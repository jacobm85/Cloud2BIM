"""
Cloud2BIM Web Interface — FastAPI backend
Run from project root:  uvicorn web.main:app --host 0.0.0.0 --port 8001
"""

import asyncio
import json
import os
import threading
import uuid
from pathlib import Path
from typing import AsyncGenerator, List, Optional

import aiofiles
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from web.job_manager import JobManager

# ── Configuration ────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).parent.parent
_WEB_CONFIG_PATH = _PROJECT_ROOT / "web_config.yaml"

if _WEB_CONFIG_PATH.exists():
    with open(_WEB_CONFIG_PATH) as _f:
        _web_cfg = yaml.safe_load(_f)
else:
    _web_cfg = {}

UPLOAD_DIR = _PROJECT_ROOT / _web_cfg.get("upload_dir", "web/uploads")
JOBS_DIR = _PROJECT_ROOT / _web_cfg.get("jobs_dir", "web/jobs")

# NETWORK_DRIVES: auto-scan /drives/* first, then merge NETWORK_DRIVES env var
# and web_config.yaml for backward compatibility.
_DRIVES_ROOT = Path(os.environ.get("DRIVES_DIR", "/drives"))
NETWORK_DRIVES: list = []
if _DRIVES_ROOT.is_dir():
    for _d in sorted(_DRIVES_ROOT.iterdir()):
        if _d.is_dir():
            NETWORK_DRIVES.append({"name": _d.name, "path": str(_d)})

_env_drives = os.environ.get("NETWORK_DRIVES", "")
if _env_drives:
    try:
        NETWORK_DRIVES += json.loads(_env_drives)
    except json.JSONDecodeError:
        pass
else:
    NETWORK_DRIVES += _web_cfg.get("network_drives") or []

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Cloud2BIM Web Interface", docs_url="/api/docs")
job_manager = JobManager(JOBS_DIR)

# Rehydrate every job_dir we find on disk into the in-memory JobManager.
# Without this, a container/Uvicorn restart wipes _jobs and any job that
# was running before vanishes from both /api/jobs/active (filters on
# in-memory status='running') and /api/jobs/reusable (filters on
# labels.npy or converted_input.xyz existing). The user can't tell why
# their job went silent or how to recover it. Rehydration registers
# every disk-side job as status='failed' so the wizard surfaces it
# under "Aktiva jobb" with a Retry button.
try:
    _rehydrated = job_manager.rehydrate_from_disk()
    if _rehydrated:
        print(f"[startup] Rehydrated {_rehydrated} job(s) from disk into JobManager")
except Exception as _exc:
    print(f"[startup] Job rehydration failed: {_exc}")

app.mount("/static", StaticFiles(directory=str(_PROJECT_ROOT / "web" / "static")), name="static")


# ── Root ──────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(str(_PROJECT_ROOT / "web" / "static" / "index.html"))


# ── Version info ──────────────────────────────────────────────────────────────
# Resolved once at startup so the wizard can show "what code is running"
# next to the tagline. Three sources, tried in order:
#   1. VERSION file in project root (pre-build script can write the exact
#      commit SHA + date when .git is stripped from a Docker image)
#   2. git rev-parse on the project root (development checkout)
#   3. mtime of cloud2bim/pipeline.py as a "build timestamp" fallback
def _resolve_version() -> dict:
    import datetime as _dt
    import subprocess

    version_file = _PROJECT_ROOT / "VERSION"
    if version_file.exists():
        try:
            content = version_file.read_text(encoding="utf-8").strip()
            # Accept either "sha date branch" tokens or full json
            if content.startswith("{"):
                return {"source": "VERSION", **json.loads(content)}
            parts = content.split()
            return {
                "source": "VERSION",
                "sha": parts[0] if parts else "unknown",
                "date": parts[1] if len(parts) > 1 else "",
                "branch": parts[2] if len(parts) > 2 else "",
            }
        except Exception:
            pass

    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_PROJECT_ROOT), capture_output=True, text=True, timeout=2,
        ).stdout.strip()
        date = subprocess.run(
            ["git", "log", "-1", "--format=%cI"],
            cwd=str(_PROJECT_ROOT), capture_output=True, text=True, timeout=2,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(_PROJECT_ROOT), capture_output=True, text=True, timeout=2,
        ).stdout.strip()
        if sha:
            return {"source": "git", "sha": sha, "date": date, "branch": branch}
    except Exception:
        pass

    # Fallback: file mtime of a stable pipeline file as a build timestamp
    try:
        proxy = _PROJECT_ROOT / "cloud2bim" / "pipeline.py"
        mtime = _dt.datetime.fromtimestamp(proxy.stat().st_mtime)
        return {"source": "mtime", "sha": "dev",
                "date": mtime.isoformat(timespec="seconds"), "branch": ""}
    except Exception:
        return {"source": "unknown", "sha": "dev", "date": "", "branch": ""}


_VERSION_INFO = _resolve_version()


@app.get("/api/version")
async def get_version():
    return _VERSION_INFO


# ── Chunked upload ────────────────────────────────────────────────────────────

@app.post("/api/upload/init")
async def upload_init(filename: str = Form(...), total_size: int = Form(...)):
    upload_id = str(uuid.uuid4())
    upload_dir = UPLOAD_DIR / upload_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    meta = {"filename": filename, "total_size": total_size, "uploaded_bytes": 0, "status": "uploading"}
    (upload_dir / "meta.json").write_text(json.dumps(meta))
    return {"upload_id": upload_id}


@app.post("/api/upload/{upload_id}/chunk")
async def upload_chunk(
    upload_id: str,
    offset: int = Form(...),
    chunk: UploadFile = File(...),
):
    meta_path = UPLOAD_DIR / upload_id / "meta.json"
    if not meta_path.exists():
        raise HTTPException(404, "Upload not found")

    meta = json.loads(meta_path.read_text())
    file_path = UPLOAD_DIR / upload_id / meta["filename"]
    data = await chunk.read()

    mode = "r+b" if file_path.exists() and offset > 0 else "wb"
    async with aiofiles.open(file_path, mode) as fh:
        if offset > 0:
            await fh.seek(offset)
        await fh.write(data)

    meta["uploaded_bytes"] = offset + len(data)
    if meta["uploaded_bytes"] >= meta["total_size"]:
        meta["status"] = "complete"
    meta_path.write_text(json.dumps(meta))

    return {"uploaded_bytes": meta["uploaded_bytes"], "status": meta["status"]}


@app.get("/api/upload/{upload_id}/status")
async def upload_status(upload_id: str):
    meta_path = UPLOAD_DIR / upload_id / "meta.json"
    if not meta_path.exists():
        raise HTTPException(404, "Upload not found")
    return json.loads(meta_path.read_text())


# ── Network drive browser ─────────────────────────────────────────────────────

@app.get("/api/browse")
async def browse(path: Optional[str] = None):
    if path is None:
        return {"drives": NETWORK_DRIVES, "items": []}

    browse_path = Path(path)

    # Security: only allow paths under configured drives
    if NETWORK_DRIVES:
        allowed = any(
            str(browse_path).startswith(str(Path(d["path"])))
            for d in NETWORK_DRIVES
        )
        if not allowed:
            raise HTTPException(403, "Path not in allowed network drives")

    if not browse_path.exists():
        raise HTTPException(404, "Path not found")

    SUPPORTED = {".xyz", ".e57", ".las", ".laz", ".ptx"}
    items = []
    try:
        for entry in sorted(browse_path.iterdir()):
            if entry.is_dir():
                items.append({"name": entry.name, "type": "dir", "path": str(entry)})
            elif entry.suffix.lower() in SUPPORTED:
                items.append({
                    "name": entry.name,
                    "type": "file",
                    "path": str(entry),
                    "size": entry.stat().st_size,
                })
    except PermissionError:
        raise HTTPException(403, "Permission denied")

    return {"current": str(browse_path), "items": items}


# ── Job management ────────────────────────────────────────────────────────────

class CreateJobRequest(BaseModel):
    # Input source (one required)
    upload_id: Optional[str] = None
    network_path: Optional[str] = None
    source_job_id: Optional[str] = None  # re-use converted_input.xyz from a previous job

    # Input format
    e57_input: bool = False
    exterior_scan: bool = False

    # Run mode: "full" runs the whole pipeline at once; "stepwise" pauses
    # between stages so the user can inspect previews and tweak params.
    mode: str = "full"

    # Detection algorithm: "v1" (original Cloud2BIM, default) or "v2"
    # (current rewrite, experimental).
    algorithm: str = "v1"

    # Point cloud options
    dilute: bool = True
    dilution_factor: int = 10
    pc_resolution: float = 0.002
    grid_coefficient: int = 5

    # Pipeline mode: geometric (histogram) / hybrid (ML + fallback) / ml (ML only)
    pipeline_mode: str = "geometric"
    hybrid_min_class_points: int = 5_000

    # Building type drives the default cross-section band when the user
    # hasn't set one manually per storey. office=upstream's 85-120% of
    # storey height. industrial=25-35 cm above floor. custom=130-160 cm.
    building_type: str = "office"

    # Vertical-continuity algorithm parameters (only used if algorithm=vertical)
    vertical_slice_thickness: float = 0.05
    vertical_min_fill: float = 0.70
    vertical_min_points_per_slice: int = 5
    vertical_sample_count: int = 5
    vertical_min_hits: int = 3
    vertical_pixel_size_cm: float = 5.0

    # New wall pairing/merging parameters (decoupled from max_thickness)
    collinear_merge_distance: float = 1.5
    pair_min_overlap: float = 0.20

    # ML semantic segmentation
    seg_enabled: bool = False
    seg_backend: str = "ptv3"
    seg_dataset: str = "s3dis"        # s3dis (indoor) | semantickitti (outdoor+vehicles)
    seg_weights: Optional[str] = None
    ml_voxel_size: float = 0.05       # 5 cm — matches S3DIS training
    geometry_resolution: float = 0.01  # 1 cm — final BIM precision
    has_rgb: str = "auto"             # auto / true / false

    # Element-type toggles (controls whether each detection stage runs)
    slabs_enabled: bool = True
    walls_enabled: bool = True
    openings_enabled: bool = True
    columns_enabled: bool = False
    stairs_enabled: bool = False
    roofs_enabled: bool = False

    # Slab thicknesses + peak detection
    bfs_thickness: float = 0.3
    tfs_thickness: float = 0.4
    max_slab_thickness: float = 0.5
    slab_peak_height_ratio: float = 0.25
    slab_z_step: float = 0.15

    # Wall options
    min_wall_length: float = 0.10
    min_wall_thickness: float = 0.05
    max_wall_thickness: float = 0.75
    exterior_walls_thickness: float = 0.3

    # IFC project metadata
    ifc_project_name: str = "Cloud2BIM Project"
    ifc_project_long_name: str = "Scan to BIM"
    ifc_project_version: str = "1.0"
    ifc_author_name: str = ""
    ifc_author_surname: str = ""
    ifc_author_organization: str = ""
    ifc_building_name: str = ""
    ifc_building_type: str = ""
    ifc_building_phase: str = ""
    ifc_site_latitude: List[int] = Field(default_factory=lambda: [0, 0, 0])
    ifc_site_longitude: List[int] = Field(default_factory=lambda: [0, 0, 0])
    ifc_site_elevation: float = 0.0
    material_for_objects: str = "Concrete"


def _convert_las_to_xyz(las_path: str, xyz_path: str, log_fn=None):
    """Convert .las/.laz to ASCII .xyz using laspy, with progress logging."""
    try:
        import laspy
        import numpy as np
    except ImportError:
        raise ImportError("laspy is not installed. Cannot process .las/.laz files.")

    if log_fn:
        log_fn(f"[INFO] Läser {Path(las_path).name} …")
    las = laspy.read(las_path)
    pts = np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)])
    total = len(pts)
    if log_fn:
        log_fn(f"[INFO] {total:,} punkter lästa. Skriver XYZ …")

    # Write in the tab-separated format with header that the pipeline expects
    # (same format as e57_data_to_xyz: header "//X\tY\tZ", then tab-separated rows)
    chunk = 500_000
    with open(xyz_path, "w") as fh:
        fh.write("//X\tY\tZ\n")
        for i in range(0, total, chunk):
            np.savetxt(fh, pts[i : i + chunk], fmt="%.3f", delimiter="\t", comments="")
            done = min(i + chunk, total)
            pct = int(done / total * 100)
            if log_fn:
                log_fn(f"[INFO] Skriver XYZ … {done:,} / {total:,} ({pct}%)")

    if log_fn:
        log_fn(f"[INFO] XYZ sparat: {Path(xyz_path).name}")


# ── Segmentation config helper ────────────────────────────────────────────────
#
# Maps wizard request → SegmentationConfig dict. Picks dataset-appropriate
# default class lists (wall_classes / floor_classes / …) so a "switch
# dataset" radio in the wizard doesn't require the user to also retype
# seven class-mapping lists. The wizard can still send explicit overrides
# if a user wants them, but right now buildConfig doesn't expose those —
# the defaults below are what every job uses for the chosen dataset.
def _build_segmentation_cfg(request) -> dict:
    dataset = request.seg_dataset if request.seg_dataset in ("s3dis", "semantickitti") else "s3dis"
    cfg = {
        "enabled": request.seg_enabled or request.pipeline_mode in ("ml", "hybrid"),
        "backend": request.seg_backend,
        "dataset": dataset,
        "weights_path": request.seg_weights,
        "ml_voxel_size": request.ml_voxel_size,
        "geometry_resolution": request.geometry_resolution,
        "has_rgb": request.has_rgb if request.has_rgb in ("auto", "true", "false") else "auto",
        "device": "auto",
        "cache_labels": True,
    }
    if dataset == "semantickitti":
        # SemanticKITTI was trained on outdoor LiDAR; there's no
        # ceiling class, "building" is the wall analogue, ground-level
        # surfaces split across several labels, and vehicles get their
        # own classes (useful for garages).
        cfg.update({
            "wall_classes": ["building", "fence"],
            "floor_classes": ["road", "parking", "sidewalk", "other-ground", "terrain"],
            "ceiling_classes": [],
            "column_classes": ["pole", "trunk"],
            "clutter_classes": [
                "car", "bicycle", "motorcycle", "truck", "other-vehicle",
                "person", "bicyclist", "motorcyclist",
                "vegetation", "traffic-sign",
            ],
            "door_classes": [],
            "window_classes": [],
        })
    # S3DIS uses SegmentationConfig's built-in defaults.
    return cfg


@app.get("/api/jobs/reusable")
async def list_reusable_jobs():
    """Return jobs that can be re-run.

    Lists any job whose work_dir has cached state — either ``labels.npy``
    (skips the slow ML step on re-run) or ``converted_input.xyz`` (v1
    legacy XYZ that skips conversion). The ``output_ifc_exists`` flag
    tells the wizard whether to jump straight to step 4 (results) on
    click vs fall back to step 2 (settings for a fresh run).
    """
    result = []
    if not JOBS_DIR.exists():
        return result
    for job_dir in sorted(JOBS_DIR.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True):
        if not job_dir.is_dir():
            continue
        labels = job_dir / "labels.npy"
        xyz = job_dir / "converted_input.xyz"
        if not labels.exists() and not xyz.exists():
            continue
        info_path = job_dir / "job_info.json"
        info = json.loads(info_path.read_text()) if info_path.exists() else {}
        cached_size_mb = round(((labels if labels.exists() else xyz).stat().st_size) / 1_000_000, 1)
        ifc_path = job_dir / "output.ifc"
        result.append({
            "job_id": job_dir.name,
            "created_at": info.get("created_at", ""),
            "original_filename": info.get("original_filename", job_dir.name),
            "xyz_size_mb": cached_size_mb,
            "has_labels": labels.exists(),
            "output_ifc_exists": ifc_path.exists(),
        })
    return result


@app.get("/api/resources")
async def get_resources():
    """Return CPU, RAM and GPU utilisation for the wizard's resource widget.

    Polled every couple of seconds. psutil covers CPU/RAM; GPU is
    optional and depends on pynvml being importable inside the
    container (it usually is when the image was built from the
    pytorch CUDA base — the libnvidia-ml.so is part of the runtime
    libs). Returns null for missing measurements so the widget can
    grey them out instead of breaking.
    """
    def _collect():
        out: dict = {}
        try:
            import psutil
            out["cpu_pct"] = float(psutil.cpu_percent(interval=None))
            vm = psutil.virtual_memory()
            out["ram_used_gb"] = round(vm.used / (1024 ** 3), 2)
            out["ram_total_gb"] = round(vm.total / (1024 ** 3), 2)
            out["ram_pct"] = float(vm.percent)
        except Exception:
            out["cpu_pct"] = None
            out["ram_used_gb"] = out["ram_total_gb"] = out["ram_pct"] = None
        # GPU via pynvml. Skip silently if not available.
        try:
            import pynvml
            pynvml.nvmlInit()
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", errors="replace")
                out["gpu_name"] = name
                out["gpu_util_pct"] = float(util.gpu)
                out["gpu_mem_used_gb"] = round(mem.used / (1024 ** 3), 2)
                out["gpu_mem_total_gb"] = round(mem.total / (1024 ** 3), 2)
                out["gpu_mem_pct"] = round(mem.used / mem.total * 100, 1) if mem.total else 0.0
            finally:
                pynvml.nvmlShutdown()
        except Exception:
            out["gpu_name"] = None
            out["gpu_util_pct"] = None
            out["gpu_mem_used_gb"] = out["gpu_mem_total_gb"] = out["gpu_mem_pct"] = None
        return out

    return await asyncio.to_thread(_collect)


def _describe_jobs_for_active_list() -> list:
    """Build the Active-jobs list by combining JobManager memory + disk state.

    Returns one entry per job_dir whose logical status is anything other
    than 'completed' (those belong in the reuse panel). Status is derived
    as follows:

      running     — JobManager says so
      failed      — JobManager says so, or state.json says only some
                    stages finished and no process is running
      interrupted — partial state.json on disk but no JobManager entry
                    (e.g. server died mid-job before rehydration ran)
      pending     — has config but no state.json yet (uploaded but never
                    started, or prepare hasn't written its state line
                    yet — rare race window)

    Also returns ``next_stage`` so the UI can label the Retry button with
    the actual stage to run.
    """
    from datetime import datetime as _dt
    from cloud2bim.stepwise import STAGES
    if not JOBS_DIR.exists():
        return []
    in_memory = {j["job_id"]: j for j in job_manager.list_jobs()}
    seen: set = set()
    result = []
    for job_dir in JOBS_DIR.iterdir():
        if not job_dir.is_dir():
            continue
        if not (job_dir / "config.yaml").exists():
            continue
        job_id = job_dir.name
        seen.add(job_id)
        # Skip completed jobs — they're handled by /api/jobs/reusable.
        if (job_dir / "output.ifc").exists():
            continue

        info = {}
        info_path = job_dir / "job_info.json"
        if info_path.exists():
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
            except Exception:
                info = {}
        state = {}
        state_path = job_dir / "state.json"
        if state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except Exception:
                state = {}
        completed_stages = list(state.keys())

        mem = in_memory.get(job_id)
        if mem and mem.get("status") == "running":
            status = "running"
            current_stage = mem.get("current_stage")
        elif mem and mem.get("status") == "failed":
            # Rehydrated jobs were inferred from disk-only state — we
            # don't actually know if the subprocess died or the server
            # died. Surface those as "interrupted" so the wizard text
            # ("Avbrutet") is honest about what we know.
            status = "interrupted" if mem.get("rehydrated") else "failed"
            current_stage = mem.get("current_stage")
        elif mem and mem.get("status") == "completed":
            # A stage finished cleanly but the whole pipeline isn't done
            # (no output.ifc yet). The wizard is parked between stages
            # waiting for the user to inspect the review and click
            # Fortsätt — exactly the case we used to lose from the
            # active list because the in-memory status moved from
            # 'running' to 'completed'.
            status = "awaiting_input"
            current_stage = None
        elif not completed_stages:
            status = "pending"
            current_stage = None
        else:
            status = "interrupted"
            current_stage = None

        # Next un-done stage (what Retry will run).
        next_stage = next((s for s in STAGES if s not in completed_stages), None)

        created = (mem.get("created_at") if mem else None) or info.get("created_at", "")
        elapsed_s = None
        if created:
            try:
                elapsed_s = int((_dt.now() - _dt.fromisoformat(created)).total_seconds())
            except Exception:
                pass

        result.append({
            "job_id": job_id,
            "status": status,
            "mode": (mem or {}).get("mode") or "stepwise",
            "current_stage": current_stage,
            "next_stage": next_stage,
            "completed_stages": completed_stages,
            "created_at": created,
            "elapsed_seconds": elapsed_s,
            "original_filename": info.get("original_filename", job_id),
        })

    # In-memory-only jobs (config not on disk yet — rare, but include
    # them so the wizard doesn't lose track during the brief gap between
    # POST /api/jobs and the first config.yaml write).
    for job_id, mem in in_memory.items():
        if job_id in seen:
            continue
        if mem.get("status") == "completed":
            continue
        from datetime import datetime as _dt2
        created = mem.get("created_at") or ""
        elapsed_s = None
        if created:
            try:
                elapsed_s = int((_dt2.now() - _dt2.fromisoformat(created)).total_seconds())
            except Exception:
                pass
        result.append({
            "job_id": job_id,
            "status": mem.get("status") or "pending",
            "mode": mem.get("mode") or "stepwise",
            "current_stage": mem.get("current_stage"),
            "next_stage": None,
            "completed_stages": [],
            "created_at": created,
            "elapsed_seconds": elapsed_s,
            "original_filename": job_id,
        })

    # Sort: running first (so the busy stuff is at the top), then
    # newest-created — feels right when checking back later.
    _status_order = {
        "running": 0,
        "awaiting_input": 1,
        "failed": 2,
        "interrupted": 3,
        "pending": 4,
    }
    result.sort(key=lambda r: (_status_order.get(r["status"], 9), r.get("created_at") or "", r["job_id"]))
    return result


@app.get("/api/jobs/active")
async def list_active_jobs():
    """Return all jobs that haven't produced an IFC yet.

    Combines in-memory JobManager status with on-disk state.json so the
    wizard can show: still-running jobs (with a "Hoppa in"-button),
    failed/interrupted jobs (with Retry + Visa logg + Ta bort), and any
    job that was started but never made it past upload. The wizard tab
    badge counts everything here; the count is split by status so the
    UI can differentiate "running" from "needs attention".
    """
    return _describe_jobs_for_active_list()


@app.get("/api/jobs/{job_id}/wizard_state")
async def get_wizard_state(job_id: str):
    """Return the persisted wizard JS state for this job, or {} if none."""
    path = JOBS_DIR / job_id / "wizard_state.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@app.post("/api/jobs/{job_id}/wizard_state")
async def set_wizard_state(job_id: str, state: dict):
    """Persist the wizard JS state for this job.

    Frontend debounces (~500 ms) so we're not hit on every keystroke.
    Stored as plain JSON next to the job's other artefacts so a user
    can re-attach from any browser tab and pick up where they left off,
    including band overrides, algorithm choice, and ML settings.
    """
    job_dir = JOBS_DIR / job_id
    if not job_dir.exists() or not job_dir.is_dir():
        raise HTTPException(404, "Job not found")
    (job_dir / "wizard_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"ok": True}


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    import shutil
    job_dir = JOBS_DIR / job_id
    if not job_dir.exists() or not job_dir.is_dir():
        raise HTTPException(404, "Job not found")
    shutil.rmtree(job_dir)
    job_manager._jobs.pop(job_id, None)
    return {"deleted": job_id}


@app.delete("/api/jobs")
async def delete_all_jobs():
    """Remove every job directory + in-memory job entry. Used by the
    'Rensa alla' button on the reuse panel — handy after a series of
    failed runs leaves dozens of dead jobs cluttering the list."""
    import shutil
    deleted: list[str] = []
    if JOBS_DIR.exists():
        for job_dir in JOBS_DIR.iterdir():
            if not job_dir.is_dir():
                continue
            try:
                shutil.rmtree(job_dir)
                deleted.append(job_dir.name)
            except Exception:
                pass
    job_manager._jobs.clear()
    return {"deleted_count": len(deleted), "deleted": deleted}


@app.post("/api/jobs")
async def create_job(request: CreateJobRequest):
    if not request.source_job_id and not request.upload_id and not request.network_path:
        raise HTTPException(400, "Either source_job_id, upload_id or network_path is required")

    preprocess_fn = None

    # ── Re-use an existing converted XYZ from a previous job ─────────────
    if request.source_job_id:
        source_xyz = JOBS_DIR / request.source_job_id / "converted_input.xyz"
        if not source_xyz.exists():
            raise HTTPException(404, "Source job XYZ not found")
        source_info_path = JOBS_DIR / request.source_job_id / "job_info.json"
        source_info = json.loads(source_info_path.read_text()) if source_info_path.exists() else {}
        input_path = str(source_xyz)
        original_filename = source_info.get("original_filename", request.source_job_id)
        e57_input = False
        pipeline_input = input_path

    # ── Resolve uploaded or network file ─────────────────────────────────
    elif request.upload_id:
        meta_path = UPLOAD_DIR / request.upload_id / "meta.json"
        if not meta_path.exists():
            raise HTTPException(404, "Upload not found")
        meta = json.loads(meta_path.read_text())
        if meta["status"] != "complete":
            raise HTTPException(400, "Upload not complete yet")
        input_path = str(UPLOAD_DIR / request.upload_id / meta["filename"])
        original_filename = Path(input_path).name
    else:
        input_path = request.network_path
        if not Path(input_path).exists():
            raise HTTPException(404, f"File not found: {input_path}")
        original_filename = Path(input_path).name

    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    output_ifc = str(job_dir / "output.ifc")

    # ── Resolve pipeline input (v2 reads E57/LAS/XYZ natively) ───────────
    if not request.source_job_id:
        # v2 readers handle .e57/.las/.laz/.xyz directly — no preprocess step
        pipeline_input = input_path

    # ── Persist job metadata for later re-use ────────────────────────────
    from datetime import datetime as _dt
    (job_dir / "job_info.json").write_text(json.dumps({
        "created_at": _dt.now().isoformat(),
        "original_filename": original_filename,
    }))

    # ── v2 config schema ─────────────────────────────────────────────────
    config = {
        "io": {
            "input_files": [pipeline_input],
            "output_ifc": output_ifc,
            "work_dir": str(job_dir),
            "dilute": request.dilute,
            "dilution_factor": request.dilution_factor,
            "center_coordinates": True,
        },
        "segmentation": _build_segmentation_cfg(request),
        "slabs": {
            "enabled": request.slabs_enabled,
            "bottom_floor_thickness": request.bfs_thickness,
            "top_floor_thickness": request.tfs_thickness,
            "pc_resolution": request.pc_resolution,
            "grid_coefficient": request.grid_coefficient,
            "z_step": request.slab_z_step,
            "max_slab_thickness": request.max_slab_thickness,
            "peak_height_ratio": request.slab_peak_height_ratio,
        },
        "walls": {
            "enabled": request.walls_enabled,
            "min_length": request.min_wall_length,
            "min_thickness": request.min_wall_thickness,
            "max_thickness": request.max_wall_thickness,
            "exterior_thickness": request.exterior_walls_thickness,
            "use_ml_filter": True,
            "enable_ransac_fallback": True,
            "collinear_merge_distance": request.collinear_merge_distance,
            "pair_min_overlap": request.pair_min_overlap,
            "vertical_slice_thickness": request.vertical_slice_thickness,
            "vertical_min_fill": request.vertical_min_fill,
            "vertical_min_points_per_slice": request.vertical_min_points_per_slice,
            "vertical_sample_count": request.vertical_sample_count,
            "vertical_min_hits": request.vertical_min_hits,
            "vertical_pixel_size_cm": request.vertical_pixel_size_cm,
        },
        "openings": {"enabled": request.openings_enabled},
        "columns": {"enabled": request.columns_enabled},
        "stairs": {"enabled": request.stairs_enabled},
        "roofs": {"enabled": request.roofs_enabled},
        "ifc": {
            "project": {
                "name": request.ifc_project_name,
                "long_name": request.ifc_project_long_name,
                "version": request.ifc_project_version,
            },
            "author": {
                "given_name": request.ifc_author_name,
                "family_name": request.ifc_author_surname,
                "organization": request.ifc_author_organization,
            },
            "building": {
                "name": request.ifc_building_name,
                "type": request.ifc_building_type,
                "phase": request.ifc_building_phase,
            },
            "site": {
                "latitude": list(request.ifc_site_latitude),
                "longitude": list(request.ifc_site_longitude),
                "elevation": request.ifc_site_elevation,
            },
            "default_material": request.material_for_objects,
            "revit_compatible": True,
        },
        "exterior_scan": request.exterior_scan,
        "algorithm": request.algorithm if request.algorithm in ("v1", "v2", "vertical") else "v1",
        "pipeline_mode": request.pipeline_mode if request.pipeline_mode in ("geometric", "hybrid", "ml") else "geometric",
        "hybrid_min_class_points": request.hybrid_min_class_points,
        "building_type": request.building_type if request.building_type in ("office", "industrial", "custom") else "office",
    }

    config_path = job_dir / "config.yaml"
    with open(config_path, "w") as fh:
        yaml.dump(config, fh, allow_unicode=True)

    job_manager.create_job(job_id, input_path, mode=request.mode)

    if request.mode == "stepwise":
        # Run only `prepare` automatically; pause so the user can crop the
        # point cloud with a polygon on the top-down preview before the rest
        # of the pipeline runs.
        thread = threading.Thread(
            target=job_manager.run_stages_async,
            args=(job_id, str(config_path), ["prepare"]),
            daemon=True,
        )
    else:
        thread = threading.Thread(
            target=job_manager.run_job,
            args=(job_id, str(config_path), preprocess_fn),
            daemon=True,
        )
    thread.start()

    return {"job_id": job_id, "mode": request.mode}


@app.get("/api/jobs")
async def list_jobs():
    return job_manager.list_jobs()


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.get("/api/jobs/{job_id}/logs")
async def stream_logs(job_id: str):
    """Server-Sent Events stream of log lines."""
    if not job_manager.get_job(job_id):
        raise HTTPException(404, "Job not found")

    async def generate() -> AsyncGenerator[str, None]:
        last_idx = 0
        while True:
            job = job_manager.get_job(job_id)
            new_lines = job["log_lines"][last_idx:]
            for line in new_lines:
                yield f"data: {json.dumps({'line': line})}\n\n"
            last_idx += len(new_lines)

            if job["status"] in ("completed", "failed"):
                yield f"data: {json.dumps({'done': True, 'status': job['status']})}\n\n"
                break

            await asyncio.sleep(0.4)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _generate_geometry_json(ifc_path: str, json_path: str):
    """Extract triangulated mesh geometry from IFC using ifcopenshell."""
    import ifcopenshell
    import ifcopenshell.geom
    import json as _json

    COLORS = {
        'IfcWall': [0.5, 0.65, 0.8], 'IfcWallStandardCase': [0.5, 0.65, 0.8],
        'IfcSlab': [0.55, 0.55, 0.62],
        'IfcWindow': [0.55, 0.82, 1.0],
        'IfcDoor': [0.78, 0.62, 0.5],
        'IfcColumn': [0.64, 0.56, 0.90], 'IfcBeam': [0.3, 0.72, 0.3],
        'IfcStair': [0.90, 0.66, 0.35], 'IfcStairFlight': [0.90, 0.66, 0.35],
        'IfcSpace': [0.92, 0.92, 0.72],
    }
    DEFAULT_COLOR = [0.65, 0.65, 0.65]

    ifc = ifcopenshell.open(ifc_path)
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    objects = []
    it = ifcopenshell.geom.iterator(settings, ifc)
    if it.initialize():
        while True:
            try:
                shape = it.get()
                geo = shape.geometry
                v = list(geo.verts)
                f = list(geo.faces)
                if v and f:
                    t = shape.type
                    objects.append({
                        't': t,
                        'n': (shape.name or '')[:64],
                        'v': [round(x, 3) for x in v],
                        'f': f,
                        'c': COLORS.get(t, DEFAULT_COLOR),
                    })
            except Exception:
                pass
            if not it.next():
                break

    with open(json_path, 'w') as fh:
        _json.dump({'objects': objects}, fh, separators=(',', ':'))


@app.get("/api/jobs/{job_id}/geometry")
async def get_geometry(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "completed":
        raise HTTPException(409, "Job not completed")
    ifc_path = JOBS_DIR / job_id / "output.ifc"
    if not ifc_path.exists():
        raise HTTPException(404, "IFC not found")
    geo_path = JOBS_DIR / job_id / "geometry.json"
    if not geo_path.exists():
        await asyncio.to_thread(_generate_geometry_json, str(ifc_path), str(geo_path))
    return FileResponse(str(geo_path), media_type="application/json")


@app.get("/api/jobs/{job_id}/topdown")
async def topdown_preview(job_id: str):
    """Render a top-down density preview of the prepared point cloud.

    Returns the image URL plus the world-coordinate bounds so the frontend
    can map pixel clicks → world XY for the polygon-crop tool.
    """
    job_dir = JOBS_DIR / job_id
    pts_path = job_dir / "points.npz"
    if not pts_path.exists():
        raise HTTPException(404, "points.npz missing — run prepare stage first")

    out_png = job_dir / "topdown.png"
    out_meta = job_dir / "topdown.json"

    def _render():
        import numpy as _np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        data = _np.load(str(pts_path))
        xy = data["xyz"][:, :2]
        if len(xy) > 250_000:
            xy = xy[:: len(xy) // 250_000]
        x_min, y_min = float(xy[:, 0].min()), float(xy[:, 1].min())
        x_max, y_max = float(xy[:, 0].max()), float(xy[:, 1].max())
        # Fixed aspect, no tight bbox — so PNG pixels map linearly to world XY
        dpi = 100
        size_in = (8, 8 * (y_max - y_min) / max(1e-6, x_max - x_min))
        fig, ax = _plt.subplots(figsize=size_in, dpi=dpi)
        fig.patch.set_facecolor("#1a1d27")
        ax.set_facecolor("#0f1117")
        ax.scatter(xy[:, 0], xy[:, 1], s=0.3, c="#76c8e8", alpha=0.45)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig.savefig(out_png, dpi=dpi, pad_inches=0)
        _plt.close(fig)
        meta = {
            "bounds": [x_min, y_min, x_max, y_max],
            "point_count": int(len(data["xyz"])),
        }
        out_meta.write_text(json.dumps(meta))
        return meta

    meta = await asyncio.to_thread(_render)
    return {
        "image_url": f"/api/jobs/{job_id}/topdown.png?t={int(meta['point_count'])}",
        "bounds": meta["bounds"],
        "point_count": meta["point_count"],
    }


@app.get("/api/jobs/{job_id}/topdown.png")
async def topdown_image(job_id: str):
    out_png = JOBS_DIR / job_id / "topdown.png"
    if not out_png.exists():
        raise HTTPException(404, "Top-down preview not yet generated; call /topdown first")
    return FileResponse(str(out_png), media_type="image/png")


@app.get("/api/jobs/{job_id}/sideview")
async def sideview_preview(job_id: str, axis: str = "auto"):
    """Render a side-view density preview onto either XZ or YZ.

    Counterpart to /topdown for the vertical polygon crop tool.
    ``axis='auto'`` picks whichever horizontal direction has the wider
    extent (good default — most informative silhouette of the building);
    ``axis='x'`` and ``axis='y'`` force the projection so the user can
    flip the view when the auto pick isn't the one they want. Returns
    image URL + world bounds [h_min, z_min, h_max, z_max] so the
    frontend can map pixel clicks → world coords, plus the resolved axis
    so the polygon can be sent back in matching coordinates.
    """
    if axis not in ("auto", "x", "y"):
        raise HTTPException(400, "axis must be 'auto', 'x', or 'y'")
    job_dir = JOBS_DIR / job_id
    pts_path = job_dir / "points.npz"
    if not pts_path.exists():
        raise HTTPException(404, "points.npz missing — run prepare stage first")

    def _render():
        import numpy as _np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        data = _np.load(str(pts_path))
        xyz = data["xyz"]
        x_spread = float(xyz[:, 0].max() - xyz[:, 0].min())
        y_spread = float(xyz[:, 1].max() - xyz[:, 1].min())
        if axis == "auto":
            chosen = "x" if x_spread >= y_spread else "y"
        else:
            chosen = axis
        h = xyz[:, 0] if chosen == "x" else xyz[:, 1]
        z = xyz[:, 2]
        if len(h) > 250_000:
            stride = len(h) // 250_000
            h = h[::stride]
            z = z[::stride]
        h_min, h_max = float(h.min()), float(h.max())
        z_min, z_max = float(z.min()), float(z.max())
        # Cache file keyed on the chosen axis so re-renders for the same
        # axis hit cache, and the URL we return matches the file we
        # wrote (axis="auto" used to write sideview_auto.png while
        # returning ?axis=x/y — that mismatch was why the image 404'd).
        out_png = job_dir / f"sideview_{chosen}.png"
        out_meta = job_dir / f"sideview_{chosen}.json"
        dpi = 100
        width_in = 12.0
        height_in = max(2.0, width_in * (z_max - z_min) / max(1e-6, h_max - h_min))
        fig, ax = _plt.subplots(figsize=(width_in, height_in), dpi=dpi)
        fig.patch.set_facecolor("#1a1d27")
        ax.set_facecolor("#0f1117")
        ax.scatter(h, z, s=0.3, c="#76c8e8", alpha=0.45)
        ax.set_xlim(h_min, h_max)
        ax.set_ylim(z_min, z_max)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig.savefig(out_png, dpi=dpi, pad_inches=0)
        _plt.close(fig)
        meta = {
            "bounds": [h_min, z_min, h_max, z_max],
            "axis": chosen,
            "point_count": int(len(xyz)),
        }
        out_meta.write_text(json.dumps(meta))
        return meta

    meta = await asyncio.to_thread(_render)
    return {
        "image_url": f"/api/jobs/{job_id}/sideview.png?axis={meta['axis']}&t={meta['point_count']}",
        "bounds": meta["bounds"],
        "axis": meta["axis"],
        "point_count": meta["point_count"],
    }


@app.get("/api/jobs/{job_id}/sideview.png")
async def sideview_image(job_id: str, axis: str = "x"):
    if axis not in ("x", "y"):
        raise HTTPException(400, "axis must be 'x' or 'y'")
    out_png = JOBS_DIR / job_id / f"sideview_{axis}.png"
    if not out_png.exists():
        raise HTTPException(404, "Side-view preview not yet generated; call /sideview first")
    return FileResponse(str(out_png), media_type="image/png")


class CropRequest(BaseModel):
    """Polygon in world XY coords (m). At least 3 vertices required."""
    polygon: List[List[float]]


class ZCropRequest(BaseModel):
    """Vertical crop: keep points with z in [z_min, z_max] (world Z, metres)."""
    z_min: float
    z_max: float


class VerticalPolygonCropRequest(BaseModel):
    """Crop in a side-view: polygon in (h, z) world coords, where h is the
    horizontal projection axis (either X or Y). At least 3 vertices."""
    polygon: List[List[float]]
    axis: str  # "x" or "y"


@app.post("/api/jobs/{job_id}/crop_z")
async def crop_points_z(job_id: str, req: ZCropRequest):
    """Filter points.npz to a vertical Z-band.

    Same workflow as the horizontal polygon crop — overwrites points.npz
    (and labels.npy if present, filtered with the same mask) so every
    downstream stage automatically reads the trimmed cloud. Invalidates
    the top-down preview cache so /topdown re-renders.
    """
    if req.z_max <= req.z_min:
        raise HTTPException(400, "z_max must be > z_min")
    job_dir = JOBS_DIR / job_id
    pts_path = job_dir / "points.npz"
    if not pts_path.exists():
        raise HTTPException(404, "points.npz missing — run prepare stage first")

    def _crop():
        import numpy as _np
        data = _np.load(str(pts_path))
        xyz = data["xyz"]
        offset = data["offset"]
        rgb = data["rgb"] if "rgb" in data.files else None
        mask = (xyz[:, 2] >= req.z_min) & (xyz[:, 2] <= req.z_max)
        kept = xyz[mask]
        if len(kept) == 0:
            return {"error": "z-range contains no points"}
        save_kwargs = {"xyz": kept.astype(_np.float32), "offset": offset}
        if rgb is not None:
            save_kwargs["rgb"] = rgb[mask].astype(_np.float32)
        _np.savez(str(pts_path), **save_kwargs)

        # Filter labels.npy in lock-step (same logic as the polygon crop).
        lbl_path = job_dir / "labels.npy"
        labels_after = None
        if lbl_path.exists():
            try:
                labels_obj = _np.load(str(lbl_path), allow_pickle=True).item()
                ids = labels_obj["ids"]
                if len(ids) == len(xyz):
                    labels_obj["ids"] = ids[mask]
                    _np.save(str(lbl_path), labels_obj, allow_pickle=True)
                    labels_after = int(len(labels_obj["ids"]))
                else:
                    lbl_path.unlink()
            except Exception:
                lbl_path.unlink(missing_ok=True)

        return {
            "before": int(len(xyz)),
            "after": int(len(kept)),
            "kept_fraction": float(len(kept) / len(xyz)),
            "labels_after": labels_after,
            "z_min": float(req.z_min),
            "z_max": float(req.z_max),
        }

    result = await asyncio.to_thread(_crop)
    if "error" in result:
        raise HTTPException(400, result["error"])

    # Invalidate top-down + side-view previews so the next render reflects
    # the trimmed cloud rather than the cached image of the full one.
    for cached in ("topdown.png", "topdown.json",
                   "sideview_auto.png", "sideview_auto.json",
                   "sideview_x.png", "sideview_x.json",
                   "sideview_y.png", "sideview_y.json"):
        try:
            (job_dir / cached).unlink()
        except FileNotFoundError:
            pass
    return result


@app.post("/api/jobs/{job_id}/crop_vertical_polygon")
async def crop_points_vertical_polygon(job_id: str, req: VerticalPolygonCropRequest):
    """Filter points.npz to those inside a polygon drawn on the side view.

    Polygon vertices are in world (h, z) where ``h`` is whichever
    horizontal axis the side view was rendered against. This is the
    vertical counterpart of the horizontal polygon crop and gives the
    user a way to remove e.g. a sloping ceiling, scaffolding, or
    overhanging trees that a flat Z-band can't surgically isolate.
    """
    if req.axis not in ("x", "y"):
        raise HTTPException(400, "axis must be 'x' or 'y'")
    if len(req.polygon) < 3:
        raise HTTPException(400, "polygon must have at least 3 vertices")

    job_dir = JOBS_DIR / job_id
    pts_path = job_dir / "points.npz"
    if not pts_path.exists():
        raise HTTPException(404, "points.npz missing — run prepare stage first")

    def _crop():
        import numpy as _np
        from matplotlib.path import Path as _MPath
        data = _np.load(str(pts_path))
        xyz = data["xyz"]
        offset = data["offset"]
        rgb = data["rgb"] if "rgb" in data.files else None
        h = xyz[:, 0] if req.axis == "x" else xyz[:, 1]
        z = xyz[:, 2]
        points_2d = _np.column_stack([h, z])
        polygon = _np.array(req.polygon, dtype=_np.float64)
        path = _MPath(polygon)
        mask = path.contains_points(points_2d)
        kept = xyz[mask]
        if len(kept) == 0:
            return {"error": "polygon contains no points"}
        save_kwargs = {"xyz": kept.astype(_np.float32), "offset": offset}
        if rgb is not None:
            save_kwargs["rgb"] = rgb[mask].astype(_np.float32)
        _np.savez(str(pts_path), **save_kwargs)

        # Filter labels.npy in lock-step (same logic as the other crops).
        lbl_path = job_dir / "labels.npy"
        labels_after = None
        if lbl_path.exists():
            try:
                labels_obj = _np.load(str(lbl_path), allow_pickle=True).item()
                ids = labels_obj["ids"]
                if len(ids) == len(xyz):
                    labels_obj["ids"] = ids[mask]
                    _np.save(str(lbl_path), labels_obj, allow_pickle=True)
                    labels_after = int(len(labels_obj["ids"]))
                else:
                    lbl_path.unlink()
            except Exception:
                lbl_path.unlink(missing_ok=True)

        return {
            "before": int(len(xyz)),
            "after": int(len(kept)),
            "kept_fraction": float(len(kept) / len(xyz)),
            "labels_after": labels_after,
            "axis": req.axis,
        }

    result = await asyncio.to_thread(_crop)
    if "error" in result:
        raise HTTPException(400, result["error"])

    # Invalidate both the top-down and the side-view caches so the next
    # render of either reflects the new cloud.
    for cached in ("topdown.png", "topdown.json",
                   "sideview_auto.png", "sideview_auto.json",
                   "sideview_x.png", "sideview_x.json",
                   "sideview_y.png", "sideview_y.json"):
        try:
            (job_dir / cached).unlink()
        except FileNotFoundError:
            pass
    return result


@app.get("/api/jobs/{job_id}/z_bounds")
async def get_z_bounds(job_id: str):
    """Return the current Z extent of points.npz.

    Used by the prepare-stage vertical-crop UI to pre-fill min/max
    inputs with the actual scan bounds rather than 0/0.
    """
    pts_path = JOBS_DIR / job_id / "points.npz"
    if not pts_path.exists():
        raise HTTPException(404, "points.npz missing — run prepare stage first")

    def _bounds():
        import numpy as _np
        data = _np.load(str(pts_path))
        z = data["xyz"][:, 2]
        return {"z_min": float(z.min()), "z_max": float(z.max()), "n_points": int(len(z))}

    return await asyncio.to_thread(_bounds)


@app.post("/api/jobs/{job_id}/crop")
async def crop_points(job_id: str, req: CropRequest):
    """Filter points.npz to points inside the given XY polygon.

    Downstream stages (slabs/walls/openings/roofs/ifc) re-read points.npz,
    so cropping here automatically tightens everything that follows. The
    user runs this from the prepare-stage review screen before letting the
    rest of the pipeline through.
    """
    job_dir = JOBS_DIR / job_id
    pts_path = job_dir / "points.npz"
    if not pts_path.exists():
        raise HTTPException(404, "points.npz missing — run prepare stage first")
    if len(req.polygon) < 3:
        raise HTTPException(400, "polygon must have at least 3 vertices")

    def _crop():
        import numpy as _np
        from matplotlib.path import Path as _MPath
        data = _np.load(str(pts_path))
        xyz = data["xyz"]
        offset = data["offset"]
        rgb = data["rgb"] if "rgb" in data.files else None
        polygon = _np.array(req.polygon, dtype=_np.float64)
        path = _MPath(polygon)
        mask = path.contains_points(xyz[:, :2])
        kept = xyz[mask]
        if len(kept) == 0:
            return {"error": "polygon contains no points"}
        save_kwargs = {"xyz": kept.astype(_np.float32), "offset": offset}
        if rgb is not None:
            save_kwargs["rgb"] = rgb[mask].astype(_np.float32)
        _np.savez(str(pts_path), **save_kwargs)

        # Also filter labels.npy if it exists — downstream stages expect
        # label_count == point_count. Without this, cropping after segment
        # would silently desync the two and crash the wall stage.
        lbl_path = job_dir / "labels.npy"
        labels_after = None
        if lbl_path.exists():
            try:
                labels_obj = _np.load(str(lbl_path), allow_pickle=True).item()
                ids = labels_obj["ids"]
                if len(ids) == len(xyz):
                    labels_obj["ids"] = ids[mask]
                    _np.save(str(lbl_path), labels_obj, allow_pickle=True)
                    labels_after = int(len(labels_obj["ids"]))
                else:
                    # Length mismatch already exists — drop the file so the
                    # user is forced to re-segment.
                    lbl_path.unlink()
            except Exception:
                # Corrupt or unexpected format — drop it; re-segmentation
                # will regenerate.
                lbl_path.unlink(missing_ok=True)

        return {
            "before": int(len(xyz)),
            "after": int(len(kept)),
            "kept_fraction": float(len(kept) / len(xyz)),
            "labels_after": labels_after,
        }

    result = await asyncio.to_thread(_crop)
    if "error" in result:
        raise HTTPException(400, result["error"])

    # Invalidate top-down + side-view previews so the next render reflects
    # the trimmed cloud rather than the cached image of the full one.
    for cached in ("topdown.png", "topdown.json",
                   "sideview_auto.png", "sideview_auto.json",
                   "sideview_x.png", "sideview_x.json",
                   "sideview_y.png", "sideview_y.json"):
        try:
            (job_dir / cached).unlink()
        except FileNotFoundError:
            pass
    return result


@app.get("/api/jobs/{job_id}/pointcloud.bin")
async def pointcloud_binary(job_id: str, max_points: int = 80000):
    """Return a decimated point cloud as raw Float32Array bytes (XYZ triplets).

    Used by the 3D viewer to overlay the prepared point cloud on top of the
    IFC mesh for visual verification. Decimated to keep WebGL happy and the
    network payload bounded — points.npz can be millions of points.
    """
    from fastapi.responses import Response
    pts_path = JOBS_DIR / job_id / "points.npz"
    if not pts_path.exists():
        raise HTTPException(404, "points.npz not found — run prepare stage first")

    def _load():
        import numpy as _np
        data = _np.load(str(pts_path))
        xyz = data["xyz"]
        n = len(xyz)
        if n > max_points and max_points > 0:
            stride = max(1, n // max_points)
            xyz = xyz[::stride]
        return xyz.astype(_np.float32, copy=False).tobytes()

    payload = await asyncio.to_thread(_load)
    return Response(content=payload, media_type="application/octet-stream")


_CLASS_ROLE_FIELDS = (
    ("wall",    "wall_classes"),
    ("floor",   "floor_classes"),
    ("ceiling", "ceiling_classes"),
    ("column",  "column_classes"),
    ("door",    "door_classes"),
    ("window",  "window_classes"),
)


def _role_of(class_name: str, seg_cfg: dict) -> str:
    """Map a class name → semantic role using the segmentation config.

    Returns 'wall' / 'floor' / 'ceiling' / 'column' / 'door' / 'window'
    if the class appears in the matching list, else 'ignore' — which
    means the class is invisible to every downstream extractor (they
    only pull labels matching their role's class-list).
    """
    for role, key in _CLASS_ROLE_FIELDS:
        if class_name in (seg_cfg.get(key) or []):
            return role
    return "ignore"


@app.get("/api/jobs/{job_id}/segment_classes")
async def get_segment_classes(job_id: str):
    """List every class the segmenter produced + its current BIM role.

    Used by the wizard's segment-stage review so the user can override
    role assignments before continuing. The downstream stages
    (slabs / walls / openings / columns / IFC) all look up points by
    role-specific class lists from config.yaml (wall_classes,
    floor_classes, …) — *not* by raw class id — so toggling roles here
    surgically changes what each extractor sees without touching the
    underlying labels.
    """
    job_dir = JOBS_DIR / job_id
    pts_path = job_dir / "points.npz"
    lbl_path = job_dir / "labels.npy"
    cfg_path = job_dir / "config.yaml"
    if not lbl_path.exists():
        raise HTTPException(404, "labels.npy missing — run segment stage first")
    if not cfg_path.exists():
        raise HTTPException(404, "config.yaml missing")

    def _load():
        import numpy as _np
        labels_obj = _np.load(str(lbl_path), allow_pickle=True).item()
        ids = labels_obj["ids"]
        names = list(labels_obj["names"])
        # Count points per label id; only classes the model actually
        # emitted appear in the output, even if more were in the vocab.
        unique, counts = _np.unique(ids, return_counts=True)
        return [(int(uid), int(c)) for uid, c in zip(unique, counts)], names

    counts_list, names = await asyncio.to_thread(_load)
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh) or {}
    seg_cfg = cfg.get("segmentation") or {}

    result = []
    for label_id, count in counts_list:
        if 0 <= label_id < len(names):
            class_name = names[label_id]
        else:
            class_name = f"class_{label_id}"
        result.append({
            "label_id": label_id,
            "name": class_name,
            "count": count,
            "role": _role_of(class_name, seg_cfg),
        })
    # Sort by count desc — the big classes are the ones the user cares
    # about deciding on first.
    result.sort(key=lambda r: -r["count"])
    return {
        "classes": result,
        "roles": ["ignore"] + [r for r, _ in _CLASS_ROLE_FIELDS],
    }


class SegmentClassesUpdate(BaseModel):
    """Map of class name → role. Role is one of:
       ignore / wall / floor / ceiling / column / door / window."""
    assignments: dict[str, str]


class ClassFilterRequest(BaseModel):
    """List of class names whose points should be removed from points.npz."""
    remove: List[str]


class SegmentModelUpdate(BaseModel):
    """Switch the ML model used by the segment stage.

    Used by the wizard's filter-only flow when the user wants to run a
    second pass with a different model (e.g. switch from SemanticKITTI
    outdoor → S3DIS indoor after stripping cars and vegetation).
    """
    backend: Optional[str] = None    # 'ptv3' | 'randla' | 'none'
    dataset: Optional[str] = None    # 's3dis' | 'semantickitti'


@app.post("/api/jobs/{job_id}/apply_class_filter")
async def apply_class_filter(job_id: str, req: ClassFilterRequest):
    """Remove all points labelled with the listed classes from points.npz.

    The flow this enables: user runs ML segmentation as a clutter-finder
    rather than a classifier. They mark cars/vegetation/furniture as
    'Ignorera' in the role editor, click this, and those points are
    gone for good. labels.npy is filtered in lock-step so downstream
    stages still have aligned labels. pipeline_mode is set to
    'geometric' on apply so the rest of the wizard runs the v1/v2/v3
    geometric algorithms on the cleaned cloud (which is what the user
    is asking for when they treat ML as 'just a filter').
    """
    if not req.remove:
        raise HTTPException(400, "remove list is empty")
    job_dir = JOBS_DIR / job_id
    pts_path = job_dir / "points.npz"
    lbl_path = job_dir / "labels.npy"
    cfg_path = job_dir / "config.yaml"
    if not pts_path.exists():
        raise HTTPException(404, "points.npz missing — run prepare first")
    if not lbl_path.exists():
        raise HTTPException(404, "labels.npy missing — run segment first")

    def _filter():
        import numpy as _np
        data = _np.load(str(pts_path))
        xyz = data["xyz"]
        offset = data["offset"]
        rgb = data["rgb"] if "rgb" in data.files else None
        labels_obj = _np.load(str(lbl_path), allow_pickle=True).item()
        ids = labels_obj["ids"]
        names = list(labels_obj["names"])
        if len(ids) != len(xyz):
            return {"error": f"label count {len(ids)} ≠ point count {len(xyz)}"}
        remove_ids = [i for i, n in enumerate(names) if n in req.remove]
        if not remove_ids:
            return {"error": f"none of {list(req.remove)} are present in the cloud"}
        keep_mask = ~_np.isin(ids, remove_ids)
        n_before = int(len(xyz))
        n_after = int(keep_mask.sum())
        if n_after == 0:
            return {"error": "filter would remove every point"}
        save_kwargs = {"xyz": xyz[keep_mask].astype(_np.float32), "offset": offset}
        if rgb is not None:
            save_kwargs["rgb"] = rgb[keep_mask].astype(_np.float32)
        _np.savez(str(pts_path), **save_kwargs)
        labels_obj["ids"] = ids[keep_mask]
        _np.save(str(lbl_path), labels_obj, allow_pickle=True)
        return {
            "before": n_before,
            "after": n_after,
            "removed": n_before - n_after,
            "removed_classes": list(req.remove),
        }

    result = await asyncio.to_thread(_filter)
    if "error" in result:
        raise HTTPException(400, result["error"])

    # Switch the pipeline to geometric so downstream stages don't try
    # to use labels (the user has signalled with this action that ML
    # was only a filter; classification belongs to v1/v2/vertical now).
    if cfg_path.exists():
        with open(cfg_path) as fh:
            cfg = yaml.safe_load(fh) or {}
        cfg["pipeline_mode"] = "geometric"
        with open(cfg_path, "w") as fh:
            yaml.dump(cfg, fh, allow_unicode=True)

    # Invalidate cached previews — the cloud just shrank.
    for cached in ("topdown.png", "topdown.json",
                   "sideview_x.png", "sideview_x.json",
                   "sideview_y.png", "sideview_y.json",
                   "prepare_z_histogram.png"):
        try:
            (job_dir / cached).unlink()
        except FileNotFoundError:
            pass
    return result


@app.post("/api/jobs/{job_id}/segment_model")
async def set_segment_model(job_id: str, req: SegmentModelUpdate):
    """Update segmentation.backend + dataset in config.yaml.

    Used by the filter-only flow to switch e.g. SemanticKITTI →
    S3DIS between passes without going back to the settings step.
    Dataset changes reset the role-specific class lists to the
    dataset's natural defaults so the role editor opens with sane
    pre-assignments rather than stale ones from the previous model.
    """
    cfg_path = JOBS_DIR / job_id / "config.yaml"
    if not cfg_path.exists():
        raise HTTPException(404, "config.yaml missing")
    if req.backend not in (None, "ptv3", "randla", "none"):
        raise HTTPException(400, "backend must be ptv3, randla, or none")
    if req.dataset not in (None, "s3dis", "semantickitti"):
        raise HTTPException(400, "dataset must be s3dis or semantickitti")

    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh) or {}
    seg_cfg = cfg.setdefault("segmentation", {})
    if req.backend is not None:
        seg_cfg["backend"] = req.backend
        # Switching backend usually means the user wants ML to actually
        # run — flip enabled back on if it was disabled.
        if req.backend != "none":
            seg_cfg["enabled"] = True
    if req.dataset is not None:
        seg_cfg["dataset"] = req.dataset
        # Hard-coded mirrors of _build_segmentation_cfg's per-dataset
        # defaults. Keeping them here means a model swap inside the
        # wizard doesn't leave wall_classes pointing at a label the
        # new model never emits.
        if req.dataset == "s3dis":
            seg_cfg["wall_classes"]    = ["wall"]
            seg_cfg["floor_classes"]   = ["floor"]
            seg_cfg["ceiling_classes"] = ["ceiling"]
            seg_cfg["column_classes"]  = ["column"]
            seg_cfg["door_classes"]    = ["door"]
            seg_cfg["window_classes"]  = ["window"]
        else:  # semantickitti
            seg_cfg["wall_classes"]    = ["building", "fence"]
            seg_cfg["floor_classes"]   = ["road", "sidewalk", "parking", "other-ground", "terrain"]
            seg_cfg["ceiling_classes"] = []
            seg_cfg["column_classes"]  = ["pole", "trunk"]
            seg_cfg["door_classes"]    = []
            seg_cfg["window_classes"]  = []
    with open(cfg_path, "w") as fh:
        yaml.dump(cfg, fh, allow_unicode=True)
    return {
        "ok": True,
        "segmentation": {
            "backend": seg_cfg.get("backend"),
            "dataset": seg_cfg.get("dataset"),
        },
    }


@app.post("/api/jobs/{job_id}/segment_classes")
async def set_segment_classes(job_id: str, req: SegmentClassesUpdate):
    """Persist user's class → role overrides into config.yaml.

    Rewrites the six role-specific class lists (wall_classes etc.) from
    scratch based on ``assignments`` — anything mapped to a role goes
    into that role's list; anything mapped to 'ignore' is dropped from
    every list. The user can then click Fortsätt to run slabs/walls/…
    against the new mapping without re-running segment.
    """
    cfg_path = JOBS_DIR / job_id / "config.yaml"
    if not cfg_path.exists():
        raise HTTPException(404, "config.yaml missing")

    valid_roles = {"ignore"} | {r for r, _ in _CLASS_ROLE_FIELDS}
    for name, role in req.assignments.items():
        if role not in valid_roles:
            raise HTTPException(400, f"Bad role for {name!r}: {role!r}")

    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh) or {}
    seg_cfg = cfg.setdefault("segmentation", {})

    # Rebuild each role list from scratch so removing a class actually
    # takes effect. clutter_classes is intentionally left untouched —
    # downstream code doesn't read it (it's effectively a legacy field),
    # and we want role='ignore' to be the canonical "exclude" signal.
    role_to_classes: dict[str, list[str]] = {r: [] for r, _ in _CLASS_ROLE_FIELDS}
    for class_name, role in req.assignments.items():
        if role == "ignore":
            continue
        role_to_classes[role].append(class_name)

    for role, key in _CLASS_ROLE_FIELDS:
        seg_cfg[key] = sorted(role_to_classes[role])

    with open(cfg_path, "w") as fh:
        yaml.dump(cfg, fh, allow_unicode=True)
    return {"ok": True, "assignments": req.assignments}


@app.get("/api/jobs/{job_id}/segment_pointcloud.bin")
async def segment_pointcloud_binary(job_id: str, max_points: int = 150000):
    """Return a decimated labeled point cloud as raw Float32Array bytes.

    Layout: (x, y, z, label_id) per point, packed contiguously — 4 floats
    per point, label_id encoded as float for transport (decode with
    Math.round() on the JS side). The segment viewer uses this to colour
    points by semantic class so the user can visually verify what the ML
    segmenter will keep vs strip before the wall/opening stages run.

    Returns 404 with a clear message when segmentation hasn't run yet
    (so the viewer can show "run segment first" instead of a stack trace).
    """
    from fastapi.responses import Response
    job_dir = JOBS_DIR / job_id
    pts_path = job_dir / "points.npz"
    lbl_path = job_dir / "labels.npy"
    if not pts_path.exists():
        raise HTTPException(404, "points.npz missing — run prepare stage first")
    if not lbl_path.exists():
        raise HTTPException(404, "labels.npy missing — run segment stage first")

    def _load():
        import numpy as _np
        data = _np.load(str(pts_path))
        xyz = data["xyz"]
        labels_obj = _np.load(str(lbl_path), allow_pickle=True).item()
        labels = labels_obj["ids"]
        if len(labels) != len(xyz):
            # Shouldn't happen, but a length mismatch would silently
            # mis-colour every point — guard the user against it.
            raise HTTPException(
                500,
                f"Label count {len(labels)} ≠ point count {len(xyz)}; "
                "re-run the segment stage.",
            )
        n = len(xyz)
        if n > max_points and max_points > 0:
            stride = max(1, n // max_points)
            xyz = xyz[::stride]
            labels = labels[::stride]
        packed = _np.empty((len(xyz), 4), dtype=_np.float32)
        packed[:, :3] = xyz.astype(_np.float32, copy=False)
        packed[:, 3] = labels.astype(_np.float32)
        return packed.tobytes(), list(labels_obj["names"])

    payload, names = await asyncio.to_thread(_load)
    return Response(
        content=payload,
        media_type="application/octet-stream",
        headers={"X-Label-Names": ",".join(names)},
    )


@app.get("/api/jobs/{job_id}/preview")
async def get_preview(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    preview_path = JOBS_DIR / job_id / "output_preview.png"
    if not preview_path.exists():
        raise HTTPException(404, "Preview not available")
    return FileResponse(str(preview_path), media_type="image/png")


@app.get("/api/jobs/{job_id}/debug/bbox")
async def debug_bbox(job_id: str):
    """Return bounding boxes of points.npz and the IFC geometry, side by side.

    Lets the user verify the two are actually in the same coordinate system.
    If the IFC bbox is, say, 30x smaller than the points bbox, that's a real
    coordinate bug and this endpoint surfaces it directly.
    """
    job_dir = JOBS_DIR / job_id
    out: dict = {}

    def _summary():
        import numpy as _np
        # Points
        pts_path = job_dir / "points.npz"
        if pts_path.exists():
            data = _np.load(str(pts_path))
            xyz = data["xyz"]
            offset = data["offset"].tolist() if "offset" in data.files else None
            out["points"] = {
                "n": int(len(xyz)),
                "min": [float(v) for v in xyz.min(axis=0)],
                "max": [float(v) for v in xyz.max(axis=0)],
                "size": [float(v) for v in (xyz.max(axis=0) - xyz.min(axis=0))],
                "offset": offset,
            }
        # IFC
        ifc_path = job_dir / "output.ifc"
        if ifc_path.exists():
            try:
                import ifcopenshell
                import ifcopenshell.geom
                ifc = ifcopenshell.open(str(ifc_path))
                # Project units — surfaces unit bugs that would otherwise look
                # like a coordinate mismatch.
                units = []
                for u in ifc.by_type("IfcSIUnit"):
                    units.append({
                        "type": getattr(u, "UnitType", None),
                        "prefix": getattr(u, "Prefix", None),
                        "name": getattr(u, "Name", None),
                    })
                out["ifc_units"] = units
                settings = ifcopenshell.geom.settings()
                settings.set(settings.USE_WORLD_COORDS, True)
                it = ifcopenshell.geom.iterator(settings, ifc)
                mins = [float("inf")] * 3
                maxs = [float("-inf")] * 3
                count = 0
                if it.initialize():
                    while True:
                        try:
                            shape = it.get()
                            v = _np.asarray(shape.geometry.verts).reshape(-1, 3)
                            if len(v):
                                mins = [min(mins[k], float(v[:, k].min())) for k in range(3)]
                                maxs = [max(maxs[k], float(v[:, k].max())) for k in range(3)]
                                count += 1
                        except Exception:
                            pass
                        if not it.next():
                            break
                if count:
                    out["ifc"] = {
                        "objects": count,
                        "min": mins,
                        "max": maxs,
                        "size": [maxs[k] - mins[k] for k in range(3)],
                    }
            except Exception as exc:
                out["ifc_error"] = str(exc)
        # Slabs and walls in their stored form
        slabs_path = job_dir / "slabs.pkl"
        if slabs_path.exists():
            import pickle
            with slabs_path.open("rb") as fh:
                slabs = pickle.load(fh)
            out["slabs"] = [
                {
                    "idx": i, "bottom_z": float(s.bottom_z), "thickness": float(s.thickness),
                    "poly_min": [float(s.polygon_x.min()), float(s.polygon_y.min())],
                    "poly_max": [float(s.polygon_x.max()), float(s.polygon_y.max())],
                }
                for i, s in enumerate(slabs)
            ]
        walls_path = job_dir / "walls.pkl"
        if walls_path.exists():
            import pickle
            with walls_path.open("rb") as fh:
                walls = pickle.load(fh)
            out["walls_per_storey"] = []
            for storey_idx, ws in enumerate(walls):
                if not ws:
                    out["walls_per_storey"].append({"storey": storey_idx, "n": 0})
                    continue
                xs = [c for w in ws for c in (w.start[0], w.end[0])]
                ys = [c for w in ws for c in (w.start[1], w.end[1])]
                out["walls_per_storey"].append({
                    "storey": storey_idx, "n": len(ws),
                    "x_range": [float(min(xs)), float(max(xs))],
                    "y_range": [float(min(ys)), float(max(ys))],
                    "z_placement": float(ws[0].z_placement),
                    "z_top": float(ws[0].z_placement + ws[0].height),
                })

    await asyncio.to_thread(_summary)
    return out


# ── DXF export ───────────────────────────────────────────────────────────────

def _load_storey_data(job_dir: Path) -> dict:
    """Load slabs/walls/openings/columns/stairs/contours pickles from a job dir."""
    import pickle
    out: dict = {
        "slabs": [], "walls": [], "openings": [],
        "columns": [], "stairs": [], "contours": [],
    }
    for key, fname in [
        ("slabs", "slabs.pkl"),
        ("walls", "walls.pkl"),
        ("openings", "openings.pkl"),
        ("columns", "columns.pkl"),
        ("stairs", "stairs.pkl"),
        ("contours", "wall_contours.pkl"),
    ]:
        p = job_dir / fname
        if p.exists():
            try:
                with p.open("rb") as f:
                    out[key] = pickle.load(f)
            except Exception:
                out[key] = []
    return out


@app.get("/api/jobs/{job_id}/dxf/storeys")
async def list_dxf_storeys(job_id: str):
    """Return how many storeys are available for DXF export."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    job_dir = JOBS_DIR / job_id
    walls_pkl = job_dir / "walls.pkl"
    if not walls_pkl.exists():
        raise HTTPException(404, "walls.pkl not found — run the walls stage first")
    data = _load_storey_data(job_dir)
    return {"storeys": len(data["walls"])}


@app.get("/api/jobs/{job_id}/stats")
async def get_job_stats(job_id: str):
    """Detected-element counts read straight from the pickled stage outputs.

    Works for both pipeline and wizard runs — the log-scraper approach
    only worked for the full pipeline which writes a final summary line.
    """
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    job_dir = JOBS_DIR / job_id
    d = _load_storey_data(job_dir)
    n_slabs = len(d["slabs"])
    n_storeys = max(0, n_slabs - 1)
    n_walls = sum(len(s) for s in d["walls"])
    n_openings = sum(len(s) for s in d["openings"])
    n_doors = sum(1 for s in d["openings"] for op in s if getattr(op, "type", "") == "door")
    n_windows = sum(1 for s in d["openings"] for op in s if getattr(op, "type", "") == "window")
    n_columns = sum(len(s) for s in d["columns"])
    n_stairs = sum(len(s) for s in d["stairs"])
    # Roofs are stored separately, single list (not per-storey)
    n_roofs = 0
    roofs_pkl = job_dir / "roofs.pkl"
    if roofs_pkl.exists():
        try:
            import pickle
            with roofs_pkl.open("rb") as f:
                n_roofs = len(pickle.load(f))
        except Exception:
            pass
    return {
        "slabs": n_slabs, "storeys": n_storeys, "walls": n_walls,
        "openings": n_openings, "windows": n_windows, "doors": n_doors,
        "columns": n_columns, "stairs": n_stairs, "roofs": n_roofs,
    }


@app.get("/api/jobs/{job_id}/dxf/{storey_idx}")
async def export_storey_dxf(job_id: str, storey_idx: int):
    """Generate and return a DXF for one storey."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    job_dir = JOBS_DIR / job_id
    if not (job_dir / "walls.pkl").exists():
        raise HTTPException(404, "walls.pkl not found — run the walls stage first")

    try:
        from cloud2bim.exporters.dxf import write_storey_dxf
    except ImportError as exc:
        raise HTTPException(
            500,
            f"DXF export requires the 'ezdxf' Python package. "
            f"Install with: pip install ezdxf>=1.3.0 (in the docker image, "
            f"rebuild after adding ezdxf to requirements-docker.txt). "
            f"Underlying error: {exc}",
        )

    def _write():
        d = _load_storey_data(job_dir)
        if storey_idx < 0 or storey_idx >= len(d["walls"]):
            return None
        slab = d["slabs"][storey_idx] if storey_idx < len(d["slabs"]) else None
        openings = d["openings"][storey_idx] if storey_idx < len(d["openings"]) else []
        columns = d["columns"][storey_idx] if storey_idx < len(d["columns"]) else []
        stairs = d["stairs"][storey_idx] if storey_idx < len(d["stairs"]) else []
        contours = d["contours"][storey_idx] if storey_idx < len(d["contours"]) else []
        out = job_dir / f"plan_storey_{storey_idx}.dxf"
        write_storey_dxf(out, storey_idx, d["walls"][storey_idx],
                         openings, columns, stairs, slab,
                         cross_section_contours=contours)
        return out

    try:
        path = await asyncio.to_thread(_write)
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        raise HTTPException(500, f"DXF generation failed: {exc}\n\n{tb}")
    if path is None:
        raise HTTPException(404, f"Storey {storey_idx} not found")
    return FileResponse(
        str(path), media_type="application/dxf",
        filename=f"plan_storey_{storey_idx}.dxf",
    )


# ── Stepwise wizard endpoints ────────────────────────────────────────────────

class RunStageRequest(BaseModel):
    """Re-run a single stage, optionally with config overrides."""
    stage: str
    # Slab / wall overrides applied to the job's config.yaml before running.
    bfs_thickness: Optional[float] = None
    tfs_thickness: Optional[float] = None
    max_slab_thickness: Optional[float] = None
    slab_peak_height_ratio: Optional[float] = None
    slab_z_step: Optional[float] = None
    min_wall_length: Optional[float] = None
    min_wall_thickness: Optional[float] = None
    max_wall_thickness: Optional[float] = None
    exterior_walls_thickness: Optional[float] = None
    max_walls_per_storey: Optional[int] = None
    # Cross-section bands as a flat list of [z_min, z_max, z_min, z_max, ...]
    # one pair per storey. None entries (passed as [null, null]) keep the
    # default 130-160 cm above-floor band.
    cross_section_bands: Optional[List[Optional[List[float]]]] = None
    # Low-section bands (diagnostic + optional support-filter input).
    cross_section_bands_lower: Optional[List[Optional[List[float]]]] = None
    # When true, walls without point support in the low band are dropped.
    require_lower_support: Optional[bool] = None
    lower_support_fraction: Optional[float] = None
    # v3 vertical-continuity wall algorithm parameters.
    vertical_slice_thickness: Optional[float] = None
    vertical_min_fill: Optional[float] = None
    vertical_min_points_per_slice: Optional[int] = None
    vertical_sample_count: Optional[int] = None
    vertical_min_hits: Optional[int] = None
    vertical_pixel_size_cm: Optional[float] = None


@app.get("/api/jobs/{job_id}/state")
async def get_job_state(job_id: str):
    """Wizard state: which stages are done, current stage, and stage-aware status."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    state_path = JOBS_DIR / job_id / "state.json"
    completed = json.loads(state_path.read_text()) if state_path.exists() else {}
    from cloud2bim.stepwise import STAGES
    return {
        "job_id": job_id,
        "mode": job.get("mode", "full"),
        "status": job["status"],
        "current_stage": job.get("current_stage"),
        "completed_stages": list(completed.keys()),
        "all_stages": list(STAGES),
    }


def _apply_overrides_to_config(config_path: Path, req: RunStageRequest) -> None:
    """Merge user overrides into the job's config.yaml in-place."""
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh) or {}
    slabs = cfg.setdefault("slabs", {})
    if req.bfs_thickness is not None:
        slabs["bottom_floor_thickness"] = req.bfs_thickness
    if req.tfs_thickness is not None:
        slabs["top_floor_thickness"] = req.tfs_thickness
    if req.max_slab_thickness is not None:
        slabs["max_slab_thickness"] = req.max_slab_thickness
    if req.slab_peak_height_ratio is not None:
        slabs["peak_height_ratio"] = req.slab_peak_height_ratio
    if req.slab_z_step is not None:
        slabs["z_step"] = req.slab_z_step

    walls = cfg.setdefault("walls", {})
    if req.min_wall_length is not None:
        walls["min_length"] = req.min_wall_length
    if req.min_wall_thickness is not None:
        walls["min_thickness"] = req.min_wall_thickness
    if req.max_wall_thickness is not None:
        walls["max_thickness"] = req.max_wall_thickness
    if req.exterior_walls_thickness is not None:
        walls["exterior_thickness"] = req.exterior_walls_thickness
    if req.max_walls_per_storey is not None:
        walls["max_walls_per_storey"] = req.max_walls_per_storey
    if req.cross_section_bands is not None:
        walls["cross_section_bands"] = req.cross_section_bands
    if req.cross_section_bands_lower is not None:
        walls["cross_section_bands_lower"] = req.cross_section_bands_lower
    if req.require_lower_support is not None:
        walls["require_lower_support"] = req.require_lower_support
    if req.lower_support_fraction is not None:
        walls["lower_support_fraction"] = req.lower_support_fraction
    if req.vertical_slice_thickness is not None:
        walls["vertical_slice_thickness"] = req.vertical_slice_thickness
    if req.vertical_min_fill is not None:
        walls["vertical_min_fill"] = req.vertical_min_fill
    if req.vertical_min_points_per_slice is not None:
        walls["vertical_min_points_per_slice"] = req.vertical_min_points_per_slice
    if req.vertical_sample_count is not None:
        walls["vertical_sample_count"] = req.vertical_sample_count
    if req.vertical_min_hits is not None:
        walls["vertical_min_hits"] = req.vertical_min_hits
    if req.vertical_pixel_size_cm is not None:
        walls["vertical_pixel_size_cm"] = req.vertical_pixel_size_cm

    with open(config_path, "w") as fh:
        yaml.dump(cfg, fh, allow_unicode=True)


@app.post("/api/jobs/{job_id}/run_stage")
async def run_stage(job_id: str, req: RunStageRequest):
    """Run a single stage. If overrides are provided they're written to
    the job's config.yaml first so re-runs use the new values.

    Lazy-registers the job in JobManager if it isn't there yet — this is
    what makes "Försök igen" work for failed/interrupted jobs whose
    in-memory entry was lost (server restart without rehydration, very
    old jobs predating persistence). The job_dir must still exist on
    disk with config.yaml; we don't conjure jobs out of thin air.
    """
    config_path = JOBS_DIR / job_id / "config.yaml"
    if not config_path.exists():
        raise HTTPException(404, "Job config not found")

    job = job_manager.get_job(job_id)
    if not job:
        # Pull input_path out of config so the in-memory entry is consistent
        # with what create_job would have stored — saves the UI a round-trip.
        try:
            with open(config_path) as fh:
                _cfg = yaml.safe_load(fh) or {}
            _inputs = ((_cfg.get("io") or {}).get("input_files") or [])
            input_path = _inputs[0] if _inputs else ""
        except Exception:
            input_path = ""
        job_manager.ensure_job(job_id, input_path=input_path, mode="stepwise")

    from cloud2bim.stepwise import STAGES
    if req.stage not in STAGES:
        raise HTTPException(400, f"Unknown stage: {req.stage}")

    _apply_overrides_to_config(config_path, req)

    thread = threading.Thread(
        target=job_manager.run_stages_async,
        args=(job_id, str(config_path), [req.stage]),
        daemon=True,
    )
    thread.start()
    return {"ok": True, "stage": req.stage}


def _render_z_histogram(job_dir: Path, bands_override=None, bands_lower=None) -> Path:
    """Render the Z-histogram PNG. Returns the path."""
    import pickle
    from cloud2bim.preview import render_z_histogram
    zh_path = job_dir / "z_histogram.pkl"
    out = job_dir / "z_histogram.png"
    with open(zh_path, "rb") as fh:
        zh = pickle.load(fh)
    slabs_path = job_dir / "slabs.pkl"
    slabs = None
    if slabs_path.exists():
        with open(slabs_path, "rb") as fh:
            slabs = pickle.load(fh)
    if bands_override is None:
        cfg_path = job_dir / "config.yaml"
        bands = None
        if cfg_path.exists():
            with open(cfg_path) as fh:
                cfg = yaml.safe_load(fh) or {}
            raw = (cfg.get("walls") or {}).get("cross_section_bands") or []
            bands = [tuple(b) if b else None for b in raw]
    else:
        bands = bands_override
    render_z_histogram(out, zh.bin_centers, zh.counts, zh.peak_z,
                       slabs=slabs, cross_section_bands=bands,
                       cross_section_bands_lower=bands_lower)
    return out


@app.get("/api/jobs/{job_id}/z_histogram.png")
async def z_histogram_image(job_id: str):
    """Render the Z-histogram PNG on demand using saved state."""
    job_dir = JOBS_DIR / job_id
    zh_path = job_dir / "z_histogram.pkl"
    if not zh_path.exists():
        raise HTTPException(404, "Z-histogram not yet computed — run 'slabs' stage first")
    out = await asyncio.to_thread(_render_z_histogram, job_dir, None, None)
    return FileResponse(str(out), media_type="image/png")


@app.get("/api/jobs/{job_id}/prepare_z_histogram.png")
async def prepare_z_histogram_image(job_id: str):
    """Render a Z-histogram for the prepare-stage vertical crop tool.

    The slabs stage produces the full z_histogram.pkl with peak picks
    and band markers, but waiting for that means the user can't see
    the distribution while choosing a Z-band crop. Recomputes a quick
    histogram straight from points.npz (no peak detection, no markers)
    so the prepare-stage Z-band tab has something to look at while
    deciding crop limits. Cached as prepare_z_histogram.png and
    invalidated whenever points.npz is modified by any crop.
    """
    job_dir = JOBS_DIR / job_id
    pts_path = job_dir / "points.npz"
    if not pts_path.exists():
        raise HTTPException(404, "points.npz missing — run prepare stage first")

    out_png = job_dir / "prepare_z_histogram.png"

    def _render():
        import numpy as _np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        data = _np.load(str(pts_path))
        z = data["xyz"][:, 2]
        if z.size == 0:
            raise HTTPException(400, "points.npz is empty")
        z_min, z_max = float(z.min()), float(z.max())
        # 1 cm bins for a building-sized scan is plenty of resolution
        # without exploding memory; finer steps don't help the user
        # eyeball slab positions and just smear out peaks.
        z_step = 0.01
        bin_edges = _np.arange(z_min, z_max + z_step, z_step)
        counts, _ = _np.histogram(z, bins=bin_edges)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        fig, ax = _plt.subplots(figsize=(10, 4.5), dpi=100)
        fig.patch.set_facecolor("#1a1d27")
        ax.set_facecolor("#0f1117")
        # Horizontal-bar style: Z on Y, density on X — matches how slabs
        # show up later in the pipeline so the user can map peaks → floors.
        ax.barh(centers, counts, height=z_step, color="#76c8e8", edgecolor="none")
        ax.set_xlabel("Antal punkter", color="#cfd2dd")
        ax.set_ylabel("Z (m)", color="#cfd2dd")
        ax.tick_params(colors="#cfd2dd")
        for spine in ax.spines.values():
            spine.set_color("#2e3350")
        ax.grid(True, axis="x", alpha=0.18, color="#cfd2dd")
        ax.set_ylim(z_min, z_max)
        fig.tight_layout()
        fig.savefig(out_png, dpi=100)
        _plt.close(fig)
        return {"z_min": z_min, "z_max": z_max, "bins": int(len(counts))}

    await asyncio.to_thread(_render)
    return FileResponse(str(out_png), media_type="image/png")


class BandsRequest(BaseModel):
    """Live-preview bands for the Z-histogram. Each entry is [z_min, z_max]
    or null to use the default for that storey. ``bands_lower`` is an
    optional second band per storey shown alongside (e.g., a low-section
    used to spot windows misinterpreted as walls)."""
    bands: List[Optional[List[float]]]
    bands_lower: Optional[List[Optional[List[float]]]] = None


@app.post("/api/jobs/{job_id}/z_histogram.png")
async def z_histogram_image_with_bands(job_id: str, req: BandsRequest):
    """Same as GET but lets the client preview a band selection without
    writing the config (used for live updates while the user drags inputs)."""
    job_dir = JOBS_DIR / job_id
    zh_path = job_dir / "z_histogram.pkl"
    if not zh_path.exists():
        raise HTTPException(404, "Z-histogram not yet computed — run 'slabs' stage first")
    bands = [tuple(b) if b and len(b) == 2 else None for b in (req.bands or [])]
    bands_lower = None
    if req.bands_lower:
        bands_lower = [tuple(b) if b and len(b) == 2 else None for b in req.bands_lower]
    out = await asyncio.to_thread(_render_z_histogram, job_dir, bands, bands_lower)
    return FileResponse(str(out), media_type="image/png")


@app.get("/api/jobs/{job_id}/slabs")
async def get_slabs_data(job_id: str):
    """JSON dump of detected slabs (bottom_z, thickness, peak metadata)."""
    job_dir = JOBS_DIR / job_id
    slabs_path = job_dir / "slabs.pkl"
    zh_path = job_dir / "z_histogram.pkl"
    if not slabs_path.exists():
        raise HTTPException(404, "Slabs not yet computed")
    import pickle
    with open(slabs_path, "rb") as fh:
        slabs = pickle.load(fh)
    z_peaks = []
    if zh_path.exists():
        with open(zh_path, "rb") as fh:
            zh = pickle.load(fh)
        z_peaks = list(zh.peak_z)
    return {
        "slabs": [
            {
                "bottom_z": float(s.bottom_z),
                "thickness": float(s.thickness),
                "top_z": float(s.bottom_z + s.thickness),
            }
            for s in slabs
        ],
        "peak_z": z_peaks,
    }


class SlabSelectRequest(BaseModel):
    keep_indices: List[int]


@app.post("/api/jobs/{job_id}/slabs/select")
async def select_slabs(job_id: str, req: SlabSelectRequest):
    """Filter slabs.pkl to only the indices the user wants to keep.

    This is how the wizard supports "I see 3 slabs but only 1 is real" —
    after running the slabs stage, the user ticks the ones to keep and we
    overwrite slabs.pkl with that subset before the walls stage runs.
    """
    import pickle
    job_dir = JOBS_DIR / job_id
    slabs_path = job_dir / "slabs.pkl"
    if not slabs_path.exists():
        raise HTTPException(404, "Slabs not yet computed")
    with open(slabs_path, "rb") as fh:
        slabs = pickle.load(fh)
    keep = sorted(set(req.keep_indices))
    if not all(0 <= i < len(slabs) for i in keep):
        raise HTTPException(400, "keep_indices contain out-of-range entries")
    filtered = [slabs[i] for i in keep]
    with open(slabs_path, "wb") as fh:
        pickle.dump(filtered, fh)
    return {
        "kept": keep,
        "total_before": len(slabs),
        "total_after": len(filtered),
    }


class SlabEdit(BaseModel):
    idx: int
    bottom_z: Optional[float] = None
    thickness: Optional[float] = None


class SlabEditRequest(BaseModel):
    edits: List[SlabEdit]


@app.post("/api/jobs/{job_id}/slabs/edit")
async def edit_slabs(job_id: str, req: SlabEditRequest):
    """Apply per-slab bottom_z / thickness overrides to slabs.pkl.

    The user can adjust the floor of each detected slab and its thickness
    independently. Top is implied (bottom + thickness).
    """
    import pickle
    job_dir = JOBS_DIR / job_id
    slabs_path = job_dir / "slabs.pkl"
    if not slabs_path.exists():
        raise HTTPException(404, "Slabs not yet computed")
    with open(slabs_path, "rb") as fh:
        slabs = pickle.load(fh)
    for e in req.edits:
        if not 0 <= e.idx < len(slabs):
            raise HTTPException(400, f"Slab index {e.idx} out of range")
        s = slabs[e.idx]
        if e.bottom_z is not None:
            s.bottom_z = float(e.bottom_z)
        if e.thickness is not None:
            s.thickness = max(0.01, float(e.thickness))
    with open(slabs_path, "wb") as fh:
        pickle.dump(slabs, fh)
    return {"ok": True, "count": len(slabs)}


class CrossSectionRequest(BaseModel):
    z_min: float
    z_max: float
    storey_idx: int = 0


@app.post("/api/jobs/{job_id}/cross_section_preview")
async def cross_section_preview(job_id: str, req: CrossSectionRequest):
    """Render an XY-occupancy PNG of points within [z_min, z_max]."""
    job_dir = JOBS_DIR / job_id
    pts_path = job_dir / "points.npz"
    if not pts_path.exists():
        raise HTTPException(404, "points.npz missing — run prepare stage first")

    out = job_dir / f"cross_section_{req.storey_idx}.png"

    def _render():
        import numpy as _np
        from cloud2bim.preview import render_cross_section
        data = _np.load(str(pts_path))
        xyz = data["xyz"]
        mask = (xyz[:, 2] >= req.z_min) & (xyz[:, 2] <= req.z_max)
        xy = xyz[mask, :2]
        # Subsample if huge to keep PNG render fast
        if len(xy) > 200_000:
            stride = len(xy) // 200_000
            xy = xy[::stride]
        title = f"Snitt Z={req.z_min:.2f}–{req.z_max:.2f} m  ({mask.sum():,} pts)"
        render_cross_section(out, xy, title=title)

    await asyncio.to_thread(_render)
    return FileResponse(str(out), media_type="image/png")


class WallsTopdownRequest(BaseModel):
    storey_idx: int = 0


@app.post("/api/jobs/{job_id}/walls/topdown_overlay")
async def walls_topdown_overlay(job_id: str, req: WallsTopdownRequest):
    """Render one storey's point density with detected wall axes overlaid.

    Used by the v3 (vertical-continuity) wizard review — instead of letting
    the operator pick a horizontal slice (irrelevant for v3, which reads
    the full storey height), we show *what was detected* against the cloud.
    """
    job_dir = JOBS_DIR / job_id
    pts_path = job_dir / "points.npz"
    walls_pkl = job_dir / "walls.pkl"
    slabs_pkl = job_dir / "slabs.pkl"
    if not pts_path.exists():
        raise HTTPException(404, "points.npz missing — run prepare stage first")
    if not walls_pkl.exists():
        raise HTTPException(404, "walls.pkl missing — run walls stage first")
    if not slabs_pkl.exists():
        raise HTTPException(404, "slabs.pkl missing — run slabs stage first")

    out = job_dir / f"walls_overlay_{req.storey_idx}.png"

    def _render():
        import pickle
        import numpy as _np
        from cloud2bim.preview import render_walls_topdown_overlay
        with slabs_pkl.open("rb") as fh:
            slabs = pickle.load(fh)
        with walls_pkl.open("rb") as fh:
            storey_walls = pickle.load(fh)
        if req.storey_idx < 0 or req.storey_idx >= len(slabs) - 1:
            raise ValueError(f"storey_idx {req.storey_idx} out of range (0..{len(slabs)-2})")
        z_floor = float(slabs[req.storey_idx].bottom_z + slabs[req.storey_idx].thickness)
        z_ceiling = float(slabs[req.storey_idx + 1].bottom_z)
        data = _np.load(str(pts_path))
        xyz = data["xyz"]
        mask = (xyz[:, 2] >= z_floor) & (xyz[:, 2] <= z_ceiling)
        xy = xyz[mask, :2]
        walls = storey_walls[req.storey_idx] if req.storey_idx < len(storey_walls) else []
        title = (f"Våning {req.storey_idx} — Z {z_floor:.2f}–{z_ceiling:.2f} m  "
                 f"({len(walls)} väggar, {int(mask.sum()):,} pts)")
        bounds = render_walls_topdown_overlay(out, xy, walls, title=title)
        return {
            "bounds": list(bounds),
            "z_floor": z_floor,
            "z_ceiling": z_ceiling,
            "n_walls": len(walls),
            "n_points": int(mask.sum()),
        }

    try:
        meta = await asyncio.to_thread(_render)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {
        "image_url": f"/api/jobs/{job_id}/walls/overlay_image/{req.storey_idx}",
        **meta,
    }


@app.get("/api/jobs/{job_id}/walls/overlay_image/{storey_idx}")
async def walls_overlay_image(job_id: str, storey_idx: int):
    out = JOBS_DIR / job_id / f"walls_overlay_{storey_idx}.png"
    if not out.exists():
        raise HTTPException(404, "Overlay not yet generated — POST /walls/topdown_overlay first")
    return FileResponse(str(out), media_type="image/png")


class WallsVerticalSectionRequest(BaseModel):
    storey_idx: int
    x1: float
    y1: float
    x2: float
    y2: float
    thickness: float = 0.20  # ± half-width around the line, in metres


@app.post("/api/jobs/{job_id}/walls/vertical_section")
async def walls_vertical_section(job_id: str, req: WallsVerticalSectionRequest):
    """Render a vertical slice along a user-drawn XY line on the storey.

    Shows the column from floor to ceiling so the operator can see
    *why* v3 marked a region as wall (or didn't): is the cloud actually
    filled top-to-bottom, or is it furniture? Detected wall axes that
    the line crosses are highlighted as vertical bands.
    """
    job_dir = JOBS_DIR / job_id
    pts_path = job_dir / "points.npz"
    slabs_pkl = job_dir / "slabs.pkl"
    if not pts_path.exists():
        raise HTTPException(404, "points.npz missing — run prepare stage first")
    if not slabs_pkl.exists():
        raise HTTPException(404, "slabs.pkl missing — run slabs stage first")

    walls_pkl = job_dir / "walls.pkl"
    out = job_dir / f"walls_section_{req.storey_idx}.png"

    def _render():
        import pickle
        import numpy as _np
        from cloud2bim.preview import render_walls_vertical_section
        with slabs_pkl.open("rb") as fh:
            slabs = pickle.load(fh)
        if req.storey_idx < 0 or req.storey_idx >= len(slabs) - 1:
            raise ValueError(f"storey_idx {req.storey_idx} out of range (0..{len(slabs)-2})")
        z_floor = float(slabs[req.storey_idx].bottom_z + slabs[req.storey_idx].thickness)
        z_ceiling = float(slabs[req.storey_idx + 1].bottom_z)
        data = _np.load(str(pts_path))
        xyz = data["xyz"]

        # Compute wall_hits before rendering: intersection of each wall axis
        # with the section line, expressed as distance along the line.
        wall_hits: list[tuple[float, float, str]] = []
        if walls_pkl.exists():
            with walls_pkl.open("rb") as fh:
                storey_walls = pickle.load(fh)
            if req.storey_idx < len(storey_walls):
                sx, sy = req.x1, req.y1
                ex, ey = req.x2, req.y2
                dx, dy = ex - sx, ey - sy
                seg_len = float(_np.hypot(dx, dy)) or 1.0
                ux, uy = dx / seg_len, dy / seg_len
                for w in storey_walls[req.storey_idx]:
                    # Intersect segment line vs wall axis using parametric form
                    wsx, wsy = w.start[0], w.start[1]
                    wex, wey = w.end[0], w.end[1]
                    wdx, wdy = wex - wsx, wey - wsy
                    denom = dx * (-wdy) - dy * (-wdx)
                    if abs(denom) < 1e-9:
                        continue
                    t_line = ((wsx - sx) * (-wdy) - (wsy - sy) * (-wdx)) / denom
                    t_wall = (dx * (wsy - sy) - dy * (wsx - sx)) / denom
                    if 0.0 <= t_line <= 1.0 and 0.0 <= t_wall <= 1.0:
                        # Distance from segment start to the intersection
                        dist = t_line * seg_len
                        wall_hits.append((dist, float(w.thickness), w.label))

        title = (f"Vertikalt snitt våning {req.storey_idx} — "
                 f"längd {_np.hypot(req.x2 - req.x1, req.y2 - req.y1):.2f} m, "
                 f"bredd ±{req.thickness:.2f} m")
        render_walls_vertical_section(
            out, xyz, (req.x1, req.y1), (req.x2, req.y2),
            z_floor, z_ceiling, req.thickness, wall_hits=wall_hits, title=title,
        )
        return {"n_wall_hits": len(wall_hits)}

    try:
        await asyncio.to_thread(_render)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return FileResponse(str(out), media_type="image/png")


@app.get("/api/jobs/{job_id}/download")
async def download_ifc(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] != "completed":
        raise HTTPException(409, "Job not completed")
    ifc_path = JOBS_DIR / job_id / "output.ifc"
    if not ifc_path.exists():
        raise HTTPException(404, "Output file not found")
    return FileResponse(
        str(ifc_path),
        media_type="application/octet-stream",
        filename=f"cloud2bim_{job_id[:8]}.ifc",
    )
