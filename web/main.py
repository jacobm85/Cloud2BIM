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

app.mount("/static", StaticFiles(directory=str(_PROJECT_ROOT / "web" / "static")), name="static")


# ── Root ──────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse(str(_PROJECT_ROOT / "web" / "static" / "index.html"))


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

    SUPPORTED = {".xyz", ".e57", ".las", ".laz"}
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

    # New wall pairing/merging parameters (decoupled from max_thickness)
    collinear_merge_distance: float = 1.5
    pair_min_overlap: float = 0.20

    # ML semantic segmentation
    seg_enabled: bool = False
    seg_backend: str = "ptv3"
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


@app.get("/api/jobs/reusable")
async def list_reusable_jobs():
    """Return jobs that can be re-run.

    v2: lists any job with cached semantic labels (skips the slow ML step).
    v1 legacy: also lists jobs with converted_input.xyz (skips conversion).
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
        result.append({
            "job_id": job_dir.name,
            "created_at": info.get("created_at", ""),
            "original_filename": info.get("original_filename", job_dir.name),
            "xyz_size_mb": cached_size_mb,
            "has_labels": labels.exists(),
        })
    return result


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
        "segmentation": {
            "enabled": request.seg_enabled or request.pipeline_mode in ("ml", "hybrid"),
            "backend": request.seg_backend,
            "weights_path": request.seg_weights,
            "ml_voxel_size": request.ml_voxel_size,
            "geometry_resolution": request.geometry_resolution,
            "has_rgb": request.has_rgb if request.has_rgb in ("auto", "true", "false") else "auto",
            "device": "auto",
            "cache_labels": True,
        },
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


class CropRequest(BaseModel):
    """Polygon in world XY coords (m). At least 3 vertices required."""
    polygon: List[List[float]]


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
        polygon = _np.array(req.polygon, dtype=_np.float64)
        path = _MPath(polygon)
        mask = path.contains_points(xyz[:, :2])
        kept = xyz[mask]
        if len(kept) == 0:
            return {"error": "polygon contains no points"}
        _np.savez(str(pts_path), xyz=kept.astype(_np.float32), offset=offset)
        return {
            "before": int(len(xyz)),
            "after": int(len(kept)),
            "kept_fraction": float(len(kept) / len(xyz)),
        }

    result = await asyncio.to_thread(_crop)
    if "error" in result:
        raise HTTPException(400, result["error"])

    # Invalidate cached top-down preview so the next /topdown call re-renders
    try:
        (job_dir / "topdown.png").unlink()
        (job_dir / "topdown.json").unlink()
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

    with open(config_path, "w") as fh:
        yaml.dump(cfg, fh, allow_unicode=True)


@app.post("/api/jobs/{job_id}/run_stage")
async def run_stage(job_id: str, req: RunStageRequest):
    """Run a single stage. If overrides are provided they're written to
    the job's config.yaml first so re-runs use the new values."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    config_path = JOBS_DIR / job_id / "config.yaml"
    if not config_path.exists():
        raise HTTPException(404, "Job config not found")

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
