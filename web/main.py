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

    # Point cloud options
    dilute: bool = True
    dilution_factor: int = 10
    pc_resolution: float = 0.002
    grid_coefficient: int = 5

    # Slab thicknesses
    bfs_thickness: float = 0.3
    tfs_thickness: float = 0.4

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
    """Return jobs that have a converted_input.xyz ready to re-use."""
    result = []
    if not JOBS_DIR.exists():
        return result
    for job_dir in sorted(JOBS_DIR.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True):
        if not job_dir.is_dir():
            continue
        xyz = job_dir / "converted_input.xyz"
        if not xyz.exists():
            continue
        info_path = job_dir / "job_info.json"
        info = json.loads(info_path.read_text()) if info_path.exists() else {}
        result.append({
            "job_id": job_dir.name,
            "created_at": info.get("created_at", ""),
            "original_filename": info.get("original_filename", job_dir.name),
            "xyz_size_mb": round(xyz.stat().st_size / 1_000_000, 1),
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

    # ── Auto-detect format (skipped when re-using existing XYZ) ──────────
    if not request.source_job_id:
        suffix = Path(input_path).suffix.lower()
        if suffix == ".e57":
            e57_input = True
            pipeline_input = input_path
            xyz_converted = str(job_dir / "converted_input.xyz")
        elif suffix in (".las", ".laz"):
            e57_input = False
            xyz_converted = str(job_dir / "converted_input.xyz")
            pipeline_input = xyz_converted
            _las_src = input_path
            _xyz_dst = xyz_converted
            def preprocess_fn(log_fn):  # noqa: E731
                _convert_las_to_xyz(_las_src, _xyz_dst, log_fn)
        else:
            e57_input = False
            pipeline_input = input_path

    # ── Persist job metadata for later re-use ────────────────────────────
    from datetime import datetime as _dt
    (job_dir / "job_info.json").write_text(json.dumps({
        "created_at": _dt.now().isoformat(),
        "original_filename": original_filename,
    }))

    config = {
        "e57_input": e57_input,
        "xyz_files": [xyz_converted] if e57_input else [pipeline_input],
        "e57_files": [pipeline_input] if e57_input else [],
        "exterior_scan": request.exterior_scan,
        "dilute": request.dilute,
        "dilution_factor": request.dilution_factor,
        "pc_resolution": request.pc_resolution,
        "grid_coefficient": request.grid_coefficient,
        "bfs_thickness": request.bfs_thickness,
        "tfs_thickness": request.tfs_thickness,
        "min_wall_length": request.min_wall_length,
        "min_wall_thickness": request.min_wall_thickness,
        "max_wall_thickness": request.max_wall_thickness,
        "exterior_walls_thickness": request.exterior_walls_thickness,
        "output_ifc": output_ifc,
        "ifc_project_name": request.ifc_project_name,
        "ifc_project_long_name": request.ifc_project_long_name,
        "ifc_project_version": request.ifc_project_version,
        "ifc_author_name": request.ifc_author_name,
        "ifc_author_surname": request.ifc_author_surname,
        "ifc_author_organization": request.ifc_author_organization,
        "ifc_building_name": request.ifc_building_name,
        "ifc_building_type": request.ifc_building_type,
        "ifc_building_phase": request.ifc_building_phase,
        "ifc_site_latitude": list(request.ifc_site_latitude),
        "ifc_site_longitude": list(request.ifc_site_longitude),
        "ifc_site_elevation": request.ifc_site_elevation,
        "material_for_objects": request.material_for_objects,
    }

    config_path = job_dir / "config.yaml"
    with open(config_path, "w") as fh:
        yaml.dump(config, fh, allow_unicode=True)

    job_manager.create_job(job_id, input_path)

    thread = threading.Thread(
        target=job_manager.run_job,
        args=(job_id, str(config_path), preprocess_fn),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id}


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
        'IfcColumn': [0.6, 0.6, 0.92], 'IfcBeam': [0.3, 0.72, 0.3],
        'IfcStair': [0.9, 0.42, 0.32], 'IfcStairFlight': [0.9, 0.42, 0.32],
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


@app.get("/api/jobs/{job_id}/preview")
async def get_preview(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    preview_path = JOBS_DIR / job_id / "output_preview.png"
    if not preview_path.exists():
        raise HTTPException(404, "Preview not available")
    return FileResponse(str(preview_path), media_type="image/png")


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
