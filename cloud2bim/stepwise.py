"""Pipeline split into resumable stages.

Each stage reads its inputs from disk (set up by previous stage), runs
its detection logic, and writes outputs back to disk. The CLI exposes
each stage as its own subcommand so the web UI can pause between stages
to let the user inspect the result and tweak parameters before continuing.

State layout in ``cfg.io.work_dir``::

    state.json          stage name → completed-at timestamp
    points.npz          xyz + offset (after `prepare`)
    labels.npy          per-point semantic ids (after `segment`)
    slabs.pkl           List[Slab] (after `slabs`)
    z_histogram.pkl     ZHistogram dump (after `slabs`)
    walls.pkl           list[list[Wall]] per storey (after `walls`)
    openings.pkl        list[list[Opening]] per storey (after `openings`)
    roofs.pkl           list[RoofPlane] (after `roofs`)

Running a stage will overwrite its own outputs but never touches the
outputs of stages further down the pipeline — the UI must run those
again explicitly.
"""
from __future__ import annotations

import json
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from cloud2bim.config import Config
from cloud2bim.elements.columns import Column, detect_columns
from cloud2bim.elements.openings import detect_openings
from cloud2bim.elements.roofs import detect_roofs
from cloud2bim.elements.slabs import Slab, compute_z_histogram, detect_slabs
from cloud2bim.elements.stairs import StairFlight, detect_stairs
from cloud2bim.elements.walls import detect_walls
from cloud2bim.ifc import IfcBuilder
from cloud2bim.io import center_xy, read_pointcloud
from cloud2bim.io.coordinates import CoordinateOffset
from cloud2bim.io.readers import diluted
from cloud2bim.logging import get_logger
from cloud2bim.segmentation import SemanticLabels, create_segmenter
from cloud2bim.segmentation.base import load_cached_labels, save_cached_labels

log = get_logger(__name__)

STAGES: tuple[str, ...] = (
    "prepare", "segment", "slabs", "walls", "openings",
    "columns", "stairs", "roofs", "ifc",
)


# ── state file ──────────────────────────────────────────────────────────────

def _state_path(cfg: Config) -> Path:
    return Path(cfg.io.work_dir) / "state.json"


def read_state(cfg: Config) -> dict:
    p = _state_path(cfg)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def write_state(cfg: Config, stage: str) -> None:
    state = read_state(cfg)
    state[stage] = datetime.now().isoformat()
    # Invalidate downstream stages: once a stage is re-run, anything after
    # it is no longer trustworthy.
    idx = STAGES.index(stage)
    for downstream in STAGES[idx + 1 :]:
        state.pop(downstream, None)
    p = _state_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2))


# ── load/save helpers ───────────────────────────────────────────────────────

def _save_pickle(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)


def _load_pickle(path: Path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def load_points(cfg: Config) -> tuple[np.ndarray, CoordinateOffset]:
    path = Path(cfg.io.work_dir) / "points.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"points.npz missing — run the 'prepare' stage first ({path})"
        )
    data = np.load(path)
    offset = data["offset"]
    return data["xyz"], CoordinateOffset(float(offset[0]), float(offset[1]), float(offset[2]))


def load_labels(cfg: Config) -> SemanticLabels:
    path = Path(cfg.io.work_dir) / "labels.npy"
    cached = load_cached_labels(path)
    if cached is None:
        raise FileNotFoundError(
            f"labels.npy missing — run the 'segment' stage first ({path})"
        )
    return cached


def load_slabs(cfg: Config) -> list[Slab]:
    return _load_pickle(Path(cfg.io.work_dir) / "slabs.pkl")


def load_z_histogram(cfg: Config):
    return _load_pickle(Path(cfg.io.work_dir) / "z_histogram.pkl")


def load_walls(cfg: Config) -> list[list]:
    return _load_pickle(Path(cfg.io.work_dir) / "walls.pkl")


def load_openings(cfg: Config) -> list[list]:
    p = Path(cfg.io.work_dir) / "openings.pkl"
    return _load_pickle(p) if p.exists() else []


def load_columns(cfg: Config) -> list[list]:
    p = Path(cfg.io.work_dir) / "columns.pkl"
    return _load_pickle(p) if p.exists() else []


def load_stairs(cfg: Config) -> list[list]:
    p = Path(cfg.io.work_dir) / "stairs.pkl"
    return _load_pickle(p) if p.exists() else []


def load_roofs(cfg: Config) -> list:
    p = Path(cfg.io.work_dir) / "roofs.pkl"
    return _load_pickle(p) if p.exists() else []


# ── stages ──────────────────────────────────────────────────────────────────

def stage_prepare(cfg: Config) -> None:
    """Load inputs, optionally dilute, optionally center."""
    log.info("─── prepare ───")
    t0 = time.time()
    chunks = []
    for path in cfg.io.input_files:
        xyz, _ = read_pointcloud(path)
        chunks.append(xyz)
    if not chunks:
        raise RuntimeError("No input files produced any points")
    pts = np.vstack(chunks) if len(chunks) > 1 else chunks[0]

    if cfg.io.dilute:
        n0 = len(pts)
        pts = diluted(pts, cfg.io.dilution_factor)
        log.info("Diluted %s → %s points (1/%d)", f"{n0:,}", f"{len(pts):,}", cfg.io.dilution_factor)

    if cfg.io.center_coordinates:
        pts, offset = center_xy(pts)
        log.info("Coordinate offset applied: X=%.3f Y=%.3f", offset.x, offset.y)
    else:
        offset = CoordinateOffset(0, 0, 0)

    out = Path(cfg.io.work_dir) / "points.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, xyz=pts.astype(np.float32),
             offset=np.array([offset.x, offset.y, offset.z], dtype=np.float64))
    write_state(cfg, "prepare")
    log.info("prepare: %.1fs (%s points)", time.time() - t0, f"{len(pts):,}")


def stage_segment(cfg: Config) -> None:
    """ML or passthrough segmentation."""
    log.info("─── segment ───")
    t0 = time.time()
    pts, _ = load_points(cfg)
    cache_path = Path(cfg.io.work_dir) / "labels.npy"

    if cfg.segmentation.cache_labels:
        cached = load_cached_labels(cache_path)
        if cached is not None and len(cached.label_ids) == len(pts):
            log.info("Using cached labels: %s", cache_path)
            write_state(cfg, "segment")
            return

    seg = create_segmenter(cfg.segmentation)
    labels = seg.segment(pts)
    save_cached_labels(labels, cache_path)
    write_state(cfg, "segment")
    log.info("segment: %.1fs", time.time() - t0)


def _pca_angle_path(cfg: Config) -> Path:
    return Path(cfg.io.work_dir) / "pca_angle.json"


def load_pca_angle(cfg: Config) -> float:
    """Load the shared building PCA angle (radians); 0 if not yet computed."""
    p = _pca_angle_path(cfg)
    if not p.exists():
        return 0.0
    try:
        return float(json.loads(p.read_text()).get("pca_angle_rad", 0.0))
    except Exception:
        return 0.0


def stage_slabs(cfg: Config) -> None:
    """Z-histogram → slab detection. Also persists the shared PCA angle."""
    from cloud2bim.elements.slabs import compute_building_pca
    log.info("─── slabs ───")
    t0 = time.time()
    if not cfg.slabs.enabled:
        _save_pickle(Path(cfg.io.work_dir) / "slabs.pkl", [])
        _save_pickle(Path(cfg.io.work_dir) / "z_histogram.pkl",
                     compute_z_histogram(load_points(cfg)[0], cfg.slabs.z_step,
                                          cfg.slabs.peak_height_ratio))
        write_state(cfg, "slabs")
        log.info("slabs: disabled (skipping)")
        return
    pts, _ = load_points(cfg)
    zh = compute_z_histogram(pts, cfg.slabs.z_step, cfg.slabs.peak_height_ratio)
    pca_angle = compute_building_pca(pts, zh.peak_z)
    slabs = detect_slabs(pts, cfg.slabs, pca_angle=pca_angle)
    work = Path(cfg.io.work_dir)
    _save_pickle(work / "slabs.pkl", slabs)
    _save_pickle(work / "z_histogram.pkl", zh)
    _pca_angle_path(cfg).write_text(json.dumps({"pca_angle_rad": float(pca_angle)}))
    write_state(cfg, "slabs")
    log.info("slabs: %d in %.1fs (PCA %.1f°)",
             len(slabs), time.time() - t0, float(np.degrees(pca_angle)))


def stage_walls(cfg: Config) -> None:
    """Per-storey wall detection.

    Uses ``cfg.walls.cross_section_bands`` (one (z_min, z_max) per storey)
    to override the default 130–160 cm above-floor band when provided.
    """
    from cloud2bim.pipeline import _synthesize_slabs  # avoid circular import
    log.info("─── walls ───")
    t0 = time.time()
    if not cfg.walls.enabled:
        _save_pickle(Path(cfg.io.work_dir) / "walls.pkl", [])
        write_state(cfg, "walls")
        log.info("walls: disabled (skipping)")
        return
    pts, _ = load_points(cfg)
    labels = load_labels(cfg)
    slabs = load_slabs(cfg)
    building_pca_angle = load_pca_angle(cfg)

    placeholder_storeys: set[int] = set()
    if len(slabs) < 2:
        log.warning(
            "Fewer than 2 slabs (%d) — synthesizing storey bounds from scan Z range",
            len(slabs),
        )
        slabs = _synthesize_slabs(slabs, pts, cfg)
        placeholder_storeys = set(range(len(slabs) - 1))

    bands = list(cfg.walls.cross_section_bands or [])
    bands_lower = list(cfg.walls.cross_section_bands_lower or [])
    storey_walls: list[list] = []
    storey_contours: list[list] = []
    for i in range(len(slabs) - 1):
        z_floor = slabs[i].bottom_z + slabs[i].thickness
        z_ceiling = slabs[i + 1].bottom_z
        is_placeholder = i in placeholder_storeys
        storey_mask = (pts[:, 2] >= z_floor - 0.1) & (pts[:, 2] <= z_ceiling + 0.1)
        storey_pts = pts[storey_mask]
        storey_labels = SemanticLabels(
            label_ids=labels.label_ids[storey_mask],
            label_names=labels.label_names,
        )
        slab_polygon_xy = np.column_stack([slabs[i + 1].polygon_x, slabs[i + 1].polygon_y])
        band_override = bands[i] if i < len(bands) and bands[i] is not None else None
        band_lower = bands_lower[i] if i < len(bands_lower) and bands_lower[i] is not None else None
        contours_out: list = []

        try:
            walls = detect_walls(
                storey_points=storey_pts,
                z_floor=z_floor,
                z_ceiling=z_ceiling,
                storey_idx=i,
                cfg=cfg.walls,
                pc_resolution=cfg.slabs.pc_resolution,
                grid_coefficient=cfg.slabs.grid_coefficient,
                slab_polygon_xy=slab_polygon_xy,
                semantic_labels=storey_labels,
                exterior_scan=cfg.exterior_scan,
                cross_section_band=band_override,
                pca_angle=building_pca_angle,
                out_contours=contours_out,
                lower_section_band=band_lower,
            )
            if is_placeholder:
                for w in walls:
                    w.height = cfg.walls.placeholder_height
                    w.label = w.label + "_placeholder"
        except Exception:
            log.exception("Storey %d wall detection failed — skipping", i)
            walls = []
        storey_walls.append(walls)
        storey_contours.append(contours_out)

    _save_pickle(Path(cfg.io.work_dir) / "walls.pkl", storey_walls)
    _save_pickle(Path(cfg.io.work_dir) / "wall_contours.pkl", storey_contours)
    write_state(cfg, "walls")
    log.info("walls: %d total in %.1fs",
             sum(len(s) for s in storey_walls), time.time() - t0)


def stage_openings(cfg: Config) -> None:
    log.info("─── openings ───")
    t0 = time.time()
    if not cfg.openings.enabled:
        _save_pickle(Path(cfg.io.work_dir) / "openings.pkl", [])
        write_state(cfg, "openings")
        log.info("openings: disabled (skipping)")
        return
    pts, _ = load_points(cfg)
    labels = load_labels(cfg)
    slabs = load_slabs(cfg)
    storey_walls = load_walls(cfg)

    storey_openings: list[list] = []
    for i in range(min(len(slabs) - 1, len(storey_walls))):
        z_floor = slabs[i].bottom_z + slabs[i].thickness
        z_ceiling = slabs[i + 1].bottom_z
        storey_mask = (pts[:, 2] >= z_floor - 0.1) & (pts[:, 2] <= z_ceiling + 0.1)
        storey_pts = pts[storey_mask]
        storey_labels = SemanticLabels(
            label_ids=labels.label_ids[storey_mask],
            label_names=labels.label_names,
        )
        try:
            openings = detect_openings(
                walls=storey_walls[i],
                storey_points=storey_pts,
                cfg=cfg.openings,
                pc_resolution=cfg.slabs.pc_resolution,
                grid_coefficient=cfg.slabs.grid_coefficient,
                semantic_labels=storey_labels,
            )
        except Exception:
            log.exception("Storey %d opening detection failed — skipping", i)
            openings = []
        storey_openings.append(openings)

    _save_pickle(Path(cfg.io.work_dir) / "openings.pkl", storey_openings)
    write_state(cfg, "openings")
    log.info("openings: %d total in %.1fs",
             sum(len(s) for s in storey_openings), time.time() - t0)


def stage_columns(cfg: Config) -> None:
    """Per-storey column detection."""
    log.info("─── columns ───")
    t0 = time.time()
    if not cfg.columns.enabled:
        _save_pickle(Path(cfg.io.work_dir) / "columns.pkl", [])
        write_state(cfg, "columns")
        log.info("columns: disabled (skipping)")
        return
    pts, _ = load_points(cfg)
    slabs = load_slabs(cfg)
    storey_walls = load_walls(cfg) if (Path(cfg.io.work_dir) / "walls.pkl").exists() else []

    storey_columns: list[list[Column]] = []
    for i in range(max(0, len(slabs) - 1)):
        z_floor = slabs[i].bottom_z + slabs[i].thickness
        z_ceiling = slabs[i + 1].bottom_z
        storey_mask = (pts[:, 2] >= z_floor) & (pts[:, 2] <= z_ceiling)
        storey_pts = pts[storey_mask]
        walls_here = storey_walls[i] if i < len(storey_walls) else []
        try:
            cols = detect_columns(
                storey_points=storey_pts,
                walls=walls_here,
                z_floor=z_floor,
                z_ceiling=z_ceiling,
                storey_idx=i,
                cfg=cfg.columns,
                pc_resolution=cfg.slabs.pc_resolution,
                grid_coefficient=cfg.slabs.grid_coefficient,
            )
        except Exception:
            log.exception("Storey %d column detection failed", i)
            cols = []
        storey_columns.append(cols)

    _save_pickle(Path(cfg.io.work_dir) / "columns.pkl", storey_columns)
    write_state(cfg, "columns")
    log.info("columns: %d total in %.1fs",
             sum(len(s) for s in storey_columns), time.time() - t0)


def stage_stairs(cfg: Config) -> None:
    """Per-storey stair-flight detection."""
    log.info("─── stairs ───")
    t0 = time.time()
    if not cfg.stairs.enabled:
        _save_pickle(Path(cfg.io.work_dir) / "stairs.pkl", [])
        write_state(cfg, "stairs")
        log.info("stairs: disabled (skipping)")
        return
    pts, _ = load_points(cfg)
    slabs = load_slabs(cfg)

    storey_stairs: list[list[StairFlight]] = []
    for i in range(max(0, len(slabs) - 1)):
        z_floor = slabs[i].bottom_z + slabs[i].thickness
        z_ceiling = slabs[i + 1].bottom_z
        try:
            flights = detect_stairs(pts, z_floor, z_ceiling, i, cfg.stairs)
        except Exception:
            log.exception("Storey %d stair detection failed", i)
            flights = []
        storey_stairs.append(flights)

    _save_pickle(Path(cfg.io.work_dir) / "stairs.pkl", storey_stairs)
    write_state(cfg, "stairs")
    log.info("stairs: %d total in %.1fs",
             sum(len(s) for s in storey_stairs), time.time() - t0)


def stage_roofs(cfg: Config) -> None:
    log.info("─── roofs ───")
    t0 = time.time()
    pts, _ = load_points(cfg)
    labels = load_labels(cfg)
    slabs = load_slabs(cfg)

    roof_planes = []
    if cfg.roofs.enabled:
        roof_mask = labels.mask_for(cfg.segmentation.ceiling_classes)
        if not roof_mask.any():
            top_z = slabs[-1].bottom_z + slabs[-1].thickness
            roof_mask = pts[:, 2] > top_z - 0.5
        roof_planes = detect_roofs(pts[roof_mask], cfg.roofs)

    _save_pickle(Path(cfg.io.work_dir) / "roofs.pkl", roof_planes)
    write_state(cfg, "roofs")
    log.info("roofs: %d planes in %.1fs", len(roof_planes), time.time() - t0)


def stage_ifc(cfg: Config) -> None:
    log.info("─── ifc ───")
    t0 = time.time()
    _, offset = load_points(cfg)
    slabs = load_slabs(cfg)
    work = Path(cfg.io.work_dir)
    storey_walls = load_walls(cfg) if (work / "walls.pkl").exists() else []
    storey_openings = load_openings(cfg) if (work / "openings.pkl").exists() else [[] for _ in storey_walls]
    storey_columns = load_columns(cfg) if (work / "columns.pkl").exists() else []
    storey_stairs = load_stairs(cfg) if (work / "stairs.pkl").exists() else []
    roof_planes = load_roofs(cfg) if (work / "roofs.pkl").exists() else []
    log.info(
        "ifc inputs: %d slabs, %d storeys (%d walls), %d openings, "
        "%d columns, %d stair flights, %d roof planes",
        len(slabs), len(storey_walls), sum(len(s) for s in storey_walls),
        sum(len(s) for s in storey_openings),
        sum(len(s) for s in storey_columns),
        sum(len(s) for s in storey_stairs),
        len(roof_planes),
    )

    builder = IfcBuilder(cfg.ifc, offset=offset)

    failed_slabs = 0
    for i, slab in enumerate(slabs):
        try:
            builder.add_slab(slab, storey_idx=i)
        except Exception:
            failed_slabs += 1
            log.exception("Slab %d failed to add to IFC", i)
    if failed_slabs:
        log.warning("%d slab(s) skipped due to IFC errors", failed_slabs)

    failed_walls = failed_openings = 0
    for storey_idx, walls in enumerate(storey_walls):
        wall_ifc_by_idx: dict[int, object] = {}
        for w_idx, w in enumerate(walls):
            try:
                wall_ifc_by_idx[w_idx] = builder.add_wall(w)
            except Exception:
                failed_walls += 1
                log.exception("Storey %d wall %d failed to add to IFC", storey_idx, w_idx)
        ops = storey_openings[storey_idx] if storey_idx < len(storey_openings) else []
        for op in ops:
            if op.wall_index not in wall_ifc_by_idx:
                failed_openings += 1
                continue
            try:
                host = walls[op.wall_index]
                builder.add_opening(op, wall_ifc_by_idx[op.wall_index], host)
            except Exception:
                failed_openings += 1
                log.exception("Storey %d opening failed to add to IFC", storey_idx)
    if failed_walls or failed_openings:
        log.warning("IFC skipped: %d walls, %d openings", failed_walls, failed_openings)

    failed_cols = failed_stairs = 0
    for storey_idx, cols in enumerate(storey_columns):
        for c_idx, col in enumerate(cols):
            try:
                builder.add_column(col)
            except Exception:
                failed_cols += 1
                log.exception("Storey %d column %d failed to add to IFC", storey_idx, c_idx)
    for storey_idx, flights in enumerate(storey_stairs):
        for s_idx, stair in enumerate(flights):
            try:
                builder.add_stair_flight(stair)
            except Exception:
                failed_stairs += 1
                log.exception("Storey %d stair %d failed to add to IFC", storey_idx, s_idx)
    if failed_cols or failed_stairs:
        log.warning("IFC skipped: %d columns, %d stairs", failed_cols, failed_stairs)

    for rp_idx, rp in enumerate(roof_planes):
        try:
            builder.add_roof_plane(rp, storey_idx=max(0, len(slabs) - 1))
        except Exception:
            log.exception("Roof plane %d failed to add to IFC", rp_idx)

    try:
        builder.write(cfg.io.output_ifc)
        log.info("IFC written: %s", cfg.io.output_ifc)
    except Exception:
        log.exception("Failed to write IFC")
        raise  # writing IFC is the *output* of this stage — if it fails, fail loudly

    # Invalidate cached viewer geometry so the next /geometry call re-extracts
    # from the new IFC. Without this, the viewer kept showing an old model.
    geo_cache = Path(str(cfg.io.output_ifc)).parent / "geometry.json"
    try:
        geo_cache.unlink()
    except FileNotFoundError:
        pass

    write_state(cfg, "ifc")
    log.info("ifc: %.1fs", time.time() - t0)

    # Floor-plan preview alongside IFC — never let preview rendering kill
    # the IFC stage (we already wrote the model above).
    try:
        from cloud2bim.preview import render_floor_plan
        preview_path = Path(str(cfg.io.output_ifc).replace(".ifc", "_preview.png"))
        all_walls = [w for st in storey_walls for w in st]
        all_ops = [op for st in storey_openings for op in st]
        all_cols = [c for st in storey_columns for c in st]
        all_stairs = [s for st in storey_stairs for s in st]
        render_floor_plan(preview_path, slabs, all_walls, all_ops,
                          columns=all_cols, stairs=all_stairs)
    except Exception:
        log.exception("Preview generation failed (IFC was written successfully)")


STAGE_FNS = {
    "prepare": stage_prepare,
    "segment": stage_segment,
    "slabs": stage_slabs,
    "walls": stage_walls,
    "openings": stage_openings,
    "columns": stage_columns,
    "stairs": stage_stairs,
    "roofs": stage_roofs,
    "ifc": stage_ifc,
}


def run_stage(cfg: Config, stage: str) -> None:
    """Run one named stage. Raises if stage isn't recognised."""
    fn = STAGE_FNS.get(stage)
    if fn is None:
        raise ValueError(f"Unknown stage: {stage!r}. Valid: {STAGES}")
    fn(cfg)
