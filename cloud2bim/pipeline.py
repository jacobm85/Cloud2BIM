"""Full pipeline orchestrator.

End-to-end flow:
    1. Read each input file → concatenate → optional dilution
    2. Centre coordinates (SWEREF safety)
    3. Semantic segmentation (cached per work_dir if enabled)
    4. Slab detection (Z-histogram)
    5. Per-storey wall detection (ML-filtered 2D histogram)
    6. Per-storey opening detection
    7. Optional roof plane detection
    8. IFC export with Revit-compatible structure

Each stage logs its timing and is fault-tolerant: a failure in walls for
one storey does not kill the whole job. ``run_pipeline`` runs the lot in
one go; the web UI's wizard mode calls the same stages individually via
``cloud2bim.stepwise``.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from cloud2bim.config import Config
from cloud2bim.elements.openings import detect_openings
from cloud2bim.elements.roofs import detect_roofs
from cloud2bim.elements.slabs import Slab, compute_building_pca, compute_z_histogram, detect_slabs
from cloud2bim.elements.walls import detect_walls
from cloud2bim.ifc import IfcBuilder
from cloud2bim.io import center_xy, read_pointcloud
from cloud2bim.io.coordinates import CoordinateOffset
from cloud2bim.io.readers import diluted
from cloud2bim.logging import get_logger
from cloud2bim.segmentation import SemanticLabels, create_segmenter
from cloud2bim.segmentation.base import load_cached_labels, save_cached_labels

log = get_logger(__name__)


def run_pipeline(cfg: Config) -> int:
    """Run the full pipeline. Returns process exit code (0 = success)."""
    t_total = time.time()
    bands = list(cfg.walls.cross_section_bands or [])

    # ── 1. Read & combine inputs ────────────────────────────────────────
    points_xyz = _load_inputs(cfg)
    if len(points_xyz) == 0:
        log.error("No points loaded — check input_files")
        return 1

    # ── 2. Optional dilution ────────────────────────────────────────────
    if cfg.io.dilute:
        n_before = len(points_xyz)
        points_xyz = diluted(points_xyz, cfg.io.dilution_factor)
        log.info("Diluted: %s → %s points (1/%d)", f"{n_before:,}", f"{len(points_xyz):,}", cfg.io.dilution_factor)

    # ── 3. Centre coordinates (SWEREF safety) ───────────────────────────
    if cfg.io.center_coordinates:
        points_xyz, offset = center_xy(points_xyz)
        log.info("Coordinate offset applied: X=%.3f Y=%.3f", offset.x, offset.y)
    else:
        offset = CoordinateOffset(0, 0, 0)

    # Persist the prepared point cloud so the web viewer can overlay it on
    # the IFC. (The wizard already does this in its prepare stage; the full
    # pipeline produced no points.npz before.)
    try:
        work = Path(cfg.io.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        np.savez(work / "points.npz",
                 xyz=points_xyz.astype(np.float32),
                 offset=np.array([offset.x, offset.y, offset.z], dtype=np.float64))
    except Exception as exc:
        log.warning("Could not save points.npz for viewer overlay: %s", exc)

    # ── 4. Semantic segmentation ────────────────────────────────────────
    labels = _run_segmentation(cfg, points_xyz)

    # ── 5. Slab detection ───────────────────────────────────────────────
    log.info("─── Slab segmentation ───")
    t0 = time.time()
    zh = compute_z_histogram(points_xyz, cfg.slabs.z_step, cfg.slabs.peak_height_ratio)
    building_pca_angle = compute_building_pca(points_xyz, zh.peak_z)
    slabs = detect_slabs(points_xyz, cfg.slabs, pca_angle=building_pca_angle)
    log.info("Slabs: %d in %.1fs", len(slabs), time.time() - t0)

    # Synthesize missing floor/ceiling so we can still emit placeholder walls
    placeholder_storeys: set[int] = set()
    if len(slabs) < 2:
        log.warning(
            "Fewer than 2 slabs (%d) — synthesizing from scan Z bounds; walls "
            "will be drawn as %.0f cm placeholders so the modeller knows "
            "something is there.",
            len(slabs), cfg.walls.placeholder_height * 100,
        )
        slabs = _synthesize_slabs(slabs, points_xyz, cfg)
        placeholder_storeys = set(range(len(slabs) - 1))

    # ── 6. Per-storey walls + openings ──────────────────────────────────
    log.info("─── Wall & opening segmentation ───")
    storey_walls: list[list] = []
    storey_openings: list[list] = []
    storey_contours: list[list] = []
    for i in range(len(slabs) - 1):
        z_floor = slabs[i].bottom_z + slabs[i].thickness
        z_ceiling = slabs[i + 1].bottom_z
        is_placeholder = i in placeholder_storeys
        storey_mask = (points_xyz[:, 2] >= z_floor - 0.1) & (points_xyz[:, 2] <= z_ceiling + 0.1)
        storey_pts = points_xyz[storey_mask]
        storey_labels = SemanticLabels(
            label_ids=labels.label_ids[storey_mask],
            label_names=labels.label_names,
        )
        slab_polygon_xy = np.column_stack([slabs[i + 1].polygon_x, slabs[i + 1].polygon_y])

        band_override = bands[i] if i < len(bands) and bands[i] is not None else None
        contours_out: list = []
        try:
            t0 = time.time()
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
            )
            if is_placeholder:
                for w in walls:
                    w.height = cfg.walls.placeholder_height
                    w.label = w.label + "_placeholder"
            log.info(
                "Storey %d walls: %d in %.1fs%s",
                i, len(walls), time.time() - t0,
                " (placeholder height)" if is_placeholder else "",
            )
        except Exception as exc:
            log.exception("Storey %d wall detection failed — skipping: %s", i, exc)
            walls = []
        storey_contours.append(contours_out)

        try:
            t0 = time.time()
            openings = detect_openings(
                walls=walls,
                storey_points=storey_pts,
                cfg=cfg.openings,
                pc_resolution=cfg.slabs.pc_resolution,
                grid_coefficient=cfg.slabs.grid_coefficient,
                semantic_labels=storey_labels,
            )
            log.info("Storey %d openings: %d in %.1fs", i, len(openings), time.time() - t0)
        except Exception as exc:
            log.exception("Storey %d opening detection failed — skipping: %s", i, exc)
            openings = []

        storey_walls.append(walls)
        storey_openings.append(openings)

    # Persist raw cross-section contours so the DXF endpoint can emit the
    # "continuous line like the cross-section" the user expects.
    try:
        import pickle
        work = Path(cfg.io.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        with (work / "wall_contours.pkl").open("wb") as fh:
            pickle.dump(storey_contours, fh)
    except Exception as exc:
        log.warning("Could not save wall_contours.pkl: %s", exc)

    # ── 7a. Columns ─────────────────────────────────────────────────────
    storey_columns: list[list] = []
    if cfg.columns.enabled:
        from cloud2bim.elements.columns import detect_columns
        log.info("─── Column segmentation ───")
        t0 = time.time()
        for i in range(len(slabs) - 1):
            z_floor = slabs[i].bottom_z + slabs[i].thickness
            z_ceiling = slabs[i + 1].bottom_z
            storey_mask = (points_xyz[:, 2] >= z_floor) & (points_xyz[:, 2] <= z_ceiling)
            try:
                cols = detect_columns(
                    storey_points=points_xyz[storey_mask],
                    walls=storey_walls[i] if i < len(storey_walls) else [],
                    z_floor=z_floor, z_ceiling=z_ceiling, storey_idx=i,
                    cfg=cfg.columns,
                    pc_resolution=cfg.slabs.pc_resolution,
                    grid_coefficient=cfg.slabs.grid_coefficient,
                )
            except Exception:
                log.exception("Storey %d column detection failed", i)
                cols = []
            storey_columns.append(cols)
        log.info("Columns: %d in %.1fs",
                 sum(len(s) for s in storey_columns), time.time() - t0)
    else:
        storey_columns = [[] for _ in range(max(0, len(slabs) - 1))]

    # ── 7b. Stairs ──────────────────────────────────────────────────────
    storey_stairs: list[list] = []
    if cfg.stairs.enabled:
        from cloud2bim.elements.stairs import detect_stairs
        log.info("─── Stair segmentation ───")
        t0 = time.time()
        for i in range(len(slabs) - 1):
            z_floor = slabs[i].bottom_z + slabs[i].thickness
            z_ceiling = slabs[i + 1].bottom_z
            try:
                flights = detect_stairs(points_xyz, z_floor, z_ceiling, i, cfg.stairs)
            except Exception:
                log.exception("Storey %d stair detection failed", i)
                flights = []
            storey_stairs.append(flights)
        log.info("Stairs: %d flights in %.1fs",
                 sum(len(s) for s in storey_stairs), time.time() - t0)
    else:
        storey_stairs = [[] for _ in range(max(0, len(slabs) - 1))]

    # ── 7c. Roof detection (optional) ───────────────────────────────────
    roof_planes = []
    if cfg.roofs.enabled:
        log.info("─── Roof segmentation ───")
        t0 = time.time()
        # Use ceiling-labelled points or points above the topmost slab
        roof_mask = labels.mask_for(cfg.segmentation.ceiling_classes)
        if not roof_mask.any():
            top_z = slabs[-1].bottom_z + slabs[-1].thickness
            roof_mask = points_xyz[:, 2] > top_z - 0.5
        roof_planes = detect_roofs(points_xyz[roof_mask], cfg.roofs)
        log.info("Roofs: %d planes in %.1fs", len(roof_planes), time.time() - t0)

    # ── 8. IFC export ───────────────────────────────────────────────────
    log.info("─── IFC export ───")
    t0 = time.time()
    builder = IfcBuilder(cfg.ifc, offset=offset)
    for i, slab in enumerate(slabs):
        builder.add_slab(slab, storey_idx=i)
    for storey_idx, (walls, openings) in enumerate(zip(storey_walls, storey_openings)):
        wall_ifc_by_idx = {}
        for w_idx, w in enumerate(walls):
            wall_ifc_by_idx[w_idx] = builder.add_wall(w)
        for op in openings:
            host = walls[op.wall_index]
            builder.add_opening(op, wall_ifc_by_idx[op.wall_index], host)
    for storey_idx, cols in enumerate(storey_columns):
        for col in cols:
            try:
                builder.add_column(col)
            except Exception:
                log.exception("Column add failed")
    for storey_idx, flights in enumerate(storey_stairs):
        for stair in flights:
            try:
                builder.add_stair_flight(stair)
            except Exception:
                log.exception("Stair add failed")
    for rp in roof_planes:
        builder.add_roof_plane(rp, storey_idx=len(slabs) - 1)
    builder.write(cfg.io.output_ifc)
    log.info("IFC export: %.1fs", time.time() - t0)

    # ── 9. Floor-plan preview ───────────────────────────────────────────
    try:
        from cloud2bim.preview import render_floor_plan
        preview_path = Path(str(cfg.io.output_ifc).replace(".ifc", "_preview.png"))
        all_walls = [w for storey in storey_walls for w in storey]
        all_openings = [op for storey in storey_openings for op in storey]
        all_cols = [c for storey in storey_columns for c in storey]
        all_stairs = [s for storey in storey_stairs for s in storey]
        render_floor_plan(preview_path, slabs, all_walls, all_openings,
                          columns=all_cols, stairs=all_stairs)
    except Exception as exc:
        log.warning("Preview generation failed: %s", exc)

    log.info(
        "─── DONE in %.1fs: %d slabs, %d walls, %d openings, %d roof planes ───",
        time.time() - t_total,
        len(slabs),
        sum(len(w) for w in storey_walls),
        sum(len(o) for o in storey_openings),
        len(roof_planes),
    )
    return 0


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_inputs(cfg: Config) -> np.ndarray:
    chunks = []
    for path in cfg.io.input_files:
        xyz, _rgb = read_pointcloud(path)
        chunks.append(xyz)
    if not chunks:
        return np.empty((0, 3))
    return np.vstack(chunks) if len(chunks) > 1 else chunks[0]


def _run_segmentation(cfg: Config, points: np.ndarray) -> SemanticLabels:
    cache_path = Path(cfg.io.work_dir) / "labels.npy"
    if cfg.segmentation.cache_labels:
        cached = load_cached_labels(cache_path)
        if cached is not None and len(cached.label_ids) == len(points):
            log.info("Using cached labels: %s", cache_path)
            return cached

    seg = create_segmenter(cfg.segmentation)
    t0 = time.time()
    labels = seg.segment(points)
    log.info("Segmentation: %.1fs", time.time() - t0)

    if cfg.segmentation.cache_labels:
        save_cached_labels(labels, cache_path)
        log.info("Cached labels: %s", cache_path)
    return labels


def _synthesize_slabs(detected: list[Slab], points_xyz: np.ndarray, cfg: Config) -> list[Slab]:
    """Fill in missing floor/ceiling so wall detection can still run.

    Strategy:
        - 0 slabs: use scan z_min as floor, z_max as ceiling
        - 1 slab: use it as either floor or ceiling depending on its Z
                  relative to the scan centre; synthesize the other end.
    """
    z_min, z_max = float(points_xyz[:, 2].min()), float(points_xyz[:, 2].max())
    xs = points_xyz[:, 0]
    ys = points_xyz[:, 1]
    bbox_x = np.array([xs.min(), xs.max(), xs.max(), xs.min(), xs.min()])
    bbox_y = np.array([ys.min(), ys.min(), ys.max(), ys.max(), ys.min()])

    def _virtual_slab(z: float, thickness: float, label: str) -> Slab:
        s = Slab(bottom_z=z, thickness=thickness, polygon_x=bbox_x, polygon_y=bbox_y)
        log.info("Synthesized %s slab at z=%.2f m (placeholder geometry)", label, z)
        return s

    if not detected:
        return [
            _virtual_slab(z_min, cfg.slabs.bottom_floor_thickness, "floor"),
            _virtual_slab(z_max - cfg.slabs.top_floor_thickness,
                          cfg.slabs.top_floor_thickness, "ceiling"),
        ]
    # Exactly 1 slab: decide whether it's a floor or ceiling
    only = detected[0]
    mid = (z_min + z_max) / 2
    if only.bottom_z < mid:
        # It's a floor — synthesize a ceiling above
        return [only, _virtual_slab(
            z_max - cfg.slabs.top_floor_thickness,
            cfg.slabs.top_floor_thickness, "ceiling")]
    # It's a ceiling — synthesize a floor below
    return [_virtual_slab(z_min, cfg.slabs.bottom_floor_thickness, "floor"), only]
