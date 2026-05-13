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
one storey does not kill the whole job.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from cloud2bim.config import Config
from cloud2bim.elements.openings import detect_openings
from cloud2bim.elements.roofs import detect_roofs
from cloud2bim.elements.slabs import Slab, detect_slabs
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

    # ── 4. Semantic segmentation ────────────────────────────────────────
    labels = _run_segmentation(cfg, points_xyz)

    # ── 5. Slab detection ───────────────────────────────────────────────
    log.info("─── Slab segmentation ───")
    t0 = time.time()
    slabs = detect_slabs(points_xyz, cfg.slabs)
    log.info("Slabs: %d in %.1fs", len(slabs), time.time() - t0)
    if len(slabs) < 2:
        log.error("Need ≥ 2 slabs for wall detection (got %d)", len(slabs))
        return _emit_ifc_only_slabs(cfg, slabs, offset)

    # ── 6. Per-storey walls + openings ──────────────────────────────────
    log.info("─── Wall & opening segmentation ───")
    storey_walls: list[list] = []
    storey_openings: list[list] = []
    for i in range(len(slabs) - 1):
        z_floor = slabs[i].bottom_z + slabs[i].thickness
        z_ceiling = slabs[i + 1].bottom_z
        storey_mask = (points_xyz[:, 2] >= z_floor - 0.1) & (points_xyz[:, 2] <= z_ceiling + 0.1)
        storey_pts = points_xyz[storey_mask]
        storey_labels = SemanticLabels(
            label_ids=labels.label_ids[storey_mask],
            label_names=labels.label_names,
        )
        slab_polygon_xy = np.column_stack([slabs[i + 1].polygon_x, slabs[i + 1].polygon_y])

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
            )
            log.info("Storey %d walls: %d in %.1fs", i, len(walls), time.time() - t0)
        except Exception as exc:
            log.exception("Storey %d wall detection failed — skipping: %s", i, exc)
            walls = []

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

    # ── 7. Roof detection (optional) ────────────────────────────────────
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
    for rp in roof_planes:
        builder.add_roof_plane(rp, storey_idx=len(slabs) - 1)
    builder.write(cfg.io.output_ifc)
    log.info("IFC export: %.1fs", time.time() - t0)

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


def _emit_ifc_only_slabs(cfg: Config, slabs: list[Slab], offset: CoordinateOffset) -> int:
    """Fallback: write a slab-only IFC if wall detection cannot proceed."""
    builder = IfcBuilder(cfg.ifc, offset=offset)
    for i, slab in enumerate(slabs):
        builder.add_slab(slab, storey_idx=i)
    builder.write(cfg.io.output_ifc)
    log.warning("IFC contains slabs only (insufficient slabs for walls)")
    return 0
