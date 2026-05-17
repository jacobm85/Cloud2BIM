"""Vertical-continuity wall detection.

A wall is by definition vertical: an XY-pixel column whose points are
present from floor to ceiling without significant gaps. Furniture, by
contrast, tops out somewhere along the way — a chair fills 0–80 cm of
its column and then nothing. By scanning each pixel column slice-by-
slice along Z and keeping only the ones with a high "fill ratio" we
get a 2D mask that naturally rejects furniture without needing
semantic labels.

Pros vs the histogram + contour approach:
  - Doesn't need PCA pre-rotation: each cluster picks its own axis,
    so diagonal buildings work without preprocessing.
  - Robust against furniture in any cross-section band.
  - Single tunable that matches operator intuition: "what fraction of
    the storey height must be filled for it to count as wall?".

Cons:
  - Loses thin walls that the scan only catches one face of (the
    column is half-filled from one side, half-empty from the other).
    Histogram still wins there.
  - Slower than histogram on large clouds because we build N Z-slices
    × M XY-pixels, not one 2D grid.

Same Wall dataclass output as detect_walls() — drop-in replacement
in the pipeline dispatcher.
"""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from cloud2bim.config import WallConfig
from cloud2bim.elements.walls import Wall, _adjust_intersections, _has_nan
from cloud2bim.logging import get_logger
from cloud2bim.segmentation.base import SemanticLabels

log = get_logger(__name__)


def detect_walls_vertical(
    storey_points: np.ndarray,
    z_floor: float,
    z_ceiling: float,
    storey_idx: int,
    cfg: WallConfig,
    pc_resolution: float,
    grid_coefficient: int,
    slab_polygon_xy: Optional[np.ndarray] = None,
    semantic_labels: Optional[SemanticLabels] = None,
    exterior_scan: bool = False,
    cross_section_band: Optional[tuple[float, float]] = None,
    pca_angle: Optional[float] = None,
    out_contours: Optional[list] = None,
    lower_section_band: Optional[tuple[float, float]] = None,
) -> List[Wall]:
    """Detect walls by checking which XY-pixel columns are filled top-to-bottom.

    Signature matches ``detect_walls`` for drop-in dispatch — the
    cross-section / pca / lower-band arguments are accepted but not
    used (the algorithm reads the full storey height regardless).
    """
    n = len(storey_points)
    if n == 0:
        log.warning("Vertical walls storey %d: empty point cloud", storey_idx)
        return []
    if z_ceiling <= z_floor:
        log.warning(
            "Vertical walls storey %d: z_ceiling (%.2f) <= z_floor (%.2f)",
            storey_idx, z_ceiling, z_floor,
        )
        return []

    # Optional ML pre-filter — same as detect_walls, only relevant in
    # hybrid mode. When labels are not available we use every point.
    pts = storey_points
    if cfg.use_ml_filter and semantic_labels is not None:
        from cloud2bim.config import SegmentationConfig
        wall_mask = semantic_labels.mask_for(SegmentationConfig().wall_classes)
        if wall_mask.any():
            pts = storey_points[wall_mask]
            log.info(
                "Vertical walls storey %d: ML filter kept %d / %d points",
                storey_idx, int(wall_mask.sum()), n,
            )

    # Pixel size: v3 takes its own (in cm via config) instead of the
    # pc_resolution × grid_coefficient default, which gives 1 cm — too
    # fine for a 10–30 cm thick wall and noisier than 5 cm. Fall back
    # to the legacy default if the cm field isn't set.
    if cfg.vertical_pixel_size_cm and cfg.vertical_pixel_size_cm > 0:
        pixel_size = cfg.vertical_pixel_size_cm / 100.0
    else:
        pixel_size = pc_resolution * grid_coefficient
    slice_h = cfg.vertical_slice_thickness
    min_pts = cfg.vertical_min_points_per_slice
    n_samples = max(2, cfg.vertical_sample_count)
    min_hits = min(max(1, cfg.vertical_min_hits), n_samples)

    storey_h = z_ceiling - z_floor

    # ── 1. Pick K sample heights between floor and ceiling ──────────────
    # Evenly spaced as fractions of the storey, capped at 0.95 so we
    # stay clear of the ceiling slab itself (which would fill every
    # pixel that lives under the deck plate). The sparse K-of-N test
    # tolerates fönsterband knocking out 1–2 contiguous samples — a
    # pixel still passes as long as ≥min_hits of the K samples are
    # filled, which is the failure mode the old N-of-N fill-ratio
    # check (vertical_min_fill) had no way to handle.
    fractions = np.linspace(1.0 / (n_samples + 1), 0.95, n_samples)
    sample_zs = z_floor + fractions * storey_h
    log.info(
        "Vertical walls storey %d: %d sample heights (≥%d hits required), "
        "slice ±%.2f m around each",
        storey_idx, n_samples, min_hits, slice_h / 2,
    )

    # ── 2. Build a shared XY grid for all samples ───────────────────────
    xs = pts[:, 0]
    ys = pts[:, 1]
    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    x_edges = np.arange(x_min, x_max + pixel_size, pixel_size)
    y_edges = np.arange(y_min, y_max + pixel_size, pixel_size)
    if len(x_edges) < 2 or len(y_edges) < 2:
        log.warning("Vertical walls storey %d: XY extent too small for grid", storey_idx)
        return []
    grid_w = len(x_edges) - 1
    grid_h = len(y_edges) - 1

    # ── 3. Per sample, count fills per XY pixel ─────────────────────────
    hit_count = np.zeros((grid_h, grid_w), dtype=np.int32)
    z_coords = pts[:, 2]
    sample_fills = []
    for s_idx, z_centre in enumerate(sample_zs):
        s_lo = z_centre - slice_h / 2
        s_hi = z_centre + slice_h / 2
        slice_mask = (z_coords >= s_lo) & (z_coords < s_hi)
        if not slice_mask.any():
            sample_fills.append(0)
            continue
        slice_xy = pts[slice_mask, :2]
        # histogram2d returns (W, H); transpose to (H, W) row-major
        h, _, _ = np.histogram2d(slice_xy[:, 0], slice_xy[:, 1], bins=[x_edges, y_edges])
        h = h.T
        filled = (h >= min_pts).astype(np.int32)
        hit_count += filled
        sample_fills.append(int(filled.sum()))

    log.info(
        "Vertical walls storey %d: per-sample fills %s",
        storey_idx, sample_fills,
    )

    # ── 4. Threshold to get wall-candidate pixels ───────────────────────
    wall_mask = (hit_count >= min_hits).astype(np.uint8) * 255
    n_wall_px = int((wall_mask > 0).sum())
    log.info(
        "Vertical walls storey %d: %d wall-candidate pixels (%.1f%% of grid, "
        "≥%d of %d sample heights hit)",
        storey_idx, n_wall_px, 100.0 * n_wall_px / (grid_h * grid_w),
        min_hits, n_samples,
    )
    if n_wall_px == 0:
        return []

    # ── 5. Connected components → wall regions ──────────────────────────
    # Closing first to bridge 1-pixel gaps from scan noise.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel)
    n_labels, label_img, stats, _ = cv2.connectedComponentsWithStats(wall_mask, connectivity=8)
    log.info("Vertical walls storey %d: %d connected regions", storey_idx, n_labels - 1)

    # ── 6. Fit a wall axis per region via PCA ───────────────────────────
    wall_axes: list[list[list[float]]] = []
    wall_thicknesses: list[float] = []

    for lbl in range(1, n_labels):  # 0 is background
        ys_idx, xs_idx = np.where(label_img == lbl)
        if len(xs_idx) < 3:
            continue
        # Convert pixel indices to world XY
        wx = x_min + (xs_idx + 0.5) * pixel_size
        wy = y_min + (ys_idx + 0.5) * pixel_size
        region_xy = np.column_stack([wx, wy])

        axis, thickness = _pca_axis_and_thickness(region_xy)
        if axis is None or _has_nan(axis):
            continue
        length = float(np.hypot(axis[1][0] - axis[0][0], axis[1][1] - axis[0][1]))
        if length < cfg.min_length:
            continue
        thickness = float(np.clip(thickness, cfg.min_thickness, cfg.max_thickness))
        wall_axes.append(axis)
        wall_thicknesses.append(thickness)

    if not wall_axes:
        log.warning("Vertical walls storey %d: no axes survived filtering", storey_idx)
        return []

    # ── 7. Snap intersections (reuses v2 helper) ────────────────────────
    wall_axes = _adjust_intersections(wall_axes, cfg.max_thickness)

    # Cap to the safety limit
    if len(wall_axes) > cfg.max_walls_per_storey:
        log.warning(
            "Vertical walls storey %d: clipping %d walls down to max %d",
            storey_idx, len(wall_axes), cfg.max_walls_per_storey,
        )
        wall_axes = wall_axes[: cfg.max_walls_per_storey]
        wall_thicknesses = wall_thicknesses[: cfg.max_walls_per_storey]

    wall_height = z_ceiling - z_floor
    walls = [
        Wall(
            start=tuple(ax[0]),
            end=tuple(ax[1]),
            thickness=t,
            z_placement=z_floor,
            height=wall_height,
            storey=storey_idx,
            label="wall",
        )
        for ax, t in zip(wall_axes, wall_thicknesses)
    ]
    log.info("Vertical walls storey %d: %d walls finalised", storey_idx, len(walls))
    return walls


def _pca_axis_and_thickness(
    region_xy: np.ndarray,
) -> tuple[Optional[list[list[float]]], float]:
    """Fit an oriented bounding box via PCA.

    Returns (axis_endpoints, thickness) where axis_endpoints is the
    longest extent through the region's centroid, and thickness is the
    perpendicular extent.
    """
    if len(region_xy) < 3:
        return None, 0.0
    centroid = region_xy.mean(axis=0)
    centred = region_xy - centroid
    cov = np.cov(centred.T)
    if not np.isfinite(cov).all():
        return None, 0.0
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    axis_dir = eigvecs[:, order[0]]
    perp_dir = eigvecs[:, order[1]]

    proj_long = centred @ axis_dir
    proj_perp = centred @ perp_dir
    long_min, long_max = float(proj_long.min()), float(proj_long.max())
    perp_min, perp_max = float(proj_perp.min()), float(proj_perp.max())

    start = centroid + long_min * axis_dir
    end = centroid + long_max * axis_dir
    thickness = float(perp_max - perp_min)
    return [[float(start[0]), float(start[1])], [float(end[0]), float(end[1])]], thickness
