"""Wall detection.

Pipeline:
    1. Filter point cloud by semantic labels (wall_classes) — drops furniture
       and clutter that confused the v1 algorithm
    2. Apply PCA rotation if dominant orientation is off-axis (>3°)
    3. 2D occupancy histogram → binary mask → contours
    4. Douglas-Peucker → segment list
    5. Group parallel/collinear segments → wall axes
    6. (Optional) RANSAC fallback for wall regions the histogram misses

All numerical guards from v1 are baked in: NaN axes filtered before
adjust_intersections, NaN intersections never written back into clean axes,
empty point sets early-return cleanly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
from skimage.morphology import closing, footprint_rectangle

from cloud2bim.config import WallConfig
from cloud2bim.geometry.lines import distance_points_to_line, line_intersection
from cloud2bim.geometry.pca import dominant_angle, rotate_points_2d
from cloud2bim.geometry.polygon import swell_polygon
from cloud2bim.logging import get_logger
from cloud2bim.segmentation.base import SemanticLabels

log = get_logger(__name__)


@dataclass
class Wall:
    """Detected wall axis with thickness and storey assignment."""
    start: tuple[float, float]
    end: tuple[float, float]
    thickness: float
    z_placement: float
    height: float
    storey: int
    label: str = "interior"  # interior | exterior
    material: str = "Concrete"


def detect_walls(
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
) -> List[Wall]:
    """Extract wall axes from a single storey's points.

    ``semantic_labels`` is per-point — if provided and ``cfg.use_ml_filter``
    is true, only wall-classified points feed the histogram.
    """
    if len(storey_points) == 0:
        log.warning("Storey %d: empty point cloud — no walls", storey_idx)
        return []

    # 1. ML filter
    pts_for_walls = storey_points
    if cfg.use_ml_filter and semantic_labels is not None:
        from cloud2bim.config import SegmentationConfig
        # Note: caller passes a labels mask filtered to the storey already
        wall_mask = semantic_labels.mask_for(SegmentationConfig().wall_classes)
        if wall_mask.any():
            log.info(
                "Storey %d: ML filter kept %d / %d points (%.1f%%)",
                storey_idx, int(wall_mask.sum()), len(storey_points),
                100 * wall_mask.sum() / len(storey_points),
            )
            pts_for_walls = storey_points[wall_mask]
        else:
            log.warning("Storey %d: no wall-labelled points; using all", storey_idx)

    # 2. Horizontal cross-section 30–130 cm above the floor.
    #    This height contains wall faces but mostly misses furniture tops and
    #    open spaces between floors. Using a fixed absolute height avoids the
    #    85-120% relative-band problem that collapsed when storey height varied.
    BAND_BOTTOM = 0.30   # m above floor
    BAND_TOP    = 1.30   # m above floor
    band_mask = (
        (pts_for_walls[:, 2] >= z_floor + BAND_BOTTOM) &
        (pts_for_walls[:, 2] <= z_floor + BAND_TOP)
    )
    if not band_mask.any():
        log.warning("Storey %d: no points in cross-section band [%.2f, %.2f]",
                    storey_idx, z_floor + BAND_BOTTOM, z_floor + BAND_TOP)
        return []
    points_2d = pts_for_walls[band_mask, :2]
    log.info("Storey %d: cross-section band %.2f–%.2f m, %s points",
             storey_idx, z_floor + BAND_BOTTOM, z_floor + BAND_TOP,
             f"{band_mask.sum():,}")

    # 3. PCA rotation
    pca_angle = dominant_angle(points_2d)
    do_rotate = abs(pca_angle) > np.radians(3)
    if do_rotate:
        log.info("Storey %d: applying PCA rotation %.1f°", storey_idx, np.degrees(pca_angle))
        points_2d = rotate_points_2d(points_2d, -pca_angle)
        if slab_polygon_xy is not None:
            slab_polygon_xy = rotate_points_2d(slab_polygon_xy, -pca_angle)

    # 4. Build 2D histogram + contour extraction
    pixel_size = pc_resolution * grid_coefficient
    segments = _extract_2d_segments(points_2d, pixel_size, cfg.min_length)
    if not segments:
        log.warning("Storey %d: no wall segments after histogram", storey_idx)
        return []
    log.info("Storey %d: %d raw wall segments", storey_idx, len(segments))

    # 5. Group parallel/collinear segments into wall axes
    parallel_groups, group_labels = _group_segments(segments, cfg.max_thickness)

    # 6. Add facade candidates from the swollen slab polygon
    if not exterior_scan and slab_polygon_xy is not None and len(slab_polygon_xy) >= 3:
        facade_segments = swell_polygon(slab_polygon_xy, cfg.exterior_thickness)
        facade_groups, facade_labels = _group_segments(facade_segments, cfg.max_thickness)
        for g in facade_groups:
            parallel_groups.append(g)
            group_labels.append("exterior")

    log.info("Storey %d: %d parallel wall groups", storey_idx, len(parallel_groups))

    # 7. Compute wall axes from groups, drop NaN/degenerate
    wall_axes: list[list[list[float]]] = []
    wall_thicknesses: list[float] = []
    wall_labels: list[str] = []
    for group, label in zip(parallel_groups, group_labels):
        axis, thickness = _calculate_wall_axis(group)
        if axis is None or _has_nan(axis):
            continue
        wall_axes.append(axis)
        wall_thicknesses.append(thickness)
        wall_labels.append(label)

    if not wall_axes:
        log.warning("Storey %d: no valid wall axes after filtering", storey_idx)
        return []

    # 8. Height validation — keep only axes that have points spanning at
    #    least 50% of the expected storey height. Singletons from furniture
    #    or clutter are typically short in Z even if they appear in the
    #    cross-section band.
    storey_height = z_ceiling - z_floor
    min_span = max(0.3, storey_height * 0.4)  # at least 40% of storey or 30 cm
    valid_axes, valid_thicknesses, valid_groups, valid_labels = [], [], [], []
    for ax, th, grp, lbl in zip(wall_axes, wall_thicknesses, valid_parallel_groups, wall_labels):
        # Collect points near this axis in full storey height
        near = pts_for_walls[
            distance_points_to_line(pts_for_walls[:, :2],
                                    np.array(ax[0]), np.array(ax[1])) < max(th, cfg.min_thickness) * 2
        ]
        if len(near) == 0:
            continue
        z_span = float(near[:, 2].max() - near[:, 2].min())
        if z_span < min_span:
            log.debug("Storey %d: discarding axis (z_span=%.2f < %.2f)", storey_idx, z_span, min_span)
            continue
        valid_axes.append(ax)
        valid_thicknesses.append(th)
        valid_groups.append(grp)
        valid_labels.append(lbl)

    wall_axes, wall_thicknesses = valid_axes, valid_thicknesses
    valid_parallel_groups, wall_labels = valid_groups, valid_labels
    log.info("Storey %d: %d axes survive height validation", storey_idx, len(wall_axes))

    if not wall_axes:
        log.warning("Storey %d: no walls after height validation", storey_idx)
        return []

    # 9. Snap intersections (NaN-safe)
    wall_axes = _adjust_intersections(wall_axes, cfg.max_thickness)

    # 9. Inverse PCA rotation
    if do_rotate:
        for ax in wall_axes:
            rot = rotate_points_2d(np.array(ax), pca_angle)
            ax[0] = rot[0].tolist()
            ax[1] = rot[1].tolist()

    # 10. Cap to safety limit
    if len(wall_axes) > cfg.max_walls_per_storey:
        log.warning(
            "Storey %d: clipping %d walls down to max_walls_per_storey=%d",
            storey_idx, len(wall_axes), cfg.max_walls_per_storey,
        )
        wall_axes = wall_axes[: cfg.max_walls_per_storey]
        wall_thicknesses = wall_thicknesses[: cfg.max_walls_per_storey]
        wall_labels = wall_labels[: cfg.max_walls_per_storey]

    # 11. Wall height comes from slab spacing; passed from pipeline
    wall_height = z_ceiling - z_floor
    walls = [
        Wall(
            start=tuple(ax[0]),
            end=tuple(ax[1]),
            thickness=t,
            z_placement=z_floor,
            height=wall_height,
            storey=storey_idx,
            label=lbl,
        )
        for ax, t, lbl in zip(wall_axes, wall_thicknesses, wall_labels)
    ]
    log.info("Storey %d: %d walls finalised", storey_idx, len(walls))
    return walls


# ── internal helpers ─────────────────────────────────────────────────────────

def _extract_2d_segments(points_2d: np.ndarray, pixel_size: float, min_length: float):
    """2D histogram → binary mask → contours → Douglas-Peucker segments."""
    x_min, y_min = points_2d.min(axis=0)
    x_max, y_max = points_2d.max(axis=0)
    xs = np.arange(x_min + 0.5 * pixel_size, x_max, pixel_size)
    ys = np.arange(y_min + 0.5 * pixel_size, y_max, pixel_size)
    if len(xs) < 2 or len(ys) < 2:
        return []
    grid, _, _ = np.histogram2d(points_2d[:, 0], points_2d[:, 1], bins=[xs, ys])
    grid = grid.T

    threshold = 0.01 * grid.max()
    mask = (grid > threshold).astype(np.uint8) * 255
    mask = closing(mask, footprint_rectangle((5, 5)))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    segments: list[list[list[float]]] = []
    for cnt in contours:
        cnt = cnt.reshape(-1, 2)
        approx = cv2.approxPolyDP(cnt.reshape(-1, 1, 2).astype(np.float32), 0.04 / pixel_size, True)
        approx = approx.reshape(-1, 2)
        for i in range(len(approx)):
            p1 = approx[i]
            p2 = approx[(i + 1) % len(approx)]
            wp1 = [float(p1[0] * pixel_size + x_min), float(p1[1] * pixel_size + y_min)]
            wp2 = [float(p2[0] * pixel_size + x_min), float(p2[1] * pixel_size + y_min)]
            length = np.hypot(wp2[0] - wp1[0], wp2[1] - wp1[1])
            if length >= min_length:
                segments.append([wp1, wp2])
    return segments


def _group_segments(segments, max_thickness: float):
    """Group parallel segments within ``max_thickness`` distance.

    Singletons are kept too — an unpaired segment usually still represents a
    real wall (one side scanned, the other obscured). Caller can tell them
    apart by group length.
    """
    grouped = []
    labels = []
    remaining = list(segments)
    while remaining:
        current = remaining.pop(0)
        group = [current]
        i = 0
        while i < len(remaining):
            other = remaining[i]
            if _segments_parallel(current, other) and _segments_close(current, other, max_thickness):
                group.append(other)
                remaining.pop(i)
            else:
                i += 1
        grouped.append(group)
        labels.append("interior")
    return grouped, labels


def _segments_parallel(s1, s2, angle_tol_deg: float = 5.0) -> bool:
    v1 = np.array(s1[1]) - np.array(s1[0])
    v2 = np.array(s2[1]) - np.array(s2[0])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return False
    cos_a = abs(np.dot(v1, v2) / (n1 * n2))
    cos_a = min(1.0, cos_a)
    return np.degrees(np.arccos(cos_a)) <= angle_tol_deg


def _segments_close(s1, s2, max_dist: float) -> bool:
    return any(
        np.linalg.norm(np.array(p1) - np.array(p2)) <= max_dist
        for p1 in s1 for p2 in s2
    )


def _calculate_wall_axis(group, default_thickness: float = 0.10):
    """Find wall axis from a group of parallel segments.

    Singleton groups: use the segment itself as the axis with a default
    thickness (the other face wasn't scanned, but the wall is still real).

    Returns ``(None, 0)`` only for truly degenerate (zero-length) groups.
    """
    if len(group) == 0:
        return None, 0.0

    lengths = [np.linalg.norm(np.array(s[1]) - np.array(s[0])) for s in group]
    longest = group[int(np.argmax(lengths))]
    direction = np.array(longest[1]) - np.array(longest[0])
    norm = float(np.linalg.norm(direction))
    if norm == 0:
        return None, 0.0

    if len(group) == 1:
        # Single-face wall — axis IS the segment, thickness is a guess
        return [list(longest[0]), list(longest[1])], default_thickness

    direction /= norm
    # Pick the second-longest segment (any partner that isn't the longest)
    other_idx = int(np.argsort(lengths)[-2])
    shorter = group[other_idx]
    mid_long = (np.array(longest[0]) + np.array(longest[1])) / 2
    mid_short = (np.array(shorter[0]) + np.array(shorter[1])) / 2
    mean_dist = float(np.linalg.norm(mid_long - mid_short))
    half = mean_dist / 2
    perp = np.array([-direction[1], direction[0]])
    axis = [
        list(np.array(longest[0]) - half * perp),
        list(np.array(longest[1]) - half * perp),
    ]
    return axis, mean_dist


def _has_nan(axis) -> bool:
    return any(not np.isfinite(c) for pt in axis for c in pt)


def _adjust_intersections(wall_axes, max_thickness: float):
    """Snap wall endpoints to nearby intersections. NaN-safe."""
    half = max_thickness / 2
    for i, ax1 in enumerate(wall_axes):
        for j, ax2 in enumerate(wall_axes):
            if i == j:
                continue
            inter = line_intersection(ax1, ax2)
            if inter is None:
                continue
            for k in range(2):
                if np.linalg.norm(np.array(ax1[k]) - np.array(inter)) <= half:
                    ax1[k] = list(inter)
                if np.linalg.norm(np.array(ax2[k]) - np.array(inter)) <= half:
                    ax2[k] = list(inter)
    return wall_axes
