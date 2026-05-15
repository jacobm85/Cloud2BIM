"""V1 wall detection ported from aronfothi/Cloud2BIM master:app/core/aux_functions.py.

Bit-for-bit reproduction of ``identify_walls`` from the aronfothi fork
(https://github.com/aronfothi/Cloud2BIM) — which is the v1 baseline the
user has confirmed works. Notably the aronfothi fork does NOT apply
PCA rotation (the upstream VaclavNezerka/Cloud2BIM master added it
later, which produced inferior walls on the user's data).

Adaptations made:
    * input: full storey ``pointcloud`` (Nx3) instead of split arrays
    * output: ``List[Wall]`` instead of v1's tuple of parallel lists
    * dropped the wall-point-assignment / per-wall rotation block —
      that data was used by v1's own opening detection, which we don't
      share

Everything else (Z-band 85–120% of pc height, absolute density
threshold 0.01, +1 pixel contour shift, swell-polygon-pair for
exterior walls, midpoint wall axis) is intentionally preserved.
"""
from __future__ import annotations

import math
from typing import List, Optional

import cv2
import numpy as np
from skimage.morphology import closing, footprint_rectangle

from cloud2bim.config import WallConfig
from cloud2bim.elements.walls import Wall
from cloud2bim.logging import get_logger

log = get_logger(__name__)


# ── Geometric helpers (mirrors aronfothi) ──────────────────────────────────

def _get_line_segments(contour, pixel_size, segment_approximation_tolerance=0.02):
    """Douglas-Peucker simplification → list of (pt, pt) segments."""
    epsilon = segment_approximation_tolerance / pixel_size
    approx = cv2.approxPolyDP(contour, epsilon, True)
    segments = []
    for i in range(len(approx)):
        segment = [tuple(approx[i - 1][0]), tuple(approx[i][0])]
        segments.append(segment)
    return segments


def _distance_between_points(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def _distance_point_to_line(point, line_start, line_end):
    line_start = np.array(line_start, dtype=float)
    line_end = np.array(line_end, dtype=float)
    point = np.array(point, dtype=float)
    line_vec = line_end - line_start
    line_length = float(np.linalg.norm(line_vec))
    if np.isclose(line_length, 0):
        return float("nan")
    line_unit = line_vec / line_length
    proj_len = float(np.dot(point - line_start, line_unit))
    closest = line_start + proj_len * line_unit
    return float(np.linalg.norm(point - closest))


def _angle_between_segments(seg1, seg2) -> float:
    dx1, dy1 = seg1[1][0] - seg1[0][0], seg1[1][1] - seg1[0][1]
    dx2, dy2 = seg2[1][0] - seg2[0][0], seg2[1][1] - seg2[0][1]
    m1 = math.hypot(dx1, dy1)
    m2 = math.hypot(dx2, dy2)
    if m1 * m2 == 0:
        return 90.0
    cosa = (dx1 * dx2 + dy1 * dy2) / (m1 * m2)
    cosa = max(-1.0, min(1.0, cosa))
    return math.degrees(math.acos(cosa))


def _segments_angle(seg1, seg2, angle_tolerance=3) -> bool:
    a = _angle_between_segments(seg1, seg2)
    return abs(a) < angle_tolerance or abs(a - 180) < angle_tolerance


def _perpendicular_distance_between_segments(seg1, seg2) -> float:
    if _segments_angle(seg1, seg2):
        return _distance_point_to_line(seg2[0], seg1[0], seg1[1])
    return float("inf")


def _segments_collinearity_check(seg1, seg2, min_thickness, max_distance) -> bool:
    close_enough = any(
        _distance_between_points(p1, p2) <= max_distance
        for p1 in seg1 for p2 in seg2
    )
    collinear = any(
        _distance_point_to_line(point, seg1[0], seg1[1]) < (min_thickness / 2)
        for point in seg2
    )
    return close_enough and collinear


def _find_furthest_points(all_points):
    pts = np.asarray(all_points, dtype=float)
    if len(pts) <= 3:
        max_d, sp, ep = -1.0, None, None
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = float(np.linalg.norm(pts[i] - pts[j]))
                if d > max_d:
                    max_d, sp, ep = d, pts[i], pts[j]
        return (sp.tolist() if sp is not None else None,
                ep.tolist() if ep is not None else None)
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts)
        hull_pts = pts[hull.vertices]
    except Exception:
        hull_pts = pts
    max_d, sp, ep = -1.0, hull_pts[0], hull_pts[0]
    n = len(hull_pts)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(hull_pts[i] - hull_pts[j]))
            if d > max_d:
                max_d, sp, ep = d, hull_pts[i], hull_pts[j]
    return sp.tolist(), ep.tolist()


def _merge_collinear_segments(segments, min_thickness, max_distance):
    final_segments = []
    work = [list(s) for s in segments]
    while work:
        base = work[0]
        to_merge = [base]
        for other in work[1:]:
            if (_segments_collinearity_check(base, other, min_thickness, max_distance)
                    and _segments_angle(base, other, angle_tolerance=3)):
                to_merge.append(other)
        if len(to_merge) > 1:
            all_points = [pt for seg in to_merge for pt in seg]
            sp, ep = _find_furthest_points(all_points)
            merged = [sp, ep]
            work.append(merged)
        else:
            final_segments.append(base)
        for seg in to_merge:
            work.remove(seg)
    return final_segments


def _check_overlap_parallel_segments(seg1, seg2, min_overlap):
    a = math.atan2(seg1[1][1] - seg1[0][1], seg1[1][0] - seg1[0][0])
    c, s = math.cos(-a), math.sin(-a)
    R = np.array([[c, -s], [s, c]])
    p1 = R @ np.array(seg1[0], dtype=float)
    p2 = R @ np.array(seg1[1], dtype=float)
    q1 = R @ np.array(seg2[0], dtype=float)
    q2 = R @ np.array(seg2[1], dtype=float)
    x1a, x1b = sorted([p1[0], p2[0]])
    x2a, x2b = sorted([q1[0], q2[0]])
    start = max(x1a, x2a)
    end = min(x1b, x2b)
    return (end - start) > min_overlap


def _group_segments(segments, max_wall_thickness, wall_label, angle_tolerance=5):
    """v1 pair grouping. Returns (pairs, labels, leftover_singletons)."""
    grouped = []
    wall_labels = []
    facade_wall_candidate = []
    work = [list(s) for s in segments]
    while work:
        current = work.pop(0)
        parallel_group = [current]
        i = 0
        while i < len(work):
            seg = work[i]
            if (_segments_angle(current, seg, angle_tolerance)
                    and any(_distance_between_points(p1, p2) <= max_wall_thickness
                            for p1 in current for p2 in seg)
                    and _check_overlap_parallel_segments(current, seg,
                                                         min_overlap=max_wall_thickness)):
                parallel_group.append(seg)
                work.pop(i)
            else:
                i += 1
        if len(parallel_group) >= 2:
            grouped.append(parallel_group)
            wall_labels.append(wall_label)
        else:
            facade_wall_candidate.append(current)
    return grouped, wall_labels, facade_wall_candidate


def _calculate_wall_axis(group):
    """v1 calculate_wall_axis: midpoint between the two longest segments."""
    if len(group) < 2:
        return None, 0.0
    lengths = [_distance_between_points(s[0], s[1]) for s in group]
    longer = group[int(np.argmax(lengths))]
    shorter = group[1 - int(np.argmax(lengths))]
    dx = longer[1][0] - longer[0][0]
    dy = longer[1][1] - longer[0][1]
    norm = math.hypot(dx, dy)
    if norm == 0:
        return None, 0.0
    direction = (dx / norm, dy / norm)
    mean_distance = float(np.mean([
        _perpendicular_distance_between_segments(longer, shorter),
        _perpendicular_distance_between_segments(shorter, longer),
    ]))
    half = mean_distance / 2.0
    axis_start = [longer[0][0] - half * direction[1], longer[0][1] + half * direction[0]]
    axis_end = [longer[1][0] - half * direction[1], longer[1][1] + half * direction[0]]
    dsum_a = sum(_distance_between_points(pt, axis_start) + _distance_between_points(pt, axis_end)
                 for pt in longer + shorter)
    axis_start_f = [longer[0][0] + half * direction[1], longer[0][1] - half * direction[0]]
    axis_end_f = [longer[1][0] + half * direction[1], longer[1][1] - half * direction[0]]
    dsum_b = sum(_distance_between_points(pt, axis_start_f) + _distance_between_points(pt, axis_end_f)
                 for pt in longer + shorter)
    if dsum_b < dsum_a:
        axis_start, axis_end = axis_start_f, axis_end_f
    return [axis_start, axis_end], mean_distance


def _line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
    div = xdiff[0] * ydiff[1] - xdiff[1] * ydiff[0]
    if div == 0:
        return None
    d = ((line1[0][0] * line1[1][1] - line1[0][1] * line1[1][0]),
         (line2[0][0] * line2[1][1] - line2[0][1] * line2[1][0]))
    x = (d[0] * xdiff[1] - d[1] * xdiff[0]) / div
    y = (d[0] * ydiff[1] - d[1] * ydiff[0]) / div
    return x, y


def _adjust_intersections(wall_axes, max_wall_thickness):
    half = max_wall_thickness / 2.0
    for i, axis1 in enumerate(wall_axes):
        for j, axis2 in enumerate(wall_axes):
            if i == j:
                continue
            inter = _line_intersection(axis1, axis2)
            if inter is None or any(np.isnan(v) or np.isinf(v) for v in inter):
                continue
            for k in range(2):
                if _distance_between_points(axis1[k], inter) <= half:
                    axis1[k] = list(inter)
                if _distance_between_points(axis2[k], inter) <= half:
                    axis2[k] = list(inter)
    return wall_axes


def _swell_polygon(vertices, thickness):
    """v1 swell_polygon: offset each edge OUTWARD by ``thickness``."""
    vertices = np.asarray(vertices, dtype=float)
    if len(vertices) < 3:
        return []
    centroid = vertices.mean(axis=0)
    offset_segments = []
    n = len(vertices)
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        edge = p2 - p1
        edge_len = float(np.linalg.norm(edge))
        if edge_len == 0:
            continue
        normal = np.array([-edge[1], edge[0]]) / edge_len
        midpoint = (p1 + p2) / 2.0
        direction = midpoint - centroid
        if np.dot(normal, direction) < 0:
            normal = -normal
        op1 = (p1 + thickness * normal).tolist()
        op2 = (p2 + thickness * normal).tolist()
        offset_segments.append([op1, op2])
    return offset_segments


# ── Main entry point ───────────────────────────────────────────────────────

def detect_walls_v1(
    storey_points: np.ndarray,
    z_floor: float,
    z_ceiling: float,
    storey_idx: int,
    cfg: WallConfig,
    pc_resolution: float,
    grid_coefficient: int,
    slab_polygon_xy: Optional[np.ndarray] = None,
    exterior_scan: bool = False,
    **_unused,
) -> List[Wall]:
    """V1 wall detection. Mirrors master:identify_walls.

    Parameters not in v1 (``semantic_labels``, ``cross_section_band``,
    ``pca_angle``, ``out_contours``, ``lower_section_band``) are accepted
    via ``**_unused`` and ignored, so the dispatch in pipeline / stepwise
    can pass them unconditionally.
    """
    if len(storey_points) == 0:
        log.warning("Storey %d: empty point cloud — no walls", storey_idx)
        return []

    x_coords_arr = storey_points[:, 0].astype(float)
    y_coords_arr = storey_points[:, 1].astype(float)
    z_coords_arr = storey_points[:, 2].astype(float)

    # Z-band: 85–120% of point cloud height (v1's z_section_boundaries).
    z_section_boundaries = (0.85, 1.20)
    z_min_pc = float(z_coords_arr.min())
    z_max_pc = float(z_coords_arr.max())
    z_band_lo = z_min_pc + z_section_boundaries[0] * (z_max_pc - z_min_pc)
    z_band_hi = z_min_pc + z_section_boundaries[1] * (z_max_pc - z_min_pc)
    z_mask = (z_coords_arr >= z_band_lo) & (z_coords_arr <= z_band_hi)
    if not np.any(z_mask):
        log.warning("Storey %d: no points in v1 Z-band [%.2f, %.2f]",
                    storey_idx, z_band_lo, z_band_hi)
        return []
    points_2d = np.column_stack([x_coords_arr[z_mask], y_coords_arr[z_mask]])
    log.info("Storey %d (v1): Z-band %.2f–%.2f m, %d points",
             storey_idx, z_band_lo, z_band_hi, int(z_mask.sum()))

    # NOTE: aronfothi/Cloud2BIM master does NOT apply PCA rotation here.
    # Slab polygon stays in original frame for the swell-pair step.
    if slab_polygon_xy is not None:
        slab_polygon_xy = np.asarray(slab_polygon_xy)

    # 2D histogram → binary mask
    pixel_size = pc_resolution * grid_coefficient
    x_min, y_min = float(points_2d[:, 0].min()), float(points_2d[:, 1].min())
    x_max, y_max = float(points_2d[:, 0].max()), float(points_2d[:, 1].max())
    x_values_full = np.arange(x_min + 0.5 * pixel_size, x_max, pixel_size)
    y_values_full = np.arange(y_min + 0.5 * pixel_size, y_max, pixel_size)
    if len(x_values_full) < 2 or len(y_values_full) < 2:
        log.warning("Storey %d (v1): degenerate histogram grid", storey_idx)
        return []
    grid_full, _, _ = np.histogram2d(points_2d[:, 0], points_2d[:, 1],
                                     bins=[x_values_full, y_values_full])
    grid_full = grid_full.T

    threshold = 0.01  # absolute density threshold, v1 default
    binary_image = (grid_full > threshold).astype(np.uint8) * 255
    binary_image = closing(binary_image, footprint_rectangle((5, 5)))

    # Contours + grid-bug fix shift (+1, +1)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    shift_x, shift_y = 1, 1
    adjusted_contours = []
    for cnt in contours:
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        adjusted_contours.append(cv2.transform(cnt, M))

    # Extract segments
    all_segments = []
    for cnt in adjusted_contours:
        all_segments.extend(_get_line_segments(cnt, pixel_size,
                                               segment_approximation_tolerance=0.04))
    # Pixel → world
    segments_world = [
        [[p[0] * pixel_size + x_min, p[1] * pixel_size + y_min] for p in seg]
        for seg in all_segments
    ]
    # Length filter
    filtered_segments = [
        seg for seg in segments_world
        if _distance_between_points(seg[0], seg[1]) >= cfg.min_length
    ]
    log.info("Storey %d (v1): %d raw → %d filtered segments",
             storey_idx, len(segments_world), len(filtered_segments))

    # Merge collinear
    final_wall_segments = _merge_collinear_segments(
        filtered_segments.copy(), cfg.min_thickness, cfg.max_thickness,
    )
    log.info("Storey %d (v1): %d segments after collinear merge",
             storey_idx, len(final_wall_segments))

    # Pair-group interior walls
    parallel_groups, wall_labels, facade_candidates = _group_segments(
        final_wall_segments, cfg.max_thickness, "interior",
    )

    # Pair singletons against the swelled slab polygon for facade walls
    if not exterior_scan and slab_polygon_xy is not None and len(slab_polygon_xy) >= 3:
        swollen = _swell_polygon(np.asarray(slab_polygon_xy), cfg.exterior_thickness)
        facade_candidates.extend(swollen)
        facade_groups, facade_labels, _ = _group_segments(
            facade_candidates, cfg.max_thickness, "exterior",
        )
        parallel_groups.extend(facade_groups)
        wall_labels.extend(facade_labels)

    log.info("Storey %d (v1): %d wall groups (interior + facade)",
             storey_idx, len(parallel_groups))

    # Calculate axis + thickness per group
    wall_axes: list = []
    wall_thicknesses: list = []
    valid_labels: list = []
    for group, label in zip(parallel_groups, wall_labels):
        axis, thickness = _calculate_wall_axis(group)
        if axis is None:
            continue
        if any(np.isnan(c) for pt in axis for c in pt):
            continue
        wall_axes.append(axis)
        wall_thicknesses.append(thickness)
        valid_labels.append(label)

    if not wall_axes:
        log.warning("Storey %d (v1): no valid wall axes", storey_idx)
        return []

    # Snap intersections
    wall_axes = _adjust_intersections(wall_axes, cfg.max_thickness)

    # No inverse PCA rotation needed — aronfothi v1 never rotated.

    # Safety cap
    if len(wall_axes) > cfg.max_walls_per_storey:
        log.warning("Storey %d (v1): clipping %d → max_walls_per_storey=%d",
                    storey_idx, len(wall_axes), cfg.max_walls_per_storey)
        wall_axes = wall_axes[: cfg.max_walls_per_storey]
        wall_thicknesses = wall_thicknesses[: cfg.max_walls_per_storey]
        valid_labels = valid_labels[: cfg.max_walls_per_storey]

    wall_height = z_ceiling - z_floor
    walls: List[Wall] = []
    for ax, t, lbl in zip(wall_axes, wall_thicknesses, valid_labels):
        walls.append(Wall(
            start=(float(ax[0][0]), float(ax[0][1])),
            end=(float(ax[1][0]), float(ax[1][1])),
            thickness=float(max(t, cfg.min_thickness)),
            z_placement=z_floor,
            height=wall_height,
            storey=storey_idx,
            label=lbl,
        ))
    log.info("Storey %d (v1): %d walls finalised", storey_idx, len(walls))
    return walls
