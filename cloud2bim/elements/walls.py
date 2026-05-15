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
from cloud2bim.geometry.lines import line_intersection
from cloud2bim.geometry.pca import dominant_angle, rotate_points_2d
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
    cross_section_band: Optional[tuple[float, float]] = None,
    pca_angle: Optional[float] = None,
    out_contours: Optional[list] = None,
) -> List[Wall]:
    """Extract wall axes from a single storey's points.

    ``semantic_labels`` is per-point — if provided and ``cfg.use_ml_filter``
    is true, only wall-classified points feed the histogram.

    ``cross_section_band`` overrides the default 130–160 cm above-floor band.
    When set, the tuple is interpreted as absolute world Z (m) — useful when
    the user has hand-picked a band from the Z-histogram in the UI.

    ``pca_angle`` overrides the per-storey PCA computation. Pass the same
    angle that slab extraction used to keep walls and slab outlines in the
    same rotated frame.
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

    # 2. Horizontal cross-section.
    #    Default is 130–160 cm above the floor — that height contains wall
    #    faces but mostly misses furniture tops. The caller can override via
    #    ``cross_section_band`` after picking it from the Z-histogram in the
    #    UI; that's important when slabs were misdetected and z_floor is off.
    if cross_section_band is not None:
        band_lo, band_hi = float(cross_section_band[0]), float(cross_section_band[1])
        band_label = "absolute Z"
    else:
        band_lo = z_floor + 1.30
        band_hi = z_floor + 1.60
        band_label = "floor-relative"
    band_mask = (pts_for_walls[:, 2] >= band_lo) & (pts_for_walls[:, 2] <= band_hi)
    if not band_mask.any():
        log.warning(
            "Storey %d: no points in cross-section band [%.2f, %.2f]",
            storey_idx, band_lo, band_hi,
        )
        return []
    points_2d = pts_for_walls[band_mask, :2]
    log.info(
        "Storey %d: cross-section band %.2f–%.2f m (%s), %s points",
        storey_idx, band_lo, band_hi, band_label, f"{band_mask.sum():,}",
    )

    # 3. PCA rotation — use the caller-provided angle when given so walls
    #    and slabs end up in the same rotated frame.
    if pca_angle is None:
        pca_angle = dominant_angle(points_2d)
    do_rotate = abs(pca_angle) > np.radians(3)
    if do_rotate:
        log.info("Storey %d: applying PCA rotation %.1f°", storey_idx, np.degrees(pca_angle))
        points_2d = rotate_points_2d(points_2d, -pca_angle)
        if slab_polygon_xy is not None:
            slab_polygon_xy = rotate_points_2d(slab_polygon_xy, -pca_angle)

    # 4. Build 2D histogram + contour extraction
    pixel_size = pc_resolution * grid_coefficient
    segments, raw_contours = _extract_2d_segments(points_2d, pixel_size, cfg.min_length)
    if not segments:
        log.warning("Storey %d: no wall segments after histogram", storey_idx)
        return []
    log.info("Storey %d: %d raw wall segments", storey_idx, len(segments))

    # 4b. Merge collinear fragments before pairing — the contour-tracer often
    #     emits a wall as several short collinear pieces; merging gives the
    #     pairing step a clean view of "one wall = one segment".
    segments = _merge_collinear(segments, cfg.min_thickness, cfg.max_thickness)
    log.info("Storey %d: %d segments after collinear merge", storey_idx, len(segments))

    # 5. Strict pairing — parallel + within max_thickness + overlap along the
    #    wall length. Groups with 2+ segments are confident walls (both faces
    #    scanned, measured thickness).
    paired_groups, singletons = _group_segments_strict(segments, cfg.max_thickness)
    log.info(
        "Storey %d: %d two-sided pairs + %d singletons (one-faced walls)",
        storey_idx, len(paired_groups), len(singletons),
    )

    wall_axes: list[list[list[float]]] = []
    wall_thicknesses: list[float] = []
    wall_labels: list[str] = []

    # 5a. Interior walls — both faces visible, axis is the midline.
    for group in paired_groups:
        axis, thickness = _calculate_wall_axis(group)
        if axis is None or _has_nan(axis):
            continue
        seg_len = float(np.hypot(axis[1][0] - axis[0][0], axis[1][1] - axis[0][1]))
        if seg_len < cfg.min_length:
            continue
        wall_axes.append(axis)
        wall_thicknesses.append(max(thickness, cfg.min_thickness))
        wall_labels.append("wall")

    # 5b. One-faced walls — typical exterior walls scanned from inside only.
    #     The contour segment IS the inside face of the wall. Shift the axis
    #     OUTWARD (away from the slab centroid) by exterior_thickness/2 so
    #     the wall body grows outside the building instead of straddling the
    #     visible face — which used to put half the wall inside the room.
    centroid = None
    if not exterior_scan and slab_polygon_xy is not None and len(slab_polygon_xy) >= 3:
        centroid = (
            float(np.asarray(slab_polygon_xy)[:, 0].mean()),
            float(np.asarray(slab_polygon_xy)[:, 1].mean()),
        )

    for seg in singletons:
        a = np.array(seg[0], dtype=float)
        b = np.array(seg[1], dtype=float)
        length = float(np.linalg.norm(b - a))
        if length < cfg.singleton_min_length:
            continue
        if _has_nan([a.tolist(), b.tolist()]):
            continue

        if centroid is not None:
            mid = (a + b) / 2.0
            direction = b - a
            d_norm = float(np.linalg.norm(direction))
            if d_norm > 1e-9:
                unit = direction / d_norm
                normal = np.array([-unit[1], unit[0]])
                # Flip so it points outward (away from the slab centroid)
                to_centroid = np.array([centroid[0] - mid[0], centroid[1] - mid[1]])
                if np.dot(normal, to_centroid) > 0:
                    normal = -normal
                shift = normal * (cfg.exterior_thickness / 2.0)
                a = a + shift
                b = b + shift

        wall_axes.append([a.tolist(), b.tolist()])
        wall_thicknesses.append(cfg.exterior_thickness)
        wall_labels.append("wall")

    if not wall_axes:
        log.warning("Storey %d: no valid wall axes after filtering", storey_idx)
        return []

    # 8. Snap intersections (NaN-safe)
    wall_axes = _adjust_intersections(wall_axes, cfg.max_thickness)

    # 9. Inverse PCA rotation
    if do_rotate:
        for ax in wall_axes:
            rot = rotate_points_2d(np.array(ax), pca_angle)
            ax[0] = rot[0].tolist()
            ax[1] = rot[1].tolist()
        raw_contours = [rotate_points_2d(c, pca_angle) for c in raw_contours]

    if out_contours is not None:
        out_contours.extend(raw_contours)

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
    """2D histogram → binary mask → contours → Douglas-Peucker segments.

    Returns (segments, raw_contours) where raw_contours is a list of closed
    pixel-space contours (Nx2 arrays in world coords) — kept around so the
    DXF exporter can emit them verbatim as the "continuous line" the user
    sees in the cross-section preview.
    """
    x_min, y_min = points_2d.min(axis=0)
    x_max, y_max = points_2d.max(axis=0)
    xs = np.arange(x_min + 0.5 * pixel_size, x_max, pixel_size)
    ys = np.arange(y_min + 0.5 * pixel_size, y_max, pixel_size)
    if len(xs) < 2 or len(ys) < 2:
        return [], []
    grid, _, _ = np.histogram2d(points_2d[:, 0], points_2d[:, 1], bins=[xs, ys])
    grid = grid.T

    # v1 used an absolute density threshold of 0.01, so any bin with ≥1 point
    # was kept. A "1% of max" threshold (v2's earlier choice) dropped weak but
    # real wall signal — exactly the data the user pointed out was missing.
    mask = (grid > 0).astype(np.uint8) * 255
    mask = closing(mask, footprint_rectangle((5, 5)))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    raw_world_contours: list[np.ndarray] = []
    for cnt in contours:
        cnt = cnt.reshape(-1, 2).astype(np.float64)
        xs_world = cnt[:, 0] * pixel_size + x_min
        ys_world = cnt[:, 1] * pixel_size + y_min
        raw_world_contours.append(np.column_stack([xs_world, ys_world]))

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
    return segments, raw_world_contours


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


def _perpendicular_distance(s1, s2) -> float:
    """Perpendicular distance from s2[0] to the infinite line through s1."""
    a = np.array(s1[0], dtype=float)
    b = np.array(s1[1], dtype=float)
    p = np.array(s2[0], dtype=float)
    ab = b - a
    n = np.linalg.norm(ab)
    if n == 0:
        return float("inf")
    return float(abs(np.cross(ab, p - a)) / n)


def _segments_overlap(s1, s2, min_overlap: float) -> bool:
    """True if s2 projected onto s1's direction overlaps s1 by ≥ min_overlap.

    Port of v1's check_overlap_parallel_segments: rotates s1 to lie on the
    x-axis, projects s2 the same way, and measures x-axis overlap. Pairs
    that sit side-by-side along the wall length pass; pairs that are
    parallel but offset past each other don't.
    """
    a = np.array(s1[0], dtype=float)
    b = np.array(s1[1], dtype=float)
    angle = np.arctan2(b[1] - a[1], b[0] - a[0])
    c, s = np.cos(-angle), np.sin(-angle)
    R = np.array([[c, -s], [s, c]])
    p1 = R @ np.array(s1[0], dtype=float)
    p2 = R @ np.array(s1[1], dtype=float)
    q1 = R @ np.array(s2[0], dtype=float)
    q2 = R @ np.array(s2[1], dtype=float)
    x1a, x1b = sorted([float(p1[0]), float(p2[0])])
    x2a, x2b = sorted([float(q1[0]), float(q2[0])])
    start = max(x1a, x2a)
    end = min(x1b, x2b)
    return (end - start) >= min_overlap


def _merge_collinear(segments, min_thickness: float, max_distance: float):
    """Merge segments that are collinear (parallel + perpendicular distance
    below ``min_thickness``) and reasonably close to each other.

    Walls show up in the contour-tracer as several short collinear pieces;
    merging gives the pairing step a single segment per face. Mirrors the
    v1 approach but with iteration cap to avoid pathological inputs.
    """
    work = [list(s) for s in segments]
    out: list = []
    safety = 0
    while work and safety < 5 * (len(work) + 1):
        safety += 1
        base = work.pop(0)
        to_merge = [base]
        i = 0
        while i < len(work):
            other = work[i]
            if (_segments_parallel(base, other, angle_tol_deg=3)
                    and _perpendicular_distance(base, other) <= min_thickness
                    and _segments_close(base, other, max_distance)):
                to_merge.append(other)
                work.pop(i)
            else:
                i += 1
        if len(to_merge) == 1:
            out.append(base)
            continue
        # Find the two furthest endpoints among all merged-segment endpoints
        pts = [p for seg in to_merge for p in seg]
        merged = _furthest_pair(pts)
        # Re-feed the merged segment so it can pick up more collinear pieces
        work.append(merged)
    out.extend(work)
    return out


def _furthest_pair(points):
    """Return the two points with the greatest pairwise distance."""
    arr = np.asarray(points, dtype=float)
    best = (arr[0], arr[1])
    best_d = -1.0
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(arr[i] - arr[j]))
            if d > best_d:
                best_d = d
                best = (arr[i], arr[j])
    return [list(map(float, best[0])), list(map(float, best[1]))]


def _group_segments_strict(segments, max_thickness: float):
    """v1-style pair grouping.

    Two segments only form a pair when they are parallel, within
    ``max_thickness`` of each other AND overlap along their length by
    at least ``max_thickness`` (so a real wall's two faces line up,
    not just sit near each other in different rooms).

    Singletons are dropped from the wall list and returned separately so
    the caller can try to pair them against the building envelope.
    """
    grouped: list = []
    unpaired: list = []
    min_overlap = max_thickness
    work = [list(s) for s in segments]
    while work:
        current = work.pop(0)
        group = [current]
        i = 0
        while i < len(work):
            other = work[i]
            if (_segments_parallel(current, other)
                    and _segments_close(current, other, max_thickness)
                    and _segments_overlap(current, other, min_overlap)):
                group.append(other)
                work.pop(i)
            else:
                i += 1
        if len(group) >= 2:
            grouped.append(group)
        else:
            unpaired.append(current)
    return grouped, unpaired


def _calculate_wall_axis(group):
    """Wall axis = midline between the two scanned faces.

    Caller guarantees the group has ≥2 segments (singletons handled
    separately as facade candidates). Returns ``(None, 0)`` only for
    degenerate zero-length groups.
    """
    if len(group) < 2:
        return None, 0.0

    lengths = [np.linalg.norm(np.array(s[1]) - np.array(s[0])) for s in group]
    longest = group[int(np.argmax(lengths))]
    direction = np.array(longest[1]) - np.array(longest[0])
    norm = float(np.linalg.norm(direction))
    if norm == 0:
        return None, 0.0
    direction = direction / norm
    perp = np.array([-direction[1], direction[0]])

    # Project each midpoint onto the perpendicular axis to find the two
    # extreme offsets. The wall midline sits halfway between them.
    base_mid = (np.array(longest[0]) + np.array(longest[1])) / 2
    offsets = []
    for seg in group:
        m = (np.array(seg[0]) + np.array(seg[1])) / 2
        offsets.append(float(np.dot(m - base_mid, perp)))
    o_min, o_max = min(offsets), max(offsets)
    thickness = float(abs(o_max - o_min))
    if thickness <= 0:
        return None, 0.0
    centre_offset = (o_min + o_max) / 2
    axis = [
        list(np.array(longest[0]) + centre_offset * perp),
        list(np.array(longest[1]) + centre_offset * perp),
    ]
    return axis, thickness


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
