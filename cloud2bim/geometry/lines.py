"""Line segment geometry — distances and intersections.

All functions are NaN-safe: if inputs contain NaN/inf or produce numerically
unstable results, they return None or NaN-marked arrays instead of silently
propagating bad data into downstream histogram calculations.
"""
from __future__ import annotations

import numpy as np


def segment_length(seg) -> float:
    """Euclidean length of a 2-point segment."""
    p1, p2 = np.asarray(seg[0]), np.asarray(seg[1])
    return float(np.linalg.norm(p2 - p1))


def distance_point_to_line(point, line_start, line_end) -> float:
    """Perpendicular distance from a single point to a line segment."""
    p = np.asarray(point, dtype=float)
    a = np.asarray(line_start, dtype=float)
    b = np.asarray(line_end, dtype=float)
    line_vec = b - a
    line_len_sq = float(np.dot(line_vec, line_vec))
    if line_len_sq == 0.0:
        return float(np.linalg.norm(p - a))
    t = max(0.0, min(1.0, float(np.dot(p - a, line_vec) / line_len_sq)))
    proj = a + t * line_vec
    return float(np.linalg.norm(p - proj))


def distance_points_to_line(points: np.ndarray, line_start, line_end) -> np.ndarray:
    """Vectorised perpendicular distances from many points to one segment."""
    pts = np.asarray(points, dtype=float)
    a = np.asarray(line_start, dtype=float)
    b = np.asarray(line_end, dtype=float)
    line_vec = b - a
    line_len = float(np.linalg.norm(line_vec))
    if line_len == 0.0:
        return np.linalg.norm(pts - a, axis=1)
    line_unit = line_vec / line_len
    rel = pts - a
    proj_len = rel @ line_unit
    on_seg = (proj_len >= 0) & (proj_len <= line_len)
    closest = np.outer(proj_len, line_unit) + a
    perp = np.linalg.norm(pts - closest, axis=1)
    d_start = np.linalg.norm(pts - a, axis=1)
    d_end = np.linalg.norm(pts - b, axis=1)
    return np.where(on_seg, perp, np.minimum(d_start, d_end))


def line_intersection(line1, line2) -> tuple[float, float] | None:
    """Intersection point of two infinite lines, or None.

    Returns None when:
        - lines are parallel (det == 0)
        - either line contains NaN/inf
        - the intersection itself is NaN/inf (numerical instability)
    """
    a1, a2 = np.asarray(line1[0], dtype=float), np.asarray(line1[1], dtype=float)
    b1, b2 = np.asarray(line2[0], dtype=float), np.asarray(line2[1], dtype=float)

    if not (np.all(np.isfinite(a1)) and np.all(np.isfinite(a2))
            and np.all(np.isfinite(b1)) and np.all(np.isfinite(b2))):
        return None

    xdiff = (a1[0] - a2[0], b1[0] - b2[0])
    ydiff = (a1[1] - a2[1], b1[1] - b2[1])
    div = xdiff[0] * ydiff[1] - xdiff[1] * ydiff[0]
    if div == 0:
        return None

    d = (
        a1[0] * a2[1] - a1[1] * a2[0],
        b1[0] * b2[1] - b1[1] * b2[0],
    )
    x = (d[0] * xdiff[1] - d[1] * xdiff[0]) / div
    y = (d[0] * ydiff[1] - d[1] * ydiff[0]) / div

    if not (np.isfinite(x) and np.isfinite(y)):
        return None
    return float(x), float(y)
