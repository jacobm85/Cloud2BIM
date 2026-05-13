"""Polygon operations — contour smoothing and offsetting."""
from __future__ import annotations

import math

import cv2
import numpy as np


def smooth_contour(x_contour: np.ndarray, y_contour: np.ndarray, epsilon: float) -> tuple[np.ndarray, np.ndarray]:
    """Douglas-Peucker contour simplification.

    Snaps near-axis-aligned segments to perfect axis alignment so corners
    come out clean.
    """
    pts = np.column_stack([x_contour, y_contour]).astype(np.float32)
    pts = pts.reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(pts, epsilon, True).reshape(-1, 2)

    # Snap nearly-axis-aligned segments
    snapped = approx.copy()
    n = len(snapped)
    for i in range(n):
        j = (i + 1) % n
        dx = snapped[j, 0] - snapped[i, 0]
        dy = snapped[j, 1] - snapped[i, 1]
        if abs(dx) < epsilon * 2 and abs(dy) > abs(dx):
            snapped[j, 0] = snapped[i, 0]
        elif abs(dy) < epsilon * 2 and abs(dx) > abs(dy):
            snapped[j, 1] = snapped[i, 1]
    return snapped[:, 0], snapped[:, 1]


def swell_polygon(vertices, thickness: float) -> list:
    """Offset a closed polygon outward by ``thickness/2`` and return its segments."""
    verts = np.asarray(vertices, dtype=float)
    if len(verts) < 3:
        return []
    # Find centroid
    cx, cy = verts.mean(axis=0)
    segments = []
    n = len(verts)
    for i in range(n):
        p1, p2 = verts[i], verts[(i + 1) % n]
        # Outward normal
        edge = p2 - p1
        edge_len = math.hypot(edge[0], edge[1])
        if edge_len == 0:
            continue
        normal = np.array([-edge[1], edge[0]]) / edge_len
        # Flip if pointing inward
        midpoint = (p1 + p2) / 2
        to_center = np.array([cx - midpoint[0], cy - midpoint[1]])
        if np.dot(normal, to_center) > 0:
            normal = -normal
        offset = normal * (thickness / 2)
        segments.append([list(p1 + offset), list(p2 + offset)])
    return segments
