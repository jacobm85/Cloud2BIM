"""PCA-based orientation detection for non-axis-aligned buildings."""
from __future__ import annotations

import numpy as np


def dominant_angle(points_2d: np.ndarray) -> float:
    """Return dominant orientation angle (radians) of a 2D point set.

    Normalised to [-pi/4, pi/4] so only corrections up to 45° are applied.
    Returns 0.0 for tiny point sets where PCA is meaningless.
    """
    pts = np.asarray(points_2d)
    if len(pts) < 10:
        return 0.0
    cov = np.cov(pts.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    dominant = eigvecs[:, np.argmax(eigvals)]
    angle = np.arctan2(dominant[1], dominant[0])
    angle = angle % (np.pi / 2)
    if angle > np.pi / 4:
        angle -= np.pi / 2
    return float(angle)


def rotate_points_2d(pts: np.ndarray, angle: float) -> np.ndarray:
    """Rotate (N,2) points by angle (radians) around origin."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return (R @ np.asarray(pts).T).T
