"""Slab detection from a 1D Z-histogram of point density.

Algorithm (unchanged from Cloud2BIM v1):
    1. Bin point Z values, find peaks above density threshold
    2. Each peak pair = bottom + top of a slab
    3. For each slab, build a 2D occupancy mask in XY and extract the hull

Returns Slab dataclasses ready for IFC export.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np
from scipy.signal import find_peaks
from skimage.morphology import closing, footprint_rectangle

from cloud2bim.config import SlabConfig
from cloud2bim.geometry.polygon import smooth_contour
from cloud2bim.logging import get_logger

log = get_logger(__name__)


@dataclass
class Slab:
    """Detected horizontal element (floor or ceiling slab)."""
    bottom_z: float
    thickness: float
    polygon_x: np.ndarray  # closed polygon, XY in metres
    polygon_y: np.ndarray
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))


def detect_slabs(
    points_xyz: np.ndarray,
    cfg: SlabConfig,
    *,
    bottom_floor_thickness: float | None = None,
    top_floor_thickness: float | None = None,
) -> List[Slab]:
    """Detect horizontal slabs from a point cloud.

    ``bottom_floor_thickness`` / ``top_floor_thickness`` override config when
    they're known from external data (e.g. drawings).
    """
    bfs_t = bottom_floor_thickness or cfg.bottom_floor_thickness
    tfs_t = top_floor_thickness or cfg.top_floor_thickness

    if len(points_xyz) == 0:
        log.warning("Empty point cloud — no slabs to detect")
        return []

    z = points_xyz[:, 2]
    z_min, z_max = float(z.min()), float(z.max())
    log.info("Slab Z-range: %.2f .. %.2f m", z_min, z_max)

    # 1D Z histogram with z_step bins
    bin_edges = np.arange(z_min, z_max + cfg.z_step, cfg.z_step)
    hist, _ = np.histogram(z, bins=bin_edges)
    if hist.max() == 0:
        log.warning("Empty Z-histogram — no slabs detected")
        return []

    # Peaks above 25 % of max density (more permissive than v1 — many real
    # scans have dense floor + sparser ceiling that miss a 0.6× cut-off).
    threshold = 0.25 * hist.max()
    min_separation = max(2, int(0.5 / cfg.z_step))
    peaks, _ = find_peaks(hist, height=threshold, distance=min_separation)
    log.info("Found %d horizontal-surface peaks", len(peaks))

    if len(peaks) < 2:
        log.warning(
            "Fewer than 2 peaks (%d) — need a floor and a ceiling. "
            "Try lowering z_step or check Z coverage.",
            len(peaks),
        )
        return []

    # Pair peaks into (bottom, top) slabs
    horiz_z = [bin_edges[p] for p in peaks]
    n_slab_candidates = len(horiz_z) // 2 + (len(horiz_z) % 2)

    slabs: List[Slab] = []
    for i in range(n_slab_candidates):
        if i == 0:
            slab_top_z = float(np.median(_slice_points(points_xyz, horiz_z[0], cfg.z_step)[:, 2]))
            slab_bottom_z = slab_top_z - bfs_t
            thickness = bfs_t
            slice_pts = _slice_points(points_xyz, horiz_z[0], cfg.z_step)
        elif i == n_slab_candidates - 1 and len(horiz_z) % 2 == 1:
            slab_bottom_z = float(np.median(_slice_points(points_xyz, horiz_z[-1], cfg.z_step)[:, 2]))
            thickness = tfs_t
            slice_pts = _slice_points(points_xyz, horiz_z[-1], cfg.z_step)
        else:
            idx_bot, idx_top = 2 * i - 1, 2 * i
            if idx_top >= len(horiz_z):
                break
            bottom_pts = _slice_points(points_xyz, horiz_z[idx_bot], cfg.z_step)
            top_pts = _slice_points(points_xyz, horiz_z[idx_top], cfg.z_step)
            slab_bottom_z = float(np.median(bottom_pts[:, 2]))
            slab_top_z = float(np.median(top_pts[:, 2]))
            thickness = max(0.05, slab_top_z - slab_bottom_z)
            slice_pts = np.vstack([bottom_pts, top_pts])

        if len(slice_pts) < 100:
            log.warning("Slab %d: too few points (%d) — skipping", i, len(slice_pts))
            continue

        x, y = _hull_from_points(slice_pts, cfg.pc_resolution, cfg.grid_coefficient)
        if x is None or len(x) < 3:
            log.warning("Slab %d: failed to build polygon — skipping", i)
            continue

        slabs.append(
            Slab(
                bottom_z=slab_bottom_z,
                thickness=thickness,
                polygon_x=x,
                polygon_y=y,
                points=slice_pts,
            )
        )
        log.info(
            "Slab %d: bottom=%.3f m, thickness=%.0f mm, hull pts=%d",
            i, slab_bottom_z, thickness * 1000, len(x),
        )

    return slabs


def _slice_points(points: np.ndarray, z_center: float, z_step: float) -> np.ndarray:
    """Return points within ±z_step of the centre."""
    mask = np.abs(points[:, 2] - z_center) <= z_step
    return points[mask]


def _hull_from_points(
    points: np.ndarray,
    pc_resolution: float,
    grid_coefficient: int,
    dilation_m: float = 1.0,
    erosion_m: float = 1.0,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Build a 2D occupancy mask and extract the largest contour as a polygon."""
    pixel_size = pc_resolution * grid_coefficient
    x_min, y_min = points[:, 0].min(), points[:, 1].min()
    x_max, y_max = points[:, 0].max(), points[:, 1].max()
    xs = np.arange(x_min, x_max + pixel_size, pixel_size)
    ys = np.arange(y_min, y_max + pixel_size, pixel_size)
    if len(xs) < 2 or len(ys) < 2:
        return None, None

    grid, _, _ = np.histogram2d(points[:, 0], points[:, 1], bins=[xs, ys])
    mask = (grid > 0).astype(np.uint8) * 255
    mask = mask.T

    # Morphological cleanup
    dilate_px = max(1, int(dilation_m / pixel_size))
    erode_px = max(1, int(erosion_m / pixel_size))
    mask = closing(mask, footprint_rectangle((5, 5)))
    if dilate_px > 1:
        mask = cv2.dilate(mask, np.ones((dilate_px, dilate_px), np.uint8))
    if erode_px > 1:
        mask = cv2.erode(mask, np.ones((erode_px, erode_px), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None

    largest = max(contours, key=cv2.contourArea).reshape(-1, 2)
    # Pixel → world
    x_world = largest[:, 0] * pixel_size + x_min
    y_world = largest[:, 1] * pixel_size + y_min
    return smooth_contour(x_world, y_world, epsilon=pixel_size * 2)
