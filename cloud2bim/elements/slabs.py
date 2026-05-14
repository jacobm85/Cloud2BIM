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


@dataclass
class ZHistogram:
    """Z-density histogram + detected peak positions.

    Exposed so the UI can render the histogram alongside slab markers and
    cross-section bands.
    """
    bin_centers: np.ndarray   # m
    counts: np.ndarray        # per bin
    peak_z: list[float]       # m — detected horizontal-surface peaks
    threshold: float          # density cut-off used for peak selection


def compute_z_histogram(
    points_xyz: np.ndarray,
    z_step: float,
    peak_height_ratio: float = 0.25,
) -> ZHistogram:
    """Compute the Z-density histogram and pick horizontal-surface peaks."""
    if len(points_xyz) == 0:
        return ZHistogram(np.empty(0), np.empty(0), [], 0.0)
    z = points_xyz[:, 2]
    z_min, z_max = float(z.min()), float(z.max())
    bin_edges = np.arange(z_min, z_max + z_step, z_step)
    hist, _ = np.histogram(z, bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    if hist.max() == 0:
        return ZHistogram(bin_centers, hist, [], 0.0)
    threshold = peak_height_ratio * hist.max()
    min_separation = max(2, int(0.5 / z_step))
    peak_idx, _ = find_peaks(hist, height=threshold, distance=min_separation)
    peak_z = [float(bin_centers[p]) for p in peak_idx]
    return ZHistogram(bin_centers, hist, peak_z, float(threshold))


def detect_slabs(
    points_xyz: np.ndarray,
    cfg: SlabConfig,
    *,
    bottom_floor_thickness: float | None = None,
    top_floor_thickness: float | None = None,
) -> List[Slab]:
    """Detect horizontal slabs from a point cloud.

    Each Z-density peak is one horizontal surface. Two adjacent peaks are
    only paired as bottom+top of the same slab when they sit within
    ``max_slab_thickness`` of each other — typical RC slabs are 15–35 cm.
    Anything wider apart is two different storeys and each peak becomes a
    standalone slab (treated as a floor surface, with the default thickness
    used to derive its bottom).

    ``bottom_floor_thickness`` / ``top_floor_thickness`` override config when
    known from external data (e.g. drawings).
    """
    bfs_t = bottom_floor_thickness or cfg.bottom_floor_thickness
    tfs_t = top_floor_thickness or cfg.top_floor_thickness

    if len(points_xyz) == 0:
        log.warning("Empty point cloud — no slabs to detect")
        return []

    z = points_xyz[:, 2]
    log.info("Slab Z-range: %.2f .. %.2f m", float(z.min()), float(z.max()))

    zh = compute_z_histogram(points_xyz, cfg.z_step, cfg.peak_height_ratio)
    log.info("Found %d horizontal-surface peaks", len(zh.peak_z))

    if not zh.peak_z:
        log.warning(
            "No Z-peaks above %.0f%% of max density — try lowering "
            "peak_height_ratio or check Z coverage.",
            cfg.peak_height_ratio * 100,
        )
        return []

    # Group peaks into slab candidates: a "slab candidate" is either a
    # paired (bottom, top) or a single surface peak.
    candidates: list[tuple[float, float | None]] = []  # (bottom_z, top_z or None)
    i = 0
    while i < len(zh.peak_z):
        z_here = zh.peak_z[i]
        z_next = zh.peak_z[i + 1] if i + 1 < len(zh.peak_z) else None
        if z_next is not None and (z_next - z_here) <= cfg.max_slab_thickness:
            # Paired bottom + top of one slab
            candidates.append((z_here, z_next))
            i += 2
        else:
            # Standalone surface — treat as a floor (top of slab)
            candidates.append((z_here, None))
            i += 1

    slabs: List[Slab] = []
    n_cand = len(candidates)
    for idx, (z_a, z_b) in enumerate(candidates):
        if z_b is not None:
            # Paired: measure thickness from peak positions
            bot_pts = _slice_points(points_xyz, z_a, cfg.z_step)
            top_pts = _slice_points(points_xyz, z_b, cfg.z_step)
            slab_bottom_z = float(np.median(bot_pts[:, 2])) if len(bot_pts) else z_a
            slab_top_z = float(np.median(top_pts[:, 2])) if len(top_pts) else z_b
            thickness = max(0.05, slab_top_z - slab_bottom_z)
            slice_pts = np.vstack([bot_pts, top_pts]) if len(bot_pts) and len(top_pts) else (
                bot_pts if len(bot_pts) else top_pts
            )
        else:
            # Standalone: peak is the top surface (floor). Pick a sensible
            # default thickness — bfs_t for the bottom-most slab, tfs_t for
            # the top-most one, bfs_t for intermediates.
            slice_pts = _slice_points(points_xyz, z_a, cfg.z_step)
            slab_top_z = float(np.median(slice_pts[:, 2])) if len(slice_pts) else z_a
            if idx == 0:
                thickness = bfs_t
            elif idx == n_cand - 1:
                thickness = tfs_t
            else:
                thickness = bfs_t
            slab_bottom_z = slab_top_z - thickness

        if len(slice_pts) < 100:
            log.warning("Slab %d: too few points (%d) — skipping", idx, len(slice_pts))
            continue

        x, y = _hull_from_points(slice_pts, cfg.pc_resolution, cfg.grid_coefficient)
        if x is None or len(x) < 3:
            log.warning("Slab %d: failed to build polygon — skipping", idx)
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
            "Slab %d: bottom=%.3f m, thickness=%.0f mm, hull pts=%d%s",
            idx, slab_bottom_z, thickness * 1000, len(x),
            " (paired peaks)" if z_b is not None else " (single peak)",
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
