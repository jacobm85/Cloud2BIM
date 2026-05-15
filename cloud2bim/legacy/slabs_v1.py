"""V1 slab detection ported from aronfothi/Cloud2BIM master:app/core/aux_functions.py.

Mirrors ``identify_slabs`` + ``create_hull_from_histogram`` +
``smooth_contour``. Differences from v2:

  * Detection by **density-threshold scan** (60% of max bin density,
    aronfothi default) instead of scipy ``find_peaks``. Walks the
    Z-histogram and emits every contiguous run of bins above the
    threshold as one horizontal surface candidate.
  * Slab definition: consecutive surface candidates are paired
    (bottom + top). The very first standalone surface gets the
    bottom-floor default thickness; an odd-final surface gets the
    top-ceiling default thickness.
  * Hull morphology: 1.0 m dilation+erosion for slab 0 (floor),
    1.5 m for paired/ceiling slabs.
  * No PCA rotation on slabs (slab polygons are world-axis-aligned in
    v1).
  * Contour shift: ``np.roll(mask, (-1, -1))`` before findContours
    plus a ``(contour + 0.5)`` pixel-centre shift on scaling.
  * Smoothing: Douglas-Peucker with ``epsilon = 0.0005 * arcLength``.
"""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from cloud2bim.config import SlabConfig
from cloud2bim.elements.slabs import Slab
from cloud2bim.logging import get_logger

log = get_logger(__name__)


# ── helpers ────────────────────────────────────────────────────────────────

def _smooth_contour(x_contour: np.ndarray, y_contour: np.ndarray, epsilon_frac: float = 0.0005):
    points = np.column_stack([x_contour, y_contour]).astype(np.float32)
    if len(points) < 3:
        return x_contour, y_contour
    epsilon = epsilon_frac * float(cv2.arcLength(points, True))
    simplified = cv2.approxPolyDP(points, epsilon, True).squeeze(axis=1)
    if simplified.ndim != 2 or len(simplified) < 3:
        return x_contour, y_contour
    return simplified[:, 0], simplified[:, 1]


def _create_hull_from_histogram(
    points_3d: np.ndarray,
    pc_resolution: float,
    grid_coefficient: int,
    dilation_meters: float,
    erosion_meters: float,
    extension_meters: float = 1.0,
):
    """v1 create_hull_from_histogram. Returns (x_contour, y_contour)."""
    points_2d = points_3d[:, :2].astype(float)
    pixel_size = pc_resolution * grid_coefficient
    dil_px = max(1, int(dilation_meters / pixel_size))
    ero_px = max(1, int(erosion_meters / pixel_size))

    x_min, x_max = float(points_2d[:, 0].min()), float(points_2d[:, 0].max())
    y_min, y_max = float(points_2d[:, 1].min()), float(points_2d[:, 1].max())
    x_min_ext, x_max_ext = x_min - extension_meters, x_max + extension_meters
    y_min_ext, y_max_ext = y_min - extension_meters, y_max + extension_meters

    x_edges = np.arange(x_min_ext, x_max_ext + pixel_size, pixel_size)
    y_edges = np.arange(y_min_ext, y_max_ext + pixel_size, pixel_size)
    if len(x_edges) < 2 or len(y_edges) < 2:
        return None, None

    hist, _, _ = np.histogram2d(points_2d[:, 0], points_2d[:, 1], bins=(x_edges, y_edges))
    mask = (hist.T > 0).astype(np.uint8)

    kernel_dil = np.ones((dil_px, dil_px), np.uint8)
    mask_dilated = cv2.dilate(mask, kernel_dil, iterations=1)
    kernel_ero = np.ones((ero_px, ero_px), np.uint8)
    mask_eroded = cv2.erode(mask_dilated, kernel_ero, iterations=1)

    # v1 pixel-position correction
    shifted_mask = np.roll(mask_eroded, (-1, -1), axis=(0, 1))

    contours, _ = cv2.findContours(shifted_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    largest = max(contours, key=cv2.contourArea)
    contour = np.squeeze(largest, axis=1)
    if contour.ndim != 2 or len(contour) < 3:
        return None, None
    # +0.5 pixel-centre shift, scale, translate to world
    scaled = (contour + 0.5) * pixel_size + np.array([x_min_ext, y_min_ext])
    x_contour = scaled[:, 0]
    y_contour = scaled[:, 1]
    x_s, y_s = _smooth_contour(x_contour, y_contour, epsilon_frac=0.0005)
    return np.asarray(x_s, dtype=float), np.asarray(y_s, dtype=float)


# ── main entry point ───────────────────────────────────────────────────────

def detect_slabs_v1(
    points_xyz: np.ndarray,
    cfg: SlabConfig,
    *,
    bottom_floor_thickness: Optional[float] = None,
    top_floor_thickness: Optional[float] = None,
    **_unused,
) -> List[Slab]:
    """V1 slab detection. Mirrors master:identify_slabs.

    ``pca_angle`` and other v2 kwargs are accepted via ``**_unused`` so
    the caller can dispatch identically to either algorithm.
    """
    bfs_t = bottom_floor_thickness or cfg.bottom_floor_thickness
    tfs_t = top_floor_thickness or cfg.top_floor_thickness

    if points_xyz is None or len(points_xyz) == 0:
        log.warning("V1 slabs: empty point cloud — no slabs")
        return []

    z_min = float(points_xyz[:, 2].min())
    z_max = float(points_xyz[:, 2].max())
    z_step = float(cfg.z_step)
    if (z_max - z_min) < z_step:
        log.warning("V1 slabs: Z-range %.3f < z_step %.3f", z_max - z_min, z_step)
        return []

    n_steps = int((z_max - z_min) / z_step + 1)
    z_array = []
    n_points_array = []
    for i in range(n_steps):
        z_lo = z_min + i * z_step
        cnt = int(np.sum((points_xyz[:, 2] > z_lo) &
                         (points_xyz[:, 2] < (z_lo + z_step))))
        z_array.append(z_lo)
        n_points_array.append(cnt)

    n_arr = np.asarray(n_points_array, dtype=float)
    if n_arr.max() <= 0:
        log.warning("V1 slabs: empty density histogram")
        return []
    # 60% of peak density — aronfothi/Cloud2BIM master default
    density_threshold = 0.6 * float(n_arr.max())

    # Walk runs above threshold → horizontal-surface intervals
    candidates: list[list[float]] = []
    start = None
    for i, n in enumerate(n_points_array):
        if n > density_threshold:
            if start is None:
                start = i
        elif start is not None:
            candidates.append([z_array[start], z_array[i - 1] + z_step])
            start = None
    if start is not None:
        candidates.append([z_array[start], z_array[-1] + z_step])

    # Merge overlapping intervals (defensive — runs usually don't overlap)
    merged: list[list[float]] = []
    for interval in candidates:
        if not merged or interval[0] > merged[-1][1]:
            merged.append(interval)
        else:
            merged[-1][1] = interval[1]
    candidates = merged

    log.info("V1 slabs: %d horizontal-surface candidates above %.0f points",
             len(candidates), density_threshold)
    if not candidates:
        return []

    # Extract points for each surface
    horiz_surface_planes: list[np.ndarray] = []
    for lo, hi in candidates:
        mask = (points_xyz[:, 2] > lo) & (points_xyz[:, 2] < hi)
        horiz_surface_planes.append(points_xyz[mask])

    # Pair surfaces into slabs (v1 logic: idx 0 standalone bottom-floor,
    # then even indices end a paired slab; final odd index is top ceiling)
    slabs: List[Slab] = []
    n = len(candidates)
    for i, surface_pts in enumerate(horiz_surface_planes):
        if len(surface_pts) < 100:
            log.warning("V1 slab %d: too few points (%d) — skipping",
                        i, len(surface_pts))
            continue

        if i == 0:
            # Standalone bottom-floor slab
            slab_top_z = float(np.median(surface_pts[:, 2]))
            slab_bottom_z = slab_top_z - bfs_t
            x, y = _create_hull_from_histogram(
                surface_pts, cfg.pc_resolution, cfg.grid_coefficient,
                dilation_meters=1.0, erosion_meters=1.0,
            )
            if x is None or len(x) < 3:
                log.warning("V1 slab %d: hull failed", i)
                continue
            slabs.append(Slab(
                bottom_z=slab_bottom_z, thickness=bfs_t,
                polygon_x=x, polygon_y=y, points=surface_pts,
            ))
            log.info("V1 slab %d: bottom %.3f m, thickness %.0f mm (standalone floor)",
                     len(slabs) - 1, slab_bottom_z, bfs_t * 1000)

        elif (i % 2) == 0:
            # Paired with previous: i-1 = bottom, i = top
            bot_pts = horiz_surface_planes[i - 1]
            top_pts = surface_pts
            slab_bottom_z = float(np.median(bot_pts[:, 2]))
            slab_top_z = float(np.median(top_pts[:, 2]))
            thickness = max(0.05, slab_top_z - slab_bottom_z)
            combined = np.concatenate([bot_pts, top_pts], axis=0)
            x, y = _create_hull_from_histogram(
                combined, cfg.pc_resolution, cfg.grid_coefficient,
                dilation_meters=1.5, erosion_meters=1.5,
            )
            if x is None or len(x) < 3:
                log.warning("V1 slab paired %d: hull failed", i)
                continue
            slabs.append(Slab(
                bottom_z=slab_bottom_z, thickness=thickness,
                polygon_x=x, polygon_y=y, points=combined,
            ))
            log.info("V1 slab %d: bottom %.3f m, thickness %.0f mm (paired)",
                     len(slabs) - 1, slab_bottom_z, thickness * 1000)

        elif (i % 2) == 1 and i == n - 1:
            # Top ceiling standalone
            slab_bottom_z = float(np.median(surface_pts[:, 2]))
            x, y = _create_hull_from_histogram(
                surface_pts, cfg.pc_resolution, cfg.grid_coefficient,
                dilation_meters=1.5, erosion_meters=1.5,
            )
            if x is None or len(x) < 3:
                log.warning("V1 slab %d: hull failed", i)
                continue
            slabs.append(Slab(
                bottom_z=slab_bottom_z, thickness=tfs_t,
                polygon_x=x, polygon_y=y, points=surface_pts,
            ))
            log.info("V1 slab %d: bottom %.3f m, thickness %.0f mm (standalone ceiling)",
                     len(slabs) - 1, slab_bottom_z, tfs_t * 1000)
        # else: odd intermediate — v1 treats as "bottom-of-next-pair", skipped here

    log.info("V1 slabs: %d total", len(slabs))
    return slabs
