"""V1 opening detection ported from VaclavNezerka/Cloud2BIM master:aux_functions.py.

Reference: ``identify_openings`` + ``identify_wall_faces`` +
``assign_points_to_walls`` from upstream master (and the local 20fa12e
state, where the same algorithm was running with PCA rotation in walls).

Why a separate port: v2's :mod:`cloud2bim.elements.openings` uses a 2D
occupancy mask of points near the wall *axis* and looks for empty
rectangles. v1 projects points near the wall *faces* (within
``thickness_for_extraction`` of each peak in the y-histogram), then
finds 1D x-histogram runs below a top-10 density threshold and refines
the z-range from a second z-histogram at each candidate's midpoint.
The face-projection makes v1 much more robust to furniture *behind*
glass — chairs behind a window don't fill the wall plane the way they
fill the wall thickness band.

Per-wall point assignment + rotation to local (along-wall, perp, up)
frame is performed in :func:`detect_openings_v1` itself rather than
inheriting it from walls_v1, so the v1 opening detector is independent
of which wall algorithm is run upstream.

Adaptations from v1:
    * Returns ``List[Opening]`` (v2's dataclass) instead of v1's tuple
      of (widths, heights, types).
    * Operates on a list of ``Wall`` objects + the full storey point
      cloud, not on a pre-rotated/pre-translated wall point cluster.
    * Defensive guards: empty wall, empty projected points, degenerate
      histograms — all return cleanly instead of raising.
"""
from __future__ import annotations

import math
from typing import List

import numpy as np
from scipy.signal import find_peaks

from cloud2bim.config import OpeningConfig
from cloud2bim.elements.openings import Opening
from cloud2bim.elements.walls import Wall
from cloud2bim.logging import get_logger

log = get_logger(__name__)


# v1 runtime constants. Mirrors the cloud2entities.py call site at 20fa12e:
#   identify_openings(..., min_opening_width=0.4, min_opening_height=0.6,
#                     max_opening_aspect_ratio=4, door_z_max=0.1,
#                     door_min_height=1.6, opening_min_z_top=1.6)
_HISTOGRAM_THRESHOLD = 0.7   # x-histogram tröskel som andel av 10:e-toppen
_THICKNESS_FOR_EXTRACTION = 0.07  # m, band runt varje wall face
_FACE_PEAK_HEIGHT_FRAC = 0.3      # andel av max(hist) som peak-tröskel
_FACE_PEAK_MIN_DISTANCE = 25      # bins, peak separation in y-histogram


# ── helpers ────────────────────────────────────────────────────────────────

def _identify_wall_faces(
    wall_points: np.ndarray,
    wall_label: str,
    resolution: float,
):
    """v1 identify_wall_faces. Returns (y1, y2) — the two wall-face peaks.

    For interior walls the two peaks are inner+outer face. For exterior
    walls we only see one face (the inward one), so y2 = y1.
    """
    y_coords = wall_points[:, 1]
    y_min, y_max = float(y_coords.min()), float(y_coords.max())
    if y_max - y_min < resolution:
        return None, None
    bin_edges = np.arange(y_min, y_max + resolution, resolution)
    if len(bin_edges) < 3:
        return None, None
    hist, bin_edges = np.histogram(y_coords, bins=bin_edges)
    if hist.max() == 0:
        return None, None

    height_threshold = _FACE_PEAK_HEIGHT_FRAC * float(hist.max())
    peaks, _ = find_peaks(
        hist,
        distance=_FACE_PEAK_MIN_DISTANCE,
        height=height_threshold,
        prominence=0.25 * height_threshold,
    )

    if wall_label == "interior":
        if len(peaks) >= 2:
            y1 = (bin_edges[peaks[0]] + bin_edges[peaks[0] + 1]) / 2
            y2 = (bin_edges[peaks[1]] + bin_edges[peaks[1] + 1]) / 2
        else:
            # Fall back to highest point in each half of the histogram
            half = len(hist) // 2
            if half < 1:
                return None, None
            i1 = int(np.argmax(hist[:half]))
            i2 = int(np.argmax(hist[half:])) + half
            y1 = (bin_edges[i1] + bin_edges[i1 + 1]) / 2
            y2 = (bin_edges[i2] + bin_edges[i2 + 1]) / 2
    else:
        # Exterior — one face only. Use the highest peak; if none, fail.
        if len(peaks) == 0:
            return None, None
        peak_idx = int(np.argmax(hist[peaks]))
        peak = int(peaks[peak_idx])
        y1 = (bin_edges[peak] + bin_edges[peak + 1]) / 2
        y2 = y1
    return float(y1), float(y2)


def _identify_openings_in_wall(
    wall_points_local: np.ndarray,
    wall_label: str,
    resolution: float,
    grid_roughness: int,
    cfg: OpeningConfig,
):
    """v1 identify_openings, operating on wall-local points (x=along, y=perp, z=up).

    Returns three parallel lists: ``[(x_start, x_end), ...]``,
    ``[(z_min, z_max), ...]``, ``["window"|"door", ...]``.
    """
    widths: list[tuple[float, float]] = []
    heights: list[tuple[float, float]] = []
    types: list[str] = []

    if len(wall_points_local) < 50:
        return widths, heights, types

    try:
        y1, y2 = _identify_wall_faces(wall_points_local, wall_label, resolution)
    except Exception:
        return widths, heights, types
    if y1 is None or y2 is None:
        return widths, heights, types

    # Project points within thickness_for_extraction of either face onto x-z
    inner_threshold = y1 - _THICKNESS_FOR_EXTRACTION / 2
    outer_threshold = y2 + _THICKNESS_FOR_EXTRACTION / 2
    y_arr = wall_points_local[:, 1]
    near_face = (
        ((y_arr >= inner_threshold) & (y_arr <= y1 + _THICKNESS_FOR_EXTRACTION / 2))
        | ((y_arr >= y2 - _THICKNESS_FOR_EXTRACTION / 2) & (y_arr <= outer_threshold))
    )
    projected = wall_points_local[near_face]
    if len(projected) == 0:
        return widths, heights, types
    x_coords = projected[:, 0]
    z_coords = projected[:, 2]

    x_min, x_max = float(x_coords.min()), float(x_coords.max())
    bins = int((x_max - x_min) / (resolution * grid_roughness))
    if bins < 2:
        return widths, heights, types
    hist, edges = np.histogram(x_coords, bins=bins, range=(x_min, x_max))

    z_min_global, z_max_global = float(z_coords.min()), float(z_coords.max())
    z_bins = int((z_max_global - z_min_global) / (resolution * grid_roughness))
    if z_bins < 2:
        return widths, heights, types

    # Top-10:e bin × histogram_threshold som tröskel (v1)
    if len(hist) > 10:
        sorted_hist = np.sort(hist)[::-1]
        max10 = float(sorted_hist[10])
    else:
        max10 = float(hist[-1])
    x_threshold = max10 * _HISTOGRAM_THRESHOLD

    # Walk x-histogram to identify candidate x-intervals (runs below threshold)
    candidates: list[tuple[float, float]] = []
    in_open = False
    start = None
    for i, count in enumerate(hist):
        if count < x_threshold and not in_open:
            in_open = True
            start = float(edges[i])
        elif count >= x_threshold and in_open:
            in_open = False
            end = float(edges[i])
            if (abs(end - start) > cfg.min_window_width
                    and start >= x_min and end <= x_max):
                candidates.append((start, end))

    # Refine z-range for each candidate
    for x_start, x_end in candidates:
        middle_x = (x_start + x_end) / 2
        tolerance = cfg.min_window_width  # v1 uses min_opening_width
        near_mid = projected[
            (projected[:, 0] >= middle_x - tolerance)
            & (projected[:, 0] <= middle_x + tolerance)
        ]
        if len(near_mid) == 0:
            continue
        z_at_middle = near_mid[:, 2]
        z_hist, z_edges = np.histogram(
            z_at_middle, bins=z_bins, range=(z_min_global, z_max_global),
        )
        if len(z_hist) < 3:
            continue
        try:
            sorted_zhist = np.sort(z_hist)[::-1]
            max2 = float(sorted_zhist[2])
        except IndexError:
            continue
        z_threshold = max2 * 0.2

        # Walk z-histogram for refined runs below threshold
        cand_zs: list[tuple[float, float]] = []
        in_open = False
        rzmin = None
        for i, count in enumerate(z_hist):
            if count < z_threshold and not in_open:
                in_open = True
                rzmin = float(z_edges[i])
            elif count >= z_threshold and in_open:
                in_open = False
                rzmax = float(z_edges[i + 1])
                cand_zs.append((rzmin, rzmax))
                rzmin = None
        if not cand_zs:
            continue

        # v1 picks the candidate with the largest z-extent
        refined_z_min, refined_z_max = max(cand_zs, key=lambda p: p[1] - p[0])
        width = x_end - x_start
        height = refined_z_max - refined_z_min
        if height <= cfg.min_window_height:
            continue
        if (height / width) >= cfg.max_aspect_ratio:
            continue
        # opening_min_z_top: top of the opening must be above 1.6 m
        if refined_z_max <= 1.6:
            continue

        if min(refined_z_min, refined_z_max) > cfg.door_max_z:
            widths.append((x_start, x_end))
            heights.append((refined_z_min, refined_z_max))
            types.append("window")
        elif height > cfg.door_min_height:
            # v1: door z_min snaps to 0 (the floor)
            widths.append((x_start, x_end))
            heights.append((0.0, refined_z_max))
            types.append("door")

    return widths, heights, types


def _wall_local_points(
    wall: Wall,
    storey_points: np.ndarray,
    perp_distance: float,
) -> np.ndarray:
    """Assign storey points to a wall and rotate/translate to local frame.

    Returns Nx3 array with x along the wall (0..wall_length), y perpendicular
    to the wall plane (centred on the wall axis), z relative to the wall's
    z_placement (0..wall.height).
    """
    if len(storey_points) == 0:
        return np.empty((0, 3))

    sx, sy = float(wall.start[0]), float(wall.start[1])
    ex, ey = float(wall.end[0]), float(wall.end[1])
    dx, dy = ex - sx, ey - sy
    wall_length = math.hypot(dx, dy)
    if wall_length == 0:
        return np.empty((0, 3))

    angle = math.atan2(dy, dx)
    c, s = math.cos(-angle), math.sin(-angle)

    # Translate to wall.start, then rotate by -angle so wall lies along +x
    rel_x = storey_points[:, 0] - sx
    rel_y = storey_points[:, 1] - sy
    local_x = rel_x * c - rel_y * s
    local_y = rel_x * s + rel_y * c
    local_z = storey_points[:, 2] - wall.z_placement

    # Filter to the wall's footprint: along-wall, perpendicular band, vertical band
    in_x = (local_x >= -0.05) & (local_x <= wall_length + 0.05)
    in_y = np.abs(local_y) <= perp_distance
    in_z = (local_z >= -0.05) & (local_z <= wall.height + 0.05)
    mask = in_x & in_y & in_z
    if not np.any(mask):
        return np.empty((0, 3))

    return np.column_stack([local_x[mask], local_y[mask], local_z[mask]])


# ── main entry point ───────────────────────────────────────────────────────

def detect_openings_v1(
    walls: List[Wall],
    storey_points: np.ndarray,
    cfg: OpeningConfig,
    pc_resolution: float,
    grid_coefficient: int,
    **_unused,
) -> List[Opening]:
    """V1 opening detection. Mirrors master:identify_openings + helpers.

    Extra kwargs accepted from the v2 caller (``semantic_labels``) are
    silently dropped via ``**_unused`` — v1 doesn't use semantic labels.
    """
    if not walls:
        return []
    if len(storey_points) == 0:
        return []

    openings: List[Opening] = []
    for w_idx, wall in enumerate(walls):
        if wall.thickness <= 0 or wall.height <= 0:
            continue
        perp_distance = max(0.5 * wall.thickness + 0.2 * wall.thickness, 0.10)
        local_pts = _wall_local_points(wall, storey_points, perp_distance)
        if len(local_pts) < 100:
            continue

        widths, heights, types = _identify_openings_in_wall(
            local_pts, wall.label, pc_resolution, grid_coefficient, cfg,
        )
        for (x0, x1), (z0, z1), otype in zip(widths, heights, types):
            openings.append(Opening(
                wall_storey=wall.storey,
                wall_index=w_idx,
                type=otype,
                x_along_wall_start=float(x0),
                x_along_wall_end=float(x1),
                z_min=float(z0),
                z_max=float(z1),
            ))

    log.info("V1 openings: detected %d across %d walls", len(openings), len(walls))
    return openings
