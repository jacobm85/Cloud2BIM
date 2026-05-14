"""Stair flight detection.

Looks for stairs *between* the main slab levels:

    1. Restrict points to the inter-slab Z range
    2. Build a fine 1D Z-histogram (3 cm bins) and pick peaks — each peak
       is a tread surface
    3. Group consecutive peaks with riser ∈ [min_riser, max_riser] into
       flights of ≥ min_steps
    4. For each flight, compute the XY footprint from the union of tread
       points; drop flights with too large a footprint (likely a balcony
       or floor-fragment) or too few points

Stair extraction is geometric only — IFC-side we emit an IfcStairFlight
with the run's footprint and a fake going/tread placeholder.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.signal import find_peaks

from cloud2bim.config import StairConfig
from cloud2bim.logging import get_logger

log = get_logger(__name__)


@dataclass
class StairFlight:
    """A run of stairs between two levels."""
    step_z: List[float]
    riser: float          # avg riser height (m)
    going: float          # avg tread depth (m)
    bottom_z: float
    top_z: float
    polygon_x: np.ndarray  # footprint (XY hull)
    polygon_y: np.ndarray
    storey: int
    material: str = "Concrete"

    @property
    def n_steps(self) -> int:
        return len(self.step_z)


def detect_stairs(
    points_xyz: np.ndarray,
    z_floor: float,
    z_ceiling: float,
    storey_idx: int,
    cfg: StairConfig,
) -> List[StairFlight]:
    """Find stair flights between z_floor (top of slab) and z_ceiling
    (bottom of next slab)."""
    if not cfg.enabled:
        return []
    if len(points_xyz) == 0:
        return []

    # 1. Restrict to inter-slab Z (a small margin lets us catch the
    #    bottom and top steps near the slab faces)
    margin = 0.05
    inter = (points_xyz[:, 2] >= z_floor - margin) & (points_xyz[:, 2] <= z_ceiling + margin)
    pts = points_xyz[inter]
    if len(pts) < 200:
        return []

    # 2. Fine Z-histogram → step peaks
    z = pts[:, 2]
    edges = np.arange(float(z.min()), float(z.max()) + cfg.z_step, cfg.z_step)
    if len(edges) < 4:
        return []
    hist, _ = np.histogram(z, bins=edges)
    if hist.max() == 0:
        return []
    centers = (edges[:-1] + edges[1:]) / 2
    threshold = cfg.peak_height_ratio * hist.max()
    min_gap_px = max(1, int(cfg.min_riser / cfg.z_step))
    peak_idx, _ = find_peaks(hist, height=threshold, distance=min_gap_px)
    if len(peak_idx) < cfg.min_steps:
        return []
    peak_zs = [float(centers[p]) for p in peak_idx]

    # 3. Group consecutive peaks with riser in range
    flights_raw: list[list[float]] = []
    current: list[float] = [peak_zs[0]]
    for z_next in peak_zs[1:]:
        rise = z_next - current[-1]
        if cfg.min_riser <= rise <= cfg.max_riser:
            current.append(z_next)
        else:
            if len(current) >= cfg.min_steps:
                flights_raw.append(current)
            current = [z_next]
    if len(current) >= cfg.min_steps:
        flights_raw.append(current)

    flights: List[StairFlight] = []
    for run_zs in flights_raw:
        # Tread points: union of slabs each at z ± z_step
        tread_mask = np.zeros(len(pts), dtype=bool)
        for z_tread in run_zs:
            tread_mask |= np.abs(pts[:, 2] - z_tread) <= cfg.z_step
        tread_pts = pts[tread_mask]
        if len(tread_pts) < 100:
            continue
        # Footprint
        x_min, y_min = float(tread_pts[:, 0].min()), float(tread_pts[:, 1].min())
        x_max, y_max = float(tread_pts[:, 0].max()), float(tread_pts[:, 1].max())
        if (x_max - x_min) > cfg.max_footprint or (y_max - y_min) > cfg.max_footprint:
            continue
        polygon_x = np.array([x_min, x_max, x_max, x_min, x_min])
        polygon_y = np.array([y_min, y_min, y_max, y_max, y_min])
        risers = [run_zs[i + 1] - run_zs[i] for i in range(len(run_zs) - 1)]
        riser = float(np.mean(risers))
        # Going (tread depth) — estimate from XY extent divided by step count
        going = float(max(x_max - x_min, y_max - y_min)) / max(1, len(run_zs))
        flights.append(StairFlight(
            step_z=run_zs,
            riser=riser,
            going=going,
            bottom_z=float(run_zs[0]),
            top_z=float(run_zs[-1]),
            polygon_x=polygon_x,
            polygon_y=polygon_y,
            storey=storey_idx,
        ))

    log.info(
        "Storey %d: %d stair flight(s) — peaks=%d, candidates=%d",
        storey_idx, len(flights), len(peak_zs), len(flights_raw),
    )
    return flights
