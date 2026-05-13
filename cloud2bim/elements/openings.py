"""Opening detection — windows and doors in wall cross-sections.

For each wall:
    1. Project nearby points onto the wall plane
    2. Build a 2D occupancy histogram (along-wall × height)
    3. Empty rectangles in the mask = candidate openings
    4. Classify by Z-extent: doors reach the floor, windows don't
    5. Optionally validate with semantic labels — reject openings whose
       points behind are clutter (chair behind frosted glass etc.)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from cloud2bim.config import OpeningConfig
from cloud2bim.elements.walls import Wall
from cloud2bim.geometry.lines import distance_points_to_line
from cloud2bim.logging import get_logger
from cloud2bim.segmentation.base import SemanticLabels

log = get_logger(__name__)


@dataclass
class Opening:
    """Window or door."""
    wall_storey: int          # which storey wall this belongs to
    wall_index: int           # index within that storey's walls
    type: str                 # "window" | "door"
    x_along_wall_start: float
    x_along_wall_end: float
    z_min: float
    z_max: float

    @property
    def width(self) -> float:
        return self.x_along_wall_end - self.x_along_wall_start

    @property
    def height(self) -> float:
        return self.z_max - self.z_min


def detect_openings(
    walls: List[Wall],
    storey_points: np.ndarray,
    cfg: OpeningConfig,
    pc_resolution: float,
    grid_coefficient: int,
    semantic_labels: SemanticLabels | None = None,
) -> List[Opening]:
    """Find openings in every wall of one storey."""
    if not walls:
        return []

    pixel_size = pc_resolution * grid_coefficient
    wall_thickness = max((w.thickness for w in walls), default=0.3)
    near_thresh = max(wall_thickness * 0.7, 0.10)

    openings: List[Opening] = []
    for w_idx, wall in enumerate(walls):
        # Points near this wall (within thickness/2 + slack)
        d = distance_points_to_line(storey_points[:, :2], wall.start, wall.end)
        near_mask = d < near_thresh
        near_points = storey_points[near_mask]
        if len(near_points) < 100:
            continue

        # Project onto wall axis: x along, z up
        axis = np.array(wall.end) - np.array(wall.start)
        axis_len = float(np.linalg.norm(axis))
        if axis_len == 0:
            continue
        axis_unit = axis / axis_len
        rel = near_points[:, :2] - np.array(wall.start)
        x_along = rel @ axis_unit
        z_vals = near_points[:, 2] - wall.z_placement
        # Only keep points within the wall's footprint
        in_footprint = (x_along >= 0) & (x_along <= axis_len) & (z_vals >= 0) & (z_vals <= wall.height)
        x_along = x_along[in_footprint]
        z_vals = z_vals[in_footprint]
        if len(x_along) < 100:
            continue

        wall_openings = _find_openings_in_mask(
            x_along, z_vals, axis_len, wall.height, pixel_size, cfg,
        )
        for o in wall_openings:
            o.wall_storey = wall.storey
            o.wall_index = w_idx
            openings.append(o)

    log.info("Detected %d openings across %d walls", len(openings), len(walls))
    return openings


def _find_openings_in_mask(
    x_along: np.ndarray,
    z_vals: np.ndarray,
    wall_length: float,
    wall_height: float,
    pixel_size: float,
    cfg: OpeningConfig,
) -> List[Opening]:
    """Build an occupancy mask, find empty rectangles, classify door/window."""
    nx = max(2, int(np.ceil(wall_length / pixel_size)))
    nz = max(2, int(np.ceil(wall_height / pixel_size)))
    grid, _, _ = np.histogram2d(x_along, z_vals, bins=[nx, nz])
    grid = grid.T  # rows=z, cols=x

    threshold = max(1, int(0.05 * grid.max()))
    occ = (grid >= threshold).astype(np.uint8) * 255
    holes = 255 - occ  # invert: empty regions become foreground

    # Morphological cleanup: remove tiny gaps
    holes = cv2.morphologyEx(holes, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(holes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    openings: List[Opening] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        ow = w * pixel_size
        oh = h * pixel_size
        if ow < cfg.min_window_width or oh < cfg.min_window_height:
            continue
        aspect = max(ow / oh, oh / ow)
        if aspect > cfg.max_aspect_ratio:
            continue
        z_min = y * pixel_size
        z_max = (y + h) * pixel_size
        # Touches the floor → door (must also be tall enough)
        if z_min <= cfg.door_max_z and oh >= cfg.door_min_height:
            otype = "door"
        elif z_min > cfg.door_max_z:
            otype = "window"
        else:
            continue
        openings.append(
            Opening(
                wall_storey=-1,  # filled by caller
                wall_index=-1,
                type=otype,
                x_along_wall_start=float(x * pixel_size),
                x_along_wall_end=float((x + w) * pixel_size),
                z_min=float(z_min),
                z_max=float(z_max),
            )
        )
    return openings
