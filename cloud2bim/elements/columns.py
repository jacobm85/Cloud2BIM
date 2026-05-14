"""Column detection.

Columns are vertical free-standing elements with a compact XY footprint
that spans (close to) the full storey height. The detector:

    1. Projects all per-storey points to a 2D occupancy histogram
    2. Subtracts the wall corridor so column candidates can't sit on a wall
    3. Finds connected components in the residual mask
    4. Keeps blobs whose bounding rect is in [min_size, max_size] on BOTH
       axes (so they aren't long like walls) AND that span enough Z to
       count as floor-to-ceiling
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
from skimage.morphology import closing, footprint_rectangle

from cloud2bim.config import ColumnConfig
from cloud2bim.logging import get_logger

log = get_logger(__name__)


@dataclass
class Column:
    """Free-standing vertical structural element."""
    center_x: float
    center_y: float
    size_x: float
    size_y: float
    z_placement: float  # bottom Z (m)
    height: float       # m
    storey: int
    material: str = "Concrete"


def detect_columns(
    storey_points: np.ndarray,
    walls: list,
    z_floor: float,
    z_ceiling: float,
    storey_idx: int,
    cfg: ColumnConfig,
    pc_resolution: float,
    grid_coefficient: int,
) -> List[Column]:
    """Find columns in one storey's points.

    ``walls`` is the list of detected Wall axes — used to mask out wall
    regions so column candidates can't overlap them.
    """
    if not cfg.enabled:
        return []
    if len(storey_points) == 0:
        log.warning("Storey %d (columns): empty point cloud", storey_idx)
        return []

    storey_height = max(0.1, z_ceiling - z_floor)

    # 1. 2D occupancy histogram across the full storey Z
    pixel_size = pc_resolution * grid_coefficient
    pts_xy = storey_points[:, :2]
    x_min, y_min = float(pts_xy[:, 0].min()), float(pts_xy[:, 1].min())
    x_max, y_max = float(pts_xy[:, 0].max()), float(pts_xy[:, 1].max())
    xs = np.arange(x_min, x_max + pixel_size, pixel_size)
    ys = np.arange(y_min, y_max + pixel_size, pixel_size)
    if len(xs) < 2 or len(ys) < 2:
        return []
    grid, _, _ = np.histogram2d(pts_xy[:, 0], pts_xy[:, 1], bins=[xs, ys])
    grid = grid.T  # rows=y, cols=x
    if grid.max() == 0:
        return []

    # Binary mask: cells with enough points
    threshold = max(1.0, 0.05 * grid.max())
    mask = (grid > threshold).astype(np.uint8) * 255
    mask = closing(mask, footprint_rectangle((3, 3)))

    # 2. Mask out walls so column blobs can't sit on a wall
    if walls and cfg.wall_clearance > 0:
        wall_mask = _wall_corridor_mask(walls, cfg.wall_clearance,
                                       x_min, y_min, pixel_size, mask.shape)
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(wall_mask))

    # 3. Connected components
    n_lab, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_lab <= 1:
        return []

    min_size_px = max(1, int(cfg.min_size / pixel_size))
    max_size_px = max(min_size_px + 1, int(cfg.max_size / pixel_size))
    min_z_span = cfg.min_height_fraction * storey_height

    columns: List[Column] = []
    for lab in range(1, n_lab):
        x, y, w, h, area = stats[lab]
        if w < min_size_px or h < min_size_px:
            continue
        if w > max_size_px or h > max_size_px:
            continue
        if area < min_size_px * min_size_px // 4:
            continue

        # Re-extract the points in this blob and check Z span
        cx_world = (x + w / 2) * pixel_size + x_min
        cy_world = (y + h / 2) * pixel_size + y_min
        # Use blob bounding rect to filter points
        half_x = (w / 2 + 1) * pixel_size
        half_y = (h / 2 + 1) * pixel_size
        in_blob = (
            (storey_points[:, 0] >= cx_world - half_x)
            & (storey_points[:, 0] <= cx_world + half_x)
            & (storey_points[:, 1] >= cy_world - half_y)
            & (storey_points[:, 1] <= cy_world + half_y)
        )
        if int(in_blob.sum()) < cfg.min_points:
            continue
        zs = storey_points[in_blob, 2]
        z_span = float(zs.max() - zs.min())
        if z_span < min_z_span:
            continue

        columns.append(Column(
            center_x=float(cx_world),
            center_y=float(cy_world),
            size_x=float(w * pixel_size),
            size_y=float(h * pixel_size),
            z_placement=float(z_floor),
            height=float(storey_height),
            storey=storey_idx,
        ))

    log.info("Storey %d: %d columns detected", storey_idx, len(columns))
    return columns


def _wall_corridor_mask(walls, clearance: float, x_min: float, y_min: float,
                       pixel_size: float, shape: tuple[int, int]) -> np.ndarray:
    """Render thick lines along each wall axis into a binary mask."""
    mask = np.zeros(shape, dtype=np.uint8)
    thickness_px = max(1, int(clearance / pixel_size))
    for w in walls:
        sp, ep = w.start, w.end
        x1 = int((sp[0] - x_min) / pixel_size)
        y1 = int((sp[1] - y_min) / pixel_size)
        x2 = int((ep[0] - x_min) / pixel_size)
        y2 = int((ep[1] - y_min) / pixel_size)
        # Add half the wall's own thickness too
        wall_w_px = max(1, int((w.thickness + 2 * clearance) / pixel_size))
        cv2.line(mask, (x1, y1), (x2, y2), 255, wall_w_px)
    return mask
