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
from cloud2bim.segmentation.base import SemanticLabels

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
    semantic_labels: Optional[SemanticLabels] = None,
    column_classes: Optional[List[str]] = None,
) -> List[Column]:
    """Find columns in one storey's points.

    ``walls`` is the list of detected Wall axes — used to mask out wall
    regions so column candidates can't overlap them. When
    ``semantic_labels`` and a non-empty ``column_classes`` are supplied,
    only points labelled as one of those classes are considered — for
    outdoor SemanticKITTI scans this stops trees and lamp posts from
    spilling into the geometric column detector. Without labels (or
    when the class list is empty), the detector falls back to the
    original geometric behaviour over the whole storey.
    """
    if not cfg.enabled:
        return []
    if len(storey_points) == 0:
        log.warning("Storey %d (columns): empty point cloud", storey_idx)
        return []

    if semantic_labels is not None and column_classes:
        mask = semantic_labels.mask_for(column_classes)
        n_before = len(storey_points)
        storey_points = storey_points[mask]
        log.info(
            "Storey %d (columns): label filter %s kept %d / %d points",
            storey_idx, list(column_classes), len(storey_points), n_before,
        )
        if len(storey_points) == 0:
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
    """Render thick lines along each wall axis into a binary mask.

    Silently skips walls whose endpoints contain NaN/inf — the v1
    wall pipeline can produce those when collinear merge collapses
    parallel segments badly, and ``int(float('inf'))`` raises
    OverflowError. Better to drop a bad wall from the mask than to
    kill the entire columns stage.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    height, width = shape
    skipped = 0
    for w in walls:
        sp, ep = w.start, w.end
        if not all(np.isfinite([sp[0], sp[1], ep[0], ep[1], w.thickness])):
            skipped += 1
            continue
        x1 = int((sp[0] - x_min) / pixel_size)
        y1 = int((sp[1] - y_min) / pixel_size)
        x2 = int((ep[0] - x_min) / pixel_size)
        y2 = int((ep[1] - y_min) / pixel_size)
        # Clip to image bounds — cv2.line tolerates out-of-range
        # coords but the mask is bounded anyway and clipping keeps
        # any rasterisation cost proportional to the visible area.
        x1 = max(-width, min(2 * width, x1))
        x2 = max(-width, min(2 * width, x2))
        y1 = max(-height, min(2 * height, y1))
        y2 = max(-height, min(2 * height, y2))
        wall_w_px = max(1, int((w.thickness + 2 * clearance) / pixel_size))
        cv2.line(mask, (x1, y1), (x2, y2), 255, wall_w_px)
    if skipped:
        log.warning(
            "_wall_corridor_mask: skipped %d wall(s) with non-finite endpoints",
            skipped,
        )
    return mask
