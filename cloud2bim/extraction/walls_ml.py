"""ML-driven wall extraction.

Histogram-based detection conflates wall-faced clutter (cabinets,
bookcases, partition panels) with real walls because they all show up
as dense vertical surfaces in a 2D occupancy grid. Class labels remove
that ambiguity: we only fit walls to points the segmenter marked as
``wall``.

Pipeline per storey:
    1. Slice storey points + labels to z ∈ [floor, ceiling]
    2. Keep only wall-labelled points
    3. DBSCAN in XY → one cluster per discrete wall segment
    4. For each cluster: fit a 2D line via PCA, treat the line as the
       wall axis, derive thickness from inlier spread perpendicular to
       the axis
    5. Snap endpoints to nearby intersections (reuses v2 helper)

Skips PCA building-rotation entirely — every cluster picks its own axis.
That naturally handles non-axis-aligned and rotated buildings.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from cloud2bim.config import SegmentationConfig, WallConfig
from cloud2bim.elements.walls import Wall, _adjust_intersections, _has_nan
from cloud2bim.logging import get_logger
from cloud2bim.segmentation.base import SemanticLabels

log = get_logger(__name__)


# Tunables that don't (yet) deserve config fields.
DEFAULT_DBSCAN_EPS = 0.30        # m — points within this XY distance cluster
DEFAULT_DBSCAN_MIN_PTS = 50      # noise filter — sparse blobs rejected
WALL_INLIER_BAND = 0.40          # m — perpendicular inlier band for axis fit


def extract_walls_ml(
    storey_points: np.ndarray,
    storey_labels: SemanticLabels,
    z_floor: float,
    z_ceiling: float,
    storey_idx: int,
    cfg: WallConfig,
    seg_cfg: SegmentationConfig,
    slab_polygon_xy: Optional[np.ndarray] = None,
    exterior_scan: bool = False,
) -> List[Wall]:
    """Per-storey wall extraction from semantic labels.

    Returns the same Wall dataclass as the geometric path, so the IFC
    builder is fully compatible. Storey index is baked into each Wall so
    downstream code can group them.
    """
    if len(storey_points) == 0:
        log.warning("ML walls storey %d: empty point cloud", storey_idx)
        return []

    wall_mask = storey_labels.mask_for(seg_cfg.wall_classes)
    if not wall_mask.any():
        log.warning(
            "ML walls storey %d: no wall-labelled points among %d total",
            storey_idx, len(storey_points),
        )
        return []
    wall_pts = storey_points[wall_mask]
    log.info(
        "ML walls storey %d: %d wall points (%.1f%% of storey)",
        storey_idx, len(wall_pts), 100 * len(wall_pts) / len(storey_points),
    )

    # Project to XY for clustering and axis fitting. Wall height comes
    # from slab spacing, not point Z, so we don't need 3D here.
    clusters = _dbscan_xy(wall_pts[:, :2], DEFAULT_DBSCAN_EPS, DEFAULT_DBSCAN_MIN_PTS)
    log.info("ML walls storey %d: %d DBSCAN clusters", storey_idx, len(clusters))

    centroid = None
    if not exterior_scan and slab_polygon_xy is not None and len(slab_polygon_xy) >= 3:
        centroid = (
            float(np.asarray(slab_polygon_xy)[:, 0].mean()),
            float(np.asarray(slab_polygon_xy)[:, 1].mean()),
        )

    wall_axes: list[list[list[float]]] = []
    wall_thicknesses: list[float] = []

    for cluster_pts in clusters:
        axis, thickness = _fit_wall_axis(cluster_pts)
        if axis is None or _has_nan(axis):
            continue
        # Length filter
        seg_len = float(np.hypot(axis[1][0] - axis[0][0], axis[1][1] - axis[0][1]))
        if seg_len < cfg.min_length:
            continue
        # Thickness sanity (clip to config min/max).
        thickness = float(np.clip(thickness, cfg.min_thickness, cfg.max_thickness))
        wall_axes.append(axis)
        wall_thicknesses.append(thickness)

    if not wall_axes:
        log.warning("ML walls storey %d: no clusters survived axis fitting", storey_idx)
        return []

    # Snap intersections (reuses the v2 helper, which is NaN-safe).
    wall_axes = _adjust_intersections(wall_axes, cfg.max_thickness)

    # Cap to safety limit.
    if len(wall_axes) > cfg.max_walls_per_storey:
        log.warning(
            "ML walls storey %d: clipping %d walls down to max %d",
            storey_idx, len(wall_axes), cfg.max_walls_per_storey,
        )
        wall_axes = wall_axes[: cfg.max_walls_per_storey]
        wall_thicknesses = wall_thicknesses[: cfg.max_walls_per_storey]

    wall_height = z_ceiling - z_floor
    walls = [
        Wall(
            start=tuple(ax[0]),
            end=tuple(ax[1]),
            thickness=t,
            z_placement=z_floor,
            height=wall_height,
            storey=storey_idx,
            label="wall",
        )
        for ax, t in zip(wall_axes, wall_thicknesses)
    ]
    log.info("ML walls storey %d: %d walls finalised", storey_idx, len(walls))
    return walls


# ── internals ─────────────────────────────────────────────────────────────────


def _dbscan_xy(
    xy: np.ndarray,
    eps: float,
    min_pts: int,
) -> list[np.ndarray]:
    """DBSCAN clustering on 2D points.

    Uses open3d's cluster_dbscan (which calls into a C++ implementation)
    via a fake-Z point cloud trick — feeding it a 3D cloud with Z=0
    works identically to a 2D DBSCAN since distances are Euclidean.
    """
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("open3d required for ML wall extraction") from exc

    if len(xy) < min_pts:
        return []

    pc = o3d.geometry.PointCloud()
    xyz = np.column_stack([xy, np.zeros(len(xy))]).astype(np.float64)
    pc.points = o3d.utility.Vector3dVector(xyz)
    # print_progress=False suppresses a long stderr bar for big inputs.
    labels = np.asarray(pc.cluster_dbscan(eps=eps, min_points=min_pts, print_progress=False))

    clusters = []
    for k in range(int(labels.max()) + 1 if labels.size and labels.max() >= 0 else 0):
        mask = labels == k
        if int(mask.sum()) < min_pts:
            continue
        clusters.append(xy[mask])
    return clusters


def _fit_wall_axis(
    cluster_xy: np.ndarray,
) -> tuple[Optional[list[list[float]]], float]:
    """PCA-based axis fit. Returns (axis, thickness) or (None, 0).

    The wall's principal axis is the eigenvector of the points' XY
    covariance matrix. Length is the extent along that axis; thickness
    is the extent perpendicular to it.
    """
    if len(cluster_xy) < 3:
        return None, 0.0

    centroid = cluster_xy.mean(axis=0)
    centred = cluster_xy - centroid
    cov = np.cov(centred.T)
    if not np.isfinite(cov).all():
        return None, 0.0
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Largest eigenvalue → long axis.
    order = np.argsort(eigvals)[::-1]
    axis_dir = eigvecs[:, order[0]]
    perp_dir = eigvecs[:, order[1]]

    proj_long = centred @ axis_dir
    proj_perp = centred @ perp_dir
    long_min, long_max = float(proj_long.min()), float(proj_long.max())
    perp_min, perp_max = float(proj_perp.min()), float(proj_perp.max())

    # Drop clusters whose perpendicular spread exceeds the inlier band —
    # these are usually corners where two walls were under-segmented into
    # one cluster, and an axis fit through them is meaningless.
    if perp_max - perp_min > WALL_INLIER_BAND * 2:
        return None, 0.0

    start = centroid + long_min * axis_dir
    end = centroid + long_max * axis_dir
    thickness = float(perp_max - perp_min)
    return [[float(start[0]), float(start[1])], [float(end[0]), float(end[1])]], thickness
