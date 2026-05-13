"""Sloped roof detection via RANSAC plane fitting.

For buildings with sloped/pitched roofs the v1 horizontal-slab detector
produces nothing useful for the top surface. Here we:
    1. Take points labelled as ceiling/roof (or above the topmost slab)
    2. Iteratively RANSAC-fit planes
    3. Reject planes whose slope is below ``min_slope_deg`` (those are
       handled by the slab detector instead)
    4. Build a 3D polygon for each remaining plane

Each detected plane becomes one IfcRoof with an underlying IfcSlab.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from cloud2bim.config import RoofConfig
from cloud2bim.logging import get_logger

log = get_logger(__name__)


@dataclass
class RoofPlane:
    """A single planar roof segment."""
    normal: np.ndarray          # (3,) unit normal pointing outward
    centroid: np.ndarray        # (3,) plane reference point
    polygon: np.ndarray         # (M, 3) closed boundary
    slope_deg: float
    points: np.ndarray          # (N, 3) inliers


def detect_roofs(
    candidate_points: np.ndarray,
    cfg: RoofConfig,
    max_planes: int = 20,
) -> List[RoofPlane]:
    """RANSAC plane segmentation on candidate roof points."""
    if not cfg.enabled:
        return []
    if len(candidate_points) < cfg.min_inliers:
        log.info("Roofs: too few candidate points (%d)", len(candidate_points))
        return []

    try:
        import open3d as o3d
    except ImportError:
        log.warning("open3d not available — skipping roof detection")
        return []

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(candidate_points)

    planes: List[RoofPlane] = []
    remaining = pcd
    for plane_idx in range(max_planes):
        if len(remaining.points) < cfg.min_inliers:
            break
        plane_model, inliers = remaining.segment_plane(
            distance_threshold=cfg.ransac_distance,
            ransac_n=3,
            num_iterations=1000,
        )
        if len(inliers) < cfg.min_inliers:
            break

        a, b, c, _ = plane_model
        normal = np.array([a, b, c], dtype=float)
        normal /= np.linalg.norm(normal)
        # slope = angle from horizontal = 90° - angle to vertical Z
        slope_deg = float(np.degrees(np.arccos(min(1.0, abs(normal[2])))))

        plane_pts = np.asarray(remaining.points)[inliers]
        remaining = remaining.select_by_index(inliers, invert=True)

        if slope_deg < cfg.min_slope_deg:
            log.debug("Plane %d slope %.1f° too flat — handled by slab detector", plane_idx, slope_deg)
            continue

        polygon = _project_hull(plane_pts, normal)
        planes.append(
            RoofPlane(
                normal=normal,
                centroid=plane_pts.mean(axis=0),
                polygon=polygon,
                slope_deg=slope_deg,
                points=plane_pts,
            )
        )
        log.info("Roof plane %d: slope %.1f°, %d inliers", len(planes), slope_deg, len(plane_pts))

    return planes


def _project_hull(points: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Project points onto the plane and return convex hull as 3D polygon."""
    from scipy.spatial import ConvexHull
    centroid = points.mean(axis=0)
    # Build a local 2D basis on the plane
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, arbitrary)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    rel = points - centroid
    pts_2d = np.column_stack([rel @ u, rel @ v])
    try:
        hull = ConvexHull(pts_2d)
        hull_2d = pts_2d[hull.vertices]
    except Exception:
        # Fallback: bounding rectangle
        mn, mx = pts_2d.min(axis=0), pts_2d.max(axis=0)
        hull_2d = np.array([
            [mn[0], mn[1]], [mx[0], mn[1]], [mx[0], mx[1]], [mn[0], mx[1]]
        ])
    # Lift back to 3D
    return centroid + hull_2d[:, 0:1] * u + hull_2d[:, 1:2] * v
