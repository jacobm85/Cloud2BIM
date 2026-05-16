"""ML-driven opening extraction.

Histogram detection looks for *gaps* in a wall's cross-section — that
works on a clean wall but a chair pulled up to a window leaves the same
silhouette as a solid wall, so the window goes missing. With labels we
do the opposite: find points labelled ``door`` / ``window``, project
them onto their host wall, and use the projection AABB as the opening.

Pipeline per storey:
    1. door_mask = labels.mask_for(door_classes)
    2. window_mask = labels.mask_for(window_classes)
    3. For each labelled cluster (DBSCAN to separate windows):
        a. Find the closest wall axis (min perpendicular distance)
        b. Project cluster points onto that wall's axis
        c. AABB along [axis_direction × z] → Opening
        d. Skip if the AABB is below cfg.min_window_width / height
"""
from __future__ import annotations

from typing import List

import numpy as np

from cloud2bim.config import OpeningConfig, SegmentationConfig
from cloud2bim.elements.openings import Opening
from cloud2bim.elements.walls import Wall
from cloud2bim.logging import get_logger
from cloud2bim.segmentation.base import SemanticLabels

log = get_logger(__name__)


DBSCAN_EPS = 0.25                  # m — opening points within this XY radius cluster
DBSCAN_MIN_PTS = 30                # noise filter
MAX_WALL_DISTANCE = 0.50           # m — opening farther than this isn't matched


def extract_openings_ml(
    walls: List[Wall],
    storey_points: np.ndarray,
    storey_labels: SemanticLabels,
    cfg: OpeningConfig,
    seg_cfg: SegmentationConfig,
) -> List[Opening]:
    """Detect doors + windows in one storey from class labels."""
    if not walls:
        return []
    if len(storey_points) == 0:
        return []

    openings: List[Opening] = []
    for cls_name, kind in (("door", "door"), ("window", "window")):
        classes = seg_cfg.door_classes if cls_name == "door" else seg_cfg.window_classes
        mask = storey_labels.mask_for(classes)
        if not mask.any():
            continue
        cluster_pts_list = _dbscan_xyz(storey_points[mask], DBSCAN_EPS, DBSCAN_MIN_PTS)
        log.info(
            "ML openings: %d %s clusters from %d labelled points",
            len(cluster_pts_list), kind, int(mask.sum()),
        )
        for cluster_pts in cluster_pts_list:
            op = _opening_from_cluster(cluster_pts, walls, kind, cfg)
            if op is not None:
                openings.append(op)
    return openings


# ── internals ────────────────────────────────────────────────────────────────


def _dbscan_xyz(
    points: np.ndarray,
    eps: float,
    min_pts: int,
) -> list[np.ndarray]:
    """3D DBSCAN — distinct windows on the same wall sit metres apart in XY."""
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError("open3d required for ML opening extraction") from exc

    if len(points) < min_pts:
        return []
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    labels = np.asarray(pc.cluster_dbscan(eps=eps, min_points=min_pts, print_progress=False))
    clusters = []
    for k in range(int(labels.max()) + 1 if labels.size and labels.max() >= 0 else 0):
        m = labels == k
        if int(m.sum()) < min_pts:
            continue
        clusters.append(points[m])
    return clusters


def _opening_from_cluster(
    cluster_pts: np.ndarray,
    walls: List[Wall],
    kind: str,
    cfg: OpeningConfig,
) -> Opening | None:
    """Project a cluster onto its nearest wall, return the Opening AABB."""
    centroid_xy = cluster_pts[:, :2].mean(axis=0)
    best = _nearest_wall(centroid_xy, walls)
    if best is None:
        return None
    wall_idx, dist = best
    if dist > MAX_WALL_DISTANCE:
        log.debug(
            "ML opening: cluster too far from any wall (%.2f m) — skipping",
            dist,
        )
        return None

    wall = walls[wall_idx]
    along = _project_onto_axis(cluster_pts[:, :2], wall)
    along_min, along_max = float(along.min()), float(along.max())
    z_min = float(cluster_pts[:, 2].min())
    z_max = float(cluster_pts[:, 2].max())

    width = along_max - along_min
    height = z_max - z_min
    if kind == "window":
        if width < cfg.min_window_width or height < cfg.min_window_height:
            return None
    else:
        if height < cfg.door_min_height:
            return None

    return Opening(
        wall_storey=wall.storey,
        wall_index=wall_idx,
        type=kind,
        x_along_wall_start=along_min,
        x_along_wall_end=along_max,
        z_min=z_min,
        z_max=z_max,
    )


def _nearest_wall(
    point_xy: np.ndarray,
    walls: List[Wall],
) -> tuple[int, float] | None:
    """Return (wall_index, perpendicular_distance) for the closest wall."""
    best: tuple[int, float] | None = None
    for i, w in enumerate(walls):
        a = np.array(w.start, dtype=float)
        b = np.array(w.end, dtype=float)
        ab = b - a
        n = float(np.linalg.norm(ab))
        if n < 1e-6:
            continue
        # Perpendicular distance from point to infinite line through a-b.
        d = float(abs(np.cross(ab, point_xy - a)) / n)
        if best is None or d < best[1]:
            best = (i, d)
    return best


def _project_onto_axis(points_xy: np.ndarray, wall: Wall) -> np.ndarray:
    """Project 2D points onto a wall's axis. Returns scalar 'along' coords."""
    a = np.array(wall.start, dtype=float)
    b = np.array(wall.end, dtype=float)
    ab = b - a
    n = float(np.linalg.norm(ab))
    if n < 1e-6:
        return np.zeros(len(points_xy))
    unit = ab / n
    return (points_xy - a) @ unit
