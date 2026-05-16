"""ML-driven slab extraction.

Replaces the Z-histogram path with class-aware RANSAC: floor and ceiling
points come pre-labelled, we cluster each along Z to separate per-storey
surfaces, and fit a horizontal plane per cluster. Slab thickness is the
gap between paired floor & ceiling surfaces.

Why this beats the histogram on cluttered scans:
    - Tabletops, shelves and ductwork no longer create spurious Z peaks
      — they aren't labelled as floor/ceiling.
    - Diagonal or rotated buildings work without PCA pre-rotation
      because we never rely on axis-aligned histograms.
    - Sparse ceilings are still detected (one labelled ceiling point
      per square metre suffices for a horizontal-plane RANSAC fit).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from cloud2bim.config import SegmentationConfig, SlabConfig
from cloud2bim.elements.slabs import Slab
from cloud2bim.logging import get_logger
from cloud2bim.segmentation.base import SemanticLabels

log = get_logger(__name__)


# Tunables that don't deserve their own config field yet — promote if a
# user starts hand-tuning them.
DEFAULT_Z_CLUSTER_TOL = 0.30      # m — points within this Z range cluster
MIN_POINTS_PER_SURFACE = 200       # below this we ignore a candidate
RANSAC_THICKNESS_TOL = 0.05        # m — inlier band for horizontal plane fit
MAX_PAIRING_GAP = 6.0              # m — ceiling further than this isn't paired
MIN_STOREY_HEIGHT = 1.8            # m — drop floor-ceiling pairs below this


def extract_slabs_ml(
    points_xyz: np.ndarray,
    labels: SemanticLabels,
    cfg: SlabConfig,
    seg_cfg: SegmentationConfig,
) -> list[Slab]:
    """Detect slabs from per-point labels. Returns sorted-by-Z Slab list."""
    if len(points_xyz) == 0:
        log.warning("extract_slabs_ml: empty point cloud")
        return []

    floor_mask = labels.mask_for(seg_cfg.floor_classes)
    ceiling_mask = labels.mask_for(seg_cfg.ceiling_classes)
    log.info(
        "ML slabs: %s floor points, %s ceiling points",
        f"{int(floor_mask.sum()):,}", f"{int(ceiling_mask.sum()):,}",
    )

    floor_surfaces = _detect_horizontal_surfaces(
        points_xyz[floor_mask], cfg, label="floor",
    )
    ceiling_surfaces = _detect_horizontal_surfaces(
        points_xyz[ceiling_mask], cfg, label="ceiling",
    )
    log.info(
        "ML slabs: %d floor surfaces + %d ceiling surfaces",
        len(floor_surfaces), len(ceiling_surfaces),
    )

    return _pair_floors_with_ceilings(floor_surfaces, ceiling_surfaces, cfg)


# ── internals ─────────────────────────────────────────────────────────────────


def _detect_horizontal_surfaces(
    pts: np.ndarray,
    cfg: SlabConfig,
    label: str,
) -> list[dict]:
    """Cluster points by Z, RANSAC-fit a horizontal plane per cluster.

    Returns one dict per detected surface with keys:
        z          mean Z of inliers (m)
        thickness  vertical spread of inliers (m)
        points     (N, 3) inlier points (used later for polygon extraction)
    """
    if len(pts) < MIN_POINTS_PER_SURFACE:
        return []

    z_clusters = _cluster_1d(pts[:, 2], DEFAULT_Z_CLUSTER_TOL)
    if not z_clusters:
        return []

    surfaces: list[dict] = []
    for cluster_mask in z_clusters:
        cluster_pts = pts[cluster_mask]
        if len(cluster_pts) < MIN_POINTS_PER_SURFACE:
            continue
        plane = _ransac_horizontal_plane(cluster_pts)
        if plane is None:
            continue
        z_centre, inliers = plane
        if len(inliers) < MIN_POINTS_PER_SURFACE:
            continue
        inlier_pts = cluster_pts[inliers]
        surfaces.append({
            "z": z_centre,
            "thickness": float(np.std(inlier_pts[:, 2])) * 2.0 or RANSAC_THICKNESS_TOL,
            "points": inlier_pts,
        })
    surfaces.sort(key=lambda s: s["z"])
    log.info("ML slabs (%s): %d surfaces at z=%s",
             label, len(surfaces), [round(s["z"], 2) for s in surfaces])
    return surfaces


def _cluster_1d(values: np.ndarray, tol: float) -> list[np.ndarray]:
    """Greedy 1D clustering. Sort, group into runs where Δ ≤ tol.

    Returns a list of boolean masks (one per cluster) over the original
    unsorted ``values`` array. Used because we want to group floor points
    on the *same level*, not all floor points across all storeys.
    """
    if values.size == 0:
        return []
    order = np.argsort(values)
    sorted_v = values[order]
    breaks = np.where(np.diff(sorted_v) > tol)[0] + 1
    cluster_ids = np.empty(len(values), dtype=np.int32)
    sorted_ids = np.zeros_like(sorted_v, dtype=np.int32)
    cid = 0
    last = 0
    for b in breaks:
        sorted_ids[last:b] = cid
        cid += 1
        last = b
    sorted_ids[last:] = cid
    cluster_ids[order] = sorted_ids
    return [cluster_ids == k for k in range(cid + 1)]


def _ransac_horizontal_plane(
    pts: np.ndarray,
) -> tuple[float, np.ndarray] | None:
    """Fit z = const to the densest horizontal slab inside ``pts``.

    We don't really need open3d.segment_plane here — for a *horizontal*
    plane we can just find the Z value that maximises inliers within a
    tolerance band. That's O(N) and avoids the open3d dependency for one
    of the hot paths. (Walls/openings still use full 3D plane RANSAC.)
    """
    if len(pts) < MIN_POINTS_PER_SURFACE:
        return None

    z = pts[:, 2]
    z_min, z_max = float(z.min()), float(z.max())
    if z_max - z_min < 1e-6:
        return float(z_min), np.arange(len(pts))

    # Iterative refinement: pick the median, snap to inlier mean, repeat.
    z_est = float(np.median(z))
    for _ in range(5):
        inliers = np.abs(z - z_est) <= RANSAC_THICKNESS_TOL
        if not inliers.any():
            return None
        new_z = float(z[inliers].mean())
        if abs(new_z - z_est) < 1e-4:
            z_est = new_z
            break
        z_est = new_z
    final_inliers = np.abs(z - z_est) <= RANSAC_THICKNESS_TOL
    if not final_inliers.any():
        return None
    return z_est, np.where(final_inliers)[0]


def _pair_floors_with_ceilings(
    floors: Sequence[dict],
    ceilings: Sequence[dict],
    cfg: SlabConfig,
) -> list[Slab]:
    """Build a sequence of Slab objects from floor+ceiling surfaces.

    Each floor surface pairs with the nearest ceiling above it that's
    within ``MAX_PAIRING_GAP`` and at least ``MIN_STOREY_HEIGHT`` above.
    The slab itself is centered on the floor surface; the ceiling is
    used only to derive storey height for the wall stage.

    Unpaired ceilings become their own Slab (so the user still sees the
    roof when only one floor was detected). Unpaired floors stay too.
    """
    slabs: list[Slab] = []
    used_ceiling_ids: set[int] = set()

    for floor in floors:
        polygon_x, polygon_y = _surface_polygon(floor["points"])
        slabs.append(Slab(
            bottom_z=floor["z"],
            thickness=max(floor["thickness"], cfg.bottom_floor_thickness),
            polygon_x=polygon_x,
            polygon_y=polygon_y,
            points=floor["points"],
        ))
        # Pair to nearest valid ceiling — only logged, doesn't change Slab.
        candidates = [
            (i, c) for i, c in enumerate(ceilings)
            if i not in used_ceiling_ids
            and MIN_STOREY_HEIGHT <= c["z"] - floor["z"] <= MAX_PAIRING_GAP
        ]
        if candidates:
            i_paired, c_paired = min(candidates, key=lambda x: x[1]["z"])
            used_ceiling_ids.add(i_paired)
            log.info(
                "ML slabs: floor z=%.2f paired with ceiling z=%.2f (storey h=%.2f m)",
                floor["z"], c_paired["z"], c_paired["z"] - floor["z"],
            )

    # Emit any ceilings that didn't pair as standalone slabs so the wall
    # stage still has a ceiling reference. They get top_floor_thickness.
    for i, c in enumerate(ceilings):
        if i in used_ceiling_ids:
            # Even paired ceilings need to be emitted as Slab — the pipeline
            # iterates slabs[i+1] to get z_ceiling for storey i.
            polygon_x, polygon_y = _surface_polygon(c["points"])
            slabs.append(Slab(
                bottom_z=c["z"],
                thickness=max(c["thickness"], cfg.top_floor_thickness),
                polygon_x=polygon_x,
                polygon_y=polygon_y,
                points=c["points"],
            ))
        else:
            polygon_x, polygon_y = _surface_polygon(c["points"])
            slabs.append(Slab(
                bottom_z=c["z"],
                thickness=max(c["thickness"], cfg.top_floor_thickness),
                polygon_x=polygon_x,
                polygon_y=polygon_y,
                points=c["points"],
            ))

    slabs.sort(key=lambda s: s.bottom_z)
    return slabs


def _surface_polygon(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed XY polygon for a slab surface.

    Convex hull is the safe default — alpha shapes give a tighter
    boundary but can produce non-monotone polygons that the IFC builder
    doesn't accept. The slab boundary mostly matters for visualisation;
    walls reference their own footprint, not the slab outline.
    """
    if len(points) < 3:
        # Degenerate — return a tiny square around the centroid so IFC
        # doesn't crash on an empty polygon.
        cx, cy = float(points[:, 0].mean()), float(points[:, 1].mean())
        px = np.array([cx - 0.1, cx + 0.1, cx + 0.1, cx - 0.1, cx - 0.1])
        py = np.array([cy - 0.1, cy - 0.1, cy + 0.1, cy + 0.1, cy - 0.1])
        return px, py

    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points[:, :2])
        idx = list(hull.vertices) + [hull.vertices[0]]
        return points[idx, 0].astype(np.float64), points[idx, 1].astype(np.float64)
    except Exception as exc:
        log.warning("ConvexHull failed (%s) — falling back to bbox", exc)
        xs = points[:, 0]
        ys = points[:, 1]
        px = np.array([xs.min(), xs.max(), xs.max(), xs.min(), xs.min()])
        py = np.array([ys.min(), ys.min(), ys.max(), ys.max(), ys.min()])
        return px, py
