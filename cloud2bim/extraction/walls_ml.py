"""ML-driven wall extraction.

Histogram-based detection conflates wall-faced clutter (cabinets,
bookcases, partition panels) with real walls because they all show up
as dense vertical surfaces in a 2D occupancy grid. Class labels remove
that ambiguity: we only fit walls to points the segmenter marked as
``wall``.

Pipeline per storey:
    1. Keep only wall-labelled points, project to XY
    2. Voxel-downsample so thresholds and runtimes are density-independent
    3. DBSCAN → connected components (separates detached structures)
    4. Per component: *iterative RANSAC line extraction* — fit the best
       2D line, take its inliers, split them into contiguous runs along
       the line (a corner-connected wall network yields many lines, one
       per straight wall), remove the inliers, repeat
    5. Thickness per run from the robust perpendicular spread
    6. Regularise: snap to dominant directions, merge collinear
       fragments, close corners (``extraction.regularize``)

The previous implementation fitted ONE principal axis per DBSCAN
cluster and rejected clusters wider than 0.8 m perpendicular to that
axis. Since the walls of a storey are connected at corners, DBSCAN
returns the whole wall network as a single cluster — which was then
rejected wholesale, or collapsed onto a meaningless diagonal axis.
Iterative line extraction is the fix: it peels one straight wall at a
time out of the connected network.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from cloud2bim.config import SegmentationConfig, WallConfig
from cloud2bim.elements.walls import Wall, _has_nan
from cloud2bim.extraction.regularize import regularize_walls
from cloud2bim.logging import get_logger
from cloud2bim.segmentation.base import SemanticLabels

log = get_logger(__name__)


# Tunables that don't (yet) deserve config fields.
DOWNSAMPLE_VOXEL = 0.03      # m — XY voxel for the working copy. Density-
                             # normalises the cloud so min-point thresholds
                             # mean the same thing for every scanner.
DBSCAN_EPS = 0.35            # m — XY gap that separates detached components
DBSCAN_MIN_PTS = 15          # on the voxel-downsampled cloud
RANSAC_INLIER_DIST = 0.18    # m — captures both faces of walls ≤ ~36 cm
RANSAC_ITERS = 300           # candidate lines per extraction round
RANSAC_SCORE_SUBSET = 25_000 # points used for candidate scoring
MAX_LINES_PER_CLUSTER = 80   # safety cap on extraction rounds
RUN_GAP = 0.50               # m — gap along the line that splits two runs
                             # (door gaps re-merge later via collinear merge)
MIN_RUN_POINTS = 12          # voxelised points for a run to count


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

    # Wall height comes from slab spacing, not point Z — XY is enough.
    n_in = len(wall_pts)
    wall_xy = _voxel_downsample_xy(wall_pts[:, :2], DOWNSAMPLE_VOXEL)
    log.info(
        "ML walls storey %d: voxel-normalised %d → %d points (%.0f cm grid)",
        storey_idx, n_in, len(wall_xy), DOWNSAMPLE_VOXEL * 100,
    )

    clusters = _dbscan_xy(wall_xy, DBSCAN_EPS, DBSCAN_MIN_PTS)
    log.info("ML walls storey %d: %d connected components", storey_idx, len(clusters))

    wall_axes: list[list[list[float]]] = []
    wall_thicknesses: list[float] = []
    rng = np.random.default_rng(0)  # deterministic re-runs
    for cluster_pts in clusters:
        segments = _extract_line_segments(cluster_pts, cfg, rng)
        for axis, thickness in segments:
            if axis is None or _has_nan(axis):
                continue
            wall_axes.append(axis)
            wall_thicknesses.append(
                float(np.clip(thickness, cfg.min_thickness, cfg.max_thickness))
            )
    log.info(
        "ML walls storey %d: %d raw segments from line extraction",
        storey_idx, len(wall_axes),
    )
    if not wall_axes:
        log.warning("ML walls storey %d: no wall segments extracted", storey_idx)
        return []

    # Regularise: dominant-direction snap, collinear merge, corner close.
    wall_axes, wall_thicknesses = regularize_walls(
        wall_axes, wall_thicknesses,
        collinear_gap=cfg.collinear_merge_distance,
        corner_snap=max(cfg.max_thickness * 0.6, 0.45),
        min_length=cfg.min_length,
    )

    # Cap to safety limit, keeping the longest walls.
    if len(wall_axes) > cfg.max_walls_per_storey:
        log.warning(
            "ML walls storey %d: clipping %d walls down to max %d (keeping longest)",
            storey_idx, len(wall_axes), cfg.max_walls_per_storey,
        )
        order = np.argsort([-_axis_length(a) for a in wall_axes])
        keep = sorted(order[: cfg.max_walls_per_storey])
        wall_axes = [wall_axes[k] for k in keep]
        wall_thicknesses = [wall_thicknesses[k] for k in keep]

    exterior_flags = _classify_exterior(wall_axes, wall_xy)
    wall_height = z_ceiling - z_floor
    walls = [
        Wall(
            start=tuple(ax[0]),
            end=tuple(ax[1]),
            thickness=t,
            z_placement=z_floor,
            height=wall_height,
            storey=storey_idx,
            label="exterior" if ext else "interior",
        )
        for ax, t, ext in zip(wall_axes, wall_thicknesses, exterior_flags)
    ]
    log.info(
        "ML walls storey %d: %d walls finalised (%d exterior)",
        storey_idx, len(walls), sum(exterior_flags),
    )
    return walls


# ── internals ─────────────────────────────────────────────────────────────────


EXTERIOR_HULL_TOL = 0.40   # m — wall axis within this of the footprint hull
                           # counts as exterior. Covers hull-chord cutoff at
                           # slightly concave corners plus axis-vs-face offset.


def _classify_exterior(wall_axes: list, wall_xy: np.ndarray) -> list[bool]:
    """Flag walls on the building envelope.

    The envelope is approximated by the convex hull of all wall points in
    the storey; a wall whose start, mid and end points all lie within
    ``EXTERIOR_HULL_TOL`` of the hull boundary is exterior. Interior walls
    sit well inside the hull, so the test is forgiving about tolerance.
    Concave footprints (L/U-shapes) mark the re-entrant walls as interior
    — wrong for those segments, but IsExternal=false is also what the old
    behaviour gave, so concavity never makes things worse.
    """
    if not wall_axes:
        return []
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(wall_xy)
        hull_pts = wall_xy[hull.vertices]
    except Exception as exc:
        log.warning("Exterior classification skipped (hull failed: %s)", exc)
        return [False] * len(wall_axes)

    # Hull edges as (a, b) pairs, closed.
    edges = [
        (hull_pts[i], hull_pts[(i + 1) % len(hull_pts)])
        for i in range(len(hull_pts))
    ]

    def dist_to_hull(p: np.ndarray) -> float:
        best = np.inf
        for a, b in edges:
            ab = b - a
            denom = float(ab @ ab)
            if denom < 1e-12:
                d = float(np.hypot(*(p - a)))
            else:
                t = float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
                d = float(np.hypot(*(p - (a + t * ab))))
            if d < best:
                best = d
        return best

    flags: list[bool] = []
    for ax in wall_axes:
        a = np.asarray(ax[0], dtype=float)
        b = np.asarray(ax[1], dtype=float)
        probes = (a, (a + b) / 2, b)
        flags.append(all(dist_to_hull(p) <= EXTERIOR_HULL_TOL for p in probes))
    return flags


def _axis_length(ax) -> float:
    return float(np.hypot(ax[1][0] - ax[0][0], ax[1][1] - ax[0][1]))


def _voxel_downsample_xy(xy: np.ndarray, voxel: float) -> np.ndarray:
    """Snap to a 2D voxel grid and keep one representative per cell."""
    if len(xy) == 0:
        return xy
    grid = np.floor(xy / voxel).astype(np.int64)
    # Combine the two cell coords into a single 64-bit key for unique().
    # The shift keeps positive and negative coords separable.
    keys = (grid[:, 0] + (1 << 30)) * (1 << 32) + (grid[:, 1] + (1 << 30))
    _, unique_idx = np.unique(keys, return_index=True)
    return xy[unique_idx]


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


def _extract_line_segments(
    cluster_xy: np.ndarray,
    cfg: WallConfig,
    rng: np.random.Generator,
) -> list[tuple[list[list[float]], float]]:
    """Peel straight wall segments out of one connected component.

    Repeats: RANSAC the best line → PCA-refine on its inliers → split the
    inliers into contiguous runs along the line → emit each long-enough
    run as (axis, thickness) → remove the inliers → continue on the rest.
    """
    segments: list[tuple[list[list[float]], float]] = []
    remaining = cluster_xy
    for _ in range(MAX_LINES_PER_CLUSTER):
        if len(remaining) < MIN_RUN_POINTS:
            break
        line = _ransac_line(remaining, RANSAC_INLIER_DIST, rng)
        if line is None:
            break
        origin, direction = line
        d = _perp_distances(remaining, origin, direction)
        inlier_mask = d <= RANSAC_INLIER_DIST
        if int(inlier_mask.sum()) < MIN_RUN_POINTS:
            break

        # PCA refine: re-centre the line on its inliers (2 rounds). This
        # pulls the axis to the midline when the wall was scanned from
        # both sides (two parallel point faces).
        inliers = remaining[inlier_mask]
        for _ in range(2):
            origin = inliers.mean(axis=0)
            cov = np.cov((inliers - origin).T)
            if not np.isfinite(cov).all():
                break
            eigvals, eigvecs = np.linalg.eigh(cov)
            direction = eigvecs[:, int(np.argmax(eigvals))]
            d = _perp_distances(remaining, origin, direction)
            inlier_mask = d <= RANSAC_INLIER_DIST
            inliers = remaining[inlier_mask]
            if len(inliers) < MIN_RUN_POINTS:
                break
        if len(inliers) < MIN_RUN_POINTS:
            remaining = remaining[~inlier_mask]
            continue

        segments.extend(_runs_to_segments(inliers, origin, direction, cfg))
        remaining = remaining[~inlier_mask]
    return segments


def _perp_distances(xy: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> np.ndarray:
    normal = np.array([-direction[1], direction[0]])
    return np.abs((xy - origin) @ normal)


def _ransac_line(
    xy: np.ndarray,
    inlier_dist: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Best 2-point line by inlier count. Returns (origin, unit_direction).

    Scoring runs on a random subset so dense storeys don't make each
    candidate evaluation O(N); the caller recomputes exact inliers on the
    full set afterwards.
    """
    n = len(xy)
    if n < 2:
        return None
    score_pts = xy
    if n > RANSAC_SCORE_SUBSET:
        score_pts = xy[rng.choice(n, RANSAC_SCORE_SUBSET, replace=False)]

    best_score = 0
    best: tuple[np.ndarray, np.ndarray] | None = None
    for _ in range(RANSAC_ITERS):
        i, j = rng.integers(0, n, size=2)
        p, q = xy[i], xy[j]
        v = q - p
        norm = float(np.hypot(v[0], v[1]))
        if norm < 0.30:  # too close — direction estimate would be noise
            continue
        u = v / norm
        score = int((_perp_distances(score_pts, p, u) <= inlier_dist).sum())
        if score > best_score:
            best_score = score
            best = (p.astype(float), u.astype(float))
    if best is None or best_score < MIN_RUN_POINTS:
        return None
    return best


def _runs_to_segments(
    inliers: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
    cfg: WallConfig,
) -> list[tuple[list[list[float]], float]]:
    """Split a line's inliers into contiguous runs; one segment per run.

    A line through a building typically crosses several distinct walls
    (e.g. the same line continues through a corridor into the next room's
    wall) — the gaps between runs separate them.
    """
    proj = (inliers - origin) @ direction
    order = np.argsort(proj)
    proj_sorted = proj[order]
    pts_sorted = inliers[order]

    breaks = np.where(np.diff(proj_sorted) > RUN_GAP)[0] + 1
    segments: list[tuple[list[list[float]], float]] = []
    start = 0
    for stop in list(breaks) + [len(proj_sorted)]:
        run_pts = pts_sorted[start:stop]
        run_proj = proj_sorted[start:stop]
        start = stop
        if len(run_pts) < MIN_RUN_POINTS:
            continue
        seg_len = float(run_proj[-1] - run_proj[0])
        if seg_len < cfg.min_length:
            continue
        # Robust thickness: central 96 % of the perpendicular spread, so a
        # few stray voxels don't inflate a 10 cm partition to 40 cm.
        perp = (run_pts - origin) @ np.array([-direction[1], direction[0]])
        thickness = float(np.percentile(perp, 98) - np.percentile(perp, 2))
        mid_perp = float(np.median(perp))
        normal = np.array([-direction[1], direction[0]])
        p1 = origin + run_proj[0] * direction + mid_perp * normal
        p2 = origin + run_proj[-1] * direction + mid_perp * normal
        segments.append((
            [[float(p1[0]), float(p1[1])], [float(p2[0]), float(p2[1])]],
            thickness,
        ))
    return segments
