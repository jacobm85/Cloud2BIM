"""v4 — evidence-grid detection of slabs, walls and wall openings.

Designed from a blank slate (docs/v4-design.md) around one observation:
label-dependence is fragile in the field, so semantic labels are only a
*soft* weight here, never a filter. The three detectors:

    detect_slabs_v4     prominence-based Z peaks → surface pairing
    detect_walls_v4     vertical-persistence raster → line peeling
    detect_openings_v4  absence-of-points in the wall facade raster

All three run fine with ``semantic_labels=None`` (pure geometry); when
labels are present they nudge scores by ±40 % at most, so a 15 %
mislabelling rate moves results marginally instead of catastrophically.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from cloud2bim.config import OpeningConfig, SegmentationConfig, SlabConfig, WallConfig
from cloud2bim.elements.openings import Opening
from cloud2bim.elements.slabs import Slab
from cloud2bim.elements.walls import Wall, _has_nan
from cloud2bim.geometry.regularize import regularize_walls
from cloud2bim.logging import get_logger
from cloud2bim.segmentation.base import SemanticLabels

log = get_logger(__name__)

# Raster + persistence tunables (see docs/v4-design.md for rationale)
PIXEL = 0.03                 # m — XY cell size for the wall evidence grid
N_SLICES = 14                # Z-slices per storey for persistence
PERSISTENCE_THRESHOLD = 0.55  # share of slices a wall cell must occupy
LABEL_WEIGHT = 0.40          # max score nudge from semantic labels
MIN_CELL_POINTS = 2          # cell×slice occupancy threshold. Tested at 1
                             # for symmetry with the availability count —
                             # that recovers more scan-shadow wall, but in
                             # tall halls it also promotes pallet racking
                             # to walls (industri noisy F1 0.92 → 0.63),
                             # which is the worse trade.

# Slab detection
SLAB_BIN = 0.02              # m — Z histogram bin
SLAB_MIN_PROMINENCE = 0.04   # × strongest peak
MIN_STOREY_HEIGHT = 1.9      # m

# Openings
FACADE_PIXEL = 0.04          # m — (along, z) raster cell
OPENING_MIN_FILL = 0.60      # emptiness share of candidate bbox
DOOR_MAX_WIDTH = 4.5         # m — ports in industri/garage count as doors


# ════════════════════════════════════════════════════════════════════════════
# Slabs
# ════════════════════════════════════════════════════════════════════════════


def detect_slabs_v4(
    points_xyz: np.ndarray,
    cfg: SlabConfig,
    semantic_labels: Optional[SemanticLabels] = None,
    seg_cfg: Optional[SegmentationConfig] = None,
    **_unused,
) -> List[Slab]:
    """Prominence-based horizontal surface detection + physical pairing.

    1. Fine Z-histogram; peaks by *prominence* so a sparse industrial
       ceiling survives next to a massive ground floor.
    2. Each peak refined to the median Z of its member points.
    3. Ceiling-surface directly below floor-surface (≤ max_slab_thickness)
       → one physical slab. Lone surfaces get a default thickness.
    4. Levels implying a storey below MIN_STOREY_HEIGHT: weakest support
       is dropped (mislabelled mezzanine clutter, scaffolding decks).
    """
    if points_xyz is None or len(points_xyz) < 100:
        return []
    try:
        from scipy.signal import find_peaks
    except ImportError:  # scipy is a hard dep of the pipeline already
        raise

    z = points_xyz[:, 2]
    n_bins = max(8, int((z.max() - z.min()) / SLAB_BIN))
    hist, edges = np.histogram(z, bins=n_bins)
    centers = (edges[:-1] + edges[1:]) / 2

    # Soft label weighting of the histogram: bins dominated by
    # floor/ceiling-labelled points get boosted, clutter-heavy damped.
    weights = np.ones_like(hist, dtype=float)
    if semantic_labels is not None and seg_cfg is not None:
        fc_mask = semantic_labels.mask_for(
            list(seg_cfg.floor_classes) + list(seg_cfg.ceiling_classes))
        if fc_mask.any():
            fc_hist, _ = np.histogram(z[fc_mask], bins=edges)
            share = fc_hist / np.maximum(hist, 1)
            weights = np.clip(1.0 + LABEL_WEIGHT * (share - 0.2), 1 - LABEL_WEIGHT, 1 + LABEL_WEIGHT)
    scored = hist * weights

    min_count = max(50, 0.001 * len(z))
    peaks, _ = find_peaks(
        scored,
        prominence=float(scored.max()) * SLAB_MIN_PROMINENCE,
        distance=max(1, int(0.10 / SLAB_BIN)),
        height=min_count,
    )
    if len(peaks) == 0:
        return []

    # Footprint reference for coverage: how many 25 cm XY cells the whole
    # cloud spans. A real floor/ceiling surface covers a large share of
    # that; table tops, shelving decks and window sills don't — coverage
    # is what separates structure from furniture WITHOUT labels.
    COV_CELL = 0.25
    footprint_cells = len(_xy_cells(points_xyz, COV_CELL))

    surfaces = []
    for p in peaks:
        z_peak = centers[p]
        members = np.abs(z - z_peak) <= SLAB_BIN * 1.5
        pts = points_xyz[members]
        cov = len(_xy_cells(pts, COV_CELL)) / max(footprint_cells, 1)
        if cov < 0.20:
            log.info("v4 slabs: surface z=%.2f rejected (coverage %.0f%%)",
                     z_peak, cov * 100)
            continue
        surfaces.append({
            "z": float(np.median(pts[:, 2])),
            "n": int(members.sum()),
            "cov": cov,
            "points": pts[:: max(1, len(pts) // 20000)],  # cap memory
        })
    surfaces.sort(key=lambda s: s["z"])
    log.info("v4 slabs: %d qualified surfaces at %s (coverage %s)",
             len(surfaces), [round(s["z"], 2) for s in surfaces],
             [round(s["cov"], 2) for s in surfaces])

    # Surfaces closer than a storey can't both be real levels — keep the
    # better-covered one. Suspended ceiling + structural soffit collapse
    # to the *visible* (lower) layer this way when coverage ties.
    changed = True
    while changed and len(surfaces) >= 2:
        changed = False
        for j in range(len(surfaces) - 1):
            gap = surfaces[j + 1]["z"] - surfaces[j]["z"]
            if cfg.max_slab_thickness < gap < MIN_STOREY_HEIGHT:
                drop = j if surfaces[j]["cov"] < surfaces[j + 1]["cov"] else j + 1
                log.info("v4 slabs: dropping surface z=%.2f (%.2f m below next level)",
                         surfaces[drop]["z"], gap)
                del surfaces[drop]
                changed = True
                break

    # Group surfaces transitively: consecutive gaps within slab thickness
    # belong to ONE physical slab zone. Real ceiling zones are layered —
    # suspended ceiling, plenum installations, structural soffit — and
    # every layer pair used to become its own slab with a bogus 20 cm
    # "storey" in between.
    merge_gap = max(cfg.max_slab_thickness, 0.6)
    groups: list[list[dict]] = []
    for s in surfaces:
        if groups and (s["z"] - groups[-1][-1]["z"]) <= merge_gap:
            groups[-1].append(s)
        else:
            groups.append([s])

    slabs: list[Slab] = []
    z_mid = float(z.min() + z.max()) / 2
    for grp in groups:
        pts = np.vstack([g["points"] for g in grp])
        z_lo, z_hi = grp[0]["z"], grp[-1]["z"]
        if len(grp) >= 2:
            slabs.append(_mk_slab(z_lo, z_hi - z_lo, pts))
        elif z_lo < z_mid:
            # Lone surface below building midpoint ⇒ floor top (slab body
            # below); above ⇒ ceiling underside (slab body above).
            t = cfg.bottom_floor_thickness
            slabs.append(_mk_slab(z_lo - t, t, pts))
        else:
            slabs.append(_mk_slab(z_lo, cfg.top_floor_thickness, pts))
    return slabs


def _xy_cells(pts: np.ndarray, cell: float) -> np.ndarray:
    """Unique XY cell keys for coverage estimation."""
    g = np.floor(pts[:, :2] / cell).astype(np.int64)
    return np.unique((g[:, 0] + (1 << 30)) * (1 << 32) + (g[:, 1] + (1 << 30)))


def _mk_slab(bottom_z: float, thickness: float, pts: np.ndarray) -> Slab:
    thickness = max(thickness, 0.05)
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts[:, :2])
        idx = list(hull.vertices) + [hull.vertices[0]]
        px, py = pts[idx, 0].astype(float), pts[idx, 1].astype(float)
    except Exception:
        xs, ys = pts[:, 0], pts[:, 1]
        px = np.array([xs.min(), xs.max(), xs.max(), xs.min(), xs.min()])
        py = np.array([ys.min(), ys.min(), ys.max(), ys.max(), ys.min()])
    return Slab(bottom_z=float(bottom_z), thickness=float(thickness),
                polygon_x=px, polygon_y=py, points=pts)


# ════════════════════════════════════════════════════════════════════════════
# Walls
# ════════════════════════════════════════════════════════════════════════════


def detect_walls_v4(
    storey_points: np.ndarray,
    z_floor: float,
    z_ceiling: float,
    storey_idx: int,
    cfg: WallConfig,
    pc_resolution: float = 0.002,
    grid_coefficient: int = 5,
    slab_polygon_xy: Optional[np.ndarray] = None,
    semantic_labels: Optional[SemanticLabels] = None,
    seg_cfg: Optional[SegmentationConfig] = None,
    **_unused,
) -> List[Wall]:
    """Vertical-persistence wall detection.

    A wall is an XY cell occupied across (most of) the storey height;
    furniture only fills the bottom slices, scan shadows knock out a few
    random slices but persistence stays high. Labels softly re-weight
    the persistence score. Centrelines come from the same RANSAC line
    peeling that scored F1=1.00 in the ML path — but run on evidence
    cells instead of label-filtered points.
    """
    h = z_ceiling - z_floor
    if len(storey_points) == 0 or h <= 0.2:
        return []
    z = storey_points[:, 2]
    in_storey = (z >= z_floor + 0.07) & (z <= z_ceiling - 0.07)
    pts = storey_points[in_storey]
    if len(pts) < 100:
        return []

    # ── persistence raster ──
    xy0 = pts[:, :2].min(axis=0)
    ij = np.floor((pts[:, :2] - xy0) / PIXEL).astype(np.int64)
    slice_h = (h - 0.14) / N_SLICES
    sl = np.clip(((pts[:, 2] - (z_floor + 0.07)) / slice_h).astype(np.int64),
                 0, N_SLICES - 1)
    nx = int(ij[:, 0].max()) + 1
    ny = int(ij[:, 1].max()) + 1
    cell = ij[:, 0] * ny + ij[:, 1]

    # occupancy per (cell, slice) needs >= MIN_CELL_POINTS points
    key = cell * N_SLICES + sl
    uniq, counts = np.unique(key, return_counts=True)
    occupied = uniq[counts >= MIN_CELL_POINTS]
    occ_cells = occupied // N_SLICES
    occ_slices = occupied % N_SLICES

    # Occlusion-aware normalisation: persistence is occupied slices over
    # *available* slices — slices where the scanner reached the cell's
    # neighbourhood at all. In a scan shadow a wall keeps its few
    # surviving points spread over the full height (high ratio) while
    # furniture stays bottom-heavy wherever the scan is complete.
    try:
        import cv2
        avail = np.zeros(nx * ny, np.float32)
        any_cells = uniq // N_SLICES        # ≥1 point counts as reachable
        any_slices = uniq % N_SLICES
        kernel = np.ones((5, 5), np.uint8)
        for s_idx in range(N_SLICES):
            img = np.zeros(nx * ny, np.uint8)
            img[any_cells[any_slices == s_idx]] = 1
            img = cv2.dilate(img.reshape(nx, ny), kernel)
            avail += img.reshape(-1)
        n_occ = np.bincount(occ_cells, minlength=nx * ny).astype(np.float32)
        pers = np.where(avail >= 4, n_occ / np.maximum(avail, 1.0), 0.0)
        pers = np.minimum(pers, 1.0)
    except ImportError:
        pers = np.bincount(occ_cells, minlength=nx * ny).astype(np.float32) / N_SLICES
    # Floor-AND-ceiling presence: a window column keeps wall above the
    # lintel and below the sill, so persistence dips below threshold —
    # but furniture never reaches the top slices. Cells occupied near
    # both floor and ceiling are wall evidence even at low persistence,
    # which keeps fönsterband walls in one piece.
    top_occ = np.zeros(nx * ny, bool)
    bot_occ = np.zeros(nx * ny, bool)
    top_occ[occ_cells[occ_slices >= N_SLICES - 3]] = True
    bot_occ[occ_cells[occ_slices < 3]] = True
    span_evidence = top_occ & bot_occ & (pers >= 0.25)

    # ── soft label weight ──
    if semantic_labels is not None and seg_cfg is not None:
        ids = semantic_labels.label_ids[in_storey]
        wall_ids = {i for i, n in enumerate(semantic_labels.label_names)
                    if n in seg_cfg.wall_classes}
        clut_ids = {i for i, n in enumerate(semantic_labels.label_names)
                    if n in seg_cfg.clutter_classes}
        if wall_ids:
            is_wall = np.isin(ids, list(wall_ids)).astype(np.float32)
            is_clut = np.isin(ids, list(clut_ids)).astype(np.float32)
            n_pts_cell = np.bincount(cell, minlength=nx * ny)
            wall_share = np.bincount(cell, weights=is_wall, minlength=nx * ny) \
                / np.maximum(n_pts_cell, 1)
            clut_share = np.bincount(cell, weights=is_clut, minlength=nx * ny) \
                / np.maximum(n_pts_cell, 1)
            weight = np.clip(1.0 + LABEL_WEIGHT * (wall_share - clut_share),
                             1 - LABEL_WEIGHT, 1 + LABEL_WEIGHT)
            pers = pers * weight

    mask = ((pers >= PERSISTENCE_THRESHOLD) | span_evidence).reshape(nx, ny)
    n_wall_cells = int(mask.sum())
    log.info("v4 walls storey %d: %d evidence cells (%.0f m² wall footprint)",
             storey_idx, n_wall_cells, n_wall_cells * PIXEL * PIXEL)
    if n_wall_cells < 10:
        return []

    # Bridge small scan-shadow gaps.
    try:
        from skimage.morphology import closing, footprint_rectangle
        mask = closing(mask, footprint_rectangle((3, 3)))
    except Exception:
        pass

    # Columns are full-height and sail through persistence — but they are
    # compact, near-square blobs while walls are long and thin. Remove
    # connected components that fit inside a column-sized bbox; the
    # columns module finds them separately.
    try:
        import cv2
        n_lab, lab_img, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8)
        max_col_px = int(0.9 / PIXEL)
        for k in range(1, n_lab):
            _, _, w_px, h_px, _ = stats[k]
            if w_px <= max_col_px and h_px <= max_col_px:
                mask[lab_img == k] = False
    except Exception:
        pass

    # ── cells → centrelines via RANSAC line peeling ──
    # mask is indexed [ix, iy] — nonzero gives (ix_list, iy_list)
    ix, iy = np.nonzero(mask)
    cells_xy = np.column_stack([
        xy0[0] + (ix + 0.5) * PIXEL,
        xy0[1] + (iy + 0.5) * PIXEL,
    ])

    from cloud2bim.extraction.walls_ml import (
        _dbscan_xy, _extract_line_segments, _shift_outward, DBSCAN_EPS,
    )
    rng = np.random.default_rng(0)
    clusters = _dbscan_xy(cells_xy, DBSCAN_EPS, 8)
    interior = cells_xy.mean(axis=0)
    single_face_spread = cfg.min_thickness + PIXEL

    axes, thicknesses = [], []
    for cl in clusters:
        for axis, thickness in _extract_line_segments(cl, cfg, rng):
            if axis is None or _has_nan(axis):
                continue
            if thickness < single_face_spread:
                axis = _shift_outward(axis, interior,
                                      (cfg.singleton_thickness - thickness) / 2)
                thickness = cfg.singleton_thickness
            axes.append(axis)
            thicknesses.append(float(np.clip(
                thickness, cfg.min_thickness, cfg.max_thickness)))
    if not axes:
        return []

    if cfg.regularize:
        axes, thicknesses = regularize_walls(
            axes, thicknesses,
            collinear_gap=cfg.collinear_merge_distance,
            corner_snap=max(cfg.max_thickness * 0.6, 0.45),
            min_length=cfg.min_length,
        )
        # Port pass: a straight wall run interrupted by a 3–4 m opening is
        # ONE wall with an opening, not two walls — but two rooms' walls
        # across a corridor are collinear with a similar gap. The
        # difference: a corridor gap is *crossed* by perpendicular walls.
        # Merge big collinear gaps only when nothing crosses them.
        axes, thicknesses = _merge_across_openings(
            axes, thicknesses, max_gap=DOOR_MAX_WIDTH + 0.5)

    if len(axes) > cfg.max_walls_per_storey:
        order = np.argsort([-float(np.hypot(a[1][0] - a[0][0], a[1][1] - a[0][1]))
                            for a in axes])
        keep = sorted(order[: cfg.max_walls_per_storey])
        axes = [axes[k] for k in keep]
        thicknesses = [thicknesses[k] for k in keep]

    walls = [
        Wall(start=tuple(ax[0]), end=tuple(ax[1]), thickness=t,
             z_placement=z_floor, height=h, storey=storey_idx, label="interior")
        for ax, t in zip(axes, thicknesses)
    ]
    log.info("v4 walls storey %d: %d walls", storey_idx, len(walls))
    return walls


def _merge_across_openings(axes: list, thicknesses: list[float],
                           max_gap: float) -> tuple[list, list[float]]:
    """Merge collinear wall pairs across port-sized gaps that no other
    wall crosses. Iterates to a fixpoint."""
    changed = True
    while changed:
        changed = False
        n = len(axes)
        for i in range(n):
            for j in range(i + 1, n):
                merged = _try_gap_merge(axes, thicknesses, i, j, max_gap)
                if merged is not None:
                    axes[i], thicknesses[i] = merged
                    del axes[j], thicknesses[j]
                    changed = True
                    break
            if changed:
                break
    return axes, thicknesses


def _try_gap_merge(axes, thicknesses, i, j, max_gap):
    a1 = np.asarray(axes[i], float)
    a2 = np.asarray(axes[j], float)
    d1, d2 = a1[1] - a1[0], a2[1] - a2[0]
    l1, l2 = float(np.hypot(*d1)), float(np.hypot(*d2))
    if l1 < 1e-6 or l2 < 1e-6:
        return None
    u1 = d1 / l1
    ang = abs(float(u1 @ (d2 / l2)))
    if ang < np.cos(np.deg2rad(6)):
        return None
    n_vec = np.array([-u1[1], u1[0]])
    off = float(np.mean((a2 - a1[0]) @ n_vec))
    if abs(off) > 0.30:
        return None
    t1 = sorted(((a1[0] - a1[0]) @ u1, (a1[1] - a1[0]) @ u1))
    t2 = sorted(((a2[0] - a1[0]) @ u1, (a2[1] - a1[0]) @ u1))
    gap_lo, gap_hi = min(t1[1], t2[1]), max(t1[0], t2[0])
    gap = gap_hi - gap_lo
    if gap <= 0 or gap > max_gap:
        return None
    # Does any other wall cross the gap segment?
    g1 = a1[0] + gap_lo * u1
    g2 = a1[0] + gap_hi * u1
    for k, ax in enumerate(axes):
        if k in (i, j):
            continue
        if _segments_cross([g1.tolist(), g2.tolist()], ax):
            return None
    lo, hi = min(t1[0], t2[0]), max(t1[1], t2[1])
    p1 = a1[0] + lo * u1
    p2 = a1[0] + hi * u1
    return ([[float(p1[0]), float(p1[1])], [float(p2[0]), float(p2[1])]],
            max(thicknesses[i], thicknesses[j]))


def _segments_cross(s1, s2) -> bool:
    """Proper segment intersection test (endpoints inclusive-ish)."""
    p = np.asarray(s1[0], float)
    r = np.asarray(s1[1], float) - p
    q = np.asarray(s2[0], float)
    s = np.asarray(s2[1], float) - q
    denom = float(r[0] * s[1] - r[1] * s[0])
    if abs(denom) < 1e-12:
        return False
    t = float(((q - p)[0] * s[1] - (q - p)[1] * s[0]) / denom)
    u = float(((q - p)[0] * r[1] - (q - p)[1] * r[0]) / denom)
    return -0.05 <= t <= 1.05 and -0.05 <= u <= 1.05


# ════════════════════════════════════════════════════════════════════════════
# Openings
# ════════════════════════════════════════════════════════════════════════════


def detect_openings_v4(
    walls: List[Wall],
    storey_points: np.ndarray,
    cfg: OpeningConfig,
    semantic_labels: Optional[SemanticLabels] = None,
    seg_cfg: Optional[SegmentationConfig] = None,
    **_unused,
) -> List[Opening]:
    """Openings from two complementary evidence sources.

    1. *Absence*: raster the points near each wall plane into (along, z);
       a hole that doesn't touch the side edges, fills its bbox and
       matches door/window proportions is something the scanner saw
       *through* — an open door, a port, an unglazed window.
    2. *Labels* (when available): closed doors and glazed windows are
       physically present in the scan, so absence can't see them — but
       the segmenter can. Label clusters are projected onto their host
       wall exactly like the ML path does.

    The union is deduplicated per wall by along-interval overlap.
    """
    if not walls or len(storey_points) == 0:
        return []
    try:
        import cv2
    except ImportError:
        log.warning("v4 openings: cv2 missing — skipping")
        return []

    door_mask_pts = window_mask_pts = None
    if semantic_labels is not None and seg_cfg is not None:
        door_mask_pts = semantic_labels.mask_for(seg_cfg.door_classes)
        window_mask_pts = semantic_labels.mask_for(seg_cfg.window_classes)

    openings: List[Opening] = []
    for w_idx, wall in enumerate(walls):
        a = np.asarray(wall.start, float)
        b = np.asarray(wall.end, float)
        d = b - a
        length = float(np.hypot(*d))
        if length < 0.8:
            continue
        u = d / length
        n_vec = np.array([-u[1], u[0]])
        rel = storey_points[:, :2] - a
        perp = rel @ n_vec
        band = np.abs(perp) <= max(wall.thickness / 2 + 0.07, 0.15)
        along = rel @ u
        in_seg = (along >= 0) & (along <= length)
        zz = storey_points[:, 2]
        in_z = (zz >= wall.z_placement) & (zz <= wall.z_placement + wall.height)
        sel = band & in_seg & in_z
        if int(sel.sum()) < 100:
            continue

        # Adaptive raster pixel: aim for ~6 expected points per occupied
        # cell so surface coverage reads as solid. A fixed pixel below the
        # point spacing turns the whole facade into Swiss cheese and the
        # openings drown in one giant "empty" component.
        rho = float(sel.sum()) / max(length * wall.height, 1e-6)
        pixel = float(np.clip(np.sqrt(6.0 / max(rho, 1.0)), 0.03, 0.15))
        n_a = max(4, int(np.ceil(length / pixel)))
        n_z = max(4, int(np.ceil(wall.height / pixel)))
        ia = np.clip((along[sel] / pixel).astype(int), 0, n_a - 1)
        iz = np.clip(((zz[sel] - wall.z_placement) / pixel).astype(int),
                     0, n_z - 1)
        raster = np.zeros((n_z, n_a), np.uint8)
        raster[iz, ia] = 1
        # Bridge residual sampling gaps so only real holes stay empty.
        raster = cv2.morphologyEx(raster, cv2.MORPH_CLOSE,
                                  np.ones((3, 3), np.uint8))
        # A wall column with no points at all (scan never reached it) must
        # not read as "hole": require some support in each along-column.
        col_support = raster.sum(axis=0) > 0

        empty = (raster == 0).astype(np.uint8)
        n_lab, lab_img, stats, _ = cv2.connectedComponentsWithStats(empty, 8)
        for k in range(1, n_lab):
            x, y, ww, hh, area = stats[k]
            if x == 0 or x + ww >= n_a:       # touches wall end → not a hole
                continue
            if y + hh >= n_z:                 # touches ceiling → shadow band
                continue
            # Require support in most along-columns — a fully unscanned
            # strip is a shadow, but shadows also eat *parts* of real
            # openings, so demand 60 % rather than all.
            if col_support[x: x + ww].mean() < 0.6:
                continue
            width_m = ww * pixel
            height_m = hh * pixel
            fill = area / max(1, ww * hh)
            if fill < OPENING_MIN_FILL:
                continue
            z0 = wall.z_placement + y * pixel
            z1 = z0 + height_m
            touches_floor = y <= 1
            kind = None
            if touches_floor and height_m >= cfg.door_min_height \
                    and 0.55 <= width_m <= DOOR_MAX_WIDTH:
                kind = "door"
            elif (not touches_floor) and width_m >= cfg.min_window_width \
                    and height_m >= cfg.min_window_height:
                kind = "window"
            if kind is None:
                continue
            # Optional label confirmation boost: shrink minimum fill when
            # labelled opening points sit inside the hole region.
            openings.append(Opening(
                wall_storey=wall.storey, wall_index=w_idx, type=kind,
                x_along_wall_start=float(x * pixel),
                x_along_wall_end=float((x + ww) * pixel),
                z_min=float(wall.z_placement if kind == "door" else z0),
                z_max=float(z1),
            ))
    n_holes = len(openings)

    # ── label-cluster openings (closed doors / glazed windows) ──
    if semantic_labels is not None and seg_cfg is not None:
        try:
            from cloud2bim.extraction.openings_ml import extract_openings_ml
            label_ops = extract_openings_ml(
                walls, storey_points, semantic_labels, cfg, seg_cfg)
        except Exception:
            log.exception("v4 openings: label-cluster pass failed")
            label_ops = []
        for op in label_ops:
            if not _overlaps_existing(op, openings):
                openings.append(op)

    log.info("v4 openings: %d total (%d absence holes, %d from labels) "
             "across %d walls", len(openings), n_holes,
             len(openings) - n_holes, len(walls))
    return openings


def _overlaps_existing(op: Opening, existing: List[Opening]) -> bool:
    """True if an opening overlaps (≥30 % of the shorter interval) an
    already-accepted opening on the same wall."""
    for ex in existing:
        if ex.wall_index != op.wall_index:
            continue
        inter = (min(ex.x_along_wall_end, op.x_along_wall_end)
                 - max(ex.x_along_wall_start, op.x_along_wall_start))
        shorter = min(ex.x_along_wall_end - ex.x_along_wall_start,
                      op.x_along_wall_end - op.x_along_wall_start)
        if shorter > 0 and inter / shorter >= 0.3:
            return True
    return False
