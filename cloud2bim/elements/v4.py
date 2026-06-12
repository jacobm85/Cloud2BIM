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
    ev = wall_evidence(storey_points, z_floor, z_ceiling,
                       semantic_labels=semantic_labels, seg_cfg=seg_cfg)
    if ev is None:
        return []
    cells_raw, cells_xy, in_storey = ev
    log.info("v4 walls storey %d: %d evidence cells (%.0f m² wall footprint)",
             storey_idx, len(cells_raw), len(cells_raw) * PIXEL * PIXEL)
    if len(cells_xy) < 10:
        return []

    from cloud2bim.extraction.walls_ml import (
        _dbscan_xy, _extract_line_segments, DBSCAN_EPS,
    )
    rng = np.random.default_rng(0)
    clusters = _dbscan_xy(cells_xy, DBSCAN_EPS, 8)
    return _walls_from_clusters(
        clusters, cells_raw, h, z_floor, storey_idx, cfg,
        storey_points, in_storey, semantic_labels, seg_cfg, rng)


def wall_evidence(
    storey_points: np.ndarray,
    z_floor: float,
    z_ceiling: float,
    semantic_labels: Optional[SemanticLabels] = None,
    seg_cfg: Optional[SegmentationConfig] = None,
):
    """Shared evidence raster: persistence + occlusion normalisation +
    span evidence + soft labels + morphology.

    Returns (cells_raw, cells_xy, in_storey_mask) or None:
        cells_raw — XY centres of mask cells BEFORE closing (thickness truth)
        cells_xy  — XY centres after closing + column-blob removal (peeling)
    Used by both v4 (line peeling) and v5 (carrier accumulation).
    """
    h = z_ceiling - z_floor
    z = storey_points[:, 2]
    in_storey = (z >= z_floor + 0.07) & (z <= z_ceiling - 0.07)
    pts = storey_points[in_storey]
    if len(pts) < 100:
        return None

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
    if int(mask.sum()) < 10:
        return None

    # Keep the un-closed mask for thickness measurement: closing widens a
    # single scanned face to ~3 cell rows, which is indistinguishable from
    # a thin two-faced wall at this resolution.
    ix0, iy0 = np.nonzero(mask)
    cells_raw = np.column_stack([
        xy0[0] + (ix0 + 0.5) * PIXEL,
        xy0[1] + (iy0 + 0.5) * PIXEL,
    ])

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
        min_col_px = int(0.15 / PIXEL)
        for k in range(1, n_lab):
            _, _, w_px, h_px, _ = stats[k]
            # A column is compact AND thick in BOTH directions. A thin
            # sub-metre line is a scan-shadow fragment of a wall — on a
            # corridor-scanned hospital floor the room dividers arrive
            # exactly like that, and removing them cost 29 % of all wall
            # evidence (every vertical wall vanished from the model).
            if (min_col_px <= w_px <= max_col_px
                    and min_col_px <= h_px <= max_col_px):
                mask[lab_img == k] = False
    except Exception:
        pass

    # mask is indexed [ix, iy] — nonzero gives (ix_list, iy_list)
    ix, iy = np.nonzero(mask)
    cells_xy = np.column_stack([
        xy0[0] + (ix + 0.5) * PIXEL,
        xy0[1] + (iy + 0.5) * PIXEL,
    ])
    return cells_raw, cells_xy, in_storey


def _walls_from_clusters(clusters, cells_raw, h, z_floor, storey_idx, cfg,
                         storey_points, in_storey, semantic_labels, seg_cfg,
                         rng):
    """v4 tail: peel clusters → regularize → gap merge → thickness."""
    from cloud2bim.extraction.walls_ml import _extract_line_segments

    # Raw axes only here — thickness is estimated AFTER all merging, from
    # the evidence cells around the final axis. Line peeling often puts a
    # thick wall's two faces in separate runs; per-run estimation then
    # reports a single face (~3 cm) for every wall, and the two face-walls
    # render as a doubled fat wall.
    axes, thicknesses = [], []
    for cl in clusters:
        for axis, thickness in _extract_line_segments(cl, cfg, rng):
            if axis is None or _has_nan(axis):
                continue
            axes.append(axis)
            thicknesses.append(float(np.clip(
                thickness, cfg.min_thickness, cfg.max_thickness)))
    if not axes:
        return []

    if cfg.regularize:
        # collinear_offset 0.30: the two faces of a 20–30 cm wall arrive
        # as parallel segments that far apart — they must collapse to the
        # centreline before thickness refinement.
        axes, thicknesses = regularize_walls(
            axes, thicknesses,
            collinear_offset=0.30,
            collinear_gap=cfg.collinear_merge_distance,
            corner_snap=max(cfg.max_thickness * 0.6, 0.45),
            min_length=cfg.min_length,
        )
        # Port pass: a straight wall run interrupted by a 3–4 m opening is
        # ONE wall with an opening, not two walls — but two rooms' walls
        # across a corridor are collinear with a similar gap. The
        # difference: a corridor gap is *crossed* by perpendicular walls.
        # Merge big collinear gaps only when nothing crosses them.
        # Door/window-labelled points are soft evidence that an even
        # longer gap is an opening (port + adjacent scan shadow) rather
        # than two separate walls.
        opening_xy = None
        if semantic_labels is not None and seg_cfg is not None:
            op_mask = semantic_labels.mask_for(
                list(seg_cfg.door_classes) + list(seg_cfg.window_classes))
            op_mask = op_mask & in_storey
            if op_mask.any():
                opening_xy = storey_points[op_mask][:, :2]
        axes, thicknesses = _merge_across_openings(
            axes, thicknesses, max_gap=DOOR_MAX_WIDTH + 0.5,
            opening_xy=opening_xy)

    axes, thicknesses = _refine_thickness(axes, cells_raw, cfg)
    # Near-coincident parallel walls (a wall plus its parallel duct/
    # curtain layer, both measured "two-faced") duplicate every opening
    # they host — keep the longer of an overlapping pair.
    from cloud2bim.v5.plan import _dedupe_parallel
    axes, thicknesses = _dedupe_parallel(axes, thicknesses)

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


def _refine_thickness(
    axes: list, cells_xy: np.ndarray, cfg: WallConfig,
) -> tuple[list, list[float]]:
    """Pair wall faces and re-estimate thickness from evidence cells.

    The persistence mask is essentially one cell-row per scanned wall
    FACE, so line peeling emits each face of a thick wall as its own
    axis. Pairing: two near-parallel axes whose perpendicular distance
    is a plausible wall thickness and that overlap along their length
    are the two faces of ONE wall — replace them with the midline.
    Leftover single faces get singleton_thickness shifted away from the
    building interior.
    """
    from cloud2bim.extraction.walls_ml import _shift_outward

    interior = cells_xy.mean(axis=0)

    # Measure each axis: is the perpendicular cell distribution bimodal
    # (two faces of one wall → distance between modes = thickness) or one
    # contiguous blob (a single noisy face is 2–4 cell rows wide — plain
    # spread can't tell that apart from a thin two-faced wall)?
    measured: list[float] = []   # face distance; 0.0 = single face
    offsets: list[float] = []    # midline offset to apply when two-faced
    for ax in axes:
        a = np.asarray(ax[0], float)
        b = np.asarray(ax[1], float)
        d = b - a
        length = float(np.hypot(*d))
        if length < 1e-6:
            measured.append(0.0)
            offsets.append(0.0)
            continue
        u = d / length
        n_vec = np.array([-u[1], u[0]])
        rel = cells_xy - a
        along = rel @ u
        perp = rel @ n_vec
        # Stay clear of the segment ends: crossing walls' cell rows run
        # right through the corner zone perpendicular to this axis and
        # contaminate the perp distribution (an L-corner reads as 40 cm
        # of "thickness" otherwise).
        end_margin = min(0.35, 0.25 * length)
        near = (np.abs(perp) <= cfg.max_thickness / 2 + 2 * PIXEL) \
            & (along >= end_margin) & (along <= length - end_margin)
        face_dist, mid_off = 0.0, 0.0
        if int(near.sum()) >= 6:
            order_p = np.argsort(perp[near])
            p = perp[near][order_p]
            al = along[near][order_p]
            splits = np.where(np.diff(p) > 2 * PIXEL)[0] + 1
            # A perp-group only counts as a wall FACE if it runs along
            # most of the axis. A car/cabinet parked against the wall
            # forms a perp-group too, but covers a fraction of the
            # length — counting it as a face inflates the thickness and
            # blocks the singleton shift.
            groups = []
            for g_p, g_a in zip(np.split(p, splits), np.split(al, splits)):
                if len(g_p) < 3:
                    continue
                bins = np.unique((g_a / 0.5).astype(np.int64))
                coverage = len(bins) / max(1.0, length / 0.5)
                if coverage >= 0.5:
                    groups.append((g_p, g_a))
            if len(groups) >= 2:
                lo = float(np.median(groups[0][0]))
                hi = float(np.median(groups[-1][0]))
                face_dist = hi - lo
                mid_off = (hi + lo) / 2
            elif len(groups) == 1:
                # One contiguous blob: a thin wall's two faces have no
                # cell-row gap between them at 3 cm resolution. A single
                # noisy face is 1–2 rows; anything wider is a wall. The
                # spread must be measured AFTER removing the linear trend
                # — a slightly tilted axis makes one straight cell row
                # smear several cm of apparent perp width over 30 m.
                g_p, g_a = groups[0]
                slope, icpt = np.polyfit(g_a, g_p, 1)
                resid = g_p - (slope * g_a + icpt)
                w = float(np.percentile(resid, 98) - np.percentile(resid, 2))
                if w > 3 * PIXEL:
                    face_dist = w
                    mid_off = float(np.median(g_p))
        measured.append(face_dist)
        offsets.append(mid_off)

    # pair single faces: nearest parallel partner at wall-like distance
    n = len(axes)
    paired = [-1] * n
    candidates: list[tuple[float, int, int]] = []
    for i in range(n):
        if measured[i] > 0:
            continue
        for j in range(i + 1, n):
            if measured[j] > 0:
                continue
            geom = _face_pair_geometry(axes[i], axes[j], cfg)
            if geom is not None:
                candidates.append((geom, i, j))
    for dist, i, j in sorted(candidates):
        if paired[i] < 0 and paired[j] < 0:
            paired[i], paired[j] = j, i

    out_axes, out_t = [], []
    consumed: set[int] = set()
    for i, ax in enumerate(axes):
        if i in consumed:
            continue
        if measured[i] > 0:
            # Two faces visible around this axis — recentre on their
            # midline, thickness = face distance (+ one cell of bias).
            a = np.asarray(ax[0], float)
            b = np.asarray(ax[1], float)
            d = b - a
            length = float(np.hypot(*d))
            u = d / length
            n_vec = np.array([-u[1], u[0]])
            a = a + offsets[i] * n_vec
            b = b + offsets[i] * n_vec
            out_axes.append([[float(a[0]), float(a[1])],
                             [float(b[0]), float(b[1])]])
            out_t.append(float(np.clip(
                measured[i] + PIXEL, cfg.min_thickness, cfg.max_thickness)))
            continue
        j = paired[i]
        if j >= 0:
            consumed.add(j)
            mid, dist = _face_midline(axes[i], axes[j])
            out_axes.append(mid)
            out_t.append(float(np.clip(
                dist + PIXEL, cfg.min_thickness, cfg.max_thickness)))
            continue
        out_axes.append(_shift_outward(ax, interior,
                                       cfg.singleton_thickness / 2))
        out_t.append(cfg.singleton_thickness)
    log.info("v4 walls: thickness refinement %d axes -> %d walls "
             "(%d face pairs)", n, len(out_axes), len(consumed))
    return out_axes, out_t


def _face_pair_geometry(ax1, ax2, cfg: WallConfig) -> float | None:
    """Perpendicular distance if two axes look like faces of one wall."""
    a1 = np.asarray(ax1, float)
    a2 = np.asarray(ax2, float)
    d1 = a1[1] - a1[0]
    d2 = a2[1] - a2[0]
    l1, l2 = float(np.hypot(*d1)), float(np.hypot(*d2))
    if l1 < 1e-6 or l2 < 1e-6:
        return None
    u = d1 / l1
    if abs(float(u @ d2)) / l2 < np.cos(np.deg2rad(6)):
        return None
    n_vec = np.array([-u[1], u[0]])
    dist = abs(float(((a2[0] - a1[0]) @ n_vec + (a2[1] - a1[0]) @ n_vec) / 2))
    if not (cfg.min_thickness + PIXEL <= dist <= cfg.max_thickness):
        return None
    s = sorted([float((a2[0] - a1[0]) @ u), float((a2[1] - a1[0]) @ u)])
    overlap = min(s[1], l1) - max(s[0], 0.0)
    if overlap < 0.5 * min(l1, l2) or overlap < cfg.pair_min_overlap:
        return None
    return dist


def _face_midline(ax1, ax2) -> tuple[list, float]:
    """Midline spanning the union of two face projections + face distance."""
    a1 = np.asarray(ax1, float)
    a2 = np.asarray(ax2, float)
    d1 = a1[1] - a1[0]
    l1 = float(np.hypot(*d1))
    u = d1 / l1
    n_vec = np.array([-u[1], u[0]])
    off2 = float(((a2[0] - a1[0]) @ n_vec + (a2[1] - a1[0]) @ n_vec) / 2)
    ts = [0.0, l1, float((a2[0] - a1[0]) @ u), float((a2[1] - a1[0]) @ u)]
    lo, hi = min(ts), max(ts)
    base = a1[0] + (off2 / 2) * n_vec
    p1 = base + lo * u
    p2 = base + hi * u
    return ([[float(p1[0]), float(p1[1])], [float(p2[0]), float(p2[1])]],
            abs(off2))


def _merge_across_openings(axes: list, thicknesses: list[float],
                           max_gap: float,
                           opening_xy: np.ndarray | None = None
                           ) -> tuple[list, list[float]]:
    """Merge collinear wall pairs across port-sized gaps that no other
    wall crosses. Iterates to a fixpoint.

    ``opening_xy``: door/window-labelled points. When the gap contains
    enough of them the merge distance is extended — the gap is then an
    opening widened by an adjacent scan shadow, not two separate walls.

    Hot path on large storeys (hundreds of segments → O(n²) pairs per
    sweep) — _try_gap_merge therefore works in plain scalar floats; the
    numpy version spent 50 s on a 205-wall hospital storey.
    """
    changed = True
    while changed:
        changed = False
        n = len(axes)
        for i in range(n):
            for j in range(i + 1, n):
                merged = _try_gap_merge(axes, thicknesses, i, j, max_gap,
                                        opening_xy)
                if merged is not None:
                    axes[i], thicknesses[i] = merged
                    del axes[j], thicknesses[j]
                    changed = True
                    break
            if changed:
                break
    return axes, thicknesses


_COS6 = float(np.cos(np.deg2rad(6)))


MAX_LABELLED_GAP = 10.0   # m — gap-merge ceiling when opening labels fill it


def _try_gap_merge(axes, thicknesses, i, j, max_gap, opening_xy=None):
    (x1a, y1a), (x1b, y1b) = axes[i]
    (x2a, y2a), (x2b, y2b) = axes[j]
    d1x, d1y = x1b - x1a, y1b - y1a
    d2x, d2y = x2b - x2a, y2b - y2a
    l1 = (d1x * d1x + d1y * d1y) ** 0.5
    l2 = (d2x * d2x + d2y * d2y) ** 0.5
    if l1 < 1e-6 or l2 < 1e-6:
        return None
    ux, uy = d1x / l1, d1y / l1
    if abs((ux * d2x + uy * d2y) / l2) < _COS6:
        return None
    nx_, ny_ = -uy, ux
    off = ((x2a - x1a) * nx_ + (y2a - y1a) * ny_
           + (x2b - x1a) * nx_ + (y2b - y1a) * ny_) / 2
    if abs(off) > 0.30:
        return None
    s2a = (x2a - x1a) * ux + (y2a - y1a) * uy
    s2b = (x2b - x1a) * ux + (y2b - y1a) * uy
    t1_lo, t1_hi = (0.0, l1)
    t2_lo, t2_hi = (s2a, s2b) if s2a <= s2b else (s2b, s2a)
    gap_lo = min(t1_hi, t2_hi)
    gap_hi = max(t1_lo, t2_lo)
    gap = gap_hi - gap_lo
    if gap <= 0 or gap > MAX_LABELLED_GAP:
        return None
    if gap > max_gap:
        # Beyond plain port distance: require door/window-labelled points
        # in the gap corridor as evidence that this is one wall with an
        # opening (plus scan shadow), not two separate walls.
        if opening_xy is None or len(opening_xy) == 0:
            return None
        relx = opening_xy[:, 0] - x1a
        rely = opening_xy[:, 1] - y1a
        o_along = relx * ux + rely * uy
        o_perp = relx * nx_ + rely * ny_
        n_in_gap = int(((o_along > gap_lo) & (o_along < gap_hi)
                        & (np.abs(o_perp) < 0.4)).sum())
        if n_in_gap < 20:
            return None
    # Does any other wall cross the gap segment?
    g1 = (x1a + gap_lo * ux, y1a + gap_lo * uy)
    g2 = (x1a + gap_hi * ux, y1a + gap_hi * uy)
    for k, ax in enumerate(axes):
        if k == i or k == j:
            continue
        if _segments_cross([g1, g2], ax):
            return None
    lo = min(t1_lo, t2_lo)
    hi = max(t1_hi, t2_hi)
    p1 = (x1a + lo * ux, y1a + lo * uy)
    p2 = (x1a + hi * ux, y1a + hi * uy)
    return ([[p1[0], p1[1]], [p2[0], p2[1]]],
            max(thicknesses[i], thicknesses[j]))


def _segments_cross(s1, s2) -> bool:
    """True when s2 passes THROUGH s1 — not when it merely abuts.

    Used to protect the gap merge from bridging corridors: a corridor
    is flanked by walls that CROSS the gap line. A perpendicular wall
    that ends ON the line (a T-junction) must NOT block the merge —
    long walls are full of T-joints, and treating them as crossings
    kept window walls split at exactly the windows.

    Scalar floats on purpose — called O(n³) times in the worst case."""
    px, py = s1[0]
    rx, ry = s1[1][0] - px, s1[1][1] - py
    qx, qy = s2[0]
    sx, sy = s2[1][0] - qx, s2[1][1] - qy
    denom = rx * sy - ry * sx
    if abs(denom) < 1e-12:
        return False
    dqx, dqy = qx - px, qy - py
    t = (dqx * sy - dqy * sx) / denom
    u = (dqx * ry - dqy * rx) / denom
    # u strictly interior on the OTHER wall = it continues on both sides.
    return -0.05 <= t <= 1.05 and 0.10 <= u <= 0.90


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

    # Coarse spatial hash: each wall only needs points within ~0.5 m of
    # its own line. Filtering all N storey points per wall cost 86 s for
    # a 205-wall hospital storey (205 × six vector ops over 20M points);
    # gathering candidate bins along the wall is ~100× less data.
    BIN = 2.0
    bx = np.floor(storey_points[:, 0] / BIN).astype(np.int64)
    by = np.floor(storey_points[:, 1] / BIN).astype(np.int64)
    bin_key = (bx + (1 << 30)) * (1 << 32) + (by + (1 << 30))
    order = np.argsort(bin_key, kind="stable")
    sorted_keys = bin_key[order]
    uniq_keys, starts = np.unique(sorted_keys, return_index=True)
    bounds = {int(k): (int(s), int(e)) for k, s, e in zip(
        uniq_keys, starts, list(starts[1:]) + [len(sorted_keys)])}

    def _points_near(a, b, margin):
        cells = set()
        n_steps = max(2, int(np.hypot(b[0] - a[0], b[1] - a[1]) / BIN * 2) + 2)
        for s_frac in np.linspace(0.0, 1.0, n_steps):
            px = a[0] + s_frac * (b[0] - a[0])
            py = a[1] + s_frac * (b[1] - a[1])
            cx0 = int(np.floor((px - margin) / BIN))
            cy0 = int(np.floor((py - margin) / BIN))
            cx1 = int(np.floor((px + margin) / BIN))
            cy1 = int(np.floor((py + margin) / BIN))
            for cx in range(cx0, cx1 + 1):
                for cy in range(cy0, cy1 + 1):
                    cells.add((cx + (1 << 30)) * (1 << 32) + (cy + (1 << 30)))
        idx_parts = [order[s:e] for c in cells
                     for (s, e) in [bounds.get(int(c), (0, 0))] if e > s]
        if not idx_parts:
            return np.empty(0, np.int64)
        return np.concatenate(idx_parts)

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
        cand = _points_near(a, b, max(wall.thickness / 2 + 0.07, 0.15) + 0.1)
        if len(cand) < 100:
            continue
        pts_w = storey_points[cand]
        rel = pts_w[:, :2] - a
        perp = rel @ n_vec
        band = np.abs(perp) <= max(wall.thickness / 2 + 0.07, 0.15)
        # The facade raster must come from the wall's dominant FACE
        # plane, not the whole thickness band: when curtains, radiators
        # or ducts run parallel to a window wall, the measured thickness
        # inflates and the full band rasterises the parallel layer right
        # over the window holes. Find the strongest perpendicular mode
        # and keep a tight slab around it.
        if band.any():
            p_band = perp[band]
            h_p, e_p = np.histogram(p_band, bins=max(
                4, int((p_band.max() - p_band.min()) / 0.03) + 1))
            mode = float((e_p[np.argmax(h_p)] + e_p[np.argmax(h_p) + 1]) / 2)
            band = np.abs(perp - mode) <= 0.12
        along = rel @ u
        in_seg = (along >= 0) & (along <= length)
        zz = pts_w[:, 2]
        # Skip the floor band: floor-surface points inside the wall band
        # occupy the raster's bottom row everywhere, so no hole could
        # ever "touch the floor" and every door classified as a window.
        FLOOR_SKIP = 0.12
        z_base = wall.z_placement + FLOOR_SKIP
        in_z = (zz >= z_base) & (zz <= wall.z_placement + wall.height)
        sel = band & in_seg & in_z
        if int(sel.sum()) < 100:
            continue

        # Adaptive raster pixel: aim for ~6 expected points per occupied
        # cell so surface coverage reads as solid. A fixed pixel below the
        # point spacing turns the whole facade into Swiss cheese and the
        # openings drown in one giant "empty" component.
        raster_h = wall.height - FLOOR_SKIP
        rho = float(sel.sum()) / max(length * raster_h, 1e-6)
        pixel = float(np.clip(np.sqrt(6.0 / max(rho, 1.0)), 0.03, 0.15))
        n_a = max(4, int(np.ceil(length / pixel)))
        n_z = max(4, int(np.ceil(raster_h / pixel)))
        ia = np.clip((along[sel] / pixel).astype(int), 0, n_a - 1)
        iz = np.clip(((zz[sel] - z_base) / pixel).astype(int),
                     0, n_z - 1)
        # Density-thresholded occupancy: glass panes return SOME points
        # (speckle, frames behind), and a single point per cell used to
        # mark the whole pane as solid wall — mullioned windows then
        # shrank to fragments. A cell is wall only with a meaningful
        # share of the expected wall-point density.
        counts = np.zeros((n_z, n_a), np.float32)
        np.add.at(counts, (iz, ia), 1.0)
        cell_expect = rho * pixel * pixel
        # Only demand more than one point per cell when the wall is dense
        # enough that real wall cells carry many points — otherwise a
        # diluted scan's legitimate wall cells get blanked too.
        thr = 2.0 if cell_expect >= 12.0 else 1.0
        raster = (counts >= thr).astype(np.uint8)
        # Bridge residual sampling gaps so only real holes stay empty.
        raster = cv2.morphologyEx(raster, cv2.MORPH_CLOSE,
                                  np.ones((3, 3), np.uint8))
        # A wall column with no points at all (scan never reached it) must
        # not read as "hole": require some support in each along-column.
        col_support = raster.sum(axis=0) > 0

        empty_orig = raster == 0
        # Merge panes across mullions: a spröjs is a 5–15 cm filled bar
        # inside one window — close the EMPTY mask over that width so a
        # mullioned window is ONE opening, while real piers (≥ 0.4 m)
        # between adjacent windows survive. Closing only GROUPS panes;
        # all size/border tests below run on the component's original
        # empty cells, since the dilation step of the closing inflates
        # holes into the ceiling row and got real windows rejected.
        mull = max(3, int(round(0.18 / pixel)) | 1)
        empty = cv2.morphologyEx(empty_orig.astype(np.uint8), cv2.MORPH_CLOSE,
                                 np.ones((mull, mull), np.uint8))
        n_lab, lab_img, _stats, _ = cv2.connectedComponentsWithStats(empty, 8)
        for k in range(1, n_lab):
            sub_z, sub_a = np.nonzero((lab_img == k) & empty_orig)
            if len(sub_z) == 0:
                continue
            x = int(sub_a.min()); ww = int(sub_a.max()) - x + 1
            y = int(sub_z.min()); hh = int(sub_z.max()) - y + 1
            area = len(sub_z)
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
            # Fill is judged on the CLOSED component: glass speckle and
            # mullions inside the hole are bridged by the closing, so a
            # real (mullioned) window fills its bbox even on a diluted
            # scan where speckle cells read as wall. Size and border
            # tests above still use the original cells.
            fill = min(1.0, float((lab_img == k).sum()) / max(1, ww * hh))
            if fill < OPENING_MIN_FILL:
                continue
            z0 = z_base + y * pixel
            z1 = z0 + height_m
            # "Reaches the floor" = hole bottom within ~35 cm of the slab
            # (raster bottom already sits FLOOR_SKIP up, and thresholds,
            # skirting boards and floor spill blur the lowest rows).
            touches_floor = z0 <= wall.z_placement + 0.35
            kind = None
            if touches_floor and (z1 - wall.z_placement) >= cfg.door_min_height \
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
                # Opening contract: Z relative to the wall bottom.
                z_min=0.0 if kind == "door" else float(z0 - wall.z_placement),
                z_max=float(z1 - wall.z_placement),
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
