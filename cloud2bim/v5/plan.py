"""v5 plan detector — walls as room boundaries on global carriers.

Per-wall detection can never guarantee exact corners, no overlaps and
no fragmentation: those are relations BETWEEN walls. v5 therefore
models the whole storey plan globally (docs/v5-design.md):

    1. Evidence cells (v4's persistence raster — reused).
    2. Dominant building rotation → work in the rotated frame.
    3. CARRIERS: global axis-aligned lines found by accumulating
       evidence perpendicular offsets. A carrier is unbounded — wall
       fragments do not exist at this level. Diagonal leftovers get
       their own carriers via line peeling.
    4. REGION GRID: free space partitioned by carrier corridors;
       connected components classified room / outside / unknown from
       floor coverage and border contact.
    5. WALLS = carrier intervals whose two sides lie in different
       regions (room|room, room|outside, room|unknown). Same region on
       both sides ⇒ not a wall — this is what dismisses archive
       racking, which per-segment evidence can never do.
    6. Corners: wall interval endpoints snap to crossing carriers —
       intersections of global lines, shared exactly by both walls.
    7. Thickness via v4's modal face analysis on raw evidence cells.

No silent fallbacks: if a stage produces nothing, the result is empty
and the log says why.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from cloud2bim.config import SegmentationConfig, WallConfig
from cloud2bim.elements.walls import Wall
from cloud2bim.elements.v4 import wall_evidence, _refine_thickness, PIXEL
from cloud2bim.logging import get_logger
from cloud2bim.segmentation.base import SemanticLabels

log = get_logger(__name__)

# Carrier accumulation
OFFSET_BIN = 0.03          # m — perpendicular histogram bin (= evidence pixel)
CARRIER_MIN_RUN = 1.2      # m — a carrier needs this much covered length
CARRIER_CLAIM = 0.13       # m — cells within this of a carrier belong to it
CARRIER_GAP = 0.45         # m — gap that splits a carrier's covered intervals
# Region grid
REGION_PIX = 0.06          # m — free-space partition resolution
MIN_ROOM_AREA = 1.0        # m² — smaller free components are voids
FLOOR_COV_ROOM = 0.12      # share of region cells with floor points
# Wall extraction along carriers
STATION_STEP = 0.10        # m — sampling step along a carrier
SIDE_PROBE = (0.20, 0.35, 0.55)   # m — perpendicular probe distances
WALL_MIN_LEN = 0.35        # m — shorter boundary runs are noise
STATION_GAP = 0.45         # m — station gap that splits a wall interval


def detect_walls_v5(
    storey_points: np.ndarray,
    z_floor: float,
    z_ceiling: float,
    storey_idx: int,
    cfg: WallConfig,
    semantic_labels: Optional[SemanticLabels] = None,
    seg_cfg: Optional[SegmentationConfig] = None,
    **_unused,
) -> List[Wall]:
    h = z_ceiling - z_floor
    if len(storey_points) == 0 or h <= 0.2:
        return []
    ev = wall_evidence(storey_points, z_floor, z_ceiling,
                       semantic_labels=semantic_labels, seg_cfg=seg_cfg)
    if ev is None:
        log.warning("v5 storey %d: no wall evidence", storey_idx)
        return []
    cells_raw, cells_xy, in_storey = ev

    # ── 2. dominant rotation ──
    theta = _dominant_rotation(cells_xy)
    rot = np.array([[np.cos(-theta), -np.sin(-theta)],
                    [np.sin(-theta), np.cos(-theta)]])
    q = cells_xy @ rot.T          # rotated evidence cells
    q_raw = cells_raw @ rot.T

    # floor coverage points (slab surface band) in rotated frame
    z = storey_points[:, 2]
    floor_pts = storey_points[(z >= z_floor - 0.05) & (z <= z_floor + 0.20)]
    qf = floor_pts[:, :2] @ rot.T if len(floor_pts) else np.empty((0, 2))

    # ── 3. carriers ──
    carriers = _find_carriers(q)
    log.info("v5 storey %d: rotation %.1f°, %d carriers (%d axis-0, %d axis-1, %d diag)",
             storey_idx, np.degrees(theta), len(carriers),
             sum(1 for c in carriers if c["axis"] == 0),
             sum(1 for c in carriers if c["axis"] == 1),
             sum(1 for c in carriers if c["axis"] == 2))
    if not carriers:
        return []

    # ── 4. region grid ──
    regions, grid0, label_img = _region_grid(q, qf, carriers)
    n_rooms = sum(1 for r in regions.values() if r == "room")
    log.info("v5 storey %d: %d regions (%d rooms)", storey_idx,
             len(regions), n_rooms)

    # ── 5+6. walls along carriers, corner-snapped ──
    axes_rot = _walls_on_carriers(carriers, regions, grid0, label_img, q)
    if not axes_rot:
        log.warning("v5 storey %d: no room boundaries found", storey_idx)
        return []

    # back to world frame
    inv = rot.T
    axes = [[(np.array(p1) @ inv.T).tolist(), (np.array(p2) @ inv.T).tolist()]
            for p1, p2 in axes_rot]

    axes, thicknesses = _refine_thickness(axes, cells_raw, cfg)
    axes, thicknesses = _dedupe_parallel(axes, thicknesses)
    walls = [
        Wall(start=tuple(ax[0]), end=tuple(ax[1]), thickness=t,
             z_placement=z_floor, height=h, storey=storey_idx,
             label="interior")
        for ax, t in zip(axes, thicknesses)
    ]
    log.info("v5 storey %d: %d walls", storey_idx, len(walls))
    return walls


# ── internals ────────────────────────────────────────────────────────────────


def _dedupe_parallel(axes: list, thicknesses: list[float]):
    """Drop near-coincident parallel walls (a thin wall's two faces each
    become a carrier, and both carriers emit the same room boundary).
    Keeps the longer wall of an overlapping pair; midline distance must
    be inside the kept wall's body."""
    order = sorted(range(len(axes)), key=lambda i: -float(
        np.hypot(axes[i][1][0] - axes[i][0][0], axes[i][1][1] - axes[i][0][1])))
    kept: list[int] = []
    for i in order:
        a1 = np.asarray(axes[i], float)
        d1 = a1[1] - a1[0]
        l1 = float(np.hypot(*d1))
        if l1 < 1e-6:
            continue
        u = d1 / l1
        n_vec = np.array([-u[1], u[0]])
        dup = False
        for j in kept:
            a2 = np.asarray(axes[j], float)
            d2 = a2[1] - a2[0]
            l2 = float(np.hypot(*d2))
            if abs(float(u @ d2)) / max(l2, 1e-9) < np.cos(np.deg2rad(6)):
                continue
            off = abs(float(((a2[0] - a1[0]) @ n_vec
                             + (a2[1] - a1[0]) @ n_vec) / 2))
            if off > max(thicknesses[j], thicknesses[i]) * 0.75 + 0.05:
                continue
            s = sorted([float((a2[0] - a1[0]) @ u), float((a2[1] - a1[0]) @ u)])
            overlap = min(s[1], l1) - max(s[0], 0.0)
            if overlap >= 0.5 * l1:
                dup = True
                break
        if not dup:
            kept.append(i)
    kept.sort()
    return [axes[i] for i in kept], [thicknesses[i] for i in kept]


def _dominant_rotation(cells_xy: np.ndarray) -> float:
    """Building rotation from evidence-cell pair directions (mod 90°).

    Random nearby cell pairs mostly lie along walls; their angles mod
    90° cluster at the building rotation. Median of the cluster is
    robust against the diagonal minority.
    """
    n = len(cells_xy)
    if n < 50:
        return 0.0
    rng = np.random.default_rng(1)
    i = rng.integers(0, n, 6000)
    j = rng.integers(0, n, 6000)
    d = cells_xy[j] - cells_xy[i]
    dist = np.hypot(d[:, 0], d[:, 1])
    ok = (dist > 0.5) & (dist < 6.0)
    ang = np.arctan2(d[ok, 1], d[ok, 0]) % (np.pi / 2)
    # circular median via histogram peak + local mean
    hist, edges = np.histogram(ang, bins=90, range=(0, np.pi / 2))
    k = int(np.argmax(hist))
    centre = (edges[k] + edges[k + 1]) / 2
    near = np.abs((ang - centre + np.pi / 4) % (np.pi / 2) - np.pi / 4) < np.deg2rad(3)
    if near.any():
        centre = float(np.mean(ang[near]))
    return centre


def _find_carriers(q: np.ndarray) -> list[dict]:
    """Axis-aligned carriers by offset accumulation + diagonal leftovers.

    Carrier dict: axis 0 = runs along x (constant y), axis 1 = runs
    along y (constant x), axis 2 = free line {p, u}. ``intervals`` are
    covered [lo, hi] ranges along the run direction.
    """
    claimed = np.zeros(len(q), bool)
    carriers: list[dict] = []
    for axis in (0, 1):
        run = q[:, axis]          # coordinate along the wall
        off = q[:, 1 - axis]      # perpendicular offset
        lo, hi = float(off.min()), float(off.max())
        nbin = max(4, int((hi - lo) / OFFSET_BIN) + 1)
        hist, edges = np.histogram(off, bins=nbin, range=(lo, lo + nbin * OFFSET_BIN))
        # local peaks with a 2-bin guard on each side
        for k in range(len(hist)):
            if hist[k] < CARRIER_MIN_RUN / PIXEL:
                continue
            l_ = max(0, k - 2)
            r_ = min(len(hist), k + 3)
            if hist[k] < hist[l_:r_].max():
                continue
            o0 = edges[k] - OFFSET_BIN
            o1 = edges[k + 1] + OFFSET_BIN
            members = (off >= o0) & (off <= o1)
            intervals = _covered_intervals(np.sort(run[members]))
            if not intervals:
                continue
            offset = float(np.median(off[members]))
            carriers.append({"axis": axis, "offset": offset,
                             "intervals": intervals})
            claimed |= members & (np.abs(off - offset) <= CARRIER_CLAIM)

    # Diagonal leftovers → peeled line carriers
    rest = q[~claimed]
    if len(rest) >= 40:
        from cloud2bim.extraction.walls_ml import _extract_line_segments, _dbscan_xy
        from cloud2bim.config import WallConfig
        rng = np.random.default_rng(2)
        for cl in _dbscan_xy(rest, 0.35, 8):
            for ax, _t in _extract_line_segments(cl, WallConfig(), rng):
                p1 = np.array(ax[0])
                p2 = np.array(ax[1])
                L = float(np.hypot(*(p2 - p1)))
                if L < CARRIER_MIN_RUN:
                    continue
                u = (p2 - p1) / L
                if min(abs(u[0]), abs(u[1])) < 0.10:
                    continue  # near-axis lines already handled above
                carriers.append({"axis": 2, "p": p1, "u": u,
                                 "intervals": [(0.0, L)]})
    return carriers


def _covered_intervals(sorted_run: np.ndarray) -> list[tuple[float, float]]:
    """[lo, hi] runs of coverage, split at gaps, short runs dropped."""
    if len(sorted_run) < 2:
        return []
    breaks = np.where(np.diff(sorted_run) > CARRIER_GAP)[0] + 1
    out = []
    start = 0
    for stop in list(breaks) + [len(sorted_run)]:
        seg = sorted_run[start:stop]
        start = stop
        if len(seg) >= 8 and seg[-1] - seg[0] >= CARRIER_MIN_RUN:
            out.append((float(seg[0]), float(seg[-1])))
    return out


def _region_grid(q: np.ndarray, qf: np.ndarray, carriers: list[dict]):
    """Partition free space by carrier corridors; classify regions.

    Returns (regions: label_id -> 'room'|'outside'|'void', grid_origin,
    label_img). Carrier corridors are drawn only where the carrier has
    evidence coverage (its intervals, slightly extended) so an
    unbounded line doesn't slice rooms it never touches.
    """
    from scipy import ndimage

    # The margin must survive the door-closing erosion (0.55 m), or the
    # outside ring loses its seed and the watershed floods the whole
    # grid from one interior room — one region, zero walls.
    g0 = q.min(axis=0) - 1.2
    g1 = q.max(axis=0) + 1.2
    nx = int(np.ceil((g1[0] - g0[0]) / REGION_PIX)) + 1
    ny = int(np.ceil((g1[1] - g0[1]) / REGION_PIX)) + 1
    blocked = np.zeros((nx, ny), bool)

    for c in carriers:
        for lo, hi in c["intervals"]:
            # Seal corners: extend the corridor to crossing carriers'
            # intersections near the interval ends, else inside and
            # outside leak together around the corner and the whole
            # plan collapses into one region (= zero walls).
            for other in carriers:
                if other is c:
                    continue
                ti = _intersection_param(c, other)
                if ti is None:
                    continue
                if lo - 0.7 <= ti < lo:
                    lo = ti
                elif hi < ti <= hi + 0.7:
                    hi = ti
            lo -= 0.25
            hi += 0.25
            n = max(2, int((hi - lo) / (REGION_PIX * 0.5)))
            t = np.linspace(lo, hi, n)
            pts = _carrier_points(c, t)
            ii = np.clip(((pts[:, 0] - g0[0]) / REGION_PIX).astype(int), 0, nx - 1)
            jj = np.clip(((pts[:, 1] - g0[1]) / REGION_PIX).astype(int), 0, ny - 1)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    blocked[np.clip(ii + di, 0, nx - 1),
                            np.clip(jj + dj, 0, ny - 1)] = True

    # Rooms connected through door openings are ONE connected component,
    # and a pure connectivity labelling would then erase the wall between
    # them (same region on both sides). Erode free space until door-width
    # passages close, label the eroded seeds, and assign every free cell
    # to its nearest seed. Over-segmentation in open space is harmless:
    # a region boundary only becomes a wall where a carrier with actual
    # wall evidence lies on it.
    free = ~blocked
    erode_iters = max(1, int(0.55 / REGION_PIX))
    seeds_mask = ndimage.binary_erosion(free, iterations=erode_iters)
    seeds, n_lab = ndimage.label(seeds_mask)
    if n_lab == 0:
        label_img = ndimage.label(free)[0]
        n_lab = int(label_img.max())
    else:
        _dist, (ii, jj) = ndimage.distance_transform_edt(
            seeds == 0, return_indices=True)
        label_img = seeds[ii, jj]
        label_img[~free] = 0
    floor_grid = np.zeros((nx, ny), bool)
    if len(qf):
        fi = np.clip(((qf[:, 0] - g0[0]) / REGION_PIX).astype(int), 0, nx - 1)
        fj = np.clip(((qf[:, 1] - g0[1]) / REGION_PIX).astype(int), 0, ny - 1)
        floor_grid[fi, fj] = True

    regions: dict[int, str] = {}
    border = np.zeros((nx, ny), bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    for lab in range(1, n_lab + 1):
        m = label_img == lab
        area = int(m.sum()) * REGION_PIX * REGION_PIX
        cov = float(floor_grid[m].mean()) if m.any() else 0.0
        touches = bool((m & border).any())
        if area < MIN_ROOM_AREA:
            regions[lab] = "void"
        elif cov >= FLOOR_COV_ROOM:
            regions[lab] = "room"
        elif touches:
            regions[lab] = "outside"
        else:
            regions[lab] = "void"
    return regions, g0, label_img


def _carrier_points(c: dict, t: np.ndarray) -> np.ndarray:
    if c["axis"] == 0:
        return np.column_stack([t, np.full(len(t), c["offset"])])
    if c["axis"] == 1:
        return np.column_stack([np.full(len(t), c["offset"]), t])
    return c["p"][None, :] + t[:, None] * c["u"][None, :]


def _carrier_normal(c: dict) -> np.ndarray:
    if c["axis"] == 0:
        return np.array([0.0, 1.0])
    if c["axis"] == 1:
        return np.array([1.0, 0.0])
    return np.array([-c["u"][1], c["u"][0]])


def _walls_on_carriers(carriers, regions, g0, label_img, q_cells):
    """Carrier intervals whose sides lie in different regions → walls.

    Station sampling: at each station probe both sides at increasing
    distances until a non-blocked region cell answers. A station is
    wall-evidence when the sides answer with different regions and at
    least one side is a room. Same room both sides ⇒ furniture ridge —
    dropped (the rack rule). Contiguous stations form wall intervals;
    endpoints snap to crossing carriers.
    """
    nx, ny = label_img.shape

    def region_at(pt):
        i = int((pt[0] - g0[0]) / REGION_PIX)
        j = int((pt[1] - g0[1]) / REGION_PIX)
        if not (0 <= i < nx and 0 <= j < ny):
            return "outside", -1
        lab = label_img[i, j]
        if lab == 0:
            return None, 0          # inside a wall corridor
        return regions.get(lab, "void"), lab

    axes_out: list = []
    for c in carriers:
        n_vec = _carrier_normal(c)
        for lo, hi in c["intervals"]:
            t = np.arange(lo, hi + STATION_STEP, STATION_STEP)
            if len(t) < 2:
                continue
            pts = _carrier_points(c, t)
            good = np.zeros(len(t), bool)
            for k in range(len(t)):
                sides = []
                for sgn in (1.0, -1.0):
                    ans = None
                    for d in SIDE_PROBE:
                        r, lab = region_at(pts[k] + sgn * d * n_vec)
                        if r is not None:
                            ans = (r, lab)
                            break
                    sides.append(ans)
                a, b = sides
                if a is None or b is None:
                    continue
                if a[1] == b[1]:
                    continue        # same region both sides → rack rule
                kinds = {a[0], b[0]}
                if "room" not in kinds:
                    continue        # outside|void boundary isn't a wall
                if kinds == {"room", "void"}:
                    # room against a small void (shaft, unscanned closet)
                    # is still a wall
                    pass
                good[k] = True
            # contiguous good stations → intervals
            idx = np.where(good)[0]
            if len(idx) == 0:
                continue
            splits = np.where(np.diff(t[idx]) > STATION_GAP)[0] + 1
            for grp in np.split(idx, splits):
                if len(grp) < 2:
                    continue
                t0, t1 = float(t[grp[0]]), float(t[grp[-1]])
                if t1 - t0 < WALL_MIN_LEN:
                    continue
                t0, t1 = _snap_ends(c, t0, t1, carriers)
                p = _carrier_points(c, np.array([t0, t1]))
                axes_out.append((p[0].tolist(), p[1].tolist()))
    return axes_out


def _snap_ends(c, t0, t1, carriers, snap=0.45):
    """Snap interval ends to crossing carriers' intersection parameters."""
    for tt, setter in ((t0, 0), (t1, 1)):
        best = None
        for other in carriers:
            if other is c:
                continue
            ti = _intersection_param(c, other)
            if ti is None:
                continue
            d = abs(ti - tt)
            if d <= snap and (best is None or d < abs(best - tt)):
                best = ti
        if best is not None:
            if setter == 0:
                t0 = best
            else:
                t1 = best
    if t1 <= t0:
        t1 = t0 + 0.01
    return t0, t1


def _intersection_param(c, other) -> float | None:
    """Parameter along carrier ``c`` where it crosses ``other``."""
    if c["axis"] == 0:
        p = np.array([0.0, c["offset"]])
        u = np.array([1.0, 0.0])
    elif c["axis"] == 1:
        p = np.array([c["offset"], 0.0])
        u = np.array([0.0, 1.0])
    else:
        p, u = c["p"], c["u"]
    if other["axis"] == 0:
        q0 = np.array([0.0, other["offset"]])
        v = np.array([1.0, 0.0])
    elif other["axis"] == 1:
        q0 = np.array([other["offset"], 0.0])
        v = np.array([0.0, 1.0])
    else:
        q0, v = other["p"], other["u"]
    denom = u[0] * v[1] - u[1] * v[0]
    if abs(denom) < 1e-6:
        return None
    w = q0 - p
    return float((w[0] * v[1] - w[1] * v[0]) / denom)
