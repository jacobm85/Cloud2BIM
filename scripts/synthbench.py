"""Synthetic scan-to-BIM benchmark.

Generates point clouds of known buildings (walls, slabs, openings,
columns, furniture clutter, scanner noise) with *perfect* per-point
S3DIS labels, runs the detection algorithms on them, and scores the
output against ground truth. Oracle labels isolate the extraction
algorithms from segmentation-model quality — a detector that fails
here fails everywhere.

Usage:
    python scripts/synthbench.py            # all scenarios
    python scripts/synthbench.py kontor     # one scenario
    python scripts/synthbench.py kontor30   # rotated variant

Scenarios model the building types Cloud2BIM is used on:
    kontor    — cellkontor + korridor, fönsterband, möbler
    bostad    — små rum, tunna gipsväggar, mycket möbler
    vard      — lång korridor, många dörrar
    industri  — stor hall, pelarrutnät, högt i tak, ställage
    garage    — lågt i tak, pelare, bilar som klutter
    kontor30  — kontor roterat 30° (testar rotationsoberoende)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from cloud2bim.config import Config, IOConfig  # noqa: E402
from cloud2bim.extraction import (  # noqa: E402
    extract_openings_ml, extract_slabs_ml, extract_walls_ml,
)
from cloud2bim.segmentation.base import S3DIS_LABELS, SemanticLabels  # noqa: E402

LBL = {name: i for i, name in enumerate(S3DIS_LABELS)}
RNG = np.random.default_rng(42)

DENSITY = 400          # points / m² on surfaces (≈ 5 cm scan diluted)
NOISE_SIGMA = 0.005    # m — scanner noise on surfaces
OUTLIER_FRACTION = 0.003


# ── point cloud builder ──────────────────────────────────────────────────────


class Build:
    """Accumulates labelled point chunks + ground truth."""

    def __init__(self):
        self.chunks: list[np.ndarray] = []
        self.labels: list[np.ndarray] = []
        self.gt_walls: list[dict] = []      # axis [[x,y],[x,y]], thickness
        self.gt_slab_levels: list[float] = []   # walkable floor tops
        self.gt_openings: list[dict] = []   # wall_gt_idx, kind, along0, along1
        self.gt_columns: list[tuple] = []   # (cx, cy)

    def add(self, pts: np.ndarray, label: str):
        if len(pts):
            self.chunks.append(pts)
            self.labels.append(np.full(len(pts), LBL[label], dtype=np.int32))

    def finish(self, rotation_deg: float = 0.0):
        xyz = np.vstack(self.chunks)
        ids = np.concatenate(self.labels)
        # Scanner outliers, uniformly in an inflated bbox
        n_out = int(len(xyz) * OUTLIER_FRACTION)
        lo, hi = xyz.min(axis=0) - 1, xyz.max(axis=0) + 1
        outl = RNG.uniform(lo, hi, size=(n_out, 3))
        xyz = np.vstack([xyz, outl])
        ids = np.concatenate([ids, np.full(n_out, LBL["clutter"], dtype=np.int32)])
        if rotation_deg:
            a = np.deg2rad(rotation_deg)
            rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            xyz = xyz.copy()
            xyz[:, :2] = xyz[:, :2] @ rot.T
            for w in self.gt_walls:
                w["axis"] = (np.asarray(w["axis"]) @ rot.T).tolist()
            self.gt_columns = [tuple(rot @ np.array(c)) for c in self.gt_columns]
        order = RNG.permutation(len(xyz))
        return xyz[order], SemanticLabels(ids[order], S3DIS_LABELS)


def _rect_surface(origin, u_vec, v_vec, density=DENSITY) -> np.ndarray:
    """Jittered grid of points on the parallelogram origin + s·u + t·v."""
    area = np.linalg.norm(np.cross(u_vec, v_vec))
    n = max(8, int(area * density))
    s = RNG.uniform(0, 1, n)
    t = RNG.uniform(0, 1, n)
    pts = (np.asarray(origin)[None, :]
           + s[:, None] * np.asarray(u_vec)[None, :]
           + t[:, None] * np.asarray(v_vec)[None, :])
    return pts + RNG.normal(0, NOISE_SIGMA, pts.shape)


def add_wall(b: Build, p1, p2, z0, z1, thickness,
             openings=(), two_sided=True, label="wall"):
    """Two vertical faces ± thickness/2 around the axis, with holes.

    ``openings``: (kind, along0, along1, zlo, zhi). Hole regions are cut
    from the wall faces; door/window points are emitted on the axis plane
    at lower density (frame + glass returns).
    """
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    d = p2 - p1
    length = float(np.hypot(*d))
    u = d / length
    n_vec = np.array([-u[1], u[0]])
    gt_idx = len(b.gt_walls)
    b.gt_walls.append({"axis": [p1.tolist(), p2.tolist()], "thickness": thickness})

    offsets = [thickness / 2, -thickness / 2] if two_sided else [thickness / 2]
    for off in offsets:
        o3 = np.array([*(p1 + off * n_vec), z0])
        pts = _rect_surface(o3, [*(u * length), 0], [0, 0, z1 - z0])
        # carve opening holes
        rel = pts[:, :2] - p1[None, :]
        along = rel @ u
        keep = np.ones(len(pts), bool)
        for kind, a0, a1, zlo, zhi in openings:
            keep &= ~((along > a0) & (along < a1) & (pts[:, 2] > zlo) & (pts[:, 2] < zhi))
        b.add(pts[keep], label)

    for kind, a0, a1, zlo, zhi in openings:
        o3 = np.array([*(p1 + a0 * u), zlo])
        pts = _rect_surface(o3, [*(u * (a1 - a0)), 0], [0, 0, zhi - zlo],
                            density=DENSITY // 6)
        b.add(pts, kind)
        b.gt_openings.append({"wall": gt_idx, "kind": kind,
                              "along0": a0, "along1": a1})


def add_slab_pair(b: Build, x0, y0, x1, y1, z_top, thickness):
    """Floor top at z_top and the matching ceiling underside below it."""
    b.add(_rect_surface([x0, y0, z_top], [x1 - x0, 0, 0], [0, y1 - y0, 0]), "floor")
    b.add(_rect_surface([x0, y0, z_top - thickness],
                        [x1 - x0, 0, 0], [0, y1 - y0, 0]), "ceiling")


def add_floor(b: Build, x0, y0, x1, y1, z_top):
    b.add(_rect_surface([x0, y0, z_top], [x1 - x0, 0, 0], [0, y1 - y0, 0]), "floor")
    b.gt_slab_levels.append(z_top)


def add_ceiling(b: Build, x0, y0, x1, y1, z_under):
    b.add(_rect_surface([x0, y0, z_under], [x1 - x0, 0, 0], [0, y1 - y0, 0]), "ceiling")


def add_column(b: Build, cx, cy, size, z0, z1):
    h = size / 2
    for sx, sy, ux, uy in ((-h, -h, 1, 0), (-h, h, 1, 0), (-h, -h, 0, 1), (h, -h, 0, 1)):
        o3 = np.array([cx + sx, cy + sy, z0])
        b.add(_rect_surface(o3, [ux * size, uy * size, 0], [0, 0, z1 - z0]), "column")
    b.gt_columns.append((cx, cy))


def add_clutter(b: Build, x0, y0, x1, y1, z_floor, n_items, max_h=1.2):
    for _ in range(n_items):
        cx, cy = RNG.uniform(x0 + 1, x1 - 1), RNG.uniform(y0 + 1, y1 - 1)
        w, d_, h = RNG.uniform(0.4, 1.8), RNG.uniform(0.4, 1.2), RNG.uniform(0.4, max_h)
        lbl = RNG.choice(["table", "chair", "bookcase", "sofa", "clutter"])
        # top + two sides is enough to look like furniture
        b.add(_rect_surface([cx, cy, z_floor + h], [w, 0, 0], [0, d_, 0],
                            density=DENSITY // 2), lbl)
        b.add(_rect_surface([cx, cy, z_floor], [w, 0, 0], [0, 0, h],
                            density=DENSITY // 2), lbl)
        b.add(_rect_surface([cx, cy, z_floor], [0, d_, 0], [0, 0, h],
                            density=DENSITY // 2), lbl)


# ── scenarios ────────────────────────────────────────────────────────────────


def scen_kontor(rotation=0.0):
    """18×12 m cellkontor: korridor, 4 rum, fönsterband, dörrar, möbler."""
    b = Build()
    H = 2.7
    add_floor(b, 0, 0, 18, 12, 0.0)
    add_ceiling(b, 0, 0, 18, 12, H)
    win = [("window", a, a + 1.6, 0.9, 2.3) for a in (1.2, 4.2, 7.2, 10.2, 13.2, 16.2)]
    add_wall(b, (0, 0), (18, 0), 0, H, 0.30, openings=win, two_sided=False)
    add_wall(b, (18, 0), (18, 12), 0, H, 0.30, two_sided=False)
    add_wall(b, (18, 12), (0, 12), 0, H, 0.30,
             openings=[("window", a, a + 1.6, 0.9, 2.3) for a in (2.2, 7.2, 12.2)],
             two_sided=False)
    add_wall(b, (0, 12), (0, 0), 0, H, 0.30, two_sided=False)
    # corridor walls at y=5 and y=7, doors into rooms
    add_wall(b, (0, 5), (18, 5), 0, H, 0.12,
             openings=[("door", a, a + 0.9, 0, 2.1) for a in (2.0, 6.5, 11.0, 15.5)])
    add_wall(b, (0, 7), (18, 7), 0, H, 0.12,
             openings=[("door", a, a + 0.9, 0, 2.1) for a in (3.0, 9.0, 14.0)])
    # room dividers
    for x in (4.5, 9.0, 13.5):
        add_wall(b, (x, 0), (x, 5), 0, H, 0.12)
        add_wall(b, (x, 7), (x, 12), 0, H, 0.12)
    add_clutter(b, 0, 0, 18, 5, 0, 14)
    add_clutter(b, 0, 7, 18, 12, 0, 10)
    return b.finish(rotation)


def scen_bostad(rotation=0.0):
    """10×8 m lägenhet: tunna gipsväggar, mycket möbler."""
    b = Build()
    H = 2.5
    add_floor(b, 0, 0, 10, 8, 0.0)
    add_ceiling(b, 0, 0, 10, 8, H)
    add_wall(b, (0, 0), (10, 0), 0, H, 0.40,
             openings=[("window", 2.0, 3.4, 0.8, 2.2), ("window", 6.5, 7.9, 0.8, 2.2)],
             two_sided=False)
    add_wall(b, (10, 0), (10, 8), 0, H, 0.40,
             openings=[("window", 2.5, 3.7, 0.8, 2.2)], two_sided=False)
    add_wall(b, (10, 8), (0, 8), 0, H, 0.40, two_sided=False)
    add_wall(b, (0, 8), (0, 0), 0, H, 0.40,
             openings=[("door", 3.4, 4.4, 0, 2.1)], two_sided=False)
    add_wall(b, (4, 0), (4, 4.5), 0, H, 0.10, openings=[("door", 3.2, 4.1, 0, 2.1)])
    add_wall(b, (4, 4.5), (10, 4.5), 0, H, 0.10,
             openings=[("door", 1.0, 1.9, 0, 2.1), ("door", 4.0, 4.9, 0, 2.1)])
    add_wall(b, (7, 4.5), (7, 8), 0, H, 0.10)
    add_clutter(b, 0, 0, 10, 8, 0, 16)
    return b.finish(rotation)


def scen_vard(rotation=0.0):
    """30×11 m vårdavdelning: lång korridor, rum på båda sidor, många dörrar."""
    b = Build()
    H = 2.9
    add_floor(b, 0, 0, 30, 11, 0.0)
    add_ceiling(b, 0, 0, 30, 11, H)
    for p1, p2 in (((0, 0), (30, 0)), ((30, 0), (30, 11)),
                   ((30, 11), (0, 11)), ((0, 11), (0, 0))):
        add_wall(b, p1, p2, 0, H, 0.30, two_sided=False)
    doors_a = [("door", a, a + 1.2, 0, 2.1) for a in np.arange(1.5, 29, 4.0)]
    doors_b = [("door", a, a + 1.2, 0, 2.1) for a in np.arange(3.0, 29, 4.0)]
    add_wall(b, (0, 4), (30, 4), 0, H, 0.15, openings=doors_a)
    add_wall(b, (0, 7), (30, 7), 0, H, 0.15, openings=doors_b)
    for x in np.arange(4.0, 30, 4.0):
        add_wall(b, (x, 0), (x, 4), 0, H, 0.12)
        add_wall(b, (x, 7), (x, 11), 0, H, 0.12)
    add_clutter(b, 0, 0, 30, 4, 0, 18)
    add_clutter(b, 0, 7, 30, 11, 0, 18)
    return b.finish(rotation)


def scen_industri(rotation=0.0):
    """40×25 m hall: pelarrutnät 8×8, kontorshörna, ställage, 6 m i tak."""
    b = Build()
    H = 6.0
    add_floor(b, 0, 0, 40, 25, 0.0)
    add_ceiling(b, 0, 0, 40, 25, H)
    add_wall(b, (0, 0), (40, 0), 0, H, 0.40,
             openings=[("window", a, a + 3.0, 3.5, 5.0) for a in (4, 14, 24, 34)],
             two_sided=False)
    add_wall(b, (40, 0), (40, 25), 0, H, 0.40,
             openings=[("door", 10, 14, 0, 4.5)], two_sided=False)  # port
    add_wall(b, (40, 25), (0, 25), 0, H, 0.40, two_sided=False)
    add_wall(b, (0, 25), (0, 0), 0, H, 0.40, two_sided=False)
    # office corner
    add_wall(b, (0, 18), (10, 18), 0, 3.0, 0.15, openings=[("door", 4, 5, 0, 2.1)])
    add_wall(b, (10, 18), (10, 25), 0, 3.0, 0.15)
    for cx in np.arange(8.0, 40, 8.0):
        for cy in np.arange(8.0, 25, 8.0):
            add_column(b, cx, cy, 0.45, 0, H)
    # pallställage = stora "bookcase"-block
    for x in (16, 24, 32):
        b.add(_rect_surface([x, 4, 0], [1.2, 0, 0], [0, 0, 4.0], DENSITY // 2), "bookcase")
        b.add(_rect_surface([x, 4, 0], [0, 12, 0], [0, 0, 4.0], DENSITY // 2), "bookcase")
    return b.finish(rotation)


def scen_garage(rotation=0.0):
    """30×20 m parkeringsgarage: 2.4 m i tak, pelare, bilar som klutter."""
    b = Build()
    H = 2.4
    add_floor(b, 0, 0, 30, 20, 0.0)
    add_ceiling(b, 0, 0, 30, 20, H)
    add_wall(b, (0, 0), (30, 0), 0, H, 0.30, two_sided=False)
    add_wall(b, (30, 0), (30, 20), 0, H, 0.30,
             openings=[("door", 8, 11, 0, 2.2)], two_sided=False)  # port
    add_wall(b, (30, 20), (0, 20), 0, H, 0.30, two_sided=False)
    add_wall(b, (0, 20), (0, 0), 0, H, 0.30, two_sided=False)
    for cx in np.arange(7.5, 30, 7.5):
        for cy in np.arange(6.5, 20, 6.5):
            add_column(b, cx, cy, 0.40, 0, H)
    # cars: ~1.8×4.5×1.5 boxes
    for _ in range(10):
        cx, cy = RNG.uniform(2, 24), RNG.uniform(2, 16)
        b.add(_rect_surface([cx, cy, 1.5], [4.5, 0, 0], [0, 1.8, 0], DENSITY // 2), "clutter")
        b.add(_rect_surface([cx, cy, 0.3], [4.5, 0, 0], [0, 0, 1.2], DENSITY // 2), "clutter")
        b.add(_rect_surface([cx, cy + 1.8, 0.3], [4.5, 0, 0], [0, 0, 1.2], DENSITY // 2), "clutter")
    return b.finish(rotation)


SCENARIOS = {
    "kontor": scen_kontor,
    "bostad": scen_bostad,
    "vard": scen_vard,
    "industri": scen_industri,
    "garage": scen_garage,
    "kontor30": lambda: scen_kontor(rotation=30.0),
}


# ── metrics ──────────────────────────────────────────────────────────────────


def _seg_geom(ax):
    p1, p2 = np.asarray(ax[0], float), np.asarray(ax[1], float)
    d = p2 - p1
    length = float(np.hypot(*d))
    u = d / max(length, 1e-9)
    return p1, u, length


def _wall_match(gt, det) -> tuple[bool, float]:
    """(matches?, axis_distance). Angle<10°, midline dist<0.30, overlap≥50%."""
    g1, gu, gl = _seg_geom(gt["axis"])
    d1, du, dl = _seg_geom([det.start, det.end])
    cosang = abs(float(gu @ du))
    if cosang < np.cos(np.deg2rad(10)):
        return False, 0.0
    # overlap of det projected on gt axis
    t1 = float((np.asarray(det.start) - g1) @ gu)
    t2 = float((np.asarray(det.end) - g1) @ gu)
    lo, hi = min(t1, t2), max(t1, t2)
    overlap = min(hi, gl) - max(lo, 0.0)
    if overlap < 0.5 * min(gl, dl):
        return False, 0.0
    n_vec = np.array([-gu[1], gu[0]])
    mid = (np.asarray(det.start) + np.asarray(det.end)) / 2
    dist = abs(float((mid - g1) @ n_vec))
    if dist > 0.30:
        return False, 0.0
    return True, dist


def score_walls(gt_walls, det_walls):
    matched_gt, matched_det, dists = set(), set(), []
    for gi, gt in enumerate(gt_walls):
        for di, det in enumerate(det_walls):
            if di in matched_det:
                continue
            ok, dist = _wall_match(gt, det)
            if ok:
                matched_gt.add(gi)
                matched_det.add(di)
                dists.append(dist)
                break
    rec = len(matched_gt) / max(1, len(gt_walls))
    prec = len(matched_det) / max(1, len(det_walls))
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return prec, rec, f1, (float(np.mean(dists)) if dists else float("nan"))


def score_openings(gt_openings, gt_walls, det_openings, det_walls):
    matched_det = set()
    hit = 0
    for gt in gt_openings:
        gw = gt_walls[gt["wall"]]
        g1, gu, _ = _seg_geom(gw["axis"])
        for di, op in enumerate(det_openings):
            if di in matched_det or op.type != gt["kind"]:
                continue
            host = det_walls[op.wall_index]
            ok, _ = _wall_match(gw, host)
            if not ok:
                continue
            h1 = np.asarray(host.start, float)
            hu = (np.asarray(host.end, float) - h1)
            hu /= max(np.hypot(*hu), 1e-9)
            # opening interval in gt-axis coords
            a0 = float((h1 + op.x_along_wall_start * hu - g1) @ gu)
            a1 = float((h1 + op.x_along_wall_end * hu - g1) @ gu)
            lo, hi = min(a0, a1), max(a0, a1)
            inter = min(hi, gt["along1"]) - max(lo, gt["along0"])
            union = max(hi, gt["along1"]) - min(lo, gt["along0"])
            if union > 0 and inter / union > 0.3:
                matched_det.add(di)
                hit += 1
                break
    rec = hit / max(1, len(gt_openings))
    prec = len(matched_det) / max(1, len(det_openings))
    return prec, rec


# ── runner ───────────────────────────────────────────────────────────────────


def main(names):
    for name in names:
        global RNG
        RNG = np.random.default_rng(42)
        t0 = time.time()
        fn = SCENARIOS[name]
        # capture the Build to access ground truth
        b_holder = {}
        orig_finish = Build.finish

        def capture_finish(self, rotation_deg=0.0):
            b_holder["b"] = self
            return orig_finish(self, rotation_deg)

        Build.finish = capture_finish
        try:
            xyz, labels = fn()
        finally:
            Build.finish = orig_finish
        b = b_holder["b"]
        cfg = Config(io=IOConfig(input_files=["x.xyz"], output_ifc="x.ifc"))
        print(f"\n=== {name}: {len(xyz):,} pts, {len(b.gt_walls)} GT walls, "
              f"{len(b.gt_openings)} GT openings ===")

        # ── slabs (ML path, oracle labels) ──
        slabs = extract_slabs_ml(xyz, labels, cfg.slabs, cfg.segmentation)
        print(f"slabs: detected {len(slabs)} (expect 2: floor + ceiling) "
              f"z={[round(s.bottom_z + s.thickness, 2) for s in slabs]}")
        if len(slabs) < 2:
            print("slabs: FAILED — cannot continue to walls")
            continue
        z_floor = slabs[0].bottom_z + slabs[0].thickness
        z_ceiling = slabs[1].bottom_z
        storey_mask = (xyz[:, 2] >= z_floor - 0.1) & (xyz[:, 2] <= z_ceiling + 0.1)
        spts = xyz[storey_mask]
        slab_poly = np.column_stack([slabs[1].polygon_x, slabs[1].polygon_y])
        slabels = SemanticLabels(labels.label_ids[storey_mask], labels.label_names)

        # ── walls: ML path ──
        t1 = time.time()
        walls = extract_walls_ml(spts, slabels, z_floor, z_ceiling, 0,
                                 cfg.walls, cfg.segmentation, slab_poly)
        p, r, f1, err = score_walls(b.gt_walls, walls)
        print(f"walls[ml]      P={p:.2f} R={r:.2f} F1={f1:.2f} "
              f"axis_err={err * 100:.1f}cm  n={len(walls)}  ({time.time() - t1:.1f}s)")

        # ── walls: geometric v2 ──
        from cloud2bim.elements.walls import detect_walls
        t1 = time.time()
        try:
            walls_v2 = detect_walls(
                storey_points=spts, z_floor=z_floor, z_ceiling=z_ceiling,
                storey_idx=0, cfg=cfg.walls, pc_resolution=cfg.slabs.pc_resolution,
                grid_coefficient=cfg.slabs.grid_coefficient,
                slab_polygon_xy=slab_poly, semantic_labels=slabels,
            )
            p, r, f1, err = score_walls(b.gt_walls, walls_v2)
            print(f"walls[v2]      P={p:.2f} R={r:.2f} F1={f1:.2f} "
                  f"axis_err={err * 100:.1f}cm  n={len(walls_v2)}  ({time.time() - t1:.1f}s)")
        except Exception as exc:
            print(f"walls[v2]      CRASHED: {exc}")

        # ── walls: vertical ──
        from cloud2bim.elements.walls_vertical import detect_walls_vertical
        t1 = time.time()
        try:
            walls_vt = detect_walls_vertical(
                storey_points=spts, z_floor=z_floor, z_ceiling=z_ceiling,
                storey_idx=0, cfg=cfg.walls, pc_resolution=cfg.slabs.pc_resolution,
                grid_coefficient=cfg.slabs.grid_coefficient,
                slab_polygon_xy=slab_poly, semantic_labels=slabels,
            )
            p, r, f1, err = score_walls(b.gt_walls, walls_vt)
            print(f"walls[vert]    P={p:.2f} R={r:.2f} F1={f1:.2f} "
                  f"axis_err={err * 100:.1f}cm  n={len(walls_vt)}  ({time.time() - t1:.1f}s)")
        except Exception as exc:
            print(f"walls[vert]    CRASHED: {exc}")

        # ── openings (ML, on ML walls) ──
        ops = extract_openings_ml(walls, spts, slabels, cfg.openings, cfg.segmentation)
        p, r = score_openings(b.gt_openings, b.gt_walls, ops, walls)
        n_doors = sum(1 for o in ops if o.type == "door")
        print(f"openings[ml]   P={p:.2f} R={r:.2f}  n={len(ops)} ({n_doors} doors)")

        # ── columns ──
        if b.gt_columns:
            from cloud2bim.elements.columns import detect_columns
            cfg.columns.enabled = True
            cols = detect_columns(
                storey_points=spts, walls=walls, z_floor=z_floor,
                z_ceiling=z_ceiling, storey_idx=0, cfg=cfg.columns,
                pc_resolution=cfg.slabs.pc_resolution,
                grid_coefficient=cfg.slabs.grid_coefficient,
                semantic_labels=slabels, column_classes=["column"],
            )
            hit = sum(
                1 for (gx, gy) in b.gt_columns
                if any(np.hypot(c.center_x - gx, c.center_y - gy) < 0.5 for c in cols)
            )
            print(f"columns        R={hit / len(b.gt_columns):.2f} "
                  f"({hit}/{len(b.gt_columns)} hit, {len(cols)} detected)")
        print(f"[{name} total {time.time() - t0:.1f}s]")


if __name__ == "__main__":
    args = sys.argv[1:] or list(SCENARIOS)
    main(args)
