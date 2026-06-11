"""Wall-axis regularisation.

Raw extracted wall segments are noisy: directions wobble a few degrees,
one physical wall arrives as several fragments, and corners don't quite
meet. Real buildings are far more regular than that — most walls follow
a small set of dominant directions (usually two, orthogonal) and walls
that meet share an exact corner point.

``regularize_walls`` applies, in order:
    1. Snap segment directions to the dominant building directions
       (weighted angle histogram, mod 180°).
    2. Merge collinear fragments into single walls (iterated to fixpoint
       so chains collapse).
    3. Snap endpoints to nearby wall-wall intersections so corners close.

Both the ML and geometric extraction paths can call this — it only deals
in ``[[x1, y1], [x2, y2]]`` axes + thickness lists.
"""
from __future__ import annotations

import numpy as np

from cloud2bim.geometry.lines import line_intersection
from cloud2bim.logging import get_logger

log = get_logger(__name__)


def regularize_walls(
    axes: list,
    thicknesses: list[float],
    labels: list[str] | None = None,
    *,
    snap_angle_deg: float = 7.0,
    collinear_angle_deg: float = 4.0,
    collinear_offset: float = 0.20,
    collinear_gap: float = 1.5,
    corner_snap: float = 0.45,
    min_length: float = 0.05,
):
    """Clean up raw wall axes.

    Returns ``(axes, thicknesses)``, or ``(axes, thicknesses, labels)``
    when a ``labels`` list is supplied (the geometric v2 path tracks a
    per-wall label; merged segments keep the longer member's label).

    ``collinear_gap`` should come from ``WallConfig.collinear_merge_distance``
    so the wizard's setting keeps working; the rest have sane fixed defaults.
    """
    if not axes:
        return ([], [], []) if labels is not None else ([], [])
    axes = [[list(map(float, a[0])), list(map(float, a[1]))] for a in axes]
    thicknesses = list(thicknesses)

    dirs = _dominant_directions(axes)
    if dirs:
        n_snapped = _snap_to_directions(axes, dirs, np.deg2rad(snap_angle_deg))
        log.info(
            "Regularize: %d dominant directions %s — snapped %d/%d segments",
            len(dirs), [round(np.rad2deg(d), 1) for d in dirs], n_snapped, len(axes),
        )

    axes, thicknesses, out_labels = _merge_collinear(
        axes, thicknesses, labels,
        angle_tol=np.deg2rad(collinear_angle_deg),
        offset_tol=collinear_offset,
        gap_tol=collinear_gap,
    )

    _snap_corners(axes, corner_snap)

    # Drop segments that collapsed below min_length during merging/snapping.
    keep = [i for i, ax in enumerate(axes) if _length(ax) >= min_length]
    axes = [axes[i] for i in keep]
    thicknesses = [thicknesses[i] for i in keep]
    if labels is not None:
        return axes, thicknesses, [out_labels[i] for i in keep]
    return axes, thicknesses


# ── internals ────────────────────────────────────────────────────────────────


def _length(ax) -> float:
    return float(np.hypot(ax[1][0] - ax[0][0], ax[1][1] - ax[0][1]))


def _angle(ax) -> float:
    """Segment direction in [0, π)."""
    a = np.arctan2(ax[1][1] - ax[0][1], ax[1][0] - ax[0][0])
    return float(a % np.pi)


def _ang_dist(a: float, b: float) -> float:
    """Distance between two undirected angles (mod π)."""
    d = abs(a - b) % np.pi
    return min(d, np.pi - d)


def _dominant_directions(axes: list, max_dirs: int = 4) -> list[float]:
    """Length-weighted angle histogram → up to ``max_dirs`` peak directions.

    A direction qualifies if it carries ≥ 15 % of total wall length, so a
    single skewed fragment can't become a snapping target.
    """
    angles = np.array([_angle(ax) for ax in axes])
    weights = np.array([_length(ax) for ax in axes])
    total = float(weights.sum())
    if total <= 0:
        return []

    bin_w = np.deg2rad(2.0)
    n_bins = int(np.ceil(np.pi / bin_w))
    hist = np.zeros(n_bins)
    for a, w in zip(angles, weights):
        hist[min(int(a / bin_w), n_bins - 1)] += w

    # Smooth circularly so a direction split across two bins still peaks.
    kernel = np.array([0.25, 0.5, 0.25])
    hist = (
        kernel[0] * np.roll(hist, 1) + kernel[1] * hist + kernel[2] * np.roll(hist, -1)
    )

    dirs: list[float] = []
    suppress = np.deg2rad(15.0)
    work = hist.copy()
    for _ in range(max_dirs):
        peak_bin = int(np.argmax(work))
        if work[peak_bin] < 0.15 * total:
            break
        centre = (peak_bin + 0.5) * bin_w
        # Refine: weighted mean of member angles near the peak (mod π —
        # use doubled-angle circular mean so 1° and 179° average to 0°).
        near = np.array([_ang_dist(a, centre) <= suppress for a in angles])
        if near.any():
            w = weights[near]
            doubled = angles[near] * 2.0
            mean = np.arctan2((w * np.sin(doubled)).sum(), (w * np.cos(doubled)).sum())
            centre = float((mean / 2.0) % np.pi)
        dirs.append(centre)
        for b in range(n_bins):
            if _ang_dist((b + 0.5) * bin_w, centre) <= suppress:
                work[b] = 0.0
    return dirs


def _snap_to_directions(axes: list, dirs: list[float], tol: float) -> int:
    """Rotate each segment about its midpoint onto the nearest dominant
    direction when within ``tol``. Returns number of snapped segments."""
    n = 0
    for ax in axes:
        a = _angle(ax)
        best = min(dirs, key=lambda d: _ang_dist(a, d))
        if 0 < _ang_dist(a, best) <= tol:
            mid = np.array([(ax[0][0] + ax[1][0]) / 2, (ax[0][1] + ax[1][1]) / 2])
            half = _length(ax) / 2
            u = np.array([np.cos(best), np.sin(best)])
            # Keep the original start→end ordering roughly intact.
            v = np.array([ax[1][0] - ax[0][0], ax[1][1] - ax[0][1]])
            if float(v @ u) < 0:
                u = -u
            p1, p2 = mid - half * u, mid + half * u
            ax[0][:] = [float(p1[0]), float(p1[1])]
            ax[1][:] = [float(p2[0]), float(p2[1])]
            n += 1
    return n


def _merge_collinear(
    axes: list,
    thicknesses: list[float],
    labels: list[str] | None,
    angle_tol: float,
    offset_tol: float,
    gap_tol: float,
) -> tuple[list, list[float], list[str]]:
    """Merge near-collinear segments. Iterates until no merge applies.

    Two segments merge when their directions agree within ``angle_tol``,
    their perpendicular offset is within ``offset_tol`` and the 1D gap
    along the shared direction is below ``gap_tol``. The merged segment
    spans the union of both projections, sits on the length-weighted mean
    line, and keeps the larger thickness (and the longer member's label).
    """
    if labels is None:
        labels = [""] * len(axes)
    items = [
        {"ax": ax, "t": t, "lbl": lbl}
        for ax, t, lbl in zip(axes, thicknesses, labels)
        if _length(ax) > 1e-9
    ]
    changed = True
    while changed:
        changed = False
        for i in range(len(items)):
            if items[i] is None:
                continue
            for j in range(i + 1, len(items)):
                if items[j] is None:
                    continue
                merged = _try_merge_pair(
                    items[i], items[j], angle_tol, offset_tol, gap_tol
                )
                if merged is not None:
                    items[i] = merged
                    items[j] = None
                    changed = True
        items = [it for it in items if it is not None]
    return (
        [it["ax"] for it in items],
        [it["t"] for it in items],
        [it["lbl"] for it in items],
    )


def _try_merge_pair(it1, it2, angle_tol, offset_tol, gap_tol):
    ax1, ax2 = it1["ax"], it2["ax"]
    a1, a2 = _angle(ax1), _angle(ax2)
    if _ang_dist(a1, a2) > angle_tol:
        return None

    l1, l2 = _length(ax1), _length(ax2)
    # Length-weighted mean direction (doubled-angle circular mean).
    doubled = np.array([a1 * 2, a2 * 2])
    w = np.array([l1, l2])
    mean = np.arctan2((w * np.sin(doubled)).sum(), (w * np.cos(doubled)).sum())
    ang = float((mean / 2.0) % np.pi)
    u = np.array([np.cos(ang), np.sin(ang)])
    n_vec = np.array([-u[1], u[0]])

    pts = np.array([ax1[0], ax1[1], ax2[0], ax2[1]], dtype=float)
    offs = pts @ n_vec
    off1 = (offs[0] + offs[1]) / 2
    off2 = (offs[2] + offs[3]) / 2
    if abs(off1 - off2) > offset_tol:
        return None

    longs = pts @ u
    lo1, hi1 = sorted((longs[0], longs[1]))
    lo2, hi2 = sorted((longs[2], longs[3]))
    gap = max(lo1, lo2) - min(hi1, hi2)
    if gap > gap_tol:
        return None

    off = (off1 * l1 + off2 * l2) / (l1 + l2)
    lo, hi = min(lo1, lo2), max(hi1, hi2)
    p1 = lo * u + off * n_vec
    p2 = hi * u + off * n_vec
    return {
        "ax": [[float(p1[0]), float(p1[1])], [float(p2[0]), float(p2[1])]],
        "t": max(it1["t"], it2["t"]),
        "lbl": it1["lbl"] if l1 >= l2 else it2["lbl"],
    }


def _snap_corners(axes: list, snap: float) -> None:
    """Move endpoints onto nearby wall-wall intersections (in place).

    Each endpoint snaps to the *closest* qualifying intersection rather
    than the last one tried, so T-junction clusters resolve consistently.
    Near-parallel pairs are skipped — their intersection is far away and
    numerically wild.
    """
    n = len(axes)
    for i in range(n):
        for k in range(2):
            best_pt, best_d = None, snap
            for j in range(n):
                if j == i:
                    continue
                if _ang_dist(_angle(axes[i]), _angle(axes[j])) < np.deg2rad(10):
                    continue
                inter = line_intersection(axes[i], axes[j])
                if inter is None:
                    continue
                d = float(np.hypot(axes[i][k][0] - inter[0], axes[i][k][1] - inter[1]))
                if d <= best_d:
                    best_pt, best_d = inter, d
            if best_pt is not None:
                axes[i][k] = [float(best_pt[0]), float(best_pt[1])]
