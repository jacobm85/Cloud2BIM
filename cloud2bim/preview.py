"""2D floor plan preview generation.

Renders detected slabs, walls, openings and (if any) roof planes as a
PNG next to the output IFC. Gives the operator a quick visual check
before opening the model in Revit.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cloud2bim.elements.openings import Opening
from cloud2bim.elements.slabs import Slab
from cloud2bim.elements.walls import Wall
from cloud2bim.logging import get_logger

log = get_logger(__name__)


def render_floor_plan(
    output_path: Path,
    slabs: List[Slab],
    walls: Iterable[Wall],
    openings: Iterable[Opening],
    columns: Iterable = (),
    stairs: Iterable = (),
) -> None:
    """Save a PNG floor plan to ``output_path``."""
    fig, ax = plt.subplots(figsize=(14, 14))
    fig.patch.set_facecolor("#1a1d27")
    ax.set_facecolor("#0f1117")

    for slab in slabs:
        xs = list(slab.polygon_x) + [slab.polygon_x[0]]
        ys = list(slab.polygon_y) + [slab.polygon_y[0]]
        ax.fill(slab.polygon_x, slab.polygon_y, color="#2e3350", alpha=0.5)
        ax.plot(xs, ys, color="#4f5580", linewidth=1, linestyle="--")

    walls_list = list(walls)
    for wall in walls_list:
        sp, ep = wall.start, wall.end
        is_placeholder = "_placeholder" in wall.label
        is_exterior = wall.label.startswith("exterior")
        if is_placeholder:
            color = "#aa6666"
            lw = max(1.0, wall.thickness * 30)
            style = ":"
        else:
            color = "#6699cc" if is_exterior else "#99aacc"
            lw = max(1.5, wall.thickness * 40)
            style = "-"
        ax.plot(
            [sp[0], ep[0]], [sp[1], ep[1]],
            color=color, linewidth=lw, linestyle=style, solid_capstyle="round",
        )

    for op in openings:
        if op.wall_index < 0 or op.wall_index >= len(walls_list):
            continue
        wall = walls_list[op.wall_index]
        sp, ep = wall.start, wall.end
        dx, dy = ep[0] - sp[0], ep[1] - sp[1]
        length = float(np.hypot(dx, dy))
        if length < 1e-6:
            continue
        ux, uy = dx / length, dy / length
        center_x = sp[0] + ux * (op.x_along_wall_start + op.x_along_wall_end) / 2
        center_y = sp[1] + uy * (op.x_along_wall_start + op.x_along_wall_end) / 2
        w = op.width
        color = "#76c8e8" if op.type == "window" else "#f5a623"
        ax.plot(
            [center_x - ux * w / 2, center_x + ux * w / 2],
            [center_y - uy * w / 2, center_y + uy * w / 2],
            color=color, linewidth=3,
        )

    # Columns: small filled squares
    for col in columns:
        hx, hy = col.size_x / 2, col.size_y / 2
        ax.fill(
            [col.center_x - hx, col.center_x + hx, col.center_x + hx, col.center_x - hx],
            [col.center_y - hy, col.center_y - hy, col.center_y + hy, col.center_y + hy],
            color="#a48ee6", alpha=0.85,
        )

    # Stairs: hashed rectangles
    for stair in stairs:
        xs = list(stair.polygon_x)
        ys = list(stair.polygon_y)
        ax.fill(xs, ys, color="#e6a85a", alpha=0.3)
        ax.plot(xs, ys, color="#e6a85a", linewidth=1, linestyle="--")

    ax.set_aspect("equal")
    ax.axis("off")
    handles = [
        mpatches.Patch(color="#6699cc", label="Yttervägg"),
        mpatches.Patch(color="#99aacc", label="Innervägg"),
        mpatches.Patch(color="#aa6666", label="Platshållarvägg (10 cm)"),
        mpatches.Patch(color="#76c8e8", label="Fönster"),
        mpatches.Patch(color="#f5a623", label="Dörr"),
        mpatches.Patch(color="#a48ee6", label="Pelare"),
        mpatches.Patch(color="#e6a85a", label="Trappa"),
    ]
    ax.legend(
        handles=handles, loc="upper right",
        facecolor="#1a1d27", labelcolor="white", edgecolor="#2e3350", fontsize=9,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    log.info("Floor-plan preview saved: %s", output_path)


# ── Wizard-mode previews ────────────────────────────────────────────────────

def render_z_histogram(
    output_path: Path,
    bin_centers: np.ndarray,
    counts: np.ndarray,
    peak_z: list[float],
    slabs: List[Slab] | None = None,
    cross_section_bands: list[tuple[float, float] | None] | None = None,
    cross_section_bands_lower: list[tuple[float, float] | None] | None = None,
) -> None:
    """Horizontal Z-histogram with peak/slab/band markers.

    Y-axis is Z (m); X-axis is point count. Detected peaks are dots,
    slabs are shaded grey bands, user-picked cross-section bands are
    coloured rectangles.
    """
    fig, ax = plt.subplots(figsize=(6, 10))
    fig.patch.set_facecolor("#1a1d27")
    ax.set_facecolor("#0f1117")

    ax.barh(bin_centers, counts, height=(bin_centers[1] - bin_centers[0]) if len(bin_centers) > 1 else 0.1,
            color="#3a4a72", edgecolor="none")

    if peak_z:
        ax.scatter([counts.max() * 1.02] * len(peak_z), peak_z,
                   color="#ffcc66", s=40, zorder=5, label="Topp (yta)")
        for z in peak_z:
            ax.axhline(z, color="#ffcc66", linewidth=0.5, alpha=0.4)

    if slabs:
        x_label = counts.max() * 0.5 if len(counts) else 0
        for i, slab in enumerate(slabs):
            z_top = slab.bottom_z + slab.thickness
            ax.axhspan(slab.bottom_z, z_top, color="#888888", alpha=0.35, zorder=2)
            ax.text(
                x_label, (slab.bottom_z + z_top) / 2, f"#{i}",
                color="#ffffff", fontsize=11, fontweight="bold",
                ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#444444",
                          edgecolor="#888888", alpha=0.85),
            )
        # legend
        ax.fill_betweenx([0, 0], 0, 0, color="#888888", alpha=0.35, label="Slab")

    if cross_section_bands:
        palette = ["#76c8e8", "#f5a623", "#a3e635", "#e88adf"]
        for i, band in enumerate(cross_section_bands):
            if band is None:
                continue
            ax.axhspan(band[0], band[1], color=palette[i % len(palette)],
                       alpha=0.45, zorder=4,
                       label=f"Väggsnitt v{i}" if i < 4 else None)

    if cross_section_bands_lower:
        for i, band in enumerate(cross_section_bands_lower):
            if band is None:
                continue
            ax.axhspan(band[0], band[1], color="#c084fc",
                       alpha=0.50, zorder=4,
                       label="Lågsnitt (fönsterkoll)" if i == 0 else None)

    ax.set_xlabel("Antal punkter", color="white", fontsize=10)
    ax.set_ylabel("Z (m)", color="white", fontsize=10)
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_color("#2e3350")
    if peak_z or slabs or cross_section_bands:
        leg = ax.legend(facecolor="#1a1d27", labelcolor="white",
                        edgecolor="#2e3350", fontsize=9, loc="upper right")
        if leg:
            leg.get_frame().set_alpha(0.9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    log.info("Z-histogram preview saved: %s", output_path)


def render_cross_section(
    output_path: Path,
    points_xy: np.ndarray,
    title: str = "",
) -> None:
    """2D occupancy map of points in a Z-band — the floor-plan-like view
    the user uses to validate the band before wall detection runs."""
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor("#1a1d27")
    ax.set_facecolor("#0f1117")

    if len(points_xy) > 0:
        ax.scatter(points_xy[:, 0], points_xy[:, 1], s=0.4,
                   color="#76c8e8", alpha=0.6)
    else:
        ax.text(0.5, 0.5, "Inga punkter i bandet", ha="center", va="center",
                color="#aaaaaa", transform=ax.transAxes, fontsize=14)

    ax.set_aspect("equal")
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_color("#2e3350")
    if title:
        ax.set_title(title, color="white", fontsize=11)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    log.info("Cross-section preview saved: %s", output_path)


def render_storey_walls(
    output_path: Path,
    walls: Iterable[Wall],
    title: str = "",
) -> None:
    """Lightweight floor-plan preview for a single storey (used between
    walls and openings stages in wizard mode)."""
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor("#1a1d27")
    ax.set_facecolor("#0f1117")

    for w in walls:
        sp, ep = w.start, w.end
        is_ext = w.label.startswith("exterior")
        color = "#6699cc" if is_ext else "#99aacc"
        lw = max(1.5, w.thickness * 40)
        ax.plot([sp[0], ep[0]], [sp[1], ep[1]], color=color, linewidth=lw,
                solid_capstyle="round")

    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, color="white", fontsize=11)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
