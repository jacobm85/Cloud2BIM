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

    ax.set_aspect("equal")
    ax.axis("off")
    handles = [
        mpatches.Patch(color="#6699cc", label="Yttervägg"),
        mpatches.Patch(color="#99aacc", label="Innervägg"),
        mpatches.Patch(color="#aa6666", label="Platshållarvägg (10 cm)"),
        mpatches.Patch(color="#76c8e8", label="Fönster"),
        mpatches.Patch(color="#f5a623", label="Dörr"),
    ]
    ax.legend(
        handles=handles, loc="upper right",
        facecolor="#1a1d27", labelcolor="white", edgecolor="#2e3350", fontsize=9,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    log.info("Floor-plan preview saved: %s", output_path)
