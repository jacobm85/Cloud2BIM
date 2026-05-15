"""DXF export of detected geometry per storey.

Writes one DXF file per storey. Each element type lands on its own layer
so it can be toggled in CAD. Walls are drawn as their outer rectangle
(four LINEs) so the user gets the wall edges, not just a centreline.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Sequence

import ezdxf
import numpy as np

from cloud2bim.elements.openings import Opening
from cloud2bim.elements.slabs import Slab
from cloud2bim.elements.walls import Wall
from cloud2bim.logging import get_logger

log = get_logger(__name__)


# Layer name → ACI colour. ezdxf takes ACI ints (1=red, 2=yellow, 3=green,
# 4=cyan, 5=blue, 6=magenta, 7=white/black, 8=dark grey).
_LAYERS = {
    "CROSS_SECTION": 7,    # continuous outline from the cross-section trace
    "SLAB_OUTLINE": 8,
    "WALLS": 30,           # detected wall rectangles (analytical view)
    "WALL_AXIS": 8,
    "WINDOWS": 4,
    "DOORS": 30,
    "COLUMNS": 6,
    "STAIRS": 40,
}


def _ensure_layers(doc) -> None:
    for name, aci in _LAYERS.items():
        if name not in doc.layers:
            doc.layers.add(name=name, color=aci)


def _wall_corners(wall: Wall) -> List[tuple[float, float]]:
    """Return the four outer corners of a wall rectangle."""
    sx, sy = float(wall.start[0]), float(wall.start[1])
    ex, ey = float(wall.end[0]), float(wall.end[1])
    dx, dy = ex - sx, ey - sy
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return []
    # Perpendicular unit vector (left of wall direction)
    nx, ny = -dy / length, dx / length
    half_t = wall.thickness / 2.0
    p1 = (sx + nx * half_t, sy + ny * half_t)
    p2 = (ex + nx * half_t, ey + ny * half_t)
    p3 = (ex - nx * half_t, ey - ny * half_t)
    p4 = (sx - nx * half_t, sy - ny * half_t)
    return [p1, p2, p3, p4]


def _opening_centre(wall: Wall, op: Opening) -> tuple[float, float] | None:
    sx, sy = float(wall.start[0]), float(wall.start[1])
    ex, ey = float(wall.end[0]), float(wall.end[1])
    dx, dy = ex - sx, ey - sy
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return None
    ux, uy = dx / length, dy / length
    centre_along = (op.x_along_wall_start + op.x_along_wall_end) / 2.0
    return (sx + ux * centre_along, sy + uy * centre_along)


def write_storey_dxf(
    output_path: Path,
    storey_idx: int,
    walls: Sequence[Wall],
    openings: Iterable[Opening] = (),
    columns: Iterable = (),
    stairs: Iterable = (),
    slab: Slab | None = None,
    cross_section_contours: Sequence[np.ndarray] = (),
) -> None:
    """Write a DXF for a single storey.

    ``slab`` is the floor slab below the storey (its top polygon is used
    as the storey outline). ``walls`` is the storey's wall list.
    ``cross_section_contours`` is a list of (N,2) arrays — the closed
    contour traces from the cross-section binary mask. These give a
    continuous line that matches what the user sees in the wizard's
    cross-section preview.
    """
    doc = ezdxf.new(dxfversion="R2018", setup=True)
    doc.units = ezdxf.units.M  # metres
    msp = doc.modelspace()
    _ensure_layers(doc)

    # Continuous cross-section trace — closed polyline per detected blob.
    # This is the layer the user typically wants to print/measure: it's a
    # single line that follows the points in the section band, not a
    # collection of analytical wall rectangles.
    for contour in cross_section_contours:
        arr = np.asarray(contour, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] != 2:
            continue
        pts = [(float(x), float(y)) for x, y in arr]
        msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "CROSS_SECTION"})

    if slab is not None and len(slab.polygon_x) >= 3:
        pts = list(zip(slab.polygon_x.tolist(), slab.polygon_y.tolist()))
        msp.add_lwpolyline(pts, close=True, dxfattribs={"layer": "SLAB_OUTLINE"})

    walls_list = list(walls)
    for wall in walls_list:
        corners = _wall_corners(wall)
        if not corners:
            continue
        # Outer rectangle as closed polyline (= 4 lines, plus closes)
        msp.add_lwpolyline(corners, close=True, dxfattribs={"layer": "WALLS"})
        # Centreline (thin construction line, separate layer)
        msp.add_line(
            (float(wall.start[0]), float(wall.start[1])),
            (float(wall.end[0]), float(wall.end[1])),
            dxfattribs={"layer": "WALL_AXIS"},
        )

    for op in openings:
        if op.wall_index < 0 or op.wall_index >= len(walls_list):
            continue
        wall = walls_list[op.wall_index]
        centre = _opening_centre(wall, op)
        if centre is None:
            continue
        # Draw the opening as a short line across the wall thickness at its
        # centre, plus a width-line along the wall — gives a clear "hole" mark.
        sx, sy = float(wall.start[0]), float(wall.start[1])
        ex, ey = float(wall.end[0]), float(wall.end[1])
        dx, dy = ex - sx, ey - sy
        length = math.hypot(dx, dy)
        if length < 1e-9:
            continue
        ux, uy = dx / length, dy / length
        nx, ny = -uy, ux
        half_t = wall.thickness / 2.0
        half_w = op.width / 2.0
        cx, cy = centre
        layer = "WINDOWS" if op.type == "window" else "DOORS"
        msp.add_line(
            (cx - ux * half_w, cy - uy * half_w),
            (cx + ux * half_w, cy + uy * half_w),
            dxfattribs={"layer": layer},
        )
        msp.add_line(
            (cx - nx * half_t, cy - ny * half_t),
            (cx + nx * half_t, cy + ny * half_t),
            dxfattribs={"layer": layer},
        )

    for col in columns:
        hx, hy = col.size_x / 2.0, col.size_y / 2.0
        cx, cy = col.center_x, col.center_y
        rect = [
            (cx - hx, cy - hy), (cx + hx, cy - hy),
            (cx + hx, cy + hy), (cx - hx, cy + hy),
        ]
        msp.add_lwpolyline(rect, close=True, dxfattribs={"layer": "COLUMNS"})

    for stair in stairs:
        xs = np.asarray(stair.polygon_x).tolist()
        ys = np.asarray(stair.polygon_y).tolist()
        if len(xs) < 2:
            continue
        msp.add_lwpolyline(list(zip(xs, ys)), close=True,
                           dxfattribs={"layer": "STAIRS"})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(str(output_path))
    log.info("DXF written: %s (storey %d, %d walls)",
             output_path, storey_idx, len(walls_list))


def write_all_storeys(
    out_dir: Path,
    slabs: Sequence[Slab],
    storey_walls: Sequence[Sequence[Wall]],
    storey_openings: Sequence[Sequence[Opening]] = (),
    storey_columns: Sequence[Sequence] = (),
    storey_stairs: Sequence[Sequence] = (),
    storey_contours: Sequence[Sequence] = (),
    prefix: str = "plan_storey_",
) -> List[Path]:
    """Write one DXF per storey to ``out_dir``. Returns the file paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    n_storeys = len(storey_walls)
    for i in range(n_storeys):
        slab = slabs[i] if i < len(slabs) else None
        openings = storey_openings[i] if i < len(storey_openings) else []
        cols = storey_columns[i] if i < len(storey_columns) else []
        stairs = storey_stairs[i] if i < len(storey_stairs) else []
        contours = storey_contours[i] if i < len(storey_contours) else []
        path = out_dir / f"{prefix}{i}.dxf"
        write_storey_dxf(path, i, storey_walls[i], openings, cols, stairs,
                         slab, cross_section_contours=contours)
        paths.append(path)
    return paths
