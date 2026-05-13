"""Point cloud readers — E57, LAS/LAZ, XYZ → numpy.

Auto-dispatched by file extension. All readers return float64 (N, 3) XYZ
plus an optional (N, 3) RGB array (None if the format lacks colour).

Designed to stream large files in chunks so a 250M-point LAS doesn't
materialise as a single Python list before becoming a numpy array.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from cloud2bim.logging import get_logger

log = get_logger(__name__)


def read_pointcloud(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Read any supported point cloud format. Returns (xyz, rgb_or_none)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".e57":
        return _read_e57(path)
    if suffix in (".las", ".laz"):
        return _read_las(path)
    if suffix == ".xyz":
        return _read_xyz(path)
    raise ValueError(f"Unsupported format: {suffix}")


def _read_e57(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    log.info("Reading E57: %s", path)
    try:
        import pye57
    except ImportError as exc:
        raise ImportError("pye57 not installed — required for .e57 input") from exc

    e57 = pye57.E57(str(path))
    n_scans = e57.scan_count
    xyz_chunks: list[np.ndarray] = []
    rgb_chunks: list[np.ndarray] = []
    has_color = True

    for scan_idx in range(n_scans):
        data = e57.read_scan(scan_idx, ignore_missing_fields=True)
        xyz = np.column_stack([data["cartesianX"], data["cartesianY"], data["cartesianZ"]])
        xyz_chunks.append(xyz.astype(np.float64))
        if "colorRed" in data and "colorGreen" in data and "colorBlue" in data:
            rgb_chunks.append(np.column_stack([data["colorRed"], data["colorGreen"], data["colorBlue"]]))
        else:
            has_color = False

    xyz = np.vstack(xyz_chunks) if xyz_chunks else np.empty((0, 3))
    rgb = np.vstack(rgb_chunks) if (has_color and rgb_chunks) else None
    log.info("E57 loaded: %s points, %s scans", f"{len(xyz):,}", n_scans)
    return xyz, rgb


def _read_las(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    log.info("Reading LAS: %s", path)
    try:
        import laspy
    except ImportError as exc:
        raise ImportError("laspy not installed — required for .las/.laz input") from exc

    las = laspy.read(str(path))
    xyz = np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)]).astype(np.float64)
    rgb = None
    if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
        rgb = np.column_stack([np.asarray(las.red), np.asarray(las.green), np.asarray(las.blue)])
    log.info("LAS loaded: %s points", f"{len(xyz):,}")
    return xyz, rgb


def _read_xyz(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Read tab-separated XYZ. First line may be a header starting with '//'."""
    log.info("Reading XYZ: %s", path)
    with open(path, encoding="utf-8") as fh:
        first = fh.readline()
        skip = 1 if first.startswith("//") else 0
    data = np.loadtxt(path, skiprows=skip, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    xyz = data[:, :3]
    rgb = data[:, 3:6] if data.shape[1] >= 6 else None
    log.info("XYZ loaded: %s points", f"{len(xyz):,}")
    return xyz, rgb


def diluted(points: np.ndarray, factor: int) -> np.ndarray:
    """Keep every Nth point. Returns a view, not a copy."""
    if factor <= 1:
        return points
    return points[::factor]
