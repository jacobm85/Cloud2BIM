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
        try:
            data = e57.read_scan(scan_idx, ignore_missing_fields=True)
        except Exception as exc:
            # Some scanners write E57 without /data3D/N/pose, which makes the
            # default to_global() transform fail. Retry with transform=False.
            log.warning(
                "Scan %d transform failed (%s) — reading local coordinates",
                scan_idx, exc,
            )
            data = e57.read_scan(scan_idx, ignore_missing_fields=True, transform=False)
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


def _read_las(path: Path, chunk_size: int = 5_000_000) -> tuple[np.ndarray, np.ndarray | None]:
    """Stream a LAS/LAZ file in chunks.

    laspy.read() loads the whole file in one allocation, peaking at
    5–10× the file size in RAM — large surveying scans get OOM-killed
    silently in containers. chunk_iterator() reads in fixed-size blocks
    so peak memory is bounded by ``chunk_size`` regardless of file size.
    """
    log.info("Reading LAS: %s", path)
    try:
        import laspy
    except ImportError as exc:
        raise ImportError("laspy not installed — required for .las/.laz input") from exc

    xyz_chunks: list[np.ndarray] = []
    rgb_chunks: list[np.ndarray] = []
    has_color: bool | None = None

    with laspy.open(str(path)) as reader:
        total = reader.header.point_count
        log.info("LAS header: %s points — streaming in %s-point chunks", f"{total:,}", f"{chunk_size:,}")
        for chunk in reader.chunk_iterator(chunk_size):
            xyz_chunks.append(
                np.column_stack([
                    np.asarray(chunk.x), np.asarray(chunk.y), np.asarray(chunk.z),
                ]).astype(np.float64)
            )
            if has_color is None:
                has_color = (
                    hasattr(chunk, "red")
                    and hasattr(chunk, "green")
                    and hasattr(chunk, "blue")
                )
            if has_color:
                rgb_chunks.append(
                    np.column_stack([
                        np.asarray(chunk.red),
                        np.asarray(chunk.green),
                        np.asarray(chunk.blue),
                    ])
                )

    if not xyz_chunks:
        return np.empty((0, 3), dtype=np.float64), None
    xyz = np.vstack(xyz_chunks) if len(xyz_chunks) > 1 else xyz_chunks[0]
    rgb = (np.vstack(rgb_chunks) if len(rgb_chunks) > 1 else rgb_chunks[0]) if rgb_chunks else None
    log.info("LAS loaded: %s points (rgb=%s)", f"{len(xyz):,}", rgb is not None)
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
