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


def read_pointcloud(
    path: str | Path,
    read_stride: int = 1,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Read any supported point cloud format. Returns (xyz, rgb_or_none).

    ``read_stride`` keeps every Nth point during reading. Only the PTX
    reader honours it today (essential because PTX is ASCII and a single
    scan can be tens of GB on disk). Other readers ignore it; the caller
    can still post-dilute with ``diluted()``.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".e57":
        return _read_e57(path)
    if suffix in (".las", ".laz"):
        return _read_las(path)
    if suffix == ".xyz":
        return _read_xyz(path)
    if suffix == ".ptx":
        return _read_ptx(path, read_stride=read_stride)
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


def _read_ptx(
    path: Path,
    read_stride: int = 1,
    scan_chunk_points: int = 5_000_000,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Read a Leica Cyclone PTX file (ASCII, multi-scan).

    PTX layout, repeated per scan until EOF:
        line 1:  ncols                     (int)
        line 2:  nrows                     (int)
        line 3:  scanner translation       (x y z)
        line 4:  scanner X axis            (x y z)
        line 5:  scanner Y axis            (x y z)
        line 6:  scanner Z axis            (x y z)
        lines 7..10:  4×4 transform matrix (row-major; row 4 is translation)
        then ncols*nrows point lines, each either:
            x y z intensity                (4 cols, no colour)
            x y z intensity r g b          (7 cols, with colour)

    Points the scanner missed at a (col, row) cell are written as
    "0 0 0 0" or "0 0 0 0.5" — we skip any line whose XYZ is exactly
    (0, 0, 0). Per-scan transform is applied so all output points sit
    in a common world frame.

    Streaming: parses line-by-line and never loads the whole file into
    memory. ``read_stride`` keeps every Nth point during read so
    multi-tens-of-GB PTX exports stay tractable. Within a scan, points
    are buffered into numpy chunks of ``scan_chunk_points`` so the
    Python list of strings doesn't balloon either.
    """
    log.info("Reading PTX: %s", path)
    stride = max(1, int(read_stride))
    if stride > 1:
        log.info("PTX read stride = %d (keeping every %d:th point)", stride, stride)

    xyz_chunks: list[np.ndarray] = []
    rgb_chunks: list[np.ndarray] = []
    file_has_color = True   # AND across all scans; False if any scan lacks RGB
    n_scans = 0

    with open(path, "r", encoding="ascii", errors="replace") as fh:
        while True:
            header = _ptx_read_header(fh)
            if header is None:
                break
            ncols, nrows, transform = header
            n_points = ncols * nrows
            n_scans += 1
            log.info(
                "PTX scan %d: %s × %s = %s points",
                n_scans, f"{ncols:,}", f"{nrows:,}", f"{n_points:,}",
            )

            scan_xyz, scan_rgb = _ptx_read_scan_points(
                fh, n_points, stride=stride,
                chunk_points=scan_chunk_points,
                progress_label=f"prepare scan {n_scans}",
            )
            if len(scan_xyz) == 0:
                log.info("PTX scan %d: no valid points after stride/skip", n_scans)
                continue

            # Apply the per-scan transform: world = local @ R + t.
            # Leica's convention is row-vectors × row-major matrix; the
            # 3×3 rotation lives in the top-left and the translation is
            # the first three entries of the bottom row.
            R = transform[:3, :3]
            t = transform[3, :3]
            scan_xyz_world = scan_xyz @ R + t
            xyz_chunks.append(scan_xyz_world)
            if scan_rgb is not None:
                rgb_chunks.append(scan_rgb)
            else:
                file_has_color = False

    if not xyz_chunks:
        log.warning("PTX: parsed 0 scans / 0 valid points from %s", path)
        return np.empty((0, 3), dtype=np.float64), None

    xyz = np.vstack(xyz_chunks) if len(xyz_chunks) > 1 else xyz_chunks[0]
    rgb = None
    if file_has_color and rgb_chunks:
        rgb = np.vstack(rgb_chunks) if len(rgb_chunks) > 1 else rgb_chunks[0]
    log.info("PTX loaded: %s points, %d scans (rgb=%s)",
             f"{len(xyz):,}", n_scans, rgb is not None)
    return xyz, rgb


def _ptx_read_header(fh) -> tuple[int, int, np.ndarray] | None:
    """Read the 10-line PTX scan header. Returns None at EOF.

    Tolerates extra blank lines between scans (some exporters add them).
    """
    # Skip leading blanks to find the first header line.
    first = None
    for raw in fh:
        s = raw.strip()
        if s:
            first = s
            break
    if first is None:
        return None  # clean EOF

    header_lines = [first]
    for _ in range(9):
        line = fh.readline()
        if not line:
            log.warning("PTX: truncated header (expected 10 lines, got %d)", len(header_lines))
            return None
        header_lines.append(line.strip())

    try:
        ncols = int(header_lines[0])
        nrows = int(header_lines[1])
    except ValueError as exc:
        log.warning("PTX: bad header (%s) — stopping read", exc)
        return None

    # Header lines 6..9 (0-indexed) are the 4×4 transform, row-major,
    # each row has 4 floats.
    try:
        transform = np.array(
            [list(map(float, header_lines[i].split())) for i in range(6, 10)],
            dtype=np.float64,
        )
        if transform.shape != (4, 4):
            raise ValueError(f"transform shape {transform.shape}")
    except ValueError as exc:
        log.warning("PTX: bad transform matrix (%s) — using identity", exc)
        transform = np.eye(4, dtype=np.float64)

    return ncols, nrows, transform


def _ptx_read_scan_points(
    fh,
    n_points: int,
    stride: int,
    chunk_points: int,
    progress_label: str = "prepare",
) -> tuple[np.ndarray, np.ndarray | None]:
    """Parse n_points lines from fh, returning (xyz, rgb_or_none) for the scan.

    Buffers in numpy-friendly chunks of ~``chunk_points`` rows so we
    don't keep millions of Python floats live at once. ``stride`` is
    applied per-line so a stride=10 file reads at ~1/10 the work.

    Emits ``[PROGRESS] <label> done/total eta=s`` lines every ~1 % so
    the wizard front-end can render a progress bar — important for
    100 GB PTX exports where a single scan can take 10+ min.
    """
    import time as _time
    xyz_buf = np.empty((chunk_points, 3), dtype=np.float64)
    rgb_buf = np.empty((chunk_points, 3), dtype=np.int32)
    buf_pos = 0
    xyz_chunks: list[np.ndarray] = []
    rgb_chunks: list[np.ndarray] = []
    has_color: bool | None = None
    t_start = _time.time()
    every = max(1, n_points // 100)

    for i in range(n_points):
        if i and (i % every) == 0:
            elapsed = max(0.001, _time.time() - t_start)
            rate = (i + 1) / elapsed
            eta_s = int((n_points - i) / rate) if rate > 0 else -1
            log.info("[PROGRESS] %s %d/%d eta=%d", progress_label, i, n_points, eta_s)
        line = fh.readline()
        if not line:
            log.warning("PTX scan truncated at %d/%d points", i, n_points)
            break
        # stride-skip without parsing the line at all → big win on huge files
        if stride > 1 and (i % stride) != 0:
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2])
        except ValueError:
            continue
        # Skip no-return points (scanner cell with no hit). Standard PTX
        # marker is (0,0,0) for XYZ regardless of the intensity column.
        if x == 0.0 and y == 0.0 and z == 0.0:
            continue

        xyz_buf[buf_pos] = (x, y, z)
        if len(parts) >= 7:
            if has_color is None:
                has_color = True
            try:
                rgb_buf[buf_pos] = (int(parts[4]), int(parts[5]), int(parts[6]))
            except ValueError:
                rgb_buf[buf_pos] = (0, 0, 0)
        else:
            if has_color is None:
                has_color = False
        buf_pos += 1

        if buf_pos == chunk_points:
            xyz_chunks.append(xyz_buf[:buf_pos].copy())
            if has_color:
                rgb_chunks.append(rgb_buf[:buf_pos].copy())
            buf_pos = 0

    # Flush tail.
    if buf_pos > 0:
        xyz_chunks.append(xyz_buf[:buf_pos].copy())
        if has_color:
            rgb_chunks.append(rgb_buf[:buf_pos].copy())

    if not xyz_chunks:
        return np.empty((0, 3), dtype=np.float64), None
    xyz = np.vstack(xyz_chunks) if len(xyz_chunks) > 1 else xyz_chunks[0]
    rgb = (np.vstack(rgb_chunks).astype(np.float64)
           if has_color and rgb_chunks else None)
    return xyz, rgb


def diluted(points: np.ndarray, factor: int) -> np.ndarray:
    """Keep every Nth point. Returns a view, not a copy."""
    if factor <= 1:
        return points
    return points[::factor]


DENOISE_MAX_POINTS = 40_000_000  # KD-tree build above this takes too long


def remove_outliers(
    xyz: np.ndarray,
    rgb: np.ndarray | None = None,
    neighbors: int = 12,
    std_ratio: float = 2.5,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    """Statistical outlier removal. Returns (xyz, rgb, n_removed).

    Drops points whose mean distance to ``neighbors`` nearest points is
    more than ``std_ratio`` standard deviations above the cloud average —
    scanner ghosting, glass reflections and airborne dust. These outliers
    smear wall faces (inflating fitted thickness) and can register as
    phantom horizontal surfaces in slab detection.

    No-ops (with a log line) when the cloud is too small to estimate the
    statistics or too large to KD-tree in reasonable time.
    """
    n = len(xyz)
    if n < 1_000:
        return xyz, rgb, 0
    if n > DENOISE_MAX_POINTS:
        log.warning(
            "Denoise skipped: %s points exceeds the %s limit — increase "
            "dilution if you want outlier removal on this scan",
            f"{n:,}", f"{DENOISE_MAX_POINTS:,}",
        )
        return xyz, rgb, 0
    try:
        import open3d as o3d
    except ImportError:
        log.warning("Denoise skipped: open3d not installed")
        return xyz, rgb, 0

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    _, keep_idx = pc.remove_statistical_outlier(
        nb_neighbors=neighbors, std_ratio=std_ratio,
    )
    keep_idx = np.asarray(keep_idx)
    if len(keep_idx) == n:
        return xyz, rgb, 0
    return (
        xyz[keep_idx],
        rgb[keep_idx] if rgb is not None else None,
        n - len(keep_idx),
    )
