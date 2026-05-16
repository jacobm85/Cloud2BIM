"""Pretrained-weights cache + downloader for segmentation backends.

Resolves the cache directory in a platform-appropriate way:
  Windows:  %LOCALAPPDATA%\\cloud2bim\\models
  macOS:    ~/Library/Caches/cloud2bim/models
  Linux:    ${XDG_CACHE_HOME:-~/.cache}/cloud2bim/models

Backends register their pretrained checkpoints in ``REGISTRY``. When a
backend asks for weights via ``resolve_weights(name)`` we either return
the cached path or download it from the registered URL. Downloads are
streamed with a tqdm progress bar (no progress bar if tqdm missing) and
verified against an expected SHA-256 when one is registered.

Override the cache directory by setting ``CLOUD2BIM_MODELS_DIR``.
"""
from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from cloud2bim.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class WeightSpec:
    """A pretrained checkpoint we know how to fetch."""
    filename: str
    url: str
    sha256: Optional[str] = None
    size_mb: Optional[float] = None
    description: str = ""


# ── Registry ──────────────────────────────────────────────────────────────────
#
# Add new pretrained checkpoints here. URLs should be stable public mirrors;
# Hugging Face Hub is preferred because it has bandwidth and resumable
# downloads. If you can't find a stable mirror, leave url="" and document a
# manual download path in the error message produced by resolve_weights().

REGISTRY: dict[str, WeightSpec] = {
    # PointTransformer V3 — S3DIS Area-5 pretrained, 13 classes.
    # Source: Pointcept project (Wu et al. 2024).
    "ptv3-s3dis-area5": WeightSpec(
        filename="ptv3-s3dis-area5.pth",
        url="https://huggingface.co/Pointcept/PointTransformerV3/resolve/main/s3dis-semseg-pt-v3m1-1-rpe-bs2x4-warmup/model_best.pth",
        sha256=None,  # Fill in once we've verified a download
        size_mb=160.0,
        description="PointTransformer V3, trained on S3DIS Area 1-4+6, validated on Area 5",
    ),
    # RandLA-Net — S3DIS pretrained from Open3D-ML release.
    "randla-s3dis": WeightSpec(
        filename="randlanet_s3dis.pth",
        url="https://storage.googleapis.com/open3d-releases-master/model-zoo/randlanet_s3dis_202201071330utc.pth",
        sha256=None,
        size_mb=12.0,
        description="RandLA-Net, Open3D-ML S3DIS pretrained checkpoint",
    ),
}


def models_dir() -> Path:
    """Return the platform-appropriate model cache directory."""
    override = os.environ.get("CLOUD2BIM_MODELS_DIR")
    if override:
        return Path(override).expanduser()

    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "cloud2bim" / "models"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "cloud2bim" / "models"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "cloud2bim" / "models"


def resolve_weights(name: str, explicit_path: Optional[Path] = None) -> Path:
    """Get a local path to weights, downloading them if needed.

    If ``explicit_path`` is given, it takes precedence and we only check
    that the file exists. Otherwise we look up ``name`` in ``REGISTRY``,
    use the cached file if present, or download from the registered URL.
    """
    if explicit_path is not None:
        explicit_path = Path(explicit_path)
        if not explicit_path.is_file():
            raise FileNotFoundError(
                f"Configured weights_path does not exist: {explicit_path}\n"
                f"Either point weights_path at an existing .pth file, or "
                f"unset it to use the auto-downloaded {name!r} checkpoint."
            )
        return explicit_path

    spec = REGISTRY.get(name)
    if spec is None:
        raise KeyError(
            f"Unknown weights {name!r}. Registered: {list(REGISTRY)}. "
            f"Add a WeightSpec to cloud2bim.segmentation.weights.REGISTRY "
            f"or set weights_path to point at your own checkpoint."
        )

    cache_dir = models_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / spec.filename
    if target.is_file():
        log.info("Using cached weights: %s", target)
        if spec.sha256:
            _verify_sha256_or_redownload(target, spec)
        return target

    if not spec.url:
        raise FileNotFoundError(
            f"No download URL for {name!r}. Place the checkpoint manually at:\n"
            f"  {target}\n"
            f"({spec.description})"
        )

    log.info("Downloading %s (~%.0f MB) → %s", name, spec.size_mb or 0, target)
    _download(spec.url, target, expected_size_mb=spec.size_mb)
    if spec.sha256:
        actual = _sha256(target)
        if actual != spec.sha256:
            target.unlink(missing_ok=True)
            raise IOError(
                f"Downloaded {name!r} but SHA-256 mismatch: got {actual}, "
                f"expected {spec.sha256}. File deleted; retry or update REGISTRY."
            )
    return target


# ── internal ──────────────────────────────────────────────────────────────────

def _download(url: str, target: Path, expected_size_mb: Optional[float] = None) -> None:
    """Stream-download to ``target.part`` then atomic rename. Resumes via Range."""
    try:
        import requests
    except ImportError as exc:
        raise ImportError(
            "requests not installed — required to auto-download weights. "
            "Either pip-install requests or set weights_path manually."
        ) from exc

    tmp = target.with_suffix(target.suffix + ".part")
    headers = {}
    resume_from = 0
    if tmp.exists():
        resume_from = tmp.stat().st_size
        headers["Range"] = f"bytes={resume_from}-"

    with requests.get(url, stream=True, headers=headers, timeout=60, allow_redirects=True) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0)) + resume_from
        mode = "ab" if resume_from > 0 else "wb"
        try:
            from tqdm import tqdm
            bar = tqdm(total=total, initial=resume_from, unit="B", unit_scale=True, desc=target.name)
        except ImportError:
            bar = None

        with open(tmp, mode) as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                fh.write(chunk)
                if bar is not None:
                    bar.update(len(chunk))
        if bar is not None:
            bar.close()

    tmp.replace(target)
    log.info("Downloaded %s (%d bytes)", target.name, target.stat().st_size)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_sha256_or_redownload(path: Path, spec: WeightSpec) -> None:
    actual = _sha256(path)
    if actual == spec.sha256:
        return
    log.warning(
        "Cached %s has SHA-256 %s but expected %s — re-downloading",
        path.name, actual, spec.sha256,
    )
    path.unlink(missing_ok=True)
    _download(spec.url, path, expected_size_mb=spec.size_mb)
