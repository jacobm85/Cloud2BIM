"""Backend dispatch — picks Segmenter implementation based on config."""
from __future__ import annotations

from cloud2bim.config import SegmentationConfig
from cloud2bim.logging import get_logger
from cloud2bim.segmentation.base import PassthroughSegmenter, Segmenter

log = get_logger(__name__)


def create_segmenter(cfg: SegmentationConfig) -> Segmenter:
    """Instantiate the configured segmentation backend.

    Falls back to PassthroughSegmenter if ML dependencies are missing or
    if ``cfg.enabled = false``. Logs a warning so the user knows.
    """
    if not cfg.enabled or cfg.backend == "none":
        log.info("Semantic segmentation disabled (passthrough mode)")
        return PassthroughSegmenter()

    if cfg.backend == "ptv3":
        try:
            from cloud2bim.segmentation.ptv3 import PTv3Segmenter
            return PTv3Segmenter(cfg)
        except ImportError as exc:
            log.warning("PTv3 unavailable (%s) — falling back to passthrough", exc)
            return PassthroughSegmenter()

    if cfg.backend == "randla":
        try:
            from cloud2bim.segmentation.randla import RandLASegmenter
            return RandLASegmenter(cfg)
        except ImportError as exc:
            log.warning("RandLA-Net unavailable (%s) — falling back to passthrough", exc)
            return PassthroughSegmenter()

    log.warning("Unknown backend %r — using passthrough", cfg.backend)
    return PassthroughSegmenter()
