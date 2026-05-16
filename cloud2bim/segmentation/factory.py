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
        except RuntimeError as exc:
            # PTv3's _resolve_device raises RuntimeError when the host
            # can't run it: no CUDA, GPU too old (pre-Volta), or too
            # little VRAM. spconv has no functional CPU path so there's
            # no point pretending otherwise — drop in Passthrough so
            # the hybrid pipeline transparently switches to geometric.
            log.warning(
                "PTv3 cannot run on this host: %s\n"
                "Falling back to passthrough segmentation — hybrid "
                "pipeline will use the geometric extractor instead. "
                "For non-ML detection set pipeline_mode=geometric in "
                "the wizard to skip this step entirely.",
                exc,
            )
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
