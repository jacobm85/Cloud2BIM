"""RandLA-Net backend via Open3D-ML — fallback when PTv3 is unavailable.

Lighter weight than PTv3 (no spconv, no pointcept), still effective on
indoor S3DIS-style data.
"""
from __future__ import annotations

import numpy as np

from cloud2bim.config import SegmentationConfig
from cloud2bim.logging import get_logger
from cloud2bim.segmentation.base import S3DIS_LABELS, Segmenter, SemanticLabels
from cloud2bim.segmentation.weights import resolve_weights

log = get_logger(__name__)


# Architecture hyperparameters from Open3D-ML's bundled randlanet_s3dis.yml.
# Inlined so we don't depend on the YAML file path inside the open3d
# package, which has been a moving target across releases.
RANDLA_S3DIS_MODEL_CFG = dict(
    name="RandLANet",
    num_classes=len(S3DIS_LABELS),
    num_points=40960,
    in_channels=6,          # XYZ + RGB
    dim_features=8,
    dim_output=[16, 64, 128, 256],
    num_neighbors=16,
    sub_sampling_ratio=[4, 4, 4, 4],
    grid_size=0.04,
)


class RandLASegmenter(Segmenter):
    """RandLA-Net via Open3D-ML (torch backend)."""

    DEFAULT_WEIGHTS_KEY = "randla-s3dis"

    def __init__(self, cfg: SegmentationConfig):
        self.cfg = cfg
        self._model = None
        log.info("RandLA-Net segmenter initialised")

    def segment(
        self, points: np.ndarray, rgb: np.ndarray | None = None
    ) -> SemanticLabels:
        self._ensure_model()
        log.info("RandLA inference on %s points (rgb=%s)", f"{len(points):,}", rgb is not None)
        labels = self._infer(points, rgb)
        return SemanticLabels(label_ids=labels.astype(np.int32), label_names=S3DIS_LABELS)

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from open3d.ml.torch.models import RandLANet
        except ImportError as exc:
            raise ImportError(
                "Open3D-ML (torch backend) not installed. "
                "Run: pip install open3d torch tensorboard"
            ) from exc

        self._model = RandLANet(**RANDLA_S3DIS_MODEL_CFG)
        weights = resolve_weights(self.DEFAULT_WEIGHTS_KEY, explicit_path=self.cfg.weights_path)
        import torch
        # weights_only=True blocks pickle code-execution at load time.
        # See ptv3.py for the rationale; fall back with a warning if
        # the checkpoint contains non-allowlisted globals.
        try:
            state = torch.load(str(weights), map_location="cpu", weights_only=True)
        except Exception as exc:
            log.warning(
                "torch.load weights_only=True failed (%s). Falling back "
                "to weights_only=False — safe only because RandLA-Net "
                "weights ship from Open3D-ML's official release.",
                exc,
            )
            state = torch.load(str(weights), map_location="cpu", weights_only=False)
        self._model.load_state_dict(state.get("model_state_dict", state), strict=False)
        self._model.eval()

    def _infer(self, points: np.ndarray, rgb: np.ndarray | None) -> np.ndarray:
        import torch
        # RandLA-Net's S3DIS recipe uses 6 input channels: XYZ + RGB.
        # If the cloud has no colour we fall back to mid-grey, matching
        # PTv3's behaviour (slight accuracy loss vs real RGB).
        from cloud2bim.segmentation.ptv3 import SYNTHETIC_RGB, _normalise_rgb
        if rgb is None:
            rgb_f = np.broadcast_to(SYNTHETIC_RGB, points.shape).astype(np.float32)
        else:
            rgb_f = _normalise_rgb(rgb)
        feat = np.concatenate([points.astype(np.float32), rgb_f], axis=1)
        with torch.no_grad():
            inputs = {
                "point": torch.from_numpy(points.astype(np.float32)).unsqueeze(0),
                "feat": torch.from_numpy(feat).unsqueeze(0),
            }
            out = self._model(inputs)
            return out.argmax(dim=-1).squeeze(0).cpu().numpy()
