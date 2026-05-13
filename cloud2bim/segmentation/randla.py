"""RandLA-Net backend via Open3D-ML — fallback when PTv3 is unavailable.

Lighter weight than PTv3 (no spconv, no pointcept), still effective on
indoor S3DIS-style data.
"""
from __future__ import annotations

import numpy as np

from cloud2bim.config import SegmentationConfig
from cloud2bim.logging import get_logger
from cloud2bim.segmentation.base import S3DIS_LABELS, Segmenter, SemanticLabels

log = get_logger(__name__)


class RandLASegmenter(Segmenter):
    """RandLA-Net via Open3D-ML (torch backend)."""

    def __init__(self, cfg: SegmentationConfig):
        self.cfg = cfg
        self._model = None
        log.info("RandLA-Net segmenter initialised")

    def segment(self, points: np.ndarray) -> SemanticLabels:
        self._ensure_model()
        log.info("RandLA inference on %s points", f"{len(points):,}")
        labels = self._infer(points)
        return SemanticLabels(label_ids=labels.astype(np.int32), label_names=S3DIS_LABELS)

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import open3d.ml.torch as ml3d
            from open3d.ml.torch.models import RandLANet
        except ImportError as exc:
            raise ImportError(
                "Open3D-ML (torch backend) not installed. "
                "Run: pip install open3d torch tensorboard"
            ) from exc

        cfg = ml3d.utils.Config.load_from_file(
            "open3d.ml.configs.randlanet_s3dis.yml"
        )
        self._model = RandLANet(**cfg.model)
        if self.cfg.weights_path is not None:
            import torch
            state = torch.load(str(self.cfg.weights_path), map_location="cpu")
            self._model.load_state_dict(state.get("model_state_dict", state), strict=False)
        self._model.eval()

    def _infer(self, points: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            inputs = {
                "point": torch.from_numpy(points).float().unsqueeze(0),
                "feat": torch.from_numpy(points).float().unsqueeze(0),
            }
            out = self._model(inputs)
            return out.argmax(dim=-1).squeeze(0).cpu().numpy()
