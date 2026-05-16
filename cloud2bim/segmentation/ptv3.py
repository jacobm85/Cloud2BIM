"""PointTransformer V3 backend.

Wraps Pointcept's PTv3 implementation. Voxelises the input cloud, runs
inference per voxel batch, then nearest-neighbour-transfers labels back
to the original points.

Heavy dependencies (torch, spconv, pointops) are imported lazily so the
rest of the package works without them installed.

To install (production server):
    pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
    pip install spconv-cu118
    pip install git+https://github.com/Pointcept/Pointcept.git
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from cloud2bim.config import SegmentationConfig
from cloud2bim.logging import get_logger
from cloud2bim.segmentation.base import S3DIS_LABELS, Segmenter, SemanticLabels
from cloud2bim.segmentation.weights import resolve_weights

log = get_logger(__name__)


class PTv3Segmenter(Segmenter):
    """PointTransformer V3 via Pointcept."""

    DEFAULT_WEIGHTS_KEY = "ptv3-s3dis-area5"

    def __init__(self, cfg: SegmentationConfig):
        self.cfg = cfg
        self._model = None  # lazy-init on first segment() call
        self._device = self._resolve_device(cfg.device)
        log.info("PTv3 segmenter initialised (device=%s, voxel=%.3f m)", self._device, cfg.ml_voxel_size)

    def segment(self, points: np.ndarray) -> SemanticLabels:
        self._ensure_model()
        log.info("PTv3 inference on %s points", f"{len(points):,}")

        voxel_pts, voxel_idx = self._voxelize(points, self.cfg.ml_voxel_size)
        voxel_logits = self._infer_voxels(voxel_pts)
        voxel_labels = voxel_logits.argmax(axis=1).astype(np.int32)

        # Transfer voxel labels back to all points
        labels = voxel_labels[voxel_idx]
        log.info("PTv3 done: %d unique labels", len(np.unique(labels)))
        return SemanticLabels(label_ids=labels, label_names=S3DIS_LABELS)

    # ── internal ────────────────────────────────────────────────────────

    def _resolve_device(self, requested: str) -> str:
        if requested != "auto":
            return requested
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # noqa: F401  (verifies presence)
            from pointcept.models.point_transformer_v3 import PointTransformerV3
        except ImportError as exc:
            raise ImportError(
                "Pointcept / PointTransformerV3 not installed. "
                "Run: pip install torch spconv-cu118 git+https://github.com/Pointcept/Pointcept.git"
            ) from exc

        weights_path = self._resolve_weights_path()
        log.info("Loading PTv3 weights: %s", weights_path)
        self._model = PointTransformerV3(
            in_channels=3,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=(2, 2, 2, 2),
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=(32, 64, 128, 256, 512),
            enc_num_head=(2, 4, 8, 16, 32),
            enc_patch_size=(1024,) * 5,
            dec_depths=(2, 2, 2, 2),
            dec_channels=(64, 64, 128, 256),
            dec_num_head=(4, 4, 8, 16),
            dec_patch_size=(1024,) * 4,
            num_classes=len(S3DIS_LABELS),
        )
        import torch
        state = torch.load(str(weights_path), map_location=self._device)
        self._model.load_state_dict(state.get("state_dict", state), strict=False)
        self._model.to(self._device).eval()

    def _resolve_weights_path(self) -> Path:
        return resolve_weights(self.DEFAULT_WEIGHTS_KEY, explicit_path=self.cfg.weights_path)

    @staticmethod
    def _voxelize(points: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
        """Voxel grid downsampling. Returns (voxel_centers, point_to_voxel_idx)."""
        coords = np.floor(points / voxel_size).astype(np.int64)
        # Hash voxel coords
        keys = coords[:, 0] * 73856093 ^ coords[:, 1] * 19349663 ^ coords[:, 2] * 83492791
        unique_keys, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        # Voxel centers (mean of contained points)
        voxel_pts = np.zeros((len(unique_keys), 3))
        np.add.at(voxel_pts, inverse, points)
        voxel_pts /= counts[:, None]
        return voxel_pts, inverse

    def _infer_voxels(self, voxel_pts: np.ndarray) -> np.ndarray:
        """Run PTv3 forward pass on voxel centres. Returns (n_voxels, n_classes)."""
        import torch
        with torch.no_grad():
            data_dict = {
                "coord": torch.from_numpy(voxel_pts).float().to(self._device),
                "feat": torch.from_numpy(voxel_pts).float().to(self._device),
                "grid_coord": torch.from_numpy(
                    np.floor(voxel_pts / self.cfg.ml_voxel_size).astype(np.int64)
                ).to(self._device),
                "offset": torch.tensor([len(voxel_pts)], device=self._device),
            }
            out = self._model(data_dict)
            return out.cpu().numpy()
