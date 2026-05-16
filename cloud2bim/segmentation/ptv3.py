"""PointTransformer V3 backend.

Wraps Pointcept's PTv3 implementation. Voxelises the input cloud,
optionally chunks it spatially, runs inference per chunk, then
nearest-neighbour-transfers labels back to the original points.

Heavy dependencies (torch, spconv, pointops) are imported lazily so the
rest of the package works without them installed.

To install (production server):
    pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
    pip install spconv-cu118
    pip install git+https://github.com/Pointcept/Pointcept.git
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from cloud2bim.config import SegmentationConfig
from cloud2bim.logging import get_logger
from cloud2bim.segmentation.base import S3DIS_LABELS, Segmenter, SemanticLabels
from cloud2bim.segmentation.weights import resolve_weights

log = get_logger(__name__)


# The S3DIS PTv3 checkpoints we ship are trained with 6 input channels:
# normalised XYZ (centred per scene, scaled to [-1, 1]) + normalised RGB
# (scaled to [0, 1]). Don't change this without retraining.
PTV3_IN_CHANNELS = 6

# Default RGB used when the input cloud has no colour. Mid-grey works
# better than zeros because the S3DIS training set rarely sees pitch-black
# surfaces — feeding zeros pushes activations into an under-represented
# region of feature space.
SYNTHETIC_RGB = np.array([0.5, 0.5, 0.5], dtype=np.float32)


class PTv3Segmenter(Segmenter):
    """PointTransformer V3 via Pointcept."""

    DEFAULT_WEIGHTS_KEY = "ptv3-s3dis-area5"

    def __init__(self, cfg: SegmentationConfig):
        self.cfg = cfg
        self._model = None  # lazy-init on first segment() call
        self._device = self._resolve_device(cfg.device)
        log.info("PTv3 segmenter initialised (device=%s, voxel=%.3f m)", self._device, cfg.ml_voxel_size)

    def segment(
        self, points: np.ndarray, rgb: np.ndarray | None = None
    ) -> SemanticLabels:
        self._ensure_model()
        n = len(points)
        log.info("PTv3 inference on %s points (rgb=%s)", f"{n:,}", rgb is not None)

        # 1. Voxelise — collapse high-density points into one voxel each.
        voxel_xyz, voxel_rgb, voxel_idx = self._voxelize(
            points, rgb, self.cfg.ml_voxel_size
        )
        log.info("PTv3 voxels: %s (1 voxel per %.1f points)",
                 f"{len(voxel_xyz):,}", n / max(1, len(voxel_xyz)))

        # 2. Inference, chunked spatially to respect GPU memory.
        voxel_logits = self._infer_chunked(
            voxel_xyz, voxel_rgb,
            max_voxels=self.cfg.max_voxels_per_batch,
        )
        voxel_labels = voxel_logits.argmax(axis=1).astype(np.int32)

        # 3. NN-transfer labels back to every original point.
        labels = voxel_labels[voxel_idx]
        unique, counts = np.unique(labels, return_counts=True)
        breakdown = ", ".join(
            f"{S3DIS_LABELS[i]}={c:,}" for i, c in zip(unique, counts)
            if i < len(S3DIS_LABELS)
        )
        log.info("PTv3 done — class breakdown: %s", breakdown)
        return SemanticLabels(label_ids=labels, label_names=S3DIS_LABELS)

    # ── model / device setup ────────────────────────────────────────────

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
            in_channels=PTV3_IN_CHANNELS,
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
        # Some checkpoints wrap the model inside "state_dict" or "model".
        for key in ("state_dict", "model", "model_state_dict"):
            if isinstance(state, dict) and key in state and isinstance(state[key], dict):
                state = state[key]
                break
        self._model.load_state_dict(state, strict=False)
        self._model.to(self._device).eval()

    def _resolve_weights_path(self) -> Path:
        return resolve_weights(self.DEFAULT_WEIGHTS_KEY, explicit_path=self.cfg.weights_path)

    # ── voxelisation ────────────────────────────────────────────────────

    @staticmethod
    def _voxelize(
        points: np.ndarray,
        rgb: np.ndarray | None,
        voxel_size: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Voxel-grid downsampling. Aggregates XYZ + RGB by mean per voxel.

        Returns (voxel_xyz, voxel_rgb, point_to_voxel_idx) where
        voxel_rgb is float32 in [0, 1] and voxel_xyz preserves world metres.
        """
        coords = np.floor(points / voxel_size).astype(np.int64)
        # Hash voxel coords to a single int key so np.unique can group them.
        keys = (coords[:, 0] * np.int64(73856093)) ^ (coords[:, 1] * np.int64(19349663)) ^ (coords[:, 2] * np.int64(83492791))
        _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        n_voxels = len(counts)

        voxel_xyz = np.zeros((n_voxels, 3), dtype=np.float64)
        np.add.at(voxel_xyz, inverse, points)
        voxel_xyz /= counts[:, None]

        if rgb is not None:
            rgb_f = _normalise_rgb(rgb)
            voxel_rgb = np.zeros((n_voxels, 3), dtype=np.float32)
            np.add.at(voxel_rgb, inverse, rgb_f)
            voxel_rgb /= counts[:, None].astype(np.float32)
        else:
            voxel_rgb = np.broadcast_to(SYNTHETIC_RGB, (n_voxels, 3)).copy()

        return voxel_xyz.astype(np.float32), voxel_rgb, inverse

    # ── chunked inference ──────────────────────────────────────────────

    def _infer_chunked(
        self,
        voxel_xyz: np.ndarray,
        voxel_rgb: np.ndarray,
        max_voxels: int,
    ) -> np.ndarray:
        """Run PTv3 in spatial chunks small enough to fit on the GPU.

        When the cloud fits within ``max_voxels`` we do a single forward
        pass. Otherwise we tile the XY footprint into a grid where every
        cell holds roughly ``max_voxels`` voxels, then process each cell.
        S3DIS-trained models are tuned for room-sized inputs (~5-10 m
        across) so this also matches the training distribution better
        than feeding a whole building at once.
        """
        n_classes = len(S3DIS_LABELS)
        n = len(voxel_xyz)

        if n <= max_voxels:
            return self._infer_single(voxel_xyz, voxel_rgb)

        # Decide grid cell size so each cell holds ~max_voxels.
        bbox_min = voxel_xyz.min(axis=0)
        bbox_max = voxel_xyz.max(axis=0)
        area_xy = (bbox_max[0] - bbox_min[0]) * (bbox_max[1] - bbox_min[1])
        # voxels-per-m² in XY:
        density = n / max(area_xy, 1e-6)
        cell_area = max_voxels / max(density, 1e-6)
        cell_side = float(np.sqrt(cell_area))
        cell_side = max(cell_side, 2.0)  # don't go below 2 m
        n_cells = int(np.ceil((bbox_max[0] - bbox_min[0]) / cell_side)) * \
                  int(np.ceil((bbox_max[1] - bbox_min[1]) / cell_side))
        log.info(
            "PTv3 chunking: %d voxels > batch %d → %d-cell %.1fm grid",
            n, max_voxels, n_cells, cell_side,
        )

        logits = np.zeros((n, n_classes), dtype=np.float32)
        counts = np.zeros(n, dtype=np.int32)

        for cell_mask in self._spatial_cells(voxel_xyz, bbox_min, bbox_max, cell_side):
            if not cell_mask.any():
                continue
            cell_logits = self._infer_single(voxel_xyz[cell_mask], voxel_rgb[cell_mask])
            logits[cell_mask] += cell_logits
            counts[cell_mask] += 1

        # Voxels never covered (shouldn't happen) get a uniform prediction.
        uncovered = counts == 0
        if uncovered.any():
            log.warning("PTv3 chunking left %d voxels uncovered", int(uncovered.sum()))
            logits[uncovered] = 1.0 / n_classes
            counts[uncovered] = 1
        logits /= counts[:, None]
        return logits

    @staticmethod
    def _spatial_cells(
        voxel_xyz: np.ndarray,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        cell_side: float,
        overlap: float = 0.5,
    ) -> Iterator[np.ndarray]:
        """Yield boolean masks of voxels inside each XY tile.

        Tiles overlap by ``overlap`` metres on each side so voxels near a
        seam vote in both neighbouring chunks — _infer_chunked averages
        their logits, smoothing out any chunk-boundary artefacts.
        """
        nx = max(1, int(np.ceil((bbox_max[0] - bbox_min[0]) / cell_side)))
        ny = max(1, int(np.ceil((bbox_max[1] - bbox_min[1]) / cell_side)))
        for ix in range(nx):
            for iy in range(ny):
                x_lo = bbox_min[0] + ix * cell_side - overlap
                x_hi = bbox_min[0] + (ix + 1) * cell_side + overlap
                y_lo = bbox_min[1] + iy * cell_side - overlap
                y_hi = bbox_min[1] + (iy + 1) * cell_side + overlap
                mask = (
                    (voxel_xyz[:, 0] >= x_lo) & (voxel_xyz[:, 0] < x_hi) &
                    (voxel_xyz[:, 1] >= y_lo) & (voxel_xyz[:, 1] < y_hi)
                )
                yield mask

    def _infer_single(
        self, voxel_xyz: np.ndarray, voxel_rgb: np.ndarray
    ) -> np.ndarray:
        """Forward pass on a single chunk of voxels. Returns (n, n_classes) logits."""
        import torch

        # Normalise XYZ for the model: centre on the chunk, scale by max axis.
        centre = voxel_xyz.mean(axis=0)
        norm_xyz = voxel_xyz - centre
        scale = float(np.abs(norm_xyz).max())
        if scale > 0:
            norm_xyz = norm_xyz / scale  # → roughly [-1, 1]

        feat = np.concatenate([norm_xyz, voxel_rgb], axis=1).astype(np.float32)
        grid_coord = np.floor(voxel_xyz / self.cfg.ml_voxel_size).astype(np.int64)
        # Reset grid to start at origin per chunk (PTv3 internals assume non-negative).
        grid_coord -= grid_coord.min(axis=0)

        with torch.no_grad():
            data_dict = {
                "coord": torch.from_numpy(voxel_xyz.astype(np.float32)).to(self._device),
                "feat": torch.from_numpy(feat).to(self._device),
                "grid_coord": torch.from_numpy(grid_coord).to(self._device),
                # Pointcept expects ``offset`` to be the cumulative point
                # count per batch element. Single chunk → single entry.
                "offset": torch.tensor([len(voxel_xyz)], dtype=torch.long, device=self._device),
            }
            out = self._model(data_dict)
            if hasattr(out, "feat"):
                out = out.feat  # some Pointcept versions wrap output in a Point dict
            return out.detach().cpu().numpy().astype(np.float32)


def _normalise_rgb(rgb: np.ndarray) -> np.ndarray:
    """Coerce RGB into float32 in [0, 1].

    Accepts uint8 (0-255), uint16 (0-65535) — common in LAS files — and
    float arrays in either [0, 1] or [0, 255]. Falls back to dividing by
    the max sample so weird ranges still produce something sensible.
    """
    if rgb.dtype == np.uint8:
        return rgb.astype(np.float32) / 255.0
    if rgb.dtype == np.uint16:
        return rgb.astype(np.float32) / 65535.0
    rgb_f = rgb.astype(np.float32)
    peak = float(rgb_f.max()) if rgb_f.size else 1.0
    if peak <= 1.5:
        return np.clip(rgb_f, 0.0, 1.0)
    if peak <= 260.0:
        return np.clip(rgb_f / 255.0, 0.0, 1.0)
    return np.clip(rgb_f / peak, 0.0, 1.0)
