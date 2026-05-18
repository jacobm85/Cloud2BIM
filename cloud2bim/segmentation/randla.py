"""RandLA-Net semantic segmentation backend (pure PyTorch).

Wraps the in-tree :mod:`cloud2bim.segmentation._randla_net` model. No
compiled extensions — works with any torch the rest of the project
ships, on CPU or GPU, no ABI matching against open3d-ml or pointcept.

Inference strategy for real-world clouds:
  1. Tile the cloud spatially so each tile has at most ``MAX_POINTS``
     points (40 960 matches the S3DIS recipe RandLA-Net was trained on).
  2. Pad the last tile up to ``MIN_POINTS`` if it's too small for the
     network's KNN graphs (need ≥ num_neighbors after every sub-sample).
  3. Tiles overlap slightly so border points get smoothed across tiles
     by logit averaging.
  4. Concatenate tile-level logits back to the full cloud and argmax.

Weights:
  The auto-downloaded ``randla-s3dis`` checkpoint comes from open3d-ml's
  release zoo, whose RandLANet module has slightly different parameter
  names than our in-tree implementation. We attempt a name-remap on
  load (same strategy PTv3 uses); when the match-rate is poor the model
  still runs but predictions are essentially random. The log line at
  load time tells you exactly how many tensors transferred.
  Set ``segmentation.weights_path`` to your own .pth if you've trained
  or fine-tuned the network against this code's parameter layout.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from cloud2bim.config import SegmentationConfig
from cloud2bim.logging import get_logger
from cloud2bim.segmentation.base import (
    S3DIS_LABELS,
    SEMANTICKITTI_LABELS,
    Segmenter,
    SemanticLabels,
    labels_for_dataset,
)
from cloud2bim.segmentation.weights import resolve_weights

log = get_logger(__name__)


def _emit_progress(stage: str, done: int, total: int, t_start: float) -> None:
    """Emit a machine-parseable progress line for the wizard front-end.

    Format: ``[PROGRESS] <stage> <done>/<total> eta=<seconds>``. The
    wizard's SSE log handler regex-parses this and updates a progress
    bar in the running-stage panel. Only logs every few tiles to keep
    log volume reasonable.
    """
    import time as _time
    # Throttle: only emit every 1 % or every tile, whichever is rarer,
    # plus the first and last.
    every = max(1, total // 100)
    if done != 1 and done != total and (done % every) != 0:
        return
    elapsed = max(0.001, _time.time() - t_start)
    rate = done / elapsed  # tiles per second
    remaining = max(0, total - done)
    eta_s = int(remaining / rate) if rate > 0 else -1
    log.info("[PROGRESS] %s %d/%d eta=%d", stage, done, total, eta_s)


def _shape_distribution(state: dict) -> list[str]:
    """Group tensors by shape and return ``"<shape>: <count> [examples]"`` strings.

    Used in checkpoint diagnostics. Helps us see at a glance whether the
    checkpoint and the model have the same set of tensor shapes — if
    they do, building an explicit name remap is straightforward; if not,
    the architectures genuinely differ and weight transfer isn't going
    to work without retraining.
    """
    by_shape: dict[tuple, list[str]] = {}
    for k, v in state.items():
        shape = tuple(getattr(v, "shape", ()))
        by_shape.setdefault(shape, []).append(k)
    ranked = sorted(by_shape.items(), key=lambda kv: -len(kv[1]))
    out = []
    for shape, keys in ranked:
        sample = ", ".join(keys[:2])
        if len(keys) > 2:
            sample += f", … (+{len(keys) - 2})"
        out.append(f"{tuple(int(s) for s in shape)}: {len(keys)}  [{sample}]")
    return out


# Architecture hyperparameters per dataset. Must match Open3D-ML's
# checkpoint configs (ml3d/configs/randlanet_*.yml) so weights load.
RANDLA_S3DIS_CFG = dict(
    num_classes=len(S3DIS_LABELS),
    in_channels=6,                              # XYZ + RGB
    dim_features=8,
    dim_output=(16, 64, 128, 256, 512),         # 5 layers
    num_neighbors=16,
    sub_sampling_ratio=(4, 4, 4, 4, 2),         # final layer halves
)
RANDLA_SEMANTICKITTI_CFG = dict(
    num_classes=len(SEMANTICKITTI_LABELS),
    in_channels=3,                              # XYZ only — no RGB in KITTI
    dim_features=8,
    dim_output=(16, 64, 128, 256),              # 4 layers
    num_neighbors=16,
    sub_sampling_ratio=(4, 4, 4, 4),
)


def _config_for_dataset(dataset: str) -> dict:
    if dataset == "semantickitti":
        return RANDLA_SEMANTICKITTI_CFG
    return RANDLA_S3DIS_CFG

# RandLA-Net was trained on 40 960-point patches (S3DIS recipe). Going
# above that costs memory linearly and degrades accuracy because the
# random sub-sampling has more candidates to drop. Going below 4 096
# starves the deepest layer of points.
MAX_POINTS_PER_TILE = 40_960
MIN_POINTS_PER_TILE = 4_096

# Tile overlap so border points get smoothed across neighbouring tiles.
TILE_OVERLAP_M = 0.5


class RandLASegmenter(Segmenter):
    """RandLA-Net via the in-tree pure-PyTorch implementation."""

    def __init__(self, cfg: SegmentationConfig):
        self.cfg = cfg
        self._model = None
        self._device = "cpu"  # set in _ensure_model based on availability
        self._labels = labels_for_dataset(cfg.dataset)
        self._arch_cfg = _config_for_dataset(cfg.dataset)
        self._weights_key = (
            "randla-semantickitti" if cfg.dataset == "semantickitti"
            else "randla-s3dis"
        )
        log.info(
            "RandLA-Net segmenter initialised (dataset=%s, %d classes, in_channels=%d, lazy load)",
            cfg.dataset, self._arch_cfg["num_classes"], self._arch_cfg["in_channels"],
        )

    def segment(
        self, points: np.ndarray, rgb: np.ndarray | None = None
    ) -> SemanticLabels:
        self._ensure_model()
        n = len(points)
        log.info("RandLA inference on %s points (rgb=%s, device=%s, dataset=%s)",
                 f"{n:,}", rgb is not None, self._device, self.cfg.dataset)
        feat = self._build_features(points, rgb)
        labels = self._infer_tiled(points.astype(np.float32), feat)
        unique, counts = np.unique(labels, return_counts=True)
        breakdown = ", ".join(
            f"{self._labels[i]}={c:,}" for i, c in zip(unique, counts)
            if 0 <= i < len(self._labels)
        )
        log.info("RandLA done — class breakdown: %s", breakdown)
        return SemanticLabels(label_ids=labels, label_names=self._labels)

    # ── model / weights setup ──────────────────────────────────────────────

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from cloud2bim.segmentation._randla_net import RandLANet
        except ImportError as exc:
            raise ImportError(
                f"RandLA-Net pure-PyTorch model unavailable ({exc}). "
                "torch and scipy are required."
            ) from exc

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("RandLA-Net device: %s", self._device)

        model = RandLANet(**self._arch_cfg)

        weights_path = resolve_weights(
            self._weights_key, explicit_path=self.cfg.weights_path,
        )
        log.info("Loading RandLA-Net weights: %s", weights_path)
        try:
            state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        except Exception as exc:
            log.warning(
                "torch.load weights_only=True failed (%s). Falling back "
                "to weights_only=False — safe only when the checkpoint "
                "is from a trusted source (the default URL is the "
                "official Open3D-ML release).", exc,
            )
            state = torch.load(str(weights_path), map_location="cpu", weights_only=False)

        # Unwrap common nest patterns ("state_dict", "model_state_dict").
        for key in ("state_dict", "model_state_dict", "model"):
            if isinstance(state, dict) and key in state and isinstance(state[key], dict):
                state = state[key]
                break

        # Best-effort prefix-stripping remap, same trick PTv3 uses for
        # Pointcept's wrapper layout. The open3d-ml RandLANet uses
        # parameter names that don't match ours exactly, so even after
        # finding the right prefix the partial-load count is the most
        # honest signal we have about whether the weights are usable.
        model_param_names = set(dict(model.named_parameters()).keys()) | \
                            set(dict(model.named_buffers()).keys())
        candidate_prefixes: set[str] = {""}
        for k in state:
            segs = k.split(".")
            for i in range(1, min(4, len(segs))):
                candidate_prefixes.add(".".join(segs[:i]) + ".")

        best_prefix = ""
        best_count = 0
        best_state: dict = {}
        for prefix in candidate_prefixes:
            pfx_len = len(prefix)
            candidate = {
                k[pfx_len:]: v for k, v in state.items()
                if (not prefix) or k.startswith(prefix)
            }
            n_match = sum(1 for name in candidate if name in model_param_names)
            if n_match > best_count:
                best_count, best_prefix, best_state = n_match, prefix, candidate

        if best_count == 0:
            # Dump enough structure that we can build a per-tensor remap
            # without round-tripping the user through another debug run.
            ckpt_keys = sorted(state.keys())
            model_keys = sorted(dict(model.named_parameters()).keys())
            log.warning(
                "RandLA-Net checkpoint had 0 matching parameter names "
                "after prefix scanning. Model would run with random "
                "weights — aborting to avoid silently producing noise."
            )
            log.warning(
                "Checkpoint has %d tensors; first 30 keys:\n  %s",
                len(ckpt_keys), "\n  ".join(ckpt_keys[:30]),
            )
            try:
                ckpt_shape_summary = _shape_distribution(state)
                log.warning(
                    "Checkpoint shape distribution (top 15):\n  %s",
                    "\n  ".join(ckpt_shape_summary[:15]),
                )
            except Exception:
                pass
            log.warning(
                "Model expects %d tensors; first 30 keys:\n  %s",
                len(model_keys), "\n  ".join(model_keys[:30]),
            )
            try:
                model_shape_summary = _shape_distribution(
                    {n: p for n, p in model.named_parameters()}
                )
                log.warning(
                    "Model shape distribution (top 15):\n  %s",
                    "\n  ".join(model_shape_summary[:15]),
                )
            except Exception:
                pass
            raise RuntimeError(
                "RandLA-Net checkpoint structure doesn't match this "
                "code's parameter layout. See the warnings above for "
                "the dumped key/shape inventory — paste them when "
                "asking for a remap. Workarounds: set "
                "segmentation.weights_path to a checkpoint trained "
                "against this implementation, or switch backend to "
                "ptv3 / none."
            )
        else:
            missing, unexpected = model.load_state_dict(best_state, strict=False)
            total_params = len(model_param_names)
            log.info(
                "RandLA-Net weights: prefix='%s', %d/%d params transferred "
                "(%d missing, %d unexpected)",
                best_prefix or "<root>", best_count, total_params,
                len(missing), len(unexpected),
            )
            if best_count < total_params * 0.5:
                log.warning(
                    "Fewer than half of the RandLA-Net parameters "
                    "matched the checkpoint. Expect degraded results — "
                    "the rest were random-initialised. Train against "
                    "this code's parameter layout for production use."
                )

        model.to(self._device).eval()
        self._model = model

    # ── feature construction ───────────────────────────────────────────────

    def _build_features(self, points: np.ndarray, rgb: np.ndarray | None) -> np.ndarray:
        """Build the per-point feature side of the model input.

        For S3DIS (in_channels=6): returns (N, 3) RGB only — XYZ is
        normalised per-tile in ``_infer_single`` and concatenated there.
        For SemanticKITTI (in_channels=3): the model takes XYZ only.
        We return a zero-channel placeholder; the actual XYZ comes from
        per-tile normalisation in ``_infer_single``.
        """
        from cloud2bim.segmentation.ptv3 import SYNTHETIC_RGB, _normalise_rgb
        if self._arch_cfg["in_channels"] == 3:
            # KITTI variant: only XYZ goes into the model. Return an
            # empty (N, 0) array so _infer_tiled's per-tile masking
            # works uniformly with the S3DIS path.
            return np.empty((len(points), 0), dtype=np.float32)
        if rgb is None:
            return np.broadcast_to(SYNTHETIC_RGB, points.shape).astype(np.float32)
        return _normalise_rgb(rgb).astype(np.float32)

    # ── tiled inference ────────────────────────────────────────────────────

    def _infer_tiled(self, points: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Run the network in spatial XY tiles; vote per-point logits.

        Returns int32 labels for every input point.
        """
        n = len(points)
        n_classes = self._arch_cfg["num_classes"]
        logits = np.zeros((n, n_classes), dtype=np.float32)
        counts = np.zeros(n, dtype=np.int32)

        if n <= MAX_POINTS_PER_TILE:
            tile_logits = self._infer_single(points, features)
            return tile_logits.argmax(axis=1).astype(np.int32)

        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        area_xy = max(
            (bbox_max[0] - bbox_min[0]) * (bbox_max[1] - bbox_min[1]),
            1e-6,
        )
        density = n / area_xy
        cell_area = MAX_POINTS_PER_TILE / max(density, 1e-6)
        cell_side = max(float(np.sqrt(cell_area)), 2.0)
        # Count tiles up front so we can emit a "X/Y" progress line per
        # tile — wizard front-end parses these to drive a progress bar.
        nx = max(1, int(np.ceil((bbox_max[0] - bbox_min[0]) / cell_side)))
        ny = max(1, int(np.ceil((bbox_max[1] - bbox_min[1]) / cell_side)))
        total_tiles = nx * ny
        log.info(
            "RandLA tiling: %s points → %d tiles of ~%dm (%dx%d, overlap %.1fm)",
            f"{n:,}", total_tiles, int(cell_side), nx, ny, TILE_OVERLAP_M,
        )

        import time as _time
        t_start = _time.time()
        for tile_idx, tile_mask in enumerate(
            self._tile_masks(points, bbox_min, bbox_max, cell_side), start=1
        ):
            tile_n = int(tile_mask.sum())
            if tile_n == 0:
                _emit_progress("segment", tile_idx, total_tiles, t_start)
                continue
            tile_pts = points[tile_mask]
            tile_feat = features[tile_mask]

            if tile_n < MIN_POINTS_PER_TILE:
                # Don't waste a forward pass on a too-small tile; let the
                # neighbouring tiles' overlap cover these points. If a
                # point is in zero non-tiny tiles we'll fill it later.
                _emit_progress("segment", tile_idx, total_tiles, t_start)
                continue
            if tile_n > MAX_POINTS_PER_TILE:
                # Random subsample to MAX_POINTS within the tile; the
                # skipped points are still covered by other tiles or
                # the fill-in step below.
                sel = np.random.choice(tile_n, MAX_POINTS_PER_TILE, replace=False)
                sub_pts = tile_pts[sel]
                sub_feat = tile_feat[sel]
                tile_logits_sub = self._infer_single(sub_pts, sub_feat)
                # Scatter back to the tile-mask points by 1-NN from sub to full tile.
                from scipy.spatial import cKDTree
                tree = cKDTree(sub_pts)
                _, nn_idx = tree.query(tile_pts, k=1, workers=-1)
                tile_logits = tile_logits_sub[nn_idx]
            else:
                tile_logits = self._infer_single(tile_pts, tile_feat)

            logits[tile_mask] += tile_logits
            counts[tile_mask] += 1
            _emit_progress("segment", tile_idx, total_tiles, t_start)

        uncovered = counts == 0
        if uncovered.any():
            # Tiles too small to run got skipped — propagate from nearest
            # covered point so every input still gets a label.
            log.warning(
                "RandLA tiling left %d points uncovered (in undersized "
                "tiles); filling from nearest covered point.",
                int(uncovered.sum()),
            )
            covered_idx = np.where(~uncovered)[0]
            if len(covered_idx):
                from scipy.spatial import cKDTree
                tree = cKDTree(points[covered_idx])
                _, nn = tree.query(points[uncovered], k=1, workers=-1)
                logits[uncovered] = logits[covered_idx[nn]]
                counts[uncovered] = 1
            else:
                logits[uncovered] = 1.0 / n_classes
                counts[uncovered] = 1

        logits /= counts[:, None]
        return logits.argmax(axis=1).astype(np.int32)

    @staticmethod
    def _tile_masks(
        points: np.ndarray,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        cell_side: float,
    ) -> Iterator[np.ndarray]:
        """Yield boolean masks of points inside each XY tile (with overlap)."""
        nx = max(1, int(np.ceil((bbox_max[0] - bbox_min[0]) / cell_side)))
        ny = max(1, int(np.ceil((bbox_max[1] - bbox_min[1]) / cell_side)))
        for ix in range(nx):
            for iy in range(ny):
                x_lo = bbox_min[0] + ix * cell_side - TILE_OVERLAP_M
                x_hi = bbox_min[0] + (ix + 1) * cell_side + TILE_OVERLAP_M
                y_lo = bbox_min[1] + iy * cell_side - TILE_OVERLAP_M
                y_hi = bbox_min[1] + (iy + 1) * cell_side + TILE_OVERLAP_M
                yield (
                    (points[:, 0] >= x_lo) & (points[:, 0] < x_hi) &
                    (points[:, 1] >= y_lo) & (points[:, 1] < y_hi)
                )

    def _infer_single(self, points: np.ndarray, extra_features: np.ndarray) -> np.ndarray:
        """Forward pass on one tile; returns (N, num_classes) float32 logits.

        ``extra_features`` is the non-XYZ part of the model input — RGB
        (N, 3) for S3DIS, empty (N, 0) for SemanticKITTI. XYZ-in-features
        is built per-tile here via centre + max-abs scale so fc0 sees
        consistent magnitudes regardless of building scale.
        """
        import torch

        num_classes = self._arch_cfg["num_classes"]
        if len(points) < self._arch_cfg["num_neighbors"]:
            return np.full(
                (len(points), num_classes),
                1.0 / num_classes,
                dtype=np.float32,
            )

        # Normalise XYZ per tile. Geometric `points` (used for KNN) stays
        # in world units so the encoder's receptive field matches training.
        centre = points.mean(axis=0)
        norm_xyz = (points - centre).astype(np.float32)
        scale = float(np.abs(norm_xyz).max())
        if scale > 0:
            norm_xyz /= scale  # roughly [-1, 1]

        if extra_features.shape[1] > 0:
            feat = np.concatenate([norm_xyz, extra_features.astype(np.float32)], axis=1)
        else:
            feat = norm_xyz

        device = torch.device(self._device)
        inputs = self._model.prepare_inputs(points, feat, device=device)
        with torch.no_grad():
            logits = self._model(inputs)  # (1, num_classes, N)
        out = logits.squeeze(0).t().detach().cpu().numpy()
        return out.astype(np.float32)
