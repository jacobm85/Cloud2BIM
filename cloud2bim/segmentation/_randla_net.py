"""Pure-PyTorch RandLA-Net architecture (no compiled extensions).

Based on Hu et al. 2020 — "RandLA-Net: Efficient Semantic Segmentation
of Large-Scale Point Clouds" (CVPR). Re-implemented from the published
description rather than vendored from Open3D-ML so this module has zero
build-time dependency on a particular torch C++ ABI. Drop-in for the
sparse-convolution-free path on machines where PTv3 won't run.

Architecture (matching the S3DIS recipe):
  * Input: per-point (xyz, rgb) → 6 input channels
  * Linear projection → 8 features
  * Encoder: 4 levels of (DilatedResidualBlock + RandomSampling)
      channels = 16 → 64 → 128 → 256
      sub-sampling ratio = 4 at every level
      neighbours K = 16 at every level
  * Bottleneck: 1×1 conv on the most-downsampled features
  * Decoder: 4 levels of (NN-upsample + skip + 1×1 conv)
  * Head: 2 × (1×1 conv + dropout) → linear(num_classes)

Differences from the canonical TF / open3d-ml implementations that
matter for weight portability:
  * `nn.Conv2d` with kernel=1 is used for every per-point MLP (matches
    open3d-ml's BatchNorm2d-wrapped convs). State-dict layout uses the
    same `module.weight / module.bias / module.bn.{weight,bias,running_*}`
    triple per conv, which gives a reasonable chance of partial weight
    transfer from the open3d-ml checkpoint via name remapping.
  * KNN is computed in numpy with `scipy.cKDTree` (CPU) — fast enough
    for the per-level point counts (≤ 40 960 / 4ⁿ) and avoids any
    torch_cluster build dependency. Indices are stitched in on the GPU
    side only as int64 tensors.

This module is intentionally self-contained — no imports from
cloud2bim.* — so it can be unit-tested in isolation and so any future
refactor of the Segmenter interface doesn't churn it.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Geometric helpers ─────────────────────────────────────────────────────────


def knn_indices(xyz: np.ndarray, k: int) -> np.ndarray:
    """K-nearest-neighbour indices for every point in ``xyz``.

    Pure-CPU via scipy's cKDTree — RandLA-Net works on heavily
    downsampled point sets (≤ 40 960 / 4ⁿ per layer) where this is
    much faster than building a torch_cluster CUDA index, and it has
    no compile-time dependency.

    Returns (N, K) int32 array of indices into ``xyz``.
    """
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise ImportError(
            "scipy is required for RandLA-Net's KNN. Install scipy."
        ) from exc
    tree = cKDTree(xyz)
    # query returns (distances, indices); we keep only indices
    _, idx = tree.query(xyz, k=k, workers=-1)
    if idx.ndim == 1:  # k=1 → squeezed
        idx = idx[:, None]
    return idx.astype(np.int32)


def random_subsample(
    xyz: np.ndarray, ratio: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Random sub-sample to 1/ratio points. Returns (sub_xyz, sub_idx).

    RandLA-Net deliberately uses uniform random sampling — it's O(N) and
    the network learns to compensate via attentive pooling. Picking
    points without replacement, no replacement seed (deterministic when
    the caller sets numpy's global state).
    """
    n = len(xyz)
    target = max(1, n // ratio)
    sel = np.random.choice(n, size=target, replace=False)
    return xyz[sel], sel.astype(np.int64)


# ── Building blocks ───────────────────────────────────────────────────────────


class SharedMLP(nn.Module):
    """1×1 Conv2d wrapper with optional BN + activation.

    The 2d-conv path mirrors open3d-ml's per-point MLP wiring: features
    are shaped (B, C, N, 1) and a 1×1 kernel operates pointwise. This
    keeps the parameter shapes identical to the open3d-ml RandLANet so
    a remap of names alone may transfer weights.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        use_bn: bool = True,
        activation: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, bias=not use_bn,
        )
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else None
        # LeakyReLU(0.2) matches the open3d-ml default; consistent across
        # every per-point conv inside the network.
        self.act = nn.LeakyReLU(0.2, inplace=True) if activation else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x


class LocalSpatialEncoding(nn.Module):
    """Pack relative position + neighbour features into a richer descriptor.

    Inputs:
        coords:   (B, 3, N, 1)  XYZ of the centre points
        neigh_xyz:(B, 3, N, K)  XYZ of K neighbours per point
        neigh_f:  (B, F, N, K)  features of those neighbours
    Output:       (B, 2*F, N, K)  fused descriptors
    """

    def __init__(self, in_features: int):
        super().__init__()
        # 10 = (relative xyz 3) + (absolute xyz of neighbour 3) +
        #      (absolute xyz of centre 3) + (euclidean distance 1)
        self.mlp = SharedMLP(10, in_features)

    def forward(
        self,
        coords: torch.Tensor,
        neigh_xyz: torch.Tensor,
        neigh_features: torch.Tensor,
    ) -> torch.Tensor:
        b, _, n, k = neigh_xyz.shape
        centre = coords.expand(-1, -1, -1, k)
        rel = neigh_xyz - centre
        dist = torch.norm(rel, dim=1, keepdim=True)
        encoded = torch.cat([rel, neigh_xyz, centre, dist], dim=1)
        # (B, 10, N, K) → (B, F, N, K)
        encoded = self.mlp(encoded)
        return torch.cat([encoded, neigh_features], dim=1)


class AttentivePooling(nn.Module):
    """Learnable attention-weighted aggregation across K neighbours."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # Score-MLP: per-(feature, neighbour) gating coefficient.
        self.score_mlp = nn.Conv2d(in_features, in_features, kernel_size=1, bias=False)
        # Post-aggregation projection.
        self.out_mlp = SharedMLP(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, N, K)
        scores = F.softmax(self.score_mlp(x), dim=-1)
        # weighted sum over the K dimension → (B, C, N, 1)
        weighted = (x * scores).sum(dim=-1, keepdim=True)
        return self.out_mlp(weighted)


class DilatedResidualBlock(nn.Module):
    """Two LSE+AttentivePool stages stacked with a residual + activation.

    `d` is the output feature channel count; intermediate channels are
    `d // 2` per the RandLA-Net recipe (the paper's "expansion" stage).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        half = out_channels // 2
        self.pre_mlp = SharedMLP(in_channels, half)
        self.lse1 = LocalSpatialEncoding(half)
        self.att1 = AttentivePooling(2 * half, half)
        self.lse2 = LocalSpatialEncoding(half)
        self.att2 = AttentivePooling(2 * half, out_channels)
        self.shortcut = SharedMLP(in_channels, out_channels, activation=False)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(
        self,
        features: torch.Tensor,    # (B, C_in, N, 1)
        coords: torch.Tensor,      # (B, 3, N, 1)
        neigh_idx: torch.Tensor,   # (B, N, K)  int64
    ) -> torch.Tensor:
        shortcut = self.shortcut(features)

        f = self.pre_mlp(features)            # (B, half, N, 1)
        # Gather neighbour features + neighbour coords using neigh_idx
        neigh_f = _gather_neighbours(f, neigh_idx)        # (B, half, N, K)
        neigh_xyz = _gather_neighbours(coords, neigh_idx)  # (B, 3, N, K)
        f = self.lse1(coords, neigh_xyz, neigh_f)         # (B, 2*half, N, K)
        f = self.att1(f)                                   # (B, half, N, 1)

        neigh_f = _gather_neighbours(f, neigh_idx)         # (B, half, N, K)
        f = self.lse2(coords, neigh_xyz, neigh_f)
        f = self.att2(f)                                   # (B, out, N, 1)

        return self.act(f + shortcut)


# ── Neighbour gather helper ───────────────────────────────────────────────────


def _gather_neighbours(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Look up features at K neighbour indices per point.

    features: (B, C, N, 1)
    idx:      (B, N, K)  int64 — indices into the N dimension of features
    returns:  (B, C, N, K)
    """
    b, c, n, _ = features.shape
    k = idx.shape[-1]
    # Flatten to (B, C, N) for index_select per batch element.
    feats_flat = features.squeeze(-1)
    # Expand idx to (B, C, N*K) for torch.gather along dim=2.
    idx_flat = idx.reshape(b, 1, n * k).expand(-1, c, -1)
    gathered = torch.gather(feats_flat, dim=2, index=idx_flat)
    return gathered.reshape(b, c, n, k)


# ── Full model ────────────────────────────────────────────────────────────────


class RandLANet(nn.Module):
    """Pure-PyTorch RandLA-Net for semantic segmentation.

    Forward expects a *prepared* point set: KNN graphs at every level and
    the sub-sample indices linking adjacent levels. ``prepare_inputs``
    builds these from raw (xyz, feat) numpy arrays.
    """

    def __init__(
        self,
        num_classes: int = 13,
        in_channels: int = 6,
        dim_features: int = 8,
        dim_output: Tuple[int, ...] = (16, 64, 128, 256),
        num_neighbors: int = 16,
        sub_sampling_ratio: Tuple[int, ...] = (4, 4, 4, 4),
        dropout: float = 0.5,
    ):
        super().__init__()
        if len(dim_output) != len(sub_sampling_ratio):
            raise ValueError("dim_output and sub_sampling_ratio must match length")
        self.num_classes = num_classes
        self.num_neighbors = num_neighbors
        self.sub_sampling_ratio = list(sub_sampling_ratio)

        # Input projection: (xyz, rgb) → dim_features per point.
        self.input_fc = SharedMLP(in_channels, dim_features)

        # Encoder ladder. Each level: DRB then we'll random-subsample
        # outside the module (random-pick is a tensor op, not a learned
        # block).
        self.encoder_blocks = nn.ModuleList()
        in_c = dim_features
        for out_c in dim_output:
            self.encoder_blocks.append(DilatedResidualBlock(in_c, out_c))
            in_c = out_c

        # Bottleneck — 1×1 conv after the deepest encoder level.
        self.bottleneck = SharedMLP(in_c, in_c)

        # Decoder: mirror the encoder. At each level we (a) NN-upsample,
        # (b) skip-connect the encoder feature at the same level via
        # channel-wise concat, (c) 1×1 conv to merge.
        self.decoder_blocks = nn.ModuleList()
        reversed_dims = list(reversed(dim_output))
        for i, out_c in enumerate(reversed_dims):
            # Skip features at the layer ABOVE the current decoder level
            # have `reversed_dims[i+1]` channels (or dim_features at the
            # top). After concat, channel count = current + skip.
            skip_c = reversed_dims[i + 1] if i + 1 < len(reversed_dims) else dim_features
            self.decoder_blocks.append(SharedMLP(out_c + skip_c, skip_c))

        # Per-point classification head: 2 × Conv1×1 + dropout, then Linear.
        head_channels = dim_features
        self.head_mlp1 = SharedMLP(head_channels, 64)
        self.head_mlp2 = SharedMLP(64, 32)
        self.head_dropout = nn.Dropout2d(dropout)
        # Final classifier: 1×1 conv keeps the (B, C, N, 1) layout.
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

    # ── input preparation ──────────────────────────────────────────────────

    def prepare_inputs(
        self,
        xyz: np.ndarray,
        features: np.ndarray,
        device: torch.device,
    ) -> dict:
        """Build the multi-level KNN + sub-sample structure for one cloud.

        Args:
            xyz: (N, 3) float32 — XYZ in metres
            features: (N, C) float32 — input features (xyz + rgb typically)

        Returns a dict of torch tensors with everything ``forward`` needs.
        """
        n = len(xyz)
        if n < self.num_neighbors:
            raise ValueError(
                f"Cloud has {n} points; need at least num_neighbors="
                f"{self.num_neighbors}. Tile/upsample the input first."
            )

        levels_xyz: List[np.ndarray] = [xyz]
        levels_idx: List[np.ndarray] = []          # KNN at every level
        sub_indices: List[np.ndarray] = []         # downsample picks per level
        upsample_idx: List[np.ndarray] = []        # nearest-neighbour in the
                                                   # finer level for each
                                                   # point of the coarser one
        cur = xyz
        for ratio in self.sub_sampling_ratio:
            levels_idx.append(knn_indices(cur, self.num_neighbors))
            sub_xyz, sub_sel = random_subsample(cur, ratio)
            sub_indices.append(sub_sel)
            # For upsampling we need: for every point of `cur`, the index
            # in `sub_xyz` of the nearest sub-sampled point. That's 1-NN.
            from scipy.spatial import cKDTree
            tree = cKDTree(sub_xyz)
            _, up = tree.query(cur, k=1, workers=-1)
            upsample_idx.append(up.astype(np.int64))
            levels_xyz.append(sub_xyz)
            cur = sub_xyz

        # KNN graph at the deepest level — used by the bottleneck pass.
        levels_idx.append(knn_indices(cur, self.num_neighbors))

        def to_dev(a, dtype=torch.float32):
            return torch.from_numpy(np.ascontiguousarray(a)).to(device=device, dtype=dtype)

        return {
            "features": to_dev(features).t().unsqueeze(0).unsqueeze(-1),  # (1, C, N, 1)
            "xyz_per_level": [
                to_dev(x).t().unsqueeze(0).unsqueeze(-1) for x in levels_xyz
            ],  # each (1, 3, N_l, 1)
            "knn_per_level": [
                to_dev(i, dtype=torch.int64).unsqueeze(0) for i in levels_idx
            ],  # each (1, N_l, K)
            "subsample_idx": [
                to_dev(s, dtype=torch.int64) for s in sub_indices
            ],  # each (N_l_target,) selecting from the layer above
            "upsample_idx": [
                to_dev(u, dtype=torch.int64) for u in upsample_idx
            ],  # each (N_finer,) pointing into the coarser layer
        }

    # ── forward ─────────────────────────────────────────────────────────────

    def forward(self, inputs: dict) -> torch.Tensor:
        """Returns per-input-point logits, shape (B, num_classes, N, 1)."""
        f = self.input_fc(inputs["features"])  # (B, dim_features, N, 1)

        encoder_outputs: List[torch.Tensor] = []
        encoder_xyz: List[torch.Tensor] = []
        for level, block in enumerate(self.encoder_blocks):
            xyz_l = inputs["xyz_per_level"][level]
            knn_l = inputs["knn_per_level"][level]
            f = block(f, xyz_l, knn_l)
            encoder_outputs.append(f)
            encoder_xyz.append(xyz_l)
            # Subsample to the next level.
            sub_idx = inputs["subsample_idx"][level]
            f = _select_along_n(f, sub_idx)

        # Bottleneck — operates on the deepest level's coords/knn.
        f = self.bottleneck(f)

        # Decoder: walk back up the ladder.
        for i, dec in enumerate(self.decoder_blocks):
            up_idx = inputs["upsample_idx"][-(i + 1)]  # finer-layer indices
            # Upsample: pick the value at up_idx[n] for every n in the finer
            # layer. (B, C, N_coarse, 1) → (B, C, N_fine, 1)
            f = _select_along_n(f, up_idx)
            # Skip connection from the encoder at the corresponding (finer)
            # level. After upsampling we're back at the resolution where
            # encoder_outputs[level_finer] was produced.
            skip_level = len(self.encoder_blocks) - 2 - i
            if skip_level >= 0:
                skip = encoder_outputs[skip_level]
            else:
                # Top of the ladder — concatenate with the projected input
                # features (pre-encoder).
                skip = self.input_fc(inputs["features"])
            f = torch.cat([f, skip], dim=1)
            f = dec(f)

        # Classification head.
        f = self.head_mlp1(f)
        f = self.head_mlp2(f)
        f = self.head_dropout(f)
        logits = self.classifier(f)  # (B, num_classes, N, 1)
        return logits


def _select_along_n(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Pick rows of features along the N dimension by index.

    features: (B, C, N, 1)
    idx:      (M,) int64
    returns:  (B, C, M, 1)
    """
    b, c, _, _ = features.shape
    f = features.squeeze(-1)  # (B, C, N)
    idx_e = idx.view(1, 1, -1).expand(b, c, -1)
    out = torch.gather(f, dim=2, index=idx_e)
    return out.unsqueeze(-1)
