"""Pure-PyTorch RandLA-Net, behaviourally identical to Open3D-ML's RandLANet.

Mirrors Open3D-ML's class hierarchy + parameter naming + forward logic
so the S3DIS pretrained checkpoint
(``ml3d/configs/randlanet_s3dis.yml``) loads byte-for-byte with no
remapping, and inference produces equivalent label outputs on a
torch-2.5 stack with no compiled extensions.

S3DIS recipe applied by default:
    in_channels       = 6        (XYZ + RGB)
    dim_features      = 8
    dim_output        = (16, 64, 128, 256, 512)
    sub_sampling_ratio= (4, 4, 4, 4, 2)
    num_neighbors     = 16
    num_layers        = 5
    num_classes       = 13

Implementation notes that surprised me while reverse-engineering the
upstream forward:

  * ``random_sample`` is misleadingly named — it's actually K-nearest
    max-pooling. ``sub_idx[i]`` has shape ``(N_{i+1}, K)`` and for each
    point in level i+1 holds the indices of its K nearest neighbours
    in level i; the pooling then maxes over the K-dim.
  * ``nearest_interpolation`` is just 1-NN gather; ``interp_idx[i]``
    has shape ``(N_i, 1)`` and for each point in level i holds the
    index of its nearest neighbour in level i+1. Used in the decoder
    to walk back up the ladder.
  * ``encoder_feat_list`` stores the **post-pool** features at each
    level, with an extra ``clone()`` of the un-pooled level-0 output
    appended at i==0. Six entries for 5 layers. The decoder uses
    ``encoder_feat_list[-i - 2]`` so it skips the deepest output
    (level-5 post-pool, identical to the bottleneck input) and instead
    walks levels 4 → 3 → 2 → 1 → 0.
  * ``AttentivePooling.score_fn`` is ``Sequential(Linear, Softmax)``
    where the Linear has ``bias=True``. The checkpoint's
    ``score_fn.0.bias`` keys (10 tensors for S3DIS) won't transfer
    otherwise.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Geometric helpers ─────────────────────────────────────────────────────────


def _knn_indices(xyz: np.ndarray, k: int) -> np.ndarray:
    """K-nearest-neighbour indices for every point. CPU via scipy."""
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise ImportError("scipy is required for RandLA-Net's KNN.") from exc
    tree = cKDTree(xyz)
    _, idx = tree.query(xyz, k=k, workers=-1)
    if idx.ndim == 1:
        idx = idx[:, None]
    return idx.astype(np.int32)


def _knn_query(target: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    """For every point in ``query``, the indices of its k nearest in ``target``."""
    from scipy.spatial import cKDTree
    tree = cKDTree(target)
    _, idx = tree.query(query, k=k, workers=-1)
    if idx.ndim == 1:
        idx = idx[:, None]
    return idx.astype(np.int32)


def _random_subsample_xyz(xyz: np.ndarray, ratio: int) -> np.ndarray:
    """Random sub-sample to 1/ratio points without replacement.

    RandLA-Net uses random sampling deliberately — it's O(N) and the
    network compensates via the attentive-pooling stage.
    """
    n = len(xyz)
    target = max(1, n // ratio)
    sel = np.random.choice(n, size=target, replace=False)
    return xyz[sel]


# ── Building blocks ───────────────────────────────────────────────────────────


class SharedMLP(nn.Module):
    """1×1 Conv2d/ConvTranspose2d + optional BatchNorm + optional activation.

    Submodule names: ``conv``, ``batch_norm``. ``activation_fn`` is a
    stored attribute, not a module — Open3D-ML's checkpoint never has
    params for it.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        transpose: bool = False,
        bn: bool = False,
        activation_fn: nn.Module | None = None,
    ):
        super().__init__()
        conv_cls = nn.ConvTranspose2d if transpose else nn.Conv2d
        self.conv = conv_cls(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.batch_norm = (
            nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.01) if bn else None
        )
        self.activation_fn = activation_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        return x


class LocalSpatialEncoding(nn.Module):
    """Project either positional or feature info, then concat with K-gathered
    neighbour features. Output has 2 × out_channels per neighbour.

    Submodule name: ``mlp`` (SharedMLP).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_neighbors: int,
        encode_pos: bool = False,
    ):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.encode_pos = encode_pos
        self.mlp = SharedMLP(in_channels, out_channels, bn=True, activation_fn=nn.LeakyReLU(0.2))

    def forward(
        self,
        coords: torch.Tensor,        # (B, N, 3)
        features: torch.Tensor,      # (B, d_in_features, N, 1)
        neighbour_idx: torch.Tensor, # (B, N, K)
    ) -> torch.Tensor:
        """Returns (B, 2*out_channels, N, K)."""
        b, _, n, _ = features.shape
        k = neighbour_idx.shape[-1]
        # Gather neighbour features.
        neigh_feats = _gather_neighbours(features, neighbour_idx)  # (B, d, N, K)
        if self.encode_pos:
            # 10-channel positional encoding per (centre, neighbour).
            neigh_xyz = _gather_neighbours_xyz(coords, neighbour_idx)  # (B, 3, N, K)
            centre = coords.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, k)  # (B, 3, N, K)
            rel = neigh_xyz - centre
            dist = torch.norm(rel, dim=1, keepdim=True)
            encoded = torch.cat([rel, neigh_xyz, centre, dist], dim=1)  # (B, 10, N, K)
            encoded = self.mlp(encoded)
        else:
            encoded = self.mlp(neigh_feats)
        return torch.cat([encoded, neigh_feats], dim=1)


class AttentivePooling(nn.Module):
    """Softmax-weighted pooling over the K-neighbour dim.

    Submodule names: ``score_fn`` (Sequential of Linear+Softmax),
    ``mlp`` (SharedMLP). The Linear has ``bias=True`` to match
    Open3D-ML's checkpoint.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.score_fn = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=True),
            nn.Softmax(dim=-2),
        )
        self.mlp = SharedMLP(in_channels, out_channels, bn=True, activation_fn=nn.LeakyReLU(0.2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, N, K) → permute to (B, N, K, C) so Linear acts on channels.
        x_perm = x.permute(0, 2, 3, 1)
        scores = self.score_fn(x_perm)            # softmax over K (dim=-2)
        weighted = (x_perm * scores).sum(dim=-2)  # (B, N, C)
        feat = weighted.permute(0, 2, 1).unsqueeze(-1)  # (B, C, N, 1)
        return self.mlp(feat)


class LocalFeatureAggregation(nn.Module):
    """One encoder block: two LSE+AttPool stages stacked with a residual.

    Submodule names mirror Open3D-ML: ``mlp1``, ``lse1``, ``pool1``,
    ``lse2``, ``pool2``, ``mlp2``, ``shortcut``, ``lrelu``.
    """

    def __init__(self, d_in: int, d_out: int, num_neighbors: int):
        super().__init__()
        self.mlp1 = SharedMLP(d_in, d_out // 2, bn=True, activation_fn=nn.LeakyReLU(0.2))
        self.lse1 = LocalSpatialEncoding(10, d_out // 2, num_neighbors, encode_pos=True)
        self.pool1 = AttentivePooling(d_out, d_out // 2)
        self.lse2 = LocalSpatialEncoding(d_out // 2, d_out // 2, num_neighbors)
        self.pool2 = AttentivePooling(d_out, d_out)
        self.mlp2 = SharedMLP(d_out, 2 * d_out, bn=True)
        self.shortcut = SharedMLP(d_in, 2 * d_out, bn=True)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(
        self,
        coords: torch.Tensor,        # (B, N, 3)
        features: torch.Tensor,      # (B, d_in, N, 1)
        neighbour_idx: torch.Tensor, # (B, N, K)
    ) -> torch.Tensor:
        shortcut = self.shortcut(features)
        f = self.mlp1(features)
        f = self.lse1(coords, f, neighbour_idx)
        f = self.pool1(f)
        f = self.lse2(coords, f, neighbour_idx)
        f = self.pool2(f)
        f = self.mlp2(f)
        return self.lrelu(f + shortcut)


# ── Neighbour gather helpers ──────────────────────────────────────────────────


def _gather_neighbours(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """features: (B, C, N, 1); idx: (B, N, K) int64. → (B, C, N, K)."""
    b, c, n, _ = features.shape
    k = idx.shape[-1]
    feats_flat = features.squeeze(-1)
    idx_flat = idx.reshape(b, 1, n * k).expand(-1, c, -1)
    gathered = torch.gather(feats_flat, dim=2, index=idx_flat)
    return gathered.reshape(b, c, n, k)


def _gather_neighbours_xyz(coords: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """coords: (B, N, 3); idx: (B, N, K) int64. → (B, 3, N, K)."""
    b, n, _ = coords.shape
    k = idx.shape[-1]
    idx_exp = idx.reshape(b, n * k, 1).expand(-1, -1, 3)  # (B, N*K, 3)
    gathered = torch.gather(coords, 1, idx_exp)            # (B, N*K, 3)
    return gathered.reshape(b, n, k, 3).permute(0, 3, 1, 2)  # (B, 3, N, K)


# ── Full model ────────────────────────────────────────────────────────────────


class RandLANet(nn.Module):
    """Pure-PyTorch RandLA-Net. Parameter names match Open3D-ML's RandLANet."""

    def __init__(
        self,
        num_classes: int = 13,
        in_channels: int = 6,
        dim_features: int = 8,
        dim_output: Tuple[int, ...] = (16, 64, 128, 256, 512),
        num_neighbors: int = 16,
        sub_sampling_ratio: Tuple[int, ...] = (4, 4, 4, 4, 2),
        dropout: float = 0.5,
    ):
        super().__init__()
        if len(dim_output) != len(sub_sampling_ratio):
            raise ValueError("dim_output and sub_sampling_ratio must match length")
        self.num_classes = num_classes
        self.num_neighbors = num_neighbors
        self.sub_sampling_ratio = list(sub_sampling_ratio)
        self.in_channels = in_channels
        self.num_layers = len(dim_output)

        self.fc0 = nn.Linear(in_channels, dim_features)
        self.bn0 = nn.BatchNorm2d(dim_features, eps=1e-6, momentum=0.01)

        encoder = []
        encoder_dim_list: list[int] = []
        dim_feature = dim_features
        for i, d_out in enumerate(dim_output):
            encoder.append(LocalFeatureAggregation(dim_feature, d_out, num_neighbors))
            dim_feature = 2 * d_out
            if i == 0:
                encoder_dim_list.append(dim_feature)
            encoder_dim_list.append(dim_feature)
        self.encoder = nn.ModuleList(encoder)

        self.mlp = SharedMLP(dim_feature, dim_feature, bn=True,
                             activation_fn=nn.LeakyReLU(0.2))

        decoder = []
        for i in range(self.num_layers):
            d_in = encoder_dim_list[-i - 2] + dim_feature
            d_out = encoder_dim_list[-i - 2]
            decoder.append(SharedMLP(d_in, d_out, transpose=True, bn=True,
                                     activation_fn=nn.LeakyReLU(0.2)))
            dim_feature = d_out
        self.decoder = nn.ModuleList(decoder)

        self.fc1 = nn.Sequential(
            SharedMLP(dim_feature, 64, bn=True, activation_fn=nn.LeakyReLU(0.2)),
            SharedMLP(64, 32, bn=True, activation_fn=nn.LeakyReLU(0.2)),
            nn.Dropout(dropout),
            SharedMLP(32, num_classes, bn=False),
        )

    # ── input preparation ──────────────────────────────────────────────────

    def prepare_inputs(
        self,
        xyz: np.ndarray,
        features: np.ndarray,
        device: torch.device,
    ) -> dict:
        """Build the multi-level KNN + sub/up-sample structure.

        Mirrors Open3D-ML's data dict:
            features:           (1, N, in_channels)
            coords[i]:          (1, N_i, 3)
            neighbor_indices[i]:(1, N_i, K)      KNN within level i
            sub_idx[i]:         (1, N_{i+1}, K)  K nearest in level i for each
                                                 point of level i+1 (used by
                                                 max-pool subsampling)
            interp_idx[i]:      (1, N_i, 1)      nearest point of level i+1
                                                 for each point of level i
                                                 (used by 1-NN upsample)
        """
        n = len(xyz)
        if n < self.num_neighbors:
            raise ValueError(
                f"Cloud has {n} points; need at least num_neighbors="
                f"{self.num_neighbors}."
            )

        coords_list = [xyz]
        neighbor_indices: list[np.ndarray] = []
        sub_idx_list: list[np.ndarray] = []
        interp_idx_list: list[np.ndarray] = []
        cur = xyz
        for ratio in self.sub_sampling_ratio:
            neighbor_indices.append(_knn_indices(cur, self.num_neighbors))
            sub_xyz = _random_subsample_xyz(cur, ratio)
            # K nearest in `cur` for each point in `sub_xyz` — used by
            # random_sample's max-pool.
            sub_idx_list.append(_knn_query(cur, sub_xyz, self.num_neighbors))
            # 1-NN in `sub_xyz` for each point in `cur` — used by
            # nearest_interpolation in the decoder.
            interp_idx_list.append(_knn_query(sub_xyz, cur, 1))
            coords_list.append(sub_xyz)
            cur = sub_xyz

        def to_dev(a, dtype=torch.float32):
            return torch.from_numpy(np.ascontiguousarray(a)).to(device=device, dtype=dtype)

        return {
            "features": to_dev(features).unsqueeze(0),                      # (1, N, in_channels)
            "coords":   [to_dev(x).unsqueeze(0) for x in coords_list],      # (1, N_i, 3)
            "neighbor_indices": [
                to_dev(i, dtype=torch.int64).unsqueeze(0) for i in neighbor_indices
            ],
            "sub_idx":  [
                to_dev(s, dtype=torch.int64).unsqueeze(0) for s in sub_idx_list
            ],
            "interp_idx": [
                to_dev(u, dtype=torch.int64).unsqueeze(0) for u in interp_idx_list
            ],
        }

    # ── sampling helpers (mirror Open3D-ML's static methods) ───────────────

    @staticmethod
    def random_sample(feature: torch.Tensor, pool_idx: torch.Tensor) -> torch.Tensor:
        """K-nearest max-pool subsampling.

        feature: (B, d, N, 1)
        pool_idx: (B, N', K)  — for each of N' points, K nearest in N
        returns: (B, d, N', 1)
        """
        feat = feature.squeeze(3)                                    # (B, d, N)
        b, d, _ = feat.shape
        _, n_target, k = pool_idx.shape
        flat = pool_idx.reshape(b, -1).unsqueeze(2).expand(b, -1, d)  # (B, N'*K, d)
        feat_t = feat.transpose(1, 2)                                  # (B, N, d)
        gathered = torch.gather(feat_t, 1, flat)                       # (B, N'*K, d)
        gathered = gathered.reshape(b, n_target, k, d)                 # (B, N', K, d)
        pooled, _ = torch.max(gathered, dim=2, keepdim=True)           # (B, N', 1, d)
        return pooled.permute(0, 3, 1, 2)                              # (B, d, N', 1)

    @staticmethod
    def nearest_interpolation(feature: torch.Tensor, interp_idx: torch.Tensor) -> torch.Tensor:
        """1-NN gather upsampling.

        feature: (B, d, N', 1)
        interp_idx: (B, N, 1)  — for each of N points, nearest in N'
        returns: (B, d, N, 1)
        """
        feat = feature.squeeze(3)                                    # (B, d, N')
        b, d, _ = feat.shape
        n = interp_idx.shape[1]
        idx = interp_idx.reshape(b, n).unsqueeze(1).expand(b, d, -1)  # (B, d, N)
        gathered = torch.gather(feat, 2, idx)                          # (B, d, N)
        return gathered.unsqueeze(3)                                   # (B, d, N, 1)

    # ── forward ─────────────────────────────────────────────────────────────

    def forward(self, inputs: dict) -> torch.Tensor:
        """Returns (B, num_classes, N) — same shape as Open3D-ML's forward."""
        feat = self.fc0(inputs["features"])                # (B, N, dim_features)
        feat = feat.transpose(-2, -1).unsqueeze(-1)        # (B, d, N, 1)
        feat = self.bn0(feat)
        feat = F.leaky_relu(feat, 0.2)

        coords_list = inputs["coords"]
        neighbor_indices = inputs["neighbor_indices"]
        sub_idx_list = inputs["sub_idx"]
        interp_idx_list = inputs["interp_idx"]

        # Encoder. Saves SAMPLED features after each block, plus an extra
        # un-sampled copy at i=0 (Open3D-ML's encoder_feat_list pattern).
        encoder_feat_list: List[torch.Tensor] = []
        for i in range(self.num_layers):
            feat_enc = self.encoder[i](coords_list[i], feat, neighbor_indices[i])
            feat_sampled = self.random_sample(feat_enc, sub_idx_list[i])
            if i == 0:
                encoder_feat_list.append(feat_enc.clone())
            encoder_feat_list.append(feat_sampled.clone())
            feat = feat_sampled

        feat = self.mlp(feat)

        # Decoder. Walks back up: at iteration i upsamples once and skip-
        # connects with encoder_feat_list[-i - 2].
        for i in range(self.num_layers):
            feat_interp = self.nearest_interpolation(feat, interp_idx_list[-i - 1])
            feat = torch.cat([encoder_feat_list[-i - 2], feat_interp], dim=1)
            feat = self.decoder[i](feat)

        scores = self.fc1(feat)                            # (B, num_classes, N, 1)
        return scores.squeeze(3)                           # (B, num_classes, N)
