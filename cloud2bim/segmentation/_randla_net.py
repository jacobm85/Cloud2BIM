"""Pure-PyTorch RandLA-Net, structurally mirroring Open3D-ML's RandLANet.

Re-implementation of Open3D-ML's RandLA-Net architecture so its
S3DIS pretrained checkpoint (parameter naming convention:
``fc0 / bn0 / encoder.k.{mlp1,lse1,pool1,lse2,pool2,mlp2,shortcut} /
mlp / decoder.k.{conv,batch_norm} / fc1.0..3``) loads directly into a
torch-2.x-compatible model with no compiled extensions.

S3DIS recipe (matches `ml3d/configs/randlanet_s3dis.yml` in upstream):
    in_channels       = 6
    dim_features      = 8
    dim_output        = (16, 64, 128, 256, 512)
    sub_sampling_ratio= (4, 4, 4, 4, 2)
    num_neighbors     = 16
    num_layers        = 5
    num_classes       = 13

Architectural notes that surprised me when reverse-engineering this:

  * ``LocalFeatureAggregation`` doubles channels via ``mlp2`` from
    ``d_out → 2*d_out``. So an encoder list of d_out [16,64,128,256,512]
    produces actual output channels [32,128,256,512,1024]; the bottleneck
    operates at 1024.
  * ``fc0`` is ``nn.Linear`` over the feature dim, NOT Conv2d. The cloud
    enters as (B, N, in_channels) and only after fc0+bn0 is it reshaped
    to (B, C, N, 1) for the rest of the pipeline.
  * Decoder ``SharedMLP`` uses ``ConvTranspose2d`` (``transpose=True``).
    Same parameter shapes as Conv2d so it doesn't break weight transfer
    — pick the right one in the class.
  * ``AttentivePooling.score_fn`` is ``Sequential(Linear, Softmax)`` and
    operates on the LAST dim (K neighbours), not on channels.

KNN is computed CPU-side via scipy.cKDTree. Random sub-sampling picks
points without replacement at each level.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Geometric helpers ─────────────────────────────────────────────────────────


def knn_indices(xyz: np.ndarray, k: int) -> np.ndarray:
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


def nearest_in(target: np.ndarray, query: np.ndarray) -> np.ndarray:
    """For every point in ``query`` return the index of the closest point in ``target``."""
    from scipy.spatial import cKDTree
    tree = cKDTree(target)
    _, idx = tree.query(query, k=1, workers=-1)
    return idx.astype(np.int64)


def random_subsample(
    xyz: np.ndarray, ratio: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Random sub-sample to 1/ratio points without replacement."""
    n = len(xyz)
    target = max(1, n // ratio)
    sel = np.random.choice(n, size=target, replace=False)
    return xyz[sel], sel.astype(np.int64)


# ── Building blocks ───────────────────────────────────────────────────────────


class SharedMLP(nn.Module):
    """1×1 Conv2d/ConvTranspose2d + optional BatchNorm + optional activation.

    Mirrors Open3D-ML's ``SharedMLP`` exactly. Submodule names: ``conv``,
    ``batch_norm``. ``activation_fn`` is stored on the instance but is
    not a learnable module (no params), so it doesn't appear in
    state_dict — safe to use as an attribute.
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
        # Open3D-ML names the BN module ``batch_norm`` and uses
        # eps=1e-6, momentum=0.01 in its RandLA-Net.
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
    """Encode either positional info (encode_pos=True) or feature info into
    an (in→out) projection, then concatenate with the K-gathered neighbour
    features. Output has 2 × out_channels per neighbour.

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
        coords: torch.Tensor,        # (B, 3, N, 1)
        features: torch.Tensor,      # (B, d_in_features, N, 1)
        neighbour_idx: torch.Tensor, # (B, N, K)
    ) -> torch.Tensor:
        """Returns (B, 2*out_channels, N, K)."""
        b, _, n, _ = coords.shape
        k = neighbour_idx.shape[-1]
        # Gather neighbour features
        neigh_feats = _gather_neighbours(features, neighbour_idx)  # (B, d_in_features, N, K)

        if self.encode_pos:
            # Build 10-channel positional encoding per (centre, neighbour).
            neigh_xyz = _gather_neighbours(coords, neighbour_idx)  # (B, 3, N, K)
            centre = coords.expand(-1, -1, -1, k)
            rel = neigh_xyz - centre
            dist = torch.norm(rel, dim=1, keepdim=True)
            encoded = torch.cat([rel, neigh_xyz, centre, dist], dim=1)  # (B, 10, N, K)
            encoded = self.mlp(encoded)
        else:
            encoded = self.mlp(neigh_feats)

        return torch.cat([encoded, neigh_feats], dim=1)


class AttentivePooling(nn.Module):
    """Learnable softmax-weighted pooling over the K-neighbour dim.

    Submodule names: ``score_fn`` (Sequential of Linear+Softmax over K),
    ``mlp`` (SharedMLP).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Linear operates on the *channel* dim because Open3D-ML's
        # forward permutes features to (B, N, K, C) before applying it.
        # We do the same permute below.
        self.score_fn = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.Softmax(dim=-2),  # softmax over the K dim (after permute)
        )
        self.mlp = SharedMLP(in_channels, out_channels, bn=True, activation_fn=nn.LeakyReLU(0.2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, N, K)
        # Permute to (B, N, K, C) so the Linear acts on channels.
        b, c, n, k = x.shape
        x_perm = x.permute(0, 2, 3, 1)  # (B, N, K, C)
        scores = self.score_fn(x_perm)  # softmax over K (dim=-2) → (B, N, K, C)
        weighted = (x_perm * scores).sum(dim=-2, keepdim=False)  # (B, N, C)
        # Back to (B, C, N, 1) for the SharedMLP.
        feat = weighted.permute(0, 2, 1).unsqueeze(-1)
        return self.mlp(feat)


class LocalFeatureAggregation(nn.Module):
    """One encoder block: two LSE+AttPool stages stacked with a residual.

    Submodule names mirror Open3D-ML exactly: ``mlp1``, ``lse1``, ``pool1``,
    ``lse2``, ``pool2``, ``mlp2``, ``shortcut``, ``lrelu``.

    Channels: in d_in, internal d_out//2 in the two LSE+pool stages,
    output 2 × d_out via mlp2.
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
        coords: torch.Tensor,        # (B, 3, N, 1)
        features: torch.Tensor,      # (B, d_in, N, 1)
        neighbour_idx: torch.Tensor, # (B, N, K)
    ) -> torch.Tensor:
        shortcut = self.shortcut(features)

        f = self.mlp1(features)                           # (B, d_out//2, N, 1)
        f = self.lse1(coords, f, neighbour_idx)           # (B, d_out, N, K)
        f = self.pool1(f)                                 # (B, d_out//2, N, 1)
        f = self.lse2(coords, f, neighbour_idx)           # (B, d_out, N, K)
        f = self.pool2(f)                                 # (B, d_out, N, 1)
        f = self.mlp2(f)                                  # (B, 2*d_out, N, 1)
        return self.lrelu(f + shortcut)


# ── Neighbour gather helper ───────────────────────────────────────────────────


def _gather_neighbours(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """features: (B, C, N, 1); idx: (B, N, K) int64. → (B, C, N, K)."""
    b, c, n, _ = features.shape
    k = idx.shape[-1]
    feats_flat = features.squeeze(-1)
    idx_flat = idx.reshape(b, 1, n * k).expand(-1, c, -1)
    gathered = torch.gather(feats_flat, dim=2, index=idx_flat)
    return gathered.reshape(b, c, n, k)


def _select_along_n(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """features: (B, C, N, 1); idx: (M,) int64. → (B, C, M, 1)."""
    b, c, _, _ = features.shape
    f = features.squeeze(-1)
    idx_e = idx.view(1, 1, -1).expand(b, c, -1)
    out = torch.gather(f, dim=2, index=idx_e)
    return out.unsqueeze(-1)


# ── Full model ────────────────────────────────────────────────────────────────


class RandLANet(nn.Module):
    """Pure-PyTorch RandLA-Net, parameter-name-compatible with Open3D-ML.

    Top-level submodule names: ``fc0``, ``bn0``, ``encoder`` (ModuleList),
    ``mlp`` (bottleneck), ``decoder`` (ModuleList), ``fc1`` (Sequential).
    """

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

        # Input projection: nn.Linear over channel dim. The cloud enters
        # as (B, N, in_channels) and only after fc0+bn0 is it reshaped to
        # the (B, C, N, 1) layout for the rest of the pipeline.
        self.fc0 = nn.Linear(in_channels, dim_features)
        self.bn0 = nn.BatchNorm2d(dim_features, eps=1e-6, momentum=0.01)

        # Encoder
        encoder = []
        encoder_dim_list: list[int] = []
        dim_feature = dim_features
        for i, d_out in enumerate(dim_output):
            encoder.append(LocalFeatureAggregation(dim_feature, d_out, num_neighbors))
            dim_feature = 2 * d_out
            if i == 0:
                encoder_dim_list.append(dim_feature)  # mirror Open3D-ML's
            encoder_dim_list.append(dim_feature)      # double-append at i=0
        self.encoder = nn.ModuleList(encoder)

        # Bottleneck (mlp)
        self.mlp = SharedMLP(dim_feature, dim_feature, bn=True,
                             activation_fn=nn.LeakyReLU(0.2))

        # Decoder. SharedMLP with transpose=True (ConvTranspose2d). Channel
        # chain follows ``encoder_dim_list`` walked back from the end.
        decoder = []
        for i in range(len(dim_output)):
            d_in = encoder_dim_list[-i - 2] + dim_feature
            d_out = encoder_dim_list[-i - 2]
            decoder.append(SharedMLP(d_in, d_out, transpose=True, bn=True,
                                     activation_fn=nn.LeakyReLU(0.2)))
            dim_feature = d_out
        self.decoder = nn.ModuleList(decoder)

        # Final head: Sequential of SharedMLP, SharedMLP, Dropout, SharedMLP.
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
        """Build the multi-level KNN + sub/up-sample structure for one cloud."""
        n = len(xyz)
        if n < self.num_neighbors:
            raise ValueError(
                f"Cloud has {n} points; need at least num_neighbors="
                f"{self.num_neighbors}."
            )

        levels_xyz: List[np.ndarray] = [xyz]
        knn_per_level: List[np.ndarray] = []
        sub_idx_per_level: List[np.ndarray] = []
        upsample_per_level: List[np.ndarray] = []
        cur = xyz
        for ratio in self.sub_sampling_ratio:
            knn_per_level.append(knn_indices(cur, self.num_neighbors))
            sub_xyz, sub_sel = random_subsample(cur, ratio)
            sub_idx_per_level.append(sub_sel)
            upsample_per_level.append(nearest_in(sub_xyz, cur))
            levels_xyz.append(sub_xyz)
            cur = sub_xyz
        # Deepest level KNN — used by the bottleneck pass.
        knn_per_level.append(knn_indices(cur, self.num_neighbors))

        def to_dev(a, dtype=torch.float32):
            return torch.from_numpy(np.ascontiguousarray(a)).to(device=device, dtype=dtype)

        return {
            # fc0 expects (B, N, in_channels); we'll transpose into (B, C, N, 1)
            # after fc0 + bn0 inside forward().
            "features": to_dev(features).unsqueeze(0),     # (1, N, in_channels)
            "xyz_per_level": [
                to_dev(x).t().unsqueeze(0).unsqueeze(-1) for x in levels_xyz
            ],
            "knn_per_level": [
                to_dev(i, dtype=torch.int64).unsqueeze(0) for i in knn_per_level
            ],
            "subsample_idx": [
                to_dev(s, dtype=torch.int64) for s in sub_idx_per_level
            ],
            "upsample_idx": [
                to_dev(u, dtype=torch.int64) for u in upsample_per_level
            ],
        }

    # ── forward ─────────────────────────────────────────────────────────────

    def forward(self, inputs: dict) -> torch.Tensor:
        """Returns (B, num_classes, N, 1)."""
        # fc0 acts on the channel dim of (B, N, in_channels).
        feat = self.fc0(inputs["features"])                # (B, N, dim_features)
        # Reshape to (B, dim_features, N, 1) for the rest of the pipeline.
        feat = feat.transpose(-2, -1).unsqueeze(-1)
        feat = self.bn0(feat)
        feat = F.leaky_relu(feat, 0.2)

        encoder_outputs: List[torch.Tensor] = []
        for i, block in enumerate(self.encoder):
            xyz_l = inputs["xyz_per_level"][i]
            knn_l = inputs["knn_per_level"][i]
            feat = block(xyz_l, feat, knn_l)
            encoder_outputs.append(feat)
            sub_idx = inputs["subsample_idx"][i]
            feat = _select_along_n(feat, sub_idx)

        feat = self.mlp(feat)

        n_levels = len(self.encoder)
        for i, dec in enumerate(self.decoder):
            finer_level = n_levels - 1 - i
            up_idx = inputs["upsample_idx"][finer_level]
            feat = _select_along_n(feat, up_idx)
            skip = encoder_outputs[finer_level]
            feat = torch.cat([feat, skip], dim=1)
            feat = dec(feat)

        return self.fc1(feat)
