"""
PointMetaBase-S-X3D backbone for electron density energy prediction.

Faithfully reproduced from EDBench:
  openpoints/models/backbone/pointmetabase_X3D.py

Architecture (PointMetaBase-S-X3D, paper config):
  - 6 stages, blocks=[1,1,1,1,1,1], strides=[1,2,2,2,2,1]
  - Base width=32, channel doubles per stride-2 stage (capped at 256)
  - Ball-query local aggregation (radius, K neighbours)
  - InvResMLP blocks for feature mixing
  - X-3D explicit structure encoding (PCA + PointHop) in stages 3 & 4
  - Global max+mean pooling → flat vector
  - RegressionHead for energy prediction

Point format: (B, N, C) throughout internally;
Input expected as (B, N, 4) where 4 = [x, y, z, density].
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .x3d_utils.explicit_structure import PCAPointHopGeometry
from .x3d_utils.neighbor_context import NeighborContext
from ..cls_head import RegressionHead


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _ball_query(
    pos: torch.Tensor,         # (B, N, 3)
    center: torch.Tensor,      # (B, M, 3)
    radius: float,
    K: int,
) -> torch.Tensor:
    """
    Returns indices of K nearest neighbours within radius for each centre.

    Output: (B, M, K)  – neighbour indices into pos.
    If fewer than K points lie within radius, fills with the closest point.
    """
    B, N, _ = pos.shape
    M = center.shape[1]

    # Pairwise distances (B, M, N)
    diff = center.unsqueeze(3) - pos.unsqueeze(1)   # (B, M, N, 3)
    dist2 = (diff ** 2).sum(-1)                      # (B, M, N)

    # Mask points outside radius
    INF = 1e10
    dist2_masked = dist2.clone()
    dist2_masked[dist2 > radius ** 2] = INF

    # Take K nearest (sorted)
    topk_dist, topk_idx = dist2_masked.topk(
        min(K, N), dim=-1, largest=False, sorted=True
    )  # (B, M, K)

    if K > N:
        # Pad by repeating the last index
        pad = K - N
        topk_idx = torch.cat([topk_idx, topk_idx[:, :, -1:].expand(-1, -1, pad)], dim=-1)
        topk_dist = torch.cat([topk_dist, topk_dist[:, :, -1:].expand(-1, -1, pad)], dim=-1)

    # Replace INF entries with the nearest neighbour (fallback)
    fallback = topk_idx[:, :, :1].expand_as(topk_idx)
    invalid = (topk_dist == INF)
    topk_idx = torch.where(invalid, fallback, topk_idx)

    return topk_idx   # (B, M, K)


def _fps(pos: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Farthest Point Sampling on (B, N, 3) → returns (B, npoint) indices.
    Uses torch_cluster.fps if available; falls back to a pure-PyTorch
    implementation for compatibility.
    """
    B, N, _ = pos.shape
    if npoint >= N:
        return torch.arange(N, device=pos.device).unsqueeze(0).expand(B, -1)

    try:
        from torch_cluster import fps as _fps_cluster
        # torch_cluster.fps expects flattened (B*N, 3) + batch vector
        pos_flat = pos.reshape(B * N, 3)
        batch = torch.arange(B, device=pos.device).repeat_interleave(N)
        ratio = npoint / N
        idx_flat = _fps_cluster(pos_flat, batch, ratio=ratio, random_start=False)
        # idx_flat is global; convert to per-sample local indices
        idx_flat = idx_flat.reshape(B, -1)[:, :npoint]          # (B, npoint)
        local_idx = idx_flat - (torch.arange(B, device=pos.device) * N).unsqueeze(1)
        return local_idx
    except ImportError:
        pass

    # Pure-PyTorch greedy FPS fallback
    device = pos.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    dist = torch.full((B, N), 1e10, device=device)
    farthest = torch.zeros(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = pos[torch.arange(B), farthest, :].unsqueeze(1)   # (B,1,3)
        d = ((pos - centroid) ** 2).sum(-1)                          # (B, N)
        dist = torch.minimum(dist, d)
        farthest = dist.argmax(dim=-1)

    return centroids   # (B, npoint)


def _gather(feat: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Gather rows from feat using idx.
    feat: (B, N, C)
    idx:  (B, M) or (B, M, K)
    Returns: (B, M, C) or (B, M, K, C)
    """
    B, N, C = feat.shape
    if idx.dim() == 2:
        M = idx.shape[1]
        idx_exp = idx.unsqueeze(-1).expand(B, M, C)
        return feat.gather(1, idx_exp)           # (B, M, C)
    else:
        M, K = idx.shape[1], idx.shape[2]
        idx_exp = idx.unsqueeze(-1).expand(B, M, K, C)
        feat_exp = feat.unsqueeze(2).expand(B, N, K, C)
        # gather along dim=1 for each (m, k) pair
        idx_flat = idx.reshape(B, M * K)
        out = feat.gather(1, idx_flat.unsqueeze(-1).expand(B, M * K, C))
        return out.reshape(B, M, K, C)           # (B, M, K, C)


# ---------------------------------------------------------------------------
# InvResMLP block (Inverted Residual MLP)
# ---------------------------------------------------------------------------

class InvResMLP(nn.Module):
    """
    Inverted Residual MLP: expand → depthwise-like → contract, with residual.
    Expansion factor = 4 (matches PointMetaBase-S).
    """

    def __init__(self, channels: int, expansion: int = 4) -> None:
        super().__init__()
        mid = channels * expansion
        self.net = nn.Sequential(
            nn.Linear(channels, mid),
            nn.BatchNorm1d(mid),
            nn.GELU(),
            nn.Linear(mid, channels),
            nn.BatchNorm1d(channels),
        )

    def _apply_bn(self, x: torch.Tensor, bn: nn.BatchNorm1d) -> torch.Tensor:
        """Apply BN to (..., C) shaped tensor by temporarily flattening."""
        shape = x.shape
        return bn(x.reshape(-1, shape[-1])).reshape(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)  or  (B, C)
        B_shape = x.shape
        flat = x.reshape(-1, B_shape[-1])

        h = self.net[0](flat)                          # Linear
        h = self.net[1](h)                             # BN
        h = self.net[2](h)                             # GELU
        h = self.net[3](h)                             # Linear
        h = self.net[4](h)                             # BN

        return (flat + h).reshape(B_shape)


# ---------------------------------------------------------------------------
# Local aggregation block
# ---------------------------------------------------------------------------

class LocalAgg(nn.Module):
    """
    Local feature aggregation over ball-query neighbours.

    Feature type "dp_fj": [delta_pos | feature_j] concatenated per neighbour,
    then projected → max-pooled → added to centre feature (skip).

    Optionally applies X-3D explicit structure encoding:
      - PCA + PointHop geometry features (33-dim) → NeighborContext weighting
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_x3d: bool = False,
        struct_channels: int = 33,
        x3d_hidden: int = 32,
    ) -> None:
        super().__init__()
        self.use_x3d = use_x3d

        # Input to local MLP: [delta_xyz(3) | feat_j(in_channels)]
        local_in = 3 + in_channels

        if use_x3d:
            self.struct_extractor = PCAPointHopGeometry()   # → 33-dim
            self.neighbor_ctx = NeighborContext(
                in_channels=local_in,
                struct_channels=struct_channels,
                out_channels=out_channels,
                hidden_channels=x3d_hidden,
            )
            # After neighbor context we apply InvResMLP
            self.post_mlp = nn.Sequential(
                nn.Linear(out_channels, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
            )
        else:
            # Standard local MLP + max pool
            self.local_mlp = nn.Sequential(
                nn.Linear(local_in, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
                nn.Linear(out_channels, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.GELU(),
            )

        # Skip projection if channel mismatch
        self.skip = (
            nn.Linear(in_channels, out_channels, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def _bn(self, net: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
        """Apply Sequential net to flattened input, reshape back."""
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        for m in net:
            flat = m(flat)
        return flat.reshape(shape[:-1] + (flat.shape[-1],))

    def forward(
        self,
        feat:    torch.Tensor,   # (B, N, C_in)
        pos:     torch.Tensor,   # (B, N, 3)
        center_feat: torch.Tensor,  # (B, M, C_in)  features at SA centres
        center_pos:  torch.Tensor,  # (B, M, 3)
        knn_idx:     torch.Tensor,  # (B, M, K)
    ) -> torch.Tensor:
        """Returns (B, M, C_out)."""
        B, M, K = knn_idx.shape

        # Gather neighbour positions and features  (B, M, K, *)
        nbr_pos  = _gather(pos,  knn_idx)   # (B, M, K, 3)
        nbr_feat = _gather(feat, knn_idx)   # (B, M, K, C_in)

        # Delta position (relative to centre)
        delta_xyz = nbr_pos - center_pos.unsqueeze(2)   # (B, M, K, 3)

        # Concatenate [delta_xyz | feat_j]
        local_in = torch.cat([delta_xyz, nbr_feat], dim=-1)  # (B, M, K, 3+C)

        if self.use_x3d:
            # Explicit geometry features  (B, M, 33)
            struct = self.struct_extractor(delta_xyz)

            # NeighborContext: dynamic weighting + conv  (B, M, C_out)
            agg = self.neighbor_ctx(local_in, struct)

            # Post-MLP
            shape = agg.shape
            flat = agg.reshape(-1, shape[-1])
            for m in self.post_mlp:
                flat = m(flat)
            agg = flat.reshape(shape[:-1] + (flat.shape[-1],))
        else:
            # Standard: flatten (B*M*K, C), apply MLP, max pool
            flat = local_in.reshape(B * M * K, -1)
            for m in self.local_mlp:
                flat = m(flat)
            flat = flat.reshape(B, M, K, -1)
            agg = flat.max(dim=2).values              # (B, M, C_out)

        # Residual connection
        skip = self.skip(center_feat)
        # skip: (B, M, C_out) — might need BN if dimensions changed
        return agg + skip


# ---------------------------------------------------------------------------
# Set Abstraction (SA) stage
# ---------------------------------------------------------------------------

class SetAbstraction(nn.Module):
    """
    PointNet++-style Set Abstraction with optional X-3D local aggregation.

    Args:
        in_channels:  Feature dim of input.
        out_channels: Feature dim of output.
        radius:       Ball-query radius (Bohr).
        K:            Number of neighbours per query point.
        npoint:       Number of FPS centre points (None → global SA).
        use_x3d:      Enable X-3D structure encoding.
        n_blocks:     Number of InvResMLP blocks after local agg.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        radius: float,
        K: int,
        npoint: int | None,
        use_x3d: bool = False,
        n_blocks: int = 1,
    ) -> None:
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.K = K

        self.local_agg = LocalAgg(in_channels, out_channels, use_x3d=use_x3d)
        self.blocks = nn.ModuleList(
            [InvResMLP(out_channels) for _ in range(n_blocks)]
        )

    def forward(
        self,
        feat: torch.Tensor,  # (B, N, C_in)
        pos:  torch.Tensor,  # (B, N, 3)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (new_feat (B, M, C_out), new_pos (B, M, 3))."""
        B, N, _ = pos.shape

        if self.npoint is None or self.npoint >= N:
            # Global SA: single centre at the centroid (used in last stage)
            center_idx = None
            center_pos = pos.mean(dim=1, keepdim=True)    # (B, 1, 3)
            center_feat = feat.mean(dim=1, keepdim=True)  # (B, 1, C)
            # Use all points as neighbours
            knn_idx = torch.arange(N, device=pos.device)
            knn_idx = knn_idx.unsqueeze(0).unsqueeze(0).expand(B, 1, N)
            # Limit to K
            if N > self.K:
                knn_idx = knn_idx[:, :, :self.K]
        else:
            # FPS centres
            center_idx = _fps(pos, self.npoint)            # (B, M)
            center_pos = _gather(pos, center_idx)          # (B, M, 3)
            center_feat = _gather(feat, center_idx)        # (B, M, C)
            # Ball query
            knn_idx = _ball_query(pos, center_pos, self.radius, self.K)  # (B,M,K)

        new_feat = self.local_agg(feat, pos, center_feat, center_pos, knn_idx)

        # InvResMLP refinement blocks
        for blk in self.blocks:
            shape = new_feat.shape
            flat = new_feat.reshape(-1, shape[-1])
            flat = blk(flat)
            new_feat = flat.reshape(shape)

        return new_feat, center_pos


# ---------------------------------------------------------------------------
# Full PointMetaBase-S-X3D model
# ---------------------------------------------------------------------------

class PointMetaBaseX3D(nn.Module):
    """
    PointMetaBase-S-X3D: 6-stage hierarchical point-cloud encoder
    with X-3D explicit structure encoding in stages 3 and 4.

    Config mirrors EDBench's pointmetabase-s-x-3d.yaml:
      blocks:   [1, 1, 1, 1, 1, 1]
      strides:  [1, 2, 2, 2, 2, 1]   (stride-1 → keep npoint; stride-2 → halve)
      width:    32
      in_channels: 4                  (x, y, z, density)
      radius:   0.15, multiplier 1.5
      K:        32
      x3d_layers: {3, 4}             (1-indexed stages)
      num_targets: 6                  (energy components)
      mlp_layers: [512, 256]

    Args:
        in_channels:    Number of input feature channels (default 4).
        width:          Base channel width (default 32).
        num_targets:    Output dimension (default 6).
        npoint_start:   Number of input points (default 2048).
        radius:         Base ball-query radius (default 0.15 Bohr).
        radius_mult:    Radius multiplier per stage (default 1.5).
        K:              Neighbours per query (default 32).
        mlp_layers:     RegressionHead hidden layers.
        dropout:        Dropout in regression head.
    """

    # Channel widths per stage: width × [1, 2, 4, 8, 8, 8]
    _CHANNEL_MULT = [1, 2, 4, 8, 8, 8]
    # Stage indices (1-based) where X-3D structure encoding is applied
    _X3D_STAGES = {3, 4}

    def __init__(
        self,
        in_channels: int = 4,
        width: int = 32,
        num_targets: int = 6,
        npoint_start: int = 2048,
        radius: float = 0.15,
        radius_mult: float = 1.5,
        K: int = 32,
        mlp_layers: list[int] = None,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if mlp_layers is None:
            mlp_layers = [512, 256]

        strides = [1, 2, 2, 2, 2, 1]
        blocks  = [1, 1, 1, 1, 1, 1]

        # Compute npoints and channels per stage
        npoints = [npoint_start]
        for s in strides[1:]:
            npoints.append(max(npoints[-1] // s, 1))

        channels = [in_channels] + [width * m for m in self._CHANNEL_MULT]

        # Build SA stages
        self.stages = nn.ModuleList()
        r = radius
        for i in range(6):
            stage_idx = i + 1   # 1-based
            npt = npoints[i + 1] if strides[i] == 2 else npoints[i]
            if stage_idx == 6:
                npt = None   # global SA in last stage

            use_x3d = stage_idx in self._X3D_STAGES

            sa = SetAbstraction(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                radius=r,
                K=K,
                npoint=npt,
                use_x3d=use_x3d,
                n_blocks=blocks[i],
            )
            self.stages.append(sa)
            r *= radius_mult

        # Global descriptor dimension: last stage output (global SA → 1 point)
        global_dim = channels[-1]

        # Regression head
        self.head = RegressionHead(
            in_channels=global_dim,
            mlp_layers=mlp_layers,
            num_targets=num_targets,
            dropout=dropout,
        )

    # ------------------------------------------------------------------
    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        Args:
            point_cloud: (B, N, 4)  – [x, y, z, density]
        Returns:
            energies: (B, num_targets)
        """
        pos  = point_cloud[..., :3]       # (B, N, 3)
        feat = point_cloud                 # (B, N, 4)  – use all 4 as features

        for sa in self.stages:
            feat, pos = sa(feat, pos)

        # After last (global) SA: feat is (B, 1, C_last); squeeze
        feat = feat.squeeze(1)             # (B, C_last)

        return self.head(feat)             # (B, num_targets)
