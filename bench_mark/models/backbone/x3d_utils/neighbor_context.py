"""
Neighbour-context aggregation module for X-3D.

Faithfully reproduced from EDBench:
  openpoints/models/backbone/X_3D_utils/neighbor_context.py

NeighborContext aggregates features within a local neighbourhood using
a Conv1d pipeline followed by a max-pooling reduction, and then applies
the dynamic structure kernel (linear weight generation) from X-3D.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class NeighborContext(nn.Module):
    """
    Aggregate neighbor features and apply a dynamic structure kernel.

    Corresponds to EDBench's Neighbor_Context + X3D_Model structure kernel.

    Args:
        in_channels:     Input feature dimension per neighbor.
        struct_channels: Dimension of the explicit geometry features (33 for
                         PCA_PointHop).
        out_channels:    Output feature dimension.
        hidden_channels: Hidden dim for the dynamic weight generator.
    """

    def __init__(
        self,
        in_channels: int,
        struct_channels: int,
        out_channels: int,
        hidden_channels: int = 32,
    ) -> None:
        super().__init__()

        # 1. Project structure features to dynamic weights
        self.weight_gen = nn.Sequential(
            nn.Linear(struct_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, in_channels),
            nn.Sigmoid(),
        )

        # 2. Channel-mixing Conv1d pipeline (applied per-neighbour-set)
        self.conv_pipe = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        neighbor_feats: torch.Tensor,    # (B, N, K, C_in)
        struct_feats:   torch.Tensor,    # (B, N, C_struct)
    ) -> torch.Tensor:
        """
        Returns aggregated feature (B, N, C_out) via dynamic re-weighting +
        max-pooling.
        """
        B, N, K, C_in = neighbor_feats.shape

        # Dynamic attention weights from geometry features → (B, N, C_in)
        weights = self.weight_gen(struct_feats)          # (B, N, C_in)

        # Re-weight each neighbor's features
        weighted = neighbor_feats * weights.unsqueeze(2) # (B, N, K, C_in)

        # Max pool over neighbours → (B, N, C_in)
        pooled = weighted.max(dim=2).values              # (B, N, C_in)

        # Conv1d expects (B, C, L); treat N as the sequence dimension
        out = self.conv_pipe(pooled.transpose(1, 2))     # (B, C_out, N)
        return out.transpose(1, 2)                       # (B, N, C_out)
