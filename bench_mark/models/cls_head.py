"""
Regression head (MLP) for point-cloud energy prediction.

Faithfully reproduced from EDBench:
  openpoints/models/classification/cls_base.py  (ClsHead)

Takes a global feature vector and predicts num_targets scalar values.
Architecture: Linear → BN → ReLU → Dropout → Linear → BN → ReLU
              → Dropout → Linear (output).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RegressionHead(nn.Module):
    """
    MLP regression head that maps a global descriptor to energy predictions.

    Args:
        in_channels:  Dimension of the global point-cloud feature.
        mlp_layers:   Hidden layer widths (default [512, 256], matching paper).
        num_targets:  Number of output scalars (6 for ED5-EC).
        dropout:      Dropout probability (default 0.5, matching paper).
    """

    def __init__(
        self,
        in_channels: int,
        mlp_layers: list[int] = None,
        num_targets: int = 6,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if mlp_layers is None:
            mlp_layers = [512, 256]

        layers: list[nn.Module] = []
        prev = in_channels
        for h in mlp_layers:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, num_targets))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_channels)
        Returns:
            (B, num_targets)
        """
        return self.mlp(x)
