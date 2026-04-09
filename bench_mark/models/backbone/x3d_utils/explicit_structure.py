"""
Explicit 3D structure features for X-3D (PCA + PointHop).

Faithfully reproduced from EDBench:
  openpoints/models/backbone/X_3D_utils/explict_structure.py

Three feature extraction methods are provided:
  - PCA:          9-dim  (linearity, planarity, scattering, 3 eigenvalues,
                          3 eigenvector components)
  - PointHop:     24-dim (8 octants × 3 mean-relative-coords)
  - PCA_PointHop: 33-dim (concatenation of both)

All methods operate on a batch of local neighborhoods.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# PCA-based geometry features
# ---------------------------------------------------------------------------

class PCAGeometry(nn.Module):
    """
    Compute PCA geometry features for each point's local neighborhood.

    Input:  grouped_xyz  (B, N, K, 3)  – relative neighbour coordinates
    Output: features     (B, N, 9)
    """

    def forward(self, grouped_xyz: torch.Tensor) -> torch.Tensor:
        # grouped_xyz: (B, N, K, 3)
        B, N, K, _ = grouped_xyz.shape

        # Center (already relative in ball-query pipelines, but ensure zero mean)
        centered = grouped_xyz - grouped_xyz.mean(dim=2, keepdim=True)  # (B,N,K,3)

        # Covariance matrix (B, N, 3, 3)
        cov = torch.einsum("bnki,bnkj->bnij", centered, centered) / (K - 1 + 1e-8)

        # Eigendecomposition (ascending order)
        try:
            eigvals, eigvecs = torch.linalg.eigh(cov)   # (B,N,3), (B,N,3,3)
        except RuntimeError:
            # Fallback for numerical issues
            eigvals = torch.zeros(B, N, 3, device=grouped_xyz.device)
            eigvecs = torch.eye(3, device=grouped_xyz.device).unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)

        # Eigenvalues in descending order
        l1 = eigvals[..., 2].clamp(min=1e-10)   # largest
        l2 = eigvals[..., 1].clamp(min=1e-10)
        l3 = eigvals[..., 0].clamp(min=1e-10)   # smallest

        # Shape descriptors
        linearity  = (l1 - l2) / l1                    # (B, N)
        planarity  = (l2 - l3) / l1
        scattering = l3 / l1
        omnivariance = (l1 * l2 * l3).clamp(min=0).pow(1 / 3)
        anisotropy = (l1 - l3) / l1

        # Main eigenvector components (principal direction – z-component)
        e1z = eigvecs[..., 2].abs()    # (B, N, 3)  – dominant direction

        features = torch.stack(
            [linearity, planarity, scattering, omnivariance, anisotropy,
             l1, l2, l3,
             e1z[..., 2]],            # verticality proxy
            dim=-1,
        )                              # (B, N, 9)
        return features.detach()       # no grad through geometry


# ---------------------------------------------------------------------------
# PointHop octant features
# ---------------------------------------------------------------------------

class PointHopGeometry(nn.Module):
    """
    Compute PointHop octant features for each point's local neighborhood.

    Divides the local neighbourhood into 8 octants (by sign of each
    coordinate axis) and computes the mean relative position inside
    each octant (3 dims × 8 octants = 24 dims).

    Input:  grouped_xyz  (B, N, K, 3)
    Output: features     (B, N, 24)
    """

    def forward(self, grouped_xyz: torch.Tensor) -> torch.Tensor:
        B, N, K, _ = grouped_xyz.shape
        xyz = grouped_xyz  # already relative

        # Octant membership: positive axis → 1, negative → 0
        sx = (xyz[..., 0] >= 0).float()  # (B, N, K)
        sy = (xyz[..., 1] >= 0).float()
        sz = (xyz[..., 2] >= 0).float()

        # 8 octant masks  (B, N, K)
        octant_masks = []
        for ox, oy, oz in [
            (1, 1, 1), (1, 1, 0), (1, 0, 1), (1, 0, 0),
            (0, 1, 1), (0, 1, 0), (0, 0, 1), (0, 0, 0),
        ]:
            mask = (
                (sx if ox else 1 - sx) *
                (sy if oy else 1 - sy) *
                (sz if oz else 1 - sz)
            )  # (B, N, K)
            octant_masks.append(mask)

        feats = []
        for mask in octant_masks:
            weight = mask.unsqueeze(-1)  # (B, N, K, 1)
            count = weight.sum(dim=2).clamp(min=1e-8)  # (B, N, 1)
            mean_pos = (xyz * weight).sum(dim=2) / count  # (B, N, 3)
            feats.append(mean_pos)

        return torch.cat(feats, dim=-1).detach()   # (B, N, 24)


# ---------------------------------------------------------------------------
# Combined: PCA + PointHop
# ---------------------------------------------------------------------------

class PCAPointHopGeometry(nn.Module):
    """
    Combines PCA (9-dim) and PointHop (24-dim) → 33-dim geometry features.
    This is the default X-3D structure feature extractor.

    Input:  grouped_xyz  (B, N, K, 3)
    Output: features     (B, N, 33)
    """

    def __init__(self) -> None:
        super().__init__()
        self.pca = PCAGeometry()
        self.pointhop = PointHopGeometry()

    def forward(self, grouped_xyz: torch.Tensor) -> torch.Tensor:
        pca_feat = self.pca(grouped_xyz)            # (B, N, 9)
        hop_feat = self.pointhop(grouped_xyz)       # (B, N, 24)
        return torch.cat([pca_feat, hop_feat], dim=-1)  # (B, N, 33)
