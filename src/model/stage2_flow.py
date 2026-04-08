"""
Stage 2: Flow Matching for electron density value generation.

Given K representative point positions {r_j} produced by Stage 1, this
network generates log-density values {z_j = log(ρ_j + ε)} via flow matching.

Architecture
------------
The network is invariant (outputs scalars) and operates on a 1-D sequence of
K density values z_t ∈ ℝ^K.  It conditions on:
  - Point positions {r_j} ∈ ℝ^{K×3}  (equivariant inputs, but used invariantly
    via pairwise distances and relative-distance features)
  - Atom features {h_i} ∈ ℝ^{N×C}   (from VisNet, invariant scalar features)
  - Time t ∈ [0, 1]

The velocity field predicts how z_t should evolve (a scalar per point).

Model blocks
  1. PointAtomAttention  – cross-attention from K points to N atoms using
     distance-based keys (invariant to global rotation/translation).
  2. PointSelfAttention  – self-attention among K points using pairwise
     distances as relative biases (invariant).
  3. A lightweight MLP per point that maps the accumulated features + noisy
     z_t values + time to a velocity scalar.

CFG
---
Same 15% drop probability as Stage 1.  Null embedding replaces atom features.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


from .stage1_flow import SinusoidalTimeEmbedding


# ---------------------------------------------------------------------------
# Invariant distance feature encoder
# ---------------------------------------------------------------------------

class RBFEncoder(nn.Module):
    """Encode scalar distances with a radial basis function expansion."""

    def __init__(self, out_dim: int, num_rbf: int = 32, cutoff: float = 8.0) -> None:
        super().__init__()
        self.cutoff = cutoff
        centers = torch.linspace(0.0, cutoff, num_rbf)
        self.register_buffer("centers", centers)
        self.width = (cutoff / num_rbf) ** 2
        self.proj = nn.Linear(num_rbf, out_dim)

    def forward(self, dist: Tensor) -> Tensor:
        """dist: (...,) → (..., out_dim)"""
        rbf = torch.exp(-((dist[..., None] - self.centers) ** 2) / self.width)
        return self.proj(rbf)


# ---------------------------------------------------------------------------
# Cross-attention: point ← atoms
# ---------------------------------------------------------------------------

class PointAtomCrossAttention(nn.Module):
    """
    Each point j attends over all atoms i in the same molecule.
    Keys/values from atoms; queries from point features + distance encoding.

    Distance-based relative bias makes this rotation-invariant.
    """

    def __init__(self, hidden: int, num_heads: int = 8, cutoff: float = 8.0) -> None:
        super().__init__()
        assert hidden % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.scale = self.head_dim ** -0.5

        self.rbf = RBFEncoder(num_heads, cutoff=cutoff)   # per-head distance bias

        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(
        self,
        h_point: Tensor,        # (sum_K, C)
        h_atom: Tensor,         # (sum_N, C)
        pos_point: Tensor,      # (sum_K, 3)
        pos_atom: Tensor,       # (sum_N, 3)
        point_batch: Tensor,    # (sum_K,)
        atom_batch: Tensor,     # (sum_N,)
    ) -> Tensor:
        B = int(point_batch.max().item()) + 1
        out_chunks = []

        for b in range(B):
            pm = point_batch == b
            am = atom_batch == b
            hp = h_point[pm]        # (K_b, C)
            ha = h_atom[am]         # (N_b, C)
            pp = pos_point[pm]      # (K_b, 3)
            pa = pos_atom[am]       # (N_b, 3)

            K_b, N_b = hp.shape[0], ha.shape[0]
            H, D = self.num_heads, self.head_dim

            q = self.q_proj(hp).view(K_b, H, D)    # (K_b, H, D)
            k = self.k_proj(ha).view(N_b, H, D)    # (N_b, H, D)
            v = self.v_proj(ha).view(N_b, H, D)    # (N_b, H, D)

            # Pairwise distances → per-head bias
            dist = torch.cdist(pp, pa)              # (K_b, N_b)
            bias = self.rbf(dist)                   # (K_b, N_b, H)
            bias = bias.permute(2, 0, 1)            # (H, K_b, N_b)

            # Scaled dot-product + distance bias
            attn = torch.einsum("khd,nhd->hkn", q, k) * self.scale + bias  # (H, K_b, N_b)
            attn = F.softmax(attn, dim=-1)

            agg = torch.einsum("hkn,nhd->khd", attn, v).reshape(K_b, H * D)  # (K_b, C)
            out_chunks.append(self.out_proj(agg))

        agg_full = torch.cat(out_chunks)            # (sum_K, C)
        return self.norm(h_point + agg_full)


# ---------------------------------------------------------------------------
# Self-attention among points
# ---------------------------------------------------------------------------

class PointSelfAttention(nn.Module):
    """
    Self-attention over K points within each molecule.
    Uses pairwise distances as relative bias (rotation-invariant).
    """

    def __init__(self, hidden: int, num_heads: int = 8, cutoff: float = 8.0) -> None:
        super().__init__()
        assert hidden % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden // num_heads
        self.scale = self.head_dim ** -0.5

        self.rbf = RBFEncoder(num_heads, cutoff=cutoff)

        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.out_proj = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(
        self,
        h: Tensor,           # (sum_K, C)
        pos: Tensor,         # (sum_K, 3)
        batch: Tensor,       # (sum_K,)
    ) -> Tensor:
        B = int(batch.max().item()) + 1
        out_chunks = []

        for b in range(B):
            m = batch == b
            h_b = h[m]              # (K_b, C)
            p_b = pos[m]            # (K_b, 3)
            K_b = h_b.shape[0]
            H, D = self.num_heads, self.head_dim

            qkv = self.qkv(h_b).chunk(3, dim=-1)
            q, k, v = [x.view(K_b, H, D) for x in qkv]

            dist = torch.cdist(p_b, p_b)            # (K_b, K_b)
            bias = self.rbf(dist).permute(2, 0, 1)  # (H, K_b, K_b)

            attn = torch.einsum("khd,jhd->hkj", q, k) * self.scale + bias
            attn = F.softmax(attn, dim=-1)

            agg = torch.einsum("hkj,jhd->khd", attn, v).reshape(K_b, H * D)
            out_chunks.append(self.out_proj(agg))

        agg_full = torch.cat(out_chunks)
        return self.norm(h + agg_full)


# ---------------------------------------------------------------------------
# Stage 2 network
# ---------------------------------------------------------------------------

class Stage2FlowNet(nn.Module):
    """
    Invariant flow-matching network for log-density value generation.

    Args:
        atom_in_channels:  VisNet atom feature dimension.
        hidden_channels:   Internal dimension.
        num_layers:        Number of cross-attention + self-attention blocks.
        num_heads:         Attention heads.
        cutoff:            Distance cutoff in Bohr (for RBF encoders).
        cfg_drop_prob:     CFG condition-drop probability during training.
    """

    def __init__(
        self,
        atom_in_channels: int = 256,
        hidden_channels: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        cutoff: float = 8.0,
        cfg_drop_prob: float = 0.15,
    ) -> None:
        super().__init__()
        self.cfg_drop_prob = cfg_drop_prob

        # --- Atom feature projection ---
        self.atom_proj = nn.Linear(atom_in_channels, hidden_channels)
        self.null_atom_feat = nn.Parameter(torch.zeros(hidden_channels))

        # --- Time embedding ---
        self.time_emb = SinusoidalTimeEmbedding(hidden_channels)

        # --- Noisy z_t + position → initial point features ---
        self.point_input_proj = nn.Sequential(
            nn.Linear(1 + hidden_channels, hidden_channels),  # z_t + time_emb
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # --- Transformer blocks (cross + self attention) ---
        self.cross_attn_layers = nn.ModuleList(
            [PointAtomCrossAttention(hidden_channels, num_heads=num_heads, cutoff=cutoff)
             for _ in range(num_layers)]
        )
        self.self_attn_layers = nn.ModuleList(
            [PointSelfAttention(hidden_channels, num_heads=num_heads, cutoff=cutoff)
             for _ in range(num_layers)]
        )

        # --- Output: scalar velocity per point ---
        self.vel_out = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(
        self,
        z_t: Tensor,             # (sum_K,)    noisy log-density values at time t
        t_query: Tensor,         # (sum_K,)    time per point
        point_pos: Tensor,       # (sum_K, 3)  point positions (from Stage 1)
        atom_pos: Tensor,        # (sum_N, 3)  atom positions
        atom_feat: Tensor,       # (sum_N, C)  VisNet invariant features
        point_batch: Tensor,     # (sum_K,)
        atom_batch: Tensor,      # (sum_N,)
        drop_condition: bool = False,
    ) -> Tensor:
        """
        Returns velocity scalars (sum_K,) for the log-density values.
        """
        # --- CFG condition drop ---
        h_atom = self.atom_proj(atom_feat)
        if drop_condition or (self.training and torch.rand(1).item() < self.cfg_drop_prob):
            h_atom = self.null_atom_feat.expand_as(h_atom)

        # --- Initial point features from z_t and time ---
        t_emb = self.time_emb(t_query)                      # (sum_K, C)
        point_input = torch.cat([z_t.unsqueeze(-1), t_emb], dim=-1)  # (sum_K, 1+C)
        h_point = self.point_input_proj(point_input)         # (sum_K, C)

        # --- Iterative attention ---
        for cross_attn, self_attn in zip(self.cross_attn_layers, self.self_attn_layers):
            h_point = cross_attn(h_point, h_atom, point_pos, atom_pos, point_batch, atom_batch)
            h_point = self_attn(h_point, point_pos, point_batch)

        # --- Predict velocity (scalar) ---
        vel = self.vel_out(h_point).squeeze(-1)             # (sum_K,)
        return vel
