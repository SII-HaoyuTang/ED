"""
Stage 1: Equivariant Flow Matching for point-cloud position generation.

Architecture
------------
Given:
  - K noisy query positions  x_t  ∈ ℝ^{K×3}  (current flow state)
  - N atom positions          R    ∈ ℝ^{N×3}  (fixed condition)
  - N atom features           h    ∈ ℝ^{N×C}  (from VisNet)
  - time step                 t    ∈ [0, 1]

Predicts:
  - Velocity  v  ∈ ℝ^{K×3}  (SE(3)-equivariant)

Equivariance is achieved via an EGNN-style architecture:
  v_j = Σ_i  (x_j - R_i) · φ(h_i, h_j, d²_ij, t)
         + Σ_k  (x_j - x_k) · ψ(h_j, h_k, d²_jk, t)

where φ and ψ are invariant MLPs, and the sums of weighted relative positions
are SE(3)-equivariant vectors.

CFG
---
During training the condition (atom features) is dropped with probability
`cfg_drop_prob`, replaced by a learned null embedding. At inference,
guided velocity is computed as:
  v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn import radius


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for the flow time t ∈ [0, 1]."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        """t: (B,) → (B, dim)"""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None] * freqs[None, :]           # (B, half)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)
        return self.proj(emb)


class EGNNLayer(nn.Module):
    """
    Single EGNN-style equivariant message-passing layer.

    Operates on a heterogeneous bipartite graph:
      source nodes  →  target (query) nodes

    Updates query features and accumulates equivariant velocity contributions.
    """

    def __init__(self, src_channels: int, tgt_channels: int, time_channels: int) -> None:
        super().__init__()
        msg_in = src_channels + tgt_channels + 1 + time_channels  # +1 for d²

        self.message_mlp = nn.Sequential(
            nn.Linear(msg_in, tgt_channels * 2),
            nn.SiLU(),
            nn.Linear(tgt_channels * 2, tgt_channels),
            nn.SiLU(),
        )
        # Scalar weight for the coordinate update (velocity contribution)
        self.coord_weight = nn.Sequential(
            nn.Linear(tgt_channels, tgt_channels // 2),
            nn.SiLU(),
            nn.Linear(tgt_channels // 2, 1),
            nn.Tanh(),          # bounded weights → stable training
        )
        self.node_update = nn.Sequential(
            nn.Linear(tgt_channels * 2, tgt_channels),
            nn.SiLU(),
            nn.Linear(tgt_channels, tgt_channels),
        )
        self.norm = nn.LayerNorm(tgt_channels)

    def forward(
        self,
        h_src: Tensor,           # (M_src, C_src)
        h_tgt: Tensor,           # (M_tgt, C_tgt)
        pos_src: Tensor,         # (M_src, 3)
        pos_tgt: Tensor,         # (M_tgt, 3)
        t_emb_tgt: Tensor,       # (M_tgt, C_time)  – time emb broadcast to each tgt node
        edge_src: Tensor,        # (E,) source indices
        edge_tgt: Tensor,        # (E,) target indices
    ) -> tuple[Tensor, Tensor]:
        """
        Returns:
            h_tgt_new:       (M_tgt, C_tgt)  updated features
            vel_contrib:     (M_tgt, 3)       equivariant velocity contribution
        """
        # --- Relative positions and distances ---
        rel = pos_tgt[edge_tgt] - pos_src[edge_src]    # (E, 3)  equivariant
        sq_dist = (rel ** 2).sum(dim=-1, keepdim=True)  # (E, 1)  invariant

        # --- Message ---
        msg_input = torch.cat(
            [h_src[edge_src], h_tgt[edge_tgt], sq_dist, t_emb_tgt[edge_tgt]],
            dim=-1,
        )
        msg = self.message_mlp(msg_input)               # (E, C_tgt)

        # --- Equivariant coordinate contribution ---
        w = self.coord_weight(msg)                      # (E, 1)
        weighted_rel = w * rel                          # (E, 3)

        vel_contrib = torch.zeros_like(pos_tgt)
        vel_contrib.index_add_(0, edge_tgt, weighted_rel)  # scatter sum

        # --- Feature aggregation ---
        agg = torch.zeros_like(h_tgt)
        agg.index_add_(0, edge_tgt, msg)                # scatter sum

        h_tgt_new = self.norm(
            h_tgt + self.node_update(torch.cat([h_tgt, agg], dim=-1))
        )
        return h_tgt_new, vel_contrib


class Stage1FlowNet(nn.Module):
    """
    Equivariant flow-matching network for electron-density point positions.

    Args:
        atom_in_channels:   Dimension of VisNet atom features.
        hidden_channels:    Internal feature dimension for query nodes.
        num_layers:         Number of EGNN update rounds (atom→query + query→query).
        cutoff:             Radius graph cutoff for edge construction (Bohr).
        cfg_drop_prob:      Probability of dropping the condition during training.
    """

    def __init__(
        self,
        atom_in_channels: int = 256,
        hidden_channels: int = 128,
        num_layers: int = 4,
        cutoff: float = 8.0,
        cfg_drop_prob: float = 0.15,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.cfg_drop_prob = cfg_drop_prob

        # Time embedding
        self.time_emb = SinusoidalTimeEmbedding(hidden_channels)

        # Project atom features to hidden_channels
        self.atom_proj = nn.Linear(atom_in_channels, hidden_channels)

        # Null embedding for CFG (replaces atom features when dropped)
        self.null_atom_feat = nn.Parameter(torch.zeros(hidden_channels))

        # Initialise query node features from log-compressed mean distance to atoms.
        # Input is log(1 + mean_dist / cutoff) ∈ [0, ~2], well-scaled.
        self.query_init = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # EGNN layers: atom→query
        self.atom_to_query_layers = nn.ModuleList(
            [EGNNLayer(hidden_channels, hidden_channels, hidden_channels)
             for _ in range(num_layers)]
        )
        # EGNN layers: query→query (self-interaction)
        self.query_to_query_layers = nn.ModuleList(
            [EGNNLayer(hidden_channels, hidden_channels, hidden_channels)
             for _ in range(num_layers)]
        )

        # Final velocity scale: zero-initialised so the model starts with
        # near-zero velocity, avoiding large initial losses and gradient spikes.
        self.vel_out = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )
        nn.init.zeros_(self.vel_out[-1].weight)
        nn.init.zeros_(self.vel_out[-1].bias)

    # ------------------------------------------------------------------
    def forward(
        self,
        x_t: Tensor,            # (sum_K, 3)  noisy query positions
        t_query: Tensor,        # (sum_K,)    time per query point
        atom_pos: Tensor,       # (sum_N, 3)  atom positions
        atom_feat: Tensor,      # (sum_N, C)  VisNet features
        point_batch: Tensor,    # (sum_K,)    molecule index per query point
        atom_batch: Tensor,     # (sum_N,)    molecule index per atom
        drop_condition: bool = False,
    ) -> Tensor:
        """
        Returns velocity vectors v ∈ ℝ^{sum_K × 3} (equivariant).
        """
        # --- Condition dropout (CFG) ---
        h_atom = self.atom_proj(atom_feat)          # (sum_N, C)
        if drop_condition or (self.training and torch.rand(1).item() < self.cfg_drop_prob):
            h_atom = self.null_atom_feat.expand_as(h_atom)

        # --- Time embedding per query node ---
        t_emb = self.time_emb(t_query)              # (sum_K, C)

        # --- Radius graphs ---
        # atom → query edges
        edge_aq_src, edge_aq_tgt = self._build_radius_edges(
            atom_pos, x_t, atom_batch, point_batch
        )
        # query → query edges (self-interaction within each molecule)
        edge_qq_src, edge_qq_tgt = self._build_radius_edges(
            x_t, x_t, point_batch, point_batch, exclude_self=True
        )

        # --- Initialise query features from nearest-atom distances ---
        # Use mean squared distance to atoms as a simple invariant init
        h_query = self._init_query_features(x_t, atom_pos, point_batch, atom_batch)

        # --- Iterative EGNN updates ---
        vel = torch.zeros_like(x_t)               # accumulated equivariant velocity
        for a2q, q2q in zip(self.atom_to_query_layers, self.query_to_query_layers):
            h_query, dv = a2q(
                h_atom, h_query,
                atom_pos, x_t,
                t_emb,
                edge_aq_src, edge_aq_tgt,
            )
            vel = vel + dv

            h_query, dv = q2q(
                h_query, h_query,
                x_t, x_t,
                t_emb,
                edge_qq_src, edge_qq_tgt,
            )
            vel = vel + dv

        # --- Global scale factor from query features ---
        scale = self.vel_out(h_query)              # (sum_K, 1)
        vel = vel * scale

        return vel                                  # (sum_K, 3)

    # ------------------------------------------------------------------
    def _init_query_features(
        self,
        pos_q: Tensor,
        pos_a: Tensor,
        batch_q: Tensor,
        batch_a: Tensor,
    ) -> Tensor:
        """
        Initialise query node features from log-compressed mean distance to atoms.

        Using log(1 + mean_dist / cutoff) compresses the raw Bohr distances
        (which can exceed 20 Bohr) into a bounded [0, ~2] range, preventing
        the MLP from receiving extremely large inputs.
        """
        B = int(batch_q.max().item()) + 1
        feats = []
        ptr_q = _batch_ptr(batch_q, B)
        ptr_a = _batch_ptr(batch_a, B)
        for b in range(B):
            q = pos_q[ptr_q[b]:ptr_q[b + 1]]     # (K_b, 3)
            a = pos_a[ptr_a[b]:ptr_a[b + 1]]     # (N_b, 3)
            sq = ((q[:, None] - a[None]) ** 2).sum(-1)      # (K_b, N_b)
            mean_dist = sq.mean(-1, keepdim=True).sqrt()    # (K_b, 1)
            normalized = torch.log1p(mean_dist / self.cutoff)  # (K_b, 1), in [0, ~2]
            feats.append(normalized)
        log_dist = torch.cat(feats)                # (sum_K, 1)
        return self.query_init(log_dist)           # (sum_K, C)

    def _build_radius_edges(
        self,
        pos_src: Tensor,
        pos_tgt: Tensor,
        batch_src: Tensor,
        batch_tgt: Tensor,
        exclude_self: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Return (src_indices, tgt_indices) for pairs within cutoff."""
        assign = radius(
            pos_src, pos_tgt,
            r=self.cutoff,
            batch_x=batch_src,
            batch_y=batch_tgt,
            max_num_neighbors=64,
        )
            # radius returns (2, E) with row0=tgt, row1=src
        edge_tgt, edge_src = assign[0], assign[1]
        if exclude_self:
            mask = edge_src != edge_tgt
            edge_src, edge_tgt = edge_src[mask], edge_tgt[mask]
        return edge_src, edge_tgt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _batch_ptr(batch: Tensor, B: int) -> list[int]:
    """Return list of [start, end) indices per batch element."""
    ptr = [0] * (B + 1)
    for i in range(len(batch)):
        ptr[batch[i].item() + 1] += 1
    for b in range(B):
        ptr[b + 1] += ptr[b]
    return ptr
