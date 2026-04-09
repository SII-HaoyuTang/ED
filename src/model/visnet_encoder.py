"""
VisNet encoder wrapper.

Wraps the local ViSNetBlock (src/model/visnet/models/visnet_block.py) to produce:
  - x   : (N_atoms, hidden_channels)     invariant scalar features
  - vec : (N_atoms, 8, hidden_channels)  equivariant vector features (lmax=2)

These are used as the condition for both Stage 1 and Stage 2 flow matching.

Note: vec shape is (N, 8, C) with lmax=2 (8 spherical harmonic components).
Stage 1/2 training ignores vec (_), so no impact downstream.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .visnet.models.visnet_block import ViSNetBlock as _ViSNetBlock


class VisNetEncoder(nn.Module):
    """
    VisNet encoder that produces per-atom invariant and equivariant features.

    Uses the original ViSNetBlock from src/model/visnet/models/visnet_block.py.

    Args:
        hidden_channels: Feature dimension.
        num_layers:      Number of VisNet message-passing layers.
        num_rbf:         Number of radial basis functions.
        trainable_rbf:   Whether RBF parameters are learnable.
        max_z:           Maximum atomic number.
        cutoff:          Interaction cutoff in Angstrom/Bohr (same unit as pos).
        num_heads:       Number of attention heads.
        lmax:            Maximum angular momentum (default 2 → 8 spherical components).
        vertex:          If True use ViS_MP_Vertex_Edge layers (richer features).
        pretrained_path: Path to a checkpoint whose "representation_model" key
                         contains the ViSNetBlock state dict.
    """

    def __init__(
        self,
        hidden_channels: int = 256,
        num_layers: int = 6,
        num_rbf: int = 32,
        trainable_rbf: bool = False,
        max_z: int = 100,
        cutoff: float = 5.0,
        num_heads: int = 8,
        lmax: int = 2,
        vertex: bool = True,
        pretrained_path: str | None = None,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.lmax = lmax
        self.vertex = vertex

        self.block = _ViSNetBlock(
            lmax=lmax,
            vecnorm_type="max_min",
            trainable_vecnorm=False,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_channels=hidden_channels,
            num_rbf=num_rbf,
            rbf_type="expnorm",
            trainable_rbf=trainable_rbf,
            activation="silu",
            attn_activation="silu",
            max_z=max_z,
            cutoff=cutoff,
            max_num_neighbors=32,
            vertex_type="Edge" if vertex else "None",
        )

        if pretrained_path is not None:
            ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            if "representation_model" in ckpt:
                state = ckpt["representation_model"]
            else:
                # Strip "representation_model." prefix from full_model weights
                state = {
                    k.removeprefix("representation_model."): v
                    for k, v in ckpt["full_model"].items()
                    if k.startswith("representation_model.")
                }
            self.block.load_state_dict(state, strict=True)
            print(f"[VisNetEncoder] Loaded pretrained weights from {pretrained_path}")

    def forward(
        self,
        z: Tensor,
        pos: Tensor,
        batch: Tensor,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Args:
            z:     (N_atoms,)    atomic numbers (int64).
            pos:   (N_atoms, 3) atomic positions.
            batch: (N_atoms,)   batch indices.

        Returns:
            x:   (N_atoms, hidden_channels)       invariant scalar features.
            vec: (N_atoms, 8, hidden_channels)    equivariant vector features
                 (lmax=2 yields 8 spherical components), or None when vertex=False.
        """
        from torch_geometric.data import Data

        data = Data(z=z, pos=pos, batch=batch)
        x, vec = self.block(data)
        if not self.vertex:
            vec = None
        return x, vec
