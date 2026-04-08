"""
VisNet encoder wrapper.

Wraps PyTorch Geometric's VisNet to produce:
  - x   : (N_atoms, hidden_channels)     invariant scalar features
  - vec : (N_atoms, 3, hidden_channels)  equivariant vector features

These are used as the condition for both Stage 1 and Stage 2 flow matching.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

try:
    from torch_geometric.nn.models import ViSNet as _PYGVisNet
except ImportError as e:
    raise ImportError(
        "torch_geometric is required for VisNetEncoder. "
        "Install it with: pip install torch_geometric"
    ) from e


class VisNetEncoder(nn.Module):
    """
    VisNet encoder that produces per-atom invariant and equivariant features.

    Args:
        hidden_channels: Feature dimension.
        num_layers:      Number of VisNet message-passing layers.
        num_rbf:         Number of radial basis functions.
        trainable_rbf:   Whether RBF centers/widths are learnable.
        max_z:           Maximum atomic number (H–Rf covers 1-104).
        cutoff:          Interaction cutoff in Bohr.
        num_heads:       Number of attention heads.
        vertex:          If True, use the vertex-variant of VisNet which
                         additionally produces equivariant vector features.
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
        vertex: bool = True,
        pretrained_path: str | None = None,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.vertex = vertex

        self.visnet = _PYGVisNet(
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_rbf=num_rbf,
            trainable_rbf=trainable_rbf,
            max_z=max_z,
            cutoff=cutoff,
            num_heads=num_heads,
            vertex=vertex,
        )

        if pretrained_path is not None:
            ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            if "representation_model" in ckpt:
                state = ckpt["representation_model"]
            else:
                # Strip "representation_model." prefix from full model weights
                state = {
                    k.removeprefix("representation_model."): v
                    for k, v in ckpt["full_model"].items()
                    if k.startswith("representation_model.")
                }
            self.visnet.representation_model.load_state_dict(state, strict=True)
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
            x:   (N_atoms, hidden_channels)     invariant features.
            vec: (N_atoms, 3, hidden_channels)  equivariant features,
                 or None when vertex=False.
        """
        # Use representation_model to get per-atom features.
        # self.visnet.forward() is a property-prediction head that reduces
        # to per-molecule scalars — not what we want here.
        x, vec = self.visnet.representation_model(z, pos, batch)
        if not self.vertex:
            vec = None
        return x, vec
