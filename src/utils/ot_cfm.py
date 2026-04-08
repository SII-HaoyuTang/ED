"""
OT-CFM (Optimal-Transport Conditional Flow Matching) utilities.

Reference: Lipman et al. "Flow Matching for Generative Modeling" (2022).
           Tong et al.   "Improving and Generalizing Flow Matching" (2023).

Core idea
---------
Given a source sample x_0 ~ N(0,I) and a target sample x_1 (from data),
define the linear conditional path:

    x_t = (1 - t) * x_0 + t * x_1          t ∈ [0, 1]

The target velocity at x_t is:

    u_t(x_t | x_1) = x_1 - x_0

The CFM loss is:

    L = E_{t, x_0, x_1} [ || v_θ(x_t, t) - (x_1 - x_0) ||² ]

For the scalar density stage (Stage 2) the same formula applies in ℝ.
"""
from __future__ import annotations

import torch
from torch import Tensor


def sample_t(batch_size: int, device: torch.device) -> Tensor:
    """Sample t ∼ Uniform[0, 1] for a batch of size batch_size."""
    return torch.rand(batch_size, device=device)


def interpolate(x_0: Tensor, x_1: Tensor, t: Tensor) -> Tensor:
    """
    Linear interpolation along the flow path.

    Args:
        x_0: (B, ...) source sample (noise).
        x_1: (B, ...) target sample (data).
        t:   (B,) or scalar, time in [0, 1].

    Returns:
        x_t: (B, ...) interpolated point.
    """
    # Broadcast t to match x_0 shape
    while t.dim() < x_0.dim():
        t = t.unsqueeze(-1)
    return (1.0 - t) * x_0 + t * x_1


def cfm_target_velocity(x_0: Tensor, x_1: Tensor) -> Tensor:
    """
    Target velocity u_t = x_1 - x_0 (constant along each path).

    Args:
        x_0: (B, ...) noise.
        x_1: (B, ...) data.

    Returns:
        u_t: (B, ...) target velocity.
    """
    return x_1 - x_0


def cfm_loss(v_pred: Tensor, x_0: Tensor, x_1: Tensor) -> Tensor:
    """
    Mean-squared CFM loss.

    Args:
        v_pred: (B, ...) predicted velocity from the model.
        x_0:    (B, ...) noise sample.
        x_1:    (B, ...) data sample.

    Returns:
        loss: scalar.
    """
    u_t = cfm_target_velocity(x_0, x_1)
    return ((v_pred - u_t) ** 2).mean()


# ---------------------------------------------------------------------------
# Batched helpers for variable-length point clouds
# (all points in the batch are concatenated along dim 0)
# ---------------------------------------------------------------------------

def sample_noise_like(x: Tensor) -> Tensor:
    """Sample x_0 ~ N(0, I) with the same shape and device as x."""
    return torch.randn_like(x)


def broadcast_t_to_points(t_mol: Tensor, batch: Tensor) -> Tensor:
    """
    Given per-molecule time scalars t_mol ∈ (B,), broadcast to per-point
    time values using the batch index tensor.

    Args:
        t_mol:  (B,) one time value per molecule.
        batch:  (sum_K,) molecule index per point.

    Returns:
        t_pts:  (sum_K,) time value per point.
    """
    return t_mol[batch]
