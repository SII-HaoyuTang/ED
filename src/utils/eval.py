"""
Evaluation metrics for electron density predictions.

All functions operate on numpy arrays or tensors with explicit shape comments.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Point-cloud → continuous density reconstruction
# ---------------------------------------------------------------------------

def reconstruct_density_kde(
    point_pos: np.ndarray,   # (K, 3)
    point_dens: np.ndarray,  # (K,)   predicted density values (NOT log)
    grid_coords: np.ndarray, # (G, 3)  query grid
    bandwidth: float = 0.5,  # Gaussian kernel bandwidth in Bohr
) -> np.ndarray:
    """
    Reconstruct a continuous density field on a grid using Gaussian KDE.

    ρ̂(r) = Σ_j ρ_j · K_h(r - r_j)  /  Σ_j K_h(r - r_j)

    This produces a smooth density field normalised so that the density
    at each grid point is a weighted average of the point densities.

    Args:
        point_pos:   (K, 3) predicted point positions.
        point_dens:  (K,)   predicted density values.
        grid_coords: (G, 3) evaluation grid.
        bandwidth:   Gaussian bandwidth h (Bohr).

    Returns:
        density: (G,) reconstructed density at grid points.
    """
    # (G, K) squared distances
    diff = grid_coords[:, None, :] - point_pos[None, :, :]  # (G, K, 3)
    sq_dist = (diff ** 2).sum(-1)                            # (G, K)
    kernel = np.exp(-sq_dist / (2 * bandwidth ** 2))         # (G, K)

    numerator = (kernel * point_dens[None, :]).sum(-1)       # (G,)
    denominator = kernel.sum(-1) + 1e-12                     # (G,)
    return numerator / denominator                           # (G,)


# ---------------------------------------------------------------------------
# Scalar density metrics
# ---------------------------------------------------------------------------

def mean_absolute_error(
    pred: np.ndarray, target: np.ndarray
) -> float:
    """MAE of density values, pointwise."""
    return float(np.abs(pred - target).mean())


def root_mean_square_error(
    pred: np.ndarray, target: np.ndarray
) -> float:
    """RMSE of density values, pointwise."""
    return float(np.sqrt(((pred - target) ** 2).mean()))


def electron_count_error(
    density: np.ndarray,   # (G,)
    voxel_volume: float,   # volume of one voxel in Bohr^3
    n_electrons: int,
) -> float:
    """
    Absolute error in the total electron count (integral of density).

        |∫ρ(r)dr - N_e|  ≈  |Σ_g ρ_g · V_voxel - N_e|
    """
    integrated = float(density.sum()) * voxel_volume
    return abs(integrated - n_electrons)


# ---------------------------------------------------------------------------
# Evaluation loop helper
# ---------------------------------------------------------------------------

def evaluate_batch(
    pred_log_dens: Tensor,    # (sum_K,)  predicted log-densities
    true_log_dens: Tensor,    # (sum_K,)  ground-truth log-densities
) -> dict[str, float]:
    """
    Fast in-batch metrics on log-density values (no grid reconstruction).
    Suitable for monitoring during training.

    Returns dict with keys: mae_log, rmse_log, mae_linear, rmse_linear.
    """
    pred = pred_log_dens.detach().float()
    true = true_log_dens.detach().float()

    mae_log = (pred - true).abs().mean().item()
    rmse_log = ((pred - true) ** 2).mean().sqrt().item()

    pred_lin = pred.exp()
    true_lin = true.exp()
    mae_lin = (pred_lin - true_lin).abs().mean().item()
    rmse_lin = ((pred_lin - true_lin) ** 2).mean().sqrt().item()

    return {
        "mae_log": mae_log,
        "rmse_log": rmse_log,
        "mae_linear": mae_lin,
        "rmse_linear": rmse_lin,
    }
