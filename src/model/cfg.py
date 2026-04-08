"""
Classifier-Free Guidance (CFG) inference utilities.

For both Stage 1 and Stage 2, guided velocity is computed as:

    v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)

This module provides:
  - cfg_velocity()    : single-call CFG combining two forward passes
  - euler_ode_solve() : simple fixed-step Euler ODE integrator
  - rk4_ode_solve()   : 4th-order Runge-Kutta integrator (higher quality)
"""
from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor


VelocityFn = Callable[[Tensor, Tensor], Tensor]
"""Signature: (x_t: Tensor, t: Tensor) -> velocity: Tensor"""


def cfg_velocity(
    x_t: Tensor,
    t: Tensor,
    cond_vel_fn: VelocityFn,
    uncond_vel_fn: VelocityFn,
    guidance_scale: float,
) -> Tensor:
    """
    Compute classifier-free guided velocity.

    Args:
        x_t:            Current state (sum_K, 3) or (sum_K,).
        t:              Current time (sum_K,).
        cond_vel_fn:    v_θ(x_t, t | condition).
        uncond_vel_fn:  v_θ(x_t, t | ∅).
        guidance_scale: w ≥ 0.  w=0 → unconditional, w=1 → pure conditional.

    Returns:
        Guided velocity with same shape as x_t.
    """
    v_cond = cond_vel_fn(x_t, t)
    if guidance_scale == 1.0:
        return v_cond
    v_uncond = uncond_vel_fn(x_t, t)
    return v_uncond + guidance_scale * (v_cond - v_uncond)


def euler_ode_solve(
    x_init: Tensor,
    vel_fn: VelocityFn,
    n_steps: int = 50,
    device: torch.device | None = None,
) -> Tensor:
    """
    Euler ODE solver: integrates from t=0 to t=1.

        x_{t+dt} = x_t + dt * v_θ(x_t, t)

    Args:
        x_init:  (sum_K, d) initial state (noise).
        vel_fn:  velocity function (x_t, t_pts) → velocity.
        n_steps: number of integration steps.

    Returns:
        x_1: (sum_K, d) generated samples.
    """
    dt = 1.0 / n_steps
    x = x_init.clone()
    K = x.shape[0]
    dev = device or x.device

    for i in range(n_steps):
        t = torch.full((K,), i * dt, device=dev, dtype=x.dtype)
        v = vel_fn(x, t)
        x = x + dt * v

    return x


def rk4_ode_solve(
    x_init: Tensor,
    vel_fn: VelocityFn,
    n_steps: int = 20,
    device: torch.device | None = None,
) -> Tensor:
    """
    4th-order Runge-Kutta ODE solver: integrates from t=0 to t=1.

    Provides better accuracy than Euler at the same step count.

    Args:
        x_init:  (sum_K, d) initial state (noise).
        vel_fn:  velocity function (x_t, t_pts) → velocity.
        n_steps: number of RK4 steps.

    Returns:
        x_1: (sum_K, d) generated samples.
    """
    dt = 1.0 / n_steps
    x = x_init.clone()
    K = x.shape[0]
    dev = device or x.device

    for i in range(n_steps):
        t0 = i * dt
        t_half = t0 + 0.5 * dt
        t_next = t0 + dt

        t0_vec = torch.full((K,), t0, device=dev, dtype=x.dtype)
        th_vec = torch.full((K,), t_half, device=dev, dtype=x.dtype)
        tn_vec = torch.full((K,), t_next, device=dev, dtype=x.dtype)

        k1 = vel_fn(x, t0_vec)
        k2 = vel_fn(x + 0.5 * dt * k1, th_vec)
        k3 = vel_fn(x + 0.5 * dt * k2, th_vec)
        k4 = vel_fn(x + dt * k3, tn_vec)

        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x
