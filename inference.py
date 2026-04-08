"""
Inference: generate electron density point clouds for new molecules.

Given atomic coordinates and types, runs Stage 1 + Stage 2 to produce a
representative set of (position, density) pairs describing the electron cloud.

Usage:
    python inference.py \
        --stage1_ckpt checkpoints/stage1/final.pt \
        --stage2_ckpt checkpoints/stage2/final.pt \
        --cube_file  /path/to/molecule.cube \
        [--n_samples 5] \
        [--guidance_scale 1.5] \
        [--ode_steps 50] \
        [--output output.pt] \
        [--device cuda]

Output format (.pt file):
    {
        "atom_coords":       (N, 3)  float32,
        "atom_types":        (N,)    int64,
        "point_positions":   (n_samples, K, 3)  float32,
        "point_densities":   (n_samples, K)     float32,  # exp(log_dens)
    }
"""
from __future__ import annotations

import argparse

import torch

from src.data import parse_cube
from src.model import (
    VisNetEncoder,
    Stage1FlowNet,
    Stage2FlowNet,
    rk4_ode_solve,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_ckpt", required=True)
    p.add_argument("--stage2_ckpt", required=True)
    p.add_argument("--cube_file", required=True, help=".cube file for atom geometry")
    p.add_argument("--n_samples", type=int, default=5, help="number of samples to draw")
    p.add_argument("--guidance_scale", type=float, default=1.5)
    p.add_argument("--ode_steps", type=int, default=50)
    p.add_argument("--output", default="output.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def run_inference(
    visnet: VisNetEncoder,
    stage1: Stage1FlowNet,
    stage2: Stage2FlowNet,
    atom_types: torch.Tensor,     # (N,)
    atom_pos: torch.Tensor,       # (N, 3)
    n_per_atom: int,
    n_samples: int,
    guidance_scale: float,
    ode_steps: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate n_samples point clouds for a single molecule.

    Returns:
        positions:  (n_samples, K, 3)
        densities:  (n_samples, K)      linear density values
    """
    N = atom_pos.shape[0]
    K = n_per_atom * N

    # Batch index (single molecule → all zeros)
    atom_batch = torch.zeros(N, dtype=torch.long, device=device)
    point_batch = torch.zeros(K, dtype=torch.long, device=device)

    # VisNet encoding (shared across all samples)
    atom_feat, _ = visnet(atom_types, atom_pos, atom_batch)

    all_positions = []
    all_densities = []

    for _ in range(n_samples):
        # --- Stage 1: generate positions ---
        def s1_vel(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            v_c = stage1(x_t, t, atom_pos, atom_feat, point_batch, atom_batch,
                         drop_condition=False)
            v_u = stage1(x_t, t, atom_pos, atom_feat, point_batch, atom_batch,
                         drop_condition=True)
            return v_u + guidance_scale * (v_c - v_u)

        x_init = torch.randn(K, 3, device=device)
        r_points = rk4_ode_solve(x_init, s1_vel, n_steps=ode_steps, device=device)

        # --- Stage 2: generate log-density values ---
        def s2_vel(z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            v_c = stage2(z_t, t, r_points, atom_pos, atom_feat,
                         point_batch, atom_batch, drop_condition=False)
            v_u = stage2(z_t, t, r_points, atom_pos, atom_feat,
                         point_batch, atom_batch, drop_condition=True)
            return v_u + guidance_scale * (v_c - v_u)

        z_init = torch.randn(K, device=device)
        z_final = rk4_ode_solve(z_init, s2_vel, n_steps=ode_steps, device=device)
        rho = z_final.exp()   # back to linear density

        all_positions.append(r_points.cpu())
        all_densities.append(rho.cpu())

    positions = torch.stack(all_positions)    # (n_samples, K, 3)
    densities = torch.stack(all_densities)    # (n_samples, K)
    return positions, densities


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    # --- Load checkpoints ---
    ckpt1 = torch.load(args.stage1_ckpt, map_location=device, weights_only=True)
    ckpt2 = torch.load(args.stage2_ckpt, map_location=device, weights_only=True)
    s1_args = ckpt1["args"]
    s2_args = ckpt2["args"]

    visnet = VisNetEncoder(hidden_channels=s1_args["hidden_channels"]).to(device)
    visnet.load_state_dict(ckpt1["visnet"])
    visnet.eval()

    stage1 = Stage1FlowNet(
        atom_in_channels=s1_args["hidden_channels"],
        hidden_channels=s1_args["hidden_channels"],
        num_layers=s1_args["num_layers"],
        cutoff=s1_args["cutoff"],
    ).to(device)
    stage1.load_state_dict(ckpt1["flow"])
    stage1.eval()

    stage2 = Stage2FlowNet(
        atom_in_channels=s1_args["hidden_channels"],
        hidden_channels=s2_args["hidden_channels"],
        num_layers=s2_args["num_layers"],
        num_heads=s2_args["num_heads"],
        cutoff=s2_args["cutoff"],
    ).to(device)
    stage2.load_state_dict(ckpt2["stage2"])
    stage2.eval()

    # --- Parse molecule geometry ---
    cube = parse_cube(args.cube_file)
    atom_types = torch.from_numpy(cube.atom_types).long().to(device)
    atom_pos = torch.from_numpy(cube.atom_coords).float().to(device)

    # --- Run inference ---
    positions, densities = run_inference(
        visnet, stage1, stage2,
        atom_types, atom_pos,
        n_per_atom=s1_args["n_per_atom"],
        n_samples=args.n_samples,
        guidance_scale=args.guidance_scale,
        ode_steps=args.ode_steps,
        device=device,
    )

    # --- Save ---
    result = {
        "atom_coords": atom_pos.cpu(),
        "atom_types": atom_types.cpu(),
        "point_positions": positions,
        "point_densities": densities,
    }
    torch.save(result, args.output)
    print(f"Saved {args.n_samples} samples to {args.output}")
    print(f"  point_positions : {positions.shape}")
    print(f"  point_densities : {densities.shape}")


if __name__ == "__main__":
    main()
