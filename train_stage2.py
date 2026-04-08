"""
Train Stage 2: Flow Matching for log-density value generation.

Usage:
    python train_stage2.py \
        --pkl_path src/data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl \
        --cache_dir src/data/ed_energy_5w/cache \
        --stage1_ckpt checkpoints/stage1/final.pt \
        --output_dir checkpoints/stage2 \
        [--n_per_atom 8] \
        [--batch_size 16] \
        [--lr 1e-4] \
        [--epochs 100] \
        [--device cuda]
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from src.data import EDBenchPKLDataset, collate_fn
from src.model import VisNetEncoder, Stage1FlowNet, Stage2FlowNet, rk4_ode_solve
from src.utils.ot_cfm import (
    sample_t,
    interpolate,
    cfm_loss,
    sample_noise_like,
    broadcast_t_to_points,
)
from src.utils import evaluate_batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pkl_path", required=True)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--stage1_ckpt", required=True)
    p.add_argument("--output_dir", default="checkpoints/stage2")
    p.add_argument("--n_per_atom", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--hidden_channels", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--cutoff", type=float, default=8.0)
    p.add_argument("--cfg_drop_prob", type=float, default=0.15)
    p.add_argument("--stage1_ode_steps", type=int, default=20,
                   help="ODE steps for Stage 1 inference during Stage 2 training")
    p.add_argument("--guidance_scale", type=float, default=1.5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--pretrained_visnet", default=None,
                   help="Path to pretrained ViSNet checkpoint (from pretrain_visnet.py)")
    # W&B
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--wandb_project", default="ed-stage2")
    p.add_argument("--run_name", default=None)
    return p.parse_args()


@torch.no_grad()
def generate_point_positions(
    stage1: Stage1FlowNet,
    atom_pos: torch.Tensor,
    atom_feat: torch.Tensor,
    point_batch: torch.Tensor,
    atom_batch: torch.Tensor,
    n_ode_steps: int,
    guidance_scale: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Run Stage 1 flow matching (RK4) to generate point positions.
    Returns (sum_K, 3) positions.
    """
    sum_K = point_batch.shape[0]
    x_init = torch.randn(sum_K, 3, device=device)
    t_dummy = torch.zeros(sum_K, device=device)  # placeholder; vel_fn fills it

    def vel_fn(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if guidance_scale == 1.0:
            return stage1(x_t, t, atom_pos, atom_feat, point_batch, atom_batch,
                          drop_condition=False)
        v_cond = stage1(x_t, t, atom_pos, atom_feat, point_batch, atom_batch,
                        drop_condition=False)
        v_uncond = stage1(x_t, t, atom_pos, atom_feat, point_batch, atom_batch,
                          drop_condition=True)
        return v_uncond + guidance_scale * (v_cond - v_uncond)

    return rk4_ode_solve(x_init, vel_fn, n_steps=n_ode_steps, device=device)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Data ---
    dataset = EDBenchPKLDataset(
        pkl_path=args.pkl_path,
        cache_dir=args.cache_dir,
        n_per_atom=args.n_per_atom,
        max_samples=args.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=(args.device == "cuda"),
    )

    # --- Load Stage 1 (frozen) ---
    ckpt1 = torch.load(args.stage1_ckpt, map_location=device, weights_only=True)
    s1_args = ckpt1["args"]

    visnet = VisNetEncoder(hidden_channels=s1_args["hidden_channels"]).to(device)
    visnet.load_state_dict(ckpt1["visnet"])
    visnet.eval()
    for p in visnet.parameters():
        p.requires_grad_(False)

    stage1 = Stage1FlowNet(
        atom_in_channels=s1_args["hidden_channels"],
        hidden_channels=s1_args["hidden_channels"],
        num_layers=s1_args["num_layers"],
        cutoff=s1_args["cutoff"],
    ).to(device)
    stage1.load_state_dict(ckpt1["flow"])
    stage1.eval()
    for p in stage1.parameters():
        p.requires_grad_(False)

    # --- Stage 2 model ---
    stage2 = Stage2FlowNet(
        atom_in_channels=s1_args["hidden_channels"],
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        cutoff=args.cutoff,
        cfg_drop_prob=args.cfg_drop_prob,
    ).to(device)

    optimizer = torch.optim.AdamW(stage2.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(loader)
    )

    # --- W&B init ---
    if args.wandb:
        if not _WANDB_AVAILABLE:
            print("WARNING: wandb not installed, skipping W&B logging")
            args.wandb = False
        else:
            _wandb.init(
                project=args.wandb_project,
                name=args.run_name,
                config=vars(args),
            )

    # --- Training loop ---
    for epoch in range(1, args.epochs + 1):
        stage2.train()
        epoch_loss = 0.0

        pbar = tqdm(loader, desc=f"Stage2 {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            atom_pos = batch["atom_coords"].to(device)
            atom_types = batch["atom_types"].to(device)
            z_1 = batch["point_log_densities"].to(device)   # (sum_K,) target
            atom_batch = batch["atom_batch"].to(device)
            point_batch = batch["point_batch"].to(device)

            B = int(point_batch.max().item()) + 1

            # VisNet encoding (no grad, visnet is frozen)
            atom_feat, _ = visnet(atom_types, atom_pos, atom_batch)

            # Generate point positions from Stage 1 (no grad)
            r_points = generate_point_positions(
                stage1, atom_pos, atom_feat, point_batch, atom_batch,
                n_ode_steps=args.stage1_ode_steps,
                guidance_scale=args.guidance_scale,
                device=device,
            )

            # --- CFM on log-density values ---
            t_mol = sample_t(B, device=device)
            t_pts = broadcast_t_to_points(t_mol, point_batch)

            z_0 = sample_noise_like(z_1)
            z_t = interpolate(z_0, z_1, t_pts)

            v_pred = stage2(
                z_t, t_pts,
                r_points, atom_pos, atom_feat,
                point_batch, atom_batch,
            )

            loss = cfm_loss(v_pred, z_0, z_1)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(stage2.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(loader)
        print(f"[Stage2] Epoch {epoch:4d}/{args.epochs}  loss={avg_loss:.6f}")
        if args.wandb:
            _wandb.log({"epoch": epoch, "train_loss": avg_loss}, step=epoch)

        if epoch % args.save_every == 0:
            torch.save(
                {"epoch": epoch, "stage2": stage2.state_dict(),
                 "optimizer": optimizer.state_dict(), "args": vars(args)},
                os.path.join(args.output_dir, f"epoch_{epoch:04d}.pt"),
            )

    torch.save(
        {"stage2": stage2.state_dict(), "args": vars(args)},
        os.path.join(args.output_dir, "final.pt"),
    )
    if args.wandb:
        _wandb.finish()
    print(f"Stage 2 training done. Checkpoint saved to {args.output_dir}/final.pt")


if __name__ == "__main__":
    main()
