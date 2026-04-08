"""
Train Stage 1: Equivariant Flow Matching for point-cloud position generation.

Usage:
    python train_stage1.py \
        --pkl_path src/data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl \
        --cache_dir src/data/ed_energy_5w/cache \
        --output_dir checkpoints/stage1 \
        [--n_per_atom 8] \
        [--batch_size 16] \
        [--lr 1e-4] \
        [--epochs 100] \
        [--pretrained_visnet checkpoints/visnet_pretrained/best.pt] \
        [--device cuda]
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch_geometric.utils import scatter
from tqdm import tqdm

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from src.data import EDBenchPKLDataset, collate_fn
from src.model import VisNetEncoder, Stage1FlowNet
from src.utils.ot_cfm import (
    sample_t,
    interpolate,
    cfm_loss,
    sample_noise_like,
    broadcast_t_to_points,
)


def center_positions(
    atom_pos: torch.Tensor,
    point_pos: torch.Tensor,
    atom_batch: torch.Tensor,
    point_batch: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Subtract the per-molecule atom centroid from both atom and point positions.

    This centers each molecule at the origin, shrinking the coordinate range
    from ~[-15, 25] Bohr to ~[-12, 12] Bohr and preventing target velocities
    (x_1 - x_0) from being dominated by the absolute position offset.

    The centroid can be restored at inference time by adding it back.
    """
    centroid = scatter(atom_pos, atom_batch, dim=0, reduce="mean")  # (B, 3)
    atom_pos_c = atom_pos - centroid[atom_batch]
    point_pos_c = point_pos - centroid[point_batch]
    return atom_pos_c, point_pos_c


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pkl_path", required=True)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--output_dir", default="checkpoints/stage1")
    p.add_argument("--n_per_atom", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--hidden_channels", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--cutoff", type=float, default=8.0)
    p.add_argument("--cfg_drop_prob", type=float, default=0.15)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_samples", type=int, default=None, help="cap for debugging")
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--preprocess", action="store_true",
                   help="Pre-process all samples to cache_dir before training")
    p.add_argument("--pretrained_visnet", default=None,
                   help="Path to pretrained ViSNet checkpoint (from pretrain_visnet.py)")
    # W&B
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--wandb_project", default="ed-stage1")
    p.add_argument("--run_name", default=None)
    return p.parse_args()


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
        preprocess=args.preprocess,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=(args.device == "cuda"),
    )

    # --- Models ---
    visnet = VisNetEncoder(
        hidden_channels=args.hidden_channels,
        pretrained_path=args.pretrained_visnet,
    ).to(device)
    flow = Stage1FlowNet(
        atom_in_channels=args.hidden_channels,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        cutoff=args.cutoff,
        cfg_drop_prob=args.cfg_drop_prob,
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(visnet.parameters()) + list(flow.parameters()),
        lr=args.lr,
        weight_decay=1e-5,
    )
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
        visnet.train()
        flow.train()
        epoch_loss = 0.0

        pbar = tqdm(loader, desc=f"Stage1 {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            atom_pos = batch["atom_coords"].to(device)
            atom_types = batch["atom_types"].to(device)
            x_1 = batch["point_positions"].to(device)
            atom_batch = batch["atom_batch"].to(device)
            point_batch = batch["point_batch"].to(device)

            B = int(point_batch.max().item()) + 1

            # Center per-molecule: prevents large target velocities caused by
            # absolute Bohr coordinates dominating x_1 - x_0.
            atom_pos, x_1 = center_positions(atom_pos, x_1, atom_batch, point_batch)

            atom_feat, _ = visnet(atom_types, atom_pos, atom_batch)

            t_mol = sample_t(B, device=device)
            t_pts = broadcast_t_to_points(t_mol, point_batch)

            x_0 = sample_noise_like(x_1)
            x_t = interpolate(x_0, x_1, t_pts.unsqueeze(-1))

            v_pred = flow(
                x_t, t_pts,
                atom_pos, atom_feat,
                point_batch, atom_batch,
            )

            loss = cfm_loss(v_pred, x_0, x_1)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(visnet.parameters()) + list(flow.parameters()), max_norm=1.0
            )
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(loader)
        print(f"[Stage1] Epoch {epoch:4d}/{args.epochs}  loss={avg_loss:.6f}")
        if args.wandb:
            _wandb.log({"epoch": epoch, "train_loss": avg_loss}, step=epoch)

        if epoch % args.save_every == 0:
            torch.save(
                {"epoch": epoch, "visnet": visnet.state_dict(),
                 "flow": flow.state_dict(), "optimizer": optimizer.state_dict(),
                 "args": vars(args)},
                os.path.join(args.output_dir, f"epoch_{epoch:04d}.pt"),
            )

    torch.save(
        {"visnet": visnet.state_dict(), "flow": flow.state_dict(), "args": vars(args)},
        os.path.join(args.output_dir, "final.pt"),
    )
    if args.wandb:
        _wandb.finish()
    print(f"Stage 1 training done. Checkpoint: {args.output_dir}/final.pt")


if __name__ == "__main__":
    main()
