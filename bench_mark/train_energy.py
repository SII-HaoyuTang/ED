"""
Training script for EDBench ED5-EC energy prediction (PointMetaBase-S-X3D).

Usage:
    # Smoke test
    python bench_mark/train_energy.py \\
        --pkl_path src/data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl \\
        --csv_path src/data/ed_energy_5w/raw/ed_energy_5w.csv \\
        --cache_dir src/data/ed_energy_5w/cache_fps \\
        --max_samples 128 --epochs 2 --npoint 512 --device cpu

    # Full training (matches paper config)
    python bench_mark/train_energy.py \\
        --pkl_path src/data/ed_energy_5w/processed/mol_EDthresh0.05_data.pkl \\
        --csv_path src/data/ed_energy_5w/raw/ed_energy_5w.csv \\
        --cache_dir src/data/ed_energy_5w/cache_fps \\
        --npoint 2048 --batch_size 32 --lr 1e-3 --epochs 100 --device cuda
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

from bench_mark.data.energy_dataset import EDBenchEnergyDataset, energy_collate_fn
from bench_mark.models.backbone.pointmetabase_x3d import PointMetaBaseX3D

# Energy component names (for readable logging)
_ENERGY_NAMES = [
    "E1_Final", "E2_NucRepul", "E3_OneElec",
    "E4_TwoElec", "E5_XC", "E6_Total",
]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EDBench ED5-EC energy prediction")
    # Data
    p.add_argument("--pkl_path",  required=True, help="Path to .pkl density file")
    p.add_argument("--csv_path",  required=True, help="Path to ed_energy_5w.csv")
    p.add_argument("--cache_dir", required=True, help="Cache dir for FPS .pt files")
    p.add_argument("--npoint",    type=int,   default=2048,
                   help="Points per molecule after FPS (default 2048)")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap dataset size (debug)")
    # Model
    p.add_argument("--width",     type=int,   default=32,
                   help="Base channel width (default 32)")
    p.add_argument("--radius",    type=float, default=0.15,
                   help="Base ball-query radius in Bohr (default 0.15)")
    p.add_argument("--radius_mult", type=float, default=1.5,
                   help="Radius multiplier per stage (default 1.5)")
    p.add_argument("--K",         type=int,   default=32,
                   help="Neighbours per query point (default 32)")
    # Training
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--dropout",    type=float, default=0.5)
    p.add_argument("--max_norm",   type=float, default=1.0,
                   help="Gradient clip max norm")
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int,  default=4)
    p.add_argument("--output_dir", default="checkpoints/energy")
    p.add_argument("--save_every", type=int,   default=10)
    # W&B
    p.add_argument("--wandb",     action="store_true", help="Enable W&B logging")
    p.add_argument("--wandb_project", default="ed-energy")
    p.add_argument("--run_name",  default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """
    Returns per-target MAE, RMSE, Pearson r, Spearman ρ.
    Matches EDBench's metric_reg_multitask evaluation.
    """
    model.eval()
    all_preds = []
    all_targets = []

    for batch in loader:
        pc = batch["point_cloud"].to(device)    # (B, N, 4)
        y  = batch["energies"].to(device)       # (B, 6)
        pred = model(pc)                        # (B, 6)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(y.cpu().numpy())

    preds   = np.concatenate(all_preds,   axis=0)   # (N_total, 6)
    targets = np.concatenate(all_targets, axis=0)   # (N_total, 6)

    results: dict[str, float] = {}
    for i, name in enumerate(_ENERGY_NAMES):
        p = preds[:, i]
        t = targets[:, i]
        mae  = float(np.abs(p - t).mean())
        rmse = float(np.sqrt(((p - t) ** 2).mean()))
        pr   = float(pearsonr(p, t)[0])
        sr   = float(spearmanr(p, t)[0])
        results[f"{name}_MAE"]     = mae
        results[f"{name}_RMSE"]    = rmse
        results[f"{name}_Pearson"] = pr
        results[f"{name}_Spearman"] = sr

    results["mean_MAE"] = float(np.mean([results[f"{n}_MAE"] for n in _ENERGY_NAMES]))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Datasets ---
    train_set = EDBenchEnergyDataset(
        pkl_path=args.pkl_path, csv_path=args.csv_path,
        cache_dir=args.cache_dir, split="train",
        npoint=args.npoint, max_samples=args.max_samples,
    )
    val_set = EDBenchEnergyDataset(
        pkl_path=args.pkl_path, csv_path=args.csv_path,
        cache_dir=args.cache_dir, split="valid",
        npoint=args.npoint, max_samples=args.max_samples,
    )
    test_set = EDBenchEnergyDataset(
        pkl_path=args.pkl_path, csv_path=args.csv_path,
        cache_dir=args.cache_dir, split="test",
        npoint=args.npoint, max_samples=args.max_samples,
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        collate_fn=energy_collate_fn, num_workers=args.num_workers,
        pin_memory=(args.device == "cuda"),
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size * 2, shuffle=False,
        collate_fn=energy_collate_fn, num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size * 2, shuffle=False,
        collate_fn=energy_collate_fn, num_workers=args.num_workers,
    )

    # --- Model ---
    model = PointMetaBaseX3D(
        in_channels=4,
        width=args.width,
        num_targets=6,
        npoint_start=args.npoint,
        radius=args.radius,
        radius_mult=args.radius_mult,
        K=args.K,
        mlp_layers=[512, 256],
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"PointMetaBase-S-X3D  params: {n_params:,}")

    # --- Optimizer & Scheduler (matches paper: AdamW + CosineAnnealing) ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = nn.MSELoss()

    # --- W&B ---
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

    best_val_mae = float("inf")

    # --- Training loop ---
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            pc = batch["point_cloud"].to(device)    # (B, N, 4)
            y  = batch["energies"].to(device)       # (B, 6)

            pred = model(pc)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, device)
        val_mae = val_metrics["mean_MAE"]
        lr_now = optimizer.param_groups[0]["lr"]

        # Print per-epoch summary
        print(
            f"Epoch {epoch:4d}/{args.epochs}  "
            f"train_loss={avg_loss:.4f}  val_mean_MAE={val_mae:.4f}  "
            f"lr={lr_now:.2e}"
        )
        for name in _ENERGY_NAMES:
            print(
                f"  {name:20s}  MAE={val_metrics[name+'_MAE']:.4f}  "
                f"RMSE={val_metrics[name+'_RMSE']:.4f}  "
                f"r={val_metrics[name+'_Pearson']:.4f}"
            )

        if args.wandb:
            log_dict = {"epoch": epoch, "train_loss": avg_loss, "lr": lr_now}
            log_dict.update({f"val/{k}": v for k, v in val_metrics.items()})
            _wandb.log(log_dict, step=epoch)

        # Save best
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(
                {"epoch": epoch, "model": model.state_dict(),
                 "val_metrics": val_metrics, "args": vars(args)},
                os.path.join(args.output_dir, "best.pt"),
            )
            if args.wandb:
                _wandb.summary["best_val_mean_MAE"] = best_val_mae

        if epoch % args.save_every == 0:
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "args": vars(args)},
                os.path.join(args.output_dir, f"epoch_{epoch:04d}.pt"),
            )

    # --- Final evaluation on test set ---
    print("\n=== Test Set Evaluation ===")
    test_metrics = evaluate(model, test_loader, device)
    for name in _ENERGY_NAMES:
        print(
            f"  {name:20s}  MAE={test_metrics[name+'_MAE']:.4f}  "
            f"RMSE={test_metrics[name+'_RMSE']:.4f}  "
            f"r={test_metrics[name+'_Pearson']:.4f}"
        )
    print(f"  Mean MAE: {test_metrics['mean_MAE']:.4f}")

    torch.save(
        {"model": model.state_dict(), "test_metrics": test_metrics, "args": vars(args)},
        os.path.join(args.output_dir, "final.pt"),
    )
    if args.wandb:
        _wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
        _wandb.finish()

    print(f"\nDone. Best val mean MAE: {best_val_mae:.4f}")
    print(f"Checkpoint: {args.output_dir}/best.pt")


if __name__ == "__main__":
    main()
