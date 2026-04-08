"""
Pretrain ViSNet encoder on QM9 U0 (internal energy at 0K) energy prediction.

The pretrained representation_model weights are directly compatible with
VisNetEncoder and can be loaded into Stage 1/2 training via --pretrained_visnet.

Usage:
    # Smoke test (fast, small subset)
    python pretrain_visnet.py --data_root data/qm9 --max_samples 1000 --epochs 2

    # Full training
    python pretrain_visnet.py \
        --data_root data/qm9 \
        --output_dir checkpoints/visnet_pretrained \
        --hidden_channels 256 \
        --num_layers 6 \
        --epochs 300 \
        --batch_size 32 \
        --lr 1e-4

    # Then use pretrained weights in Stage 1:
    python train_stage1.py ... --pretrained_visnet checkpoints/visnet_pretrained/best.pt
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from tqdm import tqdm

try:
    from torch_geometric.nn.models import ViSNet as _PYGVisNet
except ImportError as e:
    raise ImportError("torch_geometric is required. pip install torch_geometric") from e

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# QM9 target index for U0 (internal energy at 0K, in eV after unit conversion)
_U0_IDX = 7

# Default mirrors tried in order when the primary AWS S3 URL returns 403.
# Override all of these at once with --qm9_url if you have a local HTTP server.
_QM9_MIRRORS = [
    # Primary (PyG default) — may be blocked outside US
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip",
    # PyG-hosted pre-processed bundle (smaller, usually accessible)
    "https://data.pyg.org/datasets/qm9_v3.zip",
]


class _QM9(QM9):
    """QM9 with overrideable download URL and local-file support.

    Args:
        url_override:   Replace the default AWS S3 URL (useful for 403 errors).
        local_raw_dir:  Directory that already contains the raw QM9 files.
                        Files are copied into ``{root}/raw/`` so PyG finds them
                        without network access.  Accepted filenames:
                          • With rdkit:    gdb9.sdf  gdb9.sdf.csv  uncharacterized.txt
                          • Without rdkit: qm9_v3.pt
    """

    def __init__(
        self,
        root: str,
        url_override: str | None = None,
        local_raw_dir: str | None = None,
        **kwargs,
    ) -> None:
        self._local_raw_dir = local_raw_dir
        if url_override is not None:
            self.raw_url = url_override
        super().__init__(root, **kwargs)

    def download(self) -> None:
        if self._local_raw_dir is not None:
            import shutil
            os.makedirs(self.raw_dir, exist_ok=True)
            for fname in self.raw_file_names:
                src = os.path.join(self._local_raw_dir, fname)
                dst = os.path.join(self.raw_dir, fname)
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    print(f"  [_QM9] Copied {fname}  {self._local_raw_dir} → {self.raw_dir}")
            missing = [
                f for f in self.raw_file_names
                if not os.path.exists(os.path.join(self.raw_dir, f))
            ]
            if not missing:
                return   # all expected files are present — skip network download
            print(f"  [_QM9] Still missing after copy: {missing}; falling back to URL download.")
        super().download()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data/qm9",
                   help="Directory for QM9 download/cache")
    p.add_argument("--output_dir", default="checkpoints/visnet_pretrained")
    p.add_argument("--hidden_channels", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--num_rbf", type=int, default=32)
    p.add_argument("--cutoff", type=float, default=5.0)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--max_z", type=int, default=100)
    p.add_argument("--vertex", action="store_true", default=True,
                   help="Use ViSNet-l (vertex variant) for richer features")
    p.add_argument("--train_size", type=int, default=110000)
    p.add_argument("--val_size", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr_patience", type=int, default=15,
                   help="ReduceLROnPlateau patience (epochs)")
    p.add_argument("--lr_factor", type=float, default=0.8)
    p.add_argument("--lr_min", type=float, default=1e-7)
    p.add_argument("--max_norm", type=float, default=1.0,
                   help="Gradient clipping max norm")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap total dataset size for debugging")
    p.add_argument("--qm9_url", default=None,
                   help=(
                       "Override the QM9 download URL (useful when the default "
                       "AWS S3 address returns 403 on your server). "
                       "The URL must point to a .zip containing gdb9.sdf and "
                       "gdb9.sdf.csv, identical in layout to the original. "
                       "Example: --qm9_url https://data.pyg.org/datasets/qm9_v3.zip"
                   ))
    p.add_argument("--qm9_raw_dir", default=None,
                   help=(
                       "Path to a local directory that already contains the QM9 raw "
                       "files. Files are copied to {data_root}/raw/ so PyG skips the "
                       "network download entirely. "
                       "With rdkit: supply gdb9.sdf, gdb9.sdf.csv, uncharacterized.txt. "
                       "Without rdkit: supply qm9_v3.pt."
                   ))
    p.add_argument("--save_every", type=int, default=10)
    # W&B
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--wandb_project", default="ed-pretrain-visnet")
    p.add_argument("--run_name", default=None)
    return p.parse_args()


def make_splits(dataset, train_size: int, val_size: int, max_samples: int | None):
    """Fixed deterministic split using first N indices."""
    n = len(dataset)
    if max_samples is not None:
        n = min(n, max_samples)
    train_size = min(train_size, n)
    val_size = min(val_size, n - train_size)

    idx = torch.arange(n)
    train_idx = idx[:train_size]
    val_idx = idx[train_size:train_size + val_size]
    test_idx = idx[train_size + val_size:]

    return (
        Subset(dataset, train_idx.tolist()),
        Subset(dataset, val_idx.tolist()),
        Subset(dataset, test_idx.tolist()),
    )


def compute_target_stats(loader, device: torch.device) -> tuple[float, float]:
    """Compute mean and std of U0 over the training set."""
    vals = []
    for data in loader:
        vals.append(data.y[:, _U0_IDX])
    vals = torch.cat(vals)
    return vals.mean().item(), vals.std().item()


@torch.no_grad()
def evaluate(model, loader, mean: float, std: float, device: torch.device) -> float:
    """Return MAE in original units (eV). Returns inf if loader is empty."""
    model.eval()
    total_mae = 0.0
    total_n = 0
    for data in loader:
        data = data.to(device)
        y_target = (data.y[:, _U0_IDX] - mean) / std
        y_pred, _ = model(data.z, data.pos, data.batch)
        y_pred = y_pred.squeeze(-1)
        # Un-standardize for MAE
        mae = (y_pred * std + mean - data.y[:, _U0_IDX]).abs().mean().item()
        n = data.y.shape[0]
        total_mae += mae * n
        total_n += n
    if total_n == 0:
        return float("inf")
    return total_mae / total_n


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Dataset ---
    print(f"Loading QM9 from {args.data_root} (auto-downloads if needed)...")
    if args.qm9_raw_dir:
        # User already has the raw files; copy them and skip any network access.
        print(f"  Using local raw files from: {args.qm9_raw_dir}")
        dataset = _QM9(root=args.data_root, local_raw_dir=args.qm9_raw_dir)
    else:
        urls_to_try = [args.qm9_url] if args.qm9_url else _QM9_MIRRORS
        dataset = None
        for url in urls_to_try:
            try:
                dataset = _QM9(root=args.data_root, url_override=url)
                break
            except Exception as exc:
                print(f"  Download failed with URL {url!r}: {exc}")
        if dataset is None:
            raise RuntimeError(
                "All QM9 download URLs failed.\n"
                "If you already have the raw files, use:\n"
                "  --qm9_raw_dir /path/to/dir/containing/gdb9.sdf\n"
                "Otherwise download manually and place gdb9.sdf + gdb9.sdf.csv + "
                f"uncharacterized.txt into {args.data_root}/raw/"
            )
    train_set, val_set, test_set = make_splits(
        dataset, args.train_size, args.val_size, args.max_samples
    )
    print(f"Split: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(args.device == "cuda"),
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size * 4, shuffle=False,
        num_workers=args.num_workers,
    )

    # --- Compute target statistics for z-score normalization ---
    print("Computing U0 statistics over training set...")
    mean, std = compute_target_stats(train_loader, device)
    print(f"U0: mean={mean:.4f} eV, std={std:.4f} eV")

    # --- Model ---
    # Hyperparameters are kept identical to VisNetEncoder defaults so that
    # weights can be loaded directly with load_state_dict(strict=True).
    model = _PYGVisNet(
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        num_rbf=args.num_rbf,
        trainable_rbf=False,
        max_z=args.max_z,
        cutoff=args.cutoff,
        num_heads=args.num_heads,
        vertex=args.vertex,
        derivative=False,   # no force prediction → faster
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"ViSNet parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=args.lr_patience,
        factor=args.lr_factor, min_lr=args.lr_min,
    )

    best_val_mae = float("inf")

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
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Pretrain {epoch}/{args.epochs}", leave=False)
        for data in pbar:
            data = data.to(device)
            y_target = (data.y[:, _U0_IDX] - mean) / std   # standardised

            y_pred, _ = model(data.z, data.pos, data.batch)
            y_pred = y_pred.squeeze(-1)

            loss = F.mse_loss(y_pred, y_target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        val_mae = evaluate(model, val_loader, mean, std, device)
        scheduler.step(val_mae)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:4d}/{args.epochs}  "
            f"train_loss={avg_loss:.6f}  val_MAE={val_mae:.6f} eV  "
            f"lr={lr_now:.2e}"
        )
        if args.wandb:
            _wandb.log({
                "epoch": epoch,
                "train_loss": avg_loss,
                "val_mae_eV": val_mae,
                "lr": lr_now,
            }, step=epoch)

        # Save best checkpoint
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(
                {
                    "epoch": epoch,
                    "representation_model": model.representation_model.state_dict(),
                    "full_model": model.state_dict(),
                    "mean": mean,
                    "std": std,
                    "val_mae": val_mae,
                    "args": vars(args),
                },
                os.path.join(args.output_dir, "best.pt"),
            )
            if args.wandb:
                _wandb.summary["best_val_mae_eV"] = best_val_mae

        if epoch % args.save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "representation_model": model.representation_model.state_dict(),
                    "full_model": model.state_dict(),
                    "mean": mean,
                    "std": std,
                    "args": vars(args),
                },
                os.path.join(args.output_dir, f"epoch_{epoch:04d}.pt"),
            )

    # --- Final checkpoint ---
    torch.save(
        {
            "representation_model": model.representation_model.state_dict(),
            "full_model": model.state_dict(),
            "mean": mean,
            "std": std,
            "args": vars(args),
        },
        os.path.join(args.output_dir, "final.pt"),
    )
    if args.wandb:
        _wandb.finish()
    print(f"\nPretraining done. Best val MAE: {best_val_mae:.6f} eV")
    print(f"Best checkpoint: {args.output_dir}/best.pt")
    print(f"\nTo use in Stage 1/2 training, add:")
    print(f"  --pretrained_visnet {args.output_dir}/best.pt")


if __name__ == "__main__":
    main()
