"""
Pretrain ViSNet encoder on QM9 U0 (internal energy at 0K) energy prediction.

Uses the original ViSNet source code from src/model/visnet/:
  - ViSNetBlock       as the representation model
  - EquivariantScalar as the output head (uses both scalar + equivariant features)
  - Atomref prior     (per-element reference energies fitted from training data)

Architecture matches the ViSNet-QM9.yml paper configuration:
  hidden_channels=512, num_layers=9, num_rbf=64, lmax=2, vertex_type=Edge

The pretrained representation_model weights are directly compatible with
VisNetEncoder and can be loaded into Stage 1/2 training via --pretrained_visnet.

Usage:
    # Smoke test (fast, small subset)
    python pretrain_visnet.py --data_root data/qm9 --max_samples 1000 --epochs 2 \\
        --hidden_channels 64 --num_layers 2

    # Full training (paper config, 4090 GPU)
    python pretrain_visnet.py \\
        --data_root data/qm9 \\
        --output_dir checkpoints/visnet_pretrained \\
        --hidden_channels 512 \\
        --num_layers 9 \\
        --num_rbf 64 \\
        --epochs 300 \\
        --batch_size 32 \\
        --lr 1e-4 \\
        --device cuda

    # Then use pretrained weights in Stage 1:
    python train_stage1.py ... \\
        --pretrained_visnet checkpoints/visnet_pretrained/best.pt \\
        --hidden_channels 512
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_scatter import scatter
from tqdm import tqdm

try:
    from src.model.visnet.models.visnet_block import ViSNetBlock
    from src.model.visnet.models.output_modules import EquivariantScalar
except ImportError as e:
    raise ImportError(
        "Cannot import from src/model/visnet/. "
        "Ensure src/model/visnet/models/ exists and torch_geometric/torch_scatter are installed."
    ) from e

try:
    from torch_geometric.datasets import QM9
except ImportError as e:
    raise ImportError("torch_geometric is required. pip install torch_geometric") from e

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# QM9 target index for U0 (internal energy at 0K, in eV)
_U0_IDX = 7


# ---------------------------------------------------------------------------
# QM9 dataset with flexible download URL
# ---------------------------------------------------------------------------

class _QM9(QM9):
    """QM9 with overrideable download URL and local-file support."""

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
                    print(f"  [_QM9] Copied {fname}")
            missing = [
                f for f in self.raw_file_names
                if not os.path.exists(os.path.join(self.raw_dir, f))
            ]
            if not missing:
                return
            print(f"  [_QM9] Still missing: {missing}; falling back to URL download.")
        super().download()


# ---------------------------------------------------------------------------
# Local ViSNet wrapper (avoids pytorch_lightning dependency in model.py)
# ---------------------------------------------------------------------------

class _LocalViSNet(nn.Module):
    """
    Minimal ViSNet wrapper combining ViSNetBlock + EquivariantScalar + Atomref.

    Avoids importing model.py / priors.py which depend on pytorch_lightning.

    Forward output: (B, 1) per-molecule energy prediction in original eV units.

    The forward computation:
        x, v  = ViSNetBlock(data)                      per-atom features
        x_out = EquivariantScalar.pre_reduce(x, v, ...) (N, 1) per-atom scalars
        x_out = x_out * std + atomref(z)               scale + add reference
        out   = scatter_sum(x_out, batch)              sum to per-molecule
        out   = out + mean                             shift by training mean
    """

    def __init__(
        self,
        representation_model: nn.Module,
        output_model: nn.Module,
        atomref: Tensor,   # (max_z,) per-element reference energies
        mean: float,
        std: float,
    ) -> None:
        super().__init__()
        self.representation_model = representation_model
        self.output_model = output_model
        self.atomref_embed = nn.Embedding(len(atomref), 1)
        self.atomref_embed.weight.data.copy_(atomref.view(-1, 1))
        self.register_buffer("mean", torch.scalar_tensor(float(mean)))
        self.register_buffer("std", torch.scalar_tensor(float(std)))

    def forward(self, data) -> Tensor:
        x, v = self.representation_model(data)
        x = self.output_model.pre_reduce(x, v, data.z, data.pos, data.batch)
        x = x * self.std
        x = x + self.atomref_embed(data.z)               # add per-atom reference
        out = scatter(x, data.batch, dim=0, reduce="add") # sum per molecule
        out = out + self.mean
        return out  # (B, 1)


# ---------------------------------------------------------------------------
# Per-element single-atom reference energies (B3LYP/6-31G(2df,p), eV)
# Extend this table when fine-tuning on datasets with new elements.
# ---------------------------------------------------------------------------

ATOMREF_TABLE: dict[int, float] = {
    1:  -13.61,    # H
    6:  -1029.86,  # C
    7:  -1485.30,  # N
    8:  -2042.61,  # O
    9:  -2715.57,  # F
    11: -4411.90,  # Na  (for Na-ion dataset fine-tuning)
}


def build_atomref(max_z: int, dataset=None, target_idx: int = 7) -> Tensor:
    """
    Build per-element reference energy tensor of shape (max_z,).

    Priority (highest to lowest):
      1. dataset.atomref(target_idx) — dataset built-in values (most accurate,
                                       same DFT level as training labels)
      2. ATOMREF_TABLE               — hard-coded fallback for extra elements
                                       (extend this dict for new element types)
      3. 0.0                         — unknown elements; the atomref embedding
                                       will adapt during training

    Returns: Tensor of shape (max_z,) with per-element energies in eV.
    """
    atomref = torch.zeros(max_z)

    # Layer 2: fill from hard-coded table
    for z, val in ATOMREF_TABLE.items():
        if z < max_z:
            atomref[z] = val

    # Layer 1: override with dataset values when available (higher priority)
    if dataset is not None:
        ds_ref = getattr(dataset, "atomref", None)
        if callable(ds_ref):
            ds_ref = ds_ref(target_idx)
        if ds_ref is not None:
            ds_ref = ds_ref.squeeze(-1)
            n = min(len(ds_ref), max_z)
            atomref[:n] = ds_ref[:n]

    present = atomref.nonzero(as_tuple=True)[0]
    print(
        "Atomref (single-atom DFT energies): "
        + ", ".join(f"z={z.item()}:{atomref[z].item():.2f} eV" for z in present)
    )
    return atomref


def compute_residual_stats(
    loader: DataLoader, atomref: Tensor
) -> tuple[float, float]:
    """Compute mean and std of (y - sum_i atomref[z_i]) over the dataset."""
    residuals = []
    for data in loader:
        y = data.y[:, _U0_IDX].cpu()
        atom_refs = atomref[data.z.cpu()]
        ref_sum = scatter(atom_refs, data.batch.cpu(), dim=0,
                          dim_size=int(data.batch.max().item()) + 1,
                          reduce="add")
        residuals.append(y - ref_sum)
    r = torch.cat(residuals)
    return r.mean().item(), r.std().item()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pretrain ViSNet on QM9 U0 energy")
    # Data
    p.add_argument("--data_root", default="data/qm9",
                   help="Directory for QM9 download/cache")
    p.add_argument("--output_dir", default="checkpoints/visnet_pretrained")
    # Model (defaults match ViSNet-QM9.yml paper config)
    p.add_argument("--hidden_channels", type=int, default=512,
                   help="Feature dimension (must match Stage 1/2 hidden_channels)")
    p.add_argument("--num_layers", type=int, default=9)
    p.add_argument("--num_rbf", type=int, default=64)
    p.add_argument("--cutoff", type=float, default=5.0)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--max_z", type=int, default=100)
    # Data split
    p.add_argument("--train_size", type=int, default=110000)
    p.add_argument("--val_size", type=int, default=10000)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap total dataset size for debugging")
    # Training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr_patience", type=int, default=15)
    p.add_argument("--lr_factor", type=float, default=0.8)
    p.add_argument("--lr_min", type=float, default=1e-7)
    p.add_argument("--max_norm", type=float, default=1.0)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--save_every", type=int, default=10)
    # QM9 download
    p.add_argument("--qm9_url", default=None,
                   help="Override QM9 download URL (useful for 403 errors)")
    p.add_argument("--qm9_raw_dir", default=None,
                   help="Path to local dir containing gdb9.sdf etc.")
    # W&B
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default="ed-pretrain-visnet")
    p.add_argument("--run_name", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

_QM9_MIRRORS = [
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/molnet_publish/qm9.zip",
    "https://data.pyg.org/datasets/qm9_v3.zip",
]


def make_splits(dataset, train_size: int, val_size: int, max_samples: int | None):
    n = len(dataset)
    if max_samples is not None:
        n = min(n, max_samples)
    train_size = min(train_size, n)
    val_size = min(val_size, n - train_size)
    idx = torch.arange(n)
    return (
        Subset(dataset, idx[:train_size].tolist()),
        Subset(dataset, idx[train_size:train_size + val_size].tolist()),
        Subset(dataset, idx[train_size + val_size:].tolist()),
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Return MAE in eV on the given loader."""
    model.eval()
    total_mae, total_n = 0.0, 0
    for data in loader:
        data = data.to(device)
        y_target = data.y[:, _U0_IDX]
        y_pred = model(data).squeeze(-1)
        total_mae += (y_pred - y_target).abs().sum().item()
        total_n += y_target.shape[0]
    return total_mae / max(total_n, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Dataset ---
    print(f"Loading QM9 from {args.data_root} ...")
    _QM9_MIRRORS_TO_TRY = [args.qm9_url] if args.qm9_url else _QM9_MIRRORS

    if args.qm9_raw_dir:
        dataset = _QM9(root=args.data_root, local_raw_dir=args.qm9_raw_dir)
    else:
        dataset = None
        for url in _QM9_MIRRORS_TO_TRY:
            try:
                dataset = _QM9(root=args.data_root, url_override=url)
                break
            except Exception as exc:
                print(f"  Download failed ({url!r}): {exc}")
        if dataset is None:
            raise RuntimeError(
                "All QM9 download URLs failed.\n"
                f"Use --qm9_raw_dir /path/to/raw/files or --qm9_url <mirror>."
            )

    train_set, val_set, _ = make_splits(
        dataset, args.train_size, args.val_size, args.max_samples
    )
    print(f"Split: train={len(train_set)}, val={len(val_set)}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(args.device == "cuda"),
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size * 4, shuffle=False,
        num_workers=args.num_workers,
    )

    # --- Atomref: single-atom DFT reference energies ---
    atomref = build_atomref(args.max_z, dataset=dataset, target_idx=_U0_IDX)
    mean, std = compute_residual_stats(train_loader, atomref)
    print(f"Residual stats: mean={mean:.4f} eV, std={std:.4f} eV")

    # --- Model ---
    representation_model = ViSNetBlock(
        lmax=2,
        vecnorm_type="max_min",
        trainable_vecnorm=False,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        hidden_channels=args.hidden_channels,
        num_rbf=args.num_rbf,
        rbf_type="expnorm",
        trainable_rbf=False,
        activation="silu",
        attn_activation="silu",
        max_z=args.max_z,
        cutoff=args.cutoff,
        max_num_neighbors=32,
        vertex_type="Edge",
    )
    output_model = EquivariantScalar(hidden_channels=args.hidden_channels)

    model = _LocalViSNet(
        representation_model=representation_model,
        output_model=output_model,
        atomref=atomref,
        mean=mean,
        std=std,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"ViSNet parameters: {param_count:,}")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=args.lr_patience,
        factor=args.lr_factor, min_lr=args.lr_min,
    )

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

        pbar = tqdm(train_loader, desc=f"Pretrain {epoch}/{args.epochs}", leave=False)
        for data in pbar:
            data = data.to(device)
            y_target = data.y[:, _U0_IDX]

            y_pred = model(data).squeeze(-1)
            loss = F.mse_loss(y_pred, y_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        val_mae = evaluate(model, val_loader, device)
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
                    "atomref": atomref,
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
                    "atomref": atomref,
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
            "atomref": atomref,
            "args": vars(args),
        },
        os.path.join(args.output_dir, "final.pt"),
    )
    if args.wandb:
        _wandb.finish()

    print(f"\nPretraining done. Best val MAE: {best_val_mae:.6f} eV")
    print(f"Best checkpoint: {args.output_dir}/best.pt")
    print(f"\nTo use in Stage 1/2 training, add:")
    print(f"  --pretrained_visnet {args.output_dir}/best.pt --hidden_channels {args.hidden_channels}")


if __name__ == "__main__":
    main()
