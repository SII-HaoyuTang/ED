"""
EDBench energy prediction dataset (ED5-EC task).

Loads electron density point clouds from the EDBench .pkl file and
6-component DFT energy labels from the accompanying CSV, applies
farthest-point sampling (FPS) to obtain a fixed-size point cloud, and
caches the results for fast multi-epoch training.

Input per sample:
    point_cloud  (npoint, 4)  – [x, y, z, electron_density]  (Bohr + a.u.)
    energies     (6,)         – DFT energy components (Hartree):
                                  0: @DF-RKS Final Energy
                                  1: Nuclear Repulsion Energy
                                  2: One-Electron Energy
                                  3: Two-Electron Energy
                                  4: DFT Exchange-Correlation Energy
                                  5: Total Energy

Dataset split: uses the scaffold_split column from the CSV
(values: "train" / "valid" / "test").
"""
from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# FPS helper
# ---------------------------------------------------------------------------

def _fps(pos: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Farthest Point Sampling on a single (M, 3) point cloud.
    Returns npoint indices (at most M).
    Uses torch_cluster.fps; falls back to random if not available.
    """
    M = pos.shape[0]
    if npoint >= M:
        return torch.arange(M)

    try:
        from torch_cluster import fps as _fps_cluster
        # fps expects (N, 3) and a batch vector; here batch_size=1
        batch = torch.zeros(M, dtype=torch.long, device=pos.device)
        ratio = npoint / M
        idx = _fps_cluster(pos, batch, ratio=ratio, random_start=False)
        return idx[:npoint]
    except ImportError:
        # Fallback: random sampling (only used if torch_cluster missing)
        return torch.randperm(M)[:npoint]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EDBenchEnergyDataset(Dataset):
    """
    Point-cloud energy prediction dataset for the EDBench ED5-EC task.

    Args:
        pkl_path:   Path to mol_EDthresh0.05_data.pkl (~9 GB).
        csv_path:   Path to ed_energy_5w.csv (energy labels + splits).
        cache_dir:  Directory for per-molecule .pt cache files.
        split:      "train", "valid", or "test"  (scaffold_split column).
        npoint:     Number of points after FPS (default 2048).
        max_samples: Cap for debugging.
    """

    _ENERGY_NAMES = [
        "Final Energy",
        "Nuclear Repulsion Energy",
        "One-Electron Energy",
        "Two-Electron Energy",
        "DFT Exchange-Correlation Energy",
        "Total Energy",
    ]

    def __init__(
        self,
        pkl_path: str,
        csv_path: str,
        cache_dir: str,
        split: str = "train",
        npoint: int = 2048,
        max_samples: Optional[int] = None,
    ) -> None:
        assert split in ("train", "valid", "test"), \
            f"split must be 'train', 'valid', or 'test', got {split!r}"

        self.pkl_path = pkl_path
        self.npoint = npoint
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir

        # --- Load CSV labels and splits ---
        df = pd.read_csv(csv_path)
        df = df[df["scaffold_split"] == split].reset_index(drop=True)

        # Parse label column: space-separated 6 floats
        energies = np.stack(
            df["label"].apply(lambda s: np.fromstring(s, sep=" ")).values
        ).astype(np.float32)           # (N_split, 6)

        self.mol_ids: list[str] = df["index"].astype(str).tolist()
        self.energies: np.ndarray = energies

        if max_samples is not None:
            self.mol_ids = self.mol_ids[:max_samples]
            self.energies = self.energies[:max_samples]

        # --- Load PKL molecule index (data kept in memory) ---
        print(f"Loading PKL from {pkl_path} …")
        with open(pkl_path, "rb") as f:
            self._raw: dict = pickle.load(f)

        # Filter to only molecules that exist in both CSV and PKL
        valid_mask = [mid in self._raw for mid in self.mol_ids]
        if not all(valid_mask):
            missing = sum(1 for v in valid_mask if not v)
            print(f"  WARNING: {missing} mol_ids not found in PKL, skipping.")
            self.mol_ids = [mid for mid, v in zip(self.mol_ids, valid_mask) if v]
            self.energies = self.energies[[i for i, v in enumerate(valid_mask) if v]]

        print(f"  {split}: {len(self.mol_ids)} molecules")

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.mol_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        mol_id = self.mol_ids[idx]
        cache_path = self._cache_path(mol_id)

        if os.path.exists(cache_path):
            sample = torch.load(cache_path, weights_only=True)
        else:
            sample = self._process(mol_id)
            torch.save(sample, cache_path)

        return {
            "point_cloud": sample["point_cloud"],           # (npoint, 4)
            "energies": torch.from_numpy(self.energies[idx]),  # (6,)
        }

    # ------------------------------------------------------------------
    def _process(self, mol_id: str) -> dict[str, torch.Tensor]:
        entry = self._raw[mol_id]
        ed = entry["electronic_density"]

        coords = ed["coords"].astype(np.float32)   # (M, 3)  Bohr
        dens = ed["density"].astype(np.float32)    # (M,)    a.u.

        # Combine to (M, 4)
        cloud = np.concatenate([coords, dens[:, None]], axis=1)  # (M, 4)

        # FPS sampling
        pos_t = torch.from_numpy(coords)
        idx = _fps(pos_t, self.npoint)
        sampled = torch.from_numpy(cloud)[idx]          # (npoint, 4)

        # Pad if the molecule has fewer than npoint density points
        if sampled.shape[0] < self.npoint:
            pad = self.npoint - sampled.shape[0]
            sampled = torch.cat([sampled, sampled[:pad]], dim=0)

        return {"point_cloud": sampled}                  # (npoint, 4)

    def _cache_path(self, mol_id: str) -> str:
        return os.path.join(self.cache_dir, f"{mol_id}_fps{self.npoint}.pt")


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def energy_collate_fn(samples: list[dict]) -> dict[str, torch.Tensor]:
    """Stack fixed-size samples into a batch."""
    return {
        "point_cloud": torch.stack([s["point_cloud"] for s in samples]),  # (B, npoint, 4)
        "energies":    torch.stack([s["energies"]    for s in samples]),  # (B, 6)
    }
