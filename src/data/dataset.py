"""
EDBench dataset.

Each sample stores:
  atom_coords         (N_atoms, 3)  float32  – atomic positions in Bohr
  atom_types          (N_atoms,)    int64    – atomic numbers (1-based)
  point_positions     (K, 3)        float32  – representative density points in Bohr
  point_log_densities (K,)          float32  – log(ρ + ε) at those positions

K = n_per_atom × N_atoms.

Two dataset classes are provided:
  - EDBenchPKLDataset : loads from the preprocessed .pkl file (primary, for EDBench)
  - EDBenchDataset    : loads from raw .cube files (kept for compatibility)

Both share the same collate_fn and output format.
"""
from __future__ import annotations

import os
import pickle
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .clustering import cluster_pointcloud, extract_representative_points

_LOG_EPS = 1e-10   # added inside log to prevent log(0)


# ---------------------------------------------------------------------------
# PKL-based dataset (primary for EDBench .pkl files)
# ---------------------------------------------------------------------------

class EDBenchPKLDataset(Dataset):
    """
    Dataset backed by the EDBench .pkl file.

    The .pkl contains ~48k molecules, each with:
      - mol.x      (N_atoms,)   atomic numbers
      - mol.coords (N_atoms, 3) atomic positions (Bohr)
      - electronic_density.coords  (M, 3) pre-filtered density point coords
      - electronic_density.density (M,)   density values at those points

    Density vacuum-filtering is already applied upstream (threshold in filename).
    This class applies K-Means clustering to obtain K = n_per_atom * N_atoms
    representative points per molecule.

    Since the .pkl is ~9 GB, it is NOT loaded into RAM. Instead, on first
    access each molecule is processed and cached as an individual .pt file
    in cache_dir. Subsequent accesses load the cached .pt directly.

    Args:
        pkl_path:    Path to the .pkl file.
        cache_dir:   Directory for per-molecule .pt cache files (required
                     for efficient multi-epoch training).
        n_per_atom:  Representative points per atom.
        max_samples: Cap dataset size (useful for debugging).
        preprocess:  If True, pre-process all molecules to cache_dir on init
                     (slow but avoids on-the-fly processing during training).
    """

    def __init__(
        self,
        pkl_path: str,
        cache_dir: str,
        n_per_atom: int = 8,
        max_samples: Optional[int] = None,
        preprocess: bool = False,
    ) -> None:
        self.pkl_path = pkl_path
        self.cache_dir = cache_dir
        self.n_per_atom = n_per_atom
        os.makedirs(cache_dir, exist_ok=True)

        # Load only the key list (not the data) to keep init fast
        print(f"Scanning {pkl_path} for molecule IDs …")
        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)
        self.mol_ids: list[str] = list(raw.keys())
        if max_samples is not None:
            self.mol_ids = self.mol_ids[:max_samples]

        # Store raw data in memory for the current session.
        # On a machine with sufficient RAM (~20 GB free) this avoids
        # repeated deserialization. For low-memory setups, set
        # preprocess=True and delete `self._raw` afterwards.
        self._raw: dict = raw

        if preprocess:
            self._preprocess_all()

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.mol_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        mol_id = self.mol_ids[idx]
        cache_path = self._cache_path(mol_id)

        if os.path.exists(cache_path):
            return torch.load(cache_path, weights_only=True)

        sample = self._process(mol_id)
        torch.save(sample, cache_path)
        return sample

    # ------------------------------------------------------------------
    def _process(self, mol_id: str) -> dict[str, torch.Tensor]:
        entry = self._raw[mol_id]
        mol = entry["mol"]
        ed = entry["electronic_density"]

        atom_types = mol["x"].astype(np.int64)          # (N,)
        atom_coords = mol["coords"].astype(np.float32)  # (N, 3)
        ed_coords = ed["coords"].astype(np.float32)     # (M, 3)
        ed_dens = ed["density"].astype(np.float32)      # (M,)

        K = self.n_per_atom * len(atom_types)
        positions, densities = cluster_pointcloud(ed_coords, ed_dens, K)

        return {
            "atom_coords": torch.from_numpy(atom_coords),          # (N, 3)
            "atom_types": torch.from_numpy(atom_types),            # (N,)
            "point_positions": torch.from_numpy(positions),        # (K, 3)
            "point_log_densities": torch.from_numpy(
                np.log(densities + _LOG_EPS).astype(np.float32)
            ),                                                      # (K,)
        }

    def _cache_path(self, mol_id: str) -> str:
        return os.path.join(self.cache_dir, f"{mol_id}_n{self.n_per_atom}.pt")

    def _preprocess_all(self) -> None:
        """Process all molecules and write .pt cache files."""
        total = len(self.mol_ids)
        for i, mol_id in enumerate(self.mol_ids):
            cache_path = self._cache_path(mol_id)
            if not os.path.exists(cache_path):
                sample = self._process(mol_id)
                torch.save(sample, cache_path)
            if (i + 1) % 1000 == 0:
                print(f"  Preprocessed {i + 1}/{total} molecules …")
        print(f"Preprocessing complete. Cache: {self.cache_dir}")


# ---------------------------------------------------------------------------
# Cube-file-based dataset (kept for compatibility)
# ---------------------------------------------------------------------------

class EDBenchDataset(Dataset):
    """
    Iterable over EDBench .cube files.

    Args:
        cube_dir:          Directory containing .cube files.
        n_per_atom:        Representative points per atom.
        density_threshold: Vacuum density cutoff for K-Means filtering.
        cache_dir:         If given, preprocessed tensors are written here and
                           reloaded on subsequent accesses.
        max_samples:       Cap on dataset size.
    """

    def __init__(
        self,
        cube_dir: str,
        n_per_atom: int = 8,
        density_threshold: float = 1e-4,
        cache_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        from .cube_parser import parse_cube as _parse_cube
        self._parse_cube = _parse_cube

        self.n_per_atom = n_per_atom
        self.density_threshold = density_threshold
        self.cache_dir = cache_dir

        self.cube_files: list[str] = sorted(
            os.path.join(cube_dir, f)
            for f in os.listdir(cube_dir)
            if f.endswith(".cube")
        )
        if max_samples is not None:
            self.cube_files = self.cube_files[:max_samples]

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def __len__(self) -> int:
        return len(self.cube_files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        cube_path = self.cube_files[idx]

        if self.cache_dir:
            cache_path = self._cache_path(cube_path)
            if os.path.exists(cache_path):
                return torch.load(cache_path, weights_only=True)

        sample = self._process(cube_path)

        if self.cache_dir:
            torch.save(sample, self._cache_path(cube_path))

        return sample

    def _process(self, cube_path: str) -> dict[str, torch.Tensor]:
        cube = self._parse_cube(cube_path)
        positions, densities = extract_representative_points(
            cube,
            n_per_atom=self.n_per_atom,
            density_threshold=self.density_threshold,
        )
        return {
            "atom_coords": torch.from_numpy(cube.atom_coords),
            "atom_types": torch.from_numpy(cube.atom_types).long(),
            "point_positions": torch.from_numpy(positions),
            "point_log_densities": torch.from_numpy(
                np.log(densities + _LOG_EPS).astype(np.float32)
            ),
        }

    def _cache_path(self, cube_path: str) -> str:
        stem = os.path.splitext(os.path.basename(cube_path))[0]
        return os.path.join(self.cache_dir, f"{stem}_n{self.n_per_atom}.pt")


# ---------------------------------------------------------------------------
# Collate function for DataLoader
# ---------------------------------------------------------------------------

def collate_fn(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Collate a list of samples into a batch.

    Since molecules have variable numbers of atoms (N) and points (K),
    we concatenate along the first axis and provide batch-index tensors
    (atom_batch and point_batch) compatible with PyTorch Geometric.
    """
    atom_coords_list = []
    atom_types_list = []
    point_positions_list = []
    point_log_dens_list = []
    atom_batch_list = []
    point_batch_list = []

    for b_idx, s in enumerate(samples):
        n = s["atom_coords"].shape[0]
        k = s["point_positions"].shape[0]

        atom_coords_list.append(s["atom_coords"])
        atom_types_list.append(s["atom_types"])
        point_positions_list.append(s["point_positions"])
        point_log_dens_list.append(s["point_log_densities"])
        atom_batch_list.append(torch.full((n,), b_idx, dtype=torch.long))
        point_batch_list.append(torch.full((k,), b_idx, dtype=torch.long))

    return {
        "atom_coords": torch.cat(atom_coords_list),            # (sum_N, 3)
        "atom_types": torch.cat(atom_types_list),              # (sum_N,)
        "point_positions": torch.cat(point_positions_list),    # (sum_K, 3)
        "point_log_densities": torch.cat(point_log_dens_list), # (sum_K,)
        "atom_batch": torch.cat(atom_batch_list),              # (sum_N,)
        "point_batch": torch.cat(point_batch_list),            # (sum_K,)
    }
