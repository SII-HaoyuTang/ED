"""
Parser for Gaussian .cube file format.

Format specification:
  Line 1-2: comment lines
  Line 3:   N_atoms  origin_x  origin_y  origin_z  (Bohr)
  Line 4-6: N_voxels  step_x  step_y  step_z  (one per axis)
  Lines 7 to 7+N_atoms-1: atomic_num  charge  x  y  z
  Remaining: density values (N_x * N_y * N_z total, row-major)

All coordinates in Bohr (atomic units). Density in e/Bohr^3.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CubeData:
    atom_types: np.ndarray    # (N_atoms,) int, atomic numbers
    atom_coords: np.ndarray   # (N_atoms, 3) float, positions in Bohr
    origin: np.ndarray        # (3,) float, grid origin in Bohr
    axes: np.ndarray          # (3, 3) float, voxel step vectors [axis0; axis1; axis2]
    n_voxels: np.ndarray      # (3,) int, number of voxels per axis
    density: np.ndarray       # (N_x, N_y, N_z) float, electron density


def parse_cube(filepath: str) -> CubeData:
    """Parse a .cube file and return a CubeData object."""
    with open(filepath, "r") as f:
        lines = f.readlines()

    # Line 2 (0-indexed): N_atoms  origin_x  origin_y  origin_z
    tokens = lines[2].split()
    n_atoms = abs(int(tokens[0]))  # negative means MO cube; take abs
    origin = np.array([float(x) for x in tokens[1:4]])

    # Lines 3-5: voxel axes
    n_voxels = np.empty(3, dtype=int)
    axes = np.empty((3, 3))
    for i in range(3):
        toks = lines[3 + i].split()
        n_voxels[i] = int(toks[0])
        axes[i] = [float(x) for x in toks[1:4]]

    # Atom lines
    atom_types = np.empty(n_atoms, dtype=int)
    atom_coords = np.empty((n_atoms, 3))
    for i in range(n_atoms):
        toks = lines[6 + i].split()
        atom_types[i] = int(toks[0])
        atom_coords[i] = [float(x) for x in toks[2:5]]

    # Density values (flat, row-major N_x × N_y × N_z)
    density_tokens: list[str] = []
    for line in lines[6 + n_atoms:]:
        density_tokens.extend(line.split())

    density = np.array(density_tokens, dtype=np.float32).reshape(n_voxels)

    return CubeData(
        atom_types=atom_types,
        atom_coords=atom_coords.astype(np.float32),
        origin=origin.astype(np.float32),
        axes=axes.astype(np.float32),
        n_voxels=n_voxels,
        density=density,
    )


def get_grid_coords(cube: CubeData) -> np.ndarray:
    """
    Compute 3D coordinates of every grid point.

    Returns:
        coords: (N_x * N_y * N_z, 3) float32, positions in Bohr
    """
    ix = np.arange(cube.n_voxels[0], dtype=np.float32)
    iy = np.arange(cube.n_voxels[1], dtype=np.float32)
    iz = np.arange(cube.n_voxels[2], dtype=np.float32)

    gx, gy, gz = np.meshgrid(ix, iy, iz, indexing="ij")  # each (Nx, Ny, Nz)

    # coords[i,j,k] = origin + i*axis0 + j*axis1 + k*axis2
    coords = (
        cube.origin
        + gx[..., None] * cube.axes[0]
        + gy[..., None] * cube.axes[1]
        + gz[..., None] * cube.axes[2]
    )  # (Nx, Ny, Nz, 3)

    return coords.reshape(-1, 3)
