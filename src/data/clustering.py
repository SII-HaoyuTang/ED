"""
Extract representative electron density points using density-weighted
K-Means clustering.

Goal: produce K = n_per_atom * N_atoms points that:
  - Are spread out across the electron cloud (K-Means guarantees this)
  - Are attracted toward high-density regions (via density weighting)
  - Cover the full extent of the density (not just atomic cores)

Two entry points are provided:
  - cluster_pointcloud()  : operates directly on a pre-filtered point cloud
                            (the primary path for the EDBench .pkl dataset)
  - extract_representative_points() : operates on a CubeData object
                            (kept for compatibility with raw .cube files)
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import MiniBatchKMeans


def cluster_pointcloud(
    coords: np.ndarray,
    densities: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract n_clusters representative points from a pre-filtered point cloud
    using density-weighted K-Means.

    The input point cloud is assumed to already have vacuum points removed
    (e.g. by a density threshold applied upstream).

    Args:
        coords:       (M, 3) float32, point coordinates.
        densities:    (M,)   float32, density values (all > 0).
        n_clusters:   K, number of representative points to produce.
        random_state: Seed for reproducibility.

    Returns:
        centers:          (K, 3) float32, cluster center coordinates.
        center_densities: (K,)   float32, density at nearest input point
                          to each cluster center.
    """
    if len(coords) < n_clusters:
        # Edge case: fewer input points than requested clusters.
        # Pad by repeating existing points.
        repeat = int(np.ceil(n_clusters / len(coords)))
        coords = np.tile(coords, (repeat, 1))[:n_clusters]
        densities = np.tile(densities, repeat)[:n_clusters]
        return coords.astype(np.float32), densities.astype(np.float32)

    weights = densities / densities.sum()

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=3,
        random_state=random_state,
        batch_size=min(10_000, len(coords)),
    )
    kmeans.fit(coords, sample_weight=weights)
    centers = kmeans.cluster_centers_.astype(np.float32)   # (K, 3)
    center_densities = _nearest_density(centers, coords, densities)
    return centers, center_densities


def extract_representative_points(
    cube,
    n_per_atom: int = 8,
    density_threshold: float = 1e-4,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract K = n_per_atom * N_atoms representative points from a CubeData
    object using density-weighted K-Means.

    Args:
        cube:              Parsed CubeData object (from cube_parser).
        n_per_atom:        Number of representative points per atom.
        density_threshold: Grid points below this value are excluded.
        random_state:      Seed for reproducibility.

    Returns:
        positions:  (K, 3) float32, representative point coordinates in Bohr.
        densities:  (K,)   float32, density values at those positions.
    """
    from .cube_parser import get_grid_coords

    K = n_per_atom * len(cube.atom_types)

    grid_coords = get_grid_coords(cube)
    dens_flat = cube.density.flatten()

    mask = dens_flat > density_threshold
    if mask.sum() < K:
        thresh_fallback = np.percentile(dens_flat, 90.0)
        mask = dens_flat > thresh_fallback

    return cluster_pointcloud(
        grid_coords[mask], dens_flat[mask], K, random_state=random_state
    )


def _nearest_density(
    queries: np.ndarray,
    source_coords: np.ndarray,
    source_dens: np.ndarray,
    chunk: int = 512,
) -> np.ndarray:
    """
    For each query point, return the density of the nearest source point.
    Operates in chunks to keep memory usage bounded.
    """
    K = len(queries)
    densities = np.empty(K, dtype=np.float32)

    for start in range(0, K, chunk):
        end = min(start + chunk, K)
        q = queries[start:end]                          # (c, 3)
        diff = q[:, None, :] - source_coords[None, :, :]  # (c, M, 3)
        sq_dist = (diff ** 2).sum(axis=-1)              # (c, M)
        nn_idx = sq_dist.argmin(axis=-1)                # (c,)
        densities[start:end] = source_dens[nn_idx]

    return densities
