"""
K-Point Grid Module

This module contains functions for generating k-point grids and neighbor lists
with periodic boundary conditions.
"""

import numpy as np
from typing import Tuple, Dict, List


def generate_kpoint_grid(k_grid: Tuple[int, int, int]) -> np.ndarray:
    """
    Generate a Monkhorst-Pack k-point grid in fractional coordinates.
    
    The k-points are uniformly distributed in the first Brillouin zone
    using fractional coordinates: k = (i/nk1, j/nk2, k/nk3).
    
    Parameters
    ----------
    k_grid : tuple of 3 ints
        Dimensions of the k-point grid (nk1, nk2, nk3)
    
    Returns
    -------
    kpoints : ndarray of shape (num_kpoints, 3)
        Array of k-points in fractional coordinates
    
    Examples
    --------
    >>> k_grid = (2, 2, 2)
    >>> kpoints = generate_kpoint_grid(k_grid)
    >>> print(kpoints.shape)
    (8, 3)
    >>> print(kpoints[0])
    [0. 0. 0.]
    """
    nk1, nk2, nk3 = k_grid
    kpoints = []
    
    for i in range(nk1):
        for j in range(nk2):
            for k in range(nk3):
                kx = i / nk1
                ky = j / nk2
                kz = k / nk3
                kpoints.append([kx, ky, kz])
    
    return np.array(kpoints)


def generate_neighbor_list(
    k_grid: Tuple[int, int, int]
) -> Dict[int, List[Tuple[int, np.ndarray]]]:
    """
    Generate a neighbor list for all k-points in the grid.
    
    Each k-point has 6 neighbors corresponding to ±1 shifts in each direction
    with periodic boundary conditions.
    
    Parameters
    ----------
    k_grid : tuple of 3 ints
        Dimensions of the k-point grid (nk1, nk2, nk3)
    
    Returns
    -------
    neighbor_list : dict
        Maps k_idx -> list of (neighbor_idx, lattice_vector_b) tuples
        where lattice_vector_b is the connecting vector in k-space
    
    Examples
    --------
    >>> k_grid = (3, 3, 3)
    >>> neighbor_list = generate_neighbor_list(k_grid)
    >>> len(neighbor_list)
    27
    >>> len(neighbor_list[0])  # Each k-point has 6 neighbors
    6
    """
    nk1, nk2, nk3 = k_grid
    neighbor_list = {}
    
    # Direction vectors for neighbors: ±x, ±y, ±z
    directions = [
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1)
    ]
    
    for i in range(nk1):
        for j in range(nk2):
            for k in range(nk3):
                k_idx = i * nk2 * nk3 + j * nk3 + k
                neighbors = []
                
                for di, dj, dk in directions:
                    # Apply periodic boundary conditions
                    ni = (i + di) % nk1
                    nj = (j + dj) % nk2
                    nk = (k + dk) % nk3
                    
                    neighbor_idx = ni * nk2 * nk3 + nj * nk3 + nk
                    
                    # Lattice vector b in k-space (integer shifts)
                    b = np.array([di, dj, dk], dtype=int)
                    
                    neighbors.append((neighbor_idx, b))
                
                neighbor_list[k_idx] = neighbors
    
    return neighbor_list


def kpoint_index_to_grid(k_idx: int, k_grid: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """
    Convert linear k-point index to grid indices.
    
    Parameters
    ----------
    k_idx : int
        Linear k-point index
    k_grid : tuple of 3 ints
        Dimensions of the k-point grid
    
    Returns
    -------
    tuple of 3 ints
        Grid indices (i, j, k)
    """
    nk1, nk2, nk3 = k_grid
    i = k_idx // (nk2 * nk3)
    remainder = k_idx % (nk2 * nk3)
    j = remainder // nk3
    k = remainder % nk3
    return i, j, k


def grid_to_kpoint_index(i: int, j: int, k: int, k_grid: Tuple[int, int, int]) -> int:
    """
    Convert grid indices to linear k-point index.
    
    Parameters
    ----------
    i, j, k : int
        Grid indices
    k_grid : tuple of 3 ints
        Dimensions of the k-point grid
    
    Returns
    -------
    int
        Linear k-point index
    """
    nk1, nk2, nk3 = k_grid
    return i * nk2 * nk3 + j * nk3 + k
