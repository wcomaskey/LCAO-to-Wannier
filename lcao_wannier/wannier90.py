"""
Wannier90 File Writer Module

This module contains functions for writing Wannier90 input files
(.eig, .amn, .mmn) according to the Wannier90 file format specification.
"""

import numpy as np
from typing import List, Dict, Tuple, Any

from .fourier import fourier_transform_to_kspace


def compute_mmn_matrix(
    k_idx: int,
    next_k_idx: int,
    kpoints: np.ndarray,
    b_vector_cart: np.ndarray,
    real_space_matrices: Dict[Tuple[int, int, int], Dict[str, np.ndarray]],
    lattice_vectors: np.ndarray,
    atom_positions: np.ndarray,
    basis_atom_map: np.ndarray,
    eigenvectors: List[np.ndarray]
) -> np.ndarray:
    """
    Compute M_mn matrix using the Symmetric Midpoint Approximation.

    This implements the correct MMN matrix calculation for LCAO methods,
    accounting for periodic boundary conditions via the physical b-vector.

    The symmetric midpoint method evaluates the overlap matrix S at the
    midpoint between k and k+b, then applies a symmetric Berry phase
    correction based on the average atomic positions.

    Parameters
    ----------
    k_idx : int
        Index of the current k-point
    next_k_idx : int
        Index of the neighbor k-point
    kpoints : ndarray of shape (num_kpoints, 3)
        K-points in fractional coordinates
    b_vector_cart : ndarray of shape (3,)
        Cartesian b-vector connecting k to k+b (in Angstrom)
    real_space_matrices : dict
        Dict mapping (n1,n2,n3) -> {'H': H_matrix, 'S': S_matrix}
    lattice_vectors : ndarray of shape (3, 3)
        Real-space lattice vectors (rows are [a1, a2, a3])
    atom_positions : ndarray of shape (num_atoms, 3)
        Atomic positions in Cartesian coordinates
    basis_atom_map : ndarray of shape (num_orbitals,)
        Maps each orbital to its atom index
    eigenvectors : list of ndarray
        Eigenvector matrices C(k), shape (num_orbitals, num_bands)

    Returns
    -------
    M_mn : ndarray
        MMN overlap matrix, shape (num_bands, num_bands)

    Notes
    -----
    The symmetric midpoint method:
    1. Computes k_mid = k + 0.5 * b_frac (using b-vector, not averaging k-points)
    2. Evaluates S(k_mid) via Fourier transform
    3. Applies symmetric Berry phase: exp[-i * b · (τ_i + τ_j) / 2]
    4. Projects onto eigenvectors: M = C†(k) @ S_cross @ C(k+b)

    This correctly handles periodic boundary crossings where k+b may wrap
    around the Brillouin zone.
    """
    # 1. Get current k-point in fractional coordinates
    k_curr = kpoints[k_idx]

    # 2. Midpoint Strategy (Corrected for PBC)
    # Convert Cartesian b_vector to fractional coordinates
    # lattice_vectors rows are [a1, a2, a3], need inverse of transpose
    inv_lattice = np.linalg.inv(lattice_vectors.T)
    b_frac = np.dot(inv_lattice, b_vector_cart)

    # k_mid is exactly half a b-step away from k_curr
    k_mid = k_curr + 0.5 * b_frac

    # 3. Get S(k_mid) using Fourier transform
    _, S_mid = fourier_transform_to_kspace(k_mid, real_space_matrices, lattice_vectors)

    # 4. Symmetric Berry Phase Correction
    # Formula: exp[-i * b · (τ_i + τ_j) / 2]

    # Get atom positions for each orbital (num_orbitals, 3)
    taus = atom_positions[basis_atom_map]

    # Calculate average position pairs: (τ_i + τ_j) / 2
    # Broadcasting: (N, 1, 3) + (1, N, 3) -> (N, N, 3)
    tau_mid = (taus[:, None, :] + taus[None, :, :]) / 2.0

    # Calculate phase exponent: b_vec · tau_mid
    # Sum over Cartesian components (axis 2)
    phase_exponent = np.sum(tau_mid * b_vector_cart, axis=2)
    phase_matrix = np.exp(-1j * phase_exponent)

    # 5. Apply phase correction element-wise to get cross-overlap
    S_cross = S_mid * phase_matrix

    # 6. Project onto eigenvectors
    C_k = eigenvectors[k_idx]
    C_next = eigenvectors[next_k_idx]

    M_mn = C_k.conj().T @ S_cross @ C_next

    return M_mn


def write_eig_file(
    filename: str,
    eigenvalues_list: List[np.ndarray],
    num_kpoints: int,
    num_wann: int
) -> None:
    """
    Write the .eig file containing eigenvalues.
    
    Format:
        band_index  k_index  eigenvalue(real)
    
    Parameters
    ----------
    filename : str
        Output filename (e.g., 'material.eig')
    eigenvalues_list : list of ndarrays
        Eigenvalues for each k-point
    num_kpoints : int
        Total number of k-points
    num_wann : int
        Number of Wannier functions (bands)
    
    Notes
    -----
    Wannier90 uses 1-based indexing for band and k-point indices.
    """
    with open(filename, 'w') as f:
        for k_idx in range(num_kpoints):
            eigenvalues = eigenvalues_list[k_idx]
            for band_idx in range(num_wann):
                # Wannier90 uses 1-based indexing
                f.write(
                    f"{band_idx + 1:5d} {k_idx + 1:5d} "
                    f"{eigenvalues[band_idx].real:18.12f}\n"
                )


def write_amn_file_lcao(
    filename: str,
    eigenvectors_list: List[np.ndarray],
    S_k_list: List[np.ndarray],
    orbital_indices: np.ndarray,
    band_indices: np.ndarray,
    num_kpoints: int,
    num_wann: int
) -> None:
    """
    Write the .amn file for LCAO methods with non-orthogonal basis correction.

    For LCAO methods with non-orthogonal basis, the projection is:
        A_mn(k) = <χ_m | ψ_n(k)> = [S(k) @ C(k)]_mn

    where:
    - χ_m are the selected LCAO trial orbitals (projection functions)
    - ψ_n(k) are the Bloch eigenstates
    - S(k) is the overlap matrix
    - C(k) are the eigenvector coefficients

    This accounts for the non-orthogonality of the LCAO basis functions.

    Parameters
    ----------
    filename : str
        Output .amn filename
    eigenvectors_list : list of ndarray
        Full eigenvector matrices C(k) for each k-point, shape (num_orbitals, num_bands_all)
    S_k_list : list of ndarray
        Full overlap matrices S(k) for each k-point, shape (num_orbitals, num_orbitals)
    orbital_indices : ndarray
        Indices of selected projection orbitals
    band_indices : ndarray
        Indices of selected bands
    num_kpoints : int
        Number of k-points
    num_wann : int
        Number of Wannier functions
    """
    num_bands = len(band_indices)  # Number of bands to use
    num_proj = len(orbital_indices)  # Number of projections

    with open(filename, 'w') as f:
        # Write header
        # Wannier90 format: num_bands num_kpoints num_wann
        f.write("Created by LCAO-to-Wannier90 (Overlap-Corrected)\n")
        f.write(f"{num_bands:5d} {num_kpoints:5d} {num_proj:5d}\n")

        # Write projection matrices
        for k_idx in range(num_kpoints):
            C_k = eigenvectors_list[k_idx]  # Shape: (num_orbitals, num_bands_all)
            S_k = S_k_list[k_idx]  # Shape: (num_orbitals, num_orbitals)

            # Compute projection: P = S @ C
            # This gives <χ_i | ψ_n> for all orbitals i and bands n
            P_k = S_k @ C_k  # Shape: (num_orbitals, num_bands_all)

            # Extract selected orbitals (rows) and selected bands (columns)
            A_k = P_k[np.ix_(orbital_indices, band_indices)]  # Shape: (num_proj, num_bands)

            # Write elements: loop over bands m, then projectors n
            # Wannier90 format: band_idx  wannier_idx  kpoint_idx  Re(A)  Im(A)
            for m in range(num_bands):
                for n in range(num_proj):
                    # Wannier90 uses 1-based indexing
                    f.write(
                        f"{m + 1:5d} {n + 1:5d} {k_idx + 1:5d} "
                        f"{A_k[n, m].real:18.12f} {A_k[n, m].imag:18.12f}\n"
                    )


def write_amn_file(
    filename: str,
    eigenvectors_list: List[np.ndarray],
    S_k_list: List[np.ndarray],
    num_kpoints: int,
    num_wann: int
) -> None:
    """
    Write the .amn file containing projection matrices A(k).
    
    A(k) = S(k)† C(k)
    
    Format:
        Header: num_bands  num_kpoints  num_wann
        For each k-point and band:
            band_m  projection_n  k_idx  Re(A_mn)  Im(A_mn)
    
    Parameters
    ----------
    filename : str
        Output filename (e.g., 'material.amn')
    eigenvectors_list : list of ndarrays
        Eigenvectors C(k) for each k-point
    S_k_list : list of ndarrays
        Overlap matrices S(k) for each k-point
    num_kpoints : int
        Total number of k-points
    num_wann : int
        Number of Wannier functions
    
    Notes
    -----
    The projection matrix is computed as A(k) = S(k)† C(k), where
    † denotes the conjugate transpose.
    """
    with open(filename, 'w') as f:
        # Write header
        f.write("Created by LCAO-to-Wannier90 Engine\n")
        f.write(f"{num_wann:5d} {num_kpoints:5d} {num_wann:5d}\n")
        
        # Write projection matrices
        for k_idx in range(num_kpoints):
            C_k = eigenvectors_list[k_idx]
            S_k = S_k_list[k_idx]

            # For LCAO methods: A(k) = projection of eigenvectors onto selected orbitals
            # If S_k is a projection subspace matrix (num_wann x num_orbitals),
            # it contains the selected orbital indices via row selection.
            # The projection is simply the selected rows of C_k.
            if S_k.shape[0] == num_wann and S_k.shape[0] < S_k.shape[1]:
                # S_k indicates which orbitals to use via its row structure
                # For LCAO: A_mn(k) = C_mn(k) where m are selected orbital indices
                # We need to extract which orbitals S_k represents
                # Actually, S_k rows ARE the projections: A(k) = S_k @ C_k normalizes them
                A_k = S_k @ C_k
            else:
                # S_k is the full overlap matrix (num_orbitals x num_orbitals)
                # Use traditional formula: A(k) = S(k)† C(k)
                A_k = S_k.conj().T @ C_k
            
            # Write elements: loop over bands m, then projectors n
            for m in range(num_wann):
                for n in range(num_wann):
                    # Wannier90 uses 1-based indexing
                    f.write(
                        f"{m + 1:5d} {n + 1:5d} {k_idx + 1:5d} "
                        f"{A_k[n, m].real:18.12f} {A_k[n, m].imag:18.12f}\n"
                    )


def write_mmn_file_lcao(
    filename: str,
    eigenvectors_list: List[np.ndarray],
    kpoints: np.ndarray,
    real_space_matrices: Dict[Tuple[int, int, int], Dict[str, np.ndarray]],
    lattice_vectors: np.ndarray,
    neighbor_list: Dict[int, List[Dict[str, Any]]],
    atom_positions: np.ndarray,
    basis_atom_map: np.ndarray,
    num_kpoints: int,
    num_wann: int,
    convention: str = 'pi'
) -> None:
    """
    Write the .mmn file for LCAO methods using the Symmetric Midpoint Method.

    This implements the correct MMN matrix calculation using the symmetric
    midpoint approximation, which properly handles periodic boundary conditions
    by using the physical b-vector rather than averaging k-point coordinates.

    The method:
        1. Computes k_mid = k + 0.5 * b_frac (using b-vector for PBC)
        2. Evaluates S(k_mid) via Fourier transform
        3. Applies symmetric Berry phase: exp[-i * b · (τ_i + τ_j) / 2]
        4. Projects onto eigenvectors: M = C†(k) @ S_cross @ C(k+b)

    Parameters
    ----------
    filename : str
        Output .mmn filename
    eigenvectors_list : list of ndarray
        Eigenvector matrices C(k), shape (num_orbitals, num_bands)
    kpoints : ndarray
        K-points in fractional coordinates, shape (num_kpoints, 3)
    real_space_matrices : dict
        Dict mapping (n1,n2,n3) -> {'H': H_matrix, 'S': S_matrix}
    lattice_vectors : ndarray
        Real-space lattice vectors, shape (3, 3)
    neighbor_list : dict
        Maps k_idx -> list of neighbor dicts with keys:
            'id': neighbor k-point index
            'G_shift': integer lattice vector (n1, n2, n3)
            'b_vec_cart': Cartesian b-vector in Angstrom
    atom_positions : ndarray
        Atomic positions in Cartesian coordinates, shape (num_atoms, 3)
    basis_atom_map : ndarray
        Maps each orbital to its atom index, shape (num_orbitals,)
    num_kpoints : int
        Number of k-points
    num_wann : int
        Number of Wannier functions (selected bands)
    convention : str, optional
        'pi' for π convention (Crystal23), '2pi' for 2π convention
        Default is 'pi' (unused in symmetric midpoint method)
    """
    with open(filename, 'w') as f:
        # Write header
        f.write("Created by LCAO-to-Wannier90 (Symmetric Midpoint Method)\n")
        num_neighbors = len(neighbor_list[0])
        f.write(f"{num_wann:5d} {num_kpoints:5d} {num_neighbors:5d}\n")

        # Process each k-point
        for k_idx in range(num_kpoints):
            # Process each neighbor
            for neighbor in neighbor_list[k_idx]:
                k_next_idx = neighbor['id']
                G_shift = neighbor['G_shift']
                b_vec_cart = neighbor['b_vec_cart']  # Cartesian b-vector

                # Compute MMN matrix using symmetric midpoint method
                M_kb = compute_mmn_matrix(
                    k_idx, k_next_idx, kpoints, b_vec_cart,
                    real_space_matrices, lattice_vectors,
                    atom_positions, basis_atom_map, eigenvectors_list
                )

                # --- WRITE TO FILE ---
                # Write neighbor identification line
                f.write(f"{k_idx+1:5d} {k_next_idx+1:5d} "
                       f"{G_shift[0]:5d} {G_shift[1]:5d} {G_shift[2]:5d}\n")

                # Write matrix elements (column-major order for Fortran compatibility)
                for n in range(num_wann):
                    for m in range(num_wann):
                        val = M_kb[m, n]
                        f.write(f"{val.real:18.12f} {val.imag:18.12f}\n")


def write_mmn_file(
    filename: str,
    eigenvectors_list: List[np.ndarray],
    S_k_list: List[np.ndarray],
    neighbor_list: Dict[int, List[Tuple[int, np.ndarray]]],
    num_kpoints: int,
    num_wann: int
) -> None:
    """
    Write the .mmn file containing overlap matrices M(k,b).

    M(k,b) = C†(k) S(k+b) C(k+b)

    Format:
        Header: num_bands  num_kpoints  num_neighbors
        For each k-point and neighbor:
            k_idx  neighbor_idx  b1  b2  b3
            M_mn matrix elements (num_wann x num_wann)
    
    Parameters
    ----------
    filename : str
        Output filename (e.g., 'material.mmn')
    eigenvectors_list : list of ndarrays
        Eigenvectors C(k) for each k-point
    S_k_list : list of ndarrays
        Overlap matrices S(k) for each k-point
    neighbor_list : dict
        Maps k_idx -> list of (neighbor_idx, b_vector) tuples
    num_kpoints : int
        Total number of k-points
    num_wann : int
        Number of Wannier functions
    
    Notes
    -----
    The overlap matrix is computed as M(k,b) = C†(k) S(k+b) C(k+b),
    where b is the lattice vector connecting k to its neighbor.
    """
    # Each k-point typically has 6 neighbors (±x, ±y, ±z)
    num_neighbors = len(neighbor_list[0])
    
    with open(filename, 'w') as f:
        # Write header
        f.write("Created by LCAO-to-Wannier90 Engine\n")
        f.write(f"{num_wann:5d} {num_kpoints:5d} {num_neighbors:5d}\n")
        
        # Loop over all k-points
        for k_idx in range(num_kpoints):
            C_k = eigenvectors_list[k_idx]
            
            # Get neighbors of this k-point
            neighbors = neighbor_list[k_idx]
            
            for neighbor_idx, b in neighbors:
                C_k_plus_b = eigenvectors_list[neighbor_idx]
                S_k_plus_b = S_k_list[neighbor_idx]
                
                # Compute M(k,b) = C†(k) S(k+b) C(k+b)
                M_kb = C_k.conj().T @ S_k_plus_b @ C_k_plus_b
                
                # Write k-point indices and lattice vector b
                # Wannier90 uses 1-based indexing
                f.write(
                    f"{k_idx + 1:5d} {neighbor_idx + 1:5d} "
                    f"{b[0]:5d} {b[1]:5d} {b[2]:5d}\n"
                )
                
                # Write M_mn matrix elements
                for m in range(num_wann):
                    for n in range(num_wann):
                        f.write(
                            f"{M_kb[m, n].real:18.12f} "
                            f"{M_kb[m, n].imag:18.12f}\n"
                        )


def write_wannier90_files(
    seedname: str,
    eigenvalues_list: List[np.ndarray],
    eigenvectors_list: List[np.ndarray],
    S_k_list: List[np.ndarray],
    neighbor_list: Dict[int, List[Tuple[int, np.ndarray]]],
    num_kpoints: int,
    num_wann: int,
    verbose: bool = True
) -> None:
    """
    Write all three Wannier90 input files (.eig, .amn, .mmn).
    
    Parameters
    ----------
    seedname : str
        Prefix for output files
    eigenvalues_list : list of ndarrays
        Eigenvalues for each k-point
    eigenvectors_list : list of ndarrays
        Eigenvectors for each k-point
    S_k_list : list of ndarrays
        Overlap matrices for each k-point
    neighbor_list : dict
        Neighbor connectivity information
    num_kpoints : int
        Total number of k-points
    num_wann : int
        Number of Wannier functions
    verbose : bool
        Whether to print progress messages
    """
    if verbose:
        print(f"\nWriting Wannier90 files with seedname '{seedname}'...")
    
    # Write .eig file
    eig_file = f"{seedname}.eig"
    write_eig_file(eig_file, eigenvalues_list, num_kpoints, num_wann)
    if verbose:
        num_entries = num_kpoints * num_wann
        print(f"  ✓ {eig_file}: {num_entries} eigenvalues")
    
    # Write .amn file
    amn_file = f"{seedname}.amn"
    write_amn_file(amn_file, eigenvectors_list, S_k_list, num_kpoints, num_wann)
    if verbose:
        num_entries = num_kpoints * num_wann * num_wann
        print(f"  ✓ {amn_file}: {num_entries} matrix elements")
    
    # Write .mmn file
    mmn_file = f"{seedname}.mmn"
    write_mmn_file(
        mmn_file, eigenvectors_list, S_k_list, neighbor_list, num_kpoints, num_wann
    )
    if verbose:
        num_neighbors = len(neighbor_list[0])
        num_entries = num_kpoints * num_neighbors * num_wann * num_wann
        print(f"  ✓ {mmn_file}: {num_entries} matrix elements")
    
    if verbose:
        print(f"\nWannier90 files generated successfully!")
        print(f"  Files: {seedname}.eig, {seedname}.amn, {seedname}.mmn")