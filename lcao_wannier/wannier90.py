"""
Wannier90 File Writer Module

This module contains functions for writing Wannier90 input files
(.eig, .amn, .mmn) according to the Wannier90 file format specification.
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional

from .fourier import fourier_transform_to_kspace, fourier_transform_vectorized, StackedMatrices


def diagnose_mmn_matrix(
    M: np.ndarray,
    k_idx: int,
    next_k_idx: int,
    G_shift: Tuple[int, int, int],
    tolerance: float = 1e-6
) -> Dict[str, Any]:
    """
    Diagnose properties of an MMN matrix for debugging.

    Parameters
    ----------
    M : ndarray
        The MMN matrix, shape (num_bands, num_bands)
    k_idx : int
        Current k-point index
    next_k_idx : int
        Neighbor k-point index
    G_shift : tuple
        The G-vector shift (n1, n2, n3)
    tolerance : float
        Tolerance for numerical checks

    Returns
    -------
    diagnostics : dict
        Dictionary with diagnostic information:
        - 'trace': Trace of M (should be real and close to num_bands for identity)
        - 'det': Determinant of M (magnitude should be close to 1 for unitary)
        - 'is_identity': True if M is close to identity (for self-overlap)
        - 'singular_values': Singular values (should all be close to 1 for unitary)
        - 'max_off_diag': Maximum off-diagonal element magnitude
        - 'hermitian_error': ||M†M - I|| / num_bands
    """
    num_bands = M.shape[0]

    trace = np.trace(M)
    det = np.linalg.det(M)

    # Singular value decomposition
    U, s, Vh = np.linalg.svd(M)

    # Check if M is identity (for self-overlap with G=0)
    identity_error = np.linalg.norm(M - np.eye(num_bands)) / num_bands
    is_identity = identity_error < tolerance

    # Check off-diagonal elements
    diag_mask = np.eye(num_bands, dtype=bool)
    off_diag = np.abs(M[~diag_mask])
    max_off_diag = np.max(off_diag) if len(off_diag) > 0 else 0.0

    # Check unitarity: M†M should be identity
    MdagM = M.conj().T @ M
    hermitian_error = np.linalg.norm(MdagM - np.eye(num_bands)) / num_bands

    return {
        'k_idx': k_idx,
        'next_k_idx': next_k_idx,
        'G_shift': G_shift,
        'trace': trace,
        'det': det,
        'det_magnitude': np.abs(det),
        'is_identity': is_identity,
        'identity_error': identity_error,
        'singular_values': s,
        'sv_min': np.min(s),
        'sv_max': np.max(s),
        'max_off_diag': max_off_diag,
        'hermitian_error': hermitian_error,
        'is_unitary': hermitian_error < tolerance and np.abs(np.abs(det) - 1) < tolerance
    }


def compute_mmn_direct(
    k_idx: int,
    next_k_idx: int,
    kpoints: np.ndarray,
    b_vector_cart: np.ndarray,
    real_space_matrices: Dict[Tuple[int, int, int], Dict[str, np.ndarray]],
    lattice_vectors: np.ndarray,
    atom_positions: np.ndarray,
    basis_atom_map: np.ndarray,
    eigenvectors: List[np.ndarray],
    symmetrize: bool = True,
    G_shift: np.ndarray = None,
    stacked: 'StackedMatrices' = None
) -> np.ndarray:
    """
    Compute M_mn using the Symmetric Midpoint Approximation.

    This implements the correct MMN matrix for LCAO methods using the
    2π phase convention, which ensures proper BZ periodicity: S(k+G) = S(k).

    The method evaluates S at the physical midpoint between k and k+b:
    M_mn = C_m^†(k) · S(k + b/2) · C_n(k+b)

    Parameters
    ----------
    k_idx : int
        Index of the current k-point
    next_k_idx : int
        Index of the neighbor k-point
    kpoints : ndarray of shape (num_kpoints, 3)
        K-points in fractional coordinates
    b_vector_cart : ndarray of shape (3,)
        Cartesian b-vector connecting k to k+b (unused, kept for API)
    real_space_matrices : dict
        Dict mapping (n1,n2,n3) -> {'H': H_matrix, 'S': S_matrix}
    lattice_vectors : ndarray of shape (3, 3)
        Real-space lattice vectors (rows are [a1, a2, a3])
    atom_positions : ndarray of shape (num_atoms, 3)
        Atomic positions (unused, kept for API compatibility)
    basis_atom_map : ndarray of shape (num_orbitals,)
        Maps each orbital to its atom index (unused, kept for API)
    eigenvectors : list of ndarray
        Eigenvector matrices C(k), shape (num_orbitals, num_bands)
    symmetrize : bool, optional
        Not used in this implementation.
    G_shift : ndarray of shape (3,), optional
        Integer G-vector shift from .nnkp file

    Returns
    -------
    M_mn : ndarray
        MMN overlap matrix, shape (num_bands, num_bands)
    """
    k_curr = kpoints[k_idx]
    k_next = kpoints[next_k_idx]

    # Use G_shift from nnkp file (passed as parameter)
    # This tells us the BZ wrapping: k' + G = k + b (physical path)
    if G_shift is None:
        G_shift = np.zeros(3, dtype=int)
    G_shift = np.asarray(G_shift)

    # Calculate PHYSICAL b-vector in fractional coordinates
    # b = k_next + G_shift - k_curr (this is the actual displacement)
    b_frac = k_next + G_shift - k_curr

    # Calculate PHYSICAL Midpoint
    # k_mid is the actual location in reciprocal space where overlap occurs
    k_mid = k_curr + 0.5 * b_frac

    # Debug for BZ boundary cases
    debug_this = (k_idx == 0 and np.any(G_shift != 0))
    if debug_this:
        print(f"\n=== DEBUG MMN for k_idx=0, BZ boundary crossing ===")
        print(f"  k_curr = {k_curr}")
        print(f"  k_next = {k_next}")
        print(f"  G_shift (from nnkp) = {G_shift}")
        print(f"  b_frac = k_next + G_shift - k_curr = {b_frac}")
        print(f"  k_mid = k_curr + 0.5*b_frac = {k_mid}")

    # Compute S at the midpoint
    # With 2π convention, S(k_mid) = S(k_mid + G) for any G, so no correction needed
    if stacked is not None:
        _, S_mid = fourier_transform_vectorized(k_mid, stacked)
    else:
        _, S_mid = fourier_transform_to_kspace(k_mid, real_space_matrices, lattice_vectors)

    # Project onto eigenvectors
    C_k = eigenvectors[k_idx]
    C_next = eigenvectors[next_k_idx]

    M_mn = C_k.conj().T @ S_mid @ C_next

    # Debug: print |det(M)| for boundary crossings
    if debug_this:
        det_M = np.linalg.det(M_mn)
        print(f"  |det(M)| = {np.abs(det_M):.6f}")
        U, s, Vh = np.linalg.svd(M_mn)
        print(f"  Singular values: min={s.min():.4f}, max={s.max():.4f}")

    return M_mn


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

    DEPRECATED: Use compute_mmn_direct() instead, which uses the correct
    real-space formula consistent with the π convention.

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
    convention: str = 'pi',
    use_direct_method: bool = True,
    verbose: bool = False,
    stacked: 'StackedMatrices' = None
) -> None:
    """
    Write the .mmn file for LCAO methods.

    This implements the MMN matrix calculation for LCAO methods with two
    available algorithms:

    1. Direct method (use_direct_method=True, default):
       Uses direct real-space summation with explicit phase factors.
       Formula: M_mn = Σ_R exp(i*π*b·R) * exp(-i*b·(τ_j-τ_i)) * C†(k) S(R) C(k+b)

    2. Symmetric midpoint method (use_direct_method=False):
       Evaluates S at midpoint k + 0.5*b with Berry phase correction.
       Formula: M = C†(k) @ S(k_mid) @ phase_matrix @ C(k+b)

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
        Default is 'pi'
    use_direct_method : bool, optional
        If True, use the direct real-space method (default).
        If False, use the symmetric midpoint method.
    verbose : bool, optional
        If True, print diagnostic information (default: False)
    """
    method_name = "Direct Real-Space" if use_direct_method else "Symmetric Midpoint"

    with open(filename, 'w') as f:
        # Write header
        f.write(f"Created by LCAO-to-Wannier90 ({method_name} Method)\n")
        num_neighbors = len(neighbor_list[0])
        f.write(f"{num_wann:5d} {num_kpoints:5d} {num_neighbors:5d}\n")

        # Diagnostic tracking
        if verbose:
            diag_traces = []  # Track trace of M matrices (should be close to num_wann for identity)

        # Process each k-point
        for k_idx in range(num_kpoints):
            # Process each neighbor
            for neighbor in neighbor_list[k_idx]:
                k_next_idx = neighbor['id']
                G_shift = neighbor['G_shift']
                b_vec_cart = neighbor['b_vec_cart']  # Cartesian b-vector

                # Compute MMN matrix using selected method
                if use_direct_method:
                    M_kb = compute_mmn_direct(
                        k_idx, k_next_idx, kpoints, b_vec_cart,
                        real_space_matrices, lattice_vectors,
                        atom_positions, basis_atom_map, eigenvectors_list,
                        symmetrize=True,
                        G_shift=G_shift,
                        stacked=stacked
                    )
                else:
                    M_kb = compute_mmn_matrix(
                        k_idx, k_next_idx, kpoints, b_vec_cart,
                        real_space_matrices, lattice_vectors,
                        atom_positions, basis_atom_map, eigenvectors_list
                    )

                # Diagnostic: check if this is a self-overlap (k_idx == k_next_idx with G=0)
                if verbose and k_idx == k_next_idx and tuple(G_shift) == (0, 0, 0):
                    trace = np.trace(M_kb)
                    diag_traces.append((k_idx, trace))

                # --- WRITE TO FILE ---
                # Write neighbor identification line
                f.write(f"{k_idx+1:5d} {k_next_idx+1:5d} "
                       f"{G_shift[0]:5d} {G_shift[1]:5d} {G_shift[2]:5d}\n")

                # Write matrix elements (column-major order for Fortran compatibility)
                for n in range(num_wann):
                    for m in range(num_wann):
                        val = M_kb[m, n]
                        f.write(f"{val.real:18.12f} {val.imag:18.12f}\n")

        if verbose and diag_traces:
            print(f"\n  MMN Diagnostics ({method_name} method):")
            print(f"  Self-overlap traces (should be ≈ {num_wann}):")
            for k_idx, trace in diag_traces:
                print(f"    k={k_idx}: Tr(M) = {trace.real:.6f} + {trace.imag:.6f}i")


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