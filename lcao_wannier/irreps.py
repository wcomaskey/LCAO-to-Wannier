"""
Irreducible Representation Analysis for Symmetry-Adapted Band Selection

Provides tools for:
1. Little group identification at high-symmetry k-points
2. Band character computation under symmetry operations
3. Irrep classification of Bloch states
4. Symmetry-adapted band selection as an alternative to projectability
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
import warnings


@dataclass
class IrrepResult:
    """Result of irrep analysis at a single k-point."""
    k_point: np.ndarray
    k_label: str
    little_group_size: int
    band_characters: np.ndarray      # (num_bands, num_ops) complex characters
    irrep_labels: List[str]          # irrep label per band (or per degenerate group)
    degenerate_groups: List[List[int]]  # groups of degenerate band indices


@dataclass
class BandSelectionResult:
    """Result of symmetry-adapted band selection."""
    selected_bands: np.ndarray       # indices of selected bands
    num_wann: int                    # recommended number of Wannier functions
    target_irreps: List[str]         # target irreps from orbital analysis
    irrep_analysis: List[IrrepResult]  # irrep results at high-symmetry k-points
    method: str = "symmetry"


# ============================================================================
# Little Group
# ============================================================================

def compute_little_group(
    k_point: np.ndarray,
    sym_operations: list,
    tolerance: float = 1e-6
) -> Tuple[List[int], List]:
    """
    Find the little group: symmetry operations that leave k invariant (mod G).

    Parameters
    ----------
    k_point : ndarray (3,)
        k-point in fractional coordinates
    sym_operations : list of SymmetryOperation
        All symmetry operations
    tolerance : float
        Tolerance for comparing k-vectors

    Returns
    -------
    indices : list of int
        Indices of operations in the little group
    operations : list of SymmetryOperation
        Little group operations
    """
    indices = []
    operations = []

    for i, op in enumerate(sym_operations):
        # k' = R^T @ k (rotation acts on k as transpose)
        k_rot = op.rotation_frac.T @ k_point
        diff = k_rot - k_point
        # k' equivalent to k if diff is integer vector
        diff_rounded = diff - np.round(diff)
        if np.linalg.norm(diff_rounded) < tolerance:
            indices.append(i)
            operations.append(op)

    return indices, operations


# ============================================================================
# Band Characters
# ============================================================================

def compute_band_characters(
    eigenvectors: np.ndarray,
    k_point: np.ndarray,
    little_group_indices: List[int],
    protmat_list: List[np.ndarray],
    S_k: Optional[np.ndarray] = None,
    sym_operations: Optional[list] = None
) -> np.ndarray:
    """
    Compute the character of each band under each little group operation.

    chi_n(g) = <psi_nk | P_g | psi_nk>

    For non-orthogonal basis (S != I), this becomes:
    chi_n(g) = C_n^dag @ S(k) @ P_g @ C_n

    where C_n is the eigenvector column.

    Parameters
    ----------
    eigenvectors : ndarray (norbs, nbands)
        Eigenvectors at this k-point (columns are bands)
    k_point : ndarray (3,)
        k-point in fractional coordinates
    little_group_indices : list of int
        Indices into protmat_list for the little group operations
    protmat_list : list of ndarray
        Full representation matrices for all symmetry operations
    S_k : ndarray (norbs, norbs), optional
        Overlap matrix at this k-point (identity if None)
    sym_operations : list of SymmetryOperation, optional
        Symmetry operations (needed for translation phase)

    Returns
    -------
    characters : ndarray (nbands, n_little_group) complex
        Character of each band under each little group operation
    """
    nbands = eigenvectors.shape[1]
    n_ops = len(little_group_indices)
    characters = np.zeros((nbands, n_ops), dtype=np.complex128)

    for op_idx, isym in enumerate(little_group_indices):
        P_g = protmat_list[isym]

        # Phase from fractional translation: exp(-i 2pi k . t_g)
        phase = 1.0
        if sym_operations is not None:
            t_g = sym_operations[isym].translation
            phase = np.exp(-2j * np.pi * np.dot(k_point, t_g))

        # M = C^dag @ S @ (phase * P_g) @ C
        if S_k is not None:
            M = eigenvectors.conj().T @ S_k @ (phase * P_g) @ eigenvectors
        else:
            M = eigenvectors.conj().T @ (phase * P_g) @ eigenvectors

        # Diagonal elements are the per-band characters
        characters[:, op_idx] = np.diag(M)

    return characters


def identify_degenerate_groups(
    eigenvalues: np.ndarray,
    tolerance: float = 1e-4
) -> List[List[int]]:
    """
    Identify groups of degenerate bands.

    Parameters
    ----------
    eigenvalues : ndarray (nbands,)
        Band energies at a k-point
    tolerance : float
        Energy tolerance for degeneracy

    Returns
    -------
    list of list of int
        Groups of degenerate band indices
    """
    groups = []
    visited = set()

    for i in range(len(eigenvalues)):
        if i in visited:
            continue
        group = [i]
        visited.add(i)
        for j in range(i + 1, len(eigenvalues)):
            if j in visited:
                continue
            if abs(eigenvalues[j] - eigenvalues[i]) < tolerance:
                group.append(j)
                visited.add(j)
        groups.append(group)

    return groups


def compute_group_characters(
    characters: np.ndarray,
    degenerate_groups: List[List[int]]
) -> List[np.ndarray]:
    """
    Compute the trace (character) for each degenerate group.

    For a degenerate group, the character is the sum of individual characters
    (trace of the representation in the degenerate subspace).
    """
    group_chars = []
    for group in degenerate_groups:
        chi = np.sum(characters[group, :], axis=0)
        group_chars.append(chi)
    return group_chars


# ============================================================================
# Irrep Identification
# ============================================================================

def decompose_representation(
    characters: np.ndarray,
    character_table: Dict[str, np.ndarray],
    group_order: int
) -> Dict[str, int]:
    """
    Decompose a representation into irreps using the character formula:
        n_mu = (1/|G|) sum_g chi(g)* chi_mu(g)

    Parameters
    ----------
    characters : ndarray (n_ops,)
        Characters of the representation to decompose
    character_table : dict
        {irrep_label: characters_array} for the group
    group_order : int
        Order of the group |G|

    Returns
    -------
    dict
        {irrep_label: multiplicity}
    """
    multiplicities = {}
    for label, chi_mu in character_table.items():
        # n_mu = (1/|G|) sum_g chi(g)* @ chi_mu(g)
        n_mu = np.sum(characters.conj() * chi_mu) / group_order
        n_mu_real = n_mu.real
        n_mu_int = int(round(n_mu_real))
        if abs(n_mu_real - n_mu_int) > 0.3:
            warnings.warn(
                f"Non-integer multiplicity {n_mu_real:.3f} for irrep {label}. "
                f"Characters may be inaccurate."
            )
        if n_mu_int > 0:
            multiplicities[label] = n_mu_int
    return multiplicities


def identify_band_irreps(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    k_point: np.ndarray,
    k_label: str,
    sym_info,
    protmat_list: List[np.ndarray],
    S_k: Optional[np.ndarray] = None,
    degeneracy_tol: float = 1e-4
) -> IrrepResult:
    """
    Identify the irrep of each band (or degenerate group) at a k-point.

    Parameters
    ----------
    eigenvalues : ndarray (nbands,)
        Eigenvalues at this k-point
    eigenvectors : ndarray (norbs, nbands)
        Eigenvectors at this k-point
    k_point : ndarray (3,)
        k-point in fractional coordinates
    k_label : str
        Label for the k-point (e.g., "Gamma", "M", "K")
    sym_info : SymmetryInfo
        Full symmetry information
    protmat_list : list of ndarray
        Representation matrices
    S_k : ndarray, optional
        Overlap matrix at this k-point
    degeneracy_tol : float
        Tolerance for identifying degenerate bands

    Returns
    -------
    IrrepResult
    """
    # Find little group
    lg_indices, lg_ops = compute_little_group(
        k_point, sym_info.operations
    )
    lg_size = len(lg_indices)

    # Compute characters
    characters = compute_band_characters(
        eigenvectors, k_point, lg_indices, protmat_list,
        S_k=S_k, sym_operations=sym_info.operations
    )

    # Identify degenerate groups
    deg_groups = identify_degenerate_groups(eigenvalues, degeneracy_tol)

    # Try to identify irreps using the little group character table
    # For now, assign labels based on dimension and character of identity
    irrep_labels = []
    for group in deg_groups:
        dim = len(group)
        chi_E = np.sum(characters[group, 0])  # Character under identity
        # Simple labeling by dimension
        if abs(chi_E.real - 1.0) < 0.3:
            irrep_labels.append(f"{k_label}_1D")
        elif abs(chi_E.real - 2.0) < 0.3:
            irrep_labels.append(f"{k_label}_2D")
        elif abs(chi_E.real - 3.0) < 0.3:
            irrep_labels.append(f"{k_label}_3D")
        elif abs(chi_E.real - 4.0) < 0.3:
            irrep_labels.append(f"{k_label}_4D")
        else:
            irrep_labels.append(f"{k_label}_{dim}D?")

    return IrrepResult(
        k_point=k_point,
        k_label=k_label,
        little_group_size=lg_size,
        band_characters=characters,
        irrep_labels=irrep_labels,
        degenerate_groups=deg_groups
    )


# ============================================================================
# High-Symmetry K-Point Detection
# ============================================================================

def find_high_symmetry_kpoints(
    kpoints: np.ndarray,
    sym_info,
    tolerance: float = 1e-6
) -> List[Tuple[int, np.ndarray, str]]:
    """
    Find k-points from the grid that are high-symmetry points.

    A high-symmetry k-point has a little group larger than the generic k-point.

    Parameters
    ----------
    kpoints : ndarray (nk, 3)
        All k-points in fractional coordinates
    sym_info : SymmetryInfo
        Symmetry information
    tolerance : float
        Tolerance for k-point comparisons

    Returns
    -------
    list of (index, k_point, label)
        High-symmetry k-points with their indices and labels
    """
    # Standard high-symmetry points to check
    special_points = {
        'Gamma': np.array([0.0, 0.0, 0.0]),
        'X': np.array([0.5, 0.0, 0.0]),
        'Y': np.array([0.0, 0.5, 0.0]),
        'Z': np.array([0.0, 0.0, 0.5]),
        'M': np.array([0.5, 0.5, 0.0]),
        'R': np.array([0.5, 0.5, 0.5]),
        'A': np.array([0.0, 0.0, 0.5]),
        'L': np.array([0.5, 0.0, 0.5]),
        'K': np.array([1.0/3.0, 1.0/3.0, 0.0]),
        'H': np.array([1.0/3.0, 1.0/3.0, 0.5]),
    }

    result = []
    for label, k_special in special_points.items():
        # Find the closest k-point in the grid
        for idx, k in enumerate(kpoints):
            diff = k - k_special
            diff -= np.round(diff)
            if np.linalg.norm(diff) < tolerance:
                result.append((idx, k, label))
                break

    # If no standard points found, use little group size criterion
    if not result:
        # Find the minimum little group size (generic k)
        min_lg_size = float('inf')
        lg_sizes = []
        for idx, k in enumerate(kpoints):
            lg_idx, _ = compute_little_group(k, sym_info.operations, tolerance)
            lg_sizes.append(len(lg_idx))
            min_lg_size = min(min_lg_size, len(lg_idx))

        # Any k-point with larger little group is "high symmetry"
        for idx, k in enumerate(kpoints):
            if lg_sizes[idx] > min_lg_size:
                result.append((idx, k, f"HS_{idx}"))

    return result


# ============================================================================
# Symmetry-Adapted Band Selection
# ============================================================================

def select_bands_by_symmetry(
    eigenvalues_list: List[np.ndarray],
    eigenvectors_list: List[np.ndarray],
    kpoints: np.ndarray,
    sym_info,
    protmat_list: List[np.ndarray],
    e_fermi: float,
    target_orbital_types: List[List[str]],
    has_soc: bool = False,
    S_k_list: Optional[List[np.ndarray]] = None,
    energy_padding: float = 2.0,
    verbose: bool = False
) -> BandSelectionResult:
    """
    Select bands using symmetry-adapted criteria.

    Strategy:
    1. Analyze irreps at high-symmetry k-points
    2. Determine target band count from orbital types
    3. Select bands near the Fermi level that form complete irrep manifolds
    4. Ensure band continuity across the BZ

    Parameters
    ----------
    eigenvalues_list : list of ndarray
        Eigenvalues at each k-point
    eigenvectors_list : list of ndarray
        Eigenvectors at each k-point
    kpoints : ndarray (nk, 3)
        K-points in fractional coordinates
    sym_info : SymmetryInfo
        Symmetry information
    protmat_list : list of ndarray
        Representation matrices
    e_fermi : float
        Fermi energy
    target_orbital_types : list of list of str
        Per-atom orbital types (determines target num_wann)
    has_soc : bool
        Whether SOC is present
    S_k_list : list of ndarray, optional
        Overlap matrices at each k-point
    energy_padding : float
        Energy padding around Fermi level for band selection (eV)
    verbose : bool
        Print analysis details

    Returns
    -------
    BandSelectionResult
    """
    nk = len(eigenvalues_list)
    nbands = len(eigenvalues_list[0])

    # Determine target number of Wannier functions from orbital types
    from .symmetry import _orbital_dim
    target_norbs = 0
    for atom_orbs in target_orbital_types:
        for orb_type in atom_orbs:
            target_norbs += _orbital_dim(orb_type)
    if has_soc:
        target_norbs *= 2

    if verbose:
        print(f"  Target num_wann from orbital types: {target_norbs}")

    # Find high-symmetry k-points
    hs_kpoints = find_high_symmetry_kpoints(kpoints, sym_info)

    if verbose:
        print(f"  Found {len(hs_kpoints)} high-symmetry k-points: "
              f"{[label for _, _, label in hs_kpoints]}")

    # Analyze irreps at high-symmetry points
    irrep_results = []
    for k_idx, k_point, k_label in hs_kpoints:
        S_k = S_k_list[k_idx] if S_k_list is not None else None
        result = identify_band_irreps(
            eigenvalues_list[k_idx],
            eigenvectors_list[k_idx],
            k_point, k_label,
            sym_info, protmat_list,
            S_k=S_k
        )
        irrep_results.append(result)

        if verbose:
            print(f"  {k_label}: little group size = {result.little_group_size}")
            for gi, group in enumerate(result.degenerate_groups[:10]):
                e_avg = np.mean(eigenvalues_list[k_idx][group])
                print(f"    Bands {group}: E={e_avg:.4f} eV, "
                      f"irrep={result.irrep_labels[gi]}")

    # Select bands: use energy window + degeneracy-respecting selection
    selected = _select_bands_near_fermi(
        eigenvalues_list, e_fermi, target_norbs,
        energy_padding, irrep_results, hs_kpoints, has_soc,
        verbose=verbose
    )

    return BandSelectionResult(
        selected_bands=selected,
        num_wann=len(selected),
        target_irreps=[],  # Will be populated when character tables are available
        irrep_analysis=irrep_results,
        method="symmetry"
    )


def _select_bands_near_fermi(
    eigenvalues_list: List[np.ndarray],
    e_fermi: float,
    target_num_wann: int,
    energy_padding: float,
    irrep_results: List[IrrepResult],
    hs_kpoints: List[Tuple[int, np.ndarray, str]],
    has_soc: bool,
    verbose: bool = False
) -> np.ndarray:
    """
    Select bands near the Fermi level respecting degeneracy at high-symmetry points.

    Algorithm:
    1. Compute average energy per band across all k-points
    2. Sort by distance from Fermi level
    3. Select target_num_wann bands, respecting degenerate grouping
    4. Ensure even number for SOC (Kramers pairs)
    """
    nk = len(eigenvalues_list)
    nbands = len(eigenvalues_list[0])

    # Average energy per band across k-points
    avg_energies = np.zeros(nbands)
    for eigs in eigenvalues_list:
        avg_energies += eigs
    avg_energies /= nk

    # Sort bands by distance from Fermi level
    distances = np.abs(avg_energies - e_fermi)
    sorted_indices = np.argsort(distances)

    # Build degenerate groups at high-symmetry points
    # A band should not be selected without its degenerate partners
    must_include = {}  # band_idx -> set of bands it's degenerate with

    for result in irrep_results:
        for group in result.degenerate_groups:
            if len(group) > 1:
                group_set = frozenset(group)
                for b in group:
                    if b not in must_include:
                        must_include[b] = set()
                    must_include[b].update(group)

    # Greedily select bands
    selected = set()
    for b in sorted_indices:
        if len(selected) >= target_num_wann:
            break

        # Add this band and all its degenerate partners
        to_add = {b}
        if b in must_include:
            to_add.update(must_include[b])

        # Check if adding this group would overshoot
        new_total = len(selected | to_add)
        if new_total <= target_num_wann + 4:  # Allow small overshoot for degeneracy
            selected.update(to_add)

    # For SOC: ensure even number (Kramers pairs)
    if has_soc and len(selected) % 2 != 0:
        # Add the next closest band
        for b in sorted_indices:
            if b not in selected:
                selected.add(b)
                break

    selected_array = np.sort(np.array(list(selected), dtype=int))

    if verbose:
        print(f"  Selected {len(selected_array)} bands (target: {target_num_wann})")
        e_min = min(eigenvalues_list[0][b] for b in selected_array)
        e_max = max(eigenvalues_list[0][b] for b in selected_array)
        print(f"  Energy range at Gamma: [{e_min:.4f}, {e_max:.4f}] eV")

    return selected_array
