"""
Orbital Character Analysis Module

This module provides functions for computing band composition and orbital character
from eigenvalues and eigenvectors.

Functions:
- compute_band_projections: Compute projection weights for a band onto orbitals
- compute_band_character: Decompose band into element/orbital contributions
- analyze_all_bands_character: Compute character for multiple bands
- identify_dominant_character: Identify dominant orbital type in a band
- analyze_orbital_type_contributions: Per-type projectability weights near E_F
- compute_symmetry_aware_num_wann: Complete-shell num_wann from selected types
- build_orbital_mask: Boolean mask for constraining SCDM to selected shell types
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BandCharacter:
    """
    Band character decomposition by element and orbital type.

    Attributes
    ----------
    band_index : int
        Band index (0-indexed)
    characters : dict
        Nested dict: element → orbital_type → contribution (0.0 to 1.0)
    dominant : str
        Description of dominant character (e.g., "Bi-p" or "Mixed")
    """
    band_index: int
    characters: Dict[str, Dict[str, float]]
    dominant: str


def compute_band_projections(
    eigenvectors_list: List[np.ndarray],
    S_k_list: List[np.ndarray],
    band_index: int
) -> np.ndarray:
    """
    Compute projection weights for a single band onto each orbital.

    Projects the band eigenstates onto the LCAO basis using the overlap matrix:
        A_n(k) = [S(k)† @ C(k)]_n,band

    The returned weights are averaged over all k-points: <|A_n|²>_k

    Parameters
    ----------
    eigenvectors_list : list of ndarray
        Eigenvector matrices C(k) for each k-point, shape (num_orbitals, num_bands)
    S_k_list : list of ndarray
        Overlap matrices S(k) for each k-point, shape (num_orbitals, num_orbitals)
    band_index : int
        Band index to analyze (0-indexed)

    Returns
    -------
    ndarray
        Projection weights for each orbital, shape (num_orbitals,)
        Values sum to approximately 1.0 (normalized over k-points)

    Examples
    --------
    >>> projections = compute_band_projections(eigenvectors_list, S_k_list, band_idx=5)
    >>> most_contributing_orbital = np.argmax(projections)
    """
    num_kpoints = len(eigenvectors_list)
    num_orbitals = eigenvectors_list[0].shape[0]

    # Accumulate |A_n(k)|² over all k-points
    projection_weights = np.zeros(num_orbitals)

    for k_idx in range(num_kpoints):
        C_k = eigenvectors_list[k_idx]  # (num_orbitals, num_bands)
        S_k = S_k_list[k_idx]           # (num_orbitals, num_orbitals)

        # Project eigenvector onto LCAO basis
        # For non-orthogonal basis: A_k = S(k) @ C(k)
        # This gives the projection in the dual basis
        A_k = S_k @ C_k  # shape (num_orbitals, num_bands)

        # Get projection for this specific band
        A_n_k = A_k[:, band_index]  # (num_orbitals,)

        # Accumulate squared magnitude
        projection_weights += np.abs(A_n_k) ** 2

    # Average over k-points
    projection_weights /= num_kpoints

    # Normalize to sum to 1.0 (account for non-orthogonal basis)
    total_weight = np.sum(projection_weights)
    if total_weight > 1e-10:
        projection_weights /= total_weight

    return projection_weights


def compute_band_character(
    band_projections: np.ndarray,
    atom_symbols: List[str],
    basis_atom_map: np.ndarray,
    orbital_types: Dict[int, str]
) -> Dict[str, Dict[str, float]]:
    """
    Decompose band projection into element and orbital type contributions.

    Groups orbital projections by element (Bi, Se, etc.) and orbital angular momentum
    type (s, p, d, f, g).

    Parameters
    ----------
    band_projections : ndarray
        Projection weights for each orbital, shape (num_orbitals,)
    atom_symbols : list of str
        Element symbols for each atom (e.g., ['Bi', 'Bi'])
    basis_atom_map : ndarray
        Maps each orbital index to its parent atom, shape (num_orbitals,)
    orbital_types : dict
        Maps orbital index (1-indexed) → orbital type ('s', 'p', 'd', etc.)

    Returns
    -------
    dict
        Nested dictionary: element → orbital_type → contribution
        Example: {'Bi': {'s': 0.05, 'p': 0.75, 'd': 0.15}, 'Se': {'p': 0.05}}

    Examples
    --------
    >>> character = compute_band_character(projections, ['Bi', 'Bi'],
    ...                                     basis_map, orbital_types)
    >>> bi_p_contribution = character['Bi']['p']
    >>> print(f"Bi-p character: {bi_p_contribution*100:.1f}%")
    """
    characters = {}

    # Group by element and orbital type
    num_spatial_basis = len(basis_atom_map)

    for orbital_idx, weight in enumerate(band_projections):
        # Handle SOC doubling: basis_atom_map may be for spatial basis only
        # Map orbital index to spatial basis index
        spatial_idx = orbital_idx % num_spatial_basis

        # Get atom index (0-indexed in basis_atom_map)
        atom_idx = int(basis_atom_map[spatial_idx])
        element = atom_symbols[atom_idx]

        # Get orbital type (orbital_types uses 1-indexed keys)
        orb_type = orbital_types.get(orbital_idx + 1, 'unknown')

        # Initialize nested dict if needed
        if element not in characters:
            characters[element] = {}
        if orb_type not in characters[element]:
            characters[element][orb_type] = 0.0

        # Add contribution
        characters[element][orb_type] += weight

    return characters


def identify_dominant_character(
    band_character: Dict[str, Dict[str, float]],
    threshold: float = 0.3
) -> str:
    """
    Identify dominant orbital character for a band.

    Determines which element-orbital combination dominates the band character.
    If multiple types exceed the threshold, returns a combined description.

    Parameters
    ----------
    band_character : dict
        Nested dict from compute_band_character: element → orbital_type → contribution
    threshold : float
        Minimum contribution to be considered "dominant" (default: 0.3 = 30%)

    Returns
    -------
    str
        Description of dominant character:
        - "Bi-p (75%)" if single type dominates
        - "Bi-p + Se-p" if multiple types exceed threshold
        - "Mixed" if no type exceeds threshold

    Examples
    --------
    >>> dominant = identify_dominant_character({'Bi': {'p': 0.75, 's': 0.05}})
    >>> print(dominant)  # "Bi-p (75%)"
    """
    # Flatten to list of (element-type, contribution) pairs
    contributions = []
    for element, orb_dict in band_character.items():
        for orb_type, weight in orb_dict.items():
            contributions.append((f"{element}-{orb_type}", weight))

    # Sort by contribution (descending)
    contributions.sort(key=lambda x: x[1], reverse=True)

    if not contributions:
        return "Unknown"

    # Find all types above threshold
    dominant_types = [(name, weight) for name, weight in contributions if weight >= threshold]

    if not dominant_types:
        return "Mixed"
    elif len(dominant_types) == 1:
        name, weight = dominant_types[0]
        return f"{name} ({weight*100:.0f}%)"
    else:
        # Multiple dominant types
        names = [name for name, weight in dominant_types]
        return " + ".join(names)


def analyze_all_bands_character(
    eigenvectors_list: List[np.ndarray],
    S_k_list: List[np.ndarray],
    atom_symbols: List[str],
    basis_atom_map: np.ndarray,
    orbital_types: Dict[int, str],
    band_indices: Optional[np.ndarray] = None,
    threshold: float = 0.3,
    verbose: bool = False
) -> List[BandCharacter]:
    """
    Compute orbital character for multiple bands.

    Analyzes each band's projection onto different elements and orbital types.

    Parameters
    ----------
    eigenvectors_list : list of ndarray
        Eigenvector matrices for each k-point
    S_k_list : list of ndarray
        Overlap matrices for each k-point
    atom_symbols : list of str
        Element symbols
    basis_atom_map : ndarray
        Orbital to atom mapping
    orbital_types : dict
        Orbital type mapping
    band_indices : ndarray, optional
        Specific bands to analyze (default: all bands)
    threshold : float
        Threshold for dominant character identification
    verbose : bool
        Print progress information

    Returns
    -------
    list of BandCharacter
        Character analysis for each band

    Examples
    --------
    >>> results = analyze_all_bands_character(eigvecs, S_list, atoms,
    ...                                        basis_map, orb_types,
    ...                                        band_indices=np.arange(10, 20))
    >>> for band_char in results:
    ...     print(f"Band {band_char.band_index}: {band_char.dominant}")
    """
    num_bands = eigenvectors_list[0].shape[1]

    if band_indices is None:
        band_indices = np.arange(num_bands)

    results = []

    for band_idx in band_indices:
        if verbose and band_idx % 10 == 0:
            print(f"Analyzing band {band_idx+1}/{num_bands}...")

        # Compute projections
        projections = compute_band_projections(
            eigenvectors_list, S_k_list, band_idx
        )

        # Decompose into character
        character = compute_band_character(
            projections, atom_symbols, basis_atom_map, orbital_types
        )

        # Identify dominant type
        dominant = identify_dominant_character(character, threshold)

        results.append(BandCharacter(
            band_index=band_idx,
            characters=character,
            dominant=dominant
        ))

    return results


def format_band_character_table(
    band_characters: List[BandCharacter],
    elements: Optional[List[str]] = None,
    orbital_types: Optional[List[str]] = None
) -> str:
    """
    Format band character analysis as a readable table.

    Parameters
    ----------
    band_characters : list of BandCharacter
        Results from analyze_all_bands_character
    elements : list of str, optional
        Elements to include in table (default: auto-detect)
    orbital_types : list of str, optional
        Orbital types to show (default: ['s', 'p', 'd'])

    Returns
    -------
    str
        Formatted table string

    Examples
    --------
    >>> table = format_band_character_table(results, elements=['Bi'],
    ...                                      orbital_types=['s', 'p', 'd'])
    >>> print(table)
    """
    if not band_characters:
        return "No band character data available"

    # Auto-detect elements if not specified
    if elements is None:
        elements_set = set()
        for bc in band_characters:
            elements_set.update(bc.characters.keys())
        elements = sorted(elements_set)

    # Default orbital types
    if orbital_types is None:
        orbital_types = ['s', 'p', 'd', 'f']

    # Build header
    header = f"{'Band':<6} {'Dominant Character':<25}"
    for element in elements:
        for orb in orbital_types:
            header += f" {element}-{orb:<4}"

    lines = ["=" * len(header), header, "-" * len(header)]

    # Add data rows
    for bc in band_characters:
        row = f"{bc.band_index+1:<6} {bc.dominant:<25}"

        for element in elements:
            for orb in orbital_types:
                contrib = bc.characters.get(element, {}).get(orb, 0.0)
                if contrib > 0.01:  # Only show if > 1%
                    row += f" {contrib*100:>5.1f}%"
                else:
                    row += "      "

        lines.append(row)

    lines.append("=" * len(header))

    return "\n".join(lines)


# ============================================================================
# Symmetry-aware orbital selection helpers
# ============================================================================

# Mapping from orbital type to angular momentum dimension
_ORBITAL_DIM = {'s': 1, 'p': 3, 'd': 5, 'f': 7, 'g': 9}


def analyze_orbital_type_contributions(
    eigenvectors_list: List[np.ndarray],
    S_k_list: List[np.ndarray],
    eigenvalues_list: List[np.ndarray],
    orbital_types: Dict[int, str],
    e_fermi: float,
    energy_window: Tuple[float, float] = (-5.0, 3.0),
    has_soc: bool = False,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Compute k-averaged projectability weights grouped by orbital type near E_F.

    For each k-point, computes the Löwdin projection of occupied/near-Fermi
    bands onto the AO basis, then groups by orbital type (s, p, d, f).
    Returns a dict of type → total weight, normalized so that the sum = 1.

    Parameters
    ----------
    eigenvectors_list : list of ndarray
        Eigenvectors C(k) for each k-point, shape (num_orbitals, num_bands)
    S_k_list : list of ndarray
        Overlap matrices S(k) for each k-point, shape (num_orbitals, num_orbitals)
    eigenvalues_list : list of ndarray
        Eigenvalues for each k-point, shape (num_bands,)
    orbital_types : dict
        Maps orbital index (1-indexed) → orbital type ('s', 'p', 'd', etc.)
    e_fermi : float
        Fermi energy in eV
    energy_window : tuple of float
        (E_min, E_max) relative to E_fermi for selecting relevant bands
    has_soc : bool
        Whether spin-orbit coupling doubles the basis
    verbose : bool
        Print diagnostic information

    Returns
    -------
    dict
        Maps orbital type → total contribution weight (0.0 to 1.0).
        Example: {'s': 0.15, 'p': 0.72, 'd': 0.13}
    """
    num_kpoints = len(eigenvectors_list)
    num_orbitals = eigenvectors_list[0].shape[0]

    # Build orbital type array for all AO indices (0-indexed)
    orb_type_array = []
    for i in range(num_orbitals):
        # orbital_types uses 1-indexed keys
        orb_type_array.append(orbital_types.get(i + 1, 'unknown'))

    # Accumulate per-type weights
    type_weights = {}
    total_weight = 0.0

    win_min = e_fermi + energy_window[0]
    win_max = e_fermi + energy_window[1]

    for ik in range(num_kpoints):
        C_k = eigenvectors_list[ik]
        S_k = S_k_list[ik]
        evals = eigenvalues_list[ik]

        # Select bands within energy window
        in_window = (evals >= win_min) & (evals <= win_max)
        band_mask = np.where(in_window)[0]

        if len(band_mask) == 0:
            continue

        C_sel = C_k[:, band_mask]  # (num_orbitals, num_bands_in_window)

        # Compute S^{1/2}
        eigvals_s, eigvecs_s = np.linalg.eigh(S_k)
        eigvals_s = np.maximum(eigvals_s, 1e-12)
        S_sqrt = eigvecs_s @ np.diag(np.sqrt(eigvals_s)) @ eigvecs_s.conj().T

        # Löwdin-transformed density matrix: P_L(k) = S^{1/2} C C^dag S^{1/2}
        # We just need the diagonal: sum_n |[S^{1/2} C]_{mu,n}|^2
        SC = S_sqrt @ C_sel  # (num_orbitals, num_bands_in_window)
        diag_weights = np.sum(np.abs(SC) ** 2, axis=1)  # (num_orbitals,)

        # Group by orbital type
        for mu in range(num_orbitals):
            otype = orb_type_array[mu]
            if otype not in type_weights:
                type_weights[otype] = 0.0
            type_weights[otype] += diag_weights[mu]
            total_weight += diag_weights[mu]

    # Normalize
    if total_weight > 1e-10:
        for key in type_weights:
            type_weights[key] /= total_weight

    # Remove 'unknown' if negligible
    type_weights.pop('unknown', None)

    if verbose:
        print(f"  Orbital type contributions (E_F ± window [{energy_window[0]:.1f}, {energy_window[1]:.1f}] eV):")
        for otype in sorted(type_weights.keys()):
            print(f"    {otype}: {type_weights[otype]*100:.1f}%")

    return type_weights


def compute_symmetry_aware_num_wann(
    selected_types: List[str],
    orbital_structure: List[List[str]],
    has_soc: bool = False,
    verbose: bool = False,
) -> Tuple[int, List[List[str]]]:
    """
    Compute num_wann for complete orbital shells on all atoms.

    Given a set of orbital types to include (e.g., ['p'] or ['s', 'p']),
    computes the total Wannier function count that forms a complete
    representation under the crystal's space group.

    Parameters
    ----------
    selected_types : list of str
        Orbital types to include, e.g. ['p'], ['s', 'p'], ['s', 'p', 'd']
    orbital_structure : list of list of str
        Per-atom list of orbital shells from get_orbital_structure_from_crystal.
        E.g., [['s', 's', 'p', 'p', 'p', 'd', 'd', 'd'], ['s', 's', ...]]
    has_soc : bool
        Whether SOC doubles the basis (spinor_factor = 2 if True)
    verbose : bool
        Print diagnostic information

    Returns
    -------
    num_wann : int
        Total number of Wannier functions
    wannier_orbital_structure : list of list of str
        Per-atom list of unique included shell types (for WannSym compatibility).
        E.g., [['p'], ['p']] for 'p' selection on 2 atoms.
        Only one shell per angular momentum type is included (multi-zeta
        basis sets may have multiple radial functions of the same type,
        but WannSym needs exactly one complete angular momentum shell).
    """
    spinor_factor = 2 if has_soc else 1
    selected_set = set(t.lower() for t in selected_types)
    type_order = ['s', 'p', 'd', 'f', 'g']

    wannier_orbital_structure = []
    total_spatial_wann = 0

    for atom_idx, atom_shells in enumerate(orbital_structure):
        # Get UNIQUE orbital types present on this atom that match selection.
        # WannSym needs ONE complete shell per angular momentum type,
        # not all multi-zeta shells of that type.
        unique_types = []
        for t in type_order:
            if t in selected_set and t in [s.lower() for s in atom_shells]:
                unique_types.append(t)
        wannier_orbital_structure.append(unique_types)

        atom_dim = sum(_ORBITAL_DIM.get(t, 0) for t in unique_types)
        total_spatial_wann += atom_dim

    num_wann = total_spatial_wann * spinor_factor

    if verbose:
        print(f"  Symmetry-aware num_wann calculation:")
        print(f"    Selected orbital types: {selected_types}")
        print(f"    Spinor factor: {spinor_factor}")
        for i, shells in enumerate(wannier_orbital_structure):
            dim = sum(_ORBITAL_DIM.get(s.lower(), 0) for s in shells)
            print(f"    Atom {i}: shells={shells}, dim={dim}")
        print(f"    Total num_wann = {total_spatial_wann} × {spinor_factor} = {num_wann}")

    return num_wann, wannier_orbital_structure


def build_orbital_mask(
    orbital_types: Dict[int, str],
    selected_types: List[str],
    num_orbitals: int,
    has_soc: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """
    Build a boolean mask for SCDM orbital constraint.

    Creates a mask of shape (num_orbitals,) where True means the orbital
    is allowed for SCDM selection (it belongs to one of the selected types).

    Parameters
    ----------
    orbital_types : dict
        Maps orbital index (1-indexed) → orbital type ('s', 'p', 'd', etc.)
    selected_types : list of str
        Orbital types to include (e.g., ['p'], ['s', 'p'])
    num_orbitals : int
        Total number of orbitals (including SOC doubling if applicable)
    has_soc : bool
        Whether SOC doubles the basis
    verbose : bool
        Print diagnostic information

    Returns
    -------
    ndarray of bool
        Boolean mask, shape (num_orbitals,)

    Examples
    --------
    >>> mask = build_orbital_mask(orb_types, ['p'], 112, has_soc=True)
    >>> # For Bi bilayer: 6 p-orbitals per atom × 2 atoms × 2 spin = 24 True entries
    """
    selected_set = set(t.lower() for t in selected_types)
    mask = np.zeros(num_orbitals, dtype=bool)

    for i in range(num_orbitals):
        otype = orbital_types.get(i + 1, 'unknown')
        if otype.lower() in selected_set:
            mask[i] = True

    if verbose:
        n_allowed = np.sum(mask)
        print(f"  Orbital mask: {n_allowed}/{num_orbitals} orbitals selected")
        for t in sorted(selected_set):
            count = sum(1 for i in range(num_orbitals)
                        if orbital_types.get(i + 1, '').lower() == t)
            print(f"    {t}: {count} orbitals")

    return mask


def auto_select_orbital_types(
    type_contributions: Dict[str, float],
    threshold: float = 0.15,
    verbose: bool = False,
) -> List[str]:
    """
    Automatically select which orbital types to include based on contributions.

    Selects orbital types that have contributions above the threshold.
    Always includes at least the dominant type.

    Parameters
    ----------
    type_contributions : dict
        Maps orbital type → contribution weight from analyze_orbital_type_contributions
    threshold : float
        Minimum contribution to include a type (default: 0.15 = 15%)
    verbose : bool
        Print diagnostic information

    Returns
    -------
    list of str
        Selected orbital types in order: s, p, d, f
    """
    # Standard ordering
    type_order = ['s', 'p', 'd', 'f', 'g']

    # Select types above threshold
    selected = [t for t in type_order
                if type_contributions.get(t, 0.0) >= threshold]

    # If nothing selected, take the dominant type
    if not selected and type_contributions:
        dominant = max(type_contributions, key=type_contributions.get)
        selected = [dominant]

    if verbose:
        print(f"  Auto-detected orbital types (threshold={threshold*100:.0f}%): {selected}")
        for t in type_order:
            w = type_contributions.get(t, 0.0)
            marker = " ←" if t in selected else ""
            if w > 0.001:
                print(f"    {t}: {w*100:.1f}%{marker}")

    return selected
