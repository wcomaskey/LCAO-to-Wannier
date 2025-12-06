"""
Band Selection Module

This module provides intelligent band and orbital selection for LCAO-to-Wannier90
conversion. It includes:
- Energy window analysis
- Automatic Fermi level detection
- Automatic frozen window determination
- Intelligent projection orbital selection

For LCAO bases, disentanglement is typically unnecessary since the basis
functions are already localized atomic orbitals.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import warnings


@dataclass
class BandWindowResult:
    """Results from band window analysis."""
    frozen_indices: np.ndarray      # Band indices fully within window
    partial_indices: np.ndarray     # Band indices crossing window boundaries
    excluded_indices: np.ndarray    # Band indices outside window
    band_ranges: List[Tuple[float, float]]  # (E_min, E_max) for each band
    num_wann: int                   # Number of frozen bands
    frozen_energy_range: Tuple[float, float]  # Actual energy range of frozen bands
    e_fermi: float                  # Fermi energy used
    outer_window: Tuple[float, float]  # User-specified window (absolute)


@dataclass 
class OrbitalSelectionResult:
    """Results from automatic orbital selection."""
    selected_indices: np.ndarray    # Indices of selected projection orbitals
    orbital_weights: np.ndarray     # Weight of each orbital (contribution to bands)
    num_selected: int               # Number of selected orbitals


def estimate_fermi_energy(
    eigenvalues_list: List[np.ndarray],
    num_electrons: Optional[int] = None,
    num_orbitals: Optional[int] = None,
    method: str = 'auto'
) -> float:
    """
    Estimate the Fermi energy from eigenvalues.
    
    Parameters
    ----------
    eigenvalues_list : list of ndarrays
        Eigenvalues for each k-point, shape (num_bands,) each
    num_electrons : int, optional
        Number of electrons in the system. If provided, E_F is set
        to fill exactly this many states.
    num_orbitals : int, optional
        Number of spatial orbitals (without spin). Used for half-filling
        estimate if num_electrons not provided.
    method : str
        Method for Fermi level estimation:
        - 'auto': Use num_electrons if provided, else 'midgap'
        - 'electron_count': Fill states up to num_electrons
        - 'midgap': Find largest gap near center of spectrum
        - 'half_filling': Assume half the bands are occupied
        
    Returns
    -------
    float
        Estimated Fermi energy in same units as eigenvalues
        
    Notes
    -----
    For spin-orbit coupled systems with 2N×2N matrices, all bands
    are already spin-resolved, so num_electrons should be the total
    electron count (not divided by 2).
    """
    num_kpoints = len(eigenvalues_list)
    num_bands = len(eigenvalues_list[0])
    
    # Collect all eigenvalues
    all_eigenvalues = np.array(eigenvalues_list)  # Shape: (num_kpoints, num_bands)
    
    if method == 'auto':
        if num_electrons is not None:
            method = 'electron_count'
        else:
            method = 'midgap'
    
    if method == 'electron_count':
        if num_electrons is None:
            raise ValueError("num_electrons required for 'electron_count' method")
        
        # Sort all eigenvalues globally
        sorted_eigenvalues = np.sort(all_eigenvalues.flatten())
        
        # For a k-point grid, each state is weighted by 1/num_kpoints
        # Total states to fill = num_electrons * num_kpoints
        states_to_fill = num_electrons * num_kpoints
        
        if states_to_fill >= len(sorted_eigenvalues):
            warnings.warn("num_electrons exceeds available states, using highest energy")
            return sorted_eigenvalues[-1] + 0.1
        
        if states_to_fill <= 0:
            warnings.warn("num_electrons <= 0, using lowest energy")
            return sorted_eigenvalues[0] - 0.1
        
        # E_F is between the last occupied and first unoccupied state
        e_homo = sorted_eigenvalues[states_to_fill - 1]
        e_lumo = sorted_eigenvalues[states_to_fill]
        e_fermi = (e_homo + e_lumo) / 2
        
        return e_fermi
    
    elif method == 'half_filling':
        # Assume half the bands are occupied
        num_occupied = num_bands // 2
        
        # Average energy of HOMO and LUMO across k-points
        homo_energies = all_eigenvalues[:, num_occupied - 1]
        lumo_energies = all_eigenvalues[:, num_occupied]
        
        e_fermi = (homo_energies.mean() + lumo_energies.mean()) / 2
        return e_fermi
    
    elif method == 'midgap':
        # Find the largest gap in the band structure near the center
        # Average band energies across k-points
        avg_band_energies = np.mean(all_eigenvalues, axis=0)
        
        # Compute gaps between consecutive bands
        gaps = np.diff(avg_band_energies)
        
        # Weight gaps toward center of spectrum (prefer gaps near middle)
        center_idx = num_bands // 2
        weights = np.exp(-0.1 * (np.arange(len(gaps)) - center_idx)**2)
        weighted_gaps = gaps * weights
        
        # Find the largest weighted gap
        gap_idx = np.argmax(weighted_gaps)
        
        # Fermi level is in the middle of this gap
        e_fermi = (avg_band_energies[gap_idx] + avg_band_energies[gap_idx + 1]) / 2
        
        return e_fermi
    
    else:
        raise ValueError(f"Unknown method: {method}")


def analyze_band_window(
    eigenvalues_list: List[np.ndarray],
    outer_window: Tuple[float, float],
    e_fermi: Optional[float] = None,
    window_is_relative: bool = True,
    num_electrons: Optional[int] = None
) -> BandWindowResult:
    """
    Analyze bands relative to an energy window and identify frozen bands.
    
    Frozen bands are those whose ENTIRE dispersion (across all k-points)
    falls within the outer window. This guarantees a consistent band count
    at all k-points, which is required for Wannier90.
    
    Parameters
    ----------
    eigenvalues_list : list of ndarrays
        Eigenvalues for each k-point
    outer_window : tuple of float
        (E_min, E_max) energy window
    e_fermi : float, optional
        Fermi energy. If None and window_is_relative=True, will be auto-detected.
    window_is_relative : bool
        If True, outer_window is relative to E_F. If False, it's absolute.
    num_electrons : int, optional
        Number of electrons (for Fermi level auto-detection)
        
    Returns
    -------
    BandWindowResult
        Dataclass containing analysis results
    """
    num_kpoints = len(eigenvalues_list)
    num_bands = len(eigenvalues_list[0])
    
    # Auto-detect Fermi level if needed
    if e_fermi is None and window_is_relative:
        e_fermi = estimate_fermi_energy(
            eigenvalues_list, 
            num_electrons=num_electrons,
            method='auto'
        )
        print(f"  Auto-detected Fermi energy: {e_fermi:.4f} eV")
    elif e_fermi is None:
        e_fermi = 0.0
    
    # Convert window to absolute energies
    if window_is_relative:
        e_min = e_fermi + outer_window[0]
        e_max = e_fermi + outer_window[1]
    else:
        e_min, e_max = outer_window
    
    # Find energy range of each band across all k-points
    all_eigenvalues = np.array(eigenvalues_list)
    band_ranges = []
    for band_idx in range(num_bands):
        band_energies = all_eigenvalues[:, band_idx]
        band_ranges.append((band_energies.min(), band_energies.max()))
    
    # Classify bands
    frozen_indices = []
    partial_indices = []
    excluded_indices = []
    
    for band_idx, (b_min, b_max) in enumerate(band_ranges):
        if b_min >= e_min and b_max <= e_max:
            # Entire band within window - frozen
            frozen_indices.append(band_idx)
        elif b_max < e_min or b_min > e_max:
            # Entire band outside window - excluded
            excluded_indices.append(band_idx)
        else:
            # Band crosses window boundary - partial
            partial_indices.append(band_idx)
    
    frozen_indices = np.array(frozen_indices, dtype=int)
    partial_indices = np.array(partial_indices, dtype=int)
    excluded_indices = np.array(excluded_indices, dtype=int)
    
    # Determine actual energy range of frozen bands
    if len(frozen_indices) > 0:
        frozen_mins = [band_ranges[i][0] for i in frozen_indices]
        frozen_maxs = [band_ranges[i][1] for i in frozen_indices]
        frozen_energy_range = (min(frozen_mins), max(frozen_maxs))
    else:
        frozen_energy_range = (0.0, 0.0)
    
    return BandWindowResult(
        frozen_indices=frozen_indices,
        partial_indices=partial_indices,
        excluded_indices=excluded_indices,
        band_ranges=band_ranges,
        num_wann=len(frozen_indices),
        frozen_energy_range=frozen_energy_range,
        e_fermi=e_fermi,
        outer_window=(e_min, e_max)
    )


def print_band_analysis(result: BandWindowResult, verbose: bool = True) -> str:
    """
    Generate a human-readable report of band window analysis.
    
    Parameters
    ----------
    result : BandWindowResult
        Results from analyze_band_window()
    verbose : bool
        If True, print detailed per-band information
        
    Returns
    -------
    str
        Formatted analysis report
    """
    lines = []
    lines.append("=" * 70)
    lines.append("BAND WINDOW ANALYSIS")
    lines.append("=" * 70)
    lines.append(f"Fermi energy: {result.e_fermi:.4f} eV")
    lines.append(f"Outer window: [{result.outer_window[0]:.2f}, {result.outer_window[1]:.2f}] eV (absolute)")
    lines.append(f"              [{result.outer_window[0] - result.e_fermi:.2f}, "
                 f"{result.outer_window[1] - result.e_fermi:.2f}] eV (relative to E_F)")
    lines.append("")
    
    num_bands = len(result.band_ranges)
    
    # Group consecutive bands for cleaner output
    def group_consecutive(indices):
        """Group consecutive indices into ranges."""
        if len(indices) == 0:
            return []
        groups = []
        start = indices[0]
        end = indices[0]
        for idx in indices[1:]:
            if idx == end + 1:
                end = idx
            else:
                groups.append((start, end))
                start = idx
                end = idx
        groups.append((start, end))
        return groups
    
    lines.append("Band classification:")
    
    # Excluded below
    excluded_below = [i for i in result.excluded_indices 
                      if result.band_ranges[i][1] < result.outer_window[0]]
    if excluded_below:
        groups = group_consecutive(excluded_below)
        for start, end in groups:
            e_min = min(result.band_ranges[i][0] for i in range(start, end + 1))
            e_max = max(result.band_ranges[i][1] for i in range(start, end + 1))
            if start == end:
                lines.append(f"  Band {start}:      E = [{e_min:.2f}, {e_max:.2f}] eV  → EXCLUDED (below window)")
            else:
                lines.append(f"  Bands {start}-{end}:   E = [{e_min:.2f}, {e_max:.2f}] eV  → EXCLUDED (below window)")
    
    # Partial (lower boundary)
    partial_lower = [i for i in result.partial_indices 
                     if result.band_ranges[i][0] < result.outer_window[0]]
    if partial_lower:
        groups = group_consecutive(partial_lower)
        for start, end in groups:
            e_min = min(result.band_ranges[i][0] for i in range(start, end + 1))
            e_max = max(result.band_ranges[i][1] for i in range(start, end + 1))
            if start == end:
                lines.append(f"  Band {start}:      E = [{e_min:.2f}, {e_max:.2f}] eV  → PARTIAL (crosses lower boundary)")
            else:
                lines.append(f"  Bands {start}-{end}:   E = [{e_min:.2f}, {e_max:.2f}] eV  → PARTIAL (crosses lower boundary)")
    
    # Frozen
    if len(result.frozen_indices) > 0:
        groups = group_consecutive(list(result.frozen_indices))
        for start, end in groups:
            e_min = min(result.band_ranges[i][0] for i in range(start, end + 1))
            e_max = max(result.band_ranges[i][1] for i in range(start, end + 1))
            if start == end:
                lines.append(f"  Band {start}:      E = [{e_min:.2f}, {e_max:.2f}] eV  → FROZEN ✓")
            else:
                lines.append(f"  Bands {start}-{end}:  E = [{e_min:.2f}, {e_max:.2f}] eV  → FROZEN ✓")
    
    # Partial (upper boundary)
    partial_upper = [i for i in result.partial_indices 
                     if result.band_ranges[i][1] > result.outer_window[1]]
    if partial_upper:
        groups = group_consecutive(partial_upper)
        for start, end in groups:
            e_min = min(result.band_ranges[i][0] for i in range(start, end + 1))
            e_max = max(result.band_ranges[i][1] for i in range(start, end + 1))
            if start == end:
                lines.append(f"  Band {start}:      E = [{e_min:.2f}, {e_max:.2f}] eV  → PARTIAL (crosses upper boundary)")
            else:
                lines.append(f"  Bands {start}-{end}:   E = [{e_min:.2f}, {e_max:.2f}] eV  → PARTIAL (crosses upper boundary)")
    
    # Excluded above
    excluded_above = [i for i in result.excluded_indices 
                      if result.band_ranges[i][0] > result.outer_window[1]]
    if excluded_above:
        groups = group_consecutive(excluded_above)
        for start, end in groups:
            e_min = min(result.band_ranges[i][0] for i in range(start, end + 1))
            e_max = max(result.band_ranges[i][1] for i in range(start, end + 1))
            if start == end:
                lines.append(f"  Band {start}:      E = [{e_min:.2f}, {e_max:.2f}] eV  → EXCLUDED (above window)")
            else:
                lines.append(f"  Bands {start}-{end}:  E = [{e_min:.2f}, {e_max:.2f}] eV  → EXCLUDED (above window)")
    
    lines.append("")
    lines.append("-" * 70)
    lines.append("RESULT:")
    lines.append(f"  Frozen bands: {result.num_wann} (indices {result.frozen_indices[0]}-{result.frozen_indices[-1]})" 
                 if result.num_wann > 0 else "  Frozen bands: 0 (WARNING: no bands fully within window!)")
    lines.append(f"  → Setting num_wann = {result.num_wann}")
    
    if result.num_wann > 0:
        lines.append("")
        lines.append(f"Exact frozen window: [{result.frozen_energy_range[0]:.4f}, "
                     f"{result.frozen_energy_range[1]:.4f}] eV")
    
    # Suggestions for partial bands
    if len(result.partial_indices) > 0:
        lines.append("")
        lines.append("SUGGESTIONS for partial bands:")
        for idx in result.partial_indices:
            b_min, b_max = result.band_ranges[idx]
            if b_min < result.outer_window[0]:
                lines.append(f"  Band {idx}: To include, expand lower bound to {b_min:.2f} eV")
            if b_max > result.outer_window[1]:
                lines.append(f"  Band {idx}: To include, expand upper bound to {b_max:.2f} eV")
    
    lines.append("=" * 70)
    
    report = "\n".join(lines)
    print(report)
    return report


def check_frozen_continuity(frozen_indices: np.ndarray) -> Tuple[bool, List[int]]:
    """
    Check if frozen bands form a contiguous set.
    
    Non-contiguous frozen bands may indicate unusual band structure
    or an inappropriately chosen energy window.
    
    Parameters
    ----------
    frozen_indices : ndarray
        Indices of frozen bands
        
    Returns
    -------
    is_contiguous : bool
        True if bands are contiguous
    gap_positions : list of int
        Indices where gaps occur in the frozen set
    """
    if len(frozen_indices) <= 1:
        return True, []
    
    sorted_indices = np.sort(frozen_indices)
    gaps = np.diff(sorted_indices)
    gap_positions = np.where(gaps > 1)[0].tolist()
    
    is_contiguous = len(gap_positions) == 0
    
    if not is_contiguous:
        warnings.warn(
            f"Frozen bands are non-contiguous! Gaps at indices: "
            f"{[sorted_indices[i] for i in gap_positions]}. "
            "This may indicate an unusual band structure."
        )
    
    return is_contiguous, gap_positions


def validate_fermi_coverage(
    frozen_indices: np.ndarray,
    eigenvalues_list: List[np.ndarray],
    e_fermi: float
) -> Dict[str, any]:
    """
    Validate that frozen window properly covers the Fermi level.
    
    For meaningful Wannier functions, the frozen window should typically
    include both occupied and unoccupied states near E_F.
    
    Parameters
    ----------
    frozen_indices : ndarray
        Indices of frozen bands
    eigenvalues_list : list of ndarrays
        Eigenvalues for each k-point
    e_fermi : float
        Fermi energy
        
    Returns
    -------
    dict
        Validation results with keys:
        - 'has_occupied': bool, whether frozen set includes occupied bands
        - 'has_unoccupied': bool, whether frozen set includes unoccupied bands
        - 'num_occupied': int, number of frozen bands below E_F
        - 'num_unoccupied': int, number of frozen bands above E_F
        - 'crosses_fermi': bool, whether any band crosses E_F
    """
    if len(frozen_indices) == 0:
        return {
            'has_occupied': False,
            'has_unoccupied': False,
            'num_occupied': 0,
            'num_unoccupied': 0,
            'crosses_fermi': False
        }
    
    all_eigenvalues = np.array(eigenvalues_list)
    
    num_occupied = 0
    num_unoccupied = 0
    crosses_fermi = False
    
    for band_idx in frozen_indices:
        band_energies = all_eigenvalues[:, band_idx]
        band_min = band_energies.min()
        band_max = band_energies.max()
        
        if band_max < e_fermi:
            num_occupied += 1
        elif band_min > e_fermi:
            num_unoccupied += 1
        else:
            # Band crosses Fermi level
            crosses_fermi = True
            # Count as both for simplicity
            num_occupied += 0.5
            num_unoccupied += 0.5
    
    return {
        'has_occupied': num_occupied > 0,
        'has_unoccupied': num_unoccupied > 0,
        'num_occupied': int(num_occupied),
        'num_unoccupied': int(num_unoccupied),
        'crosses_fermi': crosses_fermi
    }


def select_projection_orbitals(
    eigenvectors_list: List[np.ndarray],
    S_k_list: List[np.ndarray],
    num_wann: int,
    band_indices: Optional[np.ndarray] = None
) -> OrbitalSelectionResult:
    """
    Automatically select LCAO orbitals for projection based on their
    contribution to the selected bands.
    
    For LCAO bases, this provides an optimal set of projection orbitals
    without requiring user-specified projections (like Bi:s,p in wien2wannier).
    
    Parameters
    ----------
    eigenvectors_list : list of ndarrays
        Eigenvectors C(k) for each k-point, shape (num_orbitals, num_bands)
    S_k_list : list of ndarrays
        Overlap matrices S(k) for each k-point
    num_wann : int
        Number of Wannier functions (orbitals to select)
    band_indices : ndarray, optional
        If provided, only consider contributions from these bands.
        If None, use all bands in eigenvectors_list.
        
    Returns
    -------
    OrbitalSelectionResult
        Contains selected orbital indices, weights, and count
        
    Notes
    -----
    The algorithm computes the projection of bands onto each LCAO orbital:
        A_nk = S(k)† C(k)
    Then sums |A_nk|² over all selected bands and k-points for each orbital n.
    The orbitals with largest total weight are selected.
    """
    num_kpoints = len(eigenvectors_list)
    num_orbitals = eigenvectors_list[0].shape[0]
    
    if band_indices is None:
        num_bands = eigenvectors_list[0].shape[1]
        band_indices = np.arange(num_bands)
    
    orbital_weights = np.zeros(num_orbitals)
    
    for k_idx in range(num_kpoints):
        C_k = eigenvectors_list[k_idx][:, band_indices]  # Select only specified bands
        S_k = S_k_list[k_idx]
        
        # A_k = S† C gives projection of bands onto orbitals
        # Shape: (num_orbitals, num_selected_bands)
        A_k = S_k.conj().T @ C_k
        
        # Sum squared magnitude of projections for each orbital
        orbital_weights += np.sum(np.abs(A_k)**2, axis=1)
    
    # Normalize by number of k-points and bands
    orbital_weights /= (num_kpoints * len(band_indices))
    
    # Select orbitals with largest total contribution
    selected_indices = np.argsort(orbital_weights)[-num_wann:]
    selected_indices = np.sort(selected_indices)  # Sort in ascending order
    
    return OrbitalSelectionResult(
        selected_indices=selected_indices,
        orbital_weights=orbital_weights,
        num_selected=num_wann
    )


def compute_subspace_projections(
    eigenvectors_list: List[np.ndarray],
    S_k_list: List[np.ndarray],
    band_indices: np.ndarray,
    orbital_indices: np.ndarray
) -> List[np.ndarray]:
    """
    Compute projection matrices for a selected subspace.
    
    This creates the A(k) matrices for Wannier90 when using a subset
    of bands and/or orbitals.
    
    Parameters
    ----------
    eigenvectors_list : list of ndarrays
        Full eigenvectors C(k) for each k-point
    S_k_list : list of ndarrays
        Overlap matrices S(k) for each k-point
    band_indices : ndarray
        Indices of bands to include
    orbital_indices : ndarray
        Indices of LCAO orbitals to project onto
        
    Returns
    -------
    list of ndarrays
        Projection matrices A(k) for each k-point
        Shape: (num_selected_bands, num_selected_orbitals)
    """
    A_k_list = []
    
    for k_idx in range(len(eigenvectors_list)):
        C_k_full = eigenvectors_list[k_idx]
        S_k_full = S_k_list[k_idx]
        
        # Select bands
        C_k = C_k_full[:, band_indices]
        
        # Compute full projection
        A_k_full = S_k_full.conj().T @ C_k  # Shape: (num_orbitals, num_bands)
        
        # Select orbitals (rows)
        A_k = A_k_full[orbital_indices, :]  # Shape: (num_wann, num_bands)
        
        # Transpose to match Wannier90 convention: (num_bands, num_wann)
        A_k_list.append(A_k.T)
    
    return A_k_list


def suggest_optimal_window(
    eigenvalues_list: List[np.ndarray],
    e_fermi: float,
    target_num_wann: Optional[int] = None,
    padding: float = 0.1
) -> Tuple[float, float]:
    """
    Suggest an optimal energy window based on band structure.
    
    Parameters
    ----------
    eigenvalues_list : list of ndarrays
        Eigenvalues for each k-point
    e_fermi : float
        Fermi energy
    target_num_wann : int, optional
        Target number of Wannier functions. If provided, tries to find
        a window containing approximately this many bands.
    padding : float
        Extra energy padding (eV) around the tight band edges
        
    Returns
    -------
    tuple of float
        (E_min, E_max) suggested window in absolute energies
    """
    all_eigenvalues = np.array(eigenvalues_list)
    num_bands = all_eigenvalues.shape[1]
    
    # Compute band ranges
    band_mins = np.min(all_eigenvalues, axis=0)
    band_maxs = np.max(all_eigenvalues, axis=0)
    
    # Find gaps between bands
    avg_energies = np.mean(all_eigenvalues, axis=0)
    gaps = np.diff(avg_energies)
    
    if target_num_wann is None:
        # Find bands around Fermi level with significant gaps above/below
        center_bands = np.where(
            (band_mins < e_fermi + 2.0) & (band_maxs > e_fermi - 2.0)
        )[0]
        
        if len(center_bands) > 0:
            first_band = center_bands[0]
            last_band = center_bands[-1]
            
            # Expand to nearest significant gaps
            gap_threshold = np.percentile(gaps, 75)  # Use 75th percentile as threshold
            
            # Search for gap below
            for i in range(first_band - 1, -1, -1):
                if gaps[i] > gap_threshold:
                    first_band = i + 1
                    break
            else:
                first_band = 0
            
            # Search for gap above
            for i in range(last_band, len(gaps)):
                if gaps[i] > gap_threshold:
                    last_band = i
                    break
            else:
                last_band = num_bands - 1
            
            e_min = band_mins[first_band] - padding
            e_max = band_maxs[last_band] + padding
        else:
            # Fallback: center ±5 eV around Fermi level
            e_min = e_fermi - 5.0
            e_max = e_fermi + 5.0
    else:
        # Find window containing target_num_wann bands centered on E_F
        # Start from Fermi level and expand symmetrically
        
        # Find band closest to Fermi level
        distances_to_fermi = np.abs(avg_energies - e_fermi)
        center_band = np.argmin(distances_to_fermi)
        
        # Expand symmetrically
        half_bands = target_num_wann // 2
        first_band = max(0, center_band - half_bands)
        last_band = min(num_bands - 1, center_band + half_bands)
        
        # Adjust if we hit boundaries
        actual_count = last_band - first_band + 1
        if actual_count < target_num_wann:
            if first_band == 0:
                last_band = min(num_bands - 1, first_band + target_num_wann - 1)
            elif last_band == num_bands - 1:
                first_band = max(0, last_band - target_num_wann + 1)
        
        e_min = band_mins[first_band] - padding
        e_max = band_maxs[last_band] + padding
    
    return (e_min, e_max)