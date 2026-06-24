"""
Tests for Band Selection Module

Tests cover:
- Fermi energy estimation (multiple methods)
- Band window analysis
- Frozen band identification
- Orbital selection for projections
- Edge cases and error handling
"""

import numpy as np
import pytest
from typing import List, Tuple

# Import the module under test
from lcao_wannier.band_selection import (
    estimate_fermi_energy,
    compute_fermi_level,
    analyze_band_window,
    print_band_analysis,
    check_frozen_continuity,
    validate_fermi_coverage,
    select_projection_orbitals,
    compute_subspace_projections,
    suggest_optimal_window,
    BandWindowResult,
    OrbitalSelectionResult
)


# ==============================================================================
# Test Fixtures - Synthetic Band Structures
# ==============================================================================

def create_simple_bandstructure(
    num_kpoints: int = 10,
    num_bands: int = 20,
    band_gap: float = 1.0,
    bandwidth: float = 4.0,
    e_valence_top: float = 0.0
) -> List[np.ndarray]:
    """
    Create a simple semiconductor band structure with a gap.
    
    Parameters
    ----------
    num_kpoints : int
        Number of k-points
    num_bands : int  
        Total number of bands
    band_gap : float
        Band gap in eV
    bandwidth : float
        Width of valence/conduction bands
    e_valence_top : float
        Energy of valence band maximum
        
    Returns
    -------
    list of ndarrays
        Eigenvalues for each k-point
    """
    eigenvalues_list = []
    num_valence = num_bands // 2
    num_conduction = num_bands - num_valence
    
    for k_idx in range(num_kpoints):
        # Create k-dependent dispersion (cosine-like)
        k_factor = np.cos(2 * np.pi * k_idx / num_kpoints)
        
        # Valence bands: below e_valence_top with some dispersion
        valence = np.linspace(
            e_valence_top - bandwidth,
            e_valence_top,
            num_valence
        ) + 0.2 * k_factor * np.linspace(0, 1, num_valence)
        
        # Conduction bands: above valence + gap
        conduction = np.linspace(
            e_valence_top + band_gap,
            e_valence_top + band_gap + bandwidth,
            num_conduction
        ) + 0.2 * k_factor * np.linspace(0, 1, num_conduction)
        
        eigenvalues = np.concatenate([valence, conduction])
        eigenvalues_list.append(eigenvalues)
    
    return eigenvalues_list


def create_metal_bandstructure(
    num_kpoints: int = 10,
    num_bands: int = 20,
    bandwidth: float = 8.0,
    e_fermi: float = 0.0
) -> List[np.ndarray]:
    """
    Create a metallic band structure with bands crossing E_F.
    """
    eigenvalues_list = []
    
    for k_idx in range(num_kpoints):
        k_factor = np.cos(2 * np.pi * k_idx / num_kpoints)
        
        # Bands dispersing across Fermi level
        base_energies = np.linspace(e_fermi - bandwidth/2, e_fermi + bandwidth/2, num_bands)
        dispersion = 0.5 * k_factor * np.sin(np.linspace(0, np.pi, num_bands))
        
        eigenvalues = base_energies + dispersion
        eigenvalues_list.append(np.sort(eigenvalues))
    
    return eigenvalues_list


def create_mock_eigenvectors(
    num_kpoints: int,
    num_orbitals: int,
    num_bands: int,
    localized_orbitals: List[int] = None
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Create mock eigenvectors and overlap matrices for testing.
    
    Parameters
    ----------
    num_kpoints : int
        Number of k-points
    num_orbitals : int
        Number of LCAO orbitals
    num_bands : int
        Number of bands
    localized_orbitals : list of int, optional
        Orbital indices that should have strong contributions to bands.
        These will have larger weights in the eigenvectors.
        
    Returns
    -------
    eigenvectors_list : list of ndarrays
        Mock eigenvectors C(k)
    S_k_list : list of ndarrays
        Mock overlap matrices S(k) (identity for simplicity)
    """
    if localized_orbitals is None:
        localized_orbitals = list(range(min(num_bands, num_orbitals)))
    
    eigenvectors_list = []
    S_k_list = []
    
    for k_idx in range(num_kpoints):
        # Create eigenvectors with stronger contribution from localized_orbitals
        C_k = np.random.randn(num_orbitals, num_bands) + \
              1j * np.random.randn(num_orbitals, num_bands)
        
        # Enhance contribution from specified orbitals
        for orb_idx in localized_orbitals:
            if orb_idx < num_orbitals:
                C_k[orb_idx, :] *= 3.0
        
        # Normalize columns
        for band_idx in range(num_bands):
            C_k[:, band_idx] /= np.linalg.norm(C_k[:, band_idx])
        
        eigenvectors_list.append(C_k)
        
        # Use identity for overlap (orthonormal basis approximation)
        S_k_list.append(np.eye(num_orbitals, dtype=complex))
    
    return eigenvectors_list, S_k_list


# ==============================================================================
# Tests for Fermi Energy Estimation
# ==============================================================================

class TestFermiEnergyEstimation:
    """Tests for estimate_fermi_energy function."""
    
    def test_semiconductor_midgap(self):
        """Test Fermi level detection in semiconductor using midgap method."""
        eigenvalues = create_simple_bandstructure(
            num_kpoints=10, num_bands=20, band_gap=2.0, 
            e_valence_top=0.0
        )
        
        e_fermi = estimate_fermi_energy(eigenvalues, method='midgap')
        
        # Should be approximately in the middle of the gap (0 to 2 eV)
        assert 0.5 < e_fermi < 1.5, f"E_F = {e_fermi} not in expected range"
    
    def test_semiconductor_electron_count(self):
        """Test Fermi level from electron counting."""
        eigenvalues = create_simple_bandstructure(
            num_kpoints=10, num_bands=20, band_gap=2.0
        )
        
        # 10 electrons should fill 10 bands (half)
        e_fermi = estimate_fermi_energy(eigenvalues, num_electrons=10, method='electron_count')
        
        # Should be in the gap
        assert -0.5 < e_fermi < 2.5, f"E_F = {e_fermi} not in gap region"
    
    def test_metal_midgap(self):
        """Test Fermi level in metal (should find pseudo-gap near center)."""
        eigenvalues = create_metal_bandstructure(
            num_kpoints=10, num_bands=20, e_fermi=0.0
        )
        
        e_fermi = estimate_fermi_energy(eigenvalues, method='midgap')
        
        # Should be near the center (0 eV)
        assert -2.0 < e_fermi < 2.0, f"E_F = {e_fermi} not near center"
    
    def test_half_filling(self):
        """Test half-filling method."""
        eigenvalues = create_simple_bandstructure(
            num_kpoints=10, num_bands=20, band_gap=2.0, e_valence_top=0.0
        )
        
        e_fermi = estimate_fermi_energy(eigenvalues, method='half_filling')
        
        # Should be in the gap region
        assert -0.5 < e_fermi < 2.5
    
    def test_auto_method_with_electrons(self):
        """Test that 'auto' uses electron_count when num_electrons provided."""
        eigenvalues = create_simple_bandstructure(num_kpoints=10, num_bands=20)
        
        e_fermi_auto = estimate_fermi_energy(
            eigenvalues, num_electrons=10, method='auto'
        )
        e_fermi_explicit = estimate_fermi_energy(
            eigenvalues, num_electrons=10, method='electron_count'
        )
        
        assert np.isclose(e_fermi_auto, e_fermi_explicit)
    
    def test_auto_method_without_electrons(self):
        """Test that 'auto' uses midgap when num_electrons not provided."""
        eigenvalues = create_simple_bandstructure(num_kpoints=10, num_bands=20)
        
        e_fermi_auto = estimate_fermi_energy(eigenvalues, method='auto')
        e_fermi_midgap = estimate_fermi_energy(eigenvalues, method='midgap')
        
        assert np.isclose(e_fermi_auto, e_fermi_midgap)
    
    def test_invalid_method(self):
        """Test that invalid method raises error."""
        eigenvalues = create_simple_bandstructure()
        
        with pytest.raises(ValueError, match="Unknown method"):
            estimate_fermi_energy(eigenvalues, method='invalid')
    
    def test_electron_count_without_num_electrons(self):
        """Test that electron_count without num_electrons raises error."""
        eigenvalues = create_simple_bandstructure()

        with pytest.raises(ValueError, match="num_electrons required"):
            estimate_fermi_energy(eigenvalues, method='electron_count')


# ==============================================================================
# Tests for compute_fermi_level (proper smeared Fermi)
# ==============================================================================

class TestComputeFermiLevel:
    """Tests for the rigorous smeared-occupation Fermi-level solver."""

    def test_insulator_lands_in_gap(self):
        """For a clean-gap insulator, E_F must fall in the gap."""
        eigenvalues = create_simple_bandstructure(
            num_kpoints=10, num_bands=20,
            band_gap=2.0, bandwidth=4.0, e_valence_top=0.0,
        )
        # 10 valence bands → 10 electrons per k-point (spinor, 1 e/band)
        e_f = compute_fermi_level(eigenvalues, num_electrons=10, sigma=0.01)
        # Gap is [0, 2] → E_F should be inside (0, 2)
        assert 0.0 < e_f < 2.0, f"E_F={e_f} not in gap (0, 2)"
        # With σ=0.01, should be very close to midgap since gap >> σ.
        # The synthetic fixture's exact midgap depends on band dispersion;
        # just require it sits squarely inside the gap.
        assert 0.1 < e_f < 1.9, f"E_F={e_f} not interior of gap (0, 2)"

    def test_insulator_returns_midgap_not_edge(self):
        """For a clean insulator, E_F must be midgap, NOT pinned to VBM or CBM.

        Regression test: earlier versions let brentq land on the CBM because
        the residual is flat-zero across the gap. The fast-path in
        compute_fermi_level now detects the clean-gap case and returns
        midgap explicitly.
        """
        # Asymmetric gap so midgap ≠ 0, to catch accidental (VBM+CBM)/2 = 0
        eigenvalues = create_simple_bandstructure(
            num_kpoints=8, num_bands=20,
            band_gap=2.0, bandwidth=4.0, e_valence_top=-3.0,
        )
        # Expected: VBM≈-3, CBM≈-1, midgap≈-2
        e_f = compute_fermi_level(eigenvalues, num_electrons=10, sigma=1e-5,
                                  spin_degeneracy=1)
        # Must be well inside the gap and near the center (not an edge)
        all_eigs = np.asarray(eigenvalues)
        vbm = float(np.sort(all_eigs, axis=1)[:, 9].max())
        cbm = float(np.sort(all_eigs, axis=1)[:, 10].min())
        midgap = 0.5 * (vbm + cbm)
        assert abs(e_f - midgap) < 1e-6, (
            f"E_F={e_f} is not midgap ({midgap}). VBM={vbm}, CBM={cbm}"
        )

    def test_insulator_invariant_to_smearing_width(self):
        """For gap ≫ σ, E_F should be independent of σ."""
        eigenvalues = create_simple_bandstructure(
            band_gap=2.0, bandwidth=4.0, e_valence_top=0.0,
        )
        e_f_tight = compute_fermi_level(eigenvalues, num_electrons=10, sigma=0.01)
        e_f_loose = compute_fermi_level(eigenvalues, num_electrons=10, sigma=0.1)
        # In deep-gap regime, widening σ by 10× should change E_F by ≪ σ.
        assert abs(e_f_tight - e_f_loose) < 0.05

    def test_conservation_integer_occupation(self):
        """Total integrated occupation must equal num_electrons exactly."""
        from scipy.special import erfc
        eigenvalues = create_simple_bandstructure(
            band_gap=1.0, bandwidth=4.0, e_valence_top=0.0,
        )
        sigma = 0.02
        e_f = compute_fermi_level(eigenvalues, num_electrons=10, sigma=sigma)
        # Re-evaluate total occupation at the computed E_F
        all_eigs = np.asarray(eigenvalues)
        nk = all_eigs.shape[0]
        occ = 0.5 * erfc((all_eigs - e_f) / sigma)
        total = occ.sum() / nk  # equal k-weights
        assert abs(total - 10) < 1e-6

    def test_metal_gives_smooth_fermi(self):
        """For a metal (overlapping bands), E_F lands inside the overlap region."""
        # Construct a genuine metal: bands 2 and 3 overlap so that filling
        # ~2.5 bands per k-point requires partial occupation.
        nk, nb = 21, 5
        eigenvalues = []
        for ik in range(nk):
            k = ik / (nk - 1) - 0.5  # range [-0.5, 0.5]
            evals = np.array([
                -3.0,                 # core-like, always filled
                -1.5 + 0.2 * k,       # deep valence
                -0.4 + 1.6 * k,       # crosses zero, dispersive
                -0.1 + 1.2 * k,       # overlaps with band 2!
                2.0 + 0.1 * k,        # high conduction, never filled
            ])
            eigenvalues.append(np.sort(evals))
        # Fill 2.5 bands per k-point on average → partial occupation of
        # overlapping bands 2 and 3 is required.
        e_f = compute_fermi_level(
            eigenvalues, num_electrons=2, sigma=0.05)
        # The critical correctness property is that total occupation at E_F
        # equals num_electrons exactly (conservation). E_F's specific value
        # depends on band placement — we just check it's inside the
        # eigenvalue range.
        all_eigs = np.asarray(eigenvalues)
        assert all_eigs.min() < e_f < all_eigs.max()

        # Partial occupation conservation check.
        from scipy.special import erfc
        occ = 0.5 * erfc((all_eigs - e_f) / 0.05)
        total = occ.sum() / nk
        assert abs(total - 2) < 1e-6

    def test_different_smearing_types_consistent_for_insulator(self):
        """All smearing schemes should agree in the deep-gap limit."""
        eigenvalues = create_simple_bandstructure(
            band_gap=3.0, bandwidth=2.0, e_valence_top=0.0,
        )
        e_f_gauss = compute_fermi_level(
            eigenvalues, num_electrons=10, smearing='gaussian', sigma=0.01)
        e_f_fd = compute_fermi_level(
            eigenvalues, num_electrons=10, smearing='fermi_dirac', sigma=0.01)
        e_f_mp = compute_fermi_level(
            eigenvalues, num_electrons=10, smearing='methfessel_paxton', sigma=0.01)
        # All three should give essentially the same E_F when gap ≫ σ
        assert abs(e_f_gauss - e_f_fd) < 0.01
        assert abs(e_f_gauss - e_f_mp) < 0.01

    def test_unknown_smearing_raises(self):
        eigenvalues = create_simple_bandstructure()
        with pytest.raises(ValueError, match="Unknown smearing"):
            compute_fermi_level(eigenvalues, num_electrons=10, smearing='cold')

    def test_k_weights_non_normalized_stay_in_gap(self):
        """k_weights that don't sum to 1 are auto-normalized, so for an
        insulator the result still lands in the gap.

        Note: when the residual has a flat zero region (insulator gap),
        Brent's method can return any point in the zero-interval, so we
        only require the result to be valid, not bit-identical.
        """
        eigenvalues = create_simple_bandstructure(num_kpoints=10, band_gap=2.0)
        nk = len(eigenvalues)
        e_f_default = compute_fermi_level(eigenvalues, num_electrons=10)
        e_f_unnorm = compute_fermi_level(
            eigenvalues, num_electrons=10,
            k_weights=np.full(nk, 5.7),  # arbitrary scale, auto-normalized
        )
        # Both must land inside the gap (valence top = 0, gap = 2)
        assert 0.0 < e_f_default < 2.0
        assert 0.0 < e_f_unnorm < 2.0

    def test_zero_temp_legacy_method(self):
        """Legacy 'zero_temp' method should match old aufbau behavior exactly."""
        eigenvalues = create_simple_bandstructure(
            band_gap=1.0, bandwidth=4.0, e_valence_top=0.0,
        )
        # Pre-refactor behavior: midpoint of last-filled and first-empty
        # in the globally sorted eigenvalue list.
        all_eigs = np.sort(np.asarray(eigenvalues).flatten())
        nk = len(eigenvalues)
        states_to_fill = 10 * nk
        expected = (all_eigs[states_to_fill - 1] + all_eigs[states_to_fill]) / 2

        e_f = estimate_fermi_energy(
            eigenvalues, num_electrons=10, method='zero_temp')
        assert abs(e_f - expected) < 1e-10

    def test_electron_count_agrees_with_zero_temp_for_insulator(self):
        """For a clean-gap insulator both methods should agree within σ."""
        eigenvalues = create_simple_bandstructure(
            band_gap=2.0, bandwidth=4.0, e_valence_top=0.0,
        )
        e_f_new = estimate_fermi_energy(
            eigenvalues, num_electrons=10, method='electron_count')
        e_f_old = estimate_fermi_energy(
            eigenvalues, num_electrons=10, method='zero_temp')
        # Agree to within σ=0.01 eV (set inside electron_count)
        assert abs(e_f_new - e_f_old) < 0.05


# ==============================================================================
# Tests for Band Window Analysis
# ==============================================================================

class TestBandWindowAnalysis:
    """Tests for analyze_band_window function."""
    
    def test_all_bands_frozen(self):
        """Test window that contains all bands."""
        eigenvalues = create_simple_bandstructure(
            num_kpoints=10, num_bands=10, band_gap=1.0,
            e_valence_top=0.0, bandwidth=2.0
        )
        
        result = analyze_band_window(
            eigenvalues,
            outer_window=(-5.0, 10.0),  # Very wide window
            e_fermi=0.5,
            window_is_relative=False
        )
        
        assert result.num_wann == 10
        assert len(result.excluded_indices) == 0
        assert len(result.partial_indices) == 0
    
    def test_valence_only_window(self):
        """Test window that captures only valence bands."""
        eigenvalues = create_simple_bandstructure(
            num_kpoints=10, num_bands=20, band_gap=2.0,
            e_valence_top=0.0, bandwidth=3.0
        )
        
        result = analyze_band_window(
            eigenvalues,
            outer_window=(-4.0, 0.5),  # Below gap
            e_fermi=1.0,
            window_is_relative=False
        )
        
        # Should capture ~10 valence bands
        assert result.num_wann > 0
        assert result.num_wann <= 10
        # All frozen bands should be valence
        assert all(idx < 10 for idx in result.frozen_indices)
    
    def test_relative_window(self):
        """Test that relative window is correctly converted."""
        eigenvalues = create_simple_bandstructure(
            num_kpoints=10, num_bands=20, band_gap=2.0,
            e_valence_top=0.0
        )
        
        e_fermi = 1.0  # Middle of gap
        
        # Relative window: -2 to +2 around E_F
        result = analyze_band_window(
            eigenvalues,
            outer_window=(-2.0, 2.0),
            e_fermi=e_fermi,
            window_is_relative=True
        )
        
        # Absolute window should be (-1, 3)
        assert np.isclose(result.outer_window[0], e_fermi - 2.0)
        assert np.isclose(result.outer_window[1], e_fermi + 2.0)
    
    def test_partial_bands_detected(self):
        """Test that bands crossing window boundary are marked partial."""
        # Create bands with known ranges
        num_kpoints = 10
        num_bands = 5
        eigenvalues_list = []
        
        for k in range(num_kpoints):
            # Band 0: -2 to -1 (excluded below)
            # Band 1: -1.5 to 0 (partial - crosses lower)
            # Band 2: 0.5 to 1.5 (frozen)
            # Band 3: 1.0 to 2.5 (partial - crosses upper)
            # Band 4: 3 to 4 (excluded above)
            eigenvalues = np.array([
                -1.5 + 0.5 * np.sin(k),
                -0.75 + 0.75 * np.sin(k),
                1.0 + 0.5 * np.sin(k),
                1.75 + 0.75 * np.sin(k),
                3.5 + 0.5 * np.sin(k)
            ])
            eigenvalues_list.append(np.sort(eigenvalues))
        
        result = analyze_band_window(
            eigenvalues_list,
            outer_window=(0.0, 2.0),
            e_fermi=0.0,
            window_is_relative=False
        )
        
        # Check classifications
        assert len(result.partial_indices) >= 1, "Should have partial bands"
        assert len(result.frozen_indices) >= 1, "Should have frozen bands"
    
    def test_auto_fermi_detection(self):
        """Test automatic Fermi level detection when e_fermi=None."""
        eigenvalues = create_simple_bandstructure(
            num_kpoints=10, num_bands=20, band_gap=2.0
        )
        
        result = analyze_band_window(
            eigenvalues,
            outer_window=(-3.0, 3.0),
            e_fermi=None,  # Should auto-detect
            window_is_relative=True
        )
        
        # Fermi level should be set
        assert result.e_fermi is not None
        assert not np.isnan(result.e_fermi)
    
    def test_empty_window(self):
        """Test window that contains no bands."""
        eigenvalues = create_simple_bandstructure(
            num_kpoints=10, num_bands=20, band_gap=2.0,
            e_valence_top=0.0
        )
        
        result = analyze_band_window(
            eigenvalues,
            outer_window=(100.0, 200.0),  # Way above all bands
            e_fermi=0.0,
            window_is_relative=False
        )
        
        assert result.num_wann == 0
        assert len(result.frozen_indices) == 0


# ==============================================================================
# Tests for Frozen Band Continuity
# ==============================================================================

class TestFrozenContinuity:
    """Tests for check_frozen_continuity function."""
    
    def test_contiguous_bands(self):
        """Test detection of contiguous bands."""
        frozen_indices = np.array([5, 6, 7, 8, 9, 10])
        
        is_contiguous, gaps = check_frozen_continuity(frozen_indices)
        
        assert is_contiguous
        assert len(gaps) == 0
    
    def test_noncontiguous_bands(self):
        """Test detection of non-contiguous bands."""
        frozen_indices = np.array([5, 6, 7, 10, 11, 12])  # Gap at 8, 9
        
        is_contiguous, gaps = check_frozen_continuity(frozen_indices)
        
        assert not is_contiguous
        assert len(gaps) == 1
    
    def test_single_band(self):
        """Test single band is always contiguous."""
        frozen_indices = np.array([5])
        
        is_contiguous, gaps = check_frozen_continuity(frozen_indices)
        
        assert is_contiguous
    
    def test_empty_bands(self):
        """Test empty array is contiguous."""
        frozen_indices = np.array([], dtype=int)
        
        is_contiguous, gaps = check_frozen_continuity(frozen_indices)
        
        assert is_contiguous


# ==============================================================================
# Tests for Fermi Coverage Validation
# ==============================================================================

class TestFermiCoverage:
    """Tests for validate_fermi_coverage function."""
    
    def test_both_occupied_unoccupied(self):
        """Test detection when both occupied and unoccupied bands present."""
        eigenvalues = create_simple_bandstructure(
            num_kpoints=10, num_bands=20, band_gap=2.0, e_valence_top=0.0
        )
        
        # Frozen indices spanning gap
        frozen_indices = np.array([8, 9, 10, 11])  # 2 valence, 2 conduction
        e_fermi = 1.0  # In the gap
        
        result = validate_fermi_coverage(frozen_indices, eigenvalues, e_fermi)
        
        assert result['has_occupied']
        assert result['has_unoccupied']
    
    def test_only_occupied(self):
        """Test detection of only occupied bands."""
        eigenvalues = create_simple_bandstructure(
            num_kpoints=10, num_bands=20, band_gap=2.0, e_valence_top=0.0
        )
        
        frozen_indices = np.array([0, 1, 2, 3])  # Deep valence
        e_fermi = 1.0
        
        result = validate_fermi_coverage(frozen_indices, eigenvalues, e_fermi)
        
        assert result['has_occupied']
        assert not result['has_unoccupied']
    
    def test_crossing_fermi(self):
        """Test detection of bands crossing Fermi level (metallic)."""
        eigenvalues = create_metal_bandstructure(
            num_kpoints=10, num_bands=20, e_fermi=0.0
        )
        
        # Find bands that actually cross E_F
        all_eig = np.array(eigenvalues)
        crossing_bands = []
        for i in range(20):
            if all_eig[:, i].min() < 0.0 < all_eig[:, i].max():
                crossing_bands.append(i)
        
        if len(crossing_bands) > 0:
            frozen_indices = np.array(crossing_bands[:3])
            result = validate_fermi_coverage(frozen_indices, eigenvalues, 0.0)
            
            assert result['crosses_fermi']


# ==============================================================================
# Tests for Orbital Selection
# ==============================================================================

class TestOrbitalSelection:
    """Tests for select_projection_orbitals function."""
    
    def test_selects_correct_number(self):
        """Test that correct number of orbitals is selected."""
        num_orbitals = 50
        num_bands = 20
        num_wann = 10
        
        eigenvectors, S_k = create_mock_eigenvectors(
            num_kpoints=5, num_orbitals=num_orbitals, num_bands=num_bands
        )
        
        result = select_projection_orbitals(eigenvectors, S_k, num_wann)
        
        assert result.num_selected == num_wann
        assert len(result.selected_indices) == num_wann
    
    def test_selects_dominant_orbitals(self):
        """Test that orbitals with largest contributions are selected."""
        num_orbitals = 20
        num_bands = 10
        num_wann = 5
        
        # Orbitals 0, 2, 4, 6, 8 should dominate
        dominant_orbitals = [0, 2, 4, 6, 8]
        
        eigenvectors, S_k = create_mock_eigenvectors(
            num_kpoints=10, num_orbitals=num_orbitals, 
            num_bands=num_bands, localized_orbitals=dominant_orbitals
        )
        
        result = select_projection_orbitals(eigenvectors, S_k, num_wann)
        
        # Most selected orbitals should be from dominant set
        overlap = len(set(result.selected_indices) & set(dominant_orbitals))
        assert overlap >= 3, f"Only {overlap}/5 dominant orbitals selected"
    
    def test_with_band_indices(self):
        """Test selection considering only specific bands."""
        num_orbitals = 30
        num_bands = 20
        
        eigenvectors, S_k = create_mock_eigenvectors(
            num_kpoints=5, num_orbitals=num_orbitals, num_bands=num_bands
        )
        
        # Only consider bands 5-14
        band_indices = np.arange(5, 15)
        
        result = select_projection_orbitals(
            eigenvectors, S_k, num_wann=10, band_indices=band_indices
        )
        
        assert result.num_selected == 10
    
    def test_orbital_weights_sum(self):
        """Test that orbital weights are meaningful."""
        eigenvectors, S_k = create_mock_eigenvectors(
            num_kpoints=10, num_orbitals=20, num_bands=10
        )
        
        result = select_projection_orbitals(eigenvectors, S_k, num_wann=5)
        
        # Weights should all be positive
        assert np.all(result.orbital_weights >= 0)
        # Total weight should be significant
        assert np.sum(result.orbital_weights) > 0


# ==============================================================================
# Tests for Subspace Projections
# ==============================================================================

class TestSubspaceProjections:
    """Tests for compute_subspace_projections function."""
    
    def test_output_shape(self):
        """Test that output has correct shape."""
        num_kpoints = 5
        num_orbitals = 30
        num_bands = 20
        
        eigenvectors, S_k = create_mock_eigenvectors(
            num_kpoints=num_kpoints, num_orbitals=num_orbitals, num_bands=num_bands
        )
        
        band_indices = np.array([5, 6, 7, 8, 9])  # 5 bands
        orbital_indices = np.array([0, 2, 4, 6, 8])  # 5 orbitals
        
        A_k_list = compute_subspace_projections(
            eigenvectors, S_k, band_indices, orbital_indices
        )
        
        assert len(A_k_list) == num_kpoints
        # Shape should be (num_selected_bands, num_selected_orbitals)
        assert A_k_list[0].shape == (5, 5)
    
    def test_projection_normalization(self):
        """Test that projections have reasonable magnitudes."""
        eigenvectors, S_k = create_mock_eigenvectors(
            num_kpoints=5, num_orbitals=20, num_bands=10
        )
        
        band_indices = np.arange(10)
        orbital_indices = np.arange(10)
        
        A_k_list = compute_subspace_projections(
            eigenvectors, S_k, band_indices, orbital_indices
        )
        
        # Check that projections aren't all zeros or huge
        for A_k in A_k_list:
            max_element = np.max(np.abs(A_k))
            assert 0.01 < max_element < 100, f"Projection magnitude {max_element} unexpected"


# ==============================================================================
# Tests for Window Suggestion
# ==============================================================================

class TestWindowSuggestion:
    """Tests for suggest_optimal_window function."""
    
    def test_semiconductor_window(self):
        """Test window suggestion for semiconductor."""
        eigenvalues = create_simple_bandstructure(
            num_kpoints=10, num_bands=20, band_gap=2.0, 
            e_valence_top=0.0, bandwidth=3.0
        )
        
        e_min, e_max = suggest_optimal_window(eigenvalues, e_fermi=1.0)
        
        # Window should be reasonable (not too wide, not too narrow)
        window_width = e_max - e_min
        assert 2.0 < window_width < 20.0
    
    def test_window_with_target(self):
        """Test window suggestion with target num_wann."""
        eigenvalues = create_simple_bandstructure(
            num_kpoints=10, num_bands=20, band_gap=2.0
        )
        
        e_min, e_max = suggest_optimal_window(
            eigenvalues, e_fermi=1.0, target_num_wann=10
        )
        
        # Check that approximately 10 bands fit in window
        result = analyze_band_window(
            eigenvalues, 
            outer_window=(e_min, e_max),
            e_fermi=0.0,
            window_is_relative=False
        )
        
        # Allow some flexibility
        assert 5 <= result.num_wann <= 15


# ==============================================================================
# Tests for Print Function (Smoke Tests)
# ==============================================================================

class TestPrintAnalysis:
    """Smoke tests for print_band_analysis function."""
    
    def test_print_produces_output(self, capsys):
        """Test that print function produces output."""
        eigenvalues = create_simple_bandstructure(
            num_kpoints=5, num_bands=10
        )
        
        result = analyze_band_window(
            eigenvalues,
            outer_window=(-2.0, 4.0),
            e_fermi=1.0,
            window_is_relative=False
        )
        
        report = print_band_analysis(result)
        captured = capsys.readouterr()
        
        assert len(report) > 0
        assert "BAND WINDOW ANALYSIS" in report
        assert "FROZEN" in report or "num_wann = 0" in report
    
    def test_print_with_partial_bands(self, capsys):
        """Test print output includes partial band information."""
        eigenvalues = create_simple_bandstructure(
            num_kpoints=10, num_bands=20, band_gap=2.0
        )
        
        # Window that will have partial bands
        result = analyze_band_window(
            eigenvalues,
            outer_window=(-2.5, 3.5),
            e_fermi=1.0,
            window_is_relative=False
        )
        
        report = print_band_analysis(result)
        
        # Report should mention classifications
        assert "eV" in report


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_full_workflow_semiconductor(self):
        """Test complete workflow for a semiconductor."""
        # Create band structure
        eigenvalues = create_simple_bandstructure(
            num_kpoints=15, num_bands=30, band_gap=1.5,
            e_valence_top=0.0, bandwidth=4.0
        )
        
        # Step 1: Estimate Fermi level
        e_fermi = estimate_fermi_energy(eigenvalues, num_electrons=15)
        
        # Step 2: Analyze window
        result = analyze_band_window(
            eigenvalues,
            outer_window=(-3.0, 3.0),
            e_fermi=e_fermi,
            window_is_relative=True
        )
        
        # Step 3: Check continuity
        is_contiguous, gaps = check_frozen_continuity(result.frozen_indices)
        
        # Step 4: Validate Fermi coverage
        coverage = validate_fermi_coverage(
            result.frozen_indices, eigenvalues, e_fermi
        )
        
        # Assertions
        assert result.num_wann > 0, "Should have frozen bands"
        assert is_contiguous, "Frozen bands should be contiguous"
        assert coverage['has_occupied'], "Should have occupied bands"
    
    def test_full_workflow_with_projections(self):
        """Test complete workflow including orbital selection."""
        num_orbitals = 40
        num_bands = 40
        
        # Create band structure
        eigenvalues = create_simple_bandstructure(
            num_kpoints=10, num_bands=num_bands, band_gap=1.0
        )
        
        # Create mock eigenvectors
        eigenvectors, S_k = create_mock_eigenvectors(
            num_kpoints=10, num_orbitals=num_orbitals, num_bands=num_bands
        )
        
        # Analyze window
        result = analyze_band_window(
            eigenvalues,
            outer_window=(-2.0, 2.0),
            e_fermi=0.5,
            window_is_relative=True
        )
        
        if result.num_wann > 0:
            # Select projection orbitals
            orbital_result = select_projection_orbitals(
                eigenvectors, S_k,
                num_wann=result.num_wann,
                band_indices=result.frozen_indices
            )
            
            # Compute projections
            A_k_list = compute_subspace_projections(
                eigenvectors, S_k,
                band_indices=result.frozen_indices,
                orbital_indices=orbital_result.selected_indices
            )
            
            assert len(A_k_list) == 10
            assert A_k_list[0].shape == (result.num_wann, result.num_wann)


# ==============================================================================
# Edge Cases and Error Handling
# ==============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_single_kpoint(self):
        """Test with single k-point."""
        eigenvalues = [np.array([1.0, 2.0, 3.0, 4.0, 5.0])]
        
        e_fermi = estimate_fermi_energy(eigenvalues, method='midgap')
        
        result = analyze_band_window(
            eigenvalues,
            outer_window=(1.5, 4.5),
            e_fermi=0.0,
            window_is_relative=False
        )
        
        assert result.num_wann >= 0
    
    def test_single_band(self):
        """Test with single band."""
        eigenvalues = [np.array([0.0]) for _ in range(5)]
        
        result = analyze_band_window(
            eigenvalues,
            outer_window=(-1.0, 1.0),
            e_fermi=0.0,
            window_is_relative=False
        )
        
        assert result.num_wann == 1
    
    def test_degenerate_bands(self):
        """Test handling of degenerate bands."""
        # Create bands with degeneracies
        eigenvalues = []
        for _ in range(10):
            eig = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])  # Pairs of degenerate bands
            eigenvalues.append(eig)
        
        result = analyze_band_window(
            eigenvalues,
            outer_window=(-0.5, 1.5),
            e_fermi=0.0,
            window_is_relative=False
        )
        
        # Should capture bands 0-3 (two degenerate pairs)
        assert result.num_wann == 4
    
    def test_very_wide_window(self):
        """Test with window much wider than band structure."""
        eigenvalues = create_simple_bandstructure(num_bands=10)
        
        result = analyze_band_window(
            eigenvalues,
            outer_window=(-1000.0, 1000.0),
            e_fermi=0.0,
            window_is_relative=False
        )
        
        assert result.num_wann == 10  # All bands frozen
    
    def test_very_narrow_window(self):
        """Test with very narrow window."""
        eigenvalues = create_simple_bandstructure(num_bands=10, bandwidth=2.0)
        
        result = analyze_band_window(
            eigenvalues,
            outer_window=(0.0, 0.001),  # Very narrow
            e_fermi=0.0,
            window_is_relative=False
        )
        
        # Likely no bands fit entirely
        assert result.num_wann >= 0


# ==============================================================================
# Run Tests
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])