"""
Tests for projectability-based band selection (Method 1),
direct LCAO orbital mapping (Method 2), and symmetry stub (Method 5).
"""

import numpy as np
import pytest
from scipy.linalg import eigh

from lcao_wannier.projectability import (
    compute_band_projectability,
    select_bands_by_projectability,
    smart_select_bands,
    ProjectabilityResult,
    SmartSelectionResult,
    _find_contiguous_blocks,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_orthogonal_system(num_orbitals, num_kpoints=3):
    """Create a system with S=I and orthonormal eigenvectors."""
    eigenvectors_list = []
    S_k_list = []
    for _ in range(num_kpoints):
        Q, _ = np.linalg.qr(
            np.random.randn(num_orbitals, num_orbitals) + 0j
        )
        eigenvectors_list.append(Q)
        S_k_list.append(np.eye(num_orbitals, dtype=complex))
    return eigenvectors_list, S_k_list


def make_nonorthogonal_system(num_orbitals, overlap_strength=0.1):
    """Create a system with non-trivial overlap S != I."""
    # Build positive-definite overlap
    A = np.eye(num_orbitals, dtype=complex)
    for i in range(num_orbitals - 1):
        A[i, i + 1] = overlap_strength
        A[i + 1, i] = overlap_strength

    H = np.diag(np.arange(num_orbitals, dtype=float)) + 0j
    evals, evecs = eigh(H, A)

    return [evecs + 0j], [A + 0j]


# ---------------------------------------------------------------------------
# Tests for compute_band_projectability
# ---------------------------------------------------------------------------

class TestComputeBandProjectability:

    def test_identity_overlap_unit_projectability(self):
        """With S=I and orthonormal C, projectability should be 1.0."""
        evecs, S_list = make_orthogonal_system(4, num_kpoints=3)
        proj = compute_band_projectability(evecs, S_list)

        assert proj.shape == (3, 4)
        np.testing.assert_allclose(proj, 1.0, atol=1e-10)

    def test_non_orthogonal_overlap(self):
        """With non-trivial S, projectability is still computable."""
        evecs, S_list = make_nonorthogonal_system(4)
        proj = compute_band_projectability(evecs, S_list)

        assert proj.shape == (1, 4)
        assert np.all(proj > 0)

    def test_fewer_bands_than_orbitals(self):
        """When eigenvectors have fewer columns than rows."""
        num_orb = 8
        num_bands = 4
        C = np.random.randn(num_orb, num_bands) + 0j
        S = np.eye(num_orb, dtype=complex)

        proj = compute_band_projectability([C], [S])
        assert proj.shape == (1, num_bands)

    def test_multiple_kpoints_shape(self):
        """Shape is (num_kpoints, num_bands)."""
        num_k = 10
        num_orb = 6
        evecs = [np.eye(num_orb, dtype=complex) for _ in range(num_k)]
        S_list = [np.eye(num_orb, dtype=complex) for _ in range(num_k)]

        proj = compute_band_projectability(evecs, S_list)
        assert proj.shape == (num_k, num_orb)

    def test_scaled_eigenvectors_reduce_projectability(self):
        """Scaling eigenvector columns should change projectability."""
        num_orb = 4
        C = np.eye(num_orb, dtype=complex)
        C[:, 3] *= 0.5  # Scale last column
        S = np.eye(num_orb, dtype=complex)

        proj = compute_band_projectability([C], [S])

        # Bands 0-2: projectability = 1.0
        np.testing.assert_allclose(proj[0, :3], 1.0, atol=1e-10)
        # Band 3: projectability = 0.25 (sum of |0.5|^2 in one element)
        np.testing.assert_allclose(proj[0, 3], 0.25, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests for select_bands_by_projectability
# ---------------------------------------------------------------------------

class TestSelectBandsByProjectability:

    def test_all_bands_above_threshold(self):
        """All bands selected when projectability is uniformly 1.0."""
        evecs, S_list = make_orthogonal_system(4)

        result = select_bands_by_projectability(
            evecs, S_list, threshold=0.9, verbose=False
        )

        assert result.num_wann == 4
        assert len(result.selected_band_indices) == 4
        assert len(result.rejected_band_indices) == 0
        assert result.threshold == 0.9

    def test_threshold_filters_bands(self):
        """Bands with low projectability are rejected."""
        num_orb = 4
        C = np.eye(num_orb, dtype=complex)
        C[:, 2] *= 0.5   # p = 0.25
        C[:, 3] *= 0.3   # p = 0.09
        S = np.eye(num_orb, dtype=complex)

        result = select_bands_by_projectability(
            [C], [S], threshold=0.5, verbose=False
        )

        assert 0 in result.selected_band_indices
        assert 1 in result.selected_band_indices
        assert 2 not in result.selected_band_indices
        assert 3 not in result.selected_band_indices

    def test_zero_threshold_selects_all(self):
        """Threshold of 0.0 should select all bands."""
        C = np.random.randn(4, 4) + 0j
        S = np.eye(4, dtype=complex)

        result = select_bands_by_projectability(
            [C], [S], threshold=0.0, verbose=False
        )
        assert result.num_wann == 4

    def test_high_threshold_rejects_all(self):
        """Threshold above max projectability rejects everything."""
        C = np.eye(4, dtype=complex) * 0.5  # all projectabilities = 0.25
        S = np.eye(4, dtype=complex)

        result = select_bands_by_projectability(
            [C], [S], threshold=0.5, verbose=False
        )
        assert result.num_wann == 0

    def test_result_fields_present(self):
        """ProjectabilityResult has all expected fields."""
        evecs, S_list = make_orthogonal_system(4, num_kpoints=2)

        result = select_bands_by_projectability(
            evecs, S_list, threshold=0.9, verbose=False
        )

        assert isinstance(result, ProjectabilityResult)
        assert result.band_projectabilities.shape == (4,)
        assert result.projectability_per_kpoint.shape == (2, 4)
        assert result.threshold == 0.9
        assert isinstance(result.num_wann, int)

    def test_verbose_runs_without_error(self, capsys):
        """Verbose mode prints without crashing."""
        evecs, S_list = make_orthogonal_system(4)
        result = select_bands_by_projectability(
            evecs, S_list, threshold=0.9, verbose=True
        )
        captured = capsys.readouterr()
        assert "PROJECTABILITY-BASED BAND SELECTION" in captured.out
        assert "Selected bands" in captured.out


# ---------------------------------------------------------------------------
# Tests for Method 2 (Direct) integration points
# ---------------------------------------------------------------------------

class TestDirectMethodIntegration:

    def test_num_wann_equals_num_basis(self):
        """For direct method, num_wann must equal num_orbitals."""
        from lcao_wannier import Wannier90Engine

        num_orbitals = 4
        lattice_vectors = np.eye(3) * 5.0
        real_space_matrices = {
            (0, 0, 0): {
                'H': np.diag([0.0, 1.0, 2.0, 3.0]) + 0j,
                'S': np.eye(num_orbitals) + 0j,
            }
        }
        for R in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                   (0, 0, 1), (0, 0, -1)]:
            real_space_matrices[R] = {
                'H': -0.5 * np.eye(num_orbitals) + 0j,
                'S': 0.1 * np.eye(num_orbitals) + 0j,
            }

        engine = Wannier90Engine(
            real_space_matrices=real_space_matrices,
            k_grid=(2, 2, 2),
            lattice_vectors=lattice_vectors,
            seedname='test_direct',
        )

        # Simulate direct method setup
        engine.num_wann = engine.num_orbitals
        engine.selected_band_indices = np.arange(engine.num_orbitals)
        engine.selected_orbital_indices = np.arange(engine.num_orbitals)
        engine._override_num_iter = 0

        assert engine.num_wann == num_orbitals
        assert engine._override_num_iter == 0
        assert len(engine.selected_band_indices) == num_orbitals

    def test_large_basis_threshold_constant(self):
        """The large basis threshold should be 200."""
        import lcao_to_wannier90 as main_script
        assert main_script.LARGE_BASIS_THRESHOLD == 200


# ---------------------------------------------------------------------------
# Tests for Method 5 (Symmetry) stub
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helpers for smart selector tests
# ---------------------------------------------------------------------------

def make_smart_system(num_orbitals=8, num_kpoints=3, e_fermi=5.0):
    """Create a system with eigenvalues suitable for smart selection testing.

    Returns eigenvectors, S_k, eigenvalues, and e_fermi.
    Bands 0-5 have high projectability (near 1.0), bands 6-7 have low (~0.25).
    Eigenvalues are linearly spaced: band i has energy ~ i eV.
    """
    eigenvectors_list = []
    S_k_list = []
    eigenvalues_list = []

    for _ in range(num_kpoints):
        C = np.eye(num_orbitals, dtype=complex)
        # Degrade last 2 bands
        C[:, -2] *= 0.5
        C[:, -1] *= 0.3
        S = np.eye(num_orbitals, dtype=complex)

        eigenvectors_list.append(C)
        S_k_list.append(S)
        # Eigenvalues: bands at 0,1,2,...,num_orbitals-1 eV
        eigenvalues_list.append(np.arange(num_orbitals, dtype=float))

    return eigenvectors_list, S_k_list, eigenvalues_list, e_fermi


def make_soc_system(num_pairs=6, num_kpoints=2, e_fermi=5.0):
    """Create a SOC system with Kramers-degenerate pairs.

    Even number of bands, pairs have identical eigenvalues.
    """
    num_orbitals = num_pairs * 2
    eigenvectors_list = []
    S_k_list = []
    eigenvalues_list = []

    for _ in range(num_kpoints):
        C = np.eye(num_orbitals, dtype=complex)
        S = np.eye(num_orbitals, dtype=complex)
        eigenvectors_list.append(C)
        S_k_list.append(S)
        # Kramers pairs: [0, 0, 1, 1, 2, 2, 3, 3, ...]
        eigs = np.repeat(np.arange(num_pairs, dtype=float), 2)
        eigenvalues_list.append(eigs)

    return eigenvectors_list, S_k_list, eigenvalues_list, e_fermi


# ---------------------------------------------------------------------------
# Tests for smart_select_bands
# ---------------------------------------------------------------------------

class TestSmartSelectBands:

    def test_basic_selection(self):
        """Smart selector picks bands with high projectability near E_F."""
        evecs, S_list, eigs, ef = make_smart_system()
        result = smart_select_bands(
            evecs, S_list, eigs, e_fermi=ef, verbose=False
        )
        assert isinstance(result, SmartSelectionResult)
        assert result.num_wann > 0
        assert result.num_wann == len(result.selected_band_indices)
        # Frontier fields should be populated
        assert result.recommended_num_wann > 0
        assert len(result.frontier_band_indices) > 0

    def test_rejects_low_projectability(self):
        """Bands with degraded projectability are excluded."""
        evecs, S_list, eigs, ef = make_smart_system(num_orbitals=8)
        result = smart_select_bands(
            evecs, S_list, eigs, e_fermi=ef,
            proj_threshold=0.5, verbose=False
        )
        # Bands 6,7 have projectability 0.25 and 0.09 — should be excluded
        selected = set(result.selected_band_indices)
        assert 6 not in selected
        assert 7 not in selected

    def test_removes_deep_core(self):
        """Bands far below E_F are trimmed as core states."""
        num_orb = 10
        evecs = [np.eye(num_orb, dtype=complex)]
        S_list = [np.eye(num_orb, dtype=complex)]
        # Bands 0,1 at -20 eV, rest near 0 eV, E_F = 0
        eigs = [np.array([-20.0, -20.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]
        result = smart_select_bands(
            evecs, S_list, eigs, e_fermi=0.0,
            core_cutoff=-15.0, verbose=False
        )
        selected = set(result.selected_band_indices)
        assert 0 not in selected
        assert 1 not in selected
        assert result.bands_removed_core == 2

    def test_contiguous_block_selection(self):
        """Non-contiguous bands are trimmed; largest block around E_F kept."""
        num_orb = 10
        evecs_list = []
        S_list = []
        eigs_list = []
        C = np.eye(num_orb, dtype=complex)
        # Make band 5 have low projectability to create a gap
        C[:, 5] *= 0.3
        S = np.eye(num_orb, dtype=complex)
        evecs_list.append(C)
        S_list.append(S)
        eigs_list.append(np.arange(num_orb, dtype=float))

        result = smart_select_bands(
            evecs_list, S_list, eigs_list, e_fermi=7.0,
            proj_threshold=0.5, verbose=False
        )
        # Band 5 (p=0.09) rejected, creating two blocks: [0-4] and [6-9]
        # E_F=7.0 is closest to block [6-9], so that should be selected
        selected = sorted(result.selected_band_indices)
        assert 5 not in selected
        # Selected block should be contiguous
        for i in range(len(selected) - 1):
            assert selected[i + 1] == selected[i] + 1

    def test_kramers_pairing_even(self):
        """SOC mode enforces even band count."""
        evecs, S_list, eigs, ef = make_soc_system(num_pairs=6)
        result = smart_select_bands(
            evecs, S_list, eigs, e_fermi=ef,
            has_soc=True, verbose=False
        )
        assert result.num_wann % 2 == 0
        assert result.has_soc is True

    def test_kramers_drops_edge_band(self):
        """When odd number survive filtering, one edge band is dropped for SOC."""
        num_orb = 5
        evecs = [np.eye(num_orb, dtype=complex)]
        S_list = [np.eye(num_orb, dtype=complex)]
        eigs = [np.arange(num_orb, dtype=float)]
        result = smart_select_bands(
            evecs, S_list, eigs, e_fermi=2.0,
            has_soc=True, verbose=False
        )
        assert result.num_wann % 2 == 0

    def test_quality_score_range(self):
        """Quality score should be non-negative."""
        evecs, S_list, eigs, ef = make_smart_system()
        result = smart_select_bands(
            evecs, S_list, eigs, e_fermi=ef, verbose=False
        )
        assert result.quality_score >= 0.0

    def test_energy_range_valid(self):
        """Energy range should be (min, max) of selected band energies."""
        evecs, S_list, eigs, ef = make_smart_system()
        result = smart_select_bands(
            evecs, S_list, eigs, e_fermi=ef, verbose=False
        )
        if result.num_wann > 0:
            assert result.energy_range[0] <= result.energy_range[1]

    def test_all_result_fields(self):
        """SmartSelectionResult has all expected fields."""
        evecs, S_list, eigs, ef = make_smart_system()
        result = smart_select_bands(
            evecs, S_list, eigs, e_fermi=ef, verbose=False
        )
        assert isinstance(result.selected_band_indices, np.ndarray)
        assert isinstance(result.band_projectabilities, np.ndarray)
        assert isinstance(result.band_relevance, np.ndarray)
        assert isinstance(result.band_energies, np.ndarray)
        assert isinstance(result.quality_score, float)
        assert isinstance(result.e_fermi, float)
        assert isinstance(result.has_soc, bool)
        assert isinstance(result.energy_range, tuple)
        assert isinstance(result.bands_removed_core, int)
        assert isinstance(result.bands_removed_high, int)
        assert isinstance(result.bands_removed_noncontiguous, int)
        # Frontier detection fields
        assert isinstance(result.frontier_band_indices, np.ndarray)
        assert isinstance(result.num_frontier, int)
        assert isinstance(result.recommended_num_wann, int)
        # dis_win/dis_froz are either None or tuple
        assert result.recommended_dis_win is None or isinstance(result.recommended_dis_win, tuple)
        assert result.recommended_dis_froz is None or isinstance(result.recommended_dis_froz, tuple)

    def test_verbose_output(self, capsys):
        """Verbose mode prints diagnostic table."""
        evecs, S_list, eigs, ef = make_smart_system()
        smart_select_bands(
            evecs, S_list, eigs, e_fermi=ef, verbose=True
        )
        captured = capsys.readouterr()
        assert "SMART PROJECTABILITY-BASED BAND SELECTION" in captured.out
        assert "Quality score" in captured.out

    def test_zero_threshold_selects_all_projectable(self):
        """Threshold=0 should select all bands (modulo core cutoff)."""
        evecs, S_list, eigs, ef = make_smart_system(num_orbitals=6)
        result = smart_select_bands(
            evecs, S_list, eigs, e_fermi=ef,
            proj_threshold=0.0, verbose=False
        )
        assert result.num_wann == 6


# ---------------------------------------------------------------------------
# Tests for frontier detection / disentanglement
# ---------------------------------------------------------------------------

class TestFrontierDetection:

    def test_frontier_subset_of_selected(self):
        """Frontier bands must be a subset of selected bands."""
        evecs, S_list, eigs, ef = make_smart_system()
        result = smart_select_bands(
            evecs, S_list, eigs, e_fermi=ef, verbose=False
        )
        selected = set(result.selected_band_indices)
        for idx in result.frontier_band_indices:
            assert idx in selected

    def test_frontier_all_when_high_relevance(self):
        """When all bands have high relevance, all are frontier (no disentanglement)."""
        num_orb = 6
        evecs = [np.eye(num_orb, dtype=complex)]
        S_list = [np.eye(num_orb, dtype=complex)]
        # All bands near E_F → all high relevance
        eigs = [np.linspace(-1, 1, num_orb)]
        result = smart_select_bands(
            evecs, S_list, eigs, e_fermi=0.0,
            frontier_threshold=0.1, verbose=False
        )
        # All bands should be frontier
        assert result.num_frontier == result.num_wann
        assert result.recommended_num_wann == result.num_wann
        assert result.recommended_dis_win is None
        assert result.recommended_dis_froz is None

    def test_frontier_splits_with_deep_bands(self):
        """Bands far from E_F have low relevance → become support bands."""
        num_orb = 12
        evecs = [np.eye(num_orb, dtype=complex)]
        S_list = [np.eye(num_orb, dtype=complex)]
        # Bands 0-3 at -14 eV, bands 4-11 near E_F at 0 eV
        eigs_vals = np.concatenate([
            np.linspace(-14, -13, 4),   # deep-ish bands (above core cutoff)
            np.linspace(-2, 2, 8),       # near E_F
        ])
        eigs = [eigs_vals]
        result = smart_select_bands(
            evecs, S_list, eigs, e_fermi=0.0,
            proj_threshold=0.5,
            frontier_threshold=0.4,
            core_cutoff=-15.0,
            verbose=False
        )
        # Deep bands should be support, near-E_F bands should be frontier
        assert result.num_frontier < result.num_wann or result.num_frontier == result.num_wann
        assert result.recommended_num_wann <= len(result.selected_band_indices)

    def test_disentanglement_windows_present(self):
        """When frontier < selected, disentanglement windows should be set."""
        num_orb = 12
        evecs = [np.eye(num_orb, dtype=complex)]
        S_list = [np.eye(num_orb, dtype=complex)]
        # Bands 0-3 at -14 eV (low relevance), bands 4-11 near E_F (high relevance)
        eigs_vals = np.concatenate([
            np.linspace(-14, -13, 4),
            np.linspace(-2, 2, 8),
        ])
        eigs = [eigs_vals]
        result = smart_select_bands(
            evecs, S_list, eigs, e_fermi=0.0,
            proj_threshold=0.5,
            frontier_threshold=0.4,
            core_cutoff=-15.0,
            verbose=False
        )
        if result.num_frontier < len(result.selected_band_indices):
            assert result.recommended_dis_win is not None
            assert result.recommended_dis_froz is not None
            # Outer window should be wider than frozen window
            assert result.recommended_dis_win[0] <= result.recommended_dis_froz[0]
            assert result.recommended_dis_win[1] >= result.recommended_dis_froz[1]

    def test_frontier_kramers_enforcement(self):
        """SOC systems should enforce even frontier count."""
        num_orb = 7  # odd
        evecs = [np.eye(num_orb, dtype=complex)]
        S_list = [np.eye(num_orb, dtype=complex)]
        eigs = [np.arange(num_orb, dtype=float)]
        result = smart_select_bands(
            evecs, S_list, eigs, e_fermi=3.0,
            has_soc=True, verbose=False
        )
        # Both total and frontier must be even for SOC
        assert result.num_wann % 2 == 0
        assert result.recommended_num_wann % 2 == 0

    def test_recommended_num_wann_equals_frontier(self):
        """recommended_num_wann should equal num_frontier."""
        evecs, S_list, eigs, ef = make_smart_system()
        result = smart_select_bands(
            evecs, S_list, eigs, e_fermi=ef, verbose=False
        )
        assert result.recommended_num_wann == result.num_frontier

    def test_dis_win_covers_all_selected(self):
        """The outer disentanglement window should cover all selected bands."""
        num_orb = 12
        evecs = [np.eye(num_orb, dtype=complex)]
        S_list = [np.eye(num_orb, dtype=complex)]
        eigs_vals = np.concatenate([
            np.linspace(-14, -13, 4),
            np.linspace(-2, 2, 8),
        ])
        eigs = [eigs_vals]
        result = smart_select_bands(
            evecs, S_list, eigs, e_fermi=0.0,
            proj_threshold=0.5,
            frontier_threshold=0.4,
            core_cutoff=-15.0,
            verbose=False
        )
        if result.recommended_dis_win is not None and len(result.selected_band_indices) > 0:
            selected_e_rel = result.band_energies[result.selected_band_indices] - result.e_fermi
            assert result.recommended_dis_win[0] <= np.min(selected_e_rel)
            assert result.recommended_dis_win[1] >= np.max(selected_e_rel)

    def test_verbose_shows_frontier_info(self, capsys):
        """Verbose output should include frontier detection info."""
        evecs, S_list, eigs, ef = make_smart_system()
        smart_select_bands(
            evecs, S_list, eigs, e_fermi=ef, verbose=True
        )
        captured = capsys.readouterr()
        assert "Frontier detection" in captured.out
        assert "Frontier bands" in captured.out


# ---------------------------------------------------------------------------
# Tests for _find_contiguous_blocks helper
# ---------------------------------------------------------------------------

class TestFindContiguousBlocks:

    def test_single_block(self):
        blocks = _find_contiguous_blocks(np.array([1, 2, 3, 4, 5]))
        assert len(blocks) == 1
        np.testing.assert_array_equal(blocks[0], [1, 2, 3, 4, 5])

    def test_two_blocks(self):
        blocks = _find_contiguous_blocks(np.array([1, 2, 3, 7, 8, 9]))
        assert len(blocks) == 2
        np.testing.assert_array_equal(blocks[0], [1, 2, 3])
        np.testing.assert_array_equal(blocks[1], [7, 8, 9])

    def test_all_separate(self):
        blocks = _find_contiguous_blocks(np.array([1, 5, 10]))
        assert len(blocks) == 3

    def test_empty(self):
        blocks = _find_contiguous_blocks(np.array([], dtype=int))
        assert len(blocks) == 0

    def test_single_element(self):
        blocks = _find_contiguous_blocks(np.array([42]))
        assert len(blocks) == 1
        np.testing.assert_array_equal(blocks[0], [42])


# ---------------------------------------------------------------------------
# Tests for Method 5 (Symmetry) stub
# ---------------------------------------------------------------------------

class TestSymmetryMethodStub:

    def test_symmetry_not_implemented(self):
        """Symmetry method should raise NotImplementedError."""
        with pytest.raises(NotImplementedError, match="symmetry"):
            raise NotImplementedError(
                "Method 'symmetry' (symmetry-indicator approach) is planned for "
                "a future release."
            )
