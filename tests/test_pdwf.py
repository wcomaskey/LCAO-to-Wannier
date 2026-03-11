"""
Tests for the LCAO-PDWF implementation.

Covers:
  - valence_config: lookup table, num_wann computation, target mask
  - basis_parser: ShellInfo parsing from Crystal23 output
  - lcao_pdwf: Lowdin projectability, classification, windows
"""

import numpy as np
import pytest
import os

from lcao_wannier.valence_config import (
    get_valence_l, get_num_target_orbitals, compute_num_wann,
    build_target_mask, ELEMENT_Z, VALENCE_CONFIG, summarize_config,
)
from lcao_wannier.basis_parser import (
    parse_basis_shells, get_atom_list, ShellInfo,
)
from lcao_wannier.lcao_pdwf import (
    compute_lowdin_projectability, compute_matrix_sqrt,
    classify_bands, determine_windows,
    check_frozen_interlopers, check_band_count,
    ClassificationParams, BandClassification, WindowParameters,
)


# ============================================================================
# Tests for valence_config.py
# ============================================================================

class TestValenceConfig:
    def test_get_valence_l_by_symbol(self):
        assert get_valence_l('H') == {0}
        assert get_valence_l('C') == {0, 1}
        assert get_valence_l('Fe') == {0, 2}
        assert get_valence_l('Bi') == {0, 1}

    def test_get_valence_l_by_Z(self):
        assert get_valence_l(1) == {0}
        assert get_valence_l(6) == {0, 1}
        assert get_valence_l(26) == {0, 2}

    def test_extended_config(self):
        # Fe extended adds p
        assert get_valence_l('Fe', extended=True) == {0, 1, 2}
        # Bi extended adds d
        assert get_valence_l('Bi', extended=True) == {0, 1, 2}
        # C extended is same as standard
        assert get_valence_l('C', extended=True) == {0, 1}

    def test_tm_with_p(self):
        # Fe with p override
        assert get_valence_l('Fe', include_tm_p=True) == {0, 1, 2}
        # Non-TM ignores flag
        assert get_valence_l('C', include_tm_p=True) == {0, 1}

    def test_unknown_element_raises(self):
        with pytest.raises(KeyError):
            get_valence_l('Xx')
        with pytest.raises(KeyError):
            get_valence_l(999)

    def test_num_target_orbitals(self):
        # Bi: s(1) + p(3) = 4
        assert get_num_target_orbitals('Bi') == 4
        # Bi SOC: 4 * 2 = 8
        assert get_num_target_orbitals('Bi', has_soc=True) == 8
        # Fe: s(1) + d(5) = 6
        assert get_num_target_orbitals('Fe') == 6
        # Fe extended: s(1) + p(3) + d(5) = 9
        assert get_num_target_orbitals('Fe', extended=True) == 9

    def test_compute_num_wann_MgB2(self):
        # MgB2: Mg(s+p=4) + B(s+p=4) + B(s+p=4) = 12
        assert compute_num_wann(['Mg', 'B', 'B']) == 12

    def test_compute_num_wann_Sn_SOC(self):
        # alpha-Sn: 2 * Sn(s+p=4) * 2(SOC) = 16
        assert compute_num_wann(['Sn', 'Sn'], has_soc=True) == 16

    def test_compute_num_wann_Bi_SOC(self):
        # Bi2: 2 * Bi(s+p=4) * 2(SOC) = 16
        assert compute_num_wann(['Bi', 'Bi'], has_soc=True) == 16

    def test_build_target_mask_simple(self):
        """Test target mask with mock shells."""
        shells = [
            {'atom_index': 0, 'atom_Z': 83, 'l': 0, 'ao_indices': range(0, 1)},   # Bi s rad0 -> TARGET
            {'atom_index': 0, 'atom_Z': 83, 'l': 0, 'ao_indices': range(1, 2)},   # Bi s rad1 -> complement
            {'atom_index': 0, 'atom_Z': 83, 'l': 1, 'ao_indices': range(2, 5)},   # Bi p rad0 -> TARGET
            {'atom_index': 0, 'atom_Z': 83, 'l': 1, 'ao_indices': range(5, 8)},   # Bi p rad1 -> complement
            {'atom_index': 0, 'atom_Z': 83, 'l': 2, 'ao_indices': range(8, 13)},  # Bi d rad0 -> complement (d not in standard Bi)
        ]
        mask = build_target_mask(shells)
        expected = np.array([True, False, True, True, True, False, False, False,
                             False, False, False, False, False])
        np.testing.assert_array_equal(mask, expected)
        assert np.sum(mask) == 4  # 1s + 3p

    def test_build_target_mask_soc_block_convention(self):
        """SOC mask uses block convention: [mask, mask], not interleaved."""
        shells = [
            {'atom_index': 0, 'atom_Z': 83, 'l': 0, 'ao_indices': range(0, 1)},
            {'atom_index': 0, 'atom_Z': 83, 'l': 1, 'ao_indices': range(1, 4)},
        ]
        mask = build_target_mask(shells, has_soc=True)
        # Block convention: [T, T, T, T, T, T, T, T] for 4 spatial * 2
        assert len(mask) == 8  # 4 spatial * 2
        # First 4 = spatial mask, last 4 = same spatial mask
        np.testing.assert_array_equal(mask[:4], mask[4:])

    def test_summarize_config_runs(self):
        result = summarize_config(['Bi', 'Bi'], has_soc=True)
        assert 'num_wann: 16' in result


# ============================================================================
# Tests for basis_parser.py
# ============================================================================

class TestBasisParser:
    @pytest.fixture
    def bismuth_lines(self):
        """Load Bismuth test output file."""
        test_file = os.path.join(os.path.dirname(__file__), 'Bismuth_basis_40.out')
        if not os.path.exists(test_file):
            pytest.skip("Bismuth_basis_40.out not found")
        with open(test_file, 'r') as f:
            return f.readlines()

    def test_parse_bismuth_shells(self, bismuth_lines):
        shells, num_ao = parse_basis_shells(bismuth_lines, num_atoms=2)
        # Bi has: 4s + 3p + 3d = 10 shells per atom
        # For atom 1 (explicit): 4s + 3*3p + 3*5d = 4+9+15 = 28 AOs
        assert num_ao > 0
        assert len(shells) > 0

        # Check first shell is s
        assert shells[0].l == 0
        assert shells[0].atom_symbol == 'Bi'
        assert shells[0].atom_Z == 83
        assert shells[0].radial_index == 0

        # Check that p shells exist
        p_shells = [s for s in shells if s.l == 1 and s.atom_index == 0]
        assert len(p_shells) >= 1
        assert p_shells[0].num_ao == 3
        assert p_shells[0].radial_index == 0

        # Check d shells
        d_shells = [s for s in shells if s.l == 2 and s.atom_index == 0]
        assert len(d_shells) >= 1
        assert d_shells[0].num_ao == 5

    def test_get_atom_list(self, bismuth_lines):
        shells, _ = parse_basis_shells(bismuth_lines, num_atoms=2)
        atoms = get_atom_list(shells)
        assert atoms[0] == 'Bi'

    def test_shell_radial_indexing(self, bismuth_lines):
        shells, _ = parse_basis_shells(bismuth_lines, num_atoms=2)
        # For atom 0, s shells should have radial indices 0, 1, 2, 3
        s_shells = [s for s in shells if s.l == 0 and s.atom_index == 0]
        radials = [s.radial_index for s in s_shells]
        assert radials == list(range(len(s_shells)))


# ============================================================================
# Tests for lcao_pdwf.py
# ============================================================================

class TestMatrixSqrt:
    def test_identity(self):
        S = np.eye(4)
        S_half = compute_matrix_sqrt(S)
        np.testing.assert_allclose(S_half, np.eye(4), atol=1e-12)

    def test_diagonal(self):
        S = np.diag([4.0, 9.0, 16.0])
        S_half = compute_matrix_sqrt(S)
        expected = np.diag([2.0, 3.0, 4.0])
        np.testing.assert_allclose(S_half, expected, atol=1e-12)

    def test_product(self):
        """S^{1/2} @ S^{1/2} should equal S."""
        rng = np.random.default_rng(42)
        A = rng.random((5, 5))
        S = A @ A.T + np.eye(5)  # positive definite
        S_half = compute_matrix_sqrt(S)
        np.testing.assert_allclose(S_half @ S_half, S, atol=1e-10)


class TestLowdinProjectability:
    def test_orthogonal_identity(self):
        """With S=I and C=I, target projectability = 1 for target, 0 for complement."""
        n = 4
        S = np.eye(n)
        C = np.eye(n)
        target = np.array([True, True, False, False])

        proj = compute_lowdin_projectability([C], [S], target)
        assert proj.shape == (1, n)
        np.testing.assert_allclose(proj[0, 0], 1.0, atol=1e-12)
        np.testing.assert_allclose(proj[0, 1], 1.0, atol=1e-12)
        np.testing.assert_allclose(proj[0, 2], 0.0, atol=1e-12)
        np.testing.assert_allclose(proj[0, 3], 0.0, atol=1e-12)

    def test_sum_to_one(self):
        """Total projectability (target + complement) should sum to 1."""
        n = 4
        rng = np.random.default_rng(42)
        A = rng.random((n, n))
        S = A @ A.T + np.eye(n)
        # Solve generalized eigenvalue problem H C = S C E
        # to get S-orthonormal eigenvectors (C† S C = I)
        H = rng.random((n, n))
        H = H + H.T  # symmetric Hamiltonian
        eigvals, C = np.linalg.eigh(np.linalg.solve(S, H))
        # Normalize: C† S C = I
        norms = np.sqrt(np.diag(C.T @ S @ C))
        C = C / norms[np.newaxis, :]

        target = np.array([True, True, False, False])
        proj_target = compute_lowdin_projectability([C], [S], target)
        proj_complement = compute_lowdin_projectability([C], [S], ~target)

        # For each band: proj_target + proj_complement = 1
        np.testing.assert_allclose(
            proj_target[0] + proj_complement[0],
            np.ones(n), atol=1e-10,
        )

    def test_non_orthogonal_symmetric(self):
        """2x2 non-orthogonal case: symmetric bands have p=0.5 each."""
        s = 0.3  # overlap
        S = np.array([[1.0, s], [s, 1.0]])
        # Eigenvectors of S
        eigvals, eigvecs = np.linalg.eigh(S)
        # Use these as C (they satisfy C^T S C = diag)
        # Normalize: C^T S C = I
        norms = np.sqrt(np.diag(eigvecs.T @ S @ eigvecs))
        C = eigvecs / norms

        target = np.array([True, False])  # target = AO 0 only
        proj = compute_lowdin_projectability([C], [S], target)

        # By symmetry of the 2x2 system, each eigenvector has equal weight
        # on both AOs in Lowdin basis
        np.testing.assert_allclose(proj[0, 0], 0.5, atol=1e-10)
        np.testing.assert_allclose(proj[0, 1], 0.5, atol=1e-10)

    def test_range_zero_one(self):
        """Projectability should be in [0, 1]."""
        n = 6
        rng = np.random.default_rng(123)
        A = rng.random((n, n))
        S = A @ A.T + 2 * np.eye(n)
        eigvals_H, C = np.linalg.eigh(rng.random((n, n)) + rng.random((n, n)).T)
        # Normalize in S metric
        norms = np.sqrt(np.diag(C.T @ S @ C))
        C = C / norms

        target = np.array([True, True, False, False, False, False])
        proj = compute_lowdin_projectability([C], [S], target)

        assert np.all(proj >= -1e-10)
        assert np.all(proj <= 1.0 + 1e-10)


class TestClassifyBands:
    def test_clean_separation(self):
        """Clear gap: some bands high-p, some low-p."""
        nk, nb = 3, 6
        proj = np.zeros((nk, nb))
        proj[:, :3] = 0.98  # bands 0-2: high projectability
        proj[:, 3:] = 0.03  # bands 3-5: low projectability

        eigenvalues = np.zeros((nk, nb))
        for i in range(nb):
            eigenvalues[:, i] = -5.0 + i * 2.0  # spread across energy

        result = classify_bands(proj, eigenvalues, num_wann=3)

        assert len(result.frozen_indices) == 3
        assert len(result.excluded_indices) == 3
        assert result.gap_quality > 5.0

    def test_fuzzy_gap(self):
        """Some bands have intermediate projectability."""
        nk, nb = 3, 4
        proj = np.array([[0.99, 0.70, 0.40, 0.02]] * nk)
        eigenvalues = np.array([[-2.0, 0.0, 2.0, 5.0]] * nk)

        result = classify_bands(proj, eigenvalues, num_wann=2,
                                params=ClassificationParams(p_high=0.95, p_low=0.10))

        assert 'frozen' in result.category[0]   # 0.99 > 0.95
        assert 'disent' in result.category[1]    # 0.70 in [0.10, 0.95)
        assert 'disent' in result.category[2]    # 0.40 in [0.10, 0.95)
        assert 'excluded' in result.category[3]  # 0.02 < 0.10

    def test_energy_exclusion(self):
        """Bands far from E_F are excluded regardless of projectability."""
        nk, nb = 2, 3
        proj = np.array([[0.99, 0.98, 0.97]] * nk)
        eigenvalues = np.array([[-50.0, 0.0, 50.0]] * nk)

        params = ClassificationParams(e_fermi=0.0, e_above_max=20.0,
                                       e_below_min=-30.0)
        result = classify_bands(proj, eigenvalues, num_wann=3, params=params)

        assert result.category[0] == 'excluded'  # E = -50 < -30
        assert result.category[1] == 'frozen'     # E = 0, p = 0.98
        assert result.category[2] == 'excluded'  # E = 50 > 20


class TestDetermineWindows:
    def test_frozen_only(self):
        """All bands frozen: no disentanglement needed."""
        nk, nb = 2, 4
        proj = np.ones((nk, nb)) * 0.99
        eigenvalues = np.array([[-2.0, -1.0, 1.0, 2.0]] * nk)

        classification = classify_bands(proj, eigenvalues, num_wann=4)
        windows = determine_windows(classification, eigenvalues, e_fermi=0.0)

        assert windows.num_wann == 4
        assert windows.dis_froz_min is not None
        assert windows.dis_win_min is None  # no disentanglement

    def test_with_disentanglement(self):
        """Mix of frozen and disentangle bands."""
        nk, nb = 2, 5
        proj = np.array([[0.99, 0.98, 0.50, 0.30, 0.02]] * nk)
        eigenvalues = np.array([[-3.0, -1.0, 1.0, 3.0, 6.0]] * nk)

        classification = classify_bands(proj, eigenvalues, num_wann=2)
        windows = determine_windows(classification, eigenvalues, e_fermi=0.0)

        assert windows.num_wann == 2
        assert windows.dis_froz_min is not None
        assert windows.dis_win_min is not None
        # Frozen window should be inside outer window
        assert windows.dis_froz_min >= windows.dis_win_min
        assert windows.dis_froz_max <= windows.dis_win_max


class TestValidation:
    def test_interloper_detection(self):
        """Excluded band crossing frozen window should trigger warning."""
        nk, nb = 3, 4
        eigenvalues = np.array([
            [-3.0, -1.0, 0.0, 2.0],
            [-3.0, -1.0, 0.5, 2.0],
            [-3.0, -1.0, -0.5, 2.0],
        ])

        frozen_indices = np.array([0, 1])
        excluded_indices = np.array([2, 3])
        # Frozen window that captures band 2 eigenvalues
        dis_froz_min = -3.5
        dis_froz_max = 0.6

        warnings = check_frozen_interlopers(
            eigenvalues, frozen_indices, excluded_indices,
            dis_froz_min, dis_froz_max,
        )
        # Band 2 has eigenvalues 0.0, 0.5, -0.5 all inside [-3.5, 0.6]
        assert len(warnings) >= 1
        assert 'Band 2' in warnings[0]

    def test_band_count_check(self):
        """Too few bands in window should trigger warning."""
        nk, nb = 2, 4
        eigenvalues = np.array([
            [-5.0, -3.0, 1.0, 5.0],
            [-5.0, -3.0, 1.0, 5.0],
        ])
        # Window only contains 2 bands at each k-point
        warnings = check_band_count(eigenvalues, -4.0, 2.0, num_wann=3)
        assert len(warnings) == 2  # both k-points


# ============================================================================
# Integration test with Bismuth data
# ============================================================================

class TestBismuthIntegration:
    @pytest.fixture
    def bismuth_lines(self):
        test_file = os.path.join(os.path.dirname(__file__), 'Bismuth_basis_40.out')
        if not os.path.exists(test_file):
            pytest.skip("Bismuth_basis_40.out not found")
        with open(test_file, 'r') as f:
            return f.readlines()

    def test_full_pdwf_pipeline_bismuth(self, bismuth_lines):
        """Test parsing → mask → num_wann for Bismuth."""
        shells, num_ao = parse_basis_shells(bismuth_lines, num_atoms=2)
        atoms = get_atom_list(shells)

        # Should find Bi atoms
        assert 'Bi' in atoms

        # Build target mask (standard: s+p for Bi)
        mask = build_target_mask(shells, has_soc=False)
        num_target = int(np.sum(mask))

        # Each Bi contributes 1s + 3p = 4 target AOs
        # 2 atoms * 4 = 8
        assert num_target == 8

        # num_wann should match
        num_wann = compute_num_wann(atoms, has_soc=False)
        assert num_wann == 8

        # SOC version
        mask_soc = build_target_mask(shells, has_soc=True)
        assert int(np.sum(mask_soc)) == 16
        assert compute_num_wann(atoms, has_soc=True) == 16
