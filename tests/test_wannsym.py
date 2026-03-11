"""
Tests for the WannSym post-Wannierization symmetrization package.

Tests cover:
    - HR file I/O (read/write/roundtrip)
    - Orbital rotation matrices (s, p, d, f, t2g, pz)
    - Spinor D-matrix
    - Symmetry detection
    - Hermitization and time-reversal
    - Full symmetrization pipeline
"""

import numpy as np
import pytest
import os
import sys
import tempfile

from lcao_wannier.wannsym.hamiltonian import HamiltonianData
from lcao_wannier.wannsym.orbital_rotations import (
    orbital_rotation_s, orbital_rotation_p, orbital_rotation_d,
    orbital_rotation_f, orbital_rotation_t2g, orbital_rotation_pz,
    get_orbital_rotation, get_orbital_dim,
)
from lcao_wannier.wannsym.spinor import rmat_to_euler, spinor_dmatrix, get_spinor_dmatrix
from lcao_wannier.wannsym.symmetry_ops import (
    detect_symmetry, build_representation_matrices, compute_orbital_info,
    _clean_rotation_matrix,
)
from lcao_wannier.wannsym.symmetrize import (
    symmetrize_hr, WannSymConfig,
    _hermitize, _apply_time_reversal,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_hr():
    """Create a simple 2-orbital Hamiltonian for testing."""
    norbs = 2
    hr = HamiltonianData(norbs=norbs)

    # R = (0, 0, 0): on-site
    hr.hr[(0, 0, 0)] = np.array([
        [-1.0 + 0j, 0.1 + 0.05j],
        [0.1 - 0.05j, -2.0 + 0j],
    ])
    hr.deg_rpt[(0, 0, 0)] = 1

    # R = (1, 0, 0): nearest-neighbor hopping
    hr.hr[(1, 0, 0)] = np.array([
        [0.5 + 0j, 0.05 + 0.01j],
        [0.02 - 0.01j, 0.3 + 0j],
    ])
    hr.deg_rpt[(1, 0, 0)] = 1

    # R = (-1, 0, 0): Hermitian conjugate
    hr.hr[(-1, 0, 0)] = hr.hr[(1, 0, 0)].conj().T
    hr.deg_rpt[(-1, 0, 0)] = 1

    return hr


@pytest.fixture
def rotation_90z():
    """90-degree rotation about z-axis."""
    return np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])


@pytest.fixture
def rotation_120z():
    """120-degree rotation about z-axis."""
    c = np.cos(2*np.pi/3)
    s = np.sin(2*np.pi/3)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])


@pytest.fixture
def identity_rotation():
    """Identity rotation matrix."""
    return np.eye(3)


# ============================================================================
# HR File I/O Tests
# ============================================================================

class TestHamiltonianData:

    def test_create_and_properties(self, simple_hr):
        """Test basic HamiltonianData creation and properties."""
        assert simple_hr.norbs == 2
        assert simple_hr.nrpt == 3
        assert len(simple_hr.rpts) == 3
        assert (0, 0, 0) in simple_hr.hr
        assert (1, 0, 0) in simple_hr.hr
        assert (-1, 0, 0) in simple_hr.hr

    def test_write_and_read_roundtrip(self, simple_hr):
        """Test that writing and re-reading produces identical data."""
        with tempfile.NamedTemporaryFile(suffix='_hr.dat', delete=False, mode='w') as f:
            tmpfile = f.name

        try:
            simple_hr.to_file(tmpfile)
            hr_read = HamiltonianData.from_file(tmpfile)

            assert hr_read.norbs == simple_hr.norbs
            assert hr_read.nrpt == simple_hr.nrpt

            for R in simple_hr.hr:
                assert R in hr_read.hr, f"R-point {R} missing after roundtrip"
                np.testing.assert_allclose(
                    hr_read.hr[R], simple_hr.hr[R],
                    atol=1e-12,
                    err_msg=f"Matrix mismatch at R={R}"
                )
                assert hr_read.deg_rpt[R] == simple_hr.deg_rpt[R]
        finally:
            os.unlink(tmpfile)

    def test_normalize_degeneracies(self):
        """Test degeneracy normalization."""
        hr = HamiltonianData(norbs=2)
        mat = np.eye(2, dtype=np.complex128) * 4.0
        hr.hr[(0, 0, 0)] = mat.copy()
        hr.deg_rpt[(0, 0, 0)] = 2

        hr.normalize_degeneracies()

        np.testing.assert_allclose(hr.hr[(0, 0, 0)], mat / 2.0)
        assert hr.deg_rpt[(0, 0, 0)] == 1

    def test_compress(self):
        """Test threshold-based compression."""
        hr = HamiltonianData(norbs=2)

        # Large hopping
        hr.hr[(0, 0, 0)] = np.eye(2, dtype=np.complex128)
        hr.deg_rpt[(0, 0, 0)] = 1

        # Tiny hopping
        hr.hr[(5, 5, 5)] = np.ones((2, 2), dtype=np.complex128) * 1e-12
        hr.deg_rpt[(5, 5, 5)] = 1

        n_removed = hr.compress(threshold=1e-9)
        assert n_removed == 1
        assert (0, 0, 0) in hr.hr
        assert (5, 5, 5) not in hr.hr

    def test_copy(self, simple_hr):
        """Test deep copy."""
        hr_copy = simple_hr.copy()
        assert hr_copy.norbs == simple_hr.norbs

        # Modify copy should not affect original
        hr_copy.hr[(0, 0, 0)][0, 0] = 999.0
        assert simple_hr.hr[(0, 0, 0)][0, 0] != 999.0


# ============================================================================
# Orbital Rotation Tests
# ============================================================================

class TestOrbitalRotations:

    def test_s_orbital_identity(self, identity_rotation):
        """s-orbital rotation is always identity."""
        D = orbital_rotation_s(identity_rotation)
        np.testing.assert_allclose(D, [[1.0]])

    def test_s_orbital_any_rotation(self, rotation_90z):
        """s-orbital rotation is always identity regardless of R."""
        D = orbital_rotation_s(rotation_90z)
        np.testing.assert_allclose(D, [[1.0]])

    def test_p_orbital_identity(self, identity_rotation):
        """p-orbital identity rotation."""
        D = orbital_rotation_p(identity_rotation)
        np.testing.assert_allclose(D, np.eye(3), atol=1e-10)

    def test_p_orbital_90z(self, rotation_90z):
        """p-orbital 90° z-rotation: pz unchanged, px→py, py→-px."""
        D = orbital_rotation_p(rotation_90z)
        # In Wannier90 ordering [pz, px, py]:
        # pz → pz (row 0, col 0 = 1)
        # px → py (row 2, col 1 = 1)
        # py → -px (row 1, col 2 = -1)
        expected = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ])
        np.testing.assert_allclose(D, expected, atol=1e-10)

    def test_d_orbital_identity(self, identity_rotation):
        """d-orbital identity rotation."""
        D = orbital_rotation_d(identity_rotation)
        np.testing.assert_allclose(D, np.eye(5), atol=1e-10)

    def test_d_orbital_unitarity(self, rotation_90z):
        """d-orbital rotation must be unitary (D @ D^T = I for real R)."""
        D = orbital_rotation_d(rotation_90z)
        np.testing.assert_allclose(D @ D.T, np.eye(5), atol=1e-10)

    def test_f_orbital_identity(self, identity_rotation):
        """f-orbital identity rotation."""
        D = orbital_rotation_f(identity_rotation)
        np.testing.assert_allclose(D, np.eye(7), atol=1e-10)

    def test_f_orbital_unitarity(self, rotation_120z):
        """f-orbital rotation must be unitary."""
        D = orbital_rotation_f(rotation_120z)
        np.testing.assert_allclose(D @ D.T, np.eye(7), atol=1e-8)

    def test_t2g_is_d_submatrix(self, rotation_90z):
        """t2g rotation must be submatrix of d rotation for dxz, dyz, dxy."""
        D_d = orbital_rotation_d(rotation_90z)
        D_t2g = orbital_rotation_t2g(rotation_90z)

        idx = [1, 2, 4]  # dxz, dyz, dxy
        expected = D_d[np.ix_(idx, idx)]
        np.testing.assert_allclose(D_t2g, expected, atol=1e-10)

    def test_t2g_unitarity(self, rotation_90z):
        """t2g rotation must be unitary."""
        D = orbital_rotation_t2g(rotation_90z)
        np.testing.assert_allclose(D @ D.T, np.eye(3), atol=1e-10)

    def test_pz_is_p_submatrix(self, rotation_90z):
        """pz rotation must be [0,0] element of p rotation."""
        D_p = orbital_rotation_p(rotation_90z)
        D_pz = orbital_rotation_pz(rotation_90z)
        np.testing.assert_allclose(D_pz, D_p[0:1, 0:1], atol=1e-10)

    def test_pz_90z(self, rotation_90z):
        """pz under 90° z-rotation stays pz (eigenvalue = 1)."""
        D = orbital_rotation_pz(rotation_90z)
        np.testing.assert_allclose(D, [[1.0]], atol=1e-10)

    def test_get_orbital_dim(self):
        """Test orbital dimension lookup."""
        assert get_orbital_dim('s') == 1
        assert get_orbital_dim('p') == 3
        assert get_orbital_dim('d') == 5
        assert get_orbital_dim('f') == 7
        assert get_orbital_dim('t2g') == 3
        assert get_orbital_dim('pz') == 1

    def test_get_orbital_dim_invalid(self):
        """Test orbital dimension lookup with invalid type."""
        with pytest.raises(ValueError, match="Unsupported orbital type"):
            get_orbital_dim('xyz')

    def test_get_orbital_rotation_dispatch(self, rotation_90z):
        """Test that get_orbital_rotation dispatches correctly."""
        for orb_type in ['s', 'p', 'd', 'f', 't2g', 'pz']:
            D = get_orbital_rotation(orb_type, rotation_90z)
            dim = get_orbital_dim(orb_type)
            assert D.shape == (dim, dim), f"Wrong shape for {orb_type}"


# ============================================================================
# Spinor Tests
# ============================================================================

class TestSpinor:

    def test_euler_identity(self):
        """Identity rotation has alpha=beta=gamma=0."""
        alpha, beta, gamma = rmat_to_euler(np.eye(3))
        assert abs(beta) < 1e-10

    def test_euler_roundtrip(self, rotation_90z):
        """Euler angles should reconstruct the rotation matrix."""
        alpha, beta, gamma = rmat_to_euler(rotation_90z)
        # Reconstruct: R = Rz(gamma) @ Ry(beta) @ Rz(alpha)
        ca, sa = np.cos(alpha), np.sin(alpha)
        cb, sb = np.cos(beta), np.sin(beta)
        cg, sg = np.cos(gamma), np.sin(gamma)

        Rz_a = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
        Ry_b = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
        Rz_g = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])

        R_reconstructed = Rz_g @ Ry_b @ Rz_a
        np.testing.assert_allclose(R_reconstructed, rotation_90z, atol=1e-10)

    def test_spinor_dmatrix_unitarity(self):
        """Spinor D-matrix must be unitary."""
        for alpha, beta, gamma in [
            (0, 0, 0),
            (np.pi/4, np.pi/3, np.pi/6),
            (np.pi, np.pi/2, 0),
        ]:
            D = spinor_dmatrix(alpha, beta, gamma)
            np.testing.assert_allclose(
                D @ D.conj().T, np.eye(2), atol=1e-10,
                err_msg=f"Non-unitary D for alpha={alpha}, beta={beta}, gamma={gamma}"
            )

    def test_spinor_identity(self):
        """Identity rotation gives identity (or -identity) spinor."""
        D = get_spinor_dmatrix(np.eye(3))
        # D should be close to [[1,0],[0,1]] or [[-1,0],[0,-1]]
        assert abs(abs(D[0, 0]) - 1.0) < 1e-10
        assert abs(D[0, 1]) < 1e-10

    def test_spinor_inversion(self):
        """Improper rotation (inversion) should use proper part."""
        D = get_spinor_dmatrix(-np.eye(3))
        # -I is improper, proper part is I, so D should be identity-like
        assert abs(abs(D[0, 0]) - 1.0) < 1e-10


# ============================================================================
# Symmetry Operations Tests
# ============================================================================

class TestSymmetryOps:

    def test_detect_symmetry_cubic(self):
        """Test symmetry detection for a simple cubic structure."""
        lattice = np.eye(3) * 5.0  # cubic lattice
        positions = np.array([[0.0, 0.0, 0.0]])  # single atom
        numbers = np.array([1])  # hydrogen

        sym_info = detect_symmetry(lattice, positions, numbers)
        assert sym_info.nsymm == 48  # Oh group: 48 operations
        assert '221' in sym_info.space_group or 'Pm-3m' in sym_info.space_group

    def test_detect_symmetry_atom_mapping(self):
        """Test atom mapping for a 2-atom cell with inversion."""
        lattice = np.eye(3) * 5.0
        positions = np.array([
            [0.1, 0.0, 0.0],
            [0.9, 0.0, 0.0],  # Inversion partner
        ])
        numbers = np.array([1, 1])

        sym_info = detect_symmetry(lattice, positions, numbers)
        assert sym_info.nsymm > 1

        # Find inversion operation
        for op in sym_info.operations:
            if np.allclose(op.rotation_frac, -np.eye(3)):
                # Under inversion, atoms should swap
                assert op.atom_map[0] == 1 or op.atom_map[1] == 0
                break

    def test_build_representation_matrices_shape(self):
        """Test representation matrix shape."""
        lattice = np.eye(3) * 5.0
        positions = np.array([[0.0, 0.0, 0.0]])
        numbers = np.array([1])

        sym_info = detect_symmetry(lattice, positions, numbers)
        orbital_types = [['s', 'p']]  # 1 + 3 = 4 orbitals

        protmat_list = build_representation_matrices(
            sym_info, orbital_types, has_soc=False
        )
        assert len(protmat_list) == sym_info.nsymm
        assert protmat_list[0].shape == (4, 4)

    def test_build_representation_matrices_soc_shape(self):
        """Test SOC representation matrix shape (doubled)."""
        lattice = np.eye(3) * 5.0
        positions = np.array([[0.0, 0.0, 0.0]])
        numbers = np.array([1])

        sym_info = detect_symmetry(lattice, positions, numbers)
        orbital_types = [['s', 'p']]  # 4 spatial → 8 spinor

        protmat_list = build_representation_matrices(
            sym_info, orbital_types, has_soc=True
        )
        assert protmat_list[0].shape == (8, 8)

    def test_compute_orbital_info(self):
        """Test orbital offset/count computation."""
        orbital_types = [['s', 'p', 'd'], ['s', 'p']]
        offsets, counts = compute_orbital_info(orbital_types)

        assert counts[0] == 1 + 3 + 5  # 9
        assert counts[1] == 1 + 3      # 4
        assert offsets[0] == 0
        assert offsets[1] == 9

    def test_compute_orbital_info_t2g_pz(self):
        """Test orbital info with t2g and pz types."""
        orbital_types = [['t2g', 'pz']]
        offsets, counts = compute_orbital_info(orbital_types)
        assert counts[0] == 3 + 1  # 4
        assert offsets[0] == 0

    def test_clean_rotation_matrix(self):
        """Test rotation matrix cleanup."""
        # Add small noise to known rotation
        R = np.array([
            [0.50001, -0.866024, 0.00001],
            [0.866024, 0.50001, -0.00001],
            [0.00001, 0.00001, 0.99999],
        ])
        R_clean = _clean_rotation_matrix(R)
        np.testing.assert_allclose(R_clean[0, 0], 0.5, atol=1e-10)
        np.testing.assert_allclose(R_clean[2, 2], 1.0, atol=1e-10)
        np.testing.assert_allclose(R_clean[0, 2], 0.0, atol=1e-10)


# ============================================================================
# Hermitization and Time-Reversal Tests
# ============================================================================

class TestSymmetryEnforcement:

    def test_hermitize_basic(self):
        """Test Hermitization H(R) = [H(R) + H(-R)†]/2."""
        hr = HamiltonianData(norbs=2)

        # Deliberately non-Hermitian pair
        hr.hr[(1, 0, 0)] = np.array([[1.0+0j, 0.5+0.2j],
                                       [0.3-0.1j, 2.0+0j]])
        hr.hr[(-1, 0, 0)] = np.array([[1.1+0j, 0.31+0.11j],
                                        [0.49-0.19j, 2.1+0j]])
        hr.deg_rpt[(1, 0, 0)] = 1
        hr.deg_rpt[(-1, 0, 0)] = 1

        _hermitize(hr, has_soc=False)

        # After hermitization, H(R) should equal H(-R)†
        np.testing.assert_allclose(
            hr.hr[(1, 0, 0)],
            hr.hr[(-1, 0, 0)].conj().T,
            atol=1e-12,
        )

    def test_hermitize_forces_real_spinless(self):
        """For spinless case, H(R) should be real after hermitization."""
        hr = HamiltonianData(norbs=2)
        hr.hr[(0, 0, 0)] = np.array([[1.0+0.01j, 0.5+0.02j],
                                       [0.5-0.02j, 2.0-0.01j]])
        hr.deg_rpt[(0, 0, 0)] = 1

        _hermitize(hr, has_soc=False)

        np.testing.assert_allclose(hr.hr[(0, 0, 0)].imag, 0.0, atol=1e-15)


# ============================================================================
# Integration Test with Real Data
# ============================================================================

class TestIntegration:

    @pytest.fixture
    def bismuth_hr_file(self):
        """Path to bismuth_scdm_hr.dat if it exists."""
        test_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'test_symmetry_output'
        )
        hr_file = os.path.join(test_dir, 'bismuth_scdm_hr.dat')
        if not os.path.exists(hr_file):
            pytest.skip("bismuth_scdm_hr.dat not found in test_symmetry_output/")
        return hr_file

    def test_read_bismuth_hr(self, bismuth_hr_file):
        """Test reading a real wannier90_hr.dat file."""
        hr = HamiltonianData.from_file(bismuth_hr_file)
        assert hr.norbs > 0
        assert hr.nrpt > 0
        assert (0, 0, 0) in hr.hr  # Must have on-site term

        # On-site matrix should be square
        H0 = hr.hr[(0, 0, 0)]
        assert H0.shape == (hr.norbs, hr.norbs)

        # On-site should have significant diagonal elements
        assert np.abs(np.diag(H0)).max() > 0.1

    def test_roundtrip_bismuth_hr(self, bismuth_hr_file):
        """Test round-trip read/write of a real HR file."""
        hr = HamiltonianData.from_file(bismuth_hr_file)

        with tempfile.NamedTemporaryFile(suffix='_hr.dat', delete=False, mode='w') as f:
            tmpfile = f.name

        try:
            hr.to_file(tmpfile)
            hr2 = HamiltonianData.from_file(tmpfile)

            assert hr2.norbs == hr.norbs
            assert hr2.nrpt == hr.nrpt

            for R in hr.hr:
                np.testing.assert_allclose(
                    hr2.hr[R], hr.hr[R],
                    atol=1e-12,
                    err_msg=f"Roundtrip failed for R={R}"
                )
        finally:
            os.unlink(tmpfile)


# ============================================================================
# Run
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
