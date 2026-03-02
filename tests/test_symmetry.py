#!/usr/bin/env python3
"""
Tests for symmetry.py and irreps.py modules.

Tests cover:
- Orbital rotation matrices (s, p, d, f)
- Spinor D-matrix (SU(2))
- Euler angle extraction
- Symmetry detection via spglib
- Representation matrix assembly
- Hermiticity enforcement
- Time-reversal enforcement
- Little group computation
- Band character computation
"""

import numpy as np
import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lcao_wannier.symmetry import (
    orbital_rotation_s,
    orbital_rotation_p,
    orbital_rotation_d,
    orbital_rotation_f,
    get_orbital_rotation,
    rmat_to_euler,
    spinor_dmatrix,
    get_spinor_dmatrix,
    detect_symmetry_operations,
    build_representation_matrices,
    enforce_hermiticity,
    enforce_time_reversal,
    _interleaved_to_block_spin,
    _block_to_interleaved_spin,
)
from lcao_wannier.irreps import (
    compute_little_group,
    compute_band_characters,
    identify_degenerate_groups,
    find_high_symmetry_kpoints,
)


# ============================================================================
# Rotation matrices for testing
# ============================================================================

def rotation_z(angle):
    """Rotation around z-axis by angle (radians)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rotation_x(angle):
    """Rotation around x-axis by angle (radians)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rotation_y(angle):
    """Rotation around y-axis by angle (radians)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# ============================================================================
# Tests: s-orbital rotation
# ============================================================================

class TestSOrbitalRotation:
    def test_identity(self):
        """s-orbital rotation under identity is [[1]]."""
        D = orbital_rotation_s(np.eye(3))
        assert D.shape == (1, 1)
        assert np.isclose(D[0, 0], 1.0)

    def test_any_rotation(self):
        """s-orbital is always [[1]] regardless of rotation."""
        for angle in [np.pi/4, np.pi/2, np.pi, 2*np.pi/3]:
            R = rotation_z(angle)
            D = orbital_rotation_s(R)
            assert np.isclose(D[0, 0], 1.0)


# ============================================================================
# Tests: p-orbital rotation
# ============================================================================

class TestPOrbitalRotation:
    def test_identity(self):
        """p-orbital rotation under identity is 3x3 identity."""
        D = orbital_rotation_p(np.eye(3))
        np.testing.assert_allclose(D, np.eye(3), atol=1e-10)

    def test_orthogonal(self):
        """p-orbital rotation matrices should be orthogonal: D.D^T = I."""
        for angle in [np.pi/6, np.pi/4, np.pi/3, np.pi/2, 2*np.pi/3]:
            R = rotation_z(angle)
            D = orbital_rotation_p(R)
            np.testing.assert_allclose(D @ D.T, np.eye(3), atol=1e-10)

    def test_c4z(self):
        """90-degree z-rotation: pz->pz, px->py, py->-px.

        In Wannier90 basis [pz, px, py]:
        R_z(90) acting on coordinates: x->y, y->-x, z->z
        For orbital representation (using R^{-1}):
        pz(R^{-1}r) = pz  (z-component unchanged)
        px(R^{-1}r) = py  (x-component of rotated point = y)
        py(R^{-1}r) = -px (y-component of rotated point = -x)

        So D should map:
        pz -> pz:  D[0,0] = 1
        px -> py:  D[2,1] = 1 (py row, px col)
        py -> -px: D[1,2] = -1 (px row, py col)
        """
        R = rotation_z(np.pi/2)
        D = orbital_rotation_p(R)

        # pz should map to pz
        assert abs(D[0, 0] - 1.0) < 1e-10, f"D[0,0] = {D[0,0]}, expected 1.0"
        assert abs(D[0, 1]) < 1e-10
        assert abs(D[0, 2]) < 1e-10

        # px should map to py (D[2,1] nonzero)
        assert abs(D[2, 1] - 1.0) < 1e-10, f"D[2,1] = {D[2,1]}, expected 1.0"

        # py should map to -px (D[1,2] nonzero)
        assert abs(D[1, 2] - (-1.0)) < 1e-10, f"D[1,2] = {D[1,2]}, expected -1.0"

    def test_c2z(self):
        """180-degree z-rotation: pz->pz, px->-px, py->-py."""
        R = rotation_z(np.pi)
        D = orbital_rotation_p(R)

        expected = np.diag([1.0, -1.0, -1.0])
        np.testing.assert_allclose(D, expected, atol=1e-10)

    def test_inversion(self):
        """Inversion: all p-orbitals change sign."""
        R = -np.eye(3)
        D = orbital_rotation_p(R)
        np.testing.assert_allclose(D, -np.eye(3), atol=1e-10)


# ============================================================================
# Tests: d-orbital rotation
# ============================================================================

class TestDOrbitalRotation:
    def test_identity(self):
        """d-orbital rotation under identity is 5x5 identity."""
        D = orbital_rotation_d(np.eye(3))
        np.testing.assert_allclose(D, np.eye(5), atol=1e-8)

    def test_orthogonal(self):
        """d-orbital rotation matrices should be orthogonal."""
        for angle in [np.pi/6, np.pi/4, np.pi/3, np.pi/2]:
            R = rotation_z(angle)
            D = orbital_rotation_d(R)
            np.testing.assert_allclose(D @ D.T, np.eye(5), atol=1e-8)

    def test_c4z_dz2(self):
        """C4z should leave dz2 invariant."""
        R = rotation_z(np.pi/2)
        D = orbital_rotation_d(R)
        # dz2 is index 0
        assert abs(D[0, 0] - 1.0) < 1e-8, f"D[0,0] = {D[0,0]}"

    def test_inversion(self):
        """Inversion should leave all d-orbitals invariant (even parity)."""
        R = -np.eye(3)
        D = orbital_rotation_d(R)
        np.testing.assert_allclose(D, np.eye(5), atol=1e-8)

    def test_det_one(self):
        """det(D) should be +1 for proper rotations."""
        for angle in [np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3]:
            R = rotation_z(angle)
            D = orbital_rotation_d(R)
            det = np.linalg.det(D)
            assert abs(det - 1.0) < 1e-6, f"det(D) = {det}"


# ============================================================================
# Tests: f-orbital rotation
# ============================================================================

class TestFOrbitalRotation:
    def test_identity(self):
        """f-orbital rotation under identity is 7x7 identity."""
        D = orbital_rotation_f(np.eye(3))
        np.testing.assert_allclose(D, np.eye(7), atol=1e-8)

    def test_orthogonal(self):
        """f-orbital rotation matrices should be orthogonal."""
        R = rotation_z(np.pi/3)
        D = orbital_rotation_f(R)
        np.testing.assert_allclose(D @ D.T, np.eye(7), atol=1e-7)

    def test_inversion(self):
        """Inversion should negate all f-orbitals (odd parity)."""
        R = -np.eye(3)
        D = orbital_rotation_f(R)
        np.testing.assert_allclose(D, -np.eye(7), atol=1e-8)


# ============================================================================
# Tests: Spinor D-matrix
# ============================================================================

class TestSpinorDMatrix:
    def test_identity(self):
        """Identity rotation gives identity D-matrix."""
        D = spinor_dmatrix(0.0, 0.0, 0.0)
        np.testing.assert_allclose(D, np.eye(2), atol=1e-10)

    def test_unitary(self):
        """Spinor D-matrix should be unitary: D @ D^dag = I."""
        for alpha in [0, np.pi/4, np.pi/2, np.pi]:
            for beta in [0, np.pi/4, np.pi/2]:
                for gamma in [0, np.pi/3, np.pi]:
                    D = spinor_dmatrix(alpha, beta, gamma)
                    np.testing.assert_allclose(
                        D @ D.conj().T, np.eye(2), atol=1e-10
                    )

    def test_det_one(self):
        """det(D) should be 1 (SU(2))."""
        for alpha in [0, np.pi/6, np.pi/2, np.pi]:
            for beta in [0, np.pi/4, np.pi/2]:
                D = spinor_dmatrix(alpha, beta, 0.0)
                det = np.linalg.det(D)
                assert abs(det - 1.0) < 1e-10, f"det = {det}"

    def test_z_rotation_pi(self):
        """Z-rotation by pi: D = [[exp(-i*pi/2), 0], [0, exp(i*pi/2)]] = [[-i, 0], [0, i]]."""
        D = get_spinor_dmatrix(rotation_z(np.pi))
        expected = np.diag([-1j, 1j])
        np.testing.assert_allclose(D, expected, atol=1e-10)


# ============================================================================
# Tests: Euler angle extraction
# ============================================================================

class TestEulerAngles:
    def test_identity(self):
        """Identity matrix gives zero angles."""
        alpha, beta, gamma = rmat_to_euler(np.eye(3))
        assert abs(beta) < 1e-10

    def test_roundtrip(self):
        """R -> Euler -> R should recover original rotation."""
        from lcao_wannier.symmetry import spinor_dmatrix as dmat
        for angle in [np.pi/6, np.pi/4, np.pi/3, np.pi/2]:
            R = rotation_z(angle)
            a, b, g = rmat_to_euler(R)
            # Reconstruct R from Euler angles
            cA, sA = np.cos(a), np.sin(a)
            cB, sB = np.cos(b), np.sin(b)
            cG, sG = np.cos(g), np.sin(g)
            R_rec = np.array([
                [cA*cB*cG - sA*sG, -sG*cA*cB - sA*cG, cA*sB],
                [sA*cB*cG + cA*sG, -sG*sA*cB + cA*cG, sA*sB],
                [-cG*sB, sB*sG, cB]
            ])
            np.testing.assert_allclose(R, R_rec, atol=1e-10)


# ============================================================================
# Tests: Symmetry detection (spglib)
# ============================================================================

class TestSymmetryDetection:
    def test_simple_cubic(self):
        """Simple cubic with one atom: 48 symmetry operations."""
        lattice = np.eye(3) * 5.0
        positions = np.array([[0.0, 0.0, 0.0]])
        numbers = np.array([1])

        sym_info = detect_symmetry_operations(lattice, positions, numbers)
        assert sym_info.nsymm == 48  # Oh point group
        assert '225' in sym_info.space_group or 'Fm-3m' in sym_info.space_group or 'Pm-3m' in sym_info.space_group

    def test_hexagonal_2d(self):
        """2D hexagonal lattice (like Bi monolayer)."""
        a = 4.383
        lattice = np.array([
            [a, 0.0, 0.0],
            [-a/2, a*np.sqrt(3)/2, 0.0],
            [0.0, 0.0, 50.0]  # Large vacuum
        ])
        # Two atoms at 1/3, 2/3 positions
        positions = np.array([
            [1.0/3.0, 2.0/3.0, 0.5],
            [2.0/3.0, 1.0/3.0, 0.5],
        ])
        numbers = np.array([83, 83])  # Bi

        sym_info = detect_symmetry_operations(lattice, positions, numbers, tolerance=1e-3)
        assert sym_info.nsymm > 1
        # Should detect at least inversion + C3 operations
        assert sym_info.nsymm >= 6

    def test_atom_mapping_identity(self):
        """Identity operation should map each atom to itself."""
        lattice = np.eye(3) * 5.0
        positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        numbers = np.array([1, 1])

        sym_info = detect_symmetry_operations(lattice, positions, numbers)

        # Find identity operation (rotation = identity matrix)
        identity_found = False
        for op in sym_info.operations:
            if np.allclose(op.rotation_frac, np.eye(3)):
                identity_found = True
                # Identity maps each atom to itself
                np.testing.assert_array_equal(op.atom_map, [0, 1])
                np.testing.assert_allclose(op.vec_shift, 0.0, atol=1e-10)
                break
        assert identity_found, "Identity operation not found"


# ============================================================================
# Tests: Representation matrix assembly
# ============================================================================

class TestRepresentationMatrices:
    def test_identity_operation(self):
        """Representation of identity should be identity matrix."""
        lattice = np.eye(3) * 5.0
        positions = np.array([[0.0, 0.0, 0.0]])
        numbers = np.array([1])

        sym_info = detect_symmetry_operations(lattice, positions, numbers)
        orbital_types = [['s', 'p']]  # 1 + 3 = 4 orbitals
        protmat_list = build_representation_matrices(sym_info, orbital_types, has_soc=False)

        # Find identity operation
        for i, op in enumerate(sym_info.operations):
            if np.allclose(op.rotation_frac, np.eye(3)):
                np.testing.assert_allclose(
                    protmat_list[i], np.eye(4), atol=1e-10
                )
                break

    def test_soc_doubling(self):
        """With SOC, representation matrix should be 2N x 2N."""
        lattice = np.eye(3) * 5.0
        positions = np.array([[0.0, 0.0, 0.0]])
        numbers = np.array([1])

        sym_info = detect_symmetry_operations(lattice, positions, numbers)
        orbital_types = [['s', 'p']]  # 4 spatial orbitals -> 8 with SOC
        protmat_list = build_representation_matrices(sym_info, orbital_types, has_soc=True)

        assert protmat_list[0].shape == (8, 8)


# ============================================================================
# Tests: Hermiticity enforcement
# ============================================================================

class TestHermiticityEnforcement:
    def test_already_hermitian(self):
        """Hermiticity enforcement should not change already-Hermitian matrices."""
        H = np.array([[1.0, 0.5+0.1j], [0.5-0.1j, 2.0]])
        S = np.eye(2)

        matrices = {
            (0, 0, 0): {'H': H, 'S': S},
        }
        result = enforce_hermiticity(matrices)
        np.testing.assert_allclose(result[(0, 0, 0)]['H'], H, atol=1e-15)

    def test_symmetrizes_R_and_negR(self):
        """H(R) = [H(R) + H(-R)^dag] / 2."""
        H_R = np.array([[1.0, 0.3+0.1j], [0.2-0.05j, 2.0]])
        H_negR = np.array([[1.0, 0.2+0.05j], [0.3-0.1j, 2.0]])
        S = np.eye(2)

        matrices = {
            (1, 0, 0): {'H': H_R, 'S': S.copy()},
            (-1, 0, 0): {'H': H_negR, 'S': S.copy()},
        }
        result = enforce_hermiticity(matrices)

        expected_R = (H_R + H_negR.conj().T) / 2.0
        np.testing.assert_allclose(result[(1, 0, 0)]['H'], expected_R, atol=1e-15)


# ============================================================================
# Tests: Time-reversal enforcement
# ============================================================================

class TestTimeReversalEnforcement:
    def test_preserves_kramers_structure(self):
        """Time-reversal should preserve block structure for SOC."""
        n = 2  # 2 spatial orbitals
        n2 = 4  # 4 total (spin up + spin down)
        H = np.random.randn(n2, n2) + 1j * np.random.randn(n2, n2)
        H = (H + H.conj().T) / 2  # Make Hermitian
        S = np.eye(n2, dtype=np.complex128)

        matrices = {
            (0, 0, 0): {'H': H.copy(), 'S': S.copy()},
        }
        result = enforce_time_reversal(matrices, n)

        # Result should still be Hermitian
        H_result = result[(0, 0, 0)]['H']
        np.testing.assert_allclose(
            H_result, H_result.conj().T, atol=1e-14,
            err_msg="Time-reversal enforcement broke Hermiticity"
        )


# ============================================================================
# Tests: Spin ordering conversion
# ============================================================================

class TestSpinOrdering:
    def test_roundtrip(self):
        """interleaved -> block -> interleaved should be identity."""
        n = 3
        M = np.random.randn(2*n, 2*n) + 1j * np.random.randn(2*n, 2*n)
        M_block = _interleaved_to_block_spin(M, n)
        M_back = _block_to_interleaved_spin(M_block, n)
        np.testing.assert_allclose(M_back, M, atol=1e-14)


# ============================================================================
# Tests: Little group
# ============================================================================

class TestLittleGroup:
    def test_gamma_point(self):
        """At Gamma, the little group should be the full point group."""
        lattice = np.eye(3) * 5.0
        positions = np.array([[0.0, 0.0, 0.0]])
        numbers = np.array([1])

        sym_info = detect_symmetry_operations(lattice, positions, numbers)
        k_gamma = np.array([0.0, 0.0, 0.0])

        indices, ops = compute_little_group(k_gamma, sym_info.operations)
        # At Gamma, all rotations leave k=0 invariant
        assert len(indices) == sym_info.nsymm

    def test_general_k(self):
        """A general k-point should have a smaller little group."""
        lattice = np.eye(3) * 5.0
        positions = np.array([[0.0, 0.0, 0.0]])
        numbers = np.array([1])

        sym_info = detect_symmetry_operations(lattice, positions, numbers)
        k_general = np.array([0.123, 0.234, 0.345])

        indices, ops = compute_little_group(k_general, sym_info.operations)
        # A general k should have only identity (or very few ops)
        assert len(indices) < sym_info.nsymm


# ============================================================================
# Tests: Degenerate groups
# ============================================================================

class TestDegenerateGroups:
    def test_no_degeneracy(self):
        """All distinct energies: each band is its own group."""
        eigs = np.array([1.0, 2.0, 3.0, 4.0])
        groups = identify_degenerate_groups(eigs)
        assert len(groups) == 4

    def test_double_degeneracy(self):
        """Two pairs of degenerate bands."""
        eigs = np.array([1.0, 1.0, 3.0, 3.0])
        groups = identify_degenerate_groups(eigs)
        assert len(groups) == 2
        assert len(groups[0]) == 2
        assert len(groups[1]) == 2


# ============================================================================
# Tests: get_orbital_rotation dispatch
# ============================================================================

class TestOrbitalRotationDispatch:
    def test_dispatch_s(self):
        D = get_orbital_rotation('s', np.eye(3))
        assert D.shape == (1, 1)

    def test_dispatch_p(self):
        D = get_orbital_rotation('p', np.eye(3))
        assert D.shape == (3, 3)

    def test_dispatch_d(self):
        D = get_orbital_rotation('d', np.eye(3))
        assert D.shape == (5, 5)

    def test_dispatch_f(self):
        D = get_orbital_rotation('f', np.eye(3))
        assert D.shape == (7, 7)

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            get_orbital_rotation('g', np.eye(3))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
