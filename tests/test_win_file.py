#!/usr/bin/env python3
"""
Tests for Wannier90 .win file generation module.

Tests cover:
- Wannier90WinConfig validation
- .win file writing and format
- Integration with Wannier90Engine
- Atomic position parsing
- High-symmetry k-point paths
"""

import os
import sys
import tempfile
import numpy as np
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lcao_wannier.win_file import (
    Wannier90WinConfig,
    write_win_file,
    create_win_config_from_engine,
    parse_atoms_from_crystal_output,
    KPATH_HEXAGONAL_2D,
    KPATH_SQUARE_2D,
)
from lcao_wannier import Wannier90Engine


class TestWannier90WinConfig:
    """Test Wannier90WinConfig dataclass."""

    def test_valid_config(self):
        """Test creation of valid configuration."""
        config = Wannier90WinConfig(
            num_wann=10,
            num_bands=10,
            mp_grid=(4, 4, 4),
            lattice_vectors=np.eye(3) * 5.0,
            kpoints=np.random.rand(64, 3)
        )

        assert config.num_wann == 10
        assert config.num_bands == 10
        assert config.mp_grid == (4, 4, 4)
        assert config.lattice_vectors.shape == (3, 3)
        assert len(config.kpoints) == 64

    def test_num_wann_exceeds_num_bands(self):
        """Test that num_wann > num_bands raises error."""
        with pytest.raises(ValueError, match="num_wann.*cannot exceed.*num_bands"):
            Wannier90WinConfig(
                num_wann=15,
                num_bands=10,
                mp_grid=(4, 4, 4),
                lattice_vectors=np.eye(3) * 5.0,
                kpoints=np.random.rand(64, 3)
            )

    def test_invalid_lattice_shape(self):
        """Test that invalid lattice shape raises error."""
        with pytest.raises(ValueError, match="lattice_vectors must be shape"):
            Wannier90WinConfig(
                num_wann=10,
                num_bands=10,
                mp_grid=(4, 4, 4),
                lattice_vectors=np.eye(4),  # Wrong shape
                kpoints=np.random.rand(64, 3)
            )

    def test_invalid_kpoints_shape(self):
        """Test that invalid kpoints shape raises error."""
        with pytest.raises(ValueError, match="kpoints must be shape"):
            Wannier90WinConfig(
                num_wann=10,
                num_bands=10,
                mp_grid=(4, 4, 4),
                lattice_vectors=np.eye(3) * 5.0,
                kpoints=np.random.rand(64, 4)  # Wrong number of columns
            )

    def test_kpoints_count_mismatch(self):
        """Test that kpoints count must match mp_grid product."""
        with pytest.raises(ValueError, match="Number of k-points.*does not match"):
            Wannier90WinConfig(
                num_wann=10,
                num_bands=10,
                mp_grid=(4, 4, 4),
                lattice_vectors=np.eye(3) * 5.0,
                kpoints=np.random.rand(50, 3)  # Should be 64
            )

    def test_disentanglement_window_validation(self):
        """Test energy window validation for disentanglement."""
        # num_bands > num_wann requires windows
        with pytest.raises(ValueError, match="dis_win_min and dis_win_max must be specified"):
            Wannier90WinConfig(
                num_wann=10,
                num_bands=20,  # Disentanglement mode
                mp_grid=(4, 4, 4),
                lattice_vectors=np.eye(3) * 5.0,
                kpoints=np.random.rand(64, 3)
            )

    def test_window_ordering(self):
        """Test that window min < max."""
        with pytest.raises(ValueError, match="dis_win_min.*must be <.*dis_win_max"):
            Wannier90WinConfig(
                num_wann=10,
                num_bands=20,
                mp_grid=(4, 4, 4),
                lattice_vectors=np.eye(3) * 5.0,
                kpoints=np.random.rand(64, 3),
                dis_win_min=5.0,
                dis_win_max=3.0  # Invalid
            )

    def test_frozen_window_within_outer(self):
        """Test that frozen window must be within outer window."""
        with pytest.raises(ValueError, match="Frozen window must be within outer window"):
            Wannier90WinConfig(
                num_wann=10,
                num_bands=20,
                mp_grid=(4, 4, 4),
                lattice_vectors=np.eye(3) * 5.0,
                kpoints=np.random.rand(64, 3),
                dis_win_min=-3.0,
                dis_win_max=5.0,
                dis_froz_min=-4.0,  # Outside outer window
                dis_froz_max=4.0
            )


class TestWriteWinFile:
    """Test .win file writing."""

    def test_basic_win_file(self):
        """Test writing basic .win file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            seedname = os.path.join(tmpdir, 'test')

            config = Wannier90WinConfig(
                num_wann=10,
                num_bands=10,
                mp_grid=(4, 4, 4),
                lattice_vectors=np.eye(3) * 5.0,
                kpoints=np.random.rand(64, 3)
            )

            filepath = write_win_file(seedname, config, verbose=False)

            assert os.path.exists(filepath)
            assert filepath == f"{seedname}.win"

            # Read and verify content
            with open(filepath, 'r') as f:
                content = f.read()

            assert 'num_wann = 10' in content
            assert 'num_bands = 10' in content
            assert 'mp_grid = 4 4 4' in content
            assert 'begin unit_cell_cart' in content
            assert 'begin kpoints' in content

    def test_win_file_with_atoms(self):
        """Test .win file with atomic positions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            seedname = os.path.join(tmpdir, 'test_atoms')

            atoms = [
                ('C', np.array([0.0, 0.0, 0.0])),
                ('C', np.array([0.25, 0.25, 0.25])),
            ]

            config = Wannier90WinConfig(
                num_wann=8,
                num_bands=8,
                mp_grid=(4, 4, 4),
                lattice_vectors=np.eye(3) * 3.57,
                kpoints=np.random.rand(64, 3),
                atoms=atoms
            )

            filepath = write_win_file(seedname, config, verbose=False)

            with open(filepath, 'r') as f:
                content = f.read()

            assert 'begin atoms_frac' in content
            assert 'C' in content

    def test_win_file_with_projections(self):
        """Test .win file with projections."""
        with tempfile.TemporaryDirectory() as tmpdir:
            seedname = os.path.join(tmpdir, 'test_proj')

            projections = ['C:sp3', 'C:sp3']

            config = Wannier90WinConfig(
                num_wann=8,
                num_bands=8,
                mp_grid=(4, 4, 4),
                lattice_vectors=np.eye(3) * 3.57,
                kpoints=np.random.rand(64, 3),
                projections=projections
            )

            filepath = write_win_file(seedname, config, verbose=False)

            with open(filepath, 'r') as f:
                content = f.read()

            assert 'begin projections' in content
            assert 'C:sp3' in content

    def test_win_file_with_spinors(self):
        """Test .win file with spinor flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            seedname = os.path.join(tmpdir, 'test_spinor')

            config = Wannier90WinConfig(
                num_wann=10,
                num_bands=10,
                mp_grid=(4, 4, 4),
                lattice_vectors=np.eye(3) * 5.0,
                kpoints=np.random.rand(64, 3),
                spinors=True
            )

            filepath = write_win_file(seedname, config, verbose=False)

            with open(filepath, 'r') as f:
                content = f.read()

            assert 'spinors = .true.' in content

    def test_win_file_with_disentanglement(self):
        """Test .win file with disentanglement windows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            seedname = os.path.join(tmpdir, 'test_disen')

            config = Wannier90WinConfig(
                num_wann=10,
                num_bands=20,
                mp_grid=(4, 4, 4),
                lattice_vectors=np.eye(3) * 5.0,
                kpoints=np.random.rand(64, 3),
                dis_win_min=-3.0,
                dis_win_max=5.0,
                dis_froz_min=-2.0,
                dis_froz_max=4.0
            )

            filepath = write_win_file(seedname, config, verbose=False)

            with open(filepath, 'r') as f:
                content = f.read()

            assert 'DISENTANGLEMENT' in content
            assert 'dis_win_min' in content
            assert 'dis_froz_min' in content

    def test_win_file_with_kpath(self):
        """Test .win file with band structure path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            seedname = os.path.join(tmpdir, 'test_kpath')

            config = Wannier90WinConfig(
                num_wann=10,
                num_bands=10,
                mp_grid=(8, 8, 1),
                lattice_vectors=np.eye(3) * 5.0,
                kpoints=np.random.rand(64, 3),
                bands_plot=True,
                kpoint_path=KPATH_HEXAGONAL_2D
            )

            filepath = write_win_file(seedname, config, verbose=False)

            with open(filepath, 'r') as f:
                content = f.read()

            assert 'begin kpoint_path' in content
            assert 'bands_plot = T' in content


class TestEngineIntegration:
    """Test integration with Wannier90Engine."""

    def setup_method(self):
        """Set up test engine."""
        # Create simple tight-binding model
        self.lattice = np.eye(3) * 5.0
        self.real_space_matrices = {}

        # On-site
        self.real_space_matrices[(0, 0, 0)] = {
            'H': np.diag([0.0, 1.0, 2.0, 3.0]) + 0j,
            'S': np.eye(4) + 0j
        }

        # Nearest neighbors
        t = -0.5
        for R in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
            self.real_space_matrices[R] = {
                'H': t * np.eye(4) + 0j,
                'S': 0.1 * np.eye(4) + 0j
            }

    def test_create_config_from_engine(self):
        """Test creating WinConfig from engine."""
        engine = Wannier90Engine(
            real_space_matrices=self.real_space_matrices,
            k_grid=(4, 4, 4),
            lattice_vectors=self.lattice,
            num_wann=4,
            seedname='test'
        )

        # Must solve first
        engine.solve_all_kpoints(parallel=False)

        # Create config
        config = create_win_config_from_engine(engine, spinors=False)

        assert config.num_wann == 4
        assert config.num_bands == 4
        assert config.mp_grid == (4, 4, 4)
        assert np.allclose(config.lattice_vectors, self.lattice)
        assert len(config.kpoints) == 64

    def test_create_config_before_solving(self):
        """Test that creating config before solving raises error."""
        engine = Wannier90Engine(
            real_space_matrices=self.real_space_matrices,
            k_grid=(4, 4, 4),
            lattice_vectors=self.lattice,
            num_wann=4,
            seedname='test'
        )

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="has not solved eigenvalue problems"):
            create_win_config_from_engine(engine)

    def test_write_files_integration(self):
        """Test engine.write_files() generates .win file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            seedname = os.path.join(tmpdir, 'integration_test')

            engine = Wannier90Engine(
                real_space_matrices=self.real_space_matrices,
                k_grid=(4, 4, 4),
                lattice_vectors=self.lattice,
                num_wann=4,
                seedname=seedname
            )

            engine.solve_all_kpoints(parallel=False)
            engine.write_files(verbose=False, write_win=True, spinors=False)

            # Check all files exist
            assert os.path.exists(f"{seedname}.win")
            assert os.path.exists(f"{seedname}.eig")
            assert os.path.exists(f"{seedname}.amn")
            assert os.path.exists(f"{seedname}.mmn")

    def test_write_files_with_custom_params(self):
        """Test write_files with custom .win parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            seedname = os.path.join(tmpdir, 'custom_test')

            atoms = [('Si', np.array([0.0, 0.0, 0.0]))]
            projections = ['Si:sp3']

            engine = Wannier90Engine(
                real_space_matrices=self.real_space_matrices,
                k_grid=(4, 4, 4),
                lattice_vectors=self.lattice,
                num_wann=4,
                seedname=seedname
            )

            engine.solve_all_kpoints(parallel=False)
            engine.write_files(
                verbose=False,
                write_win=True,
                atoms=atoms,
                projections=projections,
                spinors=True,
                bands_plot=True,
                kpoint_path=KPATH_SQUARE_2D
            )

            # Verify .win content
            with open(f"{seedname}.win", 'r') as f:
                content = f.read()

            assert 'Si:sp3' in content
            assert 'spinors = .true.' in content
            assert 'begin kpoint_path' in content


class TestAtomParsing:
    """Test atomic position parsing from CRYSTAL output."""

    def test_parse_atoms_basic(self):
        """Test parsing atoms from CRYSTAL-style output."""
        # Mock CRYSTAL output
        lines = [
            " DIRECT LATTICE VECTOR COMPONENTS (ANGSTROM)",
            "   5.000000    0.000000    0.000000",
            "   0.000000    5.000000    0.000000",
            "   0.000000    0.000000    5.000000",
            "",
            " ATOM N.AT.  SHELL    X(A)      Y(A)      Z(A)      EXAD       N.ELECT.",
            " 1  14  SI   10    0.00000000    0.00000000    0.00000000   1.000E-01  14.000",
            " 2  14  SI   10    1.25000000    1.25000000    1.25000000   1.000E-01  14.000",
            " ****",
        ]

        atoms, lattice = parse_atoms_from_crystal_output(lines)

        assert len(atoms) == 2
        assert atoms[0][0] == 'Si'
        assert atoms[1][0] == 'Si'
        assert lattice.shape == (3, 3)
        assert np.allclose(lattice, np.eye(3) * 5.0)


class TestKpointPaths:
    """Test predefined k-point paths."""

    def test_hexagonal_path(self):
        """Test hexagonal 2D k-point path."""
        assert len(KPATH_HEXAGONAL_2D) == 6
        assert KPATH_HEXAGONAL_2D[0][0] == 'G'
        assert KPATH_HEXAGONAL_2D[1][0] == 'M'
        assert KPATH_HEXAGONAL_2D[3][0] == 'K'

    def test_square_path(self):
        """Test square 2D k-point path."""
        assert len(KPATH_SQUARE_2D) == 6
        assert KPATH_SQUARE_2D[0][0] == 'G'
        assert KPATH_SQUARE_2D[1][0] == 'X'
        assert KPATH_SQUARE_2D[3][0] == 'M'


def run_tests():
    """Run all tests."""
    print("Running Wannier90 .win file tests...")

    test_classes = [
        TestWannier90WinConfig,
        TestWriteWinFile,
        TestEngineIntegration,
        TestAtomParsing,
        TestKpointPaths,
    ]

    total_tests = 0
    passed_tests = 0
    failed_tests = []

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        test_instance = test_class()

        # Get test methods
        test_methods = [m for m in dir(test_instance) if m.startswith('test_')]

        for method_name in test_methods:
            total_tests += 1
            try:
                # Run setup if exists
                if hasattr(test_instance, 'setup_method'):
                    test_instance.setup_method()

                # Run test
                method = getattr(test_instance, method_name)
                method()

                print(f"  ✓ {method_name}")
                passed_tests += 1

            except Exception as e:
                print(f"  ✗ {method_name}: {str(e)}")
                failed_tests.append((test_class.__name__, method_name, str(e)))

    # Summary
    print("\n" + "=" * 70)
    print(f"Tests: {total_tests} total, {passed_tests} passed, {len(failed_tests)} failed")

    if failed_tests:
        print("\nFailed tests:")
        for class_name, method_name, error in failed_tests:
            print(f"  {class_name}.{method_name}: {error}")
        return False
    else:
        print("\n✓ All tests passed!")
        return True


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
