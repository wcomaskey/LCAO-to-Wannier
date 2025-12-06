"""
LCAO-to-Wannier90 Package

A Python package for converting LCAO (Linear Combination of Atomic Orbitals)
calculations to Wannier90 format.

Main Components
---------------
Wannier90Engine : class
    Main computational engine for LCAO-to-Wannier90 conversion

Parser Functions
----------------
parse_overlap_and_fock_matrices : function
    Parse CRYSTAL/LCAO output files
create_spin_block_matrices : function
    Create 2Nx2N spin-block matrices using Global Pair Symmetry Construction

Utility Functions
-----------------
prepare_real_space_matrices : function
    Convert parsed matrices to engine-compatible format
organize_matrices_by_lattice_vector : function
    Organize raw matrices by lattice vector and spin channel
"""

__version__ = "1.1.1"
__author__ = "William Comaskey"

# Main engine class
from .engine import Wannier90Engine

# Parser functions
from .parser import (
    parse_overlap_and_fock_matrices,
    create_spin_block_matrices,
    fill_raw_matrix,
    is_hermitian,
)

# Utility functions
from .utils import (
    prepare_real_space_matrices,
    organize_matrices_by_lattice_vector,
    get_basis_size,
    verify_matrix_symmetry,
    check_matrix_consistency,
    print_matrix_summary,
    print_calculation_info,
)

# K-point functions
from .kpoints import (
    generate_kpoint_grid,
    generate_neighbor_list,
    kpoint_index_to_grid,
    grid_to_kpoint_index,
)

# Fourier transform functions
from .fourier import (
    fourier_transform_to_kspace,
    inverse_fourier_transform,
    compute_phase_factors,
)

# Solver functions
from .solver import (
    solve_generalized_eigenvalue_problem,
    solve_kpoint,
    solve_all_kpoints_sequential,
    solve_all_kpoints_parallel,
)

# Verification functions
from .verification import (
    verify_real_space_symmetry,
    verify_hermiticity,
    verify_orthonormality,
    verify_eigenvalue_sorting,
    verify_energy_range,
    run_all_verifications,
)

# Wannier90 file writers
from .wannier90 import (
    write_eig_file,
    write_amn_file,
    write_mmn_file,
    write_wannier90_files,
)

from .band_selection import (
    estimate_fermi_energy,
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

# Public API
__all__ = [
    # Main class
    'Wannier90Engine',
    
    # Parser
    'parse_overlap_and_fock_matrices',
    'create_spin_block_matrices',
    'fill_raw_matrix',
    'is_hermitian',
    
    # Utils
    'prepare_real_space_matrices',
    'organize_matrices_by_lattice_vector',
    'get_basis_size',
    'verify_matrix_symmetry',
    'check_matrix_consistency',
    'print_matrix_summary',
    'print_calculation_info',
    
    # K-points
    'generate_kpoint_grid',
    'generate_neighbor_list',
    'kpoint_index_to_grid',
    'grid_to_kpoint_index',
    
    # Fourier
    'fourier_transform_to_kspace',
    'inverse_fourier_transform',
    'compute_phase_factors',
    
    # Solver
    'solve_generalized_eigenvalue_problem',
    'solve_kpoint',
    'solve_all_kpoints_sequential',
    'solve_all_kpoints_parallel',
    
    # Verification
    'verify_real_space_symmetry',
    'verify_hermiticity',
    'verify_orthonormality',
    'verify_eigenvalue_sorting',
    'verify_energy_range',
    'run_all_verifications',
    
    # Wannier90
    'write_eig_file',
    'write_amn_file',
    'write_mmn_file',
    'write_wannier90_files',
]