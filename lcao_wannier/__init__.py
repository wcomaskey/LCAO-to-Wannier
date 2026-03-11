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

__version__ = "1.3.0"
__author__ = "William Comaskey"

# Main engine class
from .engine import Wannier90Engine

# Parser functions
from .parser import (
    parse_overlap_and_fock_matrices,
    parse_calculation_parameters,  # NEW
    parse_atomic_basis_info,  # NEW
    parse_orbital_types,  # NEW
    create_spin_block_matrices,
    create_nonsoc_full_matrices,
    fill_raw_matrix,
    is_hermitian,
    CalculationParameters,  # NEW
    AtomicBasisInfo,  # NEW
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
    read_nnkp_neighbors,
    kpoint_index_to_grid,
    grid_to_kpoint_index,
)

# Fourier transform functions
from .fourier import (
    fourier_transform_to_kspace,
    fourier_transform_vectorized,
    fourier_all_kpoints,
    inverse_fourier_transform,
    compute_phase_factors,
    StackedMatrices,
    stack_real_space_matrices,
)

# Solver functions
from .solver import (
    solve_generalized_eigenvalue_problem,
    solve_kpoint,
    solve_all_kpoints_sequential,
    solve_all_kpoints_parallel,
    solve_all_kpoints_batched,
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
    write_amn_file_pdwf,
    write_mmn_file,
    write_wannier90_files,
    compute_mmn_matrix,
    compute_mmn_direct,
    compute_mmn_lowdin,
    precompute_lowdin_eigenvectors,
    unitarize_mmn,
)

# Win file writer
from .win_file import (
    write_win_file,
    create_win_config_from_engine,
    parse_atoms_from_crystal_output,
    Wannier90WinConfig,
    KPATH_HEXAGONAL_2D,
    KPATH_HEXAGONAL_3D,
    KPATH_SQUARE_2D,
    KPATH_FCC,
    KPATH_BCC,
    KPATH_SIMPLE_CUBIC,
)

# Band structure plotting
from .band_plot import (
    run_band_structure,
    compute_band_structure,
    plot_band_structure,
    text_band_summary,
    generate_kpath,
    detect_lattice_type,
    kpath_from_win_format,
    get_kpath_for_lattice,
    parse_custom_kpath,
    compute_path_projectability,
    KPathSpec,
    KPathResult,
    BandStructureData,
    PlotConfig,
)

from .band_selection import (
    estimate_fermi_energy,
    analyze_band_window,
    print_band_analysis,
    check_frozen_continuity,
    validate_fermi_coverage,
    select_projection_orbitals,
    scdm_select_projections,
    compute_subspace_projections,
    suggest_optimal_window,
    BandWindowResult,
    OrbitalSelectionResult
)

from .orbital_analysis import (
    compute_band_projections,
    compute_band_character,
    identify_dominant_character,
    analyze_all_bands_character,
    format_band_character_table,
    BandCharacter
)

from .projectability import (
    compute_band_projectability,
    select_bands_by_projectability,
    smart_select_bands,
    ProjectabilityResult,
    SmartSelectionResult,
)

# Symmetry module
from .symmetry import (
    detect_symmetry_operations,
    build_representation_matrices,
    symmetrize_real_space_matrices,
    enforce_hermiticity,
    enforce_time_reversal,
    get_orbital_rotation,
    SymmetryOperation,
    SymmetryInfo,
)

# Conditioning validation
from .conditioning import (
    validate_overlap_conditioning,
    OverlapConditioningResult,
    OverlapConditioningError,
)

# Valence configuration (PDWF)
from .valence_config import (
    get_valence_l,
    get_num_target_orbitals,
    compute_num_wann,
    build_target_mask,
    summarize_config,
    VALENCE_CONFIG,
    ELEMENT_SYMBOLS,
    ELEMENT_Z,
)

# Basis parser (PDWF)
from .basis_parser import (
    parse_basis_shells,
    get_atom_list,
    ShellInfo,
)

# LCAO-PDWF core
from .lcao_pdwf import (
    compute_lowdin_projectability,
    compute_matrix_sqrt,
    classify_bands,
    determine_windows,
    check_frozen_interlopers,
    check_band_count,
    print_pdwf_summary,
    ClassificationParams,
    BandClassification,
    WindowParameters,
)

# Irrep module
from .irreps import (
    compute_little_group,
    compute_band_characters,
    identify_band_irreps,
    select_bands_by_symmetry,
    find_high_symmetry_kpoints,
    IrrepResult,
    BandSelectionResult,
)

# Public API
__all__ = [
    # Main class
    'Wannier90Engine',
    
    # Parser
    'parse_overlap_and_fock_matrices',
    'create_spin_block_matrices',
    'create_nonsoc_full_matrices',
    'fill_raw_matrix',
    'is_hermitian',
    'parse_calculation_parameters',
    'parse_atomic_basis_info',
    'parse_orbital_types',
    'CalculationParameters',
    'AtomicBasisInfo',
    
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
    'write_amn_file_pdwf',
    'write_mmn_file',
    'write_wannier90_files',
    'compute_mmn_matrix',
    'compute_mmn_direct',
    'compute_mmn_lowdin',
    'precompute_lowdin_eigenvectors',
    'unitarize_mmn',
    'write_win_file',
    'create_win_config_from_engine',
    'parse_atoms_from_crystal_output',
    'Wannier90WinConfig',
    'KPATH_HEXAGONAL_2D',
    'KPATH_HEXAGONAL_3D',
    'KPATH_SQUARE_2D',
    'KPATH_FCC',
    'KPATH_BCC',
    'KPATH_SIMPLE_CUBIC',

    # Band structure plotting
    'run_band_structure',
    'compute_band_structure',
    'plot_band_structure',
    'text_band_summary',
    'generate_kpath',
    'detect_lattice_type',
    'kpath_from_win_format',
    'get_kpath_for_lattice',
    'parse_custom_kpath',
    'compute_path_projectability',
    'KPathSpec',
    'KPathResult',
    'BandStructureData',
    'PlotConfig',

    # Band selection
    'estimate_fermi_energy',
    'analyze_band_window',
    'print_band_analysis',
    'check_frozen_continuity',
    'validate_fermi_coverage',
    'select_projection_orbitals',
    'scdm_select_projections',
    'compute_subspace_projections',
    'suggest_optimal_window',
    'BandWindowResult',
    'OrbitalSelectionResult',

    # Orbital analysis
    'compute_band_projections',
    'compute_band_character',
    'identify_dominant_character',
    'analyze_all_bands_character',
    'format_band_character_table',
    'BandCharacter',

    # Projectability
    'compute_band_projectability',
    'select_bands_by_projectability',
    'smart_select_bands',
    'ProjectabilityResult',
    'SmartSelectionResult',

    # Symmetry
    'detect_symmetry_operations',
    'build_representation_matrices',
    'symmetrize_real_space_matrices',
    'enforce_hermiticity',
    'enforce_time_reversal',
    'get_orbital_rotation',
    'SymmetryOperation',
    'SymmetryInfo',

    # Conditioning
    'validate_overlap_conditioning',
    'OverlapConditioningResult',
    'OverlapConditioningError',

    # Valence config (PDWF)
    'get_valence_l',
    'get_num_target_orbitals',
    'compute_num_wann',
    'build_target_mask',
    'summarize_config',
    'VALENCE_CONFIG',
    'ELEMENT_SYMBOLS',
    'ELEMENT_Z',

    # Basis parser (PDWF)
    'parse_basis_shells',
    'get_atom_list',
    'ShellInfo',

    # LCAO-PDWF core
    'compute_lowdin_projectability',
    'compute_matrix_sqrt',
    'classify_bands',
    'determine_windows',
    'check_frozen_interlopers',
    'check_band_count',
    'print_pdwf_summary',
    'ClassificationParams',
    'BandClassification',
    'WindowParameters',

    # Irreps
    'compute_little_group',
    'compute_band_characters',
    'identify_band_irreps',
    'select_bands_by_symmetry',
    'find_high_symmetry_kpoints',
    'IrrepResult',
    'BandSelectionResult',
]