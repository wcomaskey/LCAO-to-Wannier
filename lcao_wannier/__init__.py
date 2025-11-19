"""
LCAO to Wannier90 Conversion Package

A professional computational engine for converting LCAO calculations
to Wannier90 input format.

Authors: Computational Materials Science Team
License: MIT
Version: 1.0.0
"""

from .engine import Wannier90Engine
from .parser import (
    parse_overlap_and_fock_matrices,
    create_spin_block_matrices,
)
from .utils import prepare_real_space_matrices
from .kpoints import generate_kpoint_grid, generate_neighbor_list
from .fourier import fourier_transform_to_kspace
from .verification import verify_hermiticity, verify_orthonormality

__version__ = "1.0.0"
__author__ = "Computational Materials Science Team"
__all__ = [
    "Wannier90Engine",
    "parse_overlap_and_fock_matrices",
    "create_spin_block_matrices",
    "prepare_real_space_matrices",
    "generate_kpoint_grid",
    "generate_neighbor_list",
    "fourier_transform_to_kspace",
    "verify_hermiticity",
    "verify_orthonormality",
]
