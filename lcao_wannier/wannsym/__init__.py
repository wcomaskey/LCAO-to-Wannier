"""
WannSym: Post-Wannierization Hamiltonian Symmetrization

Symmetrizes tight-binding Hamiltonians from Wannier90 output using the
Reynolds operator (group averaging). Supports spin-orbit coupling (SOC)
and arbitrary orbital types (s, p, d, f, t2g, pz).

Refactored from wannhr_symm by Changming Yue (arxiv:1805.12148).
Python 3 rewrite with dict-based storage (no memory limits), numerical
orbital rotations (no SymPy dependency), and full type annotations.

Usage (standalone):
    python -m wannsym wannier90_hr.dat --poscar poscar.in --wann wann.in

Usage (as library):
    from wannsym import symmetrize_hr, HamiltonianData, WannSymConfig

    hr = HamiltonianData.from_file('wannier90_hr.dat')
    hr_sym, result = symmetrize_hr(
        hr, lattice, positions, atom_numbers,
        orbital_types, has_soc=True,
    )
    hr_sym.to_file('wannier90_hr.dat_symm')

Dependencies:
    numpy, spglib
"""

__version__ = "1.3.0"
__author__ = "William Comaskey"
__original_author__ = "Changming Yue (yuechangming10@gmail.com)"

from .symmetrize import symmetrize_hr, WannSymConfig, WannSymResult
from .hamiltonian import HamiltonianData
from .symmetry_ops import (
    detect_symmetry,
    build_representation_matrices,
    compute_orbital_info,
    SymmetryInfo,
    SymmetryOperation,
)
from .orbital_rotations import get_orbital_rotation, get_orbital_dim
from .spinor import get_spinor_dmatrix, rmat_to_euler, spinor_dmatrix

__all__ = [
    # Main API
    'symmetrize_hr',
    'WannSymConfig',
    'WannSymResult',
    'HamiltonianData',

    # Symmetry
    'detect_symmetry',
    'build_representation_matrices',
    'compute_orbital_info',
    'SymmetryInfo',
    'SymmetryOperation',

    # Orbital rotations
    'get_orbital_rotation',
    'get_orbital_dim',

    # Spinor
    'get_spinor_dmatrix',
    'rmat_to_euler',
    'spinor_dmatrix',
]
