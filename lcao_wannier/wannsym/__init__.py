"""
WannSymm: Symmetrization of Wannier tight-binding Hamiltonians.

Original developer: Changming Yue, yuechangming10@gmail.com
Python 3 port: faithful reproduction of the original code.

Usage:
    python -m lcao_wannier.wannsym [workdir]

    Run from a directory containing:
        poscar.in       - Crystal structure (lattice vectors + fractional coordinates)
        wann.in         - Wannier projection specification
        locaxis.in      - Local axis definitions
        wannier90_hr.dat - Wannier90 Hamiltonian output

    Output:
        wannier90_hr.dat_nsymmN - Symmetrized Hamiltonian (N = number of symmetry ops)
        mysymmop.dat            - Symmetry operations log
        protmat.log             - Representation matrices log
"""

from .__main__ import main
