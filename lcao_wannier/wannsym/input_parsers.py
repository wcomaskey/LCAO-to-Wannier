"""
Legacy Input File Parsers for WannSym

Parses the original WannSym input file formats:
    - poscar.in  : Crystal structure (VASP POSCAR-like format)
    - wann.in    : Wannier orbital definitions
    - locaxis.in : Local coordinate axes

These are needed for standalone CLI usage. When WannSym is called from
lcao_to_wannier90.py, crystal structure comes from the CRYSTAL output parser.

Reference: lib/read_poswan.py from wannhr_symm
"""

import numpy as np
from typing import List, Tuple, Optional

# Orbital dimension mapping
_ORB_DIMS = {
    's': 1, 'p': 3, 'd': 5, 'f': 7,
    't2g': 3, 'pz': 1,
}

# Symbol-to-Z mapping for common elements
_SYMBOL_TO_Z = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
    'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
    'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
    'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
    'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
    'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
    'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57,
    'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64,
    'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71,
    'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78,
    'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85,
    'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92,
}


def parse_poscar(
    filename: str = 'poscar.in'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Parse a poscar.in file (VASP POSCAR-like format used by WannSym).

    Format:
        Line 1-2: Comments
        Lines 3-5: Lattice vectors (as columns in original, stored as rows here)
        Lines 6-7: Comments
        Line 8: natom_pos
        Lines 9+: frac_x frac_y frac_z symbol

    Parameters
    ----------
    filename : str
        Path to poscar.in

    Returns
    -------
    lattice_vectors : ndarray (3, 3)
        Lattice vectors as rows
    atom_positions_frac : ndarray (natom, 3)
        Fractional coordinates
    atom_numbers : ndarray (natom,)
        Atomic numbers
    atom_symbols : list of str
        Element symbols
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Lines 3-5: Lattice vectors
    # Original WannSym stores them as columns of Amat, then uses Amat.T for spglib
    # We store as rows directly (matching spglib convention)
    Amat_cols = np.zeros((3, 3))
    for i in range(3):
        vals = lines[2 + i].strip().split()
        Amat_cols[:, i] = [float(v) for v in vals[:3]]

    # Convert to rows (spglib convention: rows = lattice vectors)
    lattice_vectors = Amat_cols.T

    # Line 8: number of atoms
    natom = int(lines[7].strip().split()[0])

    # Lines 9+: fractional coordinates and symbols
    positions = np.zeros((natom, 3))
    symbols = []
    for iatom in range(natom):
        parts = lines[8 + iatom].strip().split()
        positions[iatom] = [float(parts[0]), float(parts[1]), float(parts[2])]
        symbols.append(parts[3])

    atom_numbers = np.array([_SYMBOL_TO_Z.get(s, 0) for s in symbols])

    return lattice_vectors, positions, atom_numbers, symbols


def parse_wann_in(
    filename: str = 'wann.in'
) -> Tuple[List[List[str]], bool, List[int]]:
    """
    Parse a wann.in file (Wannier orbital definitions).

    Format:
        Lines 1-23: Header/comments
        Line 24: natom_wan (number of atoms with Wannier projections)
        Line 25: ispinor flag (T/F)
        Lines 26+: atom_index orbital_type1 orbital_type2 ...

    Parameters
    ----------
    filename : str
        Path to wann.in

    Returns
    -------
    orbital_types_per_atom : list of list of str
        Shell types for each Wannier atom
    is_spinor : bool
        Whether SOC is present
    atom_indices : list of int
        1-indexed POSCAR atom indices for each Wannier atom
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Line 24 (0-indexed: 23): number of Wannier atoms
    natom_wan = int(lines[23].strip().split()[0])

    # Line 25 (0-indexed: 24): spinor flag
    spinor_str = lines[24].strip().split()[0].upper()
    is_spinor = spinor_str in ('T', 'TRUE', '.TRUE.')

    # Lines 26+ (0-indexed: 25+): atom definitions
    orbital_types_per_atom = []
    atom_indices = []

    for iw in range(natom_wan):
        parts = lines[25 + iw].strip().split()
        atom_idx = int(parts[0])  # 1-indexed
        atom_indices.append(atom_idx)

        orb_types = []
        for orb_str in parts[1:]:
            orb_str = orb_str.lower()
            if orb_str in _ORB_DIMS:
                orb_types.append(orb_str)
            else:
                raise ValueError(
                    f"Unknown orbital type '{orb_str}' for atom {atom_idx}. "
                    f"Supported: {list(_ORB_DIMS.keys())}"
                )

        orbital_types_per_atom.append(orb_types)

    return orbital_types_per_atom, is_spinor, atom_indices


def parse_locaxis(
    filename: str = 'locaxis.in',
    natom_wan: int = 0
) -> Optional[np.ndarray]:
    """
    Parse a locaxis.in file (local coordinate axes).

    Format:
        Lines 1-19: Header/comments
        Line 20: number of atoms with local axes (nl)
        Lines 21+: atom_id zx zy zz xx xy xz

    Parameters
    ----------
    filename : str
        Path to locaxis.in
    natom_wan : int
        Total number of Wannier atoms (for array sizing)

    Returns
    -------
    local_axes : ndarray (natom_wan, 3, 3) or None
        Local coordinate axes for each atom, or None if file not found/empty.
        axes[iatom] is a 3×3 matrix where rows are [x_axis, y_axis, z_axis].
    """
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        return None

    if len(lines) < 20:
        return None

    # Line 20 (0-indexed: 19): number of atoms with local axes
    nl = int(lines[19].strip().split()[0])

    if nl == 0 or natom_wan == 0:
        return None

    # Default: identity axes for all atoms
    local_axes = np.tile(np.eye(3), (natom_wan, 1, 1))

    for il in range(nl):
        parts = lines[20 + il].strip().split()
        atom_id = int(parts[0]) - 1  # Convert to 0-indexed

        # z-axis
        z_axis = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
        z_axis = z_axis / np.linalg.norm(z_axis)

        # x-axis
        x_axis = np.array([float(parts[4]), float(parts[5]), float(parts[6])])
        x_axis = x_axis / np.linalg.norm(x_axis)

        # y-axis from cross product (right-hand rule)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        # Store as rows: [x, y, z]
        local_axes[atom_id, 0] = x_axis
        local_axes[atom_id, 1] = y_axis
        local_axes[atom_id, 2] = z_axis

    return local_axes
