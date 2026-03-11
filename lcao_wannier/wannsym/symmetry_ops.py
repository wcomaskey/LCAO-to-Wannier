"""
Symmetry Operations Detection and Representation Matrix Construction

Detects crystal symmetry operations using spglib, then builds the full
orbital+spinor representation matrices needed for the Reynolds operator.

The two-step orbital rotation approach:
    1. Global rotation in Cartesian coordinates
    2. Local frame transformation for atoms with non-standard axes

For SOC systems, the representation matrix is:
    protmat = kron(D_orbital, D_spinor)
converted from interleaved to block spin ordering [all_up | all_dn].

Reference: lib/get_symmop.py and lib/get_point_group_rotmat_twostep.py
from wannhr_symm (Changming Yue, arxiv:1805.12148)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .orbital_rotations import get_orbital_rotation, get_orbital_dim
from .spinor import get_spinor_dmatrix


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SymmetryOperation:
    """A single space group symmetry operation."""
    rotation_frac: np.ndarray    # (3, 3) integer rotation in fractional coords
    rotation_cart: np.ndarray    # (3, 3) float rotation in Cartesian coords
    translation: np.ndarray      # (3,) fractional translation vector
    atom_map: np.ndarray         # atom_map[j] = i: atom j maps to atom i
    vec_shift: np.ndarray        # (natom, 3) lattice shifts for each atom


@dataclass
class SymmetryInfo:
    """Complete symmetry information for a crystal structure."""
    operations: List[SymmetryOperation]
    space_group: str
    point_group: str
    lattice_vectors: np.ndarray      # (3, 3) lattice vectors as rows
    atom_positions_frac: np.ndarray  # (natom, 3) fractional positions
    atom_numbers: np.ndarray         # (natom,) atomic numbers

    @property
    def nsymm(self) -> int:
        return len(self.operations)


# ============================================================================
# Symmetry Detection
# ============================================================================

def detect_symmetry(
    lattice_vectors: np.ndarray,
    atom_positions_frac: np.ndarray,
    atom_numbers: np.ndarray,
    tolerance: float = 1e-5
) -> SymmetryInfo:
    """
    Detect all space group symmetry operations using spglib.

    Parameters
    ----------
    lattice_vectors : ndarray (3, 3)
        Lattice vectors as rows
    atom_positions_frac : ndarray (natom, 3)
        Fractional coordinates of atoms
    atom_numbers : ndarray (natom,)
        Atomic numbers
    tolerance : float
        Symmetry detection tolerance for spglib

    Returns
    -------
    SymmetryInfo
        Complete symmetry information including all operations
    """
    try:
        import spglib
    except ImportError:
        raise ImportError(
            "spglib is required for symmetry detection. "
            "Install with: pip install spglib"
        )

    cell = (lattice_vectors, atom_positions_frac, atom_numbers)

    # Space group
    spacegroup = spglib.get_spacegroup(cell, symprec=tolerance)
    if spacegroup is None:
        raise RuntimeError("spglib failed to determine space group")

    # Symmetry operations
    symmetry = spglib.get_symmetry(cell, symprec=tolerance)
    if symmetry is None:
        raise RuntimeError("spglib failed to find symmetry operations")

    ptgrp_info = spglib.get_pointgroup(symmetry['rotations'])
    point_group = ptgrp_info[0] if ptgrp_info else "unknown"

    rotations = symmetry['rotations']       # (nsymm, 3, 3) integer
    translations = symmetry['translations']  # (nsymm, 3) float
    nsymm = rotations.shape[0]

    A = lattice_vectors           # rows = lattice vectors
    A_inv = np.linalg.inv(A)
    natom = len(atom_numbers)

    operations = []
    for isym in range(nsymm):
        rot_frac = rotations[isym]
        trans = translations[isym]

        # Cartesian rotation: R_cart = A^T @ R_frac @ (A^T)^{-1}
        rot_cart = A.T @ rot_frac.astype(float) @ A_inv.T
        rot_cart = _clean_rotation_matrix(rot_cart)

        # Atom mapping: for each atom j, find where it maps under this op
        atom_map = np.zeros(natom, dtype=int)
        vec_shift = np.zeros((natom, 3), dtype=float)

        for j in range(natom):
            r_new = rot_frac @ atom_positions_frac[j] + trans
            found = False
            for i in range(natom):
                if atom_numbers[i] != atom_numbers[j]:
                    continue
                diff = r_new - atom_positions_frac[i]
                diff_rounded = diff - np.round(diff)
                if np.linalg.norm(diff_rounded) < tolerance * 10:
                    atom_map[j] = i
                    vec_shift[j] = np.round(r_new - atom_positions_frac[i])
                    found = True
                    break

            if not found:
                raise RuntimeError(
                    f"Symmetry operation {isym}: could not find image of atom {j}"
                )

        operations.append(SymmetryOperation(
            rotation_frac=rot_frac.astype(float),
            rotation_cart=rot_cart,
            translation=trans,
            atom_map=atom_map,
            vec_shift=vec_shift,
        ))

    return SymmetryInfo(
        operations=operations,
        space_group=spacegroup,
        point_group=point_group,
        lattice_vectors=lattice_vectors,
        atom_positions_frac=atom_positions_frac,
        atom_numbers=atom_numbers,
    )


def _clean_rotation_matrix(R: np.ndarray, tol: float = 0.01) -> np.ndarray:
    """
    Clean up numerical noise in rotation matrix entries.

    Rotation matrices for crystals have entries from:
    {0, ±1/2, ±√2/2, ±√3/2, ±1}
    """
    special_values = [0.0, 0.5, np.sqrt(2)/2, np.sqrt(3)/2, 1.0]
    R_clean = R.copy()
    for i in range(3):
        for j in range(3):
            val = R[i, j]
            for sv in special_values:
                if abs(abs(val) - sv) < tol:
                    R_clean[i, j] = np.sign(val) * sv if sv > 0 else 0.0
                    break
    return R_clean


# ============================================================================
# Representation Matrix Construction
# ============================================================================

def build_representation_matrices(
    sym_info: SymmetryInfo,
    orbital_types_per_atom: List[List[str]],
    has_soc: bool = False,
    local_axes: Optional[np.ndarray] = None
) -> List[np.ndarray]:
    """
    Build the full representation (protmat) matrix for each symmetry operation.

    The two-step rotation procedure:
        1. R_global = A^T @ R_frac @ A^{-T}  (fractional → Cartesian)
        2. R_local = axis_i^T @ R_global @ axis_j  (local frame transform)
        3. D_orb = orbital_rotation(type, R_local)  (orbital representation)
        4. For SOC: protmat = kron(D_orb, D_spinor) → block spin ordering

    Parameters
    ----------
    sym_info : SymmetryInfo
        Symmetry operations
    orbital_types_per_atom : list of list of str
        Shell types for each atom, e.g., [['s', 's', 'p', 'p', 'd']]
    has_soc : bool
        Whether spin-orbit coupling is present
    local_axes : ndarray (natom, 3, 3), optional
        Local coordinate axes for each atom. Default: identity (global axes).

    Returns
    -------
    list of ndarray
        One representation matrix per symmetry operation.
        Shape: (norbs, norbs) without SOC, (2*norbs, 2*norbs) with SOC.
    """
    natom = len(orbital_types_per_atom)

    # Compute orbital counts and offsets
    norbs_per_atom = []
    for atom_orbs in orbital_types_per_atom:
        n = sum(get_orbital_dim(t) for t in atom_orbs)
        norbs_per_atom.append(n)

    norbs_spatial = sum(norbs_per_atom)

    if local_axes is None:
        local_axes = np.tile(np.eye(3), (natom, 1, 1))

    offsets = np.zeros(natom, dtype=int)
    for i in range(1, natom):
        offsets[i] = offsets[i-1] + norbs_per_atom[i-1]

    protmat_list = []

    for op in sym_info.operations:
        R_cart = op.rotation_cart

        # Build orbital-only (no spin) representation matrix
        prot_orb = np.zeros((norbs_spatial, norbs_spatial), dtype=np.complex128)

        for jatom in range(natom):
            iatom = op.atom_map[jatom]

            # Two-step local rotation
            axs_j = local_axes[jatom]
            axs_i = local_axes[iatom]
            R_local = axs_i.T @ R_cart @ axs_j

            off_j = offsets[jatom]
            off_i = offsets[iatom]

            block_offset = 0
            for shell_type in orbital_types_per_atom[jatom]:
                dim = get_orbital_dim(shell_type)
                D_orb = get_orbital_rotation(shell_type, R_local)
                prot_orb[
                    off_i + block_offset : off_i + block_offset + dim,
                    off_j + block_offset : off_j + block_offset + dim
                ] = D_orb
                block_offset += dim

        if has_soc:
            # SOC: protmat = kron(D_orb, D_spinor) in interleaved ordering
            dmat = get_spinor_dmatrix(R_cart)
            protmat_interleaved = np.kron(prot_orb, dmat)

            # Convert to block spin ordering: [all_up | all_dn]
            protmat = _interleaved_to_block_spin(protmat_interleaved, norbs_spatial)
        else:
            protmat = prot_orb.real if np.allclose(prot_orb.imag, 0) else prot_orb

        protmat_list.append(protmat)

    return protmat_list


def compute_orbital_info(
    orbital_types_per_atom: List[List[str]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute orbital offsets and counts from orbital type lists.

    Parameters
    ----------
    orbital_types_per_atom : list of list of str
        Shell types per atom

    Returns
    -------
    offsets : ndarray (natom,)
        Starting index of each atom's orbitals
    counts : ndarray (natom,)
        Number of orbitals per atom
    """
    natom = len(orbital_types_per_atom)
    counts = np.zeros(natom, dtype=int)
    for i, atom_orbs in enumerate(orbital_types_per_atom):
        counts[i] = sum(get_orbital_dim(t) for t in atom_orbs)

    offsets = np.zeros(natom, dtype=int)
    for i in range(1, natom):
        offsets[i] = offsets[i-1] + counts[i-1]

    return offsets, counts


# ============================================================================
# Spin Ordering Conversions
# ============================================================================

def _interleaved_to_block_spin(M_interleaved: np.ndarray, norbs: int) -> np.ndarray:
    """
    Convert matrix from interleaved spin ordering (orb1_up, orb1_dn, orb2_up, ...)
    to block spin ordering ([all_up | all_dn]).
    """
    n2 = 2 * norbs
    M_block = np.zeros((n2, n2), dtype=M_interleaved.dtype)
    M_block[:norbs, :norbs] = M_interleaved[0::2, 0::2]     # up-up
    M_block[:norbs, norbs:] = M_interleaved[0::2, 1::2]     # up-dn
    M_block[norbs:, :norbs] = M_interleaved[1::2, 0::2]     # dn-up
    M_block[norbs:, norbs:] = M_interleaved[1::2, 1::2]     # dn-dn
    return M_block


def _block_to_interleaved_spin(M_block: np.ndarray, norbs: int) -> np.ndarray:
    """
    Convert matrix from block spin ordering to interleaved spin ordering.
    Inverse of _interleaved_to_block_spin.
    """
    n2 = 2 * norbs
    M_interleaved = np.zeros((n2, n2), dtype=M_block.dtype)
    M_interleaved[0::2, 0::2] = M_block[:norbs, :norbs]
    M_interleaved[0::2, 1::2] = M_block[:norbs, norbs:]
    M_interleaved[1::2, 0::2] = M_block[norbs:, :norbs]
    M_interleaved[1::2, 1::2] = M_block[norbs:, norbs:]
    return M_interleaved
