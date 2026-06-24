"""
Pre-Wannierization Symmetry Enforcement Module

Detects crystal symmetry operations, builds orbital rotation matrices,
assembles representation matrices, and symmetrizes H(R)/S(R) matrices
BEFORE the Wannier90 pipeline.

Mathematical approach ported from wannhr_symm (Changming Yue, arxiv:1805.12148),
rewritten in Python 3 with numpy (no sympy dependency).

Core symmetrization formula:
    H'(R)_{ij} = (1/N_sym) sum_g  P_gi^dag . H(g.R + dR_gj - dR_gi)_{i'j'} . P_gj

where P_g is the representation matrix (orbital rotation x spinor D-matrix for SOC).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import warnings


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class SymmetryOperation:
    """A single space group symmetry operation."""
    rotation_frac: np.ndarray       # 3x3 integer rotation in fractional coords
    rotation_cart: np.ndarray       # 3x3 float rotation in Cartesian coords
    translation: np.ndarray         # fractional translation vector
    atom_map: np.ndarray            # atom_map[j] = i means atom j maps to atom i
    vec_shift: np.ndarray           # vec_shift[j] = R_j' - R_i in fractional coords


@dataclass
class SymmetryInfo:
    """Complete symmetry information for a crystal structure."""
    operations: List[SymmetryOperation]
    space_group: str
    point_group: str
    lattice_vectors: np.ndarray     # 3x3 lattice vectors (rows)
    atom_positions_frac: np.ndarray # (natom, 3)
    atom_numbers: np.ndarray        # atomic numbers
    nsymm: int = 0

    def __post_init__(self):
        self.nsymm = len(self.operations)


# ============================================================================
# Phase 1: Symmetry Detection
# ============================================================================

def detect_symmetry_operations(
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
        Lattice vectors as rows: [[a1x, a1y, a1z], [a2x, a2y, a2z], [a3x, a3y, a3z]]
    atom_positions_frac : ndarray (natom, 3)
        Fractional coordinates of atoms
    atom_numbers : ndarray (natom,)
        Atomic numbers (used for symmetry identification)
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

    # spglib expects lattice as rows, positions as fractional
    cell = (lattice_vectors, atom_positions_frac, atom_numbers)

    # Get space group info
    spacegroup = spglib.get_spacegroup(cell, symprec=tolerance)
    if spacegroup is None:
        raise RuntimeError("spglib failed to determine space group")

    # Get symmetry operations
    symmetry = spglib.get_symmetry(cell, symprec=tolerance)
    if symmetry is None:
        raise RuntimeError("spglib failed to find symmetry operations")

    ptgrp_info = spglib.get_pointgroup(symmetry['rotations'])
    point_group = ptgrp_info[0] if ptgrp_info else "unknown"

    rotations = symmetry['rotations']      # (nsymm, 3, 3) integer
    translations = symmetry['translations'] # (nsymm, 3) float
    nsymm = rotations.shape[0]

    A = lattice_vectors  # rows are lattice vectors
    A_inv = np.linalg.inv(A)
    natom = len(atom_numbers)

    operations = []
    for isym in range(nsymm):
        rot_frac = rotations[isym]  # 3x3 integer
        trans = translations[isym]  # 3-vector

        # Convert rotation to Cartesian: R_cart = A^T @ R_frac @ (A^T)^{-1}
        # Since A has lattice vectors as rows: r_cart = A^T @ r_frac
        rot_cart = A.T @ rot_frac.astype(float) @ A_inv.T

        # Clean up numerical noise in rotation matrix
        rot_cart = _clean_rotation_matrix(rot_cart)

        # Compute atom mapping: for each atom j, find where it maps under this operation
        atom_map = np.zeros(natom, dtype=int)
        vec_shift = np.zeros((natom, 3), dtype=float)

        for j in range(natom):
            # Apply symmetry operation in fractional coordinates
            r_new = rot_frac @ atom_positions_frac[j] + trans

            # Find which atom i this maps to (modulo lattice)
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
            vec_shift=vec_shift
        ))

    return SymmetryInfo(
        operations=operations,
        space_group=spacegroup,
        point_group=point_group,
        lattice_vectors=lattice_vectors,
        atom_positions_frac=atom_positions_frac,
        atom_numbers=atom_numbers
    )


def _clean_rotation_matrix(R: np.ndarray, tol: float = 0.01) -> np.ndarray:
    """
    Clean up numerical noise in rotation matrix entries.
    Rotation matrices for crystals have entries that are 0, +/-1/2, +/-sqrt(2)/2,
    +/-sqrt(3)/2, or +/-1.
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
# Phase 2: Orbital Rotation Matrices (no sympy)
# ============================================================================

def orbital_rotation_s(R_cart: np.ndarray) -> np.ndarray:
    """s-orbital rotation matrix: always [[1]]."""
    return np.array([[1.0]])


def orbital_rotation_p(R_cart: np.ndarray) -> np.ndarray:
    """
    p-orbital rotation matrix in Wannier90 basis [pz, px, py].

    For rotation R acting on coordinates: r' = R @ r
    The orbital transforms as: P_R f(r) = f(R^{-1} r)
    The representation matrix D[m,n] gives the coefficient of phi_m when
    phi_n is transformed: phi_n(R^{-1}r) = sum_m D[m,n] phi_m(r).

    For p-orbitals in Cartesian basis, D[m,n] = R_inv[cart_n, cart_m],
    which equals R_cart[cart_m, cart_n] for orthogonal R.
    After permuting to Wannier90 ordering [pz, px, py]:
        D = R_cart[ix_(idx, idx)]
    """
    # Cartesian order: x=0, y=1, z=2
    # Wannier90 order: pz=0, px=1, py=2
    # Map: pz->z(2), px->x(0), py->y(1)
    idx = [2, 0, 1]  # [z, x, y] indices in Cartesian
    D = R_cart[np.ix_(idx, idx)]
    return D


def orbital_rotation_d(R_cart: np.ndarray) -> np.ndarray:
    """
    d-orbital rotation matrix in Wannier90 basis [dz2, dxz, dyz, dx2-y2, dxy].

    Uses numerical polynomial evaluation to extract rotation coefficients,
    replacing the sympy approach from wannhr_symm.
    """
    R_inv = np.linalg.inv(R_cart)
    D = np.zeros((5, 5))

    # For each original orbital j, we need to find how it decomposes after rotation
    # The rotated orbital is phi_j(R^{-1} r)
    # We express this in terms of the d-orbital basis

    # Use a set of test directions to build the transformation
    # d-orbitals as functions of (x,y,z):
    #   dz2     = (2z^2 - x^2 - y^2) / (2*sqrt(3))
    #   dxz     = xz
    #   dyz     = yz
    #   dx2-y2  = (x^2 - y^2) / 2
    #   dxy     = xy

    # Generate test points on a unit sphere — 6 points along axes + extras
    test_points = _generate_test_points_quadratic()

    # Evaluate d-orbitals at test points → basis matrix B
    B = np.zeros((len(test_points), 5))
    for ip, (x, y, z) in enumerate(test_points):
        B[ip] = _d_orbitals(x, y, z)

    # For each original orbital j, evaluate it at R^{-1} @ test_points
    for j in range(5):
        vals = np.zeros(len(test_points))
        for ip, p in enumerate(test_points):
            rp = R_inv @ p
            all_d = _d_orbitals(rp[0], rp[1], rp[2])
            vals[ip] = all_d[j]

        # Solve B @ D[:,j] = vals for the coefficients
        D[:, j], _, _, _ = np.linalg.lstsq(B, vals, rcond=None)

    # Clean small values
    D[np.abs(D) < 1e-10] = 0.0
    return D


def orbital_rotation_f(R_cart: np.ndarray) -> np.ndarray:
    """
    f-orbital rotation matrix in Wannier90 basis:
    [fz3, fxz2, fyz2, fz(x2-y2), fxyz, fx(x2-3y2), fy(3x2-y2)].
    """
    R_inv = np.linalg.inv(R_cart)
    D = np.zeros((7, 7))

    test_points = _generate_test_points_cubic()
    B = np.zeros((len(test_points), 7))
    for ip, (x, y, z) in enumerate(test_points):
        B[ip] = _f_orbitals(x, y, z)

    for j in range(7):
        vals = np.zeros(len(test_points))
        for ip, p in enumerate(test_points):
            rp = R_inv @ p
            all_f = _f_orbitals(rp[0], rp[1], rp[2])
            vals[ip] = all_f[j]
        D[:, j], _, _, _ = np.linalg.lstsq(B, vals, rcond=None)

    D[np.abs(D) < 1e-10] = 0.0
    return D


def _d_orbitals(x, y, z):
    """Evaluate the 5 real d-orbitals at point (x,y,z)."""
    sq3 = np.sqrt(3.0)
    return np.array([
        (2*z*z - x*x - y*y) / (2*sq3),  # dz2
        x*z,                              # dxz
        y*z,                              # dyz
        (x*x - y*y) / 2.0,               # dx2-y2
        x*y,                              # dxy
    ])


def _f_orbitals(x, y, z):
    """Evaluate the 7 real f-orbitals at point (x,y,z)."""
    sq15 = np.sqrt(15.0)
    sq10 = np.sqrt(10.0)
    sq6 = np.sqrt(6.0)
    return np.array([
        z*(2*z*z - 3*x*x - 3*y*y) / (2*sq15),   # fz3
        x*(4*z*z - x*x - y*y) / (2*sq10),         # fxz2
        y*(4*z*z - x*x - y*y) / (2*sq10),         # fyz2
        z*(x*x - y*y) / 2.0,                       # fz(x2-y2)
        x*y*z,                                      # fxyz
        x*(x*x - 3*y*y) / (2*sq6),                 # fx(x2-3y2)
        y*(3*x*x - y*y) / (2*sq6),                 # fy(3x2-y2)
    ])


def _generate_test_points_quadratic():
    """Generate test points for fitting quadratic (d-orbital) coefficients."""
    pts = []
    # Use points on a sphere that sample all quadratic terms well
    vals = [-1.0, -0.5, 0.0, 0.5, 1.0]
    for x in vals:
        for y in vals:
            for z in vals:
                r = np.sqrt(x*x + y*y + z*z)
                if r > 0.1:
                    pts.append(np.array([x, y, z]) / r)
    # Add specific directions for numerical stability
    phi = (1 + np.sqrt(5)) / 2  # golden ratio
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            v = np.array([0, s1, s2*phi])
            pts.append(v / np.linalg.norm(v))
            v = np.array([s1, s2*phi, 0])
            pts.append(v / np.linalg.norm(v))
            v = np.array([s2*phi, 0, s1])
            pts.append(v / np.linalg.norm(v))
    return pts


def _generate_test_points_cubic():
    """Generate test points for fitting cubic (f-orbital) coefficients."""
    pts = _generate_test_points_quadratic()
    # Add more points for cubic terms
    for theta in np.linspace(0.1, np.pi - 0.1, 8):
        for phi_angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
            x = np.sin(theta) * np.cos(phi_angle)
            y = np.sin(theta) * np.sin(phi_angle)
            z = np.cos(theta)
            pts.append(np.array([x, y, z]))
    return pts


def get_orbital_rotation(orbital_type: str, R_cart: np.ndarray) -> np.ndarray:
    """
    Get the orbital rotation matrix for a given orbital type and rotation.

    Parameters
    ----------
    orbital_type : str
        One of 's', 'p', 'd', 'f'
    R_cart : ndarray (3, 3)
        Rotation matrix in Cartesian coordinates

    Returns
    -------
    ndarray
        Orbital rotation matrix D such that phi'_m = sum_n D_{mn} phi_n
    """
    if orbital_type == 's':
        return orbital_rotation_s(R_cart)
    elif orbital_type == 'p':
        return orbital_rotation_p(R_cart)
    elif orbital_type == 'd':
        return orbital_rotation_d(R_cart)
    elif orbital_type == 'f':
        return orbital_rotation_f(R_cart)
    else:
        raise ValueError(f"Unsupported orbital type: {orbital_type}")


# ============================================================================
# Euler Angles and Spinor D-matrix
# ============================================================================

def rmat_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Extract Euler angles (alpha, beta, gamma) from a rotation matrix.
    ZYZ convention matching wannhr_symm/lib/rotate.py.

    Returns
    -------
    alpha, beta, gamma : float
        Euler angles in radians. alpha, gamma in [0, 2*pi], beta in [0, pi].
    """
    if abs(R[2, 2]) < 1.0:
        beta = np.arccos(np.clip(R[2, 2], -1.0, 1.0))
        sin_beta = np.sin(beta)

        cos_gamma = -R[2, 0] / sin_beta
        sin_gamma = R[2, 1] / sin_beta
        gamma = _angle_from_sincos(sin_gamma, cos_gamma)

        cos_alpha = R[0, 2] / sin_beta
        sin_alpha = R[1, 2] / sin_beta
        alpha = _angle_from_sincos(sin_alpha, cos_alpha)
    else:
        if R[2, 2] > 0:  # beta = 0
            beta = 0.0
            gamma = 0.0
            alpha = np.arccos(np.clip(R[1, 1], -1.0, 1.0))
            if -R[0, 1] < 0.0:
                alpha = -alpha
        else:  # beta = pi
            beta = np.pi
            gamma = 0.0
            alpha = np.arccos(np.clip(R[1, 1], -1.0, 1.0))
            if -R[0, 1] < 0.0:
                alpha = -alpha

    return alpha, beta, gamma


def _angle_from_sincos(sina: float, cosa: float) -> float:
    """Determine angle in [0, 2*pi] from sin and cos values."""
    cosa = np.clip(cosa, -1.0, 1.0)
    angle = np.arccos(cosa)
    if sina < 0.0:
        angle = 2.0 * np.pi - angle
    return angle


def spinor_dmatrix(alpha: float, beta: float, gamma: float) -> np.ndarray:
    """
    Compute the spin-1/2 (spinor) representation matrix for a rotation
    specified by Euler angles.

    D[0,0] =  exp(-i(a+g)/2) * cos(b/2)
    D[0,1] = -exp(-i(a-g)/2) * sin(b/2)
    D[1,0] =  exp(+i(a-g)/2) * sin(b/2)
    D[1,1] =  exp(+i(a+g)/2) * cos(b/2)

    Parameters
    ----------
    alpha, beta, gamma : float
        Euler angles (ZYZ convention)

    Returns
    -------
    ndarray (2, 2) complex128
        Spinor D-matrix
    """
    D = np.zeros((2, 2), dtype=np.complex128)
    D[0, 0] = np.exp(-1j * (alpha + gamma) / 2.0) * np.cos(beta / 2.0)
    D[0, 1] = -np.exp(-1j * (alpha - gamma) / 2.0) * np.sin(beta / 2.0)
    D[1, 0] = np.exp(1j * (alpha - gamma) / 2.0) * np.sin(beta / 2.0)
    D[1, 1] = np.exp(1j * (alpha + gamma) / 2.0) * np.cos(beta / 2.0)
    return D


def get_spinor_dmatrix(R_cart: np.ndarray) -> np.ndarray:
    """
    Get the spinor D-matrix for a Cartesian rotation matrix.

    For improper rotations (det = -1), we use the proper part (det*R)
    since spinor representation is for SO(3), not O(3).
    """
    det = np.linalg.det(R_cart)
    R_proper = R_cart * np.sign(det)  # Remove inversion if present
    alpha, beta, gamma = rmat_to_euler(R_proper)
    D = spinor_dmatrix(alpha, beta, gamma)

    # Clean small values
    for i in range(2):
        for j in range(2):
            if abs(D[i, j].real) < 1e-10:
                D[i, j] = 1j * D[i, j].imag
            if abs(D[i, j].imag) < 1e-10:
                D[i, j] = D[i, j].real
    return D


# ============================================================================
# Phase 3: Representation Matrix Assembly
# ============================================================================

def build_representation_matrices(
    sym_info: SymmetryInfo,
    orbital_types_per_atom: List[List[str]],
    has_soc: bool = False,
    local_axes: Optional[np.ndarray] = None
) -> List[np.ndarray]:
    """
    Build the full representation (protmat) matrix for each symmetry operation.

    Parameters
    ----------
    sym_info : SymmetryInfo
        Symmetry operations from detect_symmetry_operations()
    orbital_types_per_atom : list of list of str
        For each atom, a list of orbital types (e.g., [['s', 's', 'p', 'p', 'd']] for
        shells of each atom). Each entry is one shell's angular momentum type.
    has_soc : bool
        Whether the system has spin-orbit coupling
    local_axes : ndarray (natom, 3, 3), optional
        Local coordinate axes for each atom. Default: identity (global axes).

    Returns
    -------
    list of ndarray
        One representation matrix per symmetry operation.
        Shape: (norbs, norbs) without SOC, (2*norbs, 2*norbs) with SOC.
    """
    natom = len(orbital_types_per_atom)

    # Compute number of orbitals per atom and total
    norbs_per_atom = []
    for atom_orbs in orbital_types_per_atom:
        n = 0
        for orb_type in atom_orbs:
            n += _orbital_dim(orb_type)
        norbs_per_atom.append(n)

    norbs_spatial = sum(norbs_per_atom)

    # Default local axes: identity for all atoms
    if local_axes is None:
        local_axes = np.tile(np.eye(3), (natom, 1, 1))

    # Offsets for each atom in the orbital basis
    offsets = np.zeros(natom, dtype=int)
    for i in range(1, natom):
        offsets[i] = offsets[i-1] + norbs_per_atom[i-1]

    protmat_list = []

    for op in sym_info.operations:
        R_cart = op.rotation_cart

        # Build orbital-only representation matrix (no spin)
        prot_orb = np.zeros((norbs_spatial, norbs_spatial), dtype=np.complex128)

        for jatom in range(natom):
            iatom = op.atom_map[jatom]

            # Local rotation: R_local = axis_i^T @ R_cart @ axis_j
            axs_j = local_axes[jatom]
            axs_i = local_axes[iatom]
            R_local = axs_i.T @ R_cart @ axs_j

            # Build block-diagonal orbital rotation for this atom pair
            off_j = offsets[jatom]
            off_i = offsets[iatom]

            block_offset = 0
            for shell_type in orbital_types_per_atom[jatom]:
                dim = _orbital_dim(shell_type)
                D_orb = get_orbital_rotation(shell_type, R_local)
                prot_orb[
                    off_i + block_offset : off_i + block_offset + dim,
                    off_j + block_offset : off_j + block_offset + dim
                ] = D_orb
                block_offset += dim

        if has_soc:
            # For SOC: protmat = kron(prot_orb, dmat_spinor) in interleaved ordering
            dmat = get_spinor_dmatrix(R_cart)

            # kron gives interleaved spin ordering: (orb1_up, orb1_dn, orb2_up, ...)
            protmat_interleaved = np.kron(prot_orb, dmat)

            # Convert to block spin ordering: [all_up | all_dn]
            # to match CRYSTAL's spin-block format
            protmat = _interleaved_to_block_spin(protmat_interleaved, norbs_spatial)
        else:
            protmat = prot_orb.real if np.allclose(prot_orb.imag, 0) else prot_orb

        protmat_list.append(protmat)

    return protmat_list


def _orbital_dim(orbital_type: str) -> int:
    """Return the dimension (number of m_l values) for an orbital type."""
    dims = {'s': 1, 'p': 3, 'd': 5, 'f': 7, 'g': 9}
    return dims[orbital_type.lower()]


def _interleaved_to_block_spin(M_interleaved: np.ndarray, norbs: int) -> np.ndarray:
    """
    Convert matrix from interleaved spin ordering (orb1_up, orb1_dn, orb2_up, ...)
    to block spin ordering ([all_up | all_dn]).

    Parameters
    ----------
    M_interleaved : ndarray (2*norbs, 2*norbs)
        Matrix in interleaved ordering
    norbs : int
        Number of spatial orbitals

    Returns
    -------
    ndarray (2*norbs, 2*norbs)
        Matrix in block spin ordering
    """
    n2 = 2 * norbs
    M_block = np.zeros((n2, n2), dtype=M_interleaved.dtype)

    # up-up block
    M_block[:norbs, :norbs] = M_interleaved[0::2, 0::2]
    # up-dn block
    M_block[:norbs, norbs:] = M_interleaved[0::2, 1::2]
    # dn-up block
    M_block[norbs:, :norbs] = M_interleaved[1::2, 0::2]
    # dn-dn block
    M_block[norbs:, norbs:] = M_interleaved[1::2, 1::2]

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


# ============================================================================
# Phase 4: Matrix Symmetrization
# ============================================================================

def symmetrize_real_space_matrices(
    real_space_matrices: Dict[Tuple[int, int, int], Dict[str, np.ndarray]],
    sym_info: SymmetryInfo,
    protmat_list: List[np.ndarray],
    orbital_offsets: np.ndarray,
    orbital_counts: np.ndarray,
    has_soc: bool = False,
    verbose: bool = False
) -> Dict[Tuple[int, int, int], Dict[str, np.ndarray]]:
    """
    Symmetrize H(R) and S(R) matrices using crystal symmetry operations.

    Implements the averaging formula:
        H'(R)_{ij} = (1/N_sym) sum_g P_gi^dag @ H(g.R + dR_gj - dR_gi)_{i'j'} @ P_gj

    Parameters
    ----------
    real_space_matrices : dict
        {R_int: {'H': H_matrix, 'S': S_matrix}} for each R-vector
    sym_info : SymmetryInfo
        Symmetry operations
    protmat_list : list of ndarray
        Representation matrices from build_representation_matrices()
    orbital_offsets : ndarray (natom,)
        Starting index of each atom's orbitals in the full basis
    orbital_counts : ndarray (natom,)
        Number of orbitals per atom
    has_soc : bool
        Whether SOC is enabled (affects spin-block offsets)
    verbose : bool
        Print progress information

    Returns
    -------
    dict
        Symmetrized {R_int: {'H': H_sym, 'S': S_sym}}
    """
    nsymm = sym_info.nsymm
    natom = len(orbital_offsets)

    # Spin multiplier for SOC
    ndm = 2 if has_soc else 1

    if verbose:
        print(f"  Symmetrizing with {nsymm} operations, {natom} atoms, SOC={has_soc}")

    # Step 1: Find missing R-blocks that will be generated by rotation
    existing_R = set(real_space_matrices.keys())
    new_R_blocks = _find_missing_R_blocks(
        existing_R, sym_info, orbital_offsets, orbital_counts, ndm, verbose
    )

    # Add new R-blocks with data generated from existing blocks
    all_matrices = dict(real_space_matrices)
    for R_new in new_R_blocks:
        if R_new not in all_matrices:
            all_matrices[R_new] = _generate_R_block(
                R_new, real_space_matrices, sym_info,
                protmat_list, orbital_offsets, orbital_counts, ndm
            )

    if verbose:
        print(f"  Added {len(new_R_blocks)} new R-blocks (total: {len(all_matrices)})")

    # Step 2: Symmetrize by averaging
    symmetrized = {}
    total_R = len(all_matrices)

    for idx, (R, matrices) in enumerate(all_matrices.items()):
        if verbose and idx % max(1, total_R // 10) == 0:
            print(f"  Symmetrizing: {100*idx//total_R}%...")

        R_arr = np.array(R, dtype=float)
        norbs_full = matrices['H'].shape[0]
        H_sym = np.zeros(matrices['H'].shape, dtype=np.complex128)
        S_sym = np.zeros(matrices['S'].shape, dtype=np.complex128)

        for isym, op in enumerate(sym_info.operations):
            protmat = protmat_list[isym]
            rot = op.rotation_frac

            for jatom in range(natom):
                off_j = ndm * orbital_offsets[jatom]
                nor_j = ndm * orbital_counts[jatom]
                jatom_mapped = op.atom_map[jatom]

                for iatom in range(natom):
                    off_i = ndm * orbital_offsets[iatom]
                    nor_i = ndm * orbital_counts[iatom]
                    iatom_mapped = op.atom_map[iatom]

                    # Rotated R-vector with atom shifts
                    R_rot = rot @ R_arr + op.vec_shift[jatom] - op.vec_shift[iatom]
                    R_rot_int = tuple(np.round(R_rot).astype(int))

                    if R_rot_int not in all_matrices:
                        continue

                    # Offsets in the mapped atoms
                    off_jp = ndm * orbital_offsets[jatom_mapped]
                    nor_jp = ndm * orbital_counts[jatom_mapped]
                    off_ip = ndm * orbital_offsets[iatom_mapped]
                    nor_ip = ndm * orbital_counts[iatom_mapped]

                    # Extract blocks
                    P_i = protmat[off_i:off_i+nor_i, off_i:off_i+nor_i]
                    P_j = protmat[off_j:off_j+nor_j, off_j:off_j+nor_j]
                    H_ipjp = all_matrices[R_rot_int]['H'][off_ip:off_ip+nor_ip, off_jp:off_jp+nor_jp]
                    S_ipjp = all_matrices[R_rot_int]['S'][off_ip:off_ip+nor_ip, off_jp:off_jp+nor_jp]

                    # H'(R)_{ij} += P_i^dag @ H(R')_{i'j'} @ P_j
                    H_sym[off_i:off_i+nor_i, off_j:off_j+nor_j] += (
                        P_i.conj().T @ H_ipjp @ P_j
                    )
                    S_sym[off_i:off_i+nor_i, off_j:off_j+nor_j] += (
                        P_i.conj().T @ S_ipjp @ P_j
                    )

        # Average over symmetry operations
        H_sym /= nsymm
        S_sym /= nsymm

        symmetrized[R] = {'H': H_sym, 'S': S_sym}

    if verbose:
        print(f"  Symmetrization complete.")

    return symmetrized


def _find_missing_R_blocks(
    existing_R: set,
    sym_info: SymmetryInfo,
    orbital_offsets: np.ndarray,
    orbital_counts: np.ndarray,
    ndm: int,
    verbose: bool
) -> set:
    """Find R-vectors that need to be generated from existing blocks via rotation."""
    new_R = set()
    natom = len(orbital_offsets)

    for R in existing_R:
        R_arr = np.array(R, dtype=float)
        for op in sym_info.operations:
            for jatom in range(natom):
                for iatom in range(natom):
                    R_rot = op.rotation_frac @ R_arr + op.vec_shift[jatom] - op.vec_shift[iatom]
                    R_rot_int = tuple(np.round(R_rot).astype(int))
                    if R_rot_int not in existing_R:
                        new_R.add(R_rot_int)

    return new_R


def _generate_R_block(
    R_target: Tuple[int, int, int],
    existing_matrices: Dict,
    sym_info: SymmetryInfo,
    protmat_list: List[np.ndarray],
    orbital_offsets: np.ndarray,
    orbital_counts: np.ndarray,
    ndm: int
) -> Dict[str, np.ndarray]:
    """Generate a new R-block from existing blocks using symmetry."""
    # Get matrix dimensions from any existing block
    sample = next(iter(existing_matrices.values()))
    norbs = sample['H'].shape[0]
    H_new = np.zeros((norbs, norbs), dtype=np.complex128)
    S_new = np.zeros((norbs, norbs), dtype=np.complex128)
    natom = len(orbital_offsets)

    # Find a symmetry operation that maps an existing R to R_target
    R_arr = np.array(R_target, dtype=float)

    for isym, op in enumerate(sym_info.operations):
        rot = op.rotation_frac
        protmat = protmat_list[isym]

        for jatom in range(natom):
            for iatom in range(natom):
                # We want: rot @ R_source + shift_j - shift_i = R_target
                # So: R_source = rot^{-1} @ (R_target - shift_j + shift_i)
                R_source_arr = np.linalg.solve(
                    rot.astype(float),
                    R_arr - op.vec_shift[jatom] + op.vec_shift[iatom]
                )
                R_source_int = tuple(np.round(R_source_arr).astype(int))

                if R_source_int not in existing_matrices:
                    continue

                off_j = ndm * orbital_offsets[jatom]
                nor_j = ndm * orbital_counts[jatom]
                off_i = ndm * orbital_offsets[iatom]
                nor_i = ndm * orbital_counts[iatom]

                jatom_mapped = op.atom_map[jatom]
                iatom_mapped = op.atom_map[iatom]
                off_jp = ndm * orbital_offsets[jatom_mapped]
                nor_jp = ndm * orbital_counts[jatom_mapped]
                off_ip = ndm * orbital_offsets[iatom_mapped]
                nor_ip = ndm * orbital_counts[iatom_mapped]

                P_i = protmat[off_i:off_i+nor_i, off_i:off_i+nor_i]
                P_j = protmat[off_j:off_j+nor_j, off_j:off_j+nor_j]
                H_src = existing_matrices[R_source_int]['H'][off_ip:off_ip+nor_ip, off_jp:off_jp+nor_jp]
                S_src = existing_matrices[R_source_int]['S'][off_ip:off_ip+nor_ip, off_jp:off_jp+nor_jp]

                H_new[off_i:off_i+nor_i, off_j:off_j+nor_j] = (
                    P_i.conj().T @ H_src @ P_j
                )
                S_new[off_i:off_i+nor_i, off_j:off_j+nor_j] = (
                    P_i.conj().T @ S_src @ P_j
                )
                # Once found, move to next atom pair
                break

    return {'H': H_new, 'S': S_new}


def enforce_hermiticity(
    real_space_matrices: Dict[Tuple[int, int, int], Dict[str, np.ndarray]]
) -> Dict[Tuple[int, int, int], Dict[str, np.ndarray]]:
    """
    Enforce Hermiticity: H(R) = [H(R) + H(-R)^dag] / 2

    This ensures that the Hamiltonian and overlap matrices satisfy the
    required symmetry relation between R and -R.
    """
    result = {}
    all_R = set(real_space_matrices.keys())

    for R, matrices in real_space_matrices.items():
        R_neg = tuple(-x for x in R)

        if R_neg in all_R:
            H_R = matrices['H']
            H_negR = real_space_matrices[R_neg]['H']
            H_herm = (H_R + H_negR.conj().T) / 2.0

            S_R = matrices['S']
            S_negR = real_space_matrices[R_neg]['S']
            S_herm = (S_R + S_negR.conj().T) / 2.0

            result[R] = {'H': H_herm, 'S': S_herm}
        else:
            # No -R partner: keep as-is (should not happen for proper crystal)
            result[R] = {'H': matrices['H'].copy(), 'S': matrices['S'].copy()}

    return result


def enforce_time_reversal(
    real_space_matrices: Dict[Tuple[int, int, int], Dict[str, np.ndarray]],
    norbs_spatial: int
) -> Dict[Tuple[int, int, int], Dict[str, np.ndarray]]:
    """
    Enforce time-reversal symmetry for SOC systems.

    For spinful systems in block spin ordering [up|dn]:
        H(R) = [H(R) + sigma_y @ H*(-R) @ sigma_y] / 2

    where sigma_y operates in spin space as:
        sigma_y = [[0, -i], [i, 0]] -> in block form: [[0, -I], [I, 0]] (times i)

    Actually, the time-reversal operator is T = i*sigma_y * K (complex conjugation).
    So: H(R) -> T H(-R) T^{-1} = sigma_y_block @ H*(-R) @ sigma_y_block^{-1}

    In block ordering, sigma_y_block swaps up/dn blocks with a sign.
    """
    n = norbs_spatial
    n2 = 2 * n  # Total dimension with spin

    # Build sigma_y in block form:
    # In up/dn block ordering:
    # sigma_y_block: up->dn with factor +1, dn->up with factor -1
    # Acting on [psi_up, psi_dn]: sigma_y gives [-psi_dn*, psi_up*]
    # But for matrix transformation: H -> U_L @ H* @ U_R
    # with U_L = sigma_y^T (conjugate transpose in spin space)
    # Reference: symmhr_addrptblock.py uses syl and syr matrices

    # Left: sigma_y^T * i  =  [[0, 1], [-1, 0]] in interleaved
    # Right: sigma_y * i   =  [[0, -1], [1, 0]] in interleaved
    # In block ordering:
    umat_L = np.zeros((n2, n2), dtype=np.complex128)
    umat_R = np.zeros((n2, n2), dtype=np.complex128)

    # sigma_y^T in block form: up-up=0, up-dn=I, dn-up=-I, dn-dn=0
    umat_L[:n, n:] = np.eye(n)      # up-dn block = +I
    umat_L[n:, :n] = -np.eye(n)     # dn-up block = -I

    # sigma_y in block form: up-up=0, up-dn=-I, dn-up=I, dn-dn=0
    umat_R[:n, n:] = -np.eye(n)     # up-dn block = -I
    umat_R[n:, :n] = np.eye(n)      # dn-up block = +I

    result = {}
    all_R = set(real_space_matrices.keys())

    for R, matrices in real_space_matrices.items():
        R_neg = tuple(-x for x in R)

        if R_neg in all_R:
            H_R = matrices['H']
            H_negR_conj = real_space_matrices[R_neg]['H'].conj()
            H_TR = umat_L @ H_negR_conj @ umat_R
            H_new = (H_R + H_TR) / 2.0

            # S matrix: same treatment
            S_R = matrices['S']
            S_negR_conj = real_space_matrices[R_neg]['S'].conj()
            S_TR = umat_L @ S_negR_conj @ umat_R
            S_new = (S_R + S_TR) / 2.0

            result[R] = {'H': H_new, 'S': S_new}
        else:
            result[R] = {'H': matrices['H'].copy(), 'S': matrices['S'].copy()}

    return result


# ============================================================================
# Fix degenerate subspace gauge for centrosymmetric SOC systems
# ============================================================================

def fix_degenerate_gauge(
    eigenvalues_list: List[np.ndarray],
    eigenvectors_list: List[np.ndarray],
    S_k_list: List[np.ndarray],
    basis_atom_map: np.ndarray,
    target_atom: int = 0,
    degeneracy_tol: float = 1e-8,
) -> List[np.ndarray]:
    """
    Fix the eigenvector gauge within degenerate subspaces to break site symmetry.

    For centrosymmetric SOC systems, eigenvalues come in exact Kramers pairs.
    numpy.eigh returns arbitrary linear combinations within each degenerate
    subspace, which averages out site-specific character. This function rotates
    eigenvectors within each degenerate pair so that one is primarily localized
    on one atom site and the other on its inversion partner.

    Uses the atom-difference operator P₀ - P₁ to maximally separate eigenvectors
    between two atom sites. The eigenvector with the largest eigenvalue of P₀-P₁
    is maximally on atom 0, and the one with smallest eigenvalue is maximally on
    atom 1.

    Parameters
    ----------
    eigenvalues_list : list of ndarray
        Eigenvalues at each k-point, shape (num_bands,)
    eigenvectors_list : list of ndarray
        Eigenvectors at each k-point, shape (num_orbitals, num_bands)
    S_k_list : list of ndarray
        Overlap matrices at each k-point, shape (num_orbitals, num_orbitals)
    basis_atom_map : ndarray
        Maps each orbital index to its atom index, shape (num_orbitals,)
    target_atom : int
        Primary atom index (default: 0). Its inversion partner is determined
        automatically (the other unique atom for 2-atom cells).
    degeneracy_tol : float
        Tolerance for identifying degenerate pairs (default: 1e-8)

    Returns
    -------
    list of ndarray
        Fixed eigenvectors with site-specific gauge within degenerate pairs
    """
    fixed_eigvecs = []

    # Identify the two atoms for the difference operator
    unique_atoms = np.unique(basis_atom_map)
    if len(unique_atoms) >= 2:
        # Use atom-difference operator P₀ - P₁
        atom0 = unique_atoms[0]
        atom1 = unique_atoms[1]
        atom0_mask = (basis_atom_map == atom0)
        atom1_mask = (basis_atom_map == atom1)
        use_difference = True
    else:
        # Fallback: single-atom cell, use atom-0 projector only
        atom0_mask = (basis_atom_map == unique_atoms[0])
        use_difference = False

    for ik, (eigs, vecs, S_k) in enumerate(
        zip(eigenvalues_list, eigenvectors_list, S_k_list)
    ):
        vecs_fixed = vecs.copy()
        num_bands = len(eigs)

        i = 0
        while i < num_bands:
            # Find the extent of the degenerate block
            j = i + 1
            while j < num_bands and abs(eigs[j] - eigs[i]) < degeneracy_tol:
                j += 1
            deg = j - i

            if deg >= 2:
                # Compute overlap-weighted coefficients: S @ C
                SC = S_k @ vecs_fixed[:, i:j]  # (norb, deg)

                if use_difference:
                    # Atom-difference operator: P₀ - P₁
                    # P₀[a,b] = sum_{orbs on atom0} C*[orb,a] * (SC)[orb,b]
                    P0 = vecs_fixed[atom0_mask, i:j].conj().T @ SC[atom0_mask, :]
                    P1 = vecs_fixed[atom1_mask, i:j].conj().T @ SC[atom1_mask, :]
                    P_diff = P0 - P1
                else:
                    # Single atom fallback
                    P_diff = vecs_fixed[atom0_mask, i:j].conj().T @ SC[atom0_mask, :]

                # Diagonalize: largest eigenvalue → most on atom 0,
                #              smallest eigenvalue → most on atom 1
                w, U = np.linalg.eigh(P_diff)
                vecs_fixed[:, i:j] = vecs_fixed[:, i:j] @ U

            i = j

        fixed_eigvecs.append(vecs_fixed)

    return fixed_eigvecs


# ============================================================================
# Helper: Extract orbital structure from CRYSTAL parser output
# ============================================================================

def get_orbital_structure_from_crystal(
    orbital_types_dict: Dict[int, str],
    num_basis_per_atom,
    num_atoms: int
) -> List[List[str]]:
    """
    Convert the flat orbital_types dict from parse_orbital_types() into the
    per-atom list-of-shells format needed by build_representation_matrices().

    Each shell is a group of (2l+1) basis functions of the same type:
    - s: 1 function per shell
    - p: 3 functions per shell
    - d: 5 functions per shell
    - f: 7 functions per shell

    Multiple shells of the same type are listed separately.
    E.g., 4 s-functions + 9 p-functions + 15 d-functions becomes:
    ['s', 's', 's', 's', 'p', 'p', 'p', 'd', 'd', 'd']

    Parameters
    ----------
    orbital_types_dict : dict
        {orbital_index (1-indexed): orbital_type} from parse_orbital_types()
    num_basis_per_atom : int or list of int
        Number of basis functions per atom. If int, assumed uniform for all atoms.
        If list, gives per-atom basis function counts.
    num_atoms : int
        Number of atoms

    Returns
    -------
    list of list of str
        For each atom, list of orbital shell types
    """
    shell_dims = {'s': 1, 'p': 3, 'd': 5, 'f': 7, 'g': 9}
    result = []

    # Support per-atom basis counts
    if isinstance(num_basis_per_atom, (list, tuple)):
        per_atom_counts = list(num_basis_per_atom)
    else:
        per_atom_counts = [num_basis_per_atom] * num_atoms

    # Compute per-atom offsets
    offsets = [0] * num_atoms
    for idx in range(1, num_atoms):
        offsets[idx] = offsets[idx - 1] + per_atom_counts[idx - 1]

    for atom_idx in range(num_atoms):
        offset = offsets[atom_idx]
        count = per_atom_counts[atom_idx]
        shells = []
        i = 1  # 1-indexed orbital position within atom
        while i <= count:
            key = i + offset
            if key in orbital_types_dict:
                orb_type = orbital_types_dict[key]
                shells.append(orb_type)
                # Skip remaining m_l values in this shell
                dim = shell_dims.get(orb_type, 1)
                i += dim
            else:
                i += 1  # Skip missing entries

        result.append(shells)

    return result


# ============================================================================
# Phase 5: Site Symmetry (.dmn) Support
# ============================================================================

@dataclass
class KPointSymmetryMap:
    """K-point symmetry mapping for site_symmetry mode."""
    ik2ir: np.ndarray       # (num_kpts,) maps full k-index -> irreducible k-index (0-based)
    ir2ik: np.ndarray       # (nkptirr,) maps irreducible index -> full k-index (0-based)
    kptsym: np.ndarray      # (nsymmetry, nkptirr) image of irr k under each symop (0-based)
    iks2k: np.ndarray       # (num_kpts, nsymmetry) image of any k under each symop (0-based)
    nkptirr: int
    nsymmetry: int


def compute_kpoint_symmetry_map(
    kpoints: np.ndarray,
    sym_info: 'SymmetryInfo',
    tolerance: float = 1e-5
) -> KPointSymmetryMap:
    """
    Compute k-point symmetry mapping arrays for site_symmetry .dmn file.

    For each symmetry operation R and k-point k, finds which k-point index
    corresponds to R^T @ k (mod reciprocal lattice). Identifies the irreducible
    Brillouin zone and builds all mapping arrays.

    Parameters
    ----------
    kpoints : ndarray (num_kpts, 3)
        K-points in fractional coordinates.
    sym_info : SymmetryInfo
        Symmetry operations from detect_symmetry_operations().
    tolerance : float
        Tolerance for matching k-points (mod 1).

    Returns
    -------
    KPointSymmetryMap
        All k-point symmetry mapping arrays (0-based internally).
    """
    num_kpts = len(kpoints)
    nsymmetry = len(sym_info.operations)

    # Build iks2k[ik, isym] = index of R_isym^T @ k_ik in k-grid
    iks2k = np.full((num_kpts, nsymmetry), -1, dtype=int)

    for isym, op in enumerate(sym_info.operations):
        R_frac = op.rotation_frac
        for ik in range(num_kpts):
            # K transforms contragrediently: k' = R^T @ k
            kp = R_frac.T @ kpoints[ik]
            # Find k' in k-grid (mod 1)
            diff = kpoints - kp[np.newaxis, :]
            diff -= np.round(diff)
            dists = np.linalg.norm(diff, axis=1)
            idx = np.argmin(dists)
            if dists[idx] < tolerance:
                iks2k[ik, isym] = idx
            else:
                raise RuntimeError(
                    f"Symop {isym}: k-point {kpoints[ik]} maps to {kp} "
                    f"which is not in the k-grid (min dist={dists[idx]:.2e})"
                )

    # Build irreducible k-points (same algorithm as pw2wannier90)
    found = np.zeros(num_kpts, dtype=bool)
    ir2ik_list = []
    ik2ir = np.full(num_kpts, -1, dtype=int)

    for ik in range(num_kpts):
        if found[ik]:
            continue
        found[ik] = True
        ir_idx = len(ir2ik_list)
        ir2ik_list.append(ik)
        ik2ir[ik] = ir_idx
        for isym in range(nsymmetry):
            ikp = iks2k[ik, isym]
            if not found[ikp]:
                found[ikp] = True
                ik2ir[ikp] = ir_idx

    ir2ik = np.array(ir2ik_list, dtype=int)
    nkptirr = len(ir2ik)

    # Build kptsym[isym, ir] = image of irr k-point ir under symop isym
    kptsym = np.zeros((nsymmetry, nkptirr), dtype=int)
    for ir in range(nkptirr):
        ik = ir2ik[ir]
        for isym in range(nsymmetry):
            kptsym[isym, ir] = iks2k[ik, isym]

    return KPointSymmetryMap(
        ik2ir=ik2ir,
        ir2ik=ir2ik,
        kptsym=kptsym,
        iks2k=iks2k,
        nkptirr=nkptirr,
        nsymmetry=nsymmetry
    )


def compute_d_matrix_wann(
    sym_info: 'SymmetryInfo',
    kpoints: np.ndarray,
    ksym_map: KPointSymmetryMap,
    protmat_list: list,
    wann_orbital_indices: np.ndarray,
    basis_atom_map: np.ndarray,
    has_soc: bool = False
) -> np.ndarray:
    """
    Compute the Wannier-space representation matrices d_matrix_wann.

    d_matrix_wann(m, n, isym, ir) describes how trial projections transform:
    |g_m(gk)> = sum_n D_wann_mn |g_n(k)>

    Parameters
    ----------
    sym_info : SymmetryInfo
        Symmetry operations.
    kpoints : ndarray (num_kpts, 3)
        K-points in fractional coordinates.
    ksym_map : KPointSymmetryMap
        K-point symmetry mapping.
    protmat_list : list of ndarray
        Representation matrices from build_representation_matrices().
        Shape: (norbs, norbs) or (2*norbs, 2*norbs) with SOC.
    wann_orbital_indices : ndarray
        Indices of orbitals used as Wannier projections (0-based, into full basis).
    basis_atom_map : ndarray
        Maps basis function index -> atom index.
    has_soc : bool
        Whether the system has spin-orbit coupling.

    Returns
    -------
    ndarray (num_wann, num_wann, nsymmetry, nkptirr)
        Complex representation matrices for trial projections.
    """
    num_wann = len(wann_orbital_indices)
    nsymmetry = ksym_map.nsymmetry
    nkptirr = ksym_map.nkptirr

    d_matrix_wann = np.zeros((num_wann, num_wann, nsymmetry, nkptirr),
                             dtype=np.complex128)

    for ir in range(nkptirr):
        ik = ksym_map.ir2ik[ir]
        k_frac = kpoints[ik]

        for isym in range(nsymmetry):
            op = sym_info.operations[isym]

            # Extract WF subspace block from full protmat
            # protmat[i, j] maps orbital j -> orbital i
            full_protmat = protmat_list[isym]
            wws = full_protmat[np.ix_(wann_orbital_indices, wann_orbital_indices)]

            # Build phase matrix (diagonal)
            # Phase for each WF column n: exp(2πi k · vec_shift[atom_of_wf_n])
            phs = np.zeros((num_wann, num_wann), dtype=np.complex128)
            for iw in range(num_wann):
                orb_idx = wann_orbital_indices[iw]
                atom_idx = basis_atom_map[orb_idx]
                # vec_shift[atom_j] = R·τ_j + t - τ_{atom_map[j]} (lattice vector)
                phase = 2.0 * np.pi * np.dot(op.vec_shift[atom_idx], k_frac)
                phs[iw, iw] = np.exp(1j * phase)

            # Also account for G-vector phase when Rk wraps around BZ
            ik_gk = ksym_map.iks2k[ik, isym]
            k_image = op.rotation_frac.T @ k_frac
            G_shift = kpoints[ik_gk] - k_image
            G_shift_round = np.round(G_shift)

            if np.linalg.norm(G_shift - G_shift_round) > 1e-6:
                warnings.warn(
                    f"Non-integer G-shift at isym={isym}, ir={ir}: {G_shift}"
                )

            # G-vector phase: exp(2πi tvec · (R^T @ G))
            # Following pw2wannier90: v2 = matmul(v1, sr) where v1 = k' - R@k
            if np.linalg.norm(G_shift_round) > 1e-10:
                # Phase from translation dotted with rotated G
                v2 = G_shift_round @ op.rotation_frac
                g_phase = np.exp(1j * 2.0 * np.pi * np.dot(op.translation, v2))
                phs *= g_phase

            d_matrix_wann[:, :, isym, ir] = phs @ wws

    return d_matrix_wann


def compute_d_matrix_band(
    sym_info: 'SymmetryInfo',
    kpoints: np.ndarray,
    ksym_map: KPointSymmetryMap,
    protmat_list: list,
    eigenvectors_list: list,
    S_k_list: list,
    basis_atom_map: np.ndarray,
    band_indices: np.ndarray = None,
    has_soc: bool = False
) -> np.ndarray:
    """
    Compute the band-space representation matrices d_matrix_band.

    D_band(g, k)_{mn} = <psi_m(gk) | g | psi_n(k)>

    In LCAO: D_band = C(gk)^dag S(gk) T(g,k) C(k)
    where T(g,k)_{nu,mu} = protmat_{nu,mu} * exp(2pi i k . vec_shift[atom_of_mu])

    Parameters
    ----------
    sym_info : SymmetryInfo
        Symmetry operations.
    kpoints : ndarray (num_kpts, 3)
        K-points in fractional coordinates.
    ksym_map : KPointSymmetryMap
        K-point symmetry mapping.
    protmat_list : list of ndarray
        Representation matrices from build_representation_matrices().
    eigenvectors_list : list of ndarray
        LCAO eigenvectors C(k) at each k-point. Shape (norbs, nbands_total).
    S_k_list : list of ndarray
        Overlap matrices S(k). Shape (norbs, norbs).
    basis_atom_map : ndarray
        Maps basis function index -> atom index.
    band_indices : ndarray, optional
        Indices of selected bands (0-based). If None, use all bands.
    has_soc : bool
        Whether the system has spin-orbit coupling.

    Returns
    -------
    ndarray (num_bands, num_bands, nsymmetry, nkptirr)
        Complex representation matrices for Bloch states.
    """
    nsymmetry = ksym_map.nsymmetry
    nkptirr = ksym_map.nkptirr
    norbs = eigenvectors_list[0].shape[0]

    if band_indices is not None:
        num_bands = len(band_indices)
    else:
        num_bands = eigenvectors_list[0].shape[1]

    d_matrix_band = np.zeros((num_bands, num_bands, nsymmetry, nkptirr),
                             dtype=np.complex128)

    for ir in range(nkptirr):
        ik = ksym_map.ir2ik[ir]
        k_frac = kpoints[ik]

        # C(k) with band selection
        if band_indices is not None:
            C_k = eigenvectors_list[ik][:, band_indices]
        else:
            C_k = eigenvectors_list[ik]

        for isym in range(nsymmetry):
            op = sym_info.operations[isym]

            # Find gk = R^T @ k (image k-point index)
            ik_gk = ksym_map.iks2k[ik, isym]

            # C(gk) and S(gk) with band selection
            if band_indices is not None:
                C_gk = eigenvectors_list[ik_gk][:, band_indices]
            else:
                C_gk = eigenvectors_list[ik_gk]
            S_gk = S_k_list[ik_gk]

            # Build T(g,k) = protmat * phase_columns
            # Phase for each source orbital mu: exp(-2pi i k . vec_shift[atom_of_mu])
            # The negative sign comes from the Bloch basis transformation:
            # g|chi_mu(k)> = exp(-i gk.L_mu) protmat |chi_nu(gk)>
            # where L_mu = vec_shift[atom_mu], and -gk.L = -k.L for inversion
            protmat = protmat_list[isym].copy()
            phase_vec = np.zeros(norbs, dtype=np.complex128)
            for mu in range(norbs):
                atom_idx = basis_atom_map[mu]
                phase = -2.0 * np.pi * np.dot(op.vec_shift[atom_idx], k_frac)
                phase_vec[mu] = np.exp(1j * phase)

            # Apply phase column-wise: T_{nu,mu} = protmat_{nu,mu} * phase[mu]
            T_gk = protmat * phase_vec[np.newaxis, :]

            # D_band = C(gk)^dag @ S(gk) @ T(g,k) @ C(k)
            d_matrix_band[:, :, isym, ir] = C_gk.conj().T @ S_gk @ T_gk @ C_k

    return d_matrix_band
