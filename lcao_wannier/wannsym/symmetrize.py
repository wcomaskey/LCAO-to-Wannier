"""
Hamiltonian Symmetrization via Reynolds Operator

Main orchestrator for post-Wannierization symmetry enforcement. Takes a
wannier90_hr.dat file and crystal structure information, then applies the
group-averaging (Reynolds operator) formula:

    H_symm(R)_ij = (1/N_sym) Σ_g P_gi† · H(g·R + Δ_gj - Δ_gi)_{i'j'} · P_gj

where P_g is the representation matrix (orbital rotation × spinor D-matrix
for SOC), and Δ_gi are lattice vector shifts from atom mapping.

Algorithm ported from symmhr_addrptblock.py (Changming Yue, arxiv:1805.12148),
rewritten in Python 3 with dict-based storage and no SymPy dependency.

Steps:
    1. Normalize R-point degeneracies
    2. Enforce Hermiticity: H(R) = [H(R) + H(-R)†] / 2
    3. Enforce time-reversal (SOC only): σ_y averaging
    4. Detect crystal symmetry (spglib)
    5. Build representation matrices (orbital + spinor)
    6. Find new R-blocks from rotation
    7. Fill new R-blocks from existing data
    8. Reynolds operator averaging
    9. Compress output (drop small hoppings)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

from .hamiltonian import HamiltonianData
from .symmetry_ops import (
    SymmetryInfo,
    detect_symmetry,
    build_representation_matrices,
    compute_orbital_info,
)


@dataclass
class WannSymConfig:
    """Configuration for Hamiltonian symmetrization."""
    apply_hermitization: bool = True
    apply_time_reversal: bool = True   # Only used when has_soc=True
    threshold: float = 1e-9           # Drop hoppings below this
    sym_tolerance: float = 1e-5       # spglib tolerance
    verbose: bool = True


@dataclass
class WannSymResult:
    """Results from symmetrization."""
    nrpt_original: int
    nrpt_after_newblocks: int
    nrpt_symmetrized: int
    nsymm: int
    space_group: str
    point_group: str
    max_change: float        # max |H_sym - H_orig| element-wise
    output_file: str = ""


def symmetrize_hr(
    hr_data: HamiltonianData,
    lattice_vectors: np.ndarray,
    atom_positions_frac: np.ndarray,
    atom_numbers: np.ndarray,
    orbital_types_per_atom: List[List[str]],
    has_soc: bool,
    config: WannSymConfig = WannSymConfig(),
) -> Tuple[HamiltonianData, WannSymResult]:
    """
    Symmetrize a tight-binding Hamiltonian from wannier90_hr.dat.

    Parameters
    ----------
    hr_data : HamiltonianData
        Input Hamiltonian (will not be modified; a copy is used internally)
    lattice_vectors : ndarray (3, 3)
        Lattice vectors as rows
    atom_positions_frac : ndarray (natom, 3)
        Fractional coordinates of atoms
    atom_numbers : ndarray (natom,)
        Atomic numbers
    orbital_types_per_atom : list of list of str
        Shell types for each atom, e.g., [['s', 's', 'p', 'p', 'd']]
    has_soc : bool
        Whether spin-orbit coupling is present
    config : WannSymConfig
        Configuration options

    Returns
    -------
    hr_sym : HamiltonianData
        Symmetrized Hamiltonian
    result : WannSymResult
        Summary of the symmetrization
    """
    _print = print if config.verbose else lambda *a, **k: None

    nrpt_original = hr_data.nrpt
    hr = hr_data.copy()

    # Compute orbital structure
    offsets, counts = compute_orbital_info(orbital_types_per_atom)
    norbs_spatial = int(np.sum(counts))
    ndm = 2 if has_soc else 1  # spin multiplier
    norbs_full = ndm * norbs_spatial

    _print(f"  Hamiltonian: {hr.norbs} orbitals, {hr.nrpt} R-points")
    _print(f"  Crystal: {len(atom_numbers)} atoms, SOC={has_soc}")
    if hr.norbs != norbs_full:
        raise ValueError(
            f"Mismatch: HR has {hr.norbs} orbitals but crystal structure implies "
            f"{norbs_full} ({norbs_spatial} spatial × {ndm} spin)"
        )

    # ---- Step 1: Normalize degeneracies ----
    _print("  Step 1: Normalizing R-point degeneracies...")
    hr.normalize_degeneracies()

    # Save pre-symmetrization copy for max_change calculation
    hr_orig = hr.copy()

    # ---- Step 2: Hermitize ----
    if config.apply_hermitization:
        _print("  Step 2: Enforcing Hermiticity H(R) = [H(R) + H(-R)†]/2...")
        _hermitize(hr, has_soc)

    # ---- Step 3: Time-reversal (SOC only) ----
    if has_soc and config.apply_time_reversal:
        _print("  Step 3: Enforcing time-reversal symmetry...")
        _apply_time_reversal(hr, norbs_spatial)

    # ---- Step 4: Detect symmetry ----
    _print("  Step 4: Detecting crystal symmetry...")
    sym_info = detect_symmetry(
        lattice_vectors, atom_positions_frac, atom_numbers,
        tolerance=config.sym_tolerance
    )
    _print(f"    Space group: {sym_info.space_group}")
    _print(f"    Point group: {sym_info.point_group}")
    _print(f"    {sym_info.nsymm} symmetry operations")

    # ---- Step 5: Build representation matrices ----
    _print("  Step 5: Building representation matrices...")
    protmat_list = build_representation_matrices(
        sym_info, orbital_types_per_atom, has_soc=has_soc
    )
    _print(f"    Representation matrix shape: {protmat_list[0].shape}")

    # ---- Step 6: Find new R-blocks ----
    _print("  Step 6: Finding new R-blocks from rotation...")
    existing_R = set(hr.hr.keys())
    new_R_set = _find_new_R_blocks(existing_R, sym_info, offsets, counts, ndm)
    _print(f"    Found {len(new_R_set)} new R-blocks")

    # ---- Step 7: Fill new R-blocks ----
    if new_R_set:
        _print("  Step 7: Filling new R-blocks from existing data...")
        _fill_new_R_blocks(hr, new_R_set, sym_info, protmat_list, offsets, counts, ndm)

    nrpt_after_newblocks = hr.nrpt

    # ---- Step 8: Reynolds averaging ----
    _print(f"  Step 8: Reynolds averaging over {sym_info.nsymm} operations...")
    hr_sym = _reynolds_average(hr, sym_info, protmat_list, offsets, counts, ndm, config.verbose)

    # ---- Step 9: Compress output ----
    _print(f"  Step 9: Compressing output (threshold={config.threshold})...")
    n_removed = hr_sym.compress(config.threshold)
    _print(f"    Removed {n_removed} R-points below threshold")
    _print(f"    Final: {hr_sym.nrpt} R-points")

    # Compute max change
    max_change = 0.0
    for R in hr_orig.hr:
        if R in hr_sym.hr:
            diff = np.abs(hr_sym.hr[R] - hr_orig.hr[R]).max()
            max_change = max(max_change, diff)

    result = WannSymResult(
        nrpt_original=nrpt_original,
        nrpt_after_newblocks=nrpt_after_newblocks,
        nrpt_symmetrized=hr_sym.nrpt,
        nsymm=sym_info.nsymm,
        space_group=sym_info.space_group,
        point_group=sym_info.point_group,
        max_change=max_change,
    )

    _print(f"\n  Summary:")
    _print(f"    R-points: {nrpt_original} → {nrpt_after_newblocks} → {hr_sym.nrpt}")
    _print(f"    Max element change: {max_change:.6e}")
    _print(f"    Space group: {sym_info.space_group} ({sym_info.nsymm} ops)")

    return hr_sym, result


# ============================================================================
# Internal Implementation Functions
# ============================================================================

def _hermitize(hr: HamiltonianData, has_soc: bool) -> None:
    """
    Enforce Hermiticity: H(R) = [H(R) + H(-R)†] / 2.

    For spinless systems, also forces H(R) to be real (imaginary part
    should vanish for real-orbital real-space Hamiltonian).

    Modifies hr in-place.
    """
    all_R = set(hr.hr.keys())

    for R in list(hr.hr.keys()):
        R_neg = tuple(-x for x in R)

        if R_neg in all_R:
            H_R = hr.hr[R]
            H_negR = hr.hr[R_neg]
            # Update both H(R) and H(-R) to form hermitian conjugate pair
            hr.hr[R] = (H_R + H_negR.conj().T) / 2.0
            hr.hr[R_neg] = (H_negR + H_R.conj().T) / 2.0

    if not has_soc:
        # For spinless case, force imaginary part to zero
        for R in hr.hr:
            hr.hr[R] = hr.hr[R].real.astype(np.complex128)


def _apply_time_reversal(hr: HamiltonianData, norbs_spatial: int) -> None:
    """
    Enforce time-reversal symmetry for SOC systems.

    For spinful systems in block ordering [up|dn]:
        H(R) = [H(R) + U_L @ H*(R) @ U_R] / 2

    where U_L = σ_y^T, U_R = σ_y in block form.

    Reference: symmhr_addrptblock.py lines 106-129.
    Modifies hr in-place.
    """
    n = norbs_spatial
    n2 = 2 * n

    # Build sigma_y unitary matrices in block [up|dn] ordering
    # σ_y^T in block: up-dn = +I, dn-up = -I
    umat_L = np.zeros((n2, n2), dtype=np.complex128)
    umat_L[:n, n:] = np.eye(n)       # up-dn = +I
    umat_L[n:, :n] = -np.eye(n)      # dn-up = -I

    # σ_y in block: up-dn = -I, dn-up = +I
    umat_R = np.zeros((n2, n2), dtype=np.complex128)
    umat_R[:n, n:] = -np.eye(n)      # up-dn = -I
    umat_R[n:, :n] = np.eye(n)       # dn-up = +I

    for R in list(hr.hr.keys()):
        H_R = hr.hr[R]
        H_conj = H_R.conj()
        H_TR = umat_L @ H_conj @ umat_R
        hr.hr[R] = (H_R + H_TR) / 2.0


def _find_new_R_blocks(
    existing_R: set,
    sym_info: SymmetryInfo,
    offsets: np.ndarray,
    counts: np.ndarray,
    ndm: int,
) -> set:
    """
    Find R-vectors that need to be generated from existing blocks via rotation.

    For each existing R and each symmetry operation g, computes:
        R_new = g.R + vec_shift[j] - vec_shift[i]
    for all atom pairs (i, j). If R_new is not in the existing set, it's new.

    Reference: symmhr_addrptblock.py lines 170-214.
    """
    new_R = set()
    natom = len(offsets)

    for R in existing_R:
        R_arr = np.array(R, dtype=float)
        for op in sym_info.operations:
            rot_R = op.rotation_frac @ R_arr
            for jatom in range(natom):
                for iatom in range(natom):
                    R_new = rot_R + op.vec_shift[jatom] - op.vec_shift[iatom]
                    R_new_int = tuple(np.round(R_new).astype(int))
                    if R_new_int not in existing_R:
                        new_R.add(R_new_int)

    return new_R


def _fill_new_R_blocks(
    hr: HamiltonianData,
    new_R_set: set,
    sym_info: SymmetryInfo,
    protmat_list: List[np.ndarray],
    offsets: np.ndarray,
    counts: np.ndarray,
    ndm: int,
) -> None:
    """
    Generate hopping matrices for new R-blocks using symmetry.

    For each new R-block, finds a symmetry operation that connects it to
    an existing R-block:
        H_new(R)_ij = P_i† @ H_existing(R')_{i'j'} @ P_j

    Reference: symmhr_addrptblock.py lines 221-281.
    Modifies hr in-place.
    """
    natom = len(offsets)
    norbs = hr.norbs
    existing_R = set(hr.hr.keys()) - new_R_set

    for R_new in new_R_set:
        # Initialize new block
        H_new = np.zeros((norbs, norbs), dtype=np.complex128)
        R_arr = np.array(R_new, dtype=float)

        # Track which atom-pair blocks have been filled
        filled = np.zeros((natom, natom), dtype=bool)

        for isym, op in enumerate(sym_info.operations):
            if filled.all():
                break

            protmat = protmat_list[isym]
            rot = op.rotation_frac

            for jatom in range(natom):
                for iatom in range(natom):
                    if filled[iatom, jatom]:
                        continue

                    # Check if this operation maps an existing R to R_new
                    # for this atom pair:
                    # We want: rot @ R_source + shift_j - shift_i = R_new
                    # So: R_source = rot^{-1} @ (R_new - shift_j + shift_i)
                    R_source_arr = np.linalg.solve(
                        rot.astype(float),
                        R_arr - op.vec_shift[jatom] + op.vec_shift[iatom]
                    )
                    R_source_int = tuple(np.round(R_source_arr).astype(int))

                    if R_source_int not in existing_R:
                        continue

                    # Extract blocks
                    off_j = ndm * offsets[jatom]
                    nor_j = ndm * counts[jatom]
                    off_i = ndm * offsets[iatom]
                    nor_i = ndm * counts[iatom]

                    jatom_mapped = op.atom_map[jatom]
                    iatom_mapped = op.atom_map[iatom]
                    off_jp = ndm * offsets[jatom_mapped]
                    nor_jp = ndm * counts[jatom_mapped]
                    off_ip = ndm * offsets[iatom_mapped]
                    nor_ip = ndm * counts[iatom_mapped]

                    P_i = protmat[off_i:off_i+nor_i, off_i:off_i+nor_i]
                    P_j = protmat[off_j:off_j+nor_j, off_j:off_j+nor_j]
                    H_src = hr.hr[R_source_int][off_ip:off_ip+nor_ip, off_jp:off_jp+nor_jp]

                    H_new[off_i:off_i+nor_i, off_j:off_j+nor_j] = (
                        P_i.conj().T @ H_src @ P_j
                    )
                    filled[iatom, jatom] = True

        hr.hr[R_new] = H_new
        hr.deg_rpt[R_new] = 1


def _reynolds_average(
    hr: HamiltonianData,
    sym_info: SymmetryInfo,
    protmat_list: List[np.ndarray],
    offsets: np.ndarray,
    counts: np.ndarray,
    ndm: int,
    verbose: bool = False,
) -> HamiltonianData:
    """
    Apply the Reynolds operator: average H over all symmetry operations.

    H_symm(R)_ij = (1/N_sym) Σ_g P_gi† · H(g·R + Δ_gj - Δ_gi)_{i'j'} · P_gj

    Reference: symmhr_addrptblock.py lines 305-372.
    """
    natom = len(offsets)
    nsymm = sym_info.nsymm
    norbs = hr.norbs
    all_R = set(hr.hr.keys())

    hr_sym = HamiltonianData(
        norbs=norbs,
        hr={},
        deg_rpt={R: 1 for R in all_R},
    )

    total_R = len(all_R)
    rpts_sorted = sorted(all_R)
    tenpct = max(1, total_R // 10)

    for idx, R in enumerate(rpts_sorted):
        if verbose and idx % tenpct == 0:
            print(f"    Reynolds averaging: {100*idx//total_R}%...")

        R_arr = np.array(R, dtype=float)
        H_sym = np.zeros((norbs, norbs), dtype=np.complex128)

        for isym, op in enumerate(sym_info.operations):
            protmat = protmat_list[isym]
            rot = op.rotation_frac
            rot_R = rot @ R_arr

            for jatom in range(natom):
                off_j = ndm * offsets[jatom]
                nor_j = ndm * counts[jatom]
                jatom_mapped = op.atom_map[jatom]

                for iatom in range(natom):
                    off_i = ndm * offsets[iatom]
                    nor_i = ndm * counts[iatom]
                    iatom_mapped = op.atom_map[iatom]

                    # Rotated R-vector with atom shifts
                    R_rot = rot_R + op.vec_shift[jatom] - op.vec_shift[iatom]
                    R_rot_int = tuple(np.round(R_rot).astype(int))

                    if R_rot_int not in all_R:
                        continue

                    # Offsets in the mapped atoms
                    off_jp = ndm * offsets[jatom_mapped]
                    nor_jp = ndm * counts[jatom_mapped]
                    off_ip = ndm * offsets[iatom_mapped]
                    nor_ip = ndm * counts[iatom_mapped]

                    # Extract blocks
                    P_i = protmat[off_i:off_i+nor_i, off_i:off_i+nor_i]
                    P_j = protmat[off_j:off_j+nor_j, off_j:off_j+nor_j]
                    H_ipjp = hr.hr[R_rot_int][off_ip:off_ip+nor_ip, off_jp:off_jp+nor_jp]

                    # H'(R)_ij += P_i† @ H(R')_{i'j'} @ P_j
                    H_sym[off_i:off_i+nor_i, off_j:off_j+nor_j] += (
                        P_i.conj().T @ H_ipjp @ P_j
                    )

        H_sym /= nsymm
        hr_sym.hr[R] = H_sym

    if verbose:
        print(f"    Reynolds averaging: 100%")

    return hr_sym
