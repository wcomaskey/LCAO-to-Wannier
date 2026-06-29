#!/usr/bin/env python3
"""
Numerical validation of the Design-C (float64, non-SOC) low-memory path.

Builds real-space H(R)/S(R) for a small non-SOC system two ways:
  * fast: legacy complex128 R-space
  * low : streaming float64 R-space (promote_complex=False)
then runs the SAME downstream chain (stack -> Fourier assembly -> generalized
Hermitian eigensolve) at several k-points and asserts the band energies match.
This proves the float64 arrays flow through stack/Fourier/eigh unchanged — i.e.
nothing downstream silently requires complex128.

Run: PYTHONPATH=<repo> python3 scripts/validate_lowmem_numerics.py
"""
import os
import sys
import tempfile

import numpy as np
from scipy.linalg import eigh

from lcao_wannier.parser import (
    parse_overlap_and_fock_matrices,
    parse_overlap_and_fock_matrices_streaming,
    create_nonsoc_full_matrices,
)
from lcao_wannier.utils import prepare_real_space_matrices
from lcao_wannier.fourier import stack_real_space_matrices, fourier_all_kpoints


def _block(header, diag):
    """A 4x4 lower-triangular block as CRYSTAL prints it."""
    rows = [
        f"   1    {diag[0]: .7E}",
        f"   2     1.0000000E-01 {diag[1]: .7E}",
        f"   3     5.0000000E-02  1.0000000E-01 {diag[2]: .7E}",
        f"   4     2.0000000E-02  5.0000000E-02  1.0000000E-01 {diag[3]: .7E}",
    ]
    return (header + "\n\n"
            "                 1              2              3              4\n\n"
            + "\n".join(rows) + "\n\n")


# Non-SOC fixture with proper (R, -R) pairs so create_nonsoc_full_matrices
# produces k-dependent H(k): cells (0,0,0), (1,0,0), (-1,0,0).
FIXTURE = " NUMBER OF AO                 4\n" \
    " DIRECT LATTICE VECTOR COMPONENTS (ANGSTROM)\n" \
    "         5.0000000    0.0000000    0.0000000\n" \
    "         0.0000000    5.0000000    0.0000000\n" \
    "         0.0000000    0.0000000    5.0000000\n\n"
for R, od in [("  0  0  0", 1.0), ("  1  0  0", 0.2), (" -1  0  0", 0.2)]:
    FIXTURE += _block(f" OVERLAP MATRIX - CELL N.   1({R})", [od, od, od, od])
for spin in ("ALPHA", "BETA"):
    FIXTURE += f"    {spin}      ELECTRONS\n\n"
    for R, d in [("  0  0  0", [-5, -3, -4, -3.5]),
                 ("  1  0  0", [-0.8, -0.7, -0.6, -0.5]),
                 (" -1  0  0", [-0.8, -0.7, -0.6, -0.5])]:
        FIXTURE += _block(f" FOCK MATRIX - CELL N.   1({R})", d)


def build_rsm(path, mode):
    if mode == 'fast':
        with open(path) as f:
            lines = f.readlines()
        raw, lat = parse_overlap_and_fock_matrices(lines)
        H_R, S_R = {}, {}
        for mi in raw:
            R = tuple(mi['lattice_vector'])
            if mi['type'] == 'overlap':
                S_R[R] = mi['data']
            else:
                H_R.setdefault(R, {})[mi.get('spin_channel', 0)] = mi['data']
    else:
        H_R, S_R, lat, _ = parse_overlap_and_fock_matrices_streaming(
            path, promote_complex=False)
    H_full, S_full = create_nonsoc_full_matrices(H_R, S_R, lat)
    rsm = prepare_real_space_matrices(H_full, S_full, np.array(lat))
    return rsm


def bands(rsm):
    stacked = stack_real_space_matrices(rsm)
    kpts = np.array([[0, 0, 0], [0.25, 0, 0], [0.5, 0, 0], [0.13, 0.0, 0.0]],
                    dtype=float)
    H_all, S_all = fourier_all_kpoints(kpts, stacked)
    evs = []
    for k in range(len(kpts)):
        Hk = 0.5 * (H_all[k] + H_all[k].conj().T)
        Sk = 0.5 * (S_all[k] + S_all[k].conj().T)
        evs.append(eigh(Hk, Sk, eigvals_only=True))
    return np.array(evs), stacked.H_stack.dtype, stacked.S_stack.dtype


def main():
    with tempfile.NamedTemporaryFile('w', suffix='.out', delete=False) as tf:
        tf.write(FIXTURE)
        path = tf.name
    try:
        ev_fast, hf, sf = bands(build_rsm(path, 'fast'))
        ev_low, hl, sl = bands(build_rsm(path, 'low'))
    finally:
        os.unlink(path)

    print("Design-C numerical validation (non-SOC float64 path)")
    print("=" * 60)
    print(f"  fast R-space dtype: H={hf}, S={sf}")
    print(f"  low  R-space dtype: H={hl}, S={sl}")
    assert hl == np.float64, f"low H_stack expected float64, got {hl}"
    assert hf == np.complex128, f"fast H_stack expected complex128, got {hf}"
    max_diff = np.max(np.abs(ev_fast - ev_low))
    print(f"  max |E_fast - E_low| over 4 k-points x 4 bands: {max_diff:.2e}")
    assert max_diff < 1e-10, f"band energies diverge: {max_diff}"
    print("=" * 60)
    print("PASS: float64 R-space gives identical bands, half the memory.")


if __name__ == '__main__':
    main()
