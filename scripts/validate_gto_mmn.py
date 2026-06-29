#!/usr/bin/env python3
"""Validate the analytic GTO MMN (lcao_wannier/gto_mmn.py) against the midpoint
method on a consistent eigenbasis.

Checks:
  1. M_analytic(k, b=0) == identity   (formula sanity)
  2. M_analytic(k, b) ~= M_midpoint(k, b) at small b (fine grid)   where midpoint
     is accurate, so the exact method must agree.

Run:  python3 scripts/validate_gto_mmn.py calculations/BN_1C.out
"""
import sys
import numpy as np
from scipy.linalg import eigh
from math import pi

sys.path.insert(0, '.')
from lcao_wannier.parser import parse_overlap_and_fock_matrices_streaming
from lcao_wannier import gto_mmn as G


def full_realspace(M_R):
    """Rebuild full (non-symmetric) real-space matrices from CRYSTAL's lower-tri
    off-site storage: full(g) = tril(M[g]) + triu(M[-g]^T, 1)."""
    out = {}
    for g, M in M_R.items():
        gm = tuple(-x for x in g)
        upper = np.triu(M_R[gm].T, 1) if gm in M_R else np.triu(M.T, 1)
        out[g] = np.tril(M) + upper
    return out


def at_k(full_M, kfrac, latt_bohr):
    """Fourier sum_g e^{i 2pi k.g} full_M(g)."""
    n = next(iter(full_M.values())).shape[0]
    A = np.zeros((n, n), dtype=complex)
    for g, M in full_M.items():
        A += np.exp(2j * pi * np.dot(kfrac, g)) * M
    return 0.5 * (A + A.conj().T)  # Hermitize tiny asymmetry


def main():
    path = sys.argv[1]
    with open(path, errors='ignore') as f:
        lines = f.readlines()
    aos = G.parse_gto_basis(lines)
    H_R, S_R, latt, _ = parse_overlap_and_fock_matrices_streaming(path, promote_complex='auto')
    latt = np.asarray(latt, float)
    latt_bohr = latt * G.ANG2BOHR if np.max(np.abs(latt)) < 12 else latt
    def mat(v):
        if isinstance(v, dict):           # Fock is {spin: matrix}, spin=None restricted
            v = v[list(v.keys())[0]]
        return np.real(np.asarray(v))
    H_R = {g: mat(v) for g, v in H_R.items()}
    S_R = {g: mat(v) for g, v in S_R.items()}
    Hf, Sf = full_realspace(H_R), full_realspace(S_R)
    n = len(aos)
    print(f"AOs={n}, cells={len(S_R)}")

    recip = G.reciprocal_bohr(latt_bohr)
    cells_R = [(g, np.array(g, float) @ latt_bohr) for g in S_R]

    def solve(kfrac):
        Hk, Sk = at_k(Hf, kfrac, latt_bohr), at_k(Sf, kfrac, latt_bohr)
        w, C = eigh(Hk, Sk)
        return w, C

    # small b on a fine 12x12x6-like mesh
    k0 = np.array([0.10, 0.15, 0.20])
    bfrac = np.array([1.0 / 12, 0.0, 0.0])
    k1 = k0 + bfrac
    b_cart = bfrac @ recip
    print(f"b_frac={bfrac}, |b_cart|={np.linalg.norm(b_cart):.4f} 1/Bohr")

    w0, C0 = solve(k0)
    w1, C1 = solve(k1)
    print(f"eig(k0)[:6] (Ha): {np.round(w0[:6],4)}")

    cutoff = 16.0
    Sb = G.build_Sb_cells(aos, b_cart, cells_R, cutoff=cutoff)
    S0 = G.build_Sb_cells(aos, np.zeros(3), cells_R, cutoff=cutoff)

    # isolate: does my S(k0) match the parsed-reconstructed S(k0)?
    Smine_k0 = G.Sb_at_kpb(S0, k0)
    Sk_parsed = at_k(Sf, k0, latt_bohr)
    dS = np.abs(Smine_k0 - Sk_parsed)
    print(f"\n[0] |S_mine(k0) - S_parsed(k0)| max = {dS.max():.3e} "
          f"at {np.unravel_index(dS.argmax(), dS.shape)}  (|S| up to {np.abs(Sk_parsed).max():.2f})")

    # 1. b=0 identity
    M00 = G.mmn_block(C0, C0, S0, k0)
    print(f"\n[1] |M(k0,0) - I| max = {np.max(np.abs(M00 - np.eye(n))):.3e}")

    # 2. analytic vs midpoint at small b (use lowest 12 bands)
    nb = 12
    Ma = G.mmn_block(C0[:, :nb], C1[:, :nb], Sb, k1)
    Smid = G.Sb_at_kpb(S0, k0 + 0.5 * bfrac)
    Mmid = C0[:, :nb].conj().T @ Smid @ C1[:, :nb]
    d = np.abs(Ma - Mmid)
    print(f"[2] analytic vs midpoint (b small, {nb} bands):")
    print(f"    max|M_analytic - M_midpoint| = {d.max():.3e}")
    print(f"    diag |M_nn| analytic: {np.round(np.abs(np.diag(Ma)),4)}")
    print(f"    diag |M_nn| midpoint: {np.round(np.abs(np.diag(Mmid)),4)}")
    print(f"    max singular value analytic: {np.linalg.svd(Ma, compute_uv=False).max():.4f}")


if __name__ == '__main__':
    main()
