#!/usr/bin/env python3
"""
DFT(LCAO) vs Wannier band-structure comparison plot, with the RMS band
difference inside the frozen window as the fit-goodness metric.

  * DFT/LCAO bands  : solid line   (computed from H(R)/S(R) at the W90 path k-pts)
  * Wannier bands   : '-.' dashdot, drawn on top (canonical wannier90 band.dat,
                      which uses the wsvec.dat / Wigner-Seitz correction)
  * frozen + outer disentanglement windows are shaded/annotated
  * RMS(frozen) reported in the subtitle

Requires (from a Stage-1 run with --bands-plot, then -pp / Stage 2 / wannier90):
  {seedname}_band.dat, {seedname}_band.kpt, {seedname}_band.labelinfo.dat, {seedname}.win

Usage:
  PYTHONPATH=<repo> python3 scripts/plot_band_comparison.py \
      --input material.out --seedname material [--spin alpha|beta] [-o out.png]
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import eigh

from lcao_wannier.parser import (
    parse_calculation_parameters,
    parse_overlap_and_fock_matrices_streaming,
    create_spin_block_matrices,
    create_nonsoc_full_matrices,
)
from lcao_wannier.utils import prepare_real_space_matrices, prune_zero_rvectors
from lcao_wannier.fourier import stack_real_space_matrices, fourier_all_kpoints
from lcao_wannier.band_plot import read_w90_band_outputs
from lcao_wannier.wannier_checks import parse_win_windows

HARTREE_TO_EV = 27.211386245988
_SPIN_KEY = {'alpha': 'ALPHA_ALPHA', 'beta': 'BETA_BETA', None: None}


def build_real_space(input_path, spin):
    H_R, S_R, lat, header = parse_overlap_and_fock_matrices_streaming(
        input_path, promote_complex='auto')
    params = parse_calculation_parameters(header)
    if params.has_soc:
        H_full, S_full = create_spin_block_matrices(H_R, S_R, params.num_ao, lat)
    else:
        H_full, S_full = create_nonsoc_full_matrices(
            H_R, S_R, lat, spin_channel=_SPIN_KEY.get(spin))
    rsm = prepare_real_space_matrices(H_full, S_full, np.array(lat))
    rsm, _ = prune_zero_rvectors(rsm, verbose=False)
    return rsm, params


def dft_bands_at(kpoints_frac, rsm):
    """Generalized-eigenvalue bands at the given fractional k-points (eV, absolute)."""
    stacked = stack_real_space_matrices(rsm)
    H_all, S_all = fourier_all_kpoints(np.asarray(kpoints_frac, float), stacked)
    nk = len(kpoints_frac)
    nb = stacked.num_orbitals
    bands = np.empty((nk, nb))
    for i in range(nk):
        Hk = 0.5 * (H_all[i] + H_all[i].conj().T)
        Sk = 0.5 * (S_all[i] + S_all[i].conj().T)
        bands[i] = eigh(Hk, Sk, eigvals_only=True)
    # CRYSTAL H(R) is in Hartree; the .eig / Wannier band.dat are in eV. Convert
    # here so the comparison (and the E_F shift, which is in eV) is consistent.
    return bands * HARTREE_TO_EV


def rms_in_window(dft_rel, wann_rel, lo, hi):
    """RMS interpolation error (meV) in [lo, hi]: for each DFT band in the window
    at each k, the distance to the NEAREST Wannier band. This is robust to the
    DFT/Wannier band counts differing at the window edges (sorted pairing would
    mis-pair there and inflate the RMS)."""
    diffs = []
    for k in range(dft_rel.shape[0]):
        d = dft_rel[k][(dft_rel[k] >= lo) & (dft_rel[k] <= hi)]
        w = wann_rel[k]
        if len(d) and len(w):
            for e in d:
                diffs.append(float(np.min(np.abs(w - e))))
    if not diffs:
        return float('nan'), 0
    diffs = np.asarray(diffs)
    return float(np.sqrt(np.mean(diffs ** 2)) * 1000.0), diffs.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--seedname', required=True)
    ap.add_argument('--spin', choices=['alpha', 'beta'], default=None)
    ap.add_argument('-o', '--output', default=None)
    ap.add_argument('--emargin', type=float, default=1.0,
                    help='eV margin beyond the outer window for the y-range')
    ap.add_argument('--ylim', type=float, nargs=2, default=None, metavar=('LO', 'HI'),
                    help='explicit y-axis range (eV rel E_F), overriding the '
                         'outer-window auto-range (use to zoom in)')
    args = ap.parse_args()

    w = read_w90_band_outputs(args.seedname)        # canonical Wannier bands
    p = parse_win_windows(args.seedname + '.win')   # windows (rel E_F)
    rsm, params = build_real_space(args.input, args.spin)

    # CRITICAL: the .eig (hence the Wannier band.dat) was written relative to the
    # E_F the engine actually used, which is recorded in the .win as fermi_energy.
    # That can DIFFER from CRYSTAL's printed Fermi (params.fermi_energy) when the
    # engine auto-detects E_F from band filling. Use the .win value so the DFT and
    # Wannier bands share the SAME zero; otherwise a constant offset inflates the
    # RMS (e.g. BN: 3.3 eV offset -> phantom 1900 meV instead of the true ~0.5).
    e_fermi = p.get('fermi_energy')
    if e_fermi is None:
        e_fermi = params.fermi_energy if params.fermi_energy is not None else 0.0

    wann_rel = w['eigenvalues']                     # already rel E_F (eig frame)
    dft_rel = dft_bands_at(w['kpoints_frac'], rsm) - e_fermi
    dist = w['distances']

    froz = (p.get('dis_froz_min'), p.get('dis_froz_max'))
    outer = (p.get('dis_win_min'), p.get('dis_win_max'))

    rms, ncmp = (float('nan'), 0)
    if froz[0] is not None:
        rms, ncmp = rms_in_window(dft_rel, wann_rel, froz[0], froz[1])

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(7, 6))
    # windows
    if outer[0] is not None:
        ax.axhspan(outer[0], outer[1], color='0.92', zorder=0,
                   label='disentangle (outer)')
    if froz[0] is not None:
        ax.axhspan(froz[0], froz[1], color='#cfe8ff', alpha=0.7, zorder=0,
                   label='frozen (inner)')
        for y in froz:
            ax.axhline(y, color='#1f77b4', lw=0.8, ls=':', zorder=1)
    ax.axhline(0.0, color='0.4', lw=0.8, zorder=1)  # E_F

    # DFT solid (label once)
    ax.plot(dist, dft_rel[:, 0], '-', color='k', lw=1.3, label='DFT (LCAO)')
    ax.plot(dist, dft_rel[:, 1:], '-', color='k', lw=1.3)
    # Wannier dashdot on top
    ax.plot(dist, wann_rel[:, 0], '-.', color='#d62728', lw=1.2, label='Wannier')
    ax.plot(dist, wann_rel[:, 1:], '-.', color='#d62728', lw=1.2)

    for x in w['tick_positions']:
        ax.axvline(x, color='0.6', lw=0.7)
    ax.set_xticks(w['tick_positions'])
    ax.set_xticklabels([t.replace('G', r'$\Gamma$') for t in w['tick_labels']])
    ax.set_xlim(dist.min(), dist.max())
    if args.ylim is not None:
        ax.set_ylim(args.ylim[0], args.ylim[1])
    elif outer[0] is not None:
        ax.set_ylim(outer[0] - args.emargin, outer[1] + args.emargin)
    ax.set_ylabel('E - E$_F$ (eV)')

    chan = f" ({args.spin})" if args.spin else ""
    ax.set_title(f"{args.seedname}{chan}: DFT vs Wannier bands")
    froz_str = (f"[{froz[0]:.2f}, {froz[1]:.2f}]" if froz[0] is not None else "n/a")
    fig.text(0.5, 0.915,
             f"frozen window {froz_str} eV   |   "
             f"RMS in frozen window = {rms:.2f} meV  ({ncmp} band pts)",
             ha='center', fontsize=9, color='0.25')
    ax.legend(loc='upper right', fontsize=8, framealpha=0.9)

    out = args.output or f"{args.seedname}_band_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"  frozen window: {froz_str} eV")
    print(f"  RMS(frozen) = {rms:.3f} meV over {ncmp} band points")
    print(f"  wrote {out}")


if __name__ == '__main__':
    main()
