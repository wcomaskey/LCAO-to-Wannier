#!/usr/bin/env python3
"""Scan a Wannier90 .mmn for non-unitarity: report max |M_mn| and max diagonal
|M_nn|. For true overlaps of normalized states these must be <= 1; values > 1
indicate a broken (non-unitary) overlap, which produces negative spreads."""
import sys

path = sys.argv[1]
with open(path) as f:
    f.readline()  # comment header
    nb, nk, nn = map(int, f.readline().split())
    max_all = 0.0
    max_diag = 0.0
    max_all_blk = max_diag_blk = None
    n_diag_gt1 = 0
    for _ in range(nk * nn):
        hdr = f.readline().split()
        ik, ik2 = int(hdr[0]), int(hdr[1])
        bmax = 0.0
        bdiag = 0.0
        for idx in range(nb * nb):
            re, im = f.readline().split()
            mag2 = float(re) * float(re) + float(im) * float(im)
            if mag2 > bmax:
                bmax = mag2
            # column-major: m (row) fastest, n (col) slower
            if (idx % nb) == (idx // nb):
                if mag2 > bdiag:
                    bdiag = mag2
                if mag2 > 1.0001:
                    n_diag_gt1 += 1
        if bmax > max_all:
            max_all = bmax
            max_all_blk = (ik, ik2)
        if bdiag > max_diag:
            max_diag = bdiag
            max_diag_blk = (ik, ik2)
    print(f"file: {path}")
    print(f"  num_bands={nb}  num_k={nk}  nntot={nn}")
    print(f"  max |M_mn| (any element) = {max_all**0.5:.4f}  at (k,k+b)={max_all_blk}")
    print(f"  max |M_nn| (diagonal)    = {max_diag**0.5:.4f}  at (k,k+b)={max_diag_blk}")
    print(f"  diagonal elements with |M_nn| > 1 : {n_diag_gt1}")
    print(f"  VERDICT: {'NON-UNITARY (|M|>1) -> negative spreads expected' if max_diag**0.5 > 1.01 else 'overlaps within unit disk (OK)'}")
