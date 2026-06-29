#!/usr/bin/env python3
"""
Check a Wannier90 seedname's .win + .eig against the disentanglement rules
(see lcao_wannier.wannier_checks). Catches frozen/outer-window violations that
would make wannier90.x abort.

Usage: PYTHONPATH=<repo> python3 scripts/check_win.py <path/to/seedname>
       (expects <seedname>.win and <seedname>.eig)
"""
import re
import sys

import numpy as np

from lcao_wannier.wannier_checks import check_disentanglement_rules, format_report


def read_win(path):
    keys = ('num_wann', 'num_bands', 'dis_win_min', 'dis_win_max',
            'dis_froz_min', 'dis_froz_max')
    p = {}
    for line in open(path):
        m = re.match(r'\s*(\w+)\s*=\s*([-\d.eE+]+)', line)
        if m and m.group(1) in keys:
            p[m.group(1)] = float(m.group(2))
    return p


def read_eig(path):
    """Return list of per-k energy arrays from a .eig file (band k energy)."""
    per_k = {}
    for line in open(path):
        parts = line.split()
        if len(parts) < 3:
            continue
        b, k, e = int(parts[0]), int(parts[1]), float(parts[2])
        per_k.setdefault(k, []).append(e)
    return [np.array(per_k[k]) for k in sorted(per_k)]


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: check_win.py <seedname>")
    seed = sys.argv[1]
    p = read_win(seed + '.win')
    eig = read_eig(seed + '.eig')
    rep = check_disentanglement_rules(
        eig,
        num_wann=int(p['num_wann']),
        num_bands=int(p['num_bands']),
        dis_win_min=p.get('dis_win_min'),
        dis_win_max=p.get('dis_win_max'),
        dis_froz_min=p.get('dis_froz_min'),
        dis_froz_max=p.get('dis_froz_max'),
    )
    print(format_report(rep, title=f"Disentanglement check: {seed}"))
    sys.exit(0 if rep.ok else 1)


if __name__ == '__main__':
    main()
