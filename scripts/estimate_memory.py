#!/usr/bin/env python3
"""
Estimate peak memory for the LCAO-to-Wannier90 pipeline on a given input file.

The pipeline currently reads the whole CRYSTAL/LCAO output into memory
(``f.readlines()``) and builds several dense N x N arrays per R-vector, so peak
RSS can be many times the file size. This script does a single *streaming*
(O(1)-memory) pass over the file to extract the parameters that drive memory
use, then models the peak so you can decide threads / batch settings BEFORE
launching a 20+ minute run that might OOM.

Model is calibrated against a measured data point:
    Sc_all_center_sp_f (4.6 GB, N=1424, R=101, non-SOC, 2 spin channels)
    -> measured peak RSS 24.3 GB ; this model predicts ~24 GB.

Usage
-----
    python estimate_memory.py /path/to/material.out
    python estimate_memory.py material.out --num-r 101        # skip the R scan
    python estimate_memory.py material.out --available 26     # RAM budget (GB)
    python estimate_memory.py material.out --json

Notes
-----
* The R-vector count requires scanning the whole file (counts the
  "OVERLAP MATRIX - CELL N." headers). Pass --num-r to skip that scan if you
  already know it (Stage 1 prints "Unique R-vectors for H: <N>").
* Estimates are upper-ish bounds for guidance, not exact guarantees.
"""

import argparse
import os
import re
import sys

GiB = 1024.0 ** 3
CPLX = 16   # bytes per complex128
F64 = 8     # bytes per float64

# readlines() overhead: a Python list of str is ~1.6-1.8x the raw byte size of
# an ASCII text file (str object header + list pointers). Calibrated to 1.7.
LINES_FACTOR = 1.7

# Safety margin on the headline peak. The model captures the dominant arrays but
# not every transient copy (np.tril/.conj().T temporaries, gc lag, allocator
# fragmentation). Calibrated so the Sc case (measured 24.3 GB) is not
# under-predicted. Better to warn early than OOM at minute 20.
SAFETY = 1.15


def human(nbytes):
    return f"{nbytes / GiB:6.2f} GiB"


def scan_file(path, want_r=True, max_header_lines=20000):
    """Single streaming pass: header params + (optionally) R-vector count.

    Returns a dict with num_ao, k_grid, has_soc, spin_channels, num_r.
    Memory use is O(1) — we never hold the file in memory.
    """
    num_ao = None
    k_grid = None
    has_soc = False
    spin_channels = set()
    overlap_cells = 0

    ao_re = re.compile(r'NUMBER OF AO\s+(\d+)')
    shrink_re = re.compile(r'SHRINK\. FACT\.\(MONKH\.\)\s+(\d+)\s+(\d+)\s+(\d+)')
    overlap_re = re.compile(r'OVERLAP MATRIX - CELL N\.')
    spin_re = re.compile(r'\b(ALPHA_ALPHA|ALPHA_BETA|BETA_ALPHA|BETA_BETA|ALPHA|BETA)\s+ELECTRONS',
                         re.IGNORECASE)

    with open(path, 'r', errors='replace') as f:
        for i, line in enumerate(f):
            # Header-only fields: stop checking after the header region.
            if i < max_header_lines:
                if num_ao is None:
                    m = ao_re.search(line)
                    if m:
                        num_ao = int(m.group(1))
                if k_grid is None:
                    m = shrink_re.search(line)
                    if m:
                        k_grid = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
                if not has_soc and 'TWO-COMPONENT' in line and 'SCF' in line:
                    has_soc = True
            # Spin channels and R-count can appear anywhere in the matrix dump.
            if len(spin_channels) < 4:
                m = spin_re.search(line)
                if m:
                    spin_channels.add(m.group(1).upper())
            if want_r and overlap_re.search(line):
                overlap_cells += 1

            # Fast exit if we have everything and aren't counting R.
            if (not want_r and num_ao is not None and k_grid is not None
                    and i >= max_header_lines):
                break

    return {
        'num_ao': num_ao,
        'k_grid': k_grid,
        'has_soc': has_soc,
        'spin_channels': sorted(spin_channels),
        'num_r': overlap_cells if want_r else None,
    }


def estimate(filesize, num_ao, num_r, has_soc, n_spin, k_grid, num_sample):
    """Return (peak_bytes, breakdown_list[(label, bytes, note)])."""
    # Effective matrix dimension and dtype size.
    # SOC builds 2N x 2N complex spin blocks; non-SOC builds N x N.
    N = num_ao * (2 if has_soc else 1)
    n2 = N * N
    mat_c = n2 * CPLX            # one dense complex matrix
    mat_f = n2 * F64             # one dense real matrix

    nk = (k_grid[0] * k_grid[1] * k_grid[2]) if k_grid else 1

    # --- persistent through the run ---
    lines = LINES_FACTOR * filesize
    stacked = 2 * num_r * mat_c          # H_stack + S_stack (complex128)

    # --- parse-phase transients ---
    # raw parsed overlap (R) + fock real per spin (R*n_spin) as float64,
    # plus the combined complex fock (R*n_spin) and the full N x N lists (2R).
    parsed_raw = (1 + n_spin) * num_r * mat_f
    parsed_complex = n_spin * num_r * mat_c
    full_lists = 2 * num_r * (mat_c if has_soc else mat_f)
    parse_transient = parsed_raw + parsed_complex + full_lists

    # --- solve-phase transients ---
    conditioning = num_sample * mat_c              # S_all (K_sample, N, N)
    solve_batched = 2 * nk * mat_c                 # H_all + S_all for real k-grid

    # Peak is the persistent base plus the worst single transient phase.
    solve_phase = stacked + max(conditioning, solve_batched)
    peak = (lines + max(parse_transient, solve_phase)) * SAFETY

    breakdown = [
        ("readlines() file buffer", lines, f"{LINES_FACTOR:.1f} x file size (held all run)"),
        ("stacked H+S (complex128)", stacked, f"2 x {num_r} R x {N}^2 x 16 B"),
        ("parse transient", parse_transient, "raw+complex+full N x N lists"),
        ("conditioning S_all", conditioning, f"{num_sample} k-pts x {N}^2 x 16 B"),
        ("batched solve H+S", solve_batched, f"2 x {nk} k-pts x {N}^2 x 16 B"),
    ]
    return peak, breakdown


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('input', help='CRYSTAL/LCAO output file')
    p.add_argument('--num-r', type=int, default=None,
                   help='Number of unique R-vectors (skip the full-file scan)')
    p.add_argument('--num-sample', type=int, default=100,
                   help='Conditioning-check k-point sample size (default: 100)')
    p.add_argument('--available', type=float, default=None,
                   help='Available RAM budget in GB (for the verdict)')
    p.add_argument('--json', action='store_true', help='Emit JSON')
    args = p.parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"error: no such file: {args.input}")

    filesize = os.path.getsize(args.input)
    want_r = args.num_r is None
    if want_r:
        print(f"Scanning {args.input} ({filesize/GiB:.2f} GiB) for parameters + "
              f"R-vector count (streaming)...", file=sys.stderr)
    info = scan_file(args.input, want_r=want_r)

    num_ao = info['num_ao']
    if num_ao is None:
        sys.exit("error: could not find 'NUMBER OF AO' — is this a CRYSTAL output?")
    num_r = args.num_r if args.num_r is not None else info['num_r']
    if not num_r:
        sys.exit("error: no 'OVERLAP MATRIX - CELL' headers found; pass --num-r")

    # Spin channels: distinct ALPHA/BETA blocks drive fock storage. Default 2.
    n_spin = max(1, len(info['spin_channels'])) if info['spin_channels'] else 2

    peak, breakdown = estimate(filesize, num_ao, num_r, info['has_soc'],
                               n_spin, info['k_grid'], args.num_sample)

    avail = args.available
    if avail is None:
        try:
            pages = os.sysconf('SC_PHYS_PAGES')
            psize = os.sysconf('SC_PAGE_SIZE')
            avail = pages * psize / GiB
        except (ValueError, AttributeError, OSError):
            avail = None

    if args.json:
        import json
        print(json.dumps({
            'file_gib': filesize / GiB,
            'num_ao': num_ao, 'num_r': num_r, 'has_soc': info['has_soc'],
            'n_spin': n_spin, 'k_grid': info['k_grid'],
            'estimated_peak_gib': peak / GiB,
            'available_gib': avail,
            'breakdown': {lbl: b / GiB for lbl, b, _ in breakdown},
        }, indent=2))
        return

    print()
    print("=" * 64)
    print("  LCAO-to-Wannier90 Memory Estimate")
    print("=" * 64)
    print(f"  File:            {args.input}")
    print(f"  File size:       {filesize/GiB:.2f} GiB")
    print(f"  NUMBER OF AO:    {num_ao}" + ("  (x2 SOC spin blocks)" if info['has_soc'] else ""))
    print(f"  R-vectors:       {num_r}")
    print(f"  Spin channels:   {n_spin}  {info['spin_channels'] or '(assumed 2)'}")
    print(f"  K-grid:          {info['k_grid']}")
    print(f"  SOC:             {'Yes' if info['has_soc'] else 'No'}")
    print("-" * 64)
    print("  Contribution                          size       detail")
    print("-" * 64)
    for lbl, b, note in breakdown:
        print(f"  {lbl:<28}{human(b)}   {note}")
    print("-" * 64)
    print(f"  ESTIMATED PEAK RSS:           {human(peak)}  (~{peak/filesize:.1f}x file, incl. {int((SAFETY-1)*100)}% margin)")
    print("=" * 64)

    if avail:
        margin = avail - peak / GiB
        print(f"  Available RAM:                {avail:6.2f} GiB")
        if margin < 0:
            print(f"  VERDICT: ✗ WILL LIKELY OOM (short by {-margin:.1f} GiB)")
            print("    -> raise WSL memory/swap, or use a low-memory/batch run.")
        elif margin < 0.15 * avail:
            print(f"  VERDICT: ⚠ TIGHT (only {margin:.1f} GiB headroom)")
            print("    -> expect heavy paging / GUI starvation; consider batch mode.")
        else:
            print(f"  VERDICT: ✓ FITS ({margin:.1f} GiB headroom)")
        print("=" * 64)


if __name__ == '__main__':
    main()
