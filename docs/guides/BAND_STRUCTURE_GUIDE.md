# Band Structure Plotting with Wannier90

## Quick start — the auto-PDWF → band-comparison procedure

The full procedure is driven from `lcao_to_wannier90.py`. For a material whose
CRYSTAL output is `material.out`, using `--method auto` (PDWF band selection with
automatic fallback to projectability):

```bash
# 1. Stage 1 — auto band selection + .win.  --bands-plot adds bands_plot = .true.
#    and an auto-detected kpoint_path, so Wannier90 will write material_band.dat.
python lcao_to_wannier90.py --stage 1 --input material.out --seedname material \
    --method auto --bands-plot

# 2. Wannier90 preprocessing (writes material.nnkp)
wannier90.x -pp material

# 3. Stage 2 — generate material.eig / .amn / .mmn
python lcao_to_wannier90.py --stage 2 --input material.out --seedname material \
    --method auto

# 4. Wannier90 — localize and write material_band.dat
#    (correctly interpolated, with the wsvec.dat Wannier–Seitz corrections)
wannier90.x material

# 5a. LCAO bands coloured by PDWF projectability (two-panel plot + projected DOS)
python lcao_to_wannier90.py --stage 4 --input material.out --seedname material

# 5b. LCAO-vs-Wannier overlay — read material_band.dat (do NOT hand-roll an
#     hr.dat Fourier transform) and evaluate LCAO on the same k-points.
#     See "Comparing with the LCAO reference" below and
#     WANNIER_INTERPOLATION_PITFALL.md.
```

Useful overrides: Stage 4 takes `--kpath` / `--custom-kpath` to override the
auto-detected path, and `--k-grid NX NY NZ` / `--fermi-energy EV` for slabs or
level-shift-corrupted runs. Use `--method pdwf` to force strict PDWF (no fallback).

The remainder of this guide walks through a worked Bismuth example.

## Settings Added to bismuth_final.win

The following band structure settings have been added:

```fortran
bands_plot = .true.
bands_num_points = 100

begin kpoint_path
M  0.5000  0.0000  0.0000    G  0.0000  0.0000  0.0000
G  0.0000  0.0000  0.0000    K  0.3333  0.3333  0.0000
end kpoint_path
```

## High-Symmetry Points for 2D Hexagonal System

Your Bismuth material has a hexagonal 2D Brillouin zone:

- **Γ (Gamma)** = (0.0, 0.0, 0.0) - Brillouin zone center
- **M** = (0.5, 0.0, 0.0) - Edge center (middle of Γ-Γ edge)
- **K** = (1/3, 1/3, 0.0) - Corner point (hexagonal vertex)

The path **M-Γ-K** captures the main features of the band structure for hexagonal systems.

## Running Wannier90 to Generate Band Structure

Once you've run the full Wannier90 calculation with the updated .win file:

```bash
wannier90.x bismuth_final
```

Wannier90 will generate:
- `bismuth_final_band.dat` - Band structure data (Wannier-interpolated)
- `bismuth_final_band.gnu` - Gnuplot script for plotting
- `bismuth_final_band.kpt` - K-point coordinates along path

## Plotting the Band Structure

### Method 1: Using Gnuplot (Quickest)

```bash
gnuplot bismuth_final_band.gnu
# Opens a window with the band structure plot
```

### Method 2: Using Python/Matplotlib (More Control)

```python
import numpy as np
import matplotlib.pyplot as plt

# Read band structure data
data = np.loadtxt('bismuth_final_band.dat')
kpath = data[:, 0]  # K-point distance along path
bands = data[:, 1:]  # Energy bands (each column is a band)

# Plot
plt.figure(figsize=(8, 6))
for i in range(bands.shape[1]):
    plt.plot(kpath, bands[:, i], 'b-', linewidth=1.5)

# Add high-symmetry point labels
# You'll need to find the k-point positions from the .dat file
plt.axvline(x=0.0, color='k', linestyle='--', alpha=0.3)  # M
plt.axvline(x=k_gamma, color='k', linestyle='--', alpha=0.3)  # Γ
plt.axvline(x=k_K, color='k', linestyle='--', alpha=0.3)  # K

plt.xlabel('K-path', fontsize=14)
plt.ylabel('Energy (eV)', fontsize=14)
plt.title('Bismuth Band Structure (Wannier Interpolation)', fontsize=16)
plt.axhline(y=0, color='k', linestyle=':', alpha=0.5, label='E_F')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bismuth_bandstructure.png', dpi=300)
plt.show()
```

### Method 3: Using this package's Wannier band reader (recommended)

`lcao_wannier` ships a reader that parses Wannier90's `*_band.dat`, `*_band.kpt`,
and `*_band.labelinfo.dat` — already carrying the correct Wannier–Seitz
corrections that `wannier90.x` applied:

```python
from lcao_wannier import read_w90_band_outputs
import matplotlib.pyplot as plt

w90 = read_w90_band_outputs('bismuth_final')
for b in range(w90['num_wann']):
    plt.plot(w90['distances'], w90['eigenvalues'][:, b], 'b-', lw=1.2)
plt.xticks(w90['tick_positions'], w90['tick_labels'])
plt.ylabel('Energy (eV)'); plt.axhline(0, ls=':', c='k')
plt.savefig('bismuth_bands.png', dpi=300)
```

(There is no `from wannier90 import w90` dependency — use the helper above.)

## File Formats

### bismuth_final_band.dat Format
```
# Column 1: k-point distance along path
# Column 2-N: Energy of bands 1 to num_wann (relative to E_F)
0.000000  -4.234123  -4.156789  ...
0.001234  -4.235567  -4.158901  ...
...
```

### bismuth_final_band.gnu (Gnuplot script)
Wannier90 auto-generates a plotting script. You can customize it:
```gnuplot
set terminal x11
set xlabel "K-path"
set ylabel "Energy (eV)"
set title "Bismuth Band Structure"
set xrange [0:*]
set grid
plot "bismuth_final_band.dat" using 1:2 with lines title "Band 1", \
     "bismuth_final_band.dat" using 1:3 with lines title "Band 2", \
     ...
```

## Comparing with the LCAO reference

To check that the Wannier model reproduces the LCAO band structure, overlay
Wannier90's interpolated bands on the LCAO bands evaluated **at the same
k-points**:

```python
from lcao_wannier import read_w90_band_outputs, compute_band_structure

w90 = read_w90_band_outputs('bismuth_final')        # Wannier bands (wsvec-correct)

# Evaluate LCAO on exactly the k-points Wannier90 used:
class _Path: pass
kpath = _Path()
kpath.kpoints_frac   = w90['kpoints_frac']
kpath.distances      = w90['distances']
kpath.tick_positions = w90['tick_positions']
kpath.tick_labels    = w90['tick_labels']
eig_lcao, _, _ = compute_band_structure(real_space_matrices, lattice_vectors, kpath)
eig_lcao -= e_fermi          # W90's .eig is already E_F-centred; shift LCAO to match
```

Plot `w90['eigenvalues']` (solid) against `eig_lcao` (dots) on the shared
`w90['distances']` x-axis and report the per-band RMS. Good agreement means the
Wannier functions faithfully represent the target subspace.

> **Do not** build the Wannier bands by Fourier-transforming `*_hr.dat` yourself
> — on coarse k-grids that ignores the Wannier–Seitz corrections in `*_wsvec.dat`
> and can inflate the RMS by orders of magnitude. Always read `*_band.dat`. See
> **[WANNIER_INTERPOLATION_PITFALL.md](WANNIER_INTERPOLATION_PITFALL.md)**.

## Customizing the K-Path

If you want a different path (e.g., M-K-Γ-M), modify the .win file:

```fortran
begin kpoint_path
M  0.5000  0.0000  0.0000    K  0.3333  0.3333  0.0000
K  0.3333  0.3333  0.0000    G  0.0000  0.0000  0.0000
G  0.0000  0.0000  0.0000    M  0.5000  0.0000  0.0000
end kpoint_path
```

Or add more points for a complete path:
```fortran
begin kpoint_path
K   0.3333  0.3333  0.0000    G  0.0000  0.0000  0.0000
G   0.0000  0.0000  0.0000    M  0.5000  0.0000  0.0000
M   0.5000  0.0000  0.0000    K  0.3333  0.3333  0.0000
end kpoint_path
```

## Expected Features

For Bismuth with SOC, you should see:
- **Band splitting** due to spin-orbit coupling
- **Possible band inversion** near the Fermi level (topological features)
- **12 bands total** (from 12 Wannier functions)

## Tips

1. **Increase resolution**: Use `bands_num_points = 200` for smoother curves
2. **Check convergence**: Make sure Wannier90 converged before trusting bands
3. **Energy range**: Bands are plotted over the energy range of your Wannier functions
4. **Fermi level**: Set to 0 eV (already done via fermi_energy in .win)

## Next Steps After Plotting

Once you have good band structure:
- Calculate **Berry curvature** (for topological properties)
- Calculate **Fermi surface**
- Calculate **density of states (DOS)**
- Export to **WannierTools** for advanced analysis

## ⚠️ Important: Wannier-band interpolation pitfall

If you are comparing Wannier-interpolated bands against the LCAO
reference, **read the Wannier bands from `{seedname}_band.dat`**
(written by `wannier90.x` when `bands_plot = T`), not by
Fourier-transforming `{seedname}_hr.dat` yourself. The naïve formula

```python
H(k) = Σ_R H(R) · exp(2πi k·R) / weight(R)
```

is wrong on coarse k-grids because it ignores the Wannier–Seitz cell
boundary corrections that live in `{seedname}_wsvec.dat`. On a CrI3
6×6×6 grid, ignoring `wsvec.dat` inflated the per-band RMS by ~750×.

The package provides a correct helper:

```python
from lcao_wannier import read_w90_band_outputs

w90 = read_w90_band_outputs('my_seedname')
# w90['kpoints_frac'], w90['eigenvalues'], w90['distances'], …
```

See **[WANNIER_INTERPOLATION_PITFALL.md](WANNIER_INTERPOLATION_PITFALL.md)**
for the full explanation and the list of legacy scripts that contain
the deprecated FFT path.

