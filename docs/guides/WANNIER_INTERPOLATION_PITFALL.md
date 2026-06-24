# Wannier band interpolation — the wsvec pitfall

**TL;DR — Never interpolate Wannier bands by re-Fourier-transforming
`{seedname}_hr.dat` yourself.** Read `{seedname}_band.dat` instead, which
Wannier90 has already produced with the correct
Wannier–Seitz-cell (WS) corrections from `{seedname}_wsvec.dat`. Use the
helper `lcao_wannier.read_w90_band_outputs(seedname)`.

## When does this actually bite?

**Only on coarse k-grids.** The error from skipping `wsvec.dat` scales
with the magnitude of `H(R)` at the Wannier-Seitz (WS) cell boundary
R-vectors. As the k-grid used to build the Wannier model gets denser,
those boundary R-vectors get pushed to larger |R| where the Hamiltonian
elements decay exponentially. The naïve FFT becomes asymptotically
correct as the grid is refined.

Concrete cases from this project:

| System | k-grid | WS-cell radius | Naïve-FFT verdict |
|---|---|---|---|
| MgB2 (3D metal) | 18×18×18 | ~9 lattice constants | **Fine** — produces plots indistinguishable from the wsvec-correct ones |
| CrI3 spinlock3 (2D FM SOC) | 6×6×6 | ~3 lattice constants | **Broken** — per-band median RMS inflated 750× (258 meV vs 0.34 meV) |

The cutoff isn't sharp — it depends on the Hamiltonian's decay rate,
which itself depends on band gap, localization, and dimensionality.
Roughly, **if your grid is ≳12 in every direction the FFT is usually
fine**; **below ~8 it's often misleading**; in between, check by
diffing against W90's own `band.dat`.

## Symptom (coarse grid)

- Wannier bands that look jagged, oscillatory, or asymmetric where the
  LCAO reference is smooth.
- Per-band RMS ≫ 100 meV on a model whose `Ω_total` is small and
  whose `wannier90.x`-produced band plot looks fine.
- Strong sensitivity to k-grid density — the same hr.dat gives wildly
  different interpolated bands as you change the original k-grid used
  to build the Wannier model.

We hit this on CrI3 with a 6×6×6 Monkhorst–Pack mesh. The per-band
median RMS for the spinlock3 model was reported as **258 meV** by the
naïve interpolator. With the corrected `band.dat` reader, the same
model gives **0.34 meV**. The model was always correct; the
comparison script was lying. The **identical naïve FFT** applied to
MgB2 on 18³ produces visually fine plots that match the wsvec-correct
ones — there the boundary R-vectors carry negligible weight.

## Why this happens

A coarse-grid Wannier model lives on a Wannier–Seitz cell of size
N_kx × N_ky × N_kz in real space. R-vectors that sit on the WS-cell
boundary appear in `hr.dat` with their full magnitude but are
**shared** between adjacent WS cells. The shared count is
`{seedname}_wsvec.dat`. Wannier90 reads both files and writes
`{seedname}_band.dat` correctly. A simple

```python
H(k) = Σ_R H(R) · exp(2πi k·R) / weight(R)
```

does **not** consult `wsvec.dat`. On a dense enough k-grid the boundary
R-vectors carry exponentially small `H(R)` and the error is negligible;
on coarse grids (e.g. 6³ or smaller for SOC systems) it dominates the
result.

This is documented in the Wannier90 manual, but it's an easy trap
because the naïve formula is the textbook expression for
"Fourier-transform the hr.dat back to k-space."

## How to do it correctly

### Option A (preferred). Use Wannier90's pre-computed band.dat

Run `wannier90.x` with `bands_plot = T` and a `kpoint_path` block in
the `.win` file. W90 writes:

- `{seedname}_band.dat`           — `(k_dist, energy)` pairs, one block per band
- `{seedname}_band.kpt`           — fractional k-points used
- `{seedname}_band.labelinfo.dat` — high-symmetry tick info

Then read them with the helper from this package:

```python
from lcao_wannier import read_w90_band_outputs

w90 = read_w90_band_outputs('my_seedname')
# w90 is a dict:
#   'kpoints_frac'   : (nk, 3) ndarray
#   'eigenvalues'    : (nk, num_wann) ndarray, eV
#   'distances'      : (nk,) ndarray of cumulative k-path distance
#   'tick_positions' : list[float]
#   'tick_labels'    : list[str]
#   'num_wann'       : int
```

The energies are returned in whatever reference frame Stage 2 wrote
`{seedname}.eig` in. By default the eig file is shifted to `E_F = 0`,
so the W90 band energies are already E_F-centered. **Do not subtract
the Fermi energy a second time.**

### Option B (advanced). Use `wannier90/postw90` `interp.dat` or `chk` directly

If you need eigenvalues at arbitrary k-points not on the W90 band
path, your options are:

- Call `wannier90.x` again with a new `kpoint_path` and read the new
  `band.dat`. This is the cleanest path.
- Build your own Hamiltonian interpolator that reads both `hr.dat` and
  `wsvec.dat`. Wannier90's `ws_distance.f90` is the reference
  implementation. This is not currently shipped in `lcao_wannier`.

### Option C (deprecated). Naïve `hr.dat` Fourier transform

There is a `wannier_bands(H_R, weights, kpoints_frac, num_wann)`
function in several legacy `compare_bands.py` scripts under
`calculations/`. **These now emit a `DeprecationWarning`** noting the
coarse-grid pitfall. The legacy scripts produced legitimate plots on
dense-grid systems (MgB2 18³, bismuth 12³ where the boundary R-vector
weights were already negligible), but on smaller grids they can
silently mislead. Switch to Option A for new work.

## Comparing LCAO to Wannier

Once you have W90's correctly-interpolated bands, compare against LCAO
on the **same** k-grid:

```python
import numpy as np
from lcao_wannier import read_w90_band_outputs, compute_band_structure
from lcao_wannier.parser import parse_overlap_and_fock_matrices, ...
# ... parse CRYSTAL output, build H_R, S_R as usual ...

w90 = read_w90_band_outputs('my_seedname')

# Compute LCAO at the SAME k-points W90 used (this is critical).
class _Path: pass
kpath = _Path()
kpath.kpoints_frac   = w90['kpoints_frac']
kpath.distances      = w90['distances']
kpath.tick_positions = w90['tick_positions']
kpath.tick_labels    = w90['tick_labels']
eigenvalues_lcao, _, _ = compute_band_structure(
    real_space_matrices, lattice_vectors, kpath)

# Subtract Fermi energy from LCAO (W90 is already shifted).
e_fermi = -5.116  # whatever Stage 1/2 used
eigenvalues_lcao = eigenvalues_lcao - e_fermi
```

Then plot both on the same x-axis and compute the RMS.

## How we caught this

During the CrI3 / MgB2 work in June 2026 a band-comparison plot showed
the spinlock3 CrI3 model interpolating poorly even though `Ω_total`
had dropped 2.7× and the per-WF spreads were all under 3 Å². Comparing
against the W90-produced band plot revealed that the Wannier model
was fine; only our comparison script's interpolator was broken. See
the session log in `calculations/CrI3/spinlock3_run/` for the worked
example.

## Files affected

The following scripts contain (now-deprecated) hr.dat FFT interpolators.
All of them emit a `DeprecationWarning` when called.

- `calculations/CrI3/compare_bands.py` — primary CrI3 comparator
- `calculations/MgB2/pdwf_pipeline/compare_bands.py` — multi-config MgB2
- `calculations/MgB2/pdwf_gc/compare_bands.py` — single-run MgB2
- `calculations/benchmarks/plot_band_comparison.py` — generic benchmark
- `scripts/archive/plot_alpha_sn_bands.py` — α-Sn (archived)
- `scripts/archive/plot_snte_bands.py`, `plot_snte_comparison.py` — SnTe (archived)

The canonical correct reader is `lcao_wannier.band_plot.read_w90_band_outputs`.
