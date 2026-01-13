# Band Structure Plotting with Wannier90

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

### Method 3: Using the wannier90 Python API (if installed)

```python
from wannier90 import w90

# Load results
w90_data = w90.W90(seedname='bismuth_final')
w90_data.plot_bands()
```

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

## Comparing with DFT Bands

To compare your Wannier-interpolated bands with the original DFT calculation:

1. **Generate DFT bands** along the same path in CRYSTAL
2. **Plot both** on the same graph:
   - Wannier bands: solid lines
   - DFT bands: circles or dots

Good agreement indicates your Wannier functions accurately represent the electronic structure!

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

