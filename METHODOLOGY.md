# Computational Methodology

## Mathematical Framework for LCAO to Wannier90 Conversion

### 1. Problem Statement

Given a self-consistent LCAO calculation that provides:
- Real-space Hamiltonian matrices $H(\mathbf{R})$
- Real-space overlap matrices $S(\mathbf{R})$  
- Direct lattice vectors $\mathbf{a}_1, \mathbf{a}_2, \mathbf{a}_3$

Generate input files for Wannier90 to construct maximally localized Wannier functions.

### 2. Fourier Transformation to Momentum Space

#### 2.1 Monkhorst-Pack K-Point Grid

Generate a uniform k-point grid using the Monkhorst-Pack scheme:

$$\mathbf{k}_{ijk} = \frac{2i-n_1-1}{2n_1}\mathbf{b}_1 + \frac{2j-n_2-1}{2n_2}\mathbf{b}_2 + \frac{2k-n_3-1}{2n_3}\mathbf{b}_3$$

where:
- $(i,j,k) \in [1, n_1] \times [1, n_2] \times [1, n_3]$
- $\mathbf{b}_i$ are reciprocal lattice vectors

The implementation uses fractional coordinates and periodic boundary conditions.

#### 2.2 Fourier Transform

Transform matrices from real space to momentum space:

$$H(\mathbf{k}) = \sum_{\mathbf{R}} e^{i\mathbf{k}\cdot\mathbf{R}} H(\mathbf{R})$$

$$S(\mathbf{k}) = \sum_{\mathbf{R}} e^{i\mathbf{k}\cdot\mathbf{R}} S(\mathbf{R})$$

where the sum runs over all lattice vectors $\mathbf{R}$ in the real-space representation.

Phase factors are computed as:

$$\phi(\mathbf{k}, \mathbf{R}) = e^{i 2\pi \mathbf{k} \cdot \mathbf{R}}$$

with $\mathbf{k}$ in fractional coordinates and $\mathbf{R}$ in integer lattice coordinates.

### 3. Generalized Eigenvalue Problem

#### 3.1 Problem Formulation

At each k-point, solve the generalized eigenvalue problem:

$$H(\mathbf{k}) C(\mathbf{k}) = S(\mathbf{k}) C(\mathbf{k}) E(\mathbf{k})$$

where:
- $E(\mathbf{k})$ is a diagonal matrix of eigenvalues
- $C(\mathbf{k})$ is the matrix of eigenvectors (columns)

#### 3.2 Normalization

Eigenvectors are normalized according to:

$$C^\dagger(\mathbf{k}) S(\mathbf{k}) C(\mathbf{k}) = I$$

This ensures orthonormality with respect to the overlap metric.

#### 3.3 Band Selection

Select the lowest $N_w$ eigenvalues and corresponding eigenvectors:

$$E_n(\mathbf{k}), \quad n = 1, 2, \ldots, N_w$$

$$|\psi_n(\mathbf{k})\rangle = C_n(\mathbf{k}), \quad n = 1, 2, \ldots, N_w$$

### 4. Projection Matrices

#### 4.1 Initial Guess Construction

Construct projection matrices for Wannier90 using the LCAO basis as initial guesses:

$$A_{mn}(\mathbf{k}) = \langle \psi_m(\mathbf{k}) | g_n \rangle = [S(\mathbf{k}) C(\mathbf{k})]_{mn}$$

where:
- $|\psi_m(\mathbf{k})\rangle$ are the Bloch states
- $|g_n\rangle$ are localized orbital projections
- $m$ runs over bands, $n$ runs over Wannier functions

#### 4.2 Matrix Elements

For LCAO-based initial projections:

$$A(\mathbf{k}) = S(\mathbf{k})^\dagger C(\mathbf{k})$$

This provides the overlap between the computed Bloch states and the localized orbitals.

### 5. Overlap Matrices Between Adjacent K-Points

#### 5.1 Neighbor List Construction

For each k-point $\mathbf{k}$, identify six nearest neighbors:

$$\mathbf{k} + \mathbf{b}_i, \quad i = \pm x, \pm y, \pm z$$

with periodic boundary conditions:

$$\mathbf{k}' = \mathbf{k} + \mathbf{b} + \mathbf{G}$$

where $\mathbf{G}$ is a reciprocal lattice vector ensuring $\mathbf{k}' \in \text{BZ}$.

#### 5.2 Overlap Matrix Computation

Compute overlap matrices between adjacent k-points:

$$M_{mn}(\mathbf{k}, \mathbf{b}) = \langle \psi_m(\mathbf{k}) | \psi_n(\mathbf{k}+\mathbf{b}) \rangle$$

$$M(\mathbf{k}, \mathbf{b}) = C^\dagger(\mathbf{k}) S(\mathbf{k}+\mathbf{b}) C(\mathbf{k}+\mathbf{b})$$

For proper handling of periodic boundaries:

$$M(\mathbf{k}, \mathbf{b}) = C^\dagger(\mathbf{k}) \left[\sum_{\mathbf{R}} e^{i(\mathbf{k}+\mathbf{b})\cdot\mathbf{R}} S(\mathbf{R})\right] C(\mathbf{k}+\mathbf{b})$$

### 6. Numerical Implementation

#### 6.1 Hermiticity Preservation

Ensure Hermiticity of all matrices in k-space:

$$H(\mathbf{k}) = H^\dagger(\mathbf{k})$$

$$S(\mathbf{k}) = S^\dagger(\mathbf{k})$$

Verification:

$$\epsilon_H = \max_{ij} |H_{ij}(\mathbf{k}) - H_{ji}^*(\mathbf{k})| < 10^{-15}$$

#### 6.2 Orthonormality Verification

Verify eigenvector orthonormality:

$$\epsilon_{\text{ortho}} = \max_{ij} |[C^\dagger(\mathbf{k}) S(\mathbf{k}) C(\mathbf{k})]_{ij} - \delta_{ij}| < 10^{-14}$$

#### 6.3 Eigenvalue Solver

Use scipy.linalg.eigh for the generalized eigenvalue problem:
- Exploits Hermitian structure
- Returns eigenvalues in ascending order
- Provides normalized eigenvectors satisfying $C^\dagger S C = I$

### 7. Parallelization Strategy

#### 7.1 Embarrassingly Parallel Structure

The k-point loop is embarrassingly parallel:

$$\{H(\mathbf{k}_i), S(\mathbf{k}_i)\} \rightarrow \{E_n(\mathbf{k}_i), C_n(\mathbf{k}_i)\}$$

Each k-point can be processed independently.

#### 7.2 Implementation

Use Python multiprocessing for parallel execution:
```python
with multiprocessing.Pool(num_processes) as pool:
    results = pool.starmap(solve_single_kpoint, kpoint_tasks)
```

Expected speedup:

$$S(p) \approx p \quad \text{for} \quad p \leq N_k$$

where $p$ is the number of processes and $N_k$ is the number of k-points.

### 8. File Format Specifications

#### 8.1 Energy Eigenvalue File (.eig)

Format per line:
```
band_index  k_index  energy_eV
```

Total lines: $N_b \times N_k$

#### 8.2 Projection Matrix File (.amn)

Header:
```
num_bands  num_kpoints  num_wann
```

Data per entry:
```
band  wann  k_index  A_real  A_imag
```

Total entries: $N_b \times N_w \times N_k$

#### 8.3 Overlap Matrix File (.mmn)

Header:
```
num_bands  num_kpoints  num_neighbors
```

Per k-point neighbor:
```
k_index  neighbor_k_index  b1 b2 b3
M_11_real  M_11_imag
M_12_real  M_12_imag
...
M_nn_real  M_nn_imag
```

Total neighbor entries: $N_k \times 6$
Matrix elements per entry: $N_w \times N_w$

### 9. Computational Complexity

#### 9.1 Time Complexity

Per k-point:
- Fourier transform: $O(N_R N^2)$ where $N_R$ is number of R-vectors, $N$ is number of orbitals
- Eigenvalue solve: $O(N^3)$
- Projection matrices: $O(N^2 N_w)$
- Overlap matrices: $O(N^2 N_w)$

Total for all k-points (sequential):

$$T_{\text{total}} = N_k [O(N_R N^2) + O(N^3) + O(N^2 N_w)]$$

With parallelization over $p$ processes:

$$T_{\text{parallel}} \approx \frac{T_{\text{total}}}{p}$$

#### 9.2 Memory Complexity

Storage requirements:
- Real-space matrices: $O(N_R N^2)$
- k-space matrices: $O(N_k N^2)$
- Eigenvectors: $O(N_k N N_w)$
- Total: $O(N_R N^2 + N_k N^2 + N_k N N_w)$

For typical systems:
- $N_R \sim 10-100$
- $N_k \sim 100-1000$
- $N \sim 50-500$
- $N_w \sim 10-100$

Memory estimate: 

$$M \approx 16 \times (100 \times 250^2 + 512 \times 250^2) \text{ bytes} \approx 20 \text{ GB}$$

for a moderately large system.

### 10. Numerical Precision

All computations use:
- Complex128 (double precision complex) arithmetic
- Absolute error tolerance: $10^{-15}$
- Relative error tolerance: $10^{-12}$

Key numerical checks:
1. Hermiticity: $\max|H - H^\dagger| < 10^{-15}$
2. Orthonormality: $\max|C^\dagger S C - I| < 10^{-14}$
3. Real eigenvalues: $\max|\text{Im}(E)| < 10^{-15}$
4. Energy conservation: variance in total energy < $10^{-10}$ eV

### 11. References

1. Marzari, N., and Vanderbilt, D. "Maximally localized generalized Wannier functions for composite energy bands." Physical Review B 56.20 (1997): 12847.

2. Souza, I., et al. "Maximally localized Wannier functions for entangled energy bands." Physical Review B 65.3 (2001): 035109.

3. Mostofi, A. A., et al. "wannier90: A tool for obtaining maximally-localised Wannier functions." Computer Physics Communications 178.9 (2008): 685-699.

4. Monkhorst, H. J., and Pack, J. D. "Special points for Brillouin-zone integrations." Physical Review B 13.12 (1976): 5188.
