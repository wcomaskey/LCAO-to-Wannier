#!/usr/bin/env python3
"""Test cross-overlap computation to debug MMN non-unitarity."""

import numpy as np
from lcao_wannier import (
    parse_calculation_parameters,
    parse_overlap_and_fock_matrices,
    create_spin_block_matrices,
    prepare_real_space_matrices,
    Wannier90Engine
)
from lcao_wannier.fourier import compute_cross_overlap

# Read input
input_file = "tests/Bismuth_basis_40.out"
with open(input_file, 'r') as f:
    lines = f.readlines()

params = parse_calculation_parameters(lines)
raw_matrices, lattice_vectors_list = parse_overlap_and_fock_matrices(lines)
lattice_vectors = np.array(lattice_vectors_list)

# Organize matrices
H_R_dict = {}
S_R_dict = {}
for mat_info in raw_matrices:
    R = tuple(mat_info['lattice_vector'])
    if mat_info['type'] == 'overlap':
        S_R_dict[R] = mat_info['data']
    elif mat_info['type'] == 'fock':
        spin_channel = mat_info.get('spin_channel', 0)
        if R not in H_R_dict:
            H_R_dict[R] = {}
        H_R_dict[R][spin_channel] = mat_info['data']

# Create SOC matrices
num_basis = len(raw_matrices[0]['data'])
H_full_list, S_full_list = create_spin_block_matrices(
    H_R_dict, S_R_dict, num_basis, lattice_vectors_list
)

real_space_matrices = prepare_real_space_matrices(
    H_full_list, S_full_list, lattice_vectors
)

# Initialize engine
engine = Wannier90Engine(
    real_space_matrices=real_space_matrices,
    k_grid=params.k_grid,
    lattice_vectors=lattice_vectors,
    seedname="test",
    num_wann=None,
    outer_window=None,
    e_fermi=params.fermi_energy,
)

print("Solving eigenvalue problems...")
engine.solve_all_kpoints(parallel=False, convert_to_eV=False)
print(f"✓ Solved {len(engine.eigenvalues_list)} k-points\n")

# Select bands
band_indices = list(range(40, 56))  # 16 bands
print(f"Testing bands {band_indices[0]}-{band_indices[-1]} ({len(band_indices)} bands)\n")

print("="*80)
print("TESTING CROSS-OVERLAP COMPUTATION")
print("="*80)

# Test at a non-Gamma k-point
k_idx_test = 1
k1 = engine.kpoints[k_idx_test]
C1 = engine.eigenvectors_list[k_idx_test][:, band_indices]
S1 = engine.S_k_list[k_idx_test]

# Check if C1 is S-orthonormal
C1_check = C1.conj().T @ S1 @ C1
diag1 = np.diag(C1_check).real
print(f"\nK-point {k_idx_test}: k = {k1}")
print(f"  C†SC diagonal range: [{diag1.min():.10f}, {diag1.max():.10f}]")
print(f"  → Eigenvectors are S-orthonormal: {np.allclose(C1_check, np.eye(len(band_indices)))}")

# Pick a neighbor (let's use b = (1/15, 0, 0))
k2_idx = (k_idx_test + 1) % len(engine.kpoints)
k2 = engine.kpoints[k2_idx]
C2 = engine.eigenvectors_list[k2_idx][:, band_indices]
S2 = engine.S_k_list[k2_idx]

print(f"\nNeighbor k-point {k2_idx}: k = {k2}")
print(f"  k_diff = k2 - k1 = {k2 - k1}")

# Check if C2 is S-orthonormal
C2_check = C2.conj().T @ S2 @ C2
diag2 = np.diag(C2_check).real
print(f"  C†SC diagonal range: [{diag2.min():.10f}, {diag2.max():.10f}]")
print(f"  → Eigenvectors are S-orthonormal: {np.allclose(C2_check, np.eye(len(band_indices)))}")

# Compute cross-overlap S(k1, k2)
print(f"\n" + "-"*80)
print("Computing cross-overlap S(k1, k2) using Fourier transform of S(R):")
S_cross = compute_cross_overlap(k1, k2, real_space_matrices, lattice_vectors, convention='pi')

# Compute MMN matrix: M = C1† S_cross C2
M = C1.conj().T @ S_cross @ C2

# Check unitarity: M† M should be identity
MdagM = M.conj().T @ M
diag_M = np.diag(MdagM).real
off_diag_M = np.max(np.abs(MdagM - np.diag(diag_M)))

print(f"\nMMN matrix properties:")
print(f"  M†M diagonal range: [{diag_M.min():.10f}, {diag_M.max():.10f}]")
print(f"  M†M off-diagonal max: {off_diag_M:.6e}")
print(f"  det(M) = {np.linalg.det(M):.6f}")

identity_dev = np.max(np.abs(MdagM - np.eye(len(band_indices))))
print(f"  Max deviation from identity: {identity_dev:.6e}")

if identity_dev > 1e-3:
    print(f"\n  ⚠ WARNING: MMN is NOT unitary!")
    print(f"\n  This suggests an error in the cross-overlap computation.")
else:
    print(f"\n  ✓ MMN is unitary - cross-overlap is correct!")

print("\n" + "-"*80)
print("Comparing with WRONG formula: M_wrong = C1† S2 C2")
M_wrong = C1.conj().T @ S2 @ C2
MdagM_wrong = M_wrong.conj().T @ M_wrong
diag_wrong = np.diag(MdagM_wrong).real
print(f"  M_wrong†M_wrong diagonal range: [{diag_wrong.min():.10f}, {diag_wrong.max():.10f}]")
print(f"  det(M_wrong) = {np.linalg.det(M_wrong):.6f}")

print("\n" + "="*80)
