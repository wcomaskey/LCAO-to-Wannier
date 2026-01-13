#!/usr/bin/env python3
"""Test cross-overlap for special cases to verify correctness."""

import numpy as np
from lcao_wannier import (
    parse_calculation_parameters,
    parse_overlap_and_fock_matrices,
    create_spin_block_matrices,
    prepare_real_space_matrices,
    Wannier90Engine
)
from lcao_wannier.fourier import compute_cross_overlap, fourier_transform_to_kspace

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

print("="*80)
print("TESTING CROSS-OVERLAP SPECIAL CASES")
print("="*80)

# Test 1: k1 = k2 (should give same-k overlap)
k_idx = 10
k1 = engine.kpoints[k_idx]
k2 = k1.copy()

print(f"\nTest 1: k1 = k2 = {k1}")
print("-"*80)

# Compute via standard Fourier transform
H_standard, S_standard = fourier_transform_to_kspace(k1, real_space_matrices, lattice_vectors)

# Compute via cross-overlap
S_cross = compute_cross_overlap(k1, k2, real_space_matrices, lattice_vectors, convention='pi')

# Compare
diff = np.max(np.abs(S_standard - S_cross))
print(f"  S(k) via standard FT: shape {S_standard.shape}")
print(f"  S(k,k) via cross-overlap: shape {S_cross.shape}")
print(f"  Max difference: {diff:.6e}")

if diff < 1e-10:
    print(f"  ✓ Cross-overlap correctly reduces to same-k overlap when k1=k2")
else:
    print(f"  ⚠ ERROR: Cross-overlap does NOT match same-k overlap!")

# Test 2: Check hermiticity S(k1, k2) = S(k2, k1)†
k1_idx = 10
k2_idx = 11
k1 = engine.kpoints[k1_idx]
k2 = engine.kpoints[k2_idx]

print(f"\nTest 2: Hermiticity check")
print(f"  k1 = {k1}")
print(f"  k2 = {k2}")
print("-"*80)

S_12 = compute_cross_overlap(k1, k2, real_space_matrices, lattice_vectors, convention='pi')
S_21 = compute_cross_overlap(k2, k1, real_space_matrices, lattice_vectors, convention='pi')

hermiticity_error = np.max(np.abs(S_12 - S_21.conj().T))
print(f"  Max |S(k1,k2) - S(k2,k1)†|: {hermiticity_error:.6e}")

if hermiticity_error < 1e-10:
    print(f"  ✓ Cross-overlap is hermitian")
else:
    print(f"  ⚠ WARNING: Cross-overlap is not hermitian!")

print("\n" + "="*80)
