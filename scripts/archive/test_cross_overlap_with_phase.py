#!/usr/bin/env python3
"""Test cross-overlap + phase correction to debug MMN non-unitarity."""

import numpy as np
from lcao_wannier import (
    parse_calculation_parameters,
    parse_overlap_and_fock_matrices,
    create_spin_block_matrices,
    prepare_real_space_matrices,
    Wannier90Engine,
    parse_atomic_basis_info
)
from lcao_wannier.fourier import compute_cross_overlap

# Read input
input_file = "tests/Bismuth_basis_40.out"
with open(input_file, 'r') as f:
    lines = f.readlines()

params = parse_calculation_parameters(lines)
raw_matrices, lattice_vectors_list = parse_overlap_and_fock_matrices(lines)
lattice_vectors = np.array(lattice_vectors_list)

# Parse atomic info
atomic_info = parse_atomic_basis_info(lines)
atom_positions = atomic_info.atom_positions
basis_atom_map_original = atomic_info.basis_atom_map

# For SOC, double the basis_atom_map
num_basis = len(basis_atom_map_original)
basis_atom_map = np.concatenate([basis_atom_map_original, basis_atom_map_original])
print(f"Found {len(atom_positions)} atoms, {num_basis} basis functions (SOC: doubled to {len(basis_atom_map)} orbitals)")

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

print("="*80)
print("TESTING CROSS-OVERLAP + PHASE CORRECTION")
print("="*80)

# Test at k-point 1
k_idx = 1
k1 = engine.kpoints[k_idx]
C1 = engine.eigenvectors_list[k_idx][:, band_indices]

# Neighbor: k2 = k1 + (1/15, 0, 0)
k2_idx = 2
k2 = engine.kpoints[k2_idx]
C2 = engine.eigenvectors_list[k2_idx][:, band_indices]

# Compute b-vector in Cartesian coordinates
b_vec_frac = k2 - k1
b_vec_cart = lattice_vectors.T @ (2 * np.pi * b_vec_frac)  # Note: 2π convention for Cartesian

print(f"\nK-point {k_idx}: k1 = {k1}")
print(f"K-point {k2_idx}: k2 = {k2}")
print(f"b_vec_frac = {b_vec_frac}")
print(f"b_vec_cart = {b_vec_cart}")

# Compute cross-overlap
S_cross = compute_cross_overlap(k1, k2, real_space_matrices, lattice_vectors, convention='pi')

print("\n" + "-"*80)
print("WITHOUT phase correction:")
M_no_phase = C1.conj().T @ S_cross @ C2
MdagM_no_phase = M_no_phase.conj().T @ M_no_phase
diag_no_phase = np.diag(MdagM_no_phase).real
print(f"  M†M diagonal range: [{diag_no_phase.min():.10f}, {diag_no_phase.max():.10f}]")
print(f"  det(M) = {np.linalg.det(M_no_phase):.6f}")
print(f"  Max deviation from identity: {np.max(np.abs(MdagM_no_phase - np.eye(len(band_indices)))):.6e}")

print("\n" + "-"*80)
print("WITH phase correction:")

# Apply phase correction
relevant_atom_pos = atom_positions[basis_atom_map]
dot_products = np.dot(relevant_atom_pos, b_vec_cart)
phase_factors = np.exp(-1j * dot_products)

print(f"  Atomic positions (first 5):")
for i in range(min(5, len(atom_positions))):
    print(f"    Atom {i}: {atom_positions[i]}")
print(f"  Dot products b·τ (first 10): {dot_products[:10]}")
print(f"  Phase factor magnitudes (should all be 1.0): {np.abs(phase_factors[:10])}")
print(f"  Phase factor phases (first 10): {np.angle(phase_factors[:10])}")
print(f"  Are all phases ~1.0? {np.allclose(phase_factors, 1.0)}")

S_phase = phase_factors[:, np.newaxis] * S_cross

M_with_phase = C1.conj().T @ S_phase @ C2
MdagM_with_phase = M_with_phase.conj().T @ M_with_phase
diag_with_phase = np.diag(MdagM_with_phase).real
print(f"  M†M diagonal range: [{diag_with_phase.min():.10f}, {diag_with_phase.max():.10f}]")
print(f"  det(M) = {np.linalg.det(M_with_phase):.6f}")
print(f"  Max deviation from identity: {np.max(np.abs(MdagM_with_phase - np.eye(len(band_indices)))):.6e}")

if np.max(np.abs(MdagM_with_phase - np.eye(len(band_indices)))) < 1e-3:
    print(f"\n  ✓ MMN is unitary with phase correction!")
else:
    print(f"\n  ⚠ MMN still not unitary even with phase correction!")

print("\n" + "="*80)
