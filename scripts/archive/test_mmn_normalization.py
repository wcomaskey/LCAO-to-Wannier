#!/usr/bin/env python3
"""Quick test to check if selected eigenvectors maintain S-orthonormality."""

import numpy as np
from lcao_wannier import (
    parse_calculation_parameters,
    parse_overlap_and_fock_matrices,
    parse_atomic_basis_info,
    create_spin_block_matrices,
    prepare_real_space_matrices,
    Wannier90Engine
)

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

print("Solving eigenvalue problems (serial)...")
engine.solve_all_kpoints(parallel=False, convert_to_eV=False)
print(f"✓ Solved {len(engine.eigenvalues_list)} k-points\n")

# Now test: select bands 40-55 (like in the actual run)
band_indices = list(range(40, 56))  # 16 bands
print(f"Testing band selection: indices {band_indices[0]}-{band_indices[-1]}")
print(f"Number of selected bands: {len(band_indices)}\n")

# Check S-orthonormality for several k-points
print("="*80)
print("CHECKING S-ORTHONORMALITY OF SELECTED EIGENVECTORS")
print("="*80)

for k_idx in [0, len(engine.kpoints)//2, len(engine.kpoints)-1]:
    C_full = engine.eigenvectors_list[k_idx]
    S_k = engine.S_k_list[k_idx]

    # Select bands
    C_sel = C_full[:, band_indices]

    # Compute C†SC for selected bands
    CtSC = C_sel.conj().T @ S_k @ C_sel

    # Check deviation from identity
    identity = np.eye(len(band_indices))
    deviation = np.max(np.abs(CtSC - identity))

    diagonal = np.diag(CtSC).real
    off_diag_max = np.max(np.abs(CtSC - np.diag(diagonal)))

    print(f"\nK-point {k_idx}:")
    print(f"  C_sel† S C_sel diagonal range: [{diagonal.min():.10f}, {diagonal.max():.10f}]")
    print(f"  Off-diagonal max: {off_diag_max:.2e}")
    print(f"  Max deviation from identity: {deviation:.2e}")

    if deviation > 1e-10:
        print(f"  ⚠ WARNING: Selected eigenvectors NOT S-orthonormal!")
        print(f"    This will cause MMN non-unitarity!")

print("\n" + "="*80)
print("If selected eigenvectors ARE S-orthonormal, the issue is elsewhere.")
print("If NOT, we need to re-orthonormalize after band selection.")
print("="*80)
