#!/usr/bin/env python3
"""
Check if eigenvectors are properly normalized in the overlap metric.

For LCAO with non-orthogonal basis, eigenvectors should satisfy:
    C†(k) S(k) C(k) = I

This checks if our eigenvectors satisfy this property.
"""

import numpy as np
from lcao_wannier import parse_calculation_parameters, parse_overlap_and_fock_matrices
from lcao_wannier import create_spin_block_matrices, prepare_real_space_matrices
from lcao_wannier import Wannier90Engine

# Read CRYSTAL output
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
    mat_type = mat_info['type']
    data = mat_info['data']
    if mat_type == 'overlap':
        S_R_dict[R] = data
    elif mat_type == 'fock':
        spin_channel = mat_info.get('spin_channel', 0)
        if R not in H_R_dict:
            H_R_dict[R] = {}
        H_R_dict[R][spin_channel] = data

# Create SOC matrices
has_soc = params.has_soc
num_basis = len(raw_matrices[0]['data'])

if has_soc:
    H_full_list, S_full_list = create_spin_block_matrices(
        H_R_dict, S_R_dict, num_basis, lattice_vectors_list
    )
else:
    raise NotImplementedError("Non-SOC case not implemented in this diagnostic")

# Prepare real-space matrices
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

# Solve eigenvalue problems
print("Solving eigenvalue problems...")
engine.solve_all_kpoints(parallel=False, convert_to_eV=False)  # Keep in Hartree, no parallel
print(f"✓ Solved {len(engine.eigenvalues_list)} k-points")

# Check normalization at several k-points
print("\n" + "="*80)
print("EIGENVECTOR NORMALIZATION CHECK")
print("="*80)
print("\nFor LCAO with non-orthogonal basis, should have: C†(k) S(k) C(k) = I")

for k_idx in [0, len(engine.kpoints)//2, len(engine.kpoints)-1]:
    C_k = engine.eigenvectors_list[k_idx]
    S_k = engine.S_k_list[k_idx]

    # Compute C†  S C
    CtSC = C_k.conj().T @ S_k @ C_k

    # Check deviation from identity
    num_bands = C_k.shape[1]
    identity = np.eye(num_bands)
    deviation = np.max(np.abs(CtSC - identity))

    diagonal = np.diag(CtSC).real
    off_diag_max = np.max(np.abs(CtSC - np.diag(diagonal)))

    print(f"\nK-point {k_idx}:")
    print(f"  C†SC diagonal range: [{diagonal.min():.8f}, {diagonal.max():.8f}]")
    print(f"  C†SC off-diagonal max: {off_diag_max:.2e}")
    print(f"  Max deviation from identity: {deviation:.2e}")

    if deviation > 1e-6:
        print(f"  ⚠ WARNING: Eigenvectors are NOT properly normalized!")
        print(f"  First 5 diagonal elements of C†SC:")
        print(f"    {diagonal[:5]}")

print("\n" + "="*80)
print("If eigenvectors are NOT normalized (diagonal != 1), this explains")
print("why MMN matrices are not unitary!")
print("="*80)
