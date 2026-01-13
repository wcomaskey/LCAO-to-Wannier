#!/usr/bin/env python3
"""
Diagnostic script to check AMN and MMN matrix properties.

This helps identify issues that could cause negative Omega_I.
"""

import numpy as np
import sys

def read_amn_file(filename):
    """Read .amn file and return projection matrices."""
    with open(filename, 'r') as f:
        # Skip header
        f.readline()

        # Read dimensions
        line = f.readline().split()
        num_bands = int(line[0])
        num_kpoints = int(line[1])
        num_wann = int(line[2])

        print(f"AMN dimensions: {num_bands} bands, {num_kpoints} k-points, {num_wann} projections")

        # Read projection matrices
        A_matrices = []
        for k in range(num_kpoints):
            A_k = np.zeros((num_wann, num_bands), dtype=complex)
            for m in range(num_bands):
                for n in range(num_wann):
                    line = f.readline().split()
                    # Format: band_idx wannier_idx kpoint_idx Re(A) Im(A)
                    A_k[n, m] = float(line[3]) + 1j * float(line[4])
            A_matrices.append(A_k)

        return np.array(A_matrices), num_bands, num_kpoints, num_wann

def read_mmn_file(filename):
    """Read .mmn file and return overlap matrices."""
    with open(filename, 'r') as f:
        # Skip header
        f.readline()

        # Read dimensions
        line = f.readline().split()
        num_bands = int(line[0])
        num_kpoints = int(line[1])
        num_neighbors = int(line[2])

        print(f"MMN dimensions: {num_bands} bands, {num_kpoints} k-points, {num_neighbors} neighbors")

        # Read overlap matrices
        M_matrices = []
        for k in range(num_kpoints):
            M_k_neighbors = []
            for nb in range(num_neighbors):
                # Skip neighbor identification line
                f.readline()

                # Read matrix
                M_kb = np.zeros((num_bands, num_bands), dtype=complex)
                for n in range(num_bands):
                    for m in range(num_bands):
                        line = f.readline().split()
                        M_kb[m, n] = float(line[0]) + 1j * float(line[1])

                M_k_neighbors.append(M_kb)
            M_matrices.append(M_k_neighbors)

        return M_matrices, num_bands, num_kpoints, num_neighbors

def check_amn_properties(A_matrices):
    """Check if AMN matrices have reasonable properties."""
    print("\n" + "="*80)
    print("AMN MATRIX DIAGNOSTICS")
    print("="*80)

    num_kpoints, num_wann, num_bands = A_matrices.shape

    # Check 1: Magnitude range
    magnitudes = np.abs(A_matrices)
    print(f"\nProjection magnitude statistics:")
    print(f"  Min: {magnitudes.min():.6e}")
    print(f"  Max: {magnitudes.max():.6e}")
    print(f"  Mean: {magnitudes.mean():.6e}")
    print(f"  Std: {magnitudes.std():.6e}")

    if magnitudes.max() > 10.0:
        print(f"  ⚠ WARNING: Very large projections detected!")

    # Check 2: Are projections normalized?
    # For each band at each k-point, sum |A_mn|^2 over projections
    for k_idx in [0, num_kpoints//2, num_kpoints-1]:
        A_k = A_matrices[k_idx]
        norms = np.sum(np.abs(A_k)**2, axis=0)  # Sum over projections for each band
        print(f"\nK-point {k_idx}: Projection norms per band")
        print(f"  Min: {norms.min():.6f}")
        print(f"  Max: {norms.max():.6f}")
        print(f"  Mean: {norms.mean():.6f}")

        if norms.max() > 2.0 or norms.min() < 0.01:
            print(f"  ⚠ WARNING: Unusual normalization!")

    # Check 3: Orthogonality of projections
    # A†A should be somewhat diagonal
    for k_idx in [0, num_kpoints//2]:
        A_k = A_matrices[k_idx]
        AtA = A_k.conj().T @ A_k  # Shape: (num_bands, num_bands)
        diagonal = np.diag(AtA).real
        off_diagonal = np.abs(AtA - np.diag(diagonal))

        print(f"\nK-point {k_idx}: A†A matrix")
        print(f"  Diagonal range: [{diagonal.min():.4f}, {diagonal.max():.4f}]")
        print(f"  Off-diagonal max: {off_diagonal.max():.6f}")

def check_mmn_properties(M_matrices):
    """Check if MMN matrices have reasonable properties."""
    print("\n" + "="*80)
    print("MMN MATRIX DIAGNOSTICS")
    print("="*80)

    num_kpoints = len(M_matrices)
    num_neighbors = len(M_matrices[0])
    num_bands = M_matrices[0][0].shape[0]

    # Check 1: Unitarity - M†M should be approximately identity
    print(f"\nUnitarity check (M†M should ≈ I):")

    max_deviation = 0
    for k_idx in range(min(5, num_kpoints)):
        for nb_idx in range(num_neighbors):
            M = M_matrices[k_idx][nb_idx]
            MdagM = M.conj().T @ M

            # Check deviation from identity
            identity = np.eye(num_bands)
            deviation = np.max(np.abs(MdagM - identity))
            max_deviation = max(max_deviation, deviation)

            if k_idx < 2 and nb_idx < 2:
                diag = np.diag(MdagM).real
                print(f"  K-point {k_idx}, neighbor {nb_idx}:")
                print(f"    Diagonal range: [{diag.min():.6f}, {diag.max():.6f}]")
                print(f"    Off-diagonal max: {deviation:.6e}")

    print(f"\n  Maximum deviation from unitarity: {max_deviation:.6e}")
    if max_deviation > 0.01:
        print(f"  ⚠ WARNING: MMN matrices are not unitary!")

    # Check 2: Determinant (should be ~1 for unitary matrices)
    print(f"\nDeterminant check (should be ≈ 1 with |det|=1):")
    determinants = []
    for k_idx in range(min(10, num_kpoints)):
        for nb_idx in range(num_neighbors):
            M = M_matrices[k_idx][nb_idx]
            det = np.linalg.det(M)
            determinants.append(det)

    determinants = np.array(determinants)
    det_mags = np.abs(determinants)
    det_phases = np.angle(determinants)

    print(f"  |det(M)| statistics:")
    print(f"    Min: {det_mags.min():.6f}")
    print(f"    Max: {det_mags.max():.6f}")
    print(f"    Mean: {det_mags.mean():.6f}")

    if det_mags.max() > 1.1 or det_mags.min() < 0.9:
        print(f"  ⚠ WARNING: Determinants deviate significantly from 1!")

    # Check 3: Check for NaN or Inf
    all_M = np.array([M_matrices[k][nb] for k in range(num_kpoints) for nb in range(num_neighbors)])
    if np.any(np.isnan(all_M)):
        print(f"\n  ⚠ ERROR: NaN values detected in MMN matrices!")
    if np.any(np.isinf(all_M)):
        print(f"\n  ⚠ ERROR: Inf values detected in MMN matrices!")

def compare_with_identity(M_matrices):
    """Check if any MMN matrices are accidentally identity."""
    print("\n" + "="*80)
    print("CHECKING FOR IDENTITY MATRICES")
    print("="*80)

    num_kpoints = len(M_matrices)
    num_neighbors = len(M_matrices[0])
    num_bands = M_matrices[0][0].shape[0]
    identity = np.eye(num_bands)

    identity_count = 0
    for k_idx in range(num_kpoints):
        for nb_idx in range(num_neighbors):
            M = M_matrices[k_idx][nb_idx]
            deviation = np.max(np.abs(M - identity))

            if deviation < 1e-6:
                identity_count += 1
                if identity_count <= 5:
                    print(f"  K-point {k_idx}, neighbor {nb_idx}: Nearly identity (dev={deviation:.2e})")

    total_matrices = num_kpoints * num_neighbors
    print(f"\nFound {identity_count}/{total_matrices} identity matrices ({100*identity_count/total_matrices:.1f}%)")

    if identity_count > 0.5 * total_matrices:
        print("  ⚠ WARNING: Too many identity matrices - possible phase correction issue!")

def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_wannier_matrices.py <seedname>")
        print("  Will check <seedname>.amn and <seedname>.mmn")
        sys.exit(1)

    seedname = sys.argv[1]
    amn_file = f"{seedname}.amn"
    mmn_file = f"{seedname}.mmn"

    print("="*80)
    print(f"WANNIER MATRIX DIAGNOSTICS FOR: {seedname}")
    print("="*80)

    try:
        # Read and check AMN
        A_matrices, num_bands, num_kpoints, num_wann = read_amn_file(amn_file)
        check_amn_properties(A_matrices)

        # Read and check MMN
        M_matrices, num_bands_mmn, num_kpoints_mmn, num_neighbors = read_mmn_file(mmn_file)
        check_mmn_properties(M_matrices)
        compare_with_identity(M_matrices)

        print("\n" + "="*80)
        print("DIAGNOSTICS COMPLETE")
        print("="*80)

    except FileNotFoundError as e:
        print(f"\nError: File not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
