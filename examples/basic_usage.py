"""
Basic Usage Example for LCAO-Wannier Package

This example demonstrates the simplest way to use the package
with synthetic test data.
"""

import numpy as np
import sys
import os

# Add parent directory to path for development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lcao_wannier import Wannier90Engine


def create_simple_model(num_orbitals=4):
    """Create a simple tight-binding model for demonstration."""
    # Simple cubic lattice
    lattice_vectors = np.eye(3) * 5.0  # 5 Angstrom lattice constant
    
    real_space_matrices = {}
    
    # Define R-vectors: on-site and nearest neighbors
    R_vectors = [
        (0, 0, 0),   # On-site
        (1, 0, 0), (-1, 0, 0),  # ±x neighbors
        (0, 1, 0), (0, -1, 0),  # ±y neighbors
        (0, 0, 1), (0, 0, -1),  # ±z neighbors
    ]
    
    for R in R_vectors:
        H = np.zeros((num_orbitals, num_orbitals), dtype=np.complex128)
        S = np.zeros((num_orbitals, num_orbitals), dtype=np.complex128)
        
        if R == (0, 0, 0):
            # On-site terms
            for i in range(num_orbitals):
                H[i, i] = -2.0 + 0.1 * i  # Different orbital energies
            S = np.eye(num_orbitals, dtype=np.complex128)  # Identity overlap
        else:
            # Hopping terms
            t = -0.5  # Hopping parameter
            for i in range(num_orbitals - 1):
                H[i, i+1] = t * (1 + 0.05j)
                H[i+1, i] = t * (1 - 0.05j)  # Hermitian conjugate
            
            # Small off-site overlap
            for i in range(num_orbitals):
                S[i, i] = 0.1
        
        real_space_matrices[R] = {'H': H, 'S': S}
    
    return real_space_matrices, lattice_vectors


def main():
    """Main execution function."""
    print("=" * 70)
    print("BASIC USAGE EXAMPLE - LCAO-WANNIER PACKAGE")
    print("=" * 70)
    
    # Create a simple model
    print("\nStep 1: Creating synthetic test data...")
    real_space_matrices, lattice_vectors = create_simple_model(num_orbitals=6)
    print(f"  ✓ Created {len(real_space_matrices)} R-vectors")
    print(f"  ✓ Matrix size: 6 × 6")
    
    # Initialize the engine
    print("\nStep 2: Initializing Wannier90 engine...")
    engine = Wannier90Engine(
        real_space_matrices=real_space_matrices,
        k_grid=(4, 4, 4),              # 4×4×4 k-point grid
        lattice_vectors=lattice_vectors,
        num_wann=4,                    # Extract 4 Wannier functions
        seedname='basic_example'       # Output file prefix
    )
    print("  ✓ Engine initialized")
    
    # Run the complete workflow
    print("\nStep 3: Running complete workflow...")
    results = engine.run(
        parallel=True,   # Use parallel processing
        verify=True      # Run verification checks
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  • basic_example.eig - Band energies at each k-point")
    print("  • basic_example.amn - Projection matrices")
    print("  • basic_example.mmn - Overlap matrices")
    print("\nVerification results:")
    print(f"  • Hermiticity deviation: {results['hermiticity']['H_deviation']:.2e}")
    print(f"  • Orthonormality deviation: {results['orthonormality']['max_deviation']:.2e}")
    print("=" * 70)


if __name__ == "__main__":
    main()
