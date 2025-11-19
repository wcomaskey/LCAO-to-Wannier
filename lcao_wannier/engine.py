"""
Main Engine Module

This module contains the Wannier90Engine class that coordinates all
operations for converting LCAO calculations to Wannier90 format.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import warnings

from .kpoints import generate_kpoint_grid, generate_neighbor_list
from .solver import solve_all_kpoints_sequential, solve_all_kpoints_parallel
from .wannier90 import write_wannier90_files
from .verification import run_all_verifications
from .utils import check_matrix_consistency, print_calculation_info


class Wannier90Engine:
    """
    Main computational engine for LCAO-to-Wannier90 conversion.
    
    This class coordinates the entire workflow:
    1. K-point grid generation
    2. Fourier transforms and eigenvalue solving
    3. Verification of results
    4. Writing Wannier90 input files
    
    Attributes
    ----------
    real_space_matrices : dict
        Real-space H(R) and S(R) matrices
    k_grid : tuple of 3 ints
        Dimensions of k-point grid
    lattice_vectors : ndarray
        Real-space lattice vectors
    num_wann : int
        Number of Wannier functions
    seedname : str
        Prefix for output files
    kpoints : ndarray
        Generated k-point grid
    num_kpoints : int
        Total number of k-points
    num_orbitals : int
        Number of orbitals
    neighbor_list : dict
        K-point neighbor connectivity
    eigenvalues_list : list
        Computed eigenvalues for each k-point
    eigenvectors_list : list
        Computed eigenvectors for each k-point
    H_k_list : list
        Hamiltonian matrices at each k-point
    S_k_list : list
        Overlap matrices at each k-point
    
    Examples
    --------
    >>> engine = Wannier90Engine(
    ...     real_space_matrices=matrices,
    ...     k_grid=(4, 4, 4),
    ...     lattice_vectors=lattice_vecs,
    ...     num_wann=10,
    ...     seedname='material'
    ... )
    >>> engine.run(parallel=True, verify=True)
    """
    
    def __init__(
        self,
        real_space_matrices: Dict[Tuple[int, int, int], Dict[str, np.ndarray]],
        k_grid: Tuple[int, int, int],
        lattice_vectors: np.ndarray,
        num_wann: int,
        seedname: str = "wannier90"
    ):
        """
        Initialize the Wannier90 engine.
        
        Parameters
        ----------
        real_space_matrices : dict
            Maps (R1, R2, R3) -> {'H': H_matrix, 'S': S_matrix}
        k_grid : tuple of 3 ints
            Dimensions of the k-point grid
        lattice_vectors : ndarray of shape (3, 3)
            Real-space lattice vectors (rows)
        num_wann : int
            Number of Wannier functions
        seedname : str, optional
            Prefix for output files
        """
        self.real_space_matrices = real_space_matrices
        self.k_grid = k_grid
        self.lattice_vectors = lattice_vectors
        self.num_wann = num_wann
        self.seedname = seedname
        
        # Check matrix consistency
        if not check_matrix_consistency(real_space_matrices):
            raise ValueError("Inconsistent matrix dimensions in real_space_matrices")
        
        # Get number of orbitals
        first_key = next(iter(real_space_matrices))
        self.num_orbitals = real_space_matrices[first_key]['H'].shape[0]
        
        # Validate num_wann
        if num_wann > self.num_orbitals:
            raise ValueError(
                f"num_wann ({num_wann}) cannot exceed num_orbitals ({self.num_orbitals})"
            )
        
        # Generate k-point grid and neighbor list
        self.kpoints = generate_kpoint_grid(k_grid)
        self.num_kpoints = len(self.kpoints)
        self.neighbor_list = generate_neighbor_list(k_grid)
        
        # Storage for results (initialized as empty)
        self.eigenvalues_list = []
        self.eigenvectors_list = []
        self.H_k_list = []
        self.S_k_list = []
        
        # Print initialization info
        self._print_init_info()
    
    def _print_init_info(self):
        """Print initialization information."""
        print("\n" + "=" * 70)
        print("LCAO-to-Wannier90 Engine Initialized")
        print("=" * 70)
        print(f"Seedname: {self.seedname}")
        print(f"K-grid: {self.k_grid}")
        print(f"Number of k-points: {self.num_kpoints}")
        print(f"Number of orbitals: {self.num_orbitals}")
        print(f"Number of Wannier functions: {self.num_wann}")
        print(f"Number of R-vectors: {len(self.real_space_matrices)}")
        print("=" * 70)
    
    def solve_all_kpoints(
        self,
        parallel: bool = True,
        num_processes: Optional[int] = None
    ) -> None:
        """
        Solve the generalized eigenvalue problem for all k-points.
        
        Parameters
        ----------
        parallel : bool, optional
            Use parallel processing (default: True)
        num_processes : int, optional
            Number of processes for parallel computation
            If None, uses all available CPUs
        """
        print(f"\n{'=' * 70}")
        print(f"Solving Eigenvalue Problems at {self.num_kpoints} K-Points")
        print(f"{'=' * 70}")
        
        if parallel and self.num_kpoints > 1:
            print(f"Mode: Parallel")
            results = solve_all_kpoints_parallel(
                self.kpoints,
                self.real_space_matrices,
                self.lattice_vectors,
                self.num_wann,
                num_processes
            )
        else:
            print(f"Mode: Sequential")
            results = solve_all_kpoints_sequential(
                self.kpoints,
                self.real_space_matrices,
                self.lattice_vectors,
                self.num_wann
            )
        
        # Unpack results
        self.eigenvalues_list, self.eigenvectors_list, self.H_k_list, self.S_k_list = results
        
        print(f"✓ Eigenvalue problems solved successfully")
        print(f"{'=' * 70}")
    
    def verify_results(self) -> dict:
        """
        Run all verification checks on computed results.
        
        Returns
        -------
        dict
            Dictionary containing verification results
        """
        if not self.eigenvalues_list:
            raise RuntimeError("No results to verify. Run solve_all_kpoints() first.")
        
        print(f"\n{'=' * 70}")
        print("Verification Checks")
        print(f"{'=' * 70}")
        
        results = run_all_verifications(
            self.eigenvalues_list,
            self.eigenvectors_list,
            self.H_k_list,
            self.S_k_list,
            self.num_wann,
            verbose=True
        )
        
        print(f"{'=' * 70}")
        
        return results
    
    def write_files(self, verbose: bool = True) -> None:
        """
        Write all Wannier90 input files (.eig, .amn, .mmn).
        
        Parameters
        ----------
        verbose : bool, optional
            Print progress messages (default: True)
        """
        if not self.eigenvalues_list:
            raise RuntimeError("No results to write. Run solve_all_kpoints() first.")
        
        print(f"\n{'=' * 70}")
        print("Writing Wannier90 Files")
        print(f"{'=' * 70}")
        
        write_wannier90_files(
            self.seedname,
            self.eigenvalues_list,
            self.eigenvectors_list,
            self.S_k_list,
            self.neighbor_list,
            self.num_kpoints,
            self.num_wann,
            verbose=verbose
        )
        
        print(f"{'=' * 70}")
    
    def run(
        self,
        parallel: bool = True,
        verify: bool = True,
        num_processes: Optional[int] = None
    ) -> dict:
        """
        Run the complete workflow.
        
        This method executes the entire calculation pipeline:
        1. Solve eigenvalue problems at all k-points
        2. Optionally verify results
        3. Write Wannier90 input files
        
        Parameters
        ----------
        parallel : bool, optional
            Use parallel processing (default: True)
        verify : bool, optional
            Run verification checks (default: True)
        num_processes : int, optional
            Number of processes for parallel computation
        
        Returns
        -------
        dict
            Verification results (if verify=True)
        
        Examples
        --------
        >>> engine = Wannier90Engine(...)
        >>> results = engine.run(parallel=True, verify=True)
        """
        print("\n" + "=" * 70)
        print("LCAO-to-Wannier90 Complete Workflow")
        print("=" * 70)
        
        # Print calculation info
        print_calculation_info(
            self.num_kpoints,
            self.k_grid,
            self.num_orbitals,
            self.num_wann
        )
        
        # Step 1: Solve eigenvalue problems
        self.solve_all_kpoints(parallel=parallel, num_processes=num_processes)
        
        # Step 2: Verification (optional)
        verification_results = None
        if verify:
            verification_results = self.verify_results()
        
        # Step 3: Write output files
        self.write_files(verbose=True)
        
        # Final summary
        print("\n" + "=" * 70)
        print("✓ Workflow Completed Successfully!")
        print("=" * 70)
        print(f"Output files created:")
        print(f"  • {self.seedname}.eig - Band energies")
        print(f"  • {self.seedname}.amn - Projection matrices")
        print(f"  • {self.seedname}.mmn - Overlap matrices")
        print("=" * 70)
        
        return verification_results
    
    def get_band_structure(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the computed band structure.
        
        Returns
        -------
        kpoints : ndarray
            K-points in fractional coordinates
        eigenvalues : ndarray of shape (num_kpoints, num_wann)
            Eigenvalues at each k-point
        """
        if not self.eigenvalues_list:
            raise RuntimeError("No band structure available. Run solve_all_kpoints() first.")
        
        eigenvalues_array = np.array([eigs.real for eigs in self.eigenvalues_list])
        return self.kpoints, eigenvalues_array
    
    def get_density_of_states(self, energy_range=None, num_bins=100):
        """
        Compute a simple density of states from the band structure.
        
        Parameters
        ----------
        energy_range : tuple of 2 floats, optional
            (E_min, E_max) for the DOS range
        num_bins : int, optional
            Number of energy bins
        
        Returns
        -------
        energies : ndarray
            Energy values
        dos : ndarray
            Density of states
        """
        if not self.eigenvalues_list:
            raise RuntimeError("No data available. Run solve_all_kpoints() first.")
        
        all_eigenvalues = np.concatenate([eigs.real for eigs in self.eigenvalues_list])
        
        if energy_range is None:
            energy_range = (np.min(all_eigenvalues), np.max(all_eigenvalues))
        
        dos, bin_edges = np.histogram(all_eigenvalues, bins=num_bins, range=energy_range)
        energies = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Normalize
        dos = dos.astype(float) / (self.num_kpoints * (energies[1] - energies[0]))
        
        return energies, dos
    
    def __repr__(self):
        """String representation of the engine."""
        return (
            f"Wannier90Engine(\n"
            f"  seedname='{self.seedname}',\n"
            f"  k_grid={self.k_grid},\n"
            f"  num_kpoints={self.num_kpoints},\n"
            f"  num_orbitals={self.num_orbitals},\n"
            f"  num_wann={self.num_wann}\n"
            f")"
        )
