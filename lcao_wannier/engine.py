"""
Main Engine Module

This module contains the Wannier90Engine class that coordinates all
operations for converting LCAO calculations to Wannier90 format.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
import warnings

from .kpoints import generate_kpoint_grid, generate_neighbor_list
from .solver import solve_all_kpoints_sequential, solve_all_kpoints_parallel
from .wannier90 import write_wannier90_files
from .verification import run_all_verifications
from .utils import check_matrix_consistency, print_calculation_info
from .band_selection import (
    estimate_fermi_energy,
    analyze_band_window,
    print_band_analysis,
    check_frozen_continuity,
    validate_fermi_coverage,
    select_projection_orbitals,
    compute_subspace_projections,
    suggest_optimal_window,
    BandWindowResult,
    OrbitalSelectionResult
)


class Wannier90Engine:
    """
    Main computational engine for LCAO-to-Wannier90 conversion.
    
    This class coordinates the entire workflow:
    1. K-point grid generation
    2. Fourier transforms and eigenvalue solving
    3. Band window analysis and selection
    4. Automatic projection orbital selection
    5. Verification of results
    6. Writing Wannier90 input files
    
    Attributes
    ----------
    real_space_matrices : dict
        Real-space H(R) and S(R) matrices
    k_grid : tuple of 3 ints
        Dimensions of k-point grid
    lattice_vectors : ndarray
        Real-space lattice vectors
    num_wann : int
        Number of Wannier functions (may be auto-determined from window)
    seedname : str
        Prefix for output files
    outer_window : tuple of 2 floats, optional
        Energy window (E_min, E_max) for band selection
    e_fermi : float, optional
        Fermi energy (auto-detected if None)
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
    band_analysis : BandWindowResult
        Results from band window analysis
    selected_band_indices : ndarray
        Indices of bands to include in output
    selected_orbital_indices : ndarray
        Indices of orbitals used for projections
    
    Examples
    --------
    Basic usage with all bands:
    
    >>> engine = Wannier90Engine(
    ...     real_space_matrices=matrices,
    ...     k_grid=(4, 4, 4),
    ...     lattice_vectors=lattice_vecs,
    ...     seedname='material'
    ... )
    >>> engine.run(parallel=True, verify=True)
    
    With automatic band selection using energy window:
    
    >>> engine = Wannier90Engine(
    ...     real_space_matrices=matrices,
    ...     k_grid=(15, 15, 1),
    ...     lattice_vectors=lattice_vecs,
    ...     seedname='bismuth',
    ...     outer_window=(-3.0, 5.0),  # eV relative to E_F
    ...     num_electrons=10
    ... )
    >>> engine.run(analyze_window=True)
    """
    
    def __init__(
        self,
        real_space_matrices: Dict[Tuple[int, int, int], Dict[str, np.ndarray]],
        k_grid: Tuple[int, int, int],
        lattice_vectors: np.ndarray,
        num_wann: Optional[int] = None,
        seedname: str = "wannier90",
        outer_window: Optional[Tuple[float, float]] = None,
        e_fermi: Optional[float] = None,
        num_electrons: Optional[int] = None,
        window_is_relative: bool = True
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
        num_wann : int, optional
            Number of Wannier functions. If None and outer_window is provided,
            will be determined automatically from the frozen bands.
            If None and no outer_window, uses all orbitals.
        seedname : str, optional
            Prefix for output files (default: 'wannier90')
        outer_window : tuple of 2 floats, optional
            Energy window (E_min, E_max) for band selection. Bands whose
            entire dispersion falls within this window become "frozen" bands.
            If None, all bands are used.
        e_fermi : float, optional
            Fermi energy in eV. If None and outer_window is provided,
            will be auto-detected from the band structure.
        num_electrons : int, optional
            Number of electrons in the system. Used for Fermi level
            auto-detection when e_fermi is None.
        window_is_relative : bool, optional
            If True (default), outer_window values are relative to E_F.
            If False, they are absolute energies.
        """
        self.real_space_matrices = real_space_matrices
        self.k_grid = k_grid
        self.lattice_vectors = lattice_vectors
        self.seedname = seedname
        
        # Band selection parameters
        self.outer_window = outer_window
        self.e_fermi = e_fermi
        self.num_electrons = num_electrons
        self.window_is_relative = window_is_relative
        self._user_num_wann = num_wann  # Store user-specified value
        
        # Check matrix consistency
        if not check_matrix_consistency(real_space_matrices):
            raise ValueError("Inconsistent matrix dimensions in real_space_matrices")
        
        # Get number of orbitals
        first_key = next(iter(real_space_matrices))
        self.num_orbitals = real_space_matrices[first_key]['H'].shape[0]
        
        # Set initial num_wann (may be updated after band analysis)
        if num_wann is not None:
            if num_wann > self.num_orbitals:
                raise ValueError(
                    f"num_wann ({num_wann}) cannot exceed num_orbitals ({self.num_orbitals})"
                )
            self.num_wann = num_wann
        else:
            # Will be determined later (from window analysis or full basis)
            self.num_wann = self.num_orbitals
        
        # Generate k-point grid and neighbor list
        self.kpoints = generate_kpoint_grid(k_grid)
        self.num_kpoints = len(self.kpoints)
        self.neighbor_list = generate_neighbor_list(k_grid)
        
        # Storage for results (initialized as empty)
        self.eigenvalues_list = []
        self.eigenvectors_list = []
        self.H_k_list = []
        self.S_k_list = []
        
        # Band selection results (initialized as None)
        self.band_analysis = None
        self.selected_band_indices = None
        self.selected_orbital_indices = None
        self.orbital_selection_result = None
        
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
        if self._user_num_wann is not None:
            print(f"Number of Wannier functions: {self._user_num_wann} (user-specified)")
        elif self.outer_window is not None:
            print(f"Number of Wannier functions: (to be determined from window)")
        else:
            print(f"Number of Wannier functions: {self.num_orbitals} (full basis)")
        print(f"Number of R-vectors: {len(self.real_space_matrices)}")
        
        if self.outer_window is not None:
            window_type = "relative to E_F" if self.window_is_relative else "absolute"
            print(f"Energy window: [{self.outer_window[0]:.2f}, {self.outer_window[1]:.2f}] eV ({window_type})")
            if self.e_fermi is not None:
                print(f"Fermi energy: {self.e_fermi:.4f} eV (user-specified)")
            else:
                print(f"Fermi energy: (to be auto-detected)")
        
        print("=" * 70)
    
    def solve_all_kpoints(
        self,
        parallel: bool = True,
        num_processes: Optional[int] = None,
        convert_to_eV: bool = True
    ) -> None:
        """
        Solve the generalized eigenvalue problem for all k-points.
        
        This solves for ALL bands initially. Band selection is applied
        later in analyze_bands() or write_files().
        
        Parameters
        ----------
        parallel : bool, optional
            Use parallel processing (default: True)
        num_processes : int, optional
            Number of processes for parallel computation
            If None, uses all available CPUs
        convert_to_eV : bool, optional
            Convert eigenvalues from Hartree to eV (default: True)
            Set to False if your Fock matrix is already in eV
        """
        print(f"\n{'=' * 70}")
        print(f"Solving Eigenvalue Problems at {self.num_kpoints} K-Points")
        print(f"{'=' * 70}")
        
        # Always solve for all orbitals first
        solve_num_bands = self.num_orbitals
        
        if parallel and self.num_kpoints > 1:
            print(f"Mode: Parallel")
            results = solve_all_kpoints_parallel(
                self.kpoints,
                self.real_space_matrices,
                self.lattice_vectors,
                solve_num_bands,
                num_processes
            )
        else:
            print(f"Mode: Sequential")
            results = solve_all_kpoints_sequential(
                self.kpoints,
                self.real_space_matrices,
                self.lattice_vectors,
                solve_num_bands
            )
        
        # Unpack results
        self.eigenvalues_list, self.eigenvectors_list, self.H_k_list, self.S_k_list = results
        
        # Convert eigenvalues from Hartree to eV
        if convert_to_eV:
            HARTREE_TO_EV = 27.2114
            self.eigenvalues_list = [eigs * HARTREE_TO_EV for eigs in self.eigenvalues_list]
            print(f"✓ Eigenvalues converted from Hartree to eV")
        
        print(f"✓ Eigenvalue problems solved successfully")
        print(f"{'=' * 70}")
    
    def analyze_bands(self, verbose: bool = True) -> Optional[BandWindowResult]:
        """
        Analyze band structure and determine frozen bands from energy window.
        
        This method identifies which bands fall entirely within the specified
        outer_window, making them suitable as frozen bands for Wannier90.
        
        Must be called after solve_all_kpoints().
        
        Parameters
        ----------
        verbose : bool, optional
            Print detailed analysis (default: True)
            
        Returns
        -------
        BandWindowResult or None
            Analysis results, or None if no window was specified
            
        Raises
        ------
        RuntimeError
            If called before solve_all_kpoints()
        """
        if not self.eigenvalues_list:
            raise RuntimeError("Must call solve_all_kpoints() first")
        
        # If no window specified, use all bands
        if self.outer_window is None:
            self.selected_band_indices = np.arange(self.num_orbitals)
            self.num_wann = self._user_num_wann or self.num_orbitals
            if verbose:
                print(f"\nNo energy window specified - using all {self.num_wann} bands")
            return None
        
        print(f"\n{'=' * 70}")
        print("Band Window Analysis")
        print(f"{'=' * 70}")
        
        # Auto-detect Fermi level if needed
        if self.e_fermi is None:
            self.e_fermi = estimate_fermi_energy(
                self.eigenvalues_list,
                num_electrons=self.num_electrons,
                method='auto'
            )
            if verbose:
                print(f"Auto-detected Fermi energy: {self.e_fermi:.4f} eV")
        
        # Analyze window
        self.band_analysis = analyze_band_window(
            self.eigenvalues_list,
            outer_window=self.outer_window,
            e_fermi=self.e_fermi,
            window_is_relative=self.window_is_relative,
            num_electrons=self.num_electrons
        )
        
        if verbose:
            print_band_analysis(self.band_analysis)
        
        # Check for issues
        if self.band_analysis.num_wann == 0:
            warnings.warn(
                "No bands are fully contained within the energy window! "
                "Consider widening the window."
            )
            return self.band_analysis
        
        is_contiguous, gaps = check_frozen_continuity(
            self.band_analysis.frozen_indices
        )
        
        coverage = validate_fermi_coverage(
            self.band_analysis.frozen_indices,
            self.eigenvalues_list,
            self.band_analysis.e_fermi
        )
        
        if verbose:
            if not is_contiguous:
                print("\n⚠ Warning: Frozen bands are non-contiguous!")
            if not coverage['has_occupied']:
                print("⚠ Warning: No occupied bands in frozen window!")
            if not coverage['has_unoccupied']:
                print("⚠ Warning: No unoccupied bands in frozen window!")
            if coverage['crosses_fermi']:
                print("ℹ Note: Some bands cross the Fermi level (metallic)")
        
        # Set selected bands
        self.selected_band_indices = self.band_analysis.frozen_indices
        
        # Update num_wann (user-specified takes precedence)
        if self._user_num_wann is not None:
            if self._user_num_wann > self.band_analysis.num_wann:
                warnings.warn(
                    f"User-specified num_wann ({self._user_num_wann}) exceeds frozen bands "
                    f"({self.band_analysis.num_wann}). Using frozen band count."
                )
                self.num_wann = self.band_analysis.num_wann
            else:
                self.num_wann = self._user_num_wann
        else:
            self.num_wann = self.band_analysis.num_wann
        
        print(f"{'=' * 70}")
        
        return self.band_analysis
    
    def select_projections(self, verbose: bool = True) -> OrbitalSelectionResult:
        """
        Automatically select optimal projection orbitals.
        
        For LCAO bases, this selects the orbitals with the largest
        contribution to the selected bands, providing optimal initial
        projections for Wannier90 without user specification.
        
        Must be called after analyze_bands() or after setting
        selected_band_indices manually.
        
        Parameters
        ----------
        verbose : bool, optional
            Print selection details (default: True)
            
        Returns
        -------
        OrbitalSelectionResult
            Contains selected orbital indices and weights
            
        Raises
        ------
        RuntimeError
            If called before bands are selected
        """
        if self.selected_band_indices is None:
            raise RuntimeError("Must call analyze_bands() first or set selected_band_indices")
        
        if not self.eigenvectors_list:
            raise RuntimeError("Must call solve_all_kpoints() first")
        
        print(f"\n{'=' * 70}")
        print("Automatic Projection Orbital Selection")
        print(f"{'=' * 70}")
        
        self.orbital_selection_result = select_projection_orbitals(
            self.eigenvectors_list,
            self.S_k_list,
            num_wann=self.num_wann,
            band_indices=self.selected_band_indices
        )
        
        self.selected_orbital_indices = self.orbital_selection_result.selected_indices
        
        if verbose:
            print(f"Selected {self.orbital_selection_result.num_selected} projection orbitals")
            print(f"Orbital indices: {self.selected_orbital_indices}")
            
            # Show top contributors
            weights = self.orbital_selection_result.orbital_weights
            top_indices = np.argsort(weights)[-5:][::-1]
            print("\nTop 5 contributing orbitals:")
            for idx in top_indices:
                print(f"  Orbital {idx}: weight = {weights[idx]:.4f}")
        
        print(f"{'=' * 70}")
        
        return self.orbital_selection_result
    
    def suggest_window(
        self,
        target_num_wann: Optional[int] = None,
        padding: float = 0.1
    ) -> Tuple[float, float]:
        """
        Suggest an optimal energy window based on band structure.
        
        Must be called after solve_all_kpoints().
        
        Parameters
        ----------
        target_num_wann : int, optional
            Target number of Wannier functions. If provided, attempts
            to find a window containing approximately this many bands.
        padding : float, optional
            Extra energy padding (eV) around the tight band edges
            
        Returns
        -------
        tuple of 2 floats
            Suggested (E_min, E_max) window in absolute energies
        """
        if not self.eigenvalues_list:
            raise RuntimeError("Must call solve_all_kpoints() first")
        
        if self.e_fermi is None:
            self.e_fermi = estimate_fermi_energy(
                self.eigenvalues_list,
                num_electrons=self.num_electrons,
                method='auto'
            )
        
        suggested = suggest_optimal_window(
            self.eigenvalues_list,
            e_fermi=self.e_fermi,
            target_num_wann=target_num_wann,
            padding=padding
        )
        
        print(f"\nSuggested energy window: [{suggested[0]:.2f}, {suggested[1]:.2f}] eV (absolute)")
        print(f"                         [{suggested[0] - self.e_fermi:.2f}, "
              f"{suggested[1] - self.e_fermi:.2f}] eV (relative to E_F)")
        
        return suggested
    
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
        
        Uses only the selected bands if band analysis has been performed.
        
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
        
        # Determine which bands to use
        if self.selected_band_indices is not None:
            band_indices = self.selected_band_indices
            if verbose:
                print(f"Using {len(band_indices)} selected bands: "
                      f"{band_indices[0]}-{band_indices[-1]}")
        else:
            band_indices = np.arange(self.num_wann)
            if verbose:
                print(f"Using first {self.num_wann} bands")
        
        # Extract selected bands
        eigenvalues_selected = [
            eig[band_indices] for eig in self.eigenvalues_list
        ]
        eigenvectors_selected = [
            C[:, band_indices] for C in self.eigenvectors_list
        ]
        
        write_wannier90_files(
            self.seedname,
            eigenvalues_selected,
            eigenvectors_selected,
            self.S_k_list,
            self.neighbor_list,
            self.num_kpoints,
            len(band_indices),  # Use actual number of selected bands
            verbose=verbose
        )
        
        print(f"{'=' * 70}")
    
    def run(
        self,
        parallel: bool = True,
        verify: bool = True,
        analyze_window: bool = True,
        select_orbitals: bool = True,
        num_processes: Optional[int] = None
    ) -> dict:
        """
        Run the complete workflow.
        
        This method executes the entire calculation pipeline:
        1. Solve eigenvalue problems at all k-points
        2. Analyze band window and select frozen bands (if outer_window specified)
        3. Automatically select projection orbitals (optional)
        4. Verify results (optional)
        5. Write Wannier90 input files
        
        Parameters
        ----------
        parallel : bool, optional
            Use parallel processing (default: True)
        verify : bool, optional
            Run verification checks (default: True)
        analyze_window : bool, optional
            Perform band window analysis if outer_window is set (default: True)
        select_orbitals : bool, optional
            Automatically select projection orbitals (default: True)
        num_processes : int, optional
            Number of processes for parallel computation
        
        Returns
        -------
        dict
            Results dictionary containing:
            - 'verification': Verification results (if verify=True)
            - 'band_analysis': Band window analysis (if performed)
            - 'orbital_selection': Orbital selection results (if performed)
            - 'eigenvalues': All computed eigenvalues
            - 'selected_bands': Indices of selected bands
            - 'num_wann': Final number of Wannier functions
        
        Examples
        --------
        Basic usage:
        
        >>> engine = Wannier90Engine(...)
        >>> results = engine.run()
        
        With energy window:
        
        >>> engine = Wannier90Engine(..., outer_window=(-3.0, 5.0))
        >>> results = engine.run(analyze_window=True)
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
        
        # Step 1: Solve eigenvalue problems (always for all bands)
        self.solve_all_kpoints(parallel=parallel, num_processes=num_processes)
        
        # Step 2: Band window analysis (if requested and window specified)
        if analyze_window and self.outer_window is not None:
            self.analyze_bands(verbose=True)
            
            # Step 3: Automatic orbital selection (if requested)
            if select_orbitals and self.band_analysis is not None:
                if self.band_analysis.num_wann > 0:
                    self.select_projections(verbose=True)
        else:
            # No window analysis - use all bands or user-specified num_wann
            self.selected_band_indices = np.arange(self.num_wann)
        
        # Step 4: Verification (optional)
        verification_results = None
        if verify:
            verification_results = self.verify_results()
        
        # Step 5: Write output files
        self.write_files(verbose=True)
        
        # Final summary
        print("\n" + "=" * 70)
        print("✓ Workflow Completed Successfully!")
        print("=" * 70)
        print(f"Output files created:")
        print(f"  • {self.seedname}.eig - Band energies ({self.num_wann} bands)")
        print(f"  • {self.seedname}.amn - Projection matrices")
        print(f"  • {self.seedname}.mmn - Overlap matrices")
        if self.band_analysis is not None:
            print(f"\nBand selection:")
            print(f"  • Frozen bands: {self.band_analysis.num_wann}")
            print(f"  • Energy range: [{self.band_analysis.frozen_energy_range[0]:.2f}, "
                  f"{self.band_analysis.frozen_energy_range[1]:.2f}] eV")
        print("=" * 70)
        
        return {
            'verification': verification_results,
            'band_analysis': self.band_analysis,
            'orbital_selection': self.orbital_selection_result,
            'eigenvalues': self.eigenvalues_list,
            'selected_bands': self.selected_band_indices,
            'num_wann': self.num_wann
        }
    
    def get_band_structure(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the computed band structure.
        
        Returns
        -------
        kpoints : ndarray
            K-points in fractional coordinates
        eigenvalues : ndarray of shape (num_kpoints, num_bands)
            Eigenvalues at each k-point (all bands, not just selected)
        """
        if not self.eigenvalues_list:
            raise RuntimeError("No band structure available. Run solve_all_kpoints() first.")
        
        eigenvalues_array = np.array([eigs.real for eigs in self.eigenvalues_list])
        return self.kpoints, eigenvalues_array
    
    def get_selected_band_structure(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the band structure for selected bands only.
        
        Returns
        -------
        kpoints : ndarray
            K-points in fractional coordinates
        eigenvalues : ndarray of shape (num_kpoints, num_selected_bands)
            Eigenvalues at each k-point for selected bands
        """
        if not self.eigenvalues_list:
            raise RuntimeError("No band structure available. Run solve_all_kpoints() first.")
        
        if self.selected_band_indices is None:
            return self.get_band_structure()
        
        eigenvalues_array = np.array([
            eigs[self.selected_band_indices].real 
            for eigs in self.eigenvalues_list
        ])
        return self.kpoints, eigenvalues_array
    
    def get_density_of_states(
        self,
        energy_range: Optional[Tuple[float, float]] = None,
        num_bins: int = 100,
        selected_only: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute a simple density of states from the band structure.
        
        Parameters
        ----------
        energy_range : tuple of 2 floats, optional
            (E_min, E_max) for the DOS range
        num_bins : int, optional
            Number of energy bins (default: 100)
        selected_only : bool, optional
            If True, compute DOS only for selected bands (default: False)
        
        Returns
        -------
        energies : ndarray
            Energy values at bin centers
        dos : ndarray
            Density of states
        """
        if not self.eigenvalues_list:
            raise RuntimeError("No data available. Run solve_all_kpoints() first.")
        
        if selected_only and self.selected_band_indices is not None:
            all_eigenvalues = np.concatenate([
                eigs[self.selected_band_indices].real 
                for eigs in self.eigenvalues_list
            ])
        else:
            all_eigenvalues = np.concatenate([
                eigs.real for eigs in self.eigenvalues_list
            ])
        
        if energy_range is None:
            energy_range = (np.min(all_eigenvalues), np.max(all_eigenvalues))
        
        dos, bin_edges = np.histogram(all_eigenvalues, bins=num_bins, range=energy_range)
        energies = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Normalize
        dos = dos.astype(float) / (self.num_kpoints * (energies[1] - energies[0]))
        
        return energies, dos
    
    def get_energy_statistics(self) -> Dict[str, float]:
        """
        Get energy statistics for the computed band structure.
        
        Returns
        -------
        dict
            Dictionary with keys:
            - 'e_min': Minimum eigenvalue
            - 'e_max': Maximum eigenvalue
            - 'e_fermi': Fermi energy (if determined)
            - 'bandwidth': Total bandwidth
            - 'gap': Band gap (if semiconductor, else None)
        """
        if not self.eigenvalues_list:
            raise RuntimeError("No data available. Run solve_all_kpoints() first.")
        
        all_eigenvalues = np.array(self.eigenvalues_list)
        
        e_min = np.min(all_eigenvalues)
        e_max = np.max(all_eigenvalues)
        
        stats = {
            'e_min': float(e_min),
            'e_max': float(e_max),
            'e_fermi': float(self.e_fermi) if self.e_fermi is not None else None,
            'bandwidth': float(e_max - e_min),
            'gap': None
        }
        
        # Try to detect a gap
        num_bands = all_eigenvalues.shape[1]
        avg_energies = np.mean(all_eigenvalues, axis=0)
        gaps = np.diff(avg_energies)
        
        # Find the largest gap
        if len(gaps) > 0:
            max_gap_idx = np.argmax(gaps)
            max_gap = gaps[max_gap_idx]
            if max_gap > 0.1:  # Threshold for considering it a gap
                stats['gap'] = float(max_gap)
        
        return stats
    
    def __repr__(self):
        """String representation of the engine."""
        lines = [
            "Wannier90Engine(",
            f"  seedname='{self.seedname}',",
            f"  k_grid={self.k_grid},",
            f"  num_kpoints={self.num_kpoints},",
            f"  num_orbitals={self.num_orbitals},",
            f"  num_wann={self.num_wann},"
        ]
        
        if self.outer_window is not None:
            lines.append(f"  outer_window={self.outer_window},")
        if self.e_fermi is not None:
            lines.append(f"  e_fermi={self.e_fermi:.4f},")
        if self.band_analysis is not None:
            lines.append(f"  frozen_bands={self.band_analysis.num_wann},")
        
        lines.append(")")
        
        return "\n".join(lines)