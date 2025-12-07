"""
Parser Module for LCAO Output Files

This module contains functions to parse overlap and Fock matrices from
CRYSTAL/LCAO output files and create spin-block matrices using 
Robust Global Pair-Symmetry Construction.
"""

import numpy as np
import re
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

# ==============================
# Regular Expression Patterns
# ==============================

overlap_header_pattern = re.compile(
    r'^\s*OVERLAP MATRIX - CELL N\.\s+\d+\(\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*\)'
)
fock_header_pattern = re.compile(
    r'^\s*FOCK MATRIX \((REAL|IMAG) PART\) - CELL N\.\s+\d+\(\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*\)'
)
spin_channel_pattern = re.compile(
    r'^\s*(ALPHA_ALPHA|ALPHA_BETA|BETA_ALPHA|BETA_BETA) ELECTRONS', re.IGNORECASE
)
column_indices_pattern = re.compile(r'^\s*(\d+\s+)+\d+\s*$')
data_line_pattern = re.compile(r'^\s*(\d+)\s+(.+)$')
float_pattern = re.compile(
    r'[-+]?\d*\.\d+(?:[eEdD][-+]?\d+)?|[-+]?\d+(?:[eEdD][-+]?\d+)?'
)
direct_lattice_header_pattern = re.compile(
    r'^\s*DIRECT LATTICE VECTOR COMPONENTS \(ANGSTROM\)', re.IGNORECASE
)
vector_line_pattern = re.compile(
    r'^\s*' + r'\s+'.join([float_pattern.pattern] * 3) + r'\s*$'
)


# ==============================
# Calculation Parameters Dataclass
# ==============================

@dataclass
class CalculationParameters:
    """
    Parameters parsed from CRYSTAL/LCAO output file header.
    
    Attributes
    ----------
    fermi_energy : float or None
        Fermi energy in eV (converted from Hartree)
    fermi_energy_hartree : float or None
        Fermi energy in Hartree (raw value from file)
    num_electrons : int or None
        Number of electrons per cell
    k_grid : tuple of 3 ints or None
        Monkhorst-Pack k-point grid dimensions
    num_atoms : int or None
        Number of atoms per cell
    num_shells : int or None
        Number of shells
    num_ao : int or None
        Number of atomic orbitals
    total_energy : float or None
        Total energy in Hartree
    """
    fermi_energy: Optional[float] = None
    fermi_energy_hartree: Optional[float] = None
    num_electrons: Optional[int] = None
    k_grid: Optional[Tuple[int, int, int]] = None
    num_atoms: Optional[int] = None
    num_shells: Optional[int] = None
    num_ao: Optional[int] = None
    total_energy: Optional[float] = None


def parse_calculation_parameters(lines: List[str]) -> CalculationParameters:
    """
    Parse calculation parameters from CRYSTAL/LCAO output file.
    
    Extracts key parameters from the file header including Fermi energy,
    electron count, k-grid, and other useful information.
    
    Parameters
    ----------
    lines : list of str
        Lines from the CRYSTAL output file
        
    Returns
    -------
    CalculationParameters
        Dataclass containing parsed parameters
        
    Examples
    --------
    >>> with open('crystal.out', 'r') as f:
    ...     lines = f.readlines()
    >>> params = parse_calculation_parameters(lines)
    >>> print(f"Fermi energy: {params.fermi_energy} eV")
    >>> print(f"K-grid: {params.k_grid}")
    """
    params = CalculationParameters()
    
    # Conversion factor
    HARTREE_TO_EV = 27.2114
    
    for line in lines:
        # Parse FERMI ENERGY (in Hartree)
        # Format: FERMI ENERGY              -0.137E+00
        if 'FERMI ENERGY' in line and params.fermi_energy is None:
            match = re.search(r'FERMI ENERGY\s+([-+]?\d*\.?\d+[EeDd]?[+-]?\d*)', line)
            if match:
                fermi_str = match.group(1).replace('D', 'E').replace('d', 'e')
                params.fermi_energy_hartree = float(fermi_str)
                params.fermi_energy = params.fermi_energy_hartree * HARTREE_TO_EV
        
        # Parse N. OF ELECTRONS PER CELL
        # Format: N. OF ELECTRONS PER CELL    46
        if 'N. OF ELECTRONS PER CELL' in line and params.num_electrons is None:
            match = re.search(r'N\. OF ELECTRONS PER CELL\s+(\d+)', line)
            if match:
                params.num_electrons = int(match.group(1))
        
        # Parse SHRINK. FACT.(MONKH.) k-grid
        # Format: SHRINK. FACT.(MONKH.)    15 15  1  SHRINKING FACTOR(GILAT NET)       15
        if 'SHRINK. FACT.(MONKH.)' in line and params.k_grid is None:
            match = re.search(r'SHRINK\. FACT\.\(MONKH\.\)\s+(\d+)\s+(\d+)\s+(\d+)', line)
            if match:
                params.k_grid = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
        
        # Parse N. OF ATOMS PER CELL
        # Format: N. OF ATOMS PER CELL         2
        if 'N. OF ATOMS PER CELL' in line and params.num_atoms is None:
            match = re.search(r'N\. OF ATOMS PER CELL\s+(\d+)', line)
            if match:
                params.num_atoms = int(match.group(1))
        
        # Parse NUMBER OF SHELLS
        # Format: NUMBER OF SHELLS            20
        if 'NUMBER OF SHELLS' in line and params.num_shells is None:
            match = re.search(r'NUMBER OF SHELLS\s+(\d+)', line)
            if match:
                params.num_shells = int(match.group(1))
        
        # Parse NUMBER OF AO
        # Format: NUMBER OF AO                56
        if 'NUMBER OF AO' in line and params.num_ao is None:
            match = re.search(r'NUMBER OF AO\s+(\d+)', line)
            if match:
                params.num_ao = int(match.group(1))
        
        # Parse TOTAL ENERGY
        # Format: TOTAL ENERGY -4.2943592973046E+02
        if 'TOTAL ENERGY' in line and params.total_energy is None:
            match = re.search(r'TOTAL ENERGY\s+([-+]?\d*\.?\d+[EeDd]?[+-]?\d*)', line)
            if match:
                energy_str = match.group(1).replace('D', 'E').replace('d', 'e')
                params.total_energy = float(energy_str)
    
    return params


# ==============================
# Matrix Utility Functions
# ==============================

def is_hermitian(matrix: np.ndarray, tol: float = 1e-34) -> bool:
    """Check if a matrix is Hermitian."""
    return np.allclose(matrix, matrix.conj().T, atol=tol)


def fill_raw_matrix(
    H_R_dict: Dict, 
    S_R_dict: Dict, 
    R: Tuple[int, int, int], 
    N_basis: int, 
    is_fock: bool = True
) -> np.ndarray:
    """
    Constructs a 2N x 2N matrix containing ONLY the raw lower-triangular 
    data found in the file for a specific R vector.
    """
    if is_fock:
        M_raw = np.zeros((2 * N_basis, 2 * N_basis), dtype=np.complex128)
        data_source = H_R_dict
    else:
        M_raw = np.zeros((2 * N_basis, 2 * N_basis), dtype=np.float64)
        data_source = S_R_dict

    def insert_block(spin_key, row_offset, col_offset):
        if is_fock:
            if R in data_source and spin_key in data_source[R]:
                block = data_source[R][spin_key]
                # Enforce lower triangular reading
                M_raw[row_offset:row_offset+N_basis, col_offset:col_offset+N_basis] = np.tril(block)
        else:
            if R in data_source:
                block = data_source[R]
                L = np.tril(block)
                # Apply to Top-Left (AA)
                M_raw[0:N_basis, 0:N_basis] = L
                # Apply to Bottom-Right (BB)
                M_raw[N_basis:2*N_basis, N_basis:2*N_basis] = L
    
    if is_fock:
        insert_block('ALPHA_ALPHA', 0, 0)
        insert_block('BETA_BETA', N_basis, N_basis)
        insert_block('ALPHA_BETA', 0, N_basis) 
        insert_block('BETA_ALPHA', N_basis, 0) 
    else:
        # Overlap handling (assumed symmetric for AA and BB)
        insert_block(None, 0, 0)

    return M_raw


# ==============================
# Matrix Parsing Functions
# ==============================

def parse_matrix_data(lines: List[str], start_index: int) -> Dict:
    """Parse matrix data from file lines starting at a given index."""
    data_lines = []
    i = start_index
    col_indices = []
    
    while i < len(lines):
        line = lines[i].rstrip('\n')
        line_stripped = line.strip()
        
        if (overlap_header_pattern.match(line) or
            fock_header_pattern.match(line) or
            spin_channel_pattern.match(line) or
            direct_lattice_header_pattern.match(line)):
            break
        elif line_stripped == '':
            i += 1
            continue
        elif column_indices_pattern.match(line):
            col_indices = [int(num) - 1 for num in line_stripped.split()]
            i += 1
        else:
            data_match = data_line_pattern.match(line)
            if data_match:
                row_index = int(data_match.group(1)) - 1
                data_values_str = data_match.group(2)
                data_values = [
                    float(val.replace('D', 'E').replace('d', 'e'))
                    for val in float_pattern.findall(data_values_str)
                ]
                data_lines.append((row_index, col_indices.copy(), data_values))
            i += 1
    
    if data_lines:
        max_index = max(row for row, _, _ in data_lines)
        n = max_index + 1
        matrix = np.zeros((n, n), dtype=np.float64)
        for row, cols, values in data_lines:
            for col, val in zip(cols, values):
                matrix[row, col] = val
        return {'matrix': matrix, 'next_index': i}
    else:
        print(f"Warning: No matrix data found starting at line {start_index}")
        return {'matrix': None, 'next_index': i}


def parse_overlap_and_fock_matrices(lines: List[str]) -> Tuple[List[Dict], Optional[List[List[float]]]]:
    """
    Parse overlap and Fock matrices from CRYSTAL/LCAO output file lines.
    
    NOTE: This version returns RAW complex matrices. It does NOT enforce 
    Hermiticity at this stage, as that is handled during global construction.
    """
    matrices = []
    direct_lattice_vectors = None
    current_spin_channel = None
    i = 0
    fock_temp_storage = {}
    
    while i < len(lines):
        line = lines[i].rstrip('\n')
        
        # Parse direct lattice vectors
        if direct_lattice_header_pattern.match(line):
            vectors = []
            i += 1
            for _ in range(3):
                if i >= len(lines):
                    print("Warning: Unexpected end of file while reading direct lattice vectors.")
                    break
                vector_line = lines[i].strip()
                vector_match = vector_line_pattern.match(vector_line)
                if vector_match:
                    components = [
                        float(val.replace('D', 'E').replace('d', 'e'))
                        for val in float_pattern.findall(vector_line)
                    ]
                    vectors.append(components)
                i += 1
            if len(vectors) == 3:
                direct_lattice_vectors = vectors
        
        # Parse spin channel header
        elif spin_channel_pattern.match(line):
            spin_match = spin_channel_pattern.match(line)
            current_spin_channel = spin_match.group(1).upper()
            i += 1
        
        # Parse overlap matrix
        elif overlap_header_pattern.match(line):
            header_match = overlap_header_pattern.match(line)
            lattice_vector = [int(header_match.group(j)) for j in range(1, 4)]
            matrix_type = 'overlap'
            i += 1
            S_parsed = parse_matrix_data(lines, i)
            matrices.append({
                'type': matrix_type,
                'part': 'real',
                'spin_channel': None,
                'lattice_vector': lattice_vector,
                'data': S_parsed['matrix'],
            })
            i = S_parsed['next_index']
        
        # Parse Fock matrix
        elif fock_header_pattern.match(line):
            header_match = fock_header_pattern.match(line)
            part = header_match.group(1).lower()
            lattice_vector = tuple(int(header_match.group(j)) for j in range(2, 5))
            matrix_type = 'fock'
            i += 1
            F_parsed = parse_matrix_data(lines, i)
            key = (current_spin_channel, lattice_vector)
            if key not in fock_temp_storage:
                fock_temp_storage[key] = {}
            fock_temp_storage[key][part] = F_parsed['matrix']
            i = F_parsed['next_index']
        else:
            i += 1
    
    # Combine real and imaginary parts for Fock matrices
    for key, parts in fock_temp_storage.items():
        spin_channel, lattice_vector = key
        real_part = parts.get('real')
        imag_part = parts.get('imag')
        
        if real_part is not None and imag_part is not None:
            complex_matrix = real_part + 1j * imag_part
        elif real_part is not None:
            complex_matrix = real_part.astype(np.complex128)
        elif imag_part is not None:
            complex_matrix = 1j * imag_part
        else:
            continue
        
        # We append the complex matrix AS IS.
        # Hermiticity is enforced later in create_spin_block_matrices.
        matrices.append({
            'type': 'fock',
            'part': 'complex',
            'spin_channel': spin_channel,
            'lattice_vector': lattice_vector,
            'data': complex_matrix,
        })
    
    return matrices, direct_lattice_vectors


# ==============================
# Spin Block Matrix Creation
# ==============================

def create_spin_block_matrices(
    H_R_dict: Dict[Tuple[int, int, int], Dict[str, np.ndarray]],
    S_R_dict: Dict[Tuple[int, int, int], np.ndarray],
    N_basis: int,
    direct_lattice_vectors: List[List[float]],
    PRINTOUT: bool = False
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Construct full 2N x 2N spin block Hamiltonian and overlap matrices.
    
    This function uses Global Pair Symmetry Construction:
    1. It identifies pairs of vectors (R, -R).
    2. It combines the raw lower-triangular data of R with the raw lower-triangular
       data of -R to strictly enforce the relationship H(R) = H(-R)†.
    3. It handles the origin H(0) by ensuring exact Hermiticity and real diagonal.
    
    Returns
    -------
    H_full_list : list of tuples (R_cartesian, H_matrix)
    S_full_list : list of tuples (R_cartesian, S_matrix)
    """
    H_full_list = []
    S_full_list = []
    direct_lattice_vectors = np.array(direct_lattice_vectors)
    
    all_R_vectors = set(H_R_dict.keys())
    
    # Identify valid pairs (ensure we only process (R, -R) once)
    processed_R = set()
    valid_pairs = [] 
    
    # Sort for consistent output/processing order
    sorted_R = sorted(list(all_R_vectors))
    
    for R in sorted_R:
        if R in processed_R: continue
        
        minus_R = tuple(-x for x in R)
        
        if R == (0, 0, 0):
            valid_pairs.append((R, R))
            processed_R.add(R)
        elif minus_R in all_R_vectors:
            valid_pairs.append((R, minus_R))
            processed_R.add(R)
            processed_R.add(minus_R)
        else:
            if PRINTOUT:
                print(f"Warning: Vector {R} exists but {minus_R} is missing. Excluding.")

    # Iterate over pairs and construct matrices
    for R, minus_R in valid_pairs:
        is_origin = (R == minus_R)
        
        # Calculate Cartesian vectors
        R_cart = np.dot(np.array(R), direct_lattice_vectors)
        minus_R_cart = np.dot(np.array(minus_R), direct_lattice_vectors)
        
        # ============================================================
        # 1. FOCK CONSTRUCTION
        # ============================================================
        
        # Build Raw Lower Data (Global 2N x 2N container)
        M_raw_R = fill_raw_matrix(H_R_dict, S_R_dict, R, N_basis, is_fock=True)
        M_raw_minus_R = fill_raw_matrix(H_R_dict, S_R_dict, minus_R, N_basis, is_fock=True)
        
        # Prepare the Upper part from the -R partner
        # We take the STRICT lower triangle of -R (remove diagonal)
        # Its Conjugate Transpose becomes the Strict Upper Triangle of R
        M_minus_R_nodiag = M_raw_minus_R.copy()
        rows, cols = np.diag_indices_from(M_minus_R_nodiag)
        M_minus_R_nodiag[rows, cols] = 0 
        
        # Combine: H(R) = Lower(R) + [Lower_Strict(-R)]†
        H_R_full = M_raw_R + M_minus_R_nodiag.conj().T
        
        # Enforce Origin Hermiticity (Physical Requirement)
        if is_origin:
            diags = np.diag(H_R_full)
            np.fill_diagonal(H_R_full, np.real(diags))
        
        # Add H(R) to list
        H_full_list.append((R_cart, H_R_full))
        
        # Derive and Add H(-R)
        if not is_origin:
            # Enforce exact symmetry: H(-R) = H(R)†
            H_minus_R_full = H_R_full.conj().T
            H_full_list.append((minus_R_cart, H_minus_R_full))
        
        # ============================================================
        # 2. OVERLAP CONSTRUCTION
        # ============================================================
        if R in S_R_dict and (is_origin or minus_R in S_R_dict):
            # Note: For origin, S_raw_R and S_raw_minus_R are the same
            S_raw_R = fill_raw_matrix(H_R_dict, S_R_dict, R, N_basis, is_fock=False)
            S_raw_minus_R = fill_raw_matrix(H_R_dict, S_R_dict, minus_R, N_basis, is_fock=False)
            
            # Prepare Upper part from -R partner
            S_minus_R_nodiag = S_raw_minus_R.copy()
            rows, cols = np.diag_indices_from(S_minus_R_nodiag)
            S_minus_R_nodiag[rows, cols] = 0
            
            # Combine: S(R) = Lower(R) + [Lower_Strict(-R)]^T 
            # (Transpose only, because Overlap is Real)
            S_R_full = S_raw_R + S_minus_R_nodiag.T
            
            S_full_list.append((R_cart, S_R_full))

            if not is_origin:
                # Enforce exact symmetry: S(-R) = S(R)^T
                S_minus_R_full = S_R_full.T
                S_full_list.append((minus_R_cart, S_minus_R_full))

    return H_full_list, S_full_list