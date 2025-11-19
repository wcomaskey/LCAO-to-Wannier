"""
Parser Module for LCAO Output Files

This module contains functions to parse overlap and Fock matrices from
CRYSTAL/LCAO output files and create spin-block matrices.

Based on original parsing code with enhancements for robustness.
"""

import numpy as np
import re
from typing import List, Dict, Tuple, Optional

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
# Matrix Utility Functions
# ==============================

def is_hermitian(matrix: np.ndarray, tol: float = 1e-34) -> bool:
    """
    Check if a matrix is Hermitian.
    
    Parameters
    ----------
    matrix : ndarray
        Complex matrix to check
    tol : float
        Tolerance for comparison
    
    Returns
    -------
    bool
        True if matrix is Hermitian within tolerance
    """
    return np.allclose(matrix, matrix.conj().T, atol=tol)


def make_hermitian(matrix: np.ndarray) -> np.ndarray:
    """
    Adjust a matrix to be Hermitian by averaging with its conjugate transpose.
    
    Parameters
    ----------
    matrix : ndarray
        Input matrix
    
    Returns
    -------
    ndarray
        Hermitianized matrix
    """
    if is_hermitian(matrix):
        return matrix
    
    # Use lower triangle and its conjugate transpose
    hermitian_matrix = np.tril(matrix)
    hermitian_matrix += np.tril(matrix, -1).conj().T
    return hermitian_matrix


# ==============================
# Matrix Parsing Functions
# ==============================

def parse_matrix_data(lines: List[str], start_index: int) -> Dict:
    """
    Parse matrix data from file lines starting at a given index.
    
    Parameters
    ----------
    lines : list of str
        Lines from the output file
    start_index : int
        Index to start parsing from
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'matrix': numpy array of the parsed matrix
        - 'next_index': index where parsing stopped
    """
    data_lines = []
    i = start_index
    col_indices = []
    
    while i < len(lines):
        line = lines[i].rstrip('\n')
        line_stripped = line.strip()
        
        # End parsing if we hit a new section header
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
    
    Parameters
    ----------
    lines : list of str
        Lines from the output file
    
    Returns
    -------
    matrices : list of dict
        List of matrix dictionaries containing:
        - 'type': 'overlap' or 'fock'
        - 'part': 'real', 'imag', or 'complex'
        - 'spin_channel': spin channel identifier (for Fock matrices)
        - 'lattice_vector': tuple of (R1, R2, R3)
        - 'data': numpy array of matrix data
    direct_lattice_vectors : list of lists or None
        3x3 list of lattice vectors in Angstroms
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
                else:
                    print(f"Warning: Line does not match vector format: '{vector_line}'")
                i += 1
            if len(vectors) == 3:
                direct_lattice_vectors = vectors
            else:
                print("Warning: Incomplete direct lattice vectors found.")
        
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
            part = 'real'
            i += 1
            S_parsed = parse_matrix_data(lines, i)
            matrices.append({
                'type': matrix_type,
                'part': part,
                'spin_channel': None,  # Overlap matrices are spin-independent
                'lattice_vector': lattice_vector,
                'data': S_parsed['matrix'],
            })
            i = S_parsed['next_index']
        
        # Parse Fock matrix
        elif fock_header_pattern.match(line):
            header_match = fock_header_pattern.match(line)
            part = header_match.group(1).lower()  # 'real' or 'imag'
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
        
        # Apply Hermitianization for diagonal spin blocks
        if spin_channel in ['ALPHA_ALPHA', 'BETA_BETA']:
            hermitian_matrix = make_hermitian(complex_matrix)
            matrices.append({
                'type': 'fock',
                'part': 'complex',
                'spin_channel': spin_channel,
                'lattice_vector': lattice_vector,
                'data': hermitian_matrix,
            })
        elif spin_channel in ['ALPHA_BETA', 'BETA_ALPHA']:
            matrices.append({
                'type': 'fock',
                'part': 'complex',
                'spin_channel': spin_channel,
                'lattice_vector': lattice_vector,
                'data': complex_matrix,
            })
        else:
            print(f"Warning: Unknown spin channel '{spin_channel}' encountered.")
    
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
    
    Parameters
    ----------
    H_R_dict : dict
        Maps (R1, R2, R3) to dict of spin channel matrices
    S_R_dict : dict
        Maps (R1, R2, R3) to overlap matrix
    N_basis : int
        Number of basis functions per spin channel
    direct_lattice_vectors : list of lists
        3x3 lattice vectors in Cartesian coordinates
    PRINTOUT : bool
        Whether to print warnings
    
    Returns
    -------
    H_full_list : list of tuples
        List of (R_cartesian, H_full_matrix) tuples
    S_full_list : list of tuples
        List of (R_cartesian, S_full_matrix) tuples
    """
    H_full_list = []
    S_full_list = []
    direct_lattice_vectors = np.array(direct_lattice_vectors)
    
    # Process Hamiltonian matrices
    for R in H_R_dict:
        R_integer = np.array(R)
        R_cartesian = np.matmul(R_integer, direct_lattice_vectors)
        H_full = np.zeros((2 * N_basis, 2 * N_basis), dtype=np.complex128)
        H_channels = H_R_dict.get(R, {})
        
        # Alpha-alpha block
        if 'ALPHA_ALPHA' in H_channels:
            H_aa = H_channels['ALPHA_ALPHA']
            if not is_hermitian(H_aa):
                if PRINTOUT:
                    print(f"Warning: H_{R}_ALPHA_ALPHA is not Hermitian. Adjusting it.")
                H_aa = make_hermitian(H_aa)
            H_full[0:N_basis, 0:N_basis] = H_aa
        else:
            raise ValueError(f"Error: H_{R}_ALPHA_ALPHA is missing.")
        
        # Beta-beta block
        if 'BETA_BETA' in H_channels:
            H_bb = H_channels['BETA_BETA']
            if not is_hermitian(H_bb):
                if PRINTOUT:
                    print(f"Warning: H_{R}_BETA_BETA is not Hermitian. Adjusting it.")
                H_bb = make_hermitian(H_bb)
            H_full[N_basis:2 * N_basis, N_basis:2 * N_basis] = H_bb
        else:
            raise ValueError(f"Error: H_{R}_BETA_BETA is missing.")
        
        # Off-diagonal spin blocks
        F_alpha_beta = H_channels.get('ALPHA_BETA')
        F_beta_alpha = H_channels.get('BETA_ALPHA')
        
        if F_alpha_beta is None or F_beta_alpha is None:
            raise ValueError(
                f"Error: For R = {R}, both F_alpha_beta and F_beta_alpha must be provided."
            )
        
        F_alpha_beta_dag = F_alpha_beta.conj().T
        F_alpha_beta_dag_no_diag = F_alpha_beta_dag - np.diag(np.diag(F_alpha_beta_dag))
        F_beta_alpha_adjusted = F_beta_alpha + F_alpha_beta_dag_no_diag
        
        H_full[0:N_basis, N_basis:2 * N_basis] = F_beta_alpha_adjusted.conj().T
        H_full[N_basis:2 * N_basis, 0:N_basis] = F_beta_alpha_adjusted
        
        if not is_hermitian(H_full):
            if PRINTOUT:
                print(f"Warning: Full Hamiltonian for R = {R} is not Hermitian. Adjusting it.")
            H_full = make_hermitian(H_full)
        
        H_full_list.append((R_cartesian, H_full))
    
    # Process overlap matrices (assumed spin-independent)
    for R in S_R_dict:
        R_integer = np.array(R)
        R_cartesian = np.matmul(R_integer, direct_lattice_vectors)
        S_full = np.zeros((2 * N_basis, 2 * N_basis), dtype=np.float64)
        S = S_R_dict[R]
        
        if not np.allclose(S, S.T, atol=1e-34):
            if PRINTOUT:
                print(f"Warning: Overlap matrix for R = {R} is not symmetric. Adjusting it.")
            S = (S + S.T) - np.diag(np.diag(S))
        
        S_full[0:N_basis, 0:N_basis] = S
        S_full[N_basis:2 * N_basis, N_basis:2 * N_basis] = S
        S_full_list.append((R_cartesian, S_full))
    
    return H_full_list, S_full_list
