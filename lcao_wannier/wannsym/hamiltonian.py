"""
Wannier90 Hamiltonian (wannier90_hr.dat) I/O Module

Reads and writes the real-space tight-binding Hamiltonian produced by
Wannier90. Uses dict-based storage instead of dense 5D arrays, removing
the hardcoded sdim limit and reducing memory usage significantly.

File format reference:
    Line 1:    Header comment
    Line 2:    num_wann (number of Wannier orbitals)
    Line 3:    nrpt (number of R-vectors)
    Lines 4+:  Degeneracy values (15 per line)
    Remaining: rx ry rz i j Re(H) Im(H)  [nrpt * norbs * norbs lines]
               i,j are 1-indexed orbital indices
"""

import numpy as np
import datetime
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List


@dataclass
class HamiltonianData:
    """
    Real-space tight-binding Hamiltonian from wannier90_hr.dat.

    Attributes
    ----------
    norbs : int
        Number of Wannier orbitals (num_wann)
    hr : dict
        {(rx, ry, rz): np.ndarray} mapping R-vector tuples to (norbs, norbs)
        complex Hamiltonian matrices
    deg_rpt : dict
        {(rx, ry, rz): int} mapping R-vector tuples to degeneracy values
    """
    norbs: int
    hr: Dict[Tuple[int, int, int], np.ndarray] = field(default_factory=dict)
    deg_rpt: Dict[Tuple[int, int, int], int] = field(default_factory=dict)

    @property
    def nrpt(self) -> int:
        """Number of R-vectors."""
        return len(self.hr)

    @property
    def rpts(self) -> List[Tuple[int, int, int]]:
        """Sorted list of R-vectors."""
        return sorted(self.hr.keys())

    @staticmethod
    def from_file(filename: str) -> 'HamiltonianData':
        """
        Read a wannier90_hr.dat file.

        Parameters
        ----------
        filename : str
            Path to the wannier90_hr.dat file

        Returns
        -------
        HamiltonianData
            The parsed Hamiltonian data
        """
        hr = {}
        deg_rpt = {}

        with open(filename, 'r') as f:
            # Line 1: header comment
            f.readline()

            # Line 2: number of Wannier orbitals
            norbs = int(f.readline().strip())

            # Line 3: number of R-points
            nrpt = int(f.readline().strip())

            # Degeneracy values: 15 per line
            nlines_deg = int(np.ceil(nrpt / 15.0))
            deg_values = []
            for _ in range(nlines_deg):
                deg_values.extend(int(x) for x in f.readline().strip().split())

            # Read Hamiltonian matrix elements
            # Format: rx ry rz i j Re(H) Im(H)
            # i, j are 1-indexed (Fortran convention)
            rpt_list = []
            current_rpt = None
            current_mat = None
            rpt_index = -1

            for _ in range(nrpt * norbs * norbs):
                parts = f.readline().strip().split()
                rx, ry, rz = int(parts[0]), int(parts[1]), int(parts[2])
                i_orb = int(parts[3]) - 1   # Convert to 0-indexed
                j_orb = int(parts[4]) - 1   # Convert to 0-indexed
                h_real = float(parts[5])
                h_imag = float(parts[6])

                rpt = (rx, ry, rz)

                if rpt != current_rpt:
                    if current_rpt is not None:
                        hr[current_rpt] = current_mat

                    current_rpt = rpt
                    rpt_index += 1
                    current_mat = np.zeros((norbs, norbs), dtype=np.complex128)
                    deg_rpt[rpt] = deg_values[rpt_index]

                # Note: Wannier90 writes in column-major order:
                # i runs fastest (inner loop), j runs slowest (outer loop)
                # File ordering is: (1,1), (2,1), ..., (n,1), (1,2), ...
                current_mat[i_orb, j_orb] = h_real + 1j * h_imag

            # Don't forget the last R-point
            if current_rpt is not None:
                hr[current_rpt] = current_mat

        return HamiltonianData(norbs=norbs, hr=hr, deg_rpt=deg_rpt)

    def to_file(
        self,
        filename: str,
        header: Optional[str] = None,
        threshold: float = 0.0
    ) -> int:
        """
        Write Hamiltonian data to wannier90_hr.dat format.

        Parameters
        ----------
        filename : str
            Output file path
        header : str, optional
            Header comment line. Default: auto-generated with timestamp.
        threshold : float
            Drop R-points where sum of |H(R)| < threshold.
            Set to 0 to keep all R-points.

        Returns
        -------
        int
            Number of R-points written
        """
        if header is None:
            dt = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            header = f"Symmetrized by WannSym on {dt}"

        # Collect R-points to write, optionally applying threshold
        rpts_to_write = []
        for R in sorted(self.hr.keys()):
            mat = self.hr[R]
            if threshold > 0 and np.abs(mat).sum() < threshold:
                continue
            rpts_to_write.append(R)

        nrpt = len(rpts_to_write)

        with open(filename, 'w') as f:
            # Header
            f.write(f"{header}\n")
            f.write(f"{self.norbs:12d}\n")
            f.write(f"{nrpt:12d}\n")

            # Degeneracy values, 15 per line
            for i, R in enumerate(rpts_to_write):
                deg = self.deg_rpt.get(R, 1)
                if i > 0 and i % 15 == 0:
                    f.write("\n")
                f.write(f"{deg:5d}")
            f.write("\n")

            # Matrix elements
            for R in rpts_to_write:
                rx, ry, rz = R
                mat = self.hr[R]
                # Write in Wannier90 column-major order: j (outer), i (inner)
                for j in range(self.norbs):
                    for i in range(self.norbs):
                        rp = mat[i, j].real
                        ip = mat[i, j].imag
                        f.write(
                            f"{rx:5d}{ry:5d}{rz:5d}"
                            f"{i+1:5d}{j+1:5d}"
                            f"{rp:20.14f}{ip:20.14f}\n"
                        )

        return nrpt

    def normalize_degeneracies(self) -> None:
        """
        Divide each H(R) by its degeneracy and set all degeneracies to 1.

        This normalization must be applied before symmetrization.
        Reference: symmhr_addrptblock.py lines 100-103.
        """
        for R in list(self.hr.keys()):
            deg = self.deg_rpt.get(R, 1)
            if deg != 1:
                self.hr[R] = self.hr[R] / float(deg)
                self.deg_rpt[R] = 1

    def compress(self, threshold: float = 1e-9) -> int:
        """
        Remove R-points with negligible hoppings.

        Parameters
        ----------
        threshold : float
            Drop R-points where sum(|H(R)|) < threshold

        Returns
        -------
        int
            Number of R-points removed
        """
        to_remove = []
        for R, mat in self.hr.items():
            if np.abs(mat).sum() < threshold:
                to_remove.append(R)

        for R in to_remove:
            del self.hr[R]
            if R in self.deg_rpt:
                del self.deg_rpt[R]

        return len(to_remove)

    def copy(self) -> 'HamiltonianData':
        """Return a deep copy."""
        return HamiltonianData(
            norbs=self.norbs,
            hr={R: mat.copy() for R, mat in self.hr.items()},
            deg_rpt=dict(self.deg_rpt),
        )
