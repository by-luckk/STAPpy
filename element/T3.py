#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
/******************************************************************************
*  STAPpy : A python FEM code sharing the same input data file with STAP90    *
*     Computational Dynamics Laboratory                                       *
*     School of Aerospace Engineering, Tsinghua University                    *
*                                                                             *
*     Constant‑strain triangular element (CST / T3)                           *
*                                                                             *
*     This file implements class **CT3** which derives from ``CElement``      *
*     following the coding style used in ``Bar.py``.                          *
*                                                                             *
*  Created on  : 29‑May‑2025                                                  *
*******************************************************************************/
"""
import sys
sys.path.append('../')
from element.Element import CElement
import numpy as np


class CT3(CElement):
    """Three‑node constant strain triangle element"""

    def __init__(self):
        super().__init__()
        # ---- basic data ----
        self._NEN = 3                 # number of nodes / element
        self._nodes = [None] * self._NEN
        self._ND   = self._NEN * 2    # 3 translational dof / node (x,y,z)
        self._LocationMatrix = np.zeros(self._ND, dtype=np.int_)

    # ---------------------------------------------------------------------
    #                       1.  I/O ROUTINES
    # ---------------------------------------------------------------------
    def Read(self, input_file, Ele, MaterialSets, NodeList):
        """Read element definition line.
        Expected format (one line):
            elem_id   n1   n2   n3   mset
        """
        parts = input_file.readline().split()
        if not parts:
            raise EOFError("Unexpected end of file when reading element data")

        N = int(parts[0])
        if N != Ele + 1:
            raise ValueError(f"*** Error *** Elements must be inputted in order!\n"
                             f"   Expected element : {Ele + 1}\n   Provided element : {N}")

        n1, n2, n3 = map(int, parts[1:4])
        mset       = int(parts[4])

        # store nodes / material
        self._nodes[0]        = NodeList[n1 - 1]
        self._nodes[1]        = NodeList[n2 - 1]
        self._nodes[2]        = NodeList[n3 - 1]
        self._ElementMaterial = MaterialSets[mset - 1]

    def Write(self, output_file, Ele):
        """Write element data to stream (same layout as input)."""
        line = f"{Ele + 1:5d}{self._nodes[0].NodeNumber:11d}{self._nodes[1].NodeNumber:9d}"
        line += f"{self._nodes[2].NodeNumber:9d}{self._ElementMaterial.nset:12d}\n"
        print(line, end="")            # echo on console
        output_file.write(line)        # write to *.out

    # ---------------------------------------------------------------------
    #                       2.  ASSEMBLY HELPERS
    # ---------------------------------------------------------------------
    def GenerateLocationMatrix(self):
        """Global equation numbers for each element dof."""
        idx = 0
        for node in self._nodes:
            for d in range(2):              # x, y, z dof
                self._LocationMatrix[idx] = node.bcode[d]
                idx += 1

    def SizeOfStiffnessMatrix(self):
        """Return number of entries in the upper‑triangular storage (9×9)."""
        return 21

    # ---------------------------------------------------------------------
    #                       3.  ELEMENT STIFFNESS
    # ---------------------------------------------------------------------
    def _constitutive_matrix(self):
        """Plane‑stress constitutive matrix D (3×3)."""
        mat = self._ElementMaterial
        E   = getattr(mat, "E")
        # Poisson ratio can be named v or nu
        nu  = getattr(mat, "nu", None)
        if nu is None:
            nu = getattr(mat, "v", None)
        if nu is None:
            raise AttributeError("Material must provide Poisson's ratio as 'nu' or 'v'.")

        coeff = E / (1.0 - nu ** 2)
        D = coeff * np.array([[1.0,   nu,        0.0],
                               [nu,    1.0,       0.0],
                               [0.0,   0.0,  (1.0 - nu) / 2]])
        return D

    def _B_matrix(self):
        """Strain‑displacement matrix B (3×6) for CST."""
        # local xy coordinates only
        x = [node.XYZ[0] for node in self._nodes]
        y = [node.XYZ[1] for node in self._nodes]

        # area and geometric coefficients
        b = [y[1] - y[2], y[2] - y[0], y[0] - y[1]]
        c = [x[2] - x[1], x[0] - x[2], x[1] - x[0]]
        area = np.abs(0.5 * (b[0] * c[2] - b[2] * c[0]))   # same as det/2
        if area == 0.0:
            raise ValueError("*** Error *** CT3 element has ZERO area!")

        factor = 1.0 / (2.0 * area)
        B = np.zeros((3, 6))
        for i in range(3):
            B[0, 2 * i]     = b[i]
            B[1, 2 * i + 1] = c[i]
            B[2, 2 * i]     = c[i]
            B[2, 2 * i + 1] = b[i]
        B *= factor
        return B, area

    def ElementStiffness(self, stiffness):
        """Fill *stiffness* (1‑D upper triangular array) with 9×9 matrix."""
        # prepare
        D        = self._constitutive_matrix()
        B, area  = self._B_matrix()
        t        = getattr(self._ElementMaterial, "thickness", 1.0)

        k6 = t * area * (B.T @ D @ B)      # 6×6 for in‑plane dof
        # store upper triangular part column‑wise
        idx = 0
        for col in range(6):
            for row in range(col, -1, -1):
                stiffness[idx] = k6[row, col]
                idx += 1
    # ---------------------------------------------------------------------
    #                             4.  STRESS
    # ---------------------------------------------------------------------
    def ElementStress(self, stress, displacement):
        """Compute σ_x, σ_y, τ_xy at element (constant over element)."""
        # gather in‑plane nodal displacements (u1,v1,u2,v2,u3,v3)
        u = np.zeros(6)
        map_inplane = [0, 1, 3, 4, 6, 7]
        for local_idx, lm_idx in enumerate(map_inplane):
            eq = self._LocationMatrix[lm_idx]
            if eq:                          # 0 → prescribed displacement
                u[local_idx] = displacement[eq - 1]

        B, _    = self._B_matrix()
        D       = self._constitutive_matrix()
        strain  = B @ u
        sigma   = D @ strain

        # output
        stress[0] = sigma[0]  # σ_x
        stress[1] = sigma[1]  # σ_y
        stress[2] = sigma[2]  # τ_xy
