#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
sys.path.append('../') # Assuming Element.py is in the parent directory of where this file will be
import numpy as np
from element.Element import CElement # Assuming Element.py is in a subfolder 'element' relative to the CWD, or PYTHONPATH is set

class CH8(CElement):
    """
    三维8节点六面体实体单元 (H8单元 / Trilinear Brick Element).
    每个节点有3个平动自由度 (UX, UY, UZ)。单元共24个自由度。
    采用标准 isoparametric Formulation 和 2x2x2 高斯数值积分。
    This class now conforms to the CElement interface.
    """

    def __init__(self):
        """
        构造函数 (Constructor).
        Initializes attributes based on CElement and H8 specifics.
        """
        super().__init__()
        self._NEN = 8  # Number of nodes per element
        self._nodes = [None for _ in range(self._NEN)] # Node list
        self.num_dof_per_node = 3 # Each node has 3 translational DOFs (UX, UY, UZ)
        self._ND = self._NEN * self.num_dof_per_node  # Total DOFs for the element (24)
        self._LocationMatrix = np.zeros(self._ND, dtype=np.int_)

        # 获取高斯积分点和权重 (2x2x2方案)
        # Get Gauss points and weights (2x2x2 scheme)
        self.gauss_points, self.gauss_weights = self._get_gauss_quadrature_3d(order=2)
        self._D_matrix = None # To store material elasticity matrix

    def Read(self, input_file, Ele, MaterialSets, NodeList):
        """
        Read element data from stream Input.
        Overrides CElement.Read.
        Expected input line format: N N1 N2 N3 N4 N5 N6 N7 N8 MSet
        where N is element number, N1-N8 are node numbers, MSet is material set number.

        :param input_file: (_io.TextIOWrapper) the object of input file
        :param Ele: (int) check index (current element number being read, 0-indexed)
        :param MaterialSets: (list(CMaterial)) the material list in Domain
        :param NodeList: (list(CNode)) the node list in Domain
        :return: None
        """
        line = input_file.readline().split()

        N_read = int(line[0])
        if N_read != Ele + 1:
            error_info = (f"\n*** Error *** Elements must be inputted in order !"
                          f"\n   Expected element : {Ele + 1}"
                          f"\n   Provided element : {N_read}")
            raise ValueError(error_info)

        # Node numbers (1-indexed from input file)
        node_indices = [int(line[i+1]) for i in range(self._NEN)]
        MSet = int(line[self._NEN + 1])

        self._ElementMaterial = MaterialSets[MSet - 1]
        for i in range(self._NEN):
            self._nodes[i] = NodeList[node_indices[i] - 1]

        # Pre-calculate D matrix if material is available
        self._D_matrix = self._get_material_elasticity_matrix()


    def Write(self, output_file, Ele):
        """
        Write element data to stream.
        Overrides CElement.Write.

        :param output_file: (_io.TextIOWrapper) the object of output file
        :param Ele: (int) the element number (0-indexed)
        :return: None
        """
        node_numbers_str = " ".join([f"{node.NodeNumber:5d}" for node in self._nodes])
        element_info = (f"{Ele+1:5d} Nodes: [ {node_numbers_str} ] "
                        f"MatSet: {self._ElementMaterial.nset:5d}\n") # Assuming CMaterial has 'nset'

        # print the element info on the screen
        print(element_info, end='')
        # write the element info to output file
        output_file.write(element_info)

    def GenerateLocationMatrix(self):
        """
        Generate location matrix: the global equation number that
        corresponding to each DOF of the element.
        Overrides CElement.GenerateLocationMatrix.
        Assumes each node object in self._nodes has a 'bcode' attribute
        which is an array/list of 3 global DOF numbers [UX, UY, UZ].
        """
        idx = 0
        for N_idx in range(self._NEN): # For each node in the element
            for D_idx in range(self.num_dof_per_node): # For each DOF of that node (X, Y, Z)
                if self._nodes[N_idx] is not None and hasattr(self._nodes[N_idx], 'bcode'):
                    self._LocationMatrix[idx] = self._nodes[N_idx].bcode[D_idx]
                else:
                    # This case should ideally not happen if nodes are set up correctly
                    self._LocationMatrix[idx] = 0 # Or raise an error
                idx += 1

    def SizeOfStiffnessMatrix(self):
        """
        Return the size of the element stiffness matrix
        (stored as an array column by column, upper triangular part).
        For an H8 element (24x24 matrix), size = N*(N+1)/2 = 24*25/2 = 300.
        Overrides CElement.SizeOfStiffnessMatrix.
        """
        return int(self._ND * (self._ND + 1) / 2)


    def ElementStiffness(self, stiffness_array):
        """
        Calculate element stiffness matrix (Ke).
        The result is stored in 'stiffness_array' as the upper triangular
        part, column by column.
        Overrides CElement.ElementStiffness.

        :param stiffness_array: (np.ndarray) pre-allocated 1D array to store stiffness values.
        """
        Ke_full = np.zeros((self._ND, self._ND)) # Full 24x24 element stiffness matrix

        if self._D_matrix is None:
            self._D_matrix = self._get_material_elasticity_matrix() # Ensure D matrix is calculated

        # node_coords = np.array([node.XYZ for node in self._nodes]) # (8, 3) array of [x,y,z]
        # Ensure node.XYZ exists and is in the correct format.
        # The original H8.py uses node.coordinates. We'll assume node.XYZ from STAPpy CNode.
        node_coords = np.zeros((self._NEN, 3))
        for i in range(self._NEN):
            if self._nodes[i] is not None and hasattr(self._nodes[i], 'XYZ'):
                node_coords[i,:] = self._nodes[i].XYZ
            else:
                raise AttributeError(f"Node {i} in element is not initialized or missing XYZ coordinates.")


        for gp_idx, gp_coords_natural in enumerate(self.gauss_points):
            xi, eta, zeta = gp_coords_natural
            weight = self.gauss_weights[gp_idx]

            # 1. 形函数对自然坐标的导数 dN_dXiEtaZeta (3x8)
            #    Derivatives of shape functions w.r.t. natural coordinates
            dN_dXiEtaZeta = self._shape_function_derivatives_natural(xi, eta, zeta)

            # 2. 雅可比矩阵 J (3x3) = dN_dXiEtaZeta @ node_coords
            #    Jacobian matrix J
            J_mat = dN_dXiEtaZeta @ node_coords  # (3x8) @ (8x3) = (3x3)

            det_J = np.linalg.det(J_mat)

            if det_J <= 1e-12: # Using a small tolerance instead of strict zero
                # This can happen for distorted elements.
                raise ValueError(
                    f"Element (internal ID, needs actual ID from Domain) "
                    f"at Gauss point {gp_coords_natural} has non-positive or very small Jacobian determinant: {det_J}. "
                    f"Element may be excessively distorted or flat.")

            inv_J_mat = np.linalg.inv(J_mat)

            # 3. 形函数对全局坐标的导数 dN_dXYZ (3x8) = inv_J @ dN_dXiEtaZeta
            #    Derivatives of shape functions w.r.t. global coordinates
            dN_dXYZ = inv_J_mat @ dN_dXiEtaZeta

            # 4. 应变-位移矩阵 B (6x24)
            #    Strain-displacement matrix B
            B_mat = np.zeros((6, self._ND)) # 6 strain components, 24 DOFs
            for k_node_idx in range(self._NEN):  # For each node k (0 to 7)
                dNk_dx = dN_dXYZ[0, k_node_idx]
                dNk_dy = dN_dXYZ[1, k_node_idx]
                dNk_dz = dN_dXYZ[2, k_node_idx]

                # Columns for node k's DOFs (ux_k, uy_k, uz_k)
                # ux_k is at index k_node_idx * num_dof_per_node
                # uy_k is at index k_node_idx * num_dof_per_node + 1
                # uz_k is at index k_node_idx * num_dof_per_node + 2
                col_start_idx = k_node_idx * self.num_dof_per_node

                # Strain_xx (epsilon_x) contribution
                B_mat[0, col_start_idx] = dNk_dx
                # Strain_yy (epsilon_y) contribution
                B_mat[1, col_start_idx + 1] = dNk_dy
                # Strain_zz (epsilon_z) contribution
                B_mat[2, col_start_idx + 2] = dNk_dz

                # Strain_xy (gamma_xy) contribution
                B_mat[3, col_start_idx] = dNk_dy
                B_mat[3, col_start_idx + 1] = dNk_dx

                # Strain_yz (gamma_yz) contribution
                B_mat[4, col_start_idx + 1] = dNk_dz
                B_mat[4, col_start_idx + 2] = dNk_dy

                # Strain_zx (gamma_zx) contribution
                B_mat[5, col_start_idx] = dNk_dz
                B_mat[5, col_start_idx + 2] = dNk_dx

            # 5. 累加到单元刚度矩阵
            #    Accumulate to element stiffness matrix
            # Ke_contribution = B_transpose * D * B * det(J) * weight
            Ke_full += B_mat.T @ self._D_matrix @ B_mat * det_J * weight

        # Store the upper triangular part of Ke_full into stiffness_array, column by column
        idx = 0
        for j_col in range(self._ND):  # Iterate over columns
            for i_row in range(j_col, -1, -1):  # Iterate over rows (up to and including diagonal)
                stiffness_array[idx] = Ke_full[i_row, j_col]
                idx += 1


    def ElementStress(self, stress_output_array, displacement_global):
        """
        Calculate element stress components [sigma_xx, sigma_yy, sigma_zz, tau_xy, tau_yz, tau_zx]
        at the element center (xi=0, eta=0, zeta=0).
        The result is stored in 'stress_output_array'.
        Overrides CElement.ElementStress.

        :param stress_output_array: (np.ndarray) pre-allocated 1D array (size 6) to store stress components.
        :param displacement_global: (np.ndarray) global displacement vector for the entire structure.
        :return: None
        """
        if stress_output_array.shape[0] < 6:
            raise ValueError("stress_output_array must have at least 6 components for H8 element.")

        # Extract local element displacements from the global displacement vector
        element_displacements_local = np.zeros(self._ND)
        for i in range(self._ND):
            global_dof_index = self._LocationMatrix[i]
            if global_dof_index > 0: # Equation number is 1-indexed in STAP
                element_displacements_local[i] = displacement_global[global_dof_index - 1]
            # else: DOF is constrained or not active, displacement is 0

        if self._D_matrix is None:
            self._D_matrix = self._get_material_elasticity_matrix()

        # node_coords = np.array([node.XYZ for node in self._nodes])
        node_coords = np.zeros((self._NEN, 3))
        for i in range(self._NEN):
            if self._nodes[i] is not None and hasattr(self._nodes[i], 'XYZ'):
                node_coords[i,:] = self._nodes[i].XYZ
            else:
                raise AttributeError(f"Node {i} in element is not initialized or missing XYZ coordinates for stress calculation.")

        # Calculate stress at the center of the element (xi=0, eta=0, zeta=0)
        xi, eta, zeta = 0.0, 0.0, 0.0

        # 1. Shape function derivatives at natural coordinates
        dN_dXiEtaZeta = self._shape_function_derivatives_natural(xi, eta, zeta)

        # 2. Jacobian matrix J and its inverse
        J_mat = dN_dXiEtaZeta @ node_coords
        # det_J = np.linalg.det(J_mat) # Not strictly needed here, but good for check
        # if det_J <= 1e-12: raise ValueError("Jacobian determinant zero or negative at element center.")
        try:
            inv_J_mat = np.linalg.inv(J_mat)
        except np.linalg.LinAlgError:
             raise ValueError(
                    f"Element (internal ID, needs actual ID from Domain) "
                    f"at center has singular Jacobian matrix. Element may be excessively distorted or flat.")


        # 3. Shape function derivatives at global coordinates
        dN_dXYZ = inv_J_mat @ dN_dXiEtaZeta

        # 4. Strain-displacement matrix B
        B_mat = np.zeros((6, self._ND))
        for k_node_idx in range(self._NEN):
            dNk_dx, dNk_dy, dNk_dz = dN_dXYZ[0, k_node_idx], dN_dXYZ[1, k_node_idx], dN_dXYZ[2, k_node_idx]
            col_start_idx = k_node_idx * self.num_dof_per_node

            B_mat[0, col_start_idx] = dNk_dx
            B_mat[1, col_start_idx + 1] = dNk_dy
            B_mat[2, col_start_idx + 2] = dNk_dz
            B_mat[3, col_start_idx] = dNk_dy
            B_mat[3, col_start_idx + 1] = dNk_dx
            B_mat[4, col_start_idx + 1] = dNk_dz
            B_mat[4, col_start_idx + 2] = dNk_dy
            B_mat[5, col_start_idx] = dNk_dz
            B_mat[5, col_start_idx + 2] = dNk_dx

        # 5. Calculate strain vector {epsilon} = B @ {u_e}
        strain_vector = B_mat @ element_displacements_local  # (6,)

        # 6. Calculate stress vector {sigma} = D @ {epsilon}
        stress_vector = self._D_matrix @ strain_vector  # (6,)

        # Store results in the output array
        for i in range(6):
            stress_output_array[i] = stress_vector[i]

    # Internal helper methods from the original H8.py (prefixed with _ if not already)
    # These are mostly unchanged but are now part of the CH8 class.

    def _get_gauss_quadrature_3d(self, order: int):
        """获取3D高斯积分点 (自然坐标) 和权重。
           Get 3D Gauss points (natural coordinates) and weights."""
        if order == 2:  # 2x2x2 = 8 points
            gp_val = 1.0 / np.sqrt(3.0)
            points_1d = [-gp_val, gp_val]
            weights_1d = [1.0, 1.0]
        elif order == 1:  # 1x1x1 = 1 point (reduced integration)
            points_1d = [0.0]
            weights_1d = [2.0]
        else:
            raise ValueError(f"Unsupported Gauss integration order: {order}")

        gauss_points = []
        gauss_weights = []
        for w_k, p_k in zip(weights_1d, points_1d):  # Zeta direction
            for w_j, p_j in zip(weights_1d, points_1d):  # Eta direction
                for w_i, p_i in zip(weights_1d, points_1d):  # Xi direction
                    gauss_points.append(np.array([p_i, p_j, p_k]))  # [xi, eta, zeta]
                    gauss_weights.append(w_i * w_j * w_k)
        return gauss_points, gauss_weights

    def _shape_functions(self, xi: float, eta: float, zeta: float) -> np.ndarray:
        """
        计算给定自然坐标处的8个形函数值 N_k。
        Calculate the 8 shape function values N_k at given natural coordinates.
        返回一个 (8,) 的NumPy数组。 Returns a (8,) NumPy array.
        """
        N = np.zeros(8)
        N[0] = 0.125 * (1 - xi) * (1 - eta) * (1 - zeta)
        N[1] = 0.125 * (1 + xi) * (1 - eta) * (1 - zeta)
        N[2] = 0.125 * (1 + xi) * (1 + eta) * (1 - zeta)
        N[3] = 0.125 * (1 - xi) * (1 + eta) * (1 - zeta)
        N[4] = 0.125 * (1 - xi) * (1 - eta) * (1 + zeta)
        N[5] = 0.125 * (1 + xi) * (1 - eta) * (1 + zeta)
        N[6] = 0.125 * (1 + xi) * (1 + eta) * (1 + zeta)
        N[7] = 0.125 * (1 - xi) * (1 + eta) * (1 + zeta)
        return N

    def _shape_function_derivatives_natural(self, xi: float, eta: float, zeta: float) -> np.ndarray:
        """
        计算给定自然坐标处的形函数对自然坐标 (xi, eta, zeta) 的偏导数。
        Calculate partial derivatives of shape functions w.r.t. natural coordinates (xi, eta, zeta)
        at the given natural coordinates.
        返回一个 (3, 8) 的NumPy数组 dN_dXiEtaZeta，其中: Returns a (3,8) NumPy array where:
        dN_dXiEtaZeta[0, k] = dN_k / dxi
        dN_dXiEtaZeta[1, k] = dN_k / deta
        dN_dXiEtaZeta[2, k] = dN_k / dzeta
        """
        dN_dXiEtaZeta = np.zeros((3, 8))

        # Derivatives w.r.t. xi
        dN_dXiEtaZeta[0,0] = -0.125 * (1 - eta) * (1 - zeta)
        dN_dXiEtaZeta[0,1] =  0.125 * (1 - eta) * (1 - zeta)
        dN_dXiEtaZeta[0,2] =  0.125 * (1 + eta) * (1 - zeta)
        dN_dXiEtaZeta[0,3] = -0.125 * (1 + eta) * (1 - zeta)
        dN_dXiEtaZeta[0,4] = -0.125 * (1 - eta) * (1 + zeta)
        dN_dXiEtaZeta[0,5] =  0.125 * (1 - eta) * (1 + zeta)
        dN_dXiEtaZeta[0,6] =  0.125 * (1 + eta) * (1 + zeta)
        dN_dXiEtaZeta[0,7] = -0.125 * (1 + eta) * (1 + zeta)

        # Derivatives w.r.t. eta
        dN_dXiEtaZeta[1,0] = -0.125 * (1 - xi) * (1 - zeta)
        dN_dXiEtaZeta[1,1] = -0.125 * (1 + xi) * (1 - zeta)
        dN_dXiEtaZeta[1,2] =  0.125 * (1 + xi) * (1 - zeta)
        dN_dXiEtaZeta[1,3] =  0.125 * (1 - xi) * (1 - zeta)
        dN_dXiEtaZeta[1,4] = -0.125 * (1 - xi) * (1 + zeta)
        dN_dXiEtaZeta[1,5] = -0.125 * (1 + xi) * (1 + zeta)
        dN_dXiEtaZeta[1,6] =  0.125 * (1 + xi) * (1 + zeta)
        dN_dXiEtaZeta[1,7] =  0.125 * (1 - xi) * (1 + zeta)

        # Derivatives w.r.t. zeta
        dN_dXiEtaZeta[2,0] = -0.125 * (1 - xi) * (1 - eta)
        dN_dXiEtaZeta[2,1] = -0.125 * (1 + xi) * (1 - eta)
        dN_dXiEtaZeta[2,2] = -0.125 * (1 + xi) * (1 + eta)
        dN_dXiEtaZeta[2,3] = -0.125 * (1 - xi) * (1 + eta)
        dN_dXiEtaZeta[2,4] =  0.125 * (1 - xi) * (1 - eta)
        dN_dXiEtaZeta[2,5] =  0.125 * (1 + xi) * (1 - eta)
        dN_dXiEtaZeta[2,6] =  0.125 * (1 + xi) * (1 + eta)
        dN_dXiEtaZeta[2,7] =  0.125 * (1 - xi) * (1 + eta)

        return dN_dXiEtaZeta

    def _get_material_elasticity_matrix(self) -> np.ndarray:
        """获取材料的3D弹性矩阵 D (6x6)。
           Get the material's 3D elasticity matrix D (6x6).
           Assumes self._ElementMaterial is set and has 'E' and 'NU' attributes,
           or a method 'get_elasticity_matrix_3d()'.
           STAPpy CMaterial typically has E, NU, G. Here we'll use E and NU.
        """
        if self._ElementMaterial is None:
            raise ValueError("ElementMaterial not set. Cannot compute elasticity matrix.")

        if hasattr(self._ElementMaterial, 'get_elasticity_matrix_3d'): # If a direct method exists
             return self._ElementMaterial.get_elasticity_matrix_3d()
        elif hasattr(self._ElementMaterial, 'E') and hasattr(self._ElementMaterial, 'nu'): # STAPpy uses 'nu'
            E = self._ElementMaterial.E
            nu = self._ElementMaterial.nu # Poisson's ratio
            
            # Check for G, if not present, calculate it for isotropic material
            if hasattr(self._ElementMaterial, 'G'):
                G = self._ElementMaterial.G
            else:
                G = E / (2 * (1 + nu)) # Shear modulus for isotropic material

            D_matrix = np.zeros((6, 6))
            
            # For isotropic material
            c1 = E / ((1 + nu) * (1 - 2 * nu))
            
            D_matrix[0, 0] = c1 * (1 - nu) # D11 (sigma_x / epsilon_x)
            D_matrix[1, 1] = c1 * (1 - nu) # D22 (sigma_y / epsilon_y)
            D_matrix[2, 2] = c1 * (1 - nu) # D33 (sigma_z / epsilon_z)

            D_matrix[0, 1] = c1 * nu      # D12
            D_matrix[1, 0] = c1 * nu      # D21
            D_matrix[0, 2] = c1 * nu      # D13
            D_matrix[2, 0] = c1 * nu      # D31
            D_matrix[1, 2] = c1 * nu      # D23
            D_matrix[2, 1] = c1 * nu      # D32

            D_matrix[3, 3] = G            # D44 (tau_xy / gamma_xy)
            D_matrix[4, 4] = G            # D55 (tau_yz / gamma_yz)
            D_matrix[5, 5] = G            # D66 (tau_zx / gamma_zx)
            return D_matrix
        else:
            raise AttributeError(
                "ElementMaterial object must provide 'get_elasticity_matrix_3d()' method "
                "or 'E' and 'nu' (Poisson's ratio) attributes.")