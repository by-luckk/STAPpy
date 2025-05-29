#H8求解
# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('../')
import numpy as np
from element.Element import CElement


class H8:
    """
    三维8节点六面体实体单元 (H8单元 / Trilinear Brick Element)。
    每个节点有3个平动自由度 (UX, UY, UZ)。单元共24个自由度。
    采用标准 isoparametric Formulation 和 2x2x2 高斯数值积分。
    """

    def __init__(self, id: int, nodes: list, material: object):
        """
        构造函数。

        参数:
            id (int): 单元的唯一标识符。
            nodes (list): 包含8个Node对象的列表。
                          节点顺序应遵循 isoparametric 单元的标准编号约定。
                          每个Node对象应有 'id' (int) 和 'coordinates' (np.ndarray, [x,y,z]) 属性。
            material (object): 材料对象。
                               应有 'get_elasticity_matrix_3d()' 方法返回3D弹性矩阵D (6x6 np.ndarray)
                               或 'E' (杨氏模量) 和 'NU' (泊松比) 属性。
        """
        if len(nodes) != 8:
            raise ValueError("H8单元必须有8个节点。")

        self.id = id
        self.nodes = nodes  # 列表，包含8个Node对象
        self.material = material
        self.num_dof_per_node = 3
        self.num_nodes = 8
        self.num_dof_element = self.num_nodes * self.num_dof_per_node  # 24 DOFs

        # 获取高斯积分点和权重 (2x2x2方案)
        self.gauss_points, self.gauss_weights = self._get_gauss_quadrature_3d(order=2)

    def _get_gauss_quadrature_3d(self, order: int):
        """获取3D高斯积分点 (自然坐标) 和权重。"""
        if order == 2:  # 2x2x2 = 8 points
            gp_val = 1.0 / np.sqrt(3.0)
            points_1d = [-gp_val, gp_val]
            weights_1d = [1.0, 1.0]
        elif order == 1:  # 1x1x1 = 1 point (reduced integration)
            points_1d = [0.0]
            weights_1d = [2.0]
        else:
            raise ValueError(f"不支持的高斯积分阶次: {order}")

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
        返回一个 (8,) 的NumPy数组。
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
        返回一个 (3, 8) 的NumPy数组 dN_dXiEtaZeta，其中:
        dN_dXiEtaZeta[0, k] = dN_k / dxi
        dN_dXiEtaZeta[1, k] = dN_k / deta
        dN_dXiEtaZeta[2, k] = dN_k / dzeta
        """
        dN_dXiEtaZeta = np.zeros((3, 8))

        # Derivatives w.r.t. xi
        dN_dXiEtaZeta[0,0] = -0.125 * (1 - eta) * (1 - zeta)
        dN_dXiEtaZeta[0,1] = 0.125 * (1 - eta) * (1 - zeta)
        dN_dXiEtaZeta[0,2] = 0.125 * (1 + eta) * (1 - zeta)
        dN_dXiEtaZeta[0,3] = -0.125 * (1 + eta) * (1 - zeta)
        dN_dXiEtaZeta[0,4] = -0.125 * (1 - eta) * (1 + zeta)
        dN_dXiEtaZeta[0,5] = 0.125 * (1 - eta) * (1 + zeta)
        dN_dXiEtaZeta[0,6] = 0.125 * (1 + eta) * (1 + zeta)
        dN_dXiEtaZeta[0,7] = -0.125 * (1 + eta) * (1 + zeta)

        # Derivatives w.r.t. eta
        dN_dXiEtaZeta[1,0] = -0.125 * (1 - xi) * (1 - zeta)
        dN_dXiEtaZeta[1,1] = -0.125 * (1 + xi) * (1 - zeta)
        dN_dXiEtaZeta[1,2] = 0.125 * (1 + xi) * (1 - zeta)
        dN_dXiEtaZeta[1,3] = 0.125 * (1 - xi) * (1 - zeta)
        dN_dXiEtaZeta[1,4] = -0.125 * (1 - xi) * (1 + zeta)
        dN_dXiEtaZeta[1,5] = -0.125 * (1 + xi) * (1 + zeta)
        dN_dXiEtaZeta[1,6] = 0.125 * (1 + xi) * (1 + zeta)
        dN_dXiEtaZeta[1,7] = 0.125 * (1 - xi) * (1 + zeta)

        # Derivatives w.r.t. zeta
        dN_dXiEtaZeta[2,0] = -0.125 * (1 - xi) * (1 - eta)
        dN_dXiEtaZeta[2,1] = -0.125 * (1 + xi) * (1 - eta)
        dN_dXiEtaZeta[2,2] = -0.125 * (1 + xi) * (1 + eta)
        dN_dXiEtaZeta[2,3] = -0.125 * (1 - xi) * (1 + eta)
        dN_dXiEtaZeta[2,4] = 0.125 * (1 - xi) * (1 - eta)
        dN_dXiEtaZeta[2,5] = 0.125 * (1 + xi) * (1 - eta)
        dN_dXiEtaZeta[2,6] = 0.125 * (1 + xi) * (1 + eta)
        dN_dXiEtaZeta[2,7] = 0.125 * (1 - xi) * (1 + eta)

        return dN_dXiEtaZeta

    def get_element_dof_indices(self) -> list:
        """
        获取单元的全局自由度索引列表。
        假设Node对象有 'dof_indices' 属性，该属性是一个包含其全局自由度索引的列表。
        例如: node.dof_indices = [global_idx_ux, global_idx_uy, global_idx_uz]
        """
        dof_indices = []
        for node in self.nodes:
            if not hasattr(node, 'dof_indices') or len(node.dof_indices) != self.num_dof_per_node:
                # Fallback if node.dof_indices is not set as expected
                # This part depends on how Domain class assigns DOFs.
                # For demonstration, we assume a simple contiguous block based on node ID and num_dof_per_node.
                # This is NOT a robust way for a real FEM program.
                # A proper FEM system would have a robust DOF manager.
                # print(f"Warning: Node {node.id} does not have expected dof_indices. Using fallback (may be incorrect).")
                # base_idx = node.id * self.num_dof_per_node # Example, assuming node.id starts from 0 or 1
                # for i in range(self.num_dof_per_node):
                #    dof_indices.append(base_idx + i)
                # A better fallback or error:
                raise AttributeError(f"Node {node.id} does not have 'dof_indices' or it's improperly sized.")
            dof_indices.extend(node.dof_indices)
        return dof_indices

    def _get_material_elasticity_matrix(self) -> np.ndarray:
        """获取材料的3D弹性矩阵 D (6x6)。"""
        if hasattr(self.material, 'get_elasticity_matrix_3d'):
            return self.material.get_elasticity_matrix_3d()
        elif hasattr(self.material, 'E') and hasattr(self.material, 'NU'):
            E = self.material.E
            nu = self.material.NU
            D_matrix = np.zeros((6, 6))
            factor = E / ((1 + nu) * (1 - 2 * nu))

            D_matrix[0, 0] = factor * (1 - nu)
            D_matrix[1, 1] = factor * (1 - nu)
            D_matrix[2, 2] = factor * (1 - nu)

            D_matrix[0, 1] = factor * nu
            D_matrix[1, 0] = factor * nu
            D_matrix[0, 2] = factor * nu
            D_matrix[2, 0] = factor * nu
            D_matrix[1, 2] = factor * nu
            D_matrix[2, 1] = factor * nu

            shear_modulus = E / (2 * (1 + nu))  # G
            D_matrix[3, 3] = shear_modulus
            D_matrix[4, 4] = shear_modulus
            D_matrix[5, 5] = shear_modulus
            return D_matrix
        else:
            raise AttributeError(
                "Material object must provide 'get_elasticity_matrix_3d()' method or 'E' and 'NU' attributes.")

    def compute_stiffness_matrix(self) -> np.ndarray:
        """
        计算H8单元的刚度矩阵 Ke (24x24)。
        """
        Ke = np.zeros((self.num_dof_element, self.num_dof_element))
        D_matrix = self._get_material_elasticity_matrix()

        node_coords = np.array([node.coordinates for node in self.nodes])  # (8, 3) array of [x,y,z]

        for gp_idx, gp_coords_natural in enumerate(self.gauss_points):
            xi, eta, zeta = gp_coords_natural
            weight = self.gauss_weights[gp_idx]

            # 1. 形函数对自然坐标的导数 dN_dXiEtaZeta (3x8)
            dN_dXiEtaZeta = self._shape_function_derivatives_natural(xi, eta, zeta)

            # 2. 雅可比矩阵 J (3x3) = dN_dXiEtaZeta @ node_coords
            # J[i,j] = d(x_j) / d(xi_i) where x_j are (x,y,z) and xi_i are (xi,eta,zeta)
            # J_mat[row_idx, col_idx] = sum over k (dN_k/dNaturalCoord[row_idx] * NodeCoord_k[col_idx])
            J_mat = dN_dXiEtaZeta @ node_coords  # (3x8) @ (8x3) = (3x3)

            det_J = np.linalg.det(J_mat)
            if det_J <= 0:
                # This can happen for distorted elements.
                # A robust implementation might try to handle this or provide more context.
                raise ValueError(
                    f"单元 {self.id} 在积分点 {gp_coords_natural} 处雅可比行列式非正: {det_J}. 单元可能过度扭曲。")

            inv_J_mat = np.linalg.inv(J_mat)

            # 3. 形函数对全局坐标的导数 dN_dXYZ (3x8) = inv_J @ dN_dXiEtaZeta
            dN_dXYZ = inv_J_mat @ dN_dXiEtaZeta

            # 4. 应变-位移矩阵 B (6x24)
            B_mat = np.zeros((6, self.num_dof_element))
            for k in range(self.num_nodes):  # For each node k
                dNk_dx = dN_dXYZ[0, k]
                dNk_dy = dN_dXYZ[1, k]
                dNk_dz = dN_dXYZ[2, k]

                # Columns for node k's DOFs (ux_k, uy_k, uz_k)
                # ux_k is at index k*3
                # uy_k is at index k*3 + 1
                # uz_k is at index k*3 + 2

                # Strain_xx contribution
                B_mat[0, k * 3] = dNk_dx
                # Strain_yy contribution
                B_mat[1, k * 3 + 1] = dNk_dy
                # Strain_zz contribution
                B_mat[2, k * 3 + 2] = dNk_dz

                # Strain_xy (gamma_xy) contribution
                B_mat[3, k * 3] = dNk_dy
                B_mat[3, k * 3 + 1] = dNk_dx

                # Strain_yz (gamma_yz) contribution
                B_mat[4, k * 3 + 1] = dNk_dz
                B_mat[4, k * 3 + 2] = dNk_dy

                # Strain_zx (gamma_zx) contribution
                B_mat[5, k * 3] = dNk_dz
                B_mat[5, k * 3 + 2] = dNk_dx

            # 5. 累加到单元刚度矩阵
            # Ke_contribution = B_transpose * D * B * det(J) * weight
            Ke += B_mat.T @ D_matrix @ B_mat * det_J * weight

        return Ke

    def compute_stress_strain(self, element_displacements_global: np.ndarray,
                              points_natural: list = None) -> dict:
        """
        计算单元内指定点的应变和应力。

        参数:
            element_displacements_global (np.ndarray): 单元节点位移向量 (24x1)。
            points_natural (list, optional):
                一个列表，包含希望计算应力应变的点的自然坐标 [xi, eta, zeta]。
                如果为 None，则默认在所有高斯积分点计算。

        返回:
            dict: 键为点描述 (如 "GP0", "GP1",... 或 "Center")，
                  值为一个字典 {'strain': np.ndarray(6x1), 'stress': np.ndarray(6x1)}。
                  应变/应力分量顺序: [eps_xx, eps_yy, eps_zz, gam_xy, gam_yz, gam_zx]
        """
        if element_displacements_global.shape != (self.num_dof_element,):
            element_displacements_global = element_displacements_global.reshape(
                self.num_dof_element, )  # Ensure it's a 1D array if passed as (N,1)

        results = {}
        D_matrix = self._get_material_elasticity_matrix()
        node_coords = np.array([node.coordinates for node in self.nodes])

        if points_natural is None:
            # Default to Gauss points if no specific points are given
            eval_points = self.gauss_points
            point_names = [f"GP{i}" for i in range(len(self.gauss_points))]
        else:
            eval_points = points_natural
            point_names = [f"UserPoint{i}" for i in range(len(points_natural))]
            if not eval_points:  # Handle empty list case
                return {}

        for pt_idx, (xi, eta, zeta) in enumerate(eval_points):
            # 1. 形函数对自然坐标的导数 dN_dXiEtaZeta
            dN_dXiEtaZeta = self._shape_function_derivatives_natural(xi, eta, zeta)

            # 2. 雅可比矩阵 J 和其逆
            J_mat = dN_dXiEtaZeta @ node_coords
            # det_J = np.linalg.det(J_mat) # Not strictly needed for stress/strain, but good check
            inv_J_mat = np.linalg.inv(J_mat)

            # 3. 形函数对全局坐标的导数 dN_dXYZ
            dN_dXYZ = inv_J_mat @ dN_dXiEtaZeta

            # 4. 应变-位移矩阵 B
            B_mat = np.zeros((6, self.num_dof_element))
            for k in range(self.num_nodes):
                dNk_dx, dNk_dy, dNk_dz = dN_dXYZ[0, k], dN_dXYZ[1, k], dN_dXYZ[2, k]
                B_mat[0, k * 3] = dNk_dx
                B_mat[1, k * 3 + 1] = dNk_dy
                B_mat[2, k * 3 + 2] = dNk_dz
                B_mat[3, k * 3] = dNk_dy;
                B_mat[3, k * 3 + 1] = dNk_dx
                B_mat[4, k * 3 + 1] = dNk_dz;
                B_mat[4, k * 3 + 2] = dNk_dy
                B_mat[5, k * 3] = dNk_dz;
                B_mat[5, k * 3 + 2] = dNk_dx

            # 5. 计算应变 {epsilon} = {u_e}
            strain_vector = B_mat @ element_displacements_global  # (6,)

            # 6. 计算应力 {sigma} = {epsilon}
            stress_vector = D_matrix @ strain_vector  # (6,)

            point_name = point_names[pt_idx]
            results[point_name] = {
                'strain': strain_vector.reshape(6, 1),  # Store as column vector
                'stress': stress_vector.reshape(6, 1)  # Store as column vector
            }

        return results

    def get_node_connectivity(self) -> list:
        """返回组成该单元的节点ID列表。"""
        return [node.id for node in self.nodes]
