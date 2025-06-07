#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
/*****************************************************************************/
/*  STAPpy : A python FEM code sharing the same input data file with STAP90  */
/*     Computational Dynamics Laboratory                                     */
/*     School of Aerospace Engineering, Tsinghua University                  */
/*                                                                           */
/*     Plate element implementation                                          */
/*                                                                           */
/*     @author: thurcni@163.com, xzhang@tsinghua.edu.cn                      */
/*     http://www.comdyn.cn/                                                 */
/*****************************************************************************/
"""

import sys
import numpy as np
from element.Element import CElement

class CPlate(CElement):
    """ Plate element class """
    def __init__(self):
        super().__init__()
        self._NEN = 4  # 4节点板单元
        self._nodes = [None for _ in range(self._NEN)]
        
        # 每个节点3个自由度(w,θx,θy)，共12个自由度
        self._ND = 12  
        self._LocationMatrix = np.zeros(self._ND, dtype=np.int_)
        
        # 积分点坐标和权重(2x2高斯积分)
        self._gauss_points = [
            (-1/np.sqrt(3), -1/np.sqrt(3)),
            (1/np.sqrt(3), -1/np.sqrt(3)),
            (1/np.sqrt(3), 1/np.sqrt(3)),
            (-1/np.sqrt(3), 1/np.sqrt(3))
        ]
        self._gauss_weights = [1.0, 1.0, 1.0, 1.0]

    def Read(self, input_file, Ele, MaterialSets, NodeList):
        line = input_file.readline().split()

        N = int(line[0])
        if N != Ele + 1:
            error_info = "\n*** Error *** Elements must be inputted in order !" \
                         "\n   Expected element : {}" \
                         "\n   Provided element : {}".format(Ele + 1, N)
            raise ValueError(error_info)

        # 读取节点编号
        N1 = int(line[1])
        N2 = int(line[2])
        N3 = int(line[3])
        N4 = int(line[4])
        MSet = int(line[5])
        
        self._ElementMaterial = MaterialSets[MSet - 1]
        self._nodes[0] = NodeList[N1 - 1]
        self._nodes[1] = NodeList[N2 - 1]
        self._nodes[2] = NodeList[N3 - 1]
        self._nodes[3] = NodeList[N4 - 1]

    def Write(self, output_file, Ele):
        element_info = "%5d%11d%9d%9d%9d%12d\n" % (
            Ele+1, 
            self._nodes[0].NodeNumber,
            self._nodes[1].NodeNumber,
            self._nodes[2].NodeNumber,
            self._nodes[3].NodeNumber,
            self._ElementMaterial.nset
        )
        print(element_info, end='')
        output_file.write(element_info)

    def GenerateLocationMatrix(self):
        i = 0
        for N in range(self._NEN):
            dof = self._nodes[N].NDF
            if dof == 3:
                begin = 0
            else:
                begin = 2
            for D in range(begin, begin+3):  # 板单元每个节点有3个自由度(w,θx,θy)
                self._LocationMatrix[i] = self._nodes[N].bcode[D]
                i += 1

    def SizeOfStiffnessMatrix(self):
        # 板单元刚度矩阵是12x12，上三角部分有78个元素
        return 78

    def _shape_functions(self, xi, eta):
        """ 计算形函数及其导数 """
        N = np.zeros(4)
        dN_dxi = np.zeros(4)
        dN_deta = np.zeros(4)
        
        # 形函数
        N[0] = 0.25 * (1 - xi) * (1 - eta)
        N[1] = 0.25 * (1 + xi) * (1 - eta)
        N[2] = 0.25 * (1 + xi) * (1 + eta)
        N[3] = 0.25 * (1 - xi) * (1 + eta)
        
        # 形函数对自然坐标的导数
        dN_dxi[0] = -0.25 * (1 - eta)
        dN_dxi[1] = 0.25 * (1 - eta)
        dN_dxi[2] = 0.25 * (1 + eta)
        dN_dxi[3] = -0.25 * (1 + eta)
        
        dN_deta[0] = -0.25 * (1 - xi)
        dN_deta[1] = -0.25 * (1 + xi)
        dN_deta[2] = 0.25 * (1 + xi)
        dN_deta[3] = 0.25 * (1 - xi)
        
        return N, dN_dxi, dN_deta

    def _jacobian(self, dN_dxi, dN_deta):
        """ 计算雅可比矩阵 """
        J = np.zeros((2, 2))
        for i in range(4):
            x = self._nodes[i].XYZ[0]
            y = self._nodes[i].XYZ[1]
            J[0, 0] += dN_dxi[i] * x
            J[0, 1] += dN_dxi[i] * y
            J[1, 0] += dN_deta[i] * x
            J[1, 1] += dN_deta[i] * y
        detJ = np.linalg.det(J)
        if detJ <= 0:
            raise ValueError("Negative Jacobian determinant detected")
        return J

    def _bending_strain_displacement_matrix(self, dN_dx, dN_dy):
        """ 计算弯曲应变-位移矩阵B_b """
        B_b = np.zeros((3, 12))
        for i in range(4):
            B_b[0, 3*i+1] = dN_dx[i]  # θx对x的导数
            B_b[1, 3*i+2] = -dN_dy[i] # θy对y的导数(负号)
            B_b[2, 3*i+1] = dN_dy[i]  # θx对y的导数
            B_b[2, 3*i+2] = -dN_dx[i] # θy对x的导数(负号)
        return B_b

    def _shear_strain_displacement_matrix(self, N, dN_dx, dN_dy):
        """ 计算剪切应变-位移矩阵B_s """
        B_s = np.zeros((2, 12))
        for i in range(4):
            B_s[0, 3*i] = dN_dx[i]    # w对x的导数
            B_s[0, 3*i+1] = N[i]      # θx
            B_s[1, 3*i] = dN_dy[i]     # w对y的导数
            B_s[1, 3*i+2] = -N[i]     # θy(负号)
        return B_s

    def _constitutive_matrix_bending(self):
        """ 计算弯曲本构矩阵D_b """
        material = self._ElementMaterial
        E = material.E
        nu = material.nu
        t = material.thickness
        
        D_b = (E * t**3) / (12 * (1 - nu**2)) * np.array([
            [1, nu, 0],
            [nu, 1, 0],
            [0, 0, (1-nu)/2]
        ])
        return D_b

    def _constitutive_matrix_shear(self):
        """ 计算剪切本构矩阵D_s """
        material = self._ElementMaterial
        E = material.E
        nu = material.nu
        t = material.thickness
        k = 5/6  # 剪切修正系数
        
        G = E / (2 * (1 + nu))
        D_s = k * G * t * np.eye(2)
        return D_s

    def ElementStiffness(self, stiffness):
        """ 计算单元刚度矩阵 """
        D_b = self._constitutive_matrix_bending()
        D_s = self._constitutive_matrix_shear()
        
        # 初始化刚度矩阵
        K = np.zeros((12, 12))
        
        # 高斯积分
        for gp in range(4):
            xi, eta = self._gauss_points[gp]
            weight = self._gauss_weights[gp]
            
            # 计算形函数及其导数
            N, dN_dxi, dN_deta = self._shape_functions(xi, eta)
            
            # 计算雅可比矩阵及其行列式
            J = self._jacobian(dN_dxi, dN_deta)
            detJ = np.linalg.det(J)
            
            # 计算形函数对x,y的导数
            invJ = np.linalg.inv(J)
            dN_dx = invJ[0,0]*dN_dxi + invJ[0,1]*dN_deta
            dN_dy = invJ[1,0]*dN_dxi + invJ[1,1]*dN_deta
            
            # 计算应变-位移矩阵
            B_b = self._bending_strain_displacement_matrix(dN_dx, dN_dy)
            B_s = self._shear_strain_displacement_matrix(N, dN_dx, dN_dy)
            
            # 计算单元刚度矩阵贡献
            K += np.dot(B_b.T, np.dot(D_b, B_b)) * detJ * weight  # 弯曲部分
            
            # 剪切部分
            # 减缩积分点（中心点）
            xi, eta = 0.0, 0.0
            weight = 4.0  # 减缩积分的权重总和应为4（2×2高斯积分总权重也是4）
    
            # 计算形函数及其导数
            N, dN_dxi, dN_deta = self._shape_functions(xi, eta)
    
            # 计算雅可比矩阵及其行列式
            J = self._jacobian(dN_dxi, dN_deta)
            detJ = np.linalg.det(J)
    
            # 计算形函数对x,y的导数
            invJ = np.linalg.inv(J)
            dN_dx = invJ[0,0]*dN_dxi + invJ[0,1]*dN_deta
            dN_dy = invJ[1,0]*dN_dxi + invJ[1,1]*dN_deta
    
            # 计算剪切应变-位移矩阵
            B_s = self._shear_strain_displacement_matrix(N, dN_dx, dN_dy)
    
            # 剪切部分刚度矩阵贡献
            K += np.dot(B_s.T, np.dot(D_s, B_s)) * detJ * weight

        # 将刚度矩阵转换为上三角存储格式
        index = 0
        for j in range(12):
            for i in range(j,-1,-1):

                
                stiffness[index] = K[i, j]
                index += 1

    def ElementStress(self, stress, displacement):
        """ 计算单元应力 """
        D_b = self._constitutive_matrix_bending()
        
        # 初始化应力
        stress_at_points = np.zeros((4, 3))  # 4个积分点，每个点3个弯矩分量
        
        # 高斯积分
        for gp in range(4):
            xi, eta = self._gauss_points[gp]
            
            # 计算形函数及其导数
            N, dN_dxi, dN_deta = self._shape_functions(xi, eta)
            
            # 计算雅可比矩阵及其行列式
            J = self._jacobian(dN_dxi, dN_deta)
            
            # 计算形函数对x,y的导数
            invJ = np.linalg.inv(J)
            dN_dx = invJ[0,0]*dN_dxi + invJ[0,1]*dN_deta
            dN_dy = invJ[1,0]*dN_dxi + invJ[1,1]*dN_deta
            
            # 计算弯曲应变-位移矩阵
            B_b = self._bending_strain_displacement_matrix(dN_dx, dN_dy)
            
            # 提取单元位移
            u = np.zeros(12)
            for i in range(12):
                if self._LocationMatrix[i]:
                    u[i] = displacement[self._LocationMatrix[i]-1]
            
            # 计算弯曲应变(曲率)
            curvature = np.dot(B_b, u)
            
            # 计算弯矩
            stress_at_points[gp] = np.dot(D_b, curvature)
        
        # 计算平均弯矩
        stress[0] = np.mean(stress_at_points[:, 0])  # Mx
        stress[1] = np.mean(stress_at_points[:, 1])  # My
        stress[2] = np.mean(stress_at_points[:, 2])  # Mxy

    