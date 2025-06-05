#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
/*****************************************************************************/
/*  STAPpy : A python FEM code sharing the same input data file with STAP90  */
/*     Computational Dynamics Laboratory                                     */
/*     School of Aerospace Engineering, Tsinghua University                  */
/*                                                                           */
/*     Created on Mon Jun 22, 2020                                           */
/*                                                                           */
/*     @author: thurcni@163.com, xzhang@tsinghua.edu.cn                      */
/*     http://www.comdyn.cn/                                                 */
/*****************************************************************************/
"""
import sys
sys.path.append('../')
import numpy as np
from element.Element import CElement

class CQ4(CElement):
    """ Q4 Element class """
    def __init__(self):
        super().__init__()
        self._NEN = 4  #4个节点
        self._nodes = [None for _ in range(self._NEN)]
        
        # 每个节点2个自由度(u,v)，共8个自由度
        self._ND = 8  
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

        # # 读取节点编号 错误
        # N1 = int(line[1])
        # N2 = int(line[2])
        # N3 = int(line[3])
        # N4 = int(line[4])
        # MSet = int(line[5])
        
        # self._ElementMaterial = MaterialSets[MSet - 1]
        # self._nodes[0] = NodeList[N1 - 1]
        # self._nodes[1] = NodeList[N2 - 1]
        # self._nodes[2] = NodeList[N3 - 1]
        # self._nodes[3] = NodeList[N4 - 1]
        
        # 读取节点坐标 (4个节点，每个节点有x,y两个坐标)
        coordinates = []
        index = 1
        for _ in range(4):
            # 读取每个节点的x,y坐标
            try:
                x = float(line[index])
                y = float(line[index + 1])
            except ValueError:
                raise ValueError(f"*** Error *** Invalid coordinate format for Q4 element {N}")
            coordinates.append((x, y))
            index += 2
        
        # 读取材料集
        try:
            MSet = int(line[index])
        except ValueError:
            raise ValueError(f"*** Error *** Invalid material set for Q4 element {N}")
        
        # 存储节点坐标和材料
        self._ElementMaterial = MaterialSets[MSet - 1]
        for i, (x, y) in enumerate(coordinates):
            self._nodes[i].coordinate2D = np.array([x, y], dtype=float)

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
            for D in range(2):  # Q4单元每个节点有2个自由度(u,v)
                self._LocationMatrix[i] = self._nodes[N].bcode[D]
                i += 1

    def SizeOfStiffnessMatrix(self):
        # Q4单元刚度矩阵是8x8，上三角部分有36个元素
        DOF = 0
        for i in range(self._ND):
            DOF = DOF + i + 1
        return DOF

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
        return J

    def _strain_displacement_matrix(self, dN_dx, dN_dy):
        """ 计算应变-位移矩阵B """
        B = np.zeros((3, 8))
        for i in range(4):
            B[0, 2*i] = dN_dx[i]
            B[1, 2*i+1] = dN_dy[i]
            B[2, 2*i] = dN_dy[i]
            B[2, 2*i+1] = dN_dx[i]
        return B

    def _constitutive_matrix(self):
        """ 计算本构矩阵D """
        material = self._ElementMaterial
        E = material.E
        nu = material.nu
        
        if material.plane_stress:
            # 平面应力
            factor = E / (1 - nu**2)
            D = np.array([
                [1, nu, 0],
                [nu, 1, 0],
                [0, 0, (1-nu)/2]
            ]) * factor
        else:
            # 平面应变
            factor = E / ((1 + nu) * (1 - 2 * nu))
            D = np.array([
                [1 - nu, nu, 0],
                [nu, 1 - nu, 0],
                [0, 0, (1 - 2 * nu)/2]
            ]) * factor
        
        return D

    def ElementStiffness(self, stiffness):
        """ 计算单元刚度矩阵 """
        material = self._ElementMaterial
        D = self._constitutive_matrix()
        t = material.thickness
        
        # 初始化刚度矩阵
        K = np.zeros((8, 8))
        
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
            
            # 计算应变-位移矩阵B
            B = self._strain_displacement_matrix(dN_dx, dN_dy)
            
            # 计算单元刚度矩阵贡献
            K += np.dot(B.T, np.dot(D, B)) * detJ * weight * t
        
        # 将刚度矩阵转换为上三角存储格式 # 已修改
        index = 0
        for j in range(8):
            for i in range(j,-1,-1):
                stiffness[index] = K[i, j]
                index += 1

    def ElementStress(self, stress, displacement):
        """ 计算单元应力 """
        material = self._ElementMaterial
        D = self._constitutive_matrix()
        
        # 初始化应力
        stress_at_points = np.zeros((4, 3))  # 4个积分点，每个点3个应力分量
        
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
            
            # 计算应变-位移矩阵B
            B = self._strain_displacement_matrix(dN_dx, dN_dy)
            
            # 提取单元位移
            u = np.zeros(8)
            for i in range(8):
                if self._LocationMatrix[i]:
                    u[i] = displacement[self._LocationMatrix[i]-1]
            
            # 计算应变
            strain = np.dot(B, u)
            
            # 计算应力
            stress_at_points[gp] = np.dot(D, strain)
        
        # 计算平均应力
        stress[0] = np.mean(stress_at_points[:, 0])  # σx
        stress[1] = np.mean(stress_at_points[:, 1])  # σy
        stress[2] = np.mean(stress_at_points[:, 2])  # τxy