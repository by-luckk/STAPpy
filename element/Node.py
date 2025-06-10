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
import numpy as np

class CNode(object):
    # 基础类，包含所有节点共有的属性和方法
    NDF = 3

    def __init__(self, dof_count=3):
        # 设置自由度数量
        self.NDF = dof_count
        
        # x, y 和 z 坐标
        self.XYZ = np.zeros(3)
        
        # 边界条件代码数组
        self.bcode = np.zeros(self.NDF, dtype=np.int_)
        
        # 节点编号
        self.NodeNumber = 0

    def Read(self, input_file, check_np):
        """读取节点数据，支持不同自由度的节点"""
        line = input_file.readline().split()
        
        num_bc = len(line) - 4  # 总字段数减去节点号和三坐标
        if num_bc == 3:
            self.NDF = 3
        elif num_bc == 5:
            self.NDF = 5
        elif num_bc == 6:
            self.NDF = 6
        else:
            error_info = f"\n*** Error *** Invalid number of boundary conditions: {num_bc}"
            raise ValueError(error_info)
        
        # 重新初始化边界条件数组
        self.bcode = np.zeros(self.NDF, dtype=np.int_)
        
        N = int(line[0])
        if N != check_np + 1:
            error_info = "\n*** Error *** Nodes must be inputted in order !" \
                         f"\n   Expected node number : {check_np+1}" \
                         f"\n   Provided node number : {N}"
            raise ValueError(error_info)

        self.NodeNumber = N

        # 读取边界条件
        for i in range(self.NDF):
            self.bcode[i] = np.int_(line[i+1])
            
        # 读取坐标值
        self.XYZ[0] = np.double(line[1+self.NDF])
        self.XYZ[1] = np.double(line[2+self.NDF])
        self.XYZ[2] = np.double(line[3+self.NDF])

    def Write(self, output_file):
        """输出节点数据"""
        node_info = f"{self.NodeNumber:9d}"
        
        # 输出边界条件代码
        for bc in self.bcode:
            node_info += f"{bc:5d}"
            
        # 输出坐标
        node_info += f"{self.XYZ[0]:18.6e}{self.XYZ[1]:15.6e}{self.XYZ[2]:15.6e}\n"
            
        # 打印并写入文件
        print(node_info, end='')
        output_file.write(node_info)

    def WriteEquationNo(self, output_file):
        """输出节点方程编号"""
        equation_info = f"{self.NodeNumber:9d}       "

        for dof in range(self.NDF):
            equation_info += f"{self.bcode[dof]:5d}"

        equation_info += '\n'
        print(equation_info, end='')
        output_file.write(equation_info)

    def WriteNodalDisplacement(self, output_file, displacement):
        """输出节点位移"""
        displacement_info = f"{self.NodeNumber:5d}        "
        displacement_list = []
        
        for dof in range(self.NDF):
            if self.bcode[dof] == 0:
                displacement_info += f"{0.0:25.17e}"
                displacement_list.append(0.0)
            else:
                displacement_info += f"{displacement[self.bcode[dof] - 1]:25.17e}"
                displacement_list.append(displacement[self.bcode[dof] - 1])

        displacement_info += '\n'
        print(displacement_info, end='')
        output_file.write(displacement_info)

        return displacement_list

# class CStandardNode(CNode):
#     """标准节点类，用于杆单元等，3个自由度"""
#     def __init__(self, x=0.0, y=0.0, z=0.0):
#         super().__init__(dof_count=3)
#         self.XYZ[0] = x
#         self.XYZ[1] = y
#         self.XYZ[2] = z


# class CBeamNode(CNode):
#     """梁节点，具有6个自由度(3平移+3旋转)"""
#     def __init__(self):
#         super().__init__(dof_count=6)

# class CPlateNode(CNode):
#     def __init__(self):
#         super().__init__(dof_count=5)