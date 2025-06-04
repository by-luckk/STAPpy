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

class CBeamNode(object):
    """梁单元节点类，具有6个自由度(3平动+3转动)"""
    # 覆盖父类的NDF，设置为6
    NDF = 6

    def __init__(self, x=0.0, y=0.0, z=0.0):
        super().__init__(x, y, z)
        # 扩展数组大小为6
        self.XYZ = np.zeros(CBeamNode.NDF)
        self.XYZ[0] = x; self.XYZ[1] = y; self.XYZ[2] = z
        
        # 边界条件代码数组扩展为6
        self.bcode = np.zeros(CBeamNode.NDF, dtype=np.int_)
        
        # 节点编号
        self.NodeNumber = 0

    def Read(self, input_file, check_np):
        """
        读取梁节点数据
        输入格式: NodeNumber BC1 BC2 BC3 BC4 BC5 BC6 X Y Z
        BC1-BC3: 平动自由度边界条件
        BC4-BC6: 转动自由度边界条件
        """
        line = input_file.readline().split()

        N = int(line[0])
        if N != check_np + 1:
            error_info = "\n*** Error *** Nodes must be inputted in order !" \
                         "\n   Expected node number : {}" \
                         "\n   Provided node number : {}".format(check_np+1, N)
            raise ValueError(error_info)

        self.NodeNumber = N

        # 读取6个自由度的边界条件
        for i in range(CBeamNode.NDF):
            self.bcode[i] = np.int_(line[i+1])
            
        # 读取坐标值
        self.XYZ[0] = np.double(line[7])
        self.XYZ[1] = np.double(line[8])
        self.XYZ[2] = np.double(line[9])

    def Write(self, output_file):
        """
        输出梁节点数据
        """
        node_info = "%9d" % self.NodeNumber
        
        # 输出6个边界条件代码
        for bc in self.bcode:
            node_info += "%5d" % bc
            
        # 输出坐标
        node_info += "%18.6e%15.6e%15.6e\n" % (
            self.XYZ[0], self.XYZ[1], self.XYZ[2])
            
        # 打印并写入文件
        print(node_info, end='')
        output_file.write(node_info)

    def WriteEquationNo(self, output_file):
        """
        输出梁节点方程编号
        """
        equation_info = "%9d       " % self.NodeNumber

        for dof in range(CBeamNode.NDF):
            equation_info += "%5d" % self.bcode[dof]

        equation_info += '\n'
        print(equation_info, end='')
        output_file.write(equation_info)

    def WriteNodalDisplacement(self, output_file, displacement):
        """
        输出梁节点位移(包含平动和转动)
        """
        displacement_info = "%5d        " % self.NodeNumber
        displacement_list = []
        
        for dof in range(CBeamNode.NDF):
            if self.bcode[dof] == 0:
                displacement_info += "%18.6e" % 0.0
                displacement_list.append(0.0)
            else:
                displacement_info += "%18.6e" % displacement[self.bcode[dof] - 1]
                displacement_list.append(displacement[self.bcode[dof] - 1])

        displacement_info += '\n'
        print(displacement_info, end='')
        output_file.write(displacement_info)

        return displacement_list