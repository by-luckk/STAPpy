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
from element.Node import CBeamNode


class CBeam(CElement):
    """ B31 Beam Element class """
    def __init__(self):
        super().__init__()
        self._NEN = 2  # Each element has 2 nodes
        self._nodes = [None for _ in range(self._NEN)]

        # Each node has 6 DOFs (3 translations + 3 rotations)
        self._ND = 12  
        self._LocationMatrix = np.zeros(self._ND, dtype=np.int_)

    def Read(self, input_file, Ele, MaterialSets, NodeList):
        """
        Read element data from stream Input

        :param input_file: (_io.TextIOWrapper) the object of input file
        :param Ele: (int) check index
        :param MaterialSets: (list(CMaterial)) the material list in Domain
        :param NodeList: (list(CNode)) the node list in Domain
        :return: None
        """
        line = input_file.readline().split()

        N = int(line[0])
        if N != Ele + 1:
            error_info = "\n*** Error *** Elements must be inputted in order !" \
                         "\n   Expected element : {}" \
                         "\n   Provided element : {}".format(Ele + 1, N)
            raise ValueError(error_info)

        # left node number and right node number
        N1 = int(line[1]); N2 = int(line[2])
        MSet = int(line[3])
        self._ElementMaterial = MaterialSets[MSet - 1]
        self._nodes[0] = NodeList[N1 - 1]
        self._nodes[1] = NodeList[N2 - 1]

    def Write(self, output_file, Ele):
        """
        Write element data to stream

        :param output_file: (_io.TextIOWrapper) the object of output file
        :param Ele: the element number
        :return: None
        """
        material = self._ElementMaterial  
        element_info = "%5d%11d%9d%12d%8.2f%8.2f%8.2f%8.2f%8.2f%8.2f\n"%(
            Ele+1, 
            self._nodes[0].NodeNumber,
            self._nodes[1].NodeNumber,
            material.nset,
            material.width, material.height,
            material.t1, material.t2,
            material.t3, material.t4

        )

        # print the element info on the screen
        print(element_info, end='')
        # write the element info to output file
        output_file.write(element_info)

    def GenerateLocationMatrix(self):
        """
        Generate location matrix: the global equation number that
        corresponding to each DOF of the element
        """
        i = 0
        for N in range(self._NEN):
            for D in range(6):  # 6 DOFs per node (3 translations + 3 rotations)
                self._LocationMatrix[i] = self._nodes[N].bcode[D]
                i += 1

    def SizeOfStiffnessMatrix(self):
        """
        Return the size of the element stiffness matrix
        (stored as an array column by column)
        For 2 node B31 beam element, element stiffness is a 12x12 matrix,
        whose upper triangular part has 78 elements
        """
        return 78
    
    def ElementStiffness(self, stiffness):
        """
        Calculate element stiffness matrix for B31 beam element
        Upper triangular matrix, stored as an array column by column
        starting from the diagonal element
        """
        # Initialize stiffness matrix
        for i in range(self.SizeOfStiffnessMatrix()):
            stiffness[i] = 0.0

        # Get material properties
        E = self._ElementMaterial.E
        G = E / (2 * (1 + 0.3))  # Assume Poisson's ratio = 0.3

        # Calculate section properties
        A, Iyy, Izz, J = self._calculate_section_properties()

        # Calculate beam length and direction cosines
        DX = np.zeros(3)
        for i in range(3):
            DX[i] = self._nodes[1].XYZ[i] - self._nodes[0].XYZ[i]

        L = np.sqrt(DX[0]**2 + DX[1]**2 + DX[2]**2)
        L2 = L * L
        L3 = L2 * L

        # Direction cosines
        cx = DX[0]/L
        cy = DX[1]/L
        cz = DX[2]/L

        # Local stiffness matrix (12x12)
        Klocal = np.zeros((12, 12))

        # Axial stiffness
        EA_L = E*A/L
        Klocal[0, 0] = EA_L
        Klocal[6, 6] = EA_L
        Klocal[0, 6] = -EA_L
        Klocal[6, 0] = -EA_L

        # Torsional stiffness
        GJ_L = G*J/L
        Klocal[3, 3] = GJ_L
        Klocal[9, 9] = GJ_L
        Klocal[3, 9] = -GJ_L
        Klocal[9, 3] = -GJ_L

        # Bending stiffness about local y-axis
        factor_y = E*Iyy/L3
        Klocal[1, 1] = 12*factor_y
        Klocal[7, 7] = 12*factor_y
        Klocal[1, 7] = -12*factor_y
        Klocal[7, 1] = -12*factor_y

        Klocal[5, 5] = 4*L2*factor_y
        Klocal[11, 11] = 4*L2*factor_y
        Klocal[5, 11] = 2*L2*factor_y
        Klocal[11, 5] = 2*L2*factor_y

        Klocal[1, 5] = 6*L*factor_y
        Klocal[5, 1] = 6*L*factor_y
        Klocal[1, 11] = 6*L*factor_y
        Klocal[11, 1] = 6*L*factor_y

        Klocal[5, 7] = -6*L*factor_y
        Klocal[7, 5] = -6*L*factor_y
        Klocal[7, 11] = -6*L*factor_y
        Klocal[11, 7] = -6*L*factor_y

        # Bending stiffness about local z-axis
        factor_z = E*Izz/L3
        Klocal[2, 2] = 12*factor_z
        Klocal[8, 8] = 12*factor_z
        Klocal[2, 8] = -12*factor_z
        Klocal[8, 2] = -12*factor_z

        Klocal[4, 4] = 4*L2*factor_z
        Klocal[10, 10] = 4*L2*factor_z
        Klocal[4, 10] = 2*L2*factor_z
        Klocal[10, 4] = 2*L2*factor_z

        Klocal[2, 4] = -6*L*factor_z
        Klocal[4, 2] = -6*L*factor_z
        Klocal[2, 10] = -6*L*factor_z
        Klocal[10, 2] = -6*L*factor_z

        Klocal[4, 8] = 6*L*factor_z
        Klocal[8, 4] = 6*L*factor_z
        Klocal[8, 10] = 6*L*factor_z
        Klocal[10, 8] = 6*L*factor_z


# 计算梁长度和方向向量
        DX = np.array([
            self._nodes[1].XYZ[0] - self._nodes[0].XYZ[0],
            self._nodes[1].XYZ[1] - self._nodes[0].XYZ[1],
            self._nodes[1].XYZ[2] - self._nodes[0].XYZ[2]
        ])
        L = np.linalg.norm(DX)
        v1 = DX / L  # 局部x轴

# 参考向量（取全局Z轴）
        v_ref = np.array([0, 0, 1])
# 如果v1与v_ref几乎平行，则改用Y轴
        if np.abs(np.dot(v1, v_ref)) > 0.999:
            v_ref = np.array([0, 1, 0])

# 计算局部z轴
        v3 = np.cross(v1, v_ref)
        v3_norm = np.linalg.norm(v3)
        if v3_norm < 1e-10:
    # 如果叉积结果很小，说明v1和v_ref几乎平行，换一个参考向量
            v_ref = np.array([0, 1, 0])
            v3 = np.cross(v1, v_ref)
            v3_norm = np.linalg.norm(v3)
            if v3_norm < 1e-10:
                v_ref = np.array([1, 0, 0])
            v3 = np.cross(v1, v_ref)
            v3_norm = np.linalg.norm(v3)
            if v3_norm < 1e-10:
                raise ValueError("Cannot determine local coordinate system for beam element")
        v3 = v3 / v3_norm

# 计算局部y轴
        v2 = np.cross(v3, v1)
        v2 = v2 / np.linalg.norm(v2)

# 旋转矩阵R（3x3）
        R = np.array([v1, v2, v3])

# 构建12x12的转换矩阵T
        T = np.zeros((12, 12))
        for i in range(0, 12, 3):
            T[i:i+3, i:i+3] = R

        # Transform local stiffness to global coordinates
        Kglobal = np.dot(T.T, np.dot(Klocal, T))

        # Store upper triangular part in stiffness array
        index = 0
        for j in range(12):
            for i in range(j,-1,-1):
                stiffness[index] = Kglobal[i, j]
                index += 1

def ElementStress(self, stress, displacement):
    """
    Calculate element stress for B31 beam element
    stress[0] - Axial stress (σ_axial)
    stress[1] - Bending stress about y-axis (σ_bend_y)
    stress[2] - Bending stress about z-axis (σ_bend_z)
    stress[3] - Torsional shear stress (τ_torsion)
    stress[4] - Shear stress due to Vy (τ_shear_y)
    stress[5] - Shear stress due to Vz (τ_shear_z)
    """
    # Initialize stress array
    for i in range(6):
        stress[i] = 0.0

    # Get material properties
    E = self._ElementMaterial.E
    G = E / (2 * (1 + 0.3))  # Assume Poisson's ratio = 0.3

    # Calculate section properties
    b = self._section[0]  # width
    h = self._section[1]  # height
    t1 = self._section[2]  # top thickness
    t2 = self._section[3]  # bottom thickness
    t3 = self._section[4]  # left thickness
    t4 = self._section[5]  # right thickness
    
    # Area calculation
    A = b*h - (b-t3-t4)*(h-t1-t2)
    
    # Moment of inertia calculations
    Iyy = (b*h**3 - (b-t3-t4)*(h-t1-t2)**3)/12  # about y-axis
    Izz = (h*b**3 - (h-t1-t2)*(b-t3-t4)**3)/12  # about z-axis
    
    # Torsional constant approximation for thin-walled box
    t_avg = (t1 + t2 + t3 + t4)/4
    J = 2*(b-t_avg)*(h-t_avg)*t_avg
    
    # First moment of area for shear stress calculation
    Qy = (b*h**2)/8 - ((b-t3-t4)*(h-t1-t2)**2)/8  # for shear in y-direction
    Qz = (h*b**2)/8 - ((h-t1-t2)*(b-t3-t4)**2)/8  # for shear in z-direction

    # Calculate beam length and direction cosines
        DX = np.zeros(3)
        for i in range(3):
            DX[i] = self._nodes[1].XYZ[i] - self._nodes[0].XYZ[i]

    L = np.sqrt(DX[0]**2 + DX[1]**2 + DX[2]**2)
    cx = DX[0]/L
    cy = DX[1]/L
    cz = DX[2]/L

    # Extract element displacements
        u = np.zeros(12)
        for i in range(12):
            if self._LocationMatrix[i]:
                u[i] = displacement[self._LocationMatrix[i]-1]

    # Transformation matrix from global to local coordinates
    if abs(cx) > 0.9999:
        # When beam is along x-axis
        R = np.array([[cx, 0, 0],
                      [0, cy, cz],
                      [0, -cz, cy]])
    else:
        # General case
        l = cx
        m = cy
        n = cz
        l2 = l*l
        m2 = m*m
        n2 = n*n
        lm = l*m
        ln = l*n
        mn = m*n
        
        R = np.array([[l, m, n],
                      [-m/np.sqrt(l2 + m2), l/np.sqrt(l2 + m2), 0],
                      [-l*n/np.sqrt(l2 + m2), -m*n/np.sqrt(l2 + m2), np.sqrt(l2 + m2)]])

    # Transform displacements to local coordinates
        u_local = np.zeros(12)
        u_local[0:3] = np.dot(R, u[0:3])
        u_local[3:6] = np.dot(R, u[3:6])
        u_local[6:9] = np.dot(R, u[6:9])
        u_local[9:12] = np.dot(R, u[9:12])

    # 1. Axial stress (σ = E * ε = E * (du/dx))
        axial_strain = (u_local[6] - u_local[0])/L
        stress[0] = E * axial_strain

    # 2. Bending stresses (σ = M*y/I)
    # Calculate moments in local coordinates
    # M_y = (6*E*Iyy/L^2) * (u_local[2] + u_local[8]) - (6*E*Iyy/L^2) * (u_local[5] + u_local[11])
    # M_z = (6*E*Izz/L^2) * (u_local[1] + u_local[7]) - (6*E*Izz/L^2) * (u_local[4] + u_local[10])
    
    # More accurate moment calculation using beam stiffness relations
    M_y1 = (2*E*Iyy/L) * (2*u_local[5] + u_local[11] - (6/L)*(u_local[2] - u_local[8]))
    M_y2 = (2*E*Iyy/L) * (u_local[5] + 2*u_local[11] - (6/L)*(u_local[2] - u_local[8]))
    
    M_z1 = (2*E*Izz/L) * (2*u_local[4] + u_local[10] + (6/L)*(u_local[1] - u_local[7]))
    M_z2 = (2*E*Izz/L) * (u_local[4] + 2*u_local[10] + (6/L)*(u_local[1] - u_local[7]))
    
    # Use average moment for stress calculation
    M_y = (M_y1 + M_y2)/2
    M_z = (M_z1 + M_z2)/2
    
    # Maximum bending stresses occur at extreme fibers (y=±h/2, z=±b/2)
        stress[1] = M_y * (h/2) / Iyy  # Bending stress about y-axis
        stress[2] = M_z * (b/2) / Izz  # Bending stress about z-axis

    # 3. Torsional shear stress (τ = T*r/J)
    # Calculate torque in local coordinates
        T = (G*J/L) * (u_local[9] - u_local[3])
    # Maximum torsional shear stress occurs at outer wall
        stress[3] = T * max(b/2, h/2) / J

    # 4. Shear stresses (τ = VQ/It)
    # Calculate shear forces in local coordinates
        V_y = (12*E*Iyy/L**3) * (u_local[2] - u_local[8]) + (6*E*Iyy/L**2) * (u_local[5] + u_local[11])
        V_z = (12*E*Izz/L**3) * (u_local[1] - u_local[7]) - (6*E*Izz/L**2) * (u_local[4] + u_local[10])
    
    # Shear stress due to Vy (acting in y-direction)
        stress[4] = V_y * Qy / (Iyy * (t3 + t4))  # Average shear stress in web
    
    # Shear stress due to Vz (acting in z-direction)
        stress[5] = V_z * Qz / (Izz * (t1 + t2))  # Average shear stress in flange

