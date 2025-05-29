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
import abc


class CMaterial(metaclass=abc.ABCMeta):
	"""
		Material base class which only define one data member
		All type of material classes should be derived from this base class
	"""
	def __init__(self):
		self.nset = 0			# Number of set
		self.E = 0				# Young's modulus

	@abc.abstractmethod
	def Read(self, input_file, mset):
		pass

	@abc.abstractmethod
	def Write(self, output_file):
		pass


class CBarMaterial(CMaterial):
	""" Material class for bar element """
	def __init__(self):
		super().__init__()
		self.Area = 0			# Sectional area of a bar element

	def Read(self, input_file, mset):
		"""
		Read material data from stream Input
		"""
		line = input_file.readline().split()

		self.nset = np.int_(line[0])
		if self.nset != mset + 1:
			error_info = "\n*** Error *** Material sets must be inputted in order !" \
						 "\n   Expected set : {}" \
						 "\n   Provided set : {}".format(mset + 1, self.nset)
			raise ValueError(error_info)

		self.E = float(line[1])
		self.Area = float(line[2])

	def Write(self, output_file):
		"""
		Write material data to Stream
		"""
		material_info = "%5d%16.6e%16.6e\n"%(self.nset, self.E, self.Area)

		# print the material info on the screen
		print(material_info, end='')
		# write the material info to output file
		output_file.write(material_info)

class CQ4Material(CMaterial):
    ### Q4单元
    ### 增加
    def __init__(self):
        super().__init__()
        self.thickness = 0.0  # 单元厚度
        self.nu = 0.0         # 泊松比
        self.plane_stress = True  # 平面应力或平面应变

    def Read(self, input_file, mset):
        line = input_file.readline().split()

        self.nset = np.int_(line[0])
        if self.nset != mset + 1:
            error_info = "\n*** Error *** Material sets must be inputted in order !" \
                         "\n   Expected set : {}" \
                         "\n   Provided set : {}".format(mset + 1, self.nset)
            raise ValueError(error_info)

        self.E = float(line[1])
        self.nu = float(line[2])
        self.thickness = float(line[3])
        if len(line) > 4:
            self.plane_stress = bool(int(line[4]))

    def Write(self, output_file):
        material_info = "%5d%16.6e%16.6e%16.6e%5d\n" % (
            self.nset, self.E, self.nu, self.thickness, int(self.plane_stress))
        print(material_info, end='')
        output_file.write(material_info)

class CH8Material(CMaterial):
    """ Material class for H8 (3D solid) element """
    def __init__(self):
        super().__init__()
        # E and nu are inherited from CMaterial.
        # H8 elements typically only need E and nu for isotropic material definition.

    def Read(self, input_file, mset):
        """
        Read material data from stream Input for H8 element.
        Expected format: nset E nu
        """
        line = input_file.readline().split()

        self.nset = np.int_(line[0])
        if self.nset != mset + 1:
            error_info = (f"\n*** Error *** Material sets must be inputted in order !"
                          f"\n   Expected set : {mset + 1}"
                          f"\n   Provided set : {self.nset}")
            raise ValueError(error_info)

        self.E = float(line[1])
        self.nu = float(line[2]) # Poisson's ratio

    def Write(self, output_file):
        """
        Write H8 material data to Stream.
        """
        # Format: nset E nu
        material_info = "%5d%16.6e%16.6e\n" % (self.nset, self.E, self.nu)

        # print the material info on the screen
        print(material_info, end='')
        # write the material info to output file
        output_file.write(material_info)

class CT3Material(CMaterial):
    """Material definition for a 3‑node plane‑stress triangle."""

    def __init__(self):
        super().__init__()
        self.E         = 0.0     # Young's modulus
        self.nu        = 0.0     # Poisson's ratio
        self.thickness = 1.0     # Plate thickness (default 1)

    # ------------------------------------------------------------------
    #                          I / O ROUTINES
    # ------------------------------------------------------------------
    def Read(self, input_file, mset):
        """Read a single material card.

        Expected line format:
            mset   E   nu   thickness
        """
        line = input_file.readline().split()
        if not line:
            raise EOFError("Unexpected EOF while reading material data")

        self.nset = int(line[0])
        if self.nset != mset + 1:
            error_info = ("\n*** Error *** Material sets must be inputted in order !"
                          f"\n   Expected set : {mset + 1}"
                          f"\n   Provided set : {self.nset}")
            raise ValueError(error_info)

        try:
            self.E         = float(line[1])
            self.nu        = float(line[2])
            self.thickness = float(line[3]) if len(line) > 3 else 1.0
        except (ValueError, IndexError):
            raise ValueError("*** Error *** Incorrect CT3 material card format!")

    def Write(self, output_file):
        """Echo the material card (same format as input)."""
        material_info = f"{self.nset:5d}{self.E:16.6e}{self.nu:16.6e}{self.thickness:16.6e}\n"
        print(material_info, end="")           # console echo
        output_file.write(material_info)

class CPlateMaterial(CMaterial):
    """ Material class for plate element """
    def __init__(self):
        super().__init__()
        self.thickness = 0.0      # 板厚度
        self.nu = 0.0            # 泊松比
        self.density = 0.0        # 材料密度(可选)
        self.shear_correction = 5.0/6.0  # 剪切修正系数(默认5/6)

    def Read(self, input_file, mset):
        """
        Read plate material data from input file
        输入格式: nset E nu thickness [density] [shear_correction]
        """
        line = input_file.readline().split()

        self.nset = np.int_(line[0])
        if self.nset != mset + 1:
            error_info = "\n*** Error *** Material sets must be inputted in order !" \
                         "\n   Expected set : {}" \
                         "\n   Provided set : {}".format(mset + 1, self.nset)
            raise ValueError(error_info)

        # 读取必要参数
        self.E = float(line[1])
        self.nu = float(line[2])
        self.thickness = float(line[3])
        
        # 读取可选参数
        if len(line) > 4:
            self.density = float(line[4])
        if len(line) > 5:
            self.shear_correction = float(line[5])

    def Write(self, output_file):
        """
        Write plate material data to output file
        输出格式: nset E nu thickness density shear_correction
        """
        material_info = "%5d%16.6e%16.6e%16.6e%16.6e%16.6e\n" % (
            self.nset, 
            self.E, 
            self.nu, 
            self.thickness,
            self.density,
            self.shear_correction
        )
        
        # print the material info on the screen
        print(material_info, end='')
        # write the material info to output file
        output_file.write(material_info)

    def GetBendingRigidity(self):
        """ 计算板的弯曲刚度矩阵 """
        return (self.E * self.thickness**3) / (12 * (1 - self.nu**2))

    def GetShearModulus(self):
        """ 计算剪切模量 """
        return self.E / (2 * (1 + self.nu))