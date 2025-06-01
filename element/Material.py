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

		self.E = np.double(line[1])
		self.Area = np.double(line[2])

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

        self.E = np.double(line[1])
        self.nu = np.double(line[2])
        self.thickness = np.double(line[3])
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

        self.E = np.double(line[1])
        self.nu = np.double(line[2]) # Poisson's ratio

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

class CBeamMaterial(CMaterial):
    """ Material class for beam element with BOX section """
    def __init__(self):
        super().__init__()
        self.nu = 0.3           # Poisson's ratio (default 0.3)
        self.width = 0.0        # Box width (b)
        self.height = 0.0       # Box height (h)
        self.t1 = 0.0           # Top thickness
        self.t2 = 0.0           # Bottom thickness
        self.t3 = 0.0           # Left thickness
        self.t4 = 0.0           # Right thickness

    def Read(self, input_file, mset):
        """
        Read beam material data with BOX section parameters
        Format: nset E nu width height t1 t2 t3 t4
        """
        line = input_file.readline().split()

        self.nset = np.int_(line[0])
        if self.nset != mset + 1:
            error_info = "\n*** Error *** Material sets must be inputted in order !" \
                         "\n   Expected set : {}" \
                         "\n   Provided set : {}".format(mset + 1, self.nset)
            raise ValueError(error_info)

        self.E = np.double(line[1])
        self.nu = np.double(line[2])
        self.width = np.double(line[3])
        self.height = np.double(line[4])
        self.t1 = np.double(line[5])
        self.t2 = np.double(line[6])
        self.t3 = np.double(line[7])
        self.t4 = np.double(line[8])

    def Write(self, output_file):
        """
        Write beam material data to Stream
        """
        material_info = "%5d%16.6e%16.6e%16.6e%16.6e%16.6e%16.6e%16.6e%16.6e\n" % (
            self.nset, self.E, self.nu, 
            self.width, self.height,
            self.t1, self.t2, self.t3, self.t4
        )

        # print the material info on the screen
        print(material_info, end='')
        # write the material info to output file
        output_file.write(material_info)

    def calculate_section_properties(self):
        """ Calculate section properties for BOX section """
        b = self.width
        h = self.height
        t1 = self.t1
        t2 = self.t2
        t3 = self.t3
        t4 = self.t4

        # Area calculation
        A = b*h - (b-t3-t4)*(h-t1-t2)

        # Moment of inertia calculations
        Iyy = (b*h**3 - (b-t3-t4)*(h-t1-t2)**3)/12
        Izz = (h*b**3 - (h-t1-t2)*(b-t3-t4)**3)/12

        # Torsional constant approximation for thin-walled box
        t_avg = (t1 + t2 + t3 + t4)/4
        J = 2*(b-t_avg)*(h-t_avg)*t_avg

        return A, Iyy, Izz, J