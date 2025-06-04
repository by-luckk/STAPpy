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
from utils.Singleton import Singleton
from element.ElementGroup import ElementTypes
import datetime
import numpy as np
from utils.Plot import PlotDisp

weekday = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", 
           "Saturday", "Sunday"]
month = ["January", "February", "March", "April", "May", "June",
		 "July", "August", "September", "October", "November", "December"]


@Singleton
class COutputter(object):
	""" Singleton: Outputer class is used to output results """
	def __init__(self, filename=""):
		try:
			self._output_file = open(filename, 'w')
		except FileNotFoundError as e:
			print(e)
			sys.exit(3)

	def GetOutputFile(self):
		return self._output_file

	def PrintTime(self):
		""" Output current time and date """
		t = datetime.datetime.now()
		time_info = t.strftime("        (%H:%M:%S on ")
		time_info += (month[t.month - 1] + " ")
		time_info += (str(t.day) + ", ")
		time_info += (str(t.year) + ", ")
		time_info += (weekday[t.weekday()] + ")\n\n")

		print(time_info, end="")
		self._output_file.write(time_info)

	def OutputHeading(self):
		""" Print program logo """
		from Domain import Domain
		FEMData = Domain()

		title_info = "TITLE : " + FEMData.GetTitle() + "\n"
		print(title_info, end="")
		self._output_file.write(title_info)

		self.PrintTime()

	def OutputNodeInfo(self):
		""" Print nodal data """
		from Domain import Domain
		FEMData = Domain()

		NodeList = FEMData.GetNodeList()

		pre_info = "C O N T R O L   I N F O R M A T I O N\n\n"
		print(pre_info, end="")
		self._output_file.write(pre_info)

		NUMNP = FEMData.GetNUMNP()
		NUMEG = FEMData.GetNUMEG()
		NLCASE = FEMData.GetNLCASE()
		MODEX = FEMData.GetMODEX()

		pre_info = "\t  NUMBER OF NODAL POINTS . . . . . . . . . . (NUMNP)  =%6d\n" \
				   "\t  NUMBER OF ELEMENT GROUPS . . . . . . . . . (NUMEG)  =%6d\n" \
				   "\t  NUMBER OF LOAD CASES . . . . . . . . . . . (NLCASE) =%6d\n" \
				   "\t  SOLUTION MODE  . . . . . . . . . . . . . . (MODEX)  =%6d\n" \
				   "\t\t EQ.0, DATA CHECK\n" \
				   "\t\t EQ.1, EXECUTION\n\n"%(NUMNP, NUMEG, NLCASE, MODEX)
		print(pre_info, end="")
		self._output_file.write(pre_info)

		pre_info = " N O D A L   P O I N T   D A T A\n\n" \
				   "    NODE       BOUNDARY                         NODAL POINT\n" \
				   "   NUMBER  CONDITION  CODES                     COORDINATES\n"
		print(pre_info, end="")
		self._output_file.write(pre_info)

		for n in range(NUMNP):
			NodeList[n].Write(self._output_file)

		print("\n", end="")
		self._output_file.write("\n")

	def OutputEquationNumber(self):
		""" Output equation numbers """
		from Domain import Domain
		FEMData = Domain()

		NodeList = FEMData.GetNodeList()

		NUMNP = FEMData.GetNUMNP()

		pre_info = " EQUATION NUMBERS\n\n" \
				   "   NODE NUMBER   DEGREES OF FREEDOM\n" \
				   "        N           X    Y    Z\n"
		print(pre_info, end="")
		self._output_file.write(pre_info)

		for n in range(NUMNP):
			NodeList[n].WriteEquationNo(self._output_file)

		print("\n", end="")
		self._output_file.write("\n")

	def OutputElementInfo(self):
		""" Output element data """
		# Print element group control line
		from Domain import Domain
		FEMData = Domain()

		NUMEG = FEMData.GetNUMEG()

		pre_info = " E L E M E N T   G R O U P   D A T A\n\n\n"
		print(pre_info, end="")
		self._output_file.write(pre_info)

		for EleGrp in range(NUMEG):
			ElementType = FEMData.GetEleGrpList()[EleGrp].GetElementType()
			NUME = FEMData.GetEleGrpList()[EleGrp].GetNUME()

			pre_info = " E L E M E N T   D E F I N I T I O N\n\n" \
					   " ELEMENT TYPE  . . . . . . . . . . . . .( NPAR(1) ) . . =%5d\n" \
					   "     EQ.1, TRUSS ELEMENTS\n" \
					   "     EQ.2, ELEMENTS CURRENTLY\n" \
					   "     EQ.3, NOT AVAILABLE\n\n" \
					   " NUMBER OF ELEMENTS. . . . . . . . . . .( NPAR(2) ) . . =%5d\n\n" \
					   %(ElementType, NUME)
			print(pre_info, end="")
			self._output_file.write(pre_info)

			element_type = ElementTypes.get(ElementType)
			if element_type == 'Bar':
				self.PrintBarElementData(EleGrp)
			elif element_type == 'Q4':
				self.PrintQ4ElementData(EleGrp)
			elif element_type == 'T3':
				self.PrintT3ElementData(EleGrp)
			elif element_type == 'H8':
				self.PrintH8ElementData(EleGrp)
			elif element_type == 'Beam':
				self.PrintBeamElementData(EleGrp)
				# pass  # comment or delete this line after implementation
    
			else:
				# error_info = "\n*** Error *** Elment type {} has not been " \
				# 			 "implemented.\n\n".format(ElementType)
				# raise ValueError(error_info)
				pass
				# error_info = "\n*** Error *** Elment type {} has not been " \
				# 			 "implemented.\n\n".format(ElementType)
				# raise ValueError(error_info)
				pass

	def PrintBarElementData(self, EleGrp):
		""" Output bar element data """
		from Domain import Domain
		FEMData = Domain()

		ElementGroup = FEMData.GetEleGrpList()[EleGrp]
		NUMMAT = ElementGroup.GetNUMMAT()

		pre_info = " M A T E R I A L   D E F I N I T I O N\n\n" \
				   " NUMBER OF DIFFERENT SETS OF MATERIAL\n" \
				   " AND CROSS-SECTIONAL  CONSTANTS  . . . .( NPAR(3) ) . . =%5d\n\n" \
				   "  SET       YOUNG'S     CROSS-SECTIONAL\n" \
				   " NUMBER     MODULUS          AREA\n" \
				   "               E              A\n"%NUMMAT
		print(pre_info, end="")
		self._output_file.write(pre_info)

		for mset in range(NUMMAT):
			ElementGroup.GetMaterial(mset).Write(self._output_file)

		pre_info = "\n\n E L E M E N T   I N F O R M A T I O N\n" \
				   " ELEMENT     NODE     NODE       MATERIAL\n" \
				   " NUMBER-N      I        J       SET NUMBER\n"
		print(pre_info, end="")
		self._output_file.write(pre_info)

		NUME = ElementGroup.GetNUME()
		for Ele in range(NUME):
			ElementGroup[Ele].Write(self._output_file, Ele)

		print("\n", end="")
		self._output_file.write("\n")

	def PrintT3ElementData(self, EleGrp):
		"""
		Output T3 (CST) element data in a style parallel to PrintBarElementData.
		"""
		from Domain import Domain
		FEMData = Domain()
		ElementGroup = FEMData.GetEleGrpList()[EleGrp]
		NUMMAT = ElementGroup.GetNUMMAT()

		pre_info = (
			" M A T E R I A L   D E F I N I T I O N\n\n"
			" NUMBER OF DIFFERENT SETS OF MATERIAL\n"
			" AND PLATE  CONSTANTS  . . . .( NPAR(3) ) . . =%5d\n\n"
			"  SET       YOUNG'S      POISSON     THICKNESS\n"
			" NUMBER     MODULUS        RATIO          t\n" % NUMMAT
		)
		print(pre_info, end="")
		self._output_file.write(pre_info)

		for mset in range(NUMMAT):
			ElementGroup.GetMaterial(mset).Write(self._output_file)
		pre_info = (
			"\n\n E L E M E N T   I N F O R M A T I O N\n"
			" ELEMENT     NODE     NODE     NODE     MATERIAL\n"
			" NUMBER-N      I        J        K    SET NUMBER\n"
		)
		print(pre_info, end="")
		self._output_file.write(pre_info)

		NUME = ElementGroup.GetNUME()
		for Ele in range(NUME):
			ElementGroup[Ele].Write(self._output_file, Ele)

		print("\n", end="")
		self._output_file.write("\n")
  
	## Q4单元
	def PrintQ4ElementData(self, EleGrp):
		""" Output Q4 element data """
		from Domain import Domain
		FEMData = Domain()

		ElementGroup = FEMData.GetEleGrpList()[EleGrp]
		NUMMAT = ElementGroup.GetNUMMAT()

		pre_info = " M A T E R I A L   D E F I N I T I O N\n\n" \
				" NUMBER OF DIFFERENT SETS OF MATERIAL\n" \
				" AND CROSS-SECTIONAL  CONSTANTS  . . . .( NPAR(3) ) . . =%5d\n\n" \
				"  SET       YOUNG'S     POISSON'S     THICKNESS    PLANE\n" \
				" NUMBER     MODULUS       RATIO                     STRESS\n" \
				"               E            nu            t            (1/0)\n" % NUMMAT
		print(pre_info, end='')
		self._output_file.write(pre_info)

		for mset in range(NUMMAT):
			ElementGroup.GetMaterial(mset).Write(self._output_file)

		pre_info = "\n\n E L E M E N T   I N F O R M A T I O N\n" \
				" ELEMENT     NODE     NODE     NODE     NODE       MATERIAL\n" \
				" NUMBER-N      I        J        K        L       SET NUMBER\n"
		print(pre_info, end='')
		self._output_file.write(pre_info)

		NUME = ElementGroup.GetNUME()
		for Ele in range(NUME):
			ElementGroup[Ele].Write(self._output_file, Ele)

		print("\n", end='')
		self._output_file.write("\n")
	def PrintH8ElementData(self, EleGrp): # NEW METHOD FOR H8
		""" Output H8 element data """
		from Domain import Domain
		FEMData = Domain()

		ElementGroup = FEMData.GetEleGrpList()[EleGrp]
		NUMMAT = ElementGroup.GetNUMMAT()

		pre_info = " M A T E R I A L   D E F I N I T I O N\n\n" \
				" NUMBER OF DIFFERENT SETS OF MATERIAL  . . . .( NPAR(3) ) . . =%5d\n\n" \
				"  SET       YOUNG'S     POISSON'S\n" \
				" NUMBER     MODULUS       RATIO\n" \
				"               E            nu\n" % NUMMAT
		print(pre_info, end='')
		self._output_file.write(pre_info)

		for mset in range(NUMMAT):
			ElementGroup.GetMaterial(mset).Write(self._output_file)

		pre_info = "\n\n E L E M E N T   I N F O R M A T I O N\n" \
				" ELEMENT     N1       N2       N3       N4       N5       N6       N7       N8       MATERIAL\n" \
				" NUMBER-N                                                                          SET NUMBER\n" # Adjusted for 8 nodes
		print(pre_info, end='')
		self._output_file.write(pre_info)

		NUME = ElementGroup.GetNUME()
		for Ele in range(NUME):
			ElementGroup[Ele].Write(self._output_file, Ele) # Assumes CH8.Write handles 8 nodes

		print("\n", end='')
		self._output_file.write("\n")

	def PrintBeamElementData(self, EleGrp):
		""" Output beam element data """
		from Domain import Domain
		FEMData = Domain()
		ElementGroup = FEMData.GetEleGrpList()[EleGrp]
		NUMMAT = ElementGroup.GetNUMMAT()

		pre_info = (
            " M A T E R I A L   D E F I N I T I O N\n\n"
            " NUMBER OF DIFFERENT SETS OF MATERIAL\n"
            " AND SECTION PROPERTIES . . . .( NPAR(3) ) . . =%5d\n\n"
            "  SET       YOUNG'S      SHEAR        AREA        Iyy         Izz         J\n"
            " NUMBER     MODULUS      MODULUS                  (m^4)       (m^4)       (m^4)\n"
            "               E           G            A\n" % NUMMAT
        )
		print(pre_info, end="")
		self._output_file.write(pre_info)

		for mset in range(NUMMAT):
			ElementGroup.GetMaterial(mset).Write(self._output_file)

		pre_info = (
            "\n\n E L E M E N T   I N F O R M A T I O N\n"
            " ELEMENT     NODE     NODE     MATERIAL   ORIENTATION NODE\n"
            " NUMBER-N      I        J       SET NUMBER   (OPTIONAL)\n"
        )
		print(pre_info, end="")
		self._output_file.write(pre_info)

		NUME = ElementGroup.GetNUME()
		for Ele in range(NUME):
			ElementGroup[Ele].Write(self._output_file, Ele)

		print("\n", end="")
		self._output_file.write("\n")

	
	# 其他单元Print
	def OutputLoadInfo(self):
		""" Print load data """
		from Domain import Domain
		FEMData = Domain()

		for lcase in range(FEMData.GetNLCASE()):
			LoadData = FEMData.GetLoadCases()[lcase]

			pre_info = " L O A D   C A S E   D A T A\n\n" \
					   "     LOAD CASE NUMBER . . . . . . . =%6d\n" \
					   "     NUMBER OF CONCENTRATED LOADS . =%6d\n\n" \
					   "    NODE       DIRECTION      LOAD\n" \
					   "   NUMBER                   MAGNITUDE\n"%(lcase + 1,
																  LoadData.nloads)
			print(pre_info, end="")
			self._output_file.write(pre_info)

			LoadData.Write(self._output_file, lcase+1)

			print("\n", end="")
			self._output_file.write("\n")

	def OutputNodalDisplacement(self, lcase, vis_scale=1):
		""" Print nodal displacement *并生成位移可视化图* """
		from Domain import Domain
		FEMData = Domain()
		NodeList = FEMData.GetNodeList()
		displacement = FEMData.GetDisplacement()

		pre_info = " LOAD CASE%5d\n\n\n" \
				   " D I S P L A C E M E N T S\n\n" \
				   "  NODE           X-DISPLACEMENT    Y-DISPLACEMENT    Z-DISPLACEMENT\n" \
				   %(lcase+1)
		print(pre_info, end="")
		self._output_file.write(pre_info)

		# -------- 收集数据 --------
		node_ids, disp = [], []

		for n in range(FEMData.GetNUMNP()):
			displacement_list = NodeList[n].WriteNodalDisplacement(self._output_file, displacement)

			node_ids.append(n + 1)
			disp.append(displacement_list)    

		print("\n", end="")
		self._output_file.write("\n")

		# -------- 绘图并保存 --------
		Coords = np.array([[node.XYZ[0], node.XYZ[1], node.XYZ[2]] for node in NodeList])
		PlotDisp(Coords, disp, scale=vis_scale, out_dir="output")

	def OutputElementStress(self):
		from Domain import Domain
		FEMData = Domain()

		displacement = FEMData.GetDisplacement()

		NUMEG = FEMData.GetNUMEG()

		for ELeGrpIndex in range(NUMEG):
			pre_info = " S T R E S S  C A L C U L A T I O N S  F O R  E L E M E N T  G R O U P%5d\n\n" \
					   %(ELeGrpIndex+1)
			print(pre_info, end="")
			self._output_file.write(pre_info)

			EleGrp = FEMData.GetEleGrpList()[ELeGrpIndex]
			NUME = EleGrp.GetNUME()
			ElementType = EleGrp.GetElementType()

			element_type = ElementTypes.get(ElementType)
			if element_type == 'Bar':
				pre_info = "  ELEMENT             FORCE            STRESS\n" \
						   "  NUMBER\n"
				print(pre_info, end="")
				self._output_file.write(pre_info)

				stress = np.zeros(1)

				for Ele in range(NUME):
					Element = EleGrp[Ele]
					Element.ElementStress(stress, displacement)

					material = Element.GetElementMaterial()
					stress_info = "%5d%22.6e%18.6e\n"%(Ele+1, stress[0]*material.Area, stress[0])
					print(stress_info, end="")
					self._output_file.write(stress_info)
			elif element_type == 'T3':
				pass
			elif element_type == 'Q4':
				# implementation for other element types by yourself
				# ...
				pass  # comment or delete this line after implementation
			elif element_type == 'H8':
				pre_info = ("  ELEMENT      SIGMA_XX      SIGMA_YY      SIGMA_ZZ        TAU_XY        TAU_YZ        TAU_ZX\n"
							"  NUMBER                  (Stresses at element center: xi=0, eta=0, zeta=0)\n")
				print(pre_info, end="")
				self._output_file.write(pre_info)

				stress = np.zeros(6) # H8 element has 6 stress components [s_xx, s_yy, s_zz, t_xy, t_yz, t_zx]

				for Ele in range(NUME):
					Element = EleGrp[Ele]
					Element.ElementStress(stress, displacement) # This calls CH8.ElementStress

					stress_info = "%5d%14.6e%14.6e%14.6e%14.6e%14.6e%14.6e\n"%(
						Ele+1, stress[0], stress[1], stress[2], stress[3], stress[4], stress[5])
					print(stress_info, end="")
					self._output_file.write(stress_info)
			elif element_type == 'Beam':
            # 保持原始输出标签不变
				pre_info = (
                "  ELEMENT      AXIAL       SHEAR-Y    SHEAR-Z    TORSION     MOMENT-Y    MOMENT-Z\n"
                "  NUMBER       FORCE       FORCE      FORCE      MOMENT      MOMENT      MOMENT\n"
                "               (N)         (N)        (N)        (N-m)       (N-m)       (N-m)\n"
            	)
				print(pre_info, end="")
				self._output_file.write(pre_info)

            # 创建临时数组存储应力值
				stress = np.zeros(6)

				for Ele in range(NUME):
					Element = EleGrp[Ele]
                # 调用修正后的应力计算方法
					Element.ElementStress(stress, displacement)
                
                # 将应力值转换为内力值（保持原始输出格式）
					material = Element.GetElementMaterial()
					b = material.width
					h = material.height
					t1 = material.t1
					t2 = material.t2
					t3 = material.t3
					t4 = material.t4
                
                # 计算截面属性（与Beam.py中一致）
					A = b*h - (b-t3-t4)*(h-t1-t2)
					Iyy = (b*h**3 - (b-t3-t4)*(h-t1-t2)**3)/12
					Izz = (h*b**3 - (h-t1-t2)*(b-t3-t4)**3)/12
                
                # 将应力转换为内力（保持原始输出格式）
					axial_force = stress[0] * A
					shear_y_force = stress[4] * A  # 近似处理
					shear_z_force = stress[5] * A  # 近似处理
					torsion_moment = stress[3] * (Iyy + Izz)  # 近似处理
					moment_y = stress[1] * Iyy / (h/2)
					moment_z = stress[2] * Izz / (b/2)
                
					force_info = "%5d%12.4e%12.4e%12.4e%12.4e%12.4e%12.4e\n" % (
					Ele+1, 
                    axial_force,
                    shear_y_force,
                    shear_z_force,
                    torsion_moment,
                    moment_y,
                    moment_z
                	)
					print(force_info, end="")
					self._output_file.write(force_info)
			else:
				# error_info = "\n*** Error *** Elment type {} has not been " \
				# 			 "implemented.\n\n".format(ElementType)
				# raise ValueError(error_info)
				pass

	def OutputTotalSystemData(self):
		""" Print total system data """
		from Domain import Domain
		FEMData = Domain()

		pre_info = "	TOTAL SYSTEM DATA\n\n" \
				   "     NUMBER OF EQUATIONS . . . . . . . . . . . . . .(NEQ) = {}\n" \
				   "     NUMBER OF MATRIX ELEMENTS . . . . . . . . . . .(NWK) = {}\n" \
				   "     MAXIMUM HALF BANDWIDTH  . . . . . . . . . . . .(MK ) = {}\n" \
				   "     MEAN HALF BANDWIDTH . . . . . . . . . . . . . .(MM ) = {}\n\n\n".format(
			FEMData.GetNEQ(), FEMData.GetStiffnessMatrix().size(),
			FEMData.GetStiffnessMatrix().GetMaximumHalfBandwidth(),
			FEMData.GetStiffnessMatrix().size()/FEMData.GetNEQ()
		)
		print(pre_info, end="")
		self._output_file.write(pre_info)

	def OutputSolutionTime(self, time_info):
		""" Print CPU time used for solution """
		print(time_info, end="")
		self._output_file.write(time_info)
