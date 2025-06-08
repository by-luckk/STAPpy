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
from solver.Solver import CSolver
import numpy as np
import sys
#GPU needed
# import cupy as cp  # 添加CuPy库支持
# import time


class CLDLTSolver(CSolver):
	"""
	LDLT solver: A in core solver using skyline storage
	and column reduction scheme
	"""
	def __init__(self, K, use_gpu=True):
		self.K = K			# Global Stiffness matrix in Skyline storage
		if use_gpu == True:
			import time
			import torch
		self.use_gpu = use_gpu 
		self.gpu_init_time = 0  # GPU初始化时间
        
	def _to_gpu(self,array):
		import cupy as cp
		import time
		start = time.time()
		gpu_array = cp.asarray(array)
		self.gpu_init_time += time.time() - start
		return gpu_array

	def _to_cpu(self, gpu_array):
		import cupy as cp
		import time
		"""将数据移回CPU"""
		return cp.asnumpy(gpu_array)

	def LDLT(self):
		""" LDLT分解，可选择GPU加速 """
		if self.use_gpu:
			self._gpu_LDLT()
		else:
			self._cpu_LDLT()
        
    

	def _cpu_LDLT(self):
		""" LDLT facterization (原有代码，使用CPU)"""
		N = self.K.dim()
		ColumnHeights = self.K.GetColumnHeights()

		if N > 1000:
			from tqdm import tqdm
			iteration = tqdm(range(2, N+1), desc="LDLT Decomposition", unit="column")
		else:
			iteration = range(2, N+1)
		for j in iteration: # Loop for column 2:n (Numbering starting from 1)
			# Row number of the first non-zero element in column j
			# (Numbering starting from 1)
			mj = j - ColumnHeights[j - 1]

			for i in range(mj+1, j): # Loop for mj+1:j-1 (Numbering starting from 1)
				# Row number of the first nonzero element in column i
				# (Numbering starting from 1)
				mi = i - ColumnHeights[i - 1]

				C = np.double(0.0)
				for r in range(max(mi, mj), i):
					# C += L_ri * U_rj
					C += (self.K[r, i]*self.K[r, j])

				self.K[i, j] -= C		# U_ij = K_ij - C

			for r in range(mj, j):		# Loop for mj:j-1 (column j)
				# L_rj = U_rj / D_rr
				Lrj = self.K[r, j]/self.K[r, r]
				# D_jj = K_jj - sum(L_rj*U_rj, r=mj:j-1)
				self.K[j, j] -= (Lrj*self.K[r, j])
				self.K[r, j] = Lrj

			if np.fabs(self.K[j, j] <= sys.float_info.min):
				error_info = "\n*** Error *** Stiffness matrix is not positive definite !" \
							 "\n    Euqation no = {}" \
							 "\n    Pivot = {}".format(j, self.K[j, j])
				raise ValueError(error_info)

	def _gpu_LDLT(self):
		""" GPU加速的LDLT分解 """
		print("using gpu...")
		import cupy as cp  # 添加CuPy库支持
		import time
		N = self.K.dim()
		ColumnHeights = self.K.GetColumnHeights()
        
        # 将必要数据转移到GPU
		K_data = self._to_gpu(self.K._data)
		col_heights = self._to_gpu(ColumnHeights)
        
        # GPU加速的LDLT分解
		for j in range(2, N+1): # 列2到n
			mj = j - col_heights[j - 1]
            
			for i in range(mj+1, j): # mj+1到j-1
				mi = i - col_heights[i - 1]
				r_start = int(max(mi, mj))  # 转为整数索引
				r_end = int(i)
                
                # 确保有元素可计算
				if r_end > r_start:
                    # 提取相关行
					K_i = K_data[r_start:r_end, i]
					K_j = K_data[r_start:r_end, j]
                    
                    # GPU点积计算
					C = cp.dot(K_i, K_j)
                    
                    # 原地更新矩阵元素
					K_data[i, j] -= C
            
			for r in range(int(mj), int(j)):
				Lrj = K_data[r, j] / K_data[r, r]
				K_data[j, j] -= Lrj * K_data[r, j]
				K_data[r, j] = Lrj
                
            # 检查是否正定
			if cp.abs(K_data[j, j]) <= sys.float_info.min:
                # 从GPU复制数据用于错误信息
				pivot = float(cp.asnumpy(K_data[j, j]))
				error_info = "\n*** Error *** Stiffness matrix is not positive definite !" \
                             "\n    Equation no = {}" \
                             "\n    Pivot = {}".format(j, pivot)
				raise ValueError(error_info)
        
        # 将更新后的数据移回CPU
		self.K._data = self._to_cpu(K_data)
		print(f"GPU LDLT completed. Data transfer time: {self.gpu_init_time:.4f}s")

	def BackSubstitution(self, Force):
		""" 后向替换求解位移，可选择GPU加速 """
		if self.use_gpu:
			return self._gpu_BackSubstitution(Force)
		else:
			return self._cpu_BackSubstitution(Force)
        
	def _cpu_BackSubstitution(self, Force):
		""" Solve displacement by back substitution (原始代码)"""
		N = self.K.dim()
		ColumnHeights = self.K.GetColumnHeights()

		# Reduce right-hand-side load vector (LV = R)
		for i in range(2, N+1): # Loop for i=2:N (Numering starting from 1)
			mi = i - ColumnHeights[i - 1]

			for j in range(mi, i): # Loop for j=mi:i-1
				# V_i = R_i - sum_j (L_ji V_j)
				Force[i - 1] -= (self.K[j, i]*Force[j - 1])

		# Back substitute (Vbar = D^(-1) V, L^T a = Vbar)
		for i in range(1, N+1): # Loop for i=1:N
			# Vbar = D^(-1) V
			Force[i - 1] /= self.K[i, i]

		for j in range(N, 1, -1): # Loop for j=N:2
			mj = j - ColumnHeights[j - 1]

			for i in range(mj, j): # Loop for i=mj:j-1
				# a_i = Vbar_i - sum_j(L_ij Vbar_j)
				Force[i - 1] -= (self.K[i, j]*Force[j - 1])
    
    
	def _gpu_BackSubstitution(self, Force):
		""" GPU加速的后向替换 """
		import cupy as cp  # 添加CuPy库支持
		import time
		N = self.K.dim()
		ColumnHeights = self.K.GetColumnHeights()
        
        # 将必要数据转移到GPU
		K_data = self._to_gpu(self.K._data)
		col_heights = self._to_gpu(ColumnHeights)
		gpu_force = self._to_gpu(Force)
        
        # 前向替换
		for i in range(2, N+1):
			mi = i - col_heights[i - 1]
			j_start = int(mi)
			j_end = int(i)
            
			if j_end > j_start:
                # GPU点积计算
				K_i = K_data[j_start:j_end, i]
				F_vals = gpu_force[j_start-1:j_end-1]
				gpu_force[i - 1] -= cp.dot(K_i, F_vals)
        
        # 对角线除法
		for i in range(1, N+1):
			gpu_force[i - 1] /= K_data[i, i]
        
        # 后向替换
		for j in range(N, 1, -1):
			mj = j - col_heights[j - 1]
			i_start = int(mj)
			i_end = int(j)
            
			if i_end > i_start:
                # GPU广播计算
				K_j = K_data[i_start:i_end, j]
				gpu_force[i_start-1:i_end-1] -= K_j * gpu_force[j - 1]
        
        # 将结果移回CPU
		Force[:] = self._to_cpu(gpu_force)
		print(f"GPU back substitution completed. Total GPU time: {self.gpu_init_time:.4f}s")
		return Force
