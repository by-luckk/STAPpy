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
import time
import torch

class CLDLTSolver(CSolver):
	"""
	LDLT solver: A in core solver using skyline storage
	and column reduction scheme
	"""
	def __init__(self, K, use_gpu=True):
		self.K = K			# Global Stiffness matrix in Skyline storage
		self.use_gpu = use_gpu and torch.cuda.is_available()
		self.device = torch.device("cuda" if self.use_gpu else "cpu")
		self._factorized = False      # 是否已完成分解
		self._LD = None               # (N,N) torch.Tensor, 存储 L 与 D
		self._pivots = None 
        
	def _to_dense_tensor(self):
		"""将 Skyline → dense → torch.Tensor (float64)"""
		if hasattr(self.K, "to_dense_matrix"):
			dense_np = self.K.to_dense_matrix()          # 用户需实现
		else:
			dense_np = self._skyline_to_dense_fallback() # 备用实现
		return torch.as_tensor(dense_np,
								dtype=torch.float64,
								device=self.device)

	def _skyline_to_dense_fallback(self):
		"""若未实现 to_dense_matrix，可利用 [] 运算符补一版慢速转换"""
		n = self.K.dim()
		dense = np.zeros((n, n), dtype=np.float64)
		H = self.K.GetColumnHeights()
		for i in range(1, n + 1):
			for j in range(1, i + 1):
				if i - j <= H[i-1]:                # 只有存过时才取
					val = self.K[j, i]
				else:
					val = 0.0
				dense[i-1, j-1] = val
				dense[j-1, i-1] = val
		return dense
	
	def LDLT(self):
		"""
		对 K 做对称 LDLᵀ (Bunch-Kaufman) 分解。
		调用后 self._LD 与 self._pivots 持有结果。
		"""
		if self._factorized:
			return

		t0 = time.time()
		A = self._to_dense_tensor()            # (N,N) float64, on self.device
		t_transfer = time.time() - t0

		# torch.linalg.ldl_factor_ex 返回 (LD, pivots, info)
		ld, piv, info = torch.linalg.ldl_factor_ex(
			A, hermitian=True, check_errors=False
		)
		if info.item() != 0:
			# info>0 ⇒ 紧凑选主元失败，可能非正定；info<0 ⇒ 非法参数
			raise ValueError(
				f"LDLᵀ factorization failed, info={info.item()}"
			)

		self._LD = ld
		self._pivots = piv
		self._factorized = True

		t_total = time.time() - t0
		if self.use_gpu:
			print(f"[PyTorch-GPU] LDLᵀ factorization done "
					f"(transfer {t_transfer:.3f}s, total {t_total:.3f}s)")
		else:
			print(f"[PyTorch-CPU] LDLᵀ factorization done "
					f"(total {t_total:.3f}s)")

	# ------------------------------------------------------------------ #
	def BackSubstitution(self, Force):
		if not self._factorized:
			self.LDLT()

		b = torch.as_tensor(Force,
							dtype=self._LD.dtype,
							device=self.device)

		if b.dim() == 1:                      # → (n, 1)
			b = b.unsqueeze(-1)

		x = torch.linalg.ldl_solve(self._LD, self._pivots,
								b, hermitian=True)

		if x.shape[-1] == 1:                  # → (n,)
			x = x.squeeze(-1)

		return x.cpu().numpy()

	# ------------------------------------------------------------------ #
	# 可选：一次性完成分解 + 求解（接口兼容旧 Solver）
	# ------------------------------------------------------------------ #
	def Solve(self, Force):
		"""
		兼容 CSolver 的常用“先分解再求解”调用模式。
		相当于：
			self.LDLT()
			return self.BackSubstitution(Force)
		"""
		self.LDLT()
		return self.BackSubstitution(Force)

