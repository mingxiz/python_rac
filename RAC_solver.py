import sys
sys.path.append('/Users/mingxi/Desktop/python_rac/python_rac')
import numpy as np
import math
import time
from scipy import linalg
from scipy.sparse import csc_matrix
from scipy.sparse import coo_matrix
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
from scipy import sparse
from scipy.io import loadmat
from scipy.sparse.linalg import factorized


class RAC_solver():

# build a general solver on RAC-ADMM

# Possible solver mode:
# is_sparse: Dense Sparse [0 1]
# is_par: whether Parallel computing [0 1]
# block_num
# beta
# tolerance
# max_iter

# Solvers available:

# RAC_solver.lcqp
# RAC_solver.lcqp_ineq
# RAC_solver.binary
# RAC_solver.mip


    def __init__(self, is_sparse, is_par, block_num, beta, tol, max_iter):
    # initialize solver withh correct block_num, stepsize, tolerance and max_iter
        self.is_sparse = is_sparse
        self.is_par = is_par
        self.block_num = block_num
        self.tol = tol
        self.max_iter = max_iter

    def lcqp(self, H, c, A, b):
    # problem index for lcqp : H, c, A, b
    # min_x 1/2*x'*H*x + c'*x
    # s.t. A*x = b

    if self.is_sparse == 1 and self.is_par == 0:
        return lcqp_sparse(self, H, c, A, b)
    if self.is_sparse == 1 and self.is_par == 1:
        return lcqp_sparse_par(self, H, c, A, b)
    if self.is_sparse == 0 and self.is_par == 0:
        return lcqp_dense(self, H, c, A, b)
    if self.is_sparse == 0 and self.is_par == 1:
        return lcqp_dense_par(self, H, c, A, b)
    
    
    
