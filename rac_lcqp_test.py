import sys
sys.path.append('/Users/mingxi/Desktop/python_rac')
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


def rac_lcqp(H, c, A, b, block_num, beta, tol, max_iter):
    """
    RAC ADMM
    Solver for following Quadratic Optimization Problem
    min 1/2*x^T H x + c^T x
    s.t.  A x = b
    Input (H,c,A,b,block_num, dual penalty parameter beta,tolerence of ||Ax-b||_inf, maximum number of iteration)
    Written for python3
    Written by Mingxi Zhu, Graduate School of Business,
    Stanford University, Stanford, CA 94305.
    Febrary, 2019.
    mingxiz@stanford.edu
    """

    rac_time_start = time.time()
    rac_prepare_time_start = time.time()

    # initializing x_k, y_k
    dim = A.shape[1]
    num_cons = A.shape[0]
    x_k = np.zeros((dim, 1))
    y_k = np.zeros((num_cons, 1))
    Ax = A*x_k
    Hx = H*x_k
    At = np.transpose(A)
    c_and_bA_lin = c - beta*At*b;
    
    block_size = math.floor(dim/block_num)
    num_iter = 0
    tol_temp = tol + 1

    # initiliaze time count
    rac_prepare_right_side_time = 0
    rac_sub_matrix_prepare_time = 0
    rac_sub_matrix_prepare_time_part1_select = 0
    rac_sub_matrix_prepare_time_part2_mul = 0
    rac_solver_time = 0
    rac_update_Hx_Ax_time = 0
    rac_update_Hx_Ax_time_part1_select = 0
    rac_update_Hx_Ax_time_part2_mul = 0
    rac_update_dual_time = 0
    
    # count rac prepare time
    rac_prepare_time = time.time() - rac_prepare_time_start
    #print(rac_prepare_time)

    while tol_temp > tol :

        # update dual vector
        # count time
        rac_prepare_right_side_time_start = time.time()
        c_dual_vector_1 = - At*y_k
        c_res = c_and_bA_lin + c_dual_vector_1
        x_index_perm = np.random.permutation(dim)
        rac_prepare_right_side_time += (time.time() - rac_prepare_right_side_time_start)
        #print(rac_prepare_right_side_time_start)

        for x_visited_block in range(block_num):
            
            # count time
            rac_sub_matrix_prepare_time_start = time.time()
            # select update index
            if x_visited_block == block_num - 1:
                x_update_index = x_index_perm[x_visited_block*block_size:dim]
            else:
                x_update_index = x_index_perm[x_visited_block*block_size : (x_visited_block+1)*block_size]

            # select sub matrix based on update matrix
            rac_sub_matrix_prepare_time_part1_select_start = time.time()
            A_sub = A[:, x_update_index]
            H_sub = H[x_update_index[:, None], x_update_index] ;
            rac_sub_matrix_prepare_time_part1_select += (time.time() - rac_sub_matrix_prepare_time_part1_select_start)
            
            rac_sub_matrix_prepare_time_part2_mul_start = time.time()
            A_sub_t = np.transpose(A_sub)
            A_sub_t_A_sub = A_sub_t*A_sub;
            H_current = H_sub + beta*A_sub_t_A_sub
            rac_sub_matrix_prepare_time_part2_mul += (time.time() - rac_sub_matrix_prepare_time_part2_mul_start)

            c_current_sub = Hx[x_update_index] + beta*A_sub_t*Ax - H_current*x_k[x_update_index]
            c_current = c_current_sub + c_res[x_update_index];
            right_side = - c_current
            left_side = H_current
            rac_sub_matrix_prepare_time += (time.time() - rac_sub_matrix_prepare_time_start)
            # print(rac_sub_matrix_prepare_time)

            # solve lienar system
            # count time
            rac_solver_time_start = time.time()
            # result_x = spsolve(left_side, right_side)
            # result_x = result_x.reshape(-1, 1)

            solve = factorized(left_side)
            result_x = solve(right_side)
            result_x = result_x.reshape(-1, 1)
            rac_solver_time += (time.time() - rac_solver_time_start)
            #print(rac_solver_time)
            
            # update Hx and Ax
            # count time
            rac_update_Hx_Ax_time_start = time.time()
            diff_x = x_k[x_update_index] - result_x
            
            rac_update_Hx_Ax_time_part1_select_start = time.time()
            Hx_sub = H[:, x_update_index]
            rac_update_Hx_Ax_time_part1_select += (time.time() - rac_update_Hx_Ax_time_part1_select_start)

            rac_update_Hx_Ax_time_part2_mul_start = time.time()
            Hx = Hx - Hx_sub*diff_x
            Ax = Ax - A_sub*diff_x
            rac_update_Hx_Ax_time_part2_mul += (time.time() - rac_update_Hx_Ax_time_part2_mul_start)

            x_k[x_update_index] = result_x
            rac_update_Hx_Ax_time += (time.time() - rac_update_Hx_Ax_time_start)
            #print(rac_update_Hx_Ax_time)


        # update dual
        # count time
        rac_update_dual_time_start = time.time()
        res_k = Ax - b
        y_k = y_k - beta*res_k
        tol_temp = max(abs(res_k))
        rac_update_dual_time += (time.time() - rac_update_dual_time_start)
        #print(rac_update_dual_time)
        
        num_iter += 1
        print(num_iter)
        if num_iter == max_iter:
            break


    # report time
    rac_time = time.time()-rac_time_start
    return (rac_time, tol_temp, num_iter, rac_prepare_time, rac_prepare_right_side_time, rac_sub_matrix_prepare_time, rac_sub_matrix_prepare_time_part1_select, rac_sub_matrix_prepare_time_part2_mul, rac_solver_time, rac_update_Hx_Ax_time, rac_update_Hx_Ax_time_part1_select, rac_update_Hx_Ax_time_part2_mul, rac_update_dual_time)


def test_problem():
    problem_parameters = loadmat('wide_test.mat')
    H = problem_parameters['Q']
    H_s = csc_matrix(H)
    c = problem_parameters['c']
    A = problem_parameters['A']
    A_s = csc_matrix(A)
    b = problem_parameters['b']
    
    block_num = 200
    beta = 1
    tol = 1e-4
    max_iter = 10
    
    return rac_lcqp(H_s, c, A_s, b, block_num, beta, tol, max_iter)


(rac_time, tol_temp, num_iter, rac_prepare_time, rac_prepare_right_side_time, rac_sub_matrix_prepare_time, rac_sub_matrix_prepare_time_part1_select, rac_sub_matrix_prepare_time_part2_mul, rac_solver_time, rac_update_Hx_Ax_time, rac_update_Hx_Ax_time_part1_select, rac_update_Hx_Ax_time_part2_mul, rac_update_dual_time) = test_problem()
print("rac time %f" % (rac_time))
print("rac_prepare_time %f" % (rac_prepare_time))
print("rac_prepare_right_side_time %f" % (rac_prepare_right_side_time))
print("rac_sub_matrix_prepare_time %f" % (rac_sub_matrix_prepare_time))
print("rac_sub_matrix_prepare_time_part1_select %f" % (rac_sub_matrix_prepare_time_part1_select))
print("rac_sub_matrix_prepare_time_part2_mul %f" % (rac_sub_matrix_prepare_time_part2_mul))
print("rac_solver_time %f" % (rac_solver_time))
print("rac_update_Hx_Ax_time %f" % (rac_update_Hx_Ax_time))
print("rac_update_Hx_Ax_time_part1_select %f" % (rac_update_Hx_Ax_time_part1_select))
print("rac_update_Hx_Ax_time_part2_mul %f" % (rac_update_Hx_Ax_time_part2_mul))
print("rac_update_dual_time %f" % (rac_update_dual_time))
