#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dask-parallelized version of the Haldane heterostructure analysis
"""

import numpy as np
from haldane_model import haldane_hamiltonian, haldane_positions
import scipy as sp
import scipy.sparse.linalg
import datetime
import os
from dask.distributed import Client, LocalCluster
import dask.array as da

def generate_system():
    """Generate system matrices (unchanged from original)"""
    H = haldane_hamiltonian(21, [0,0.3*np.sqrt(3),0.5*np.sqrt(3)], [1,1,1], [0.5,0,0], [np.pi/2,0,0], [0,0,0.2])
    X, Y = haldane_positions(21,1)
    return (X, Y, H)

def clifford_comp(kappa, x, y, E, X, Y, H):
    """Clifford composite operator (unchanged)"""
    N = len(X.toarray())
    clifford_comp_arr = np.zeros((2*N,2*N), dtype='complex')
    id_matrix_arr = np.identity(N, dtype='complex')
    
    x_diff = X - x*id_matrix_arr
    y_diff = Y - y*id_matrix_arr
    E_diff = H - E*id_matrix_arr
    
    clifford_comp_arr[0:N,0:N] = E_diff
    clifford_comp_arr[N:2*N,N:2*N] = -np.conj(E_diff).T
    clifford_comp_arr[0:N,N:2*N] = kappa*(x_diff - 1j*y_diff)
    clifford_comp_arr[N:2*N,0:N] = kappa*(x_diff + 1j*y_diff)
    
    return sp.sparse.lil_array(clifford_comp_arr)

def m_operator(kappa, x, y, E, X, Y, H):
    """M operator (unchanged)"""
    N = len(X.toarray())
    m_operator_arr = np.zeros((3*N,N), dtype='complex')
    id_matrix_arr = np.identity(N, dtype='complex')
    
    x_diff = X - x*id_matrix_arr
    y_diff = Y - y*id_matrix_arr
    E_diff = H - E*id_matrix_arr
    
    m_operator_arr[0:N,0:N] = kappa*kappa*x_diff
    m_operator_arr[N:2*N,0:N] = kappa*kappa*y_diff
    m_operator_arr[2*N:3*N,0:N] = E_diff
    
    return sp.sparse.lil_array(m_operator_arr)

def clifford_linear_gap(L):
    """Clifford linear gap calculation"""
    N_half = int(np.ceil(L.shape[0]/4))
    eig_val, gap_vector = sp.sparse.linalg.eigs(L, k=N_half, which='LR')
    gap = np.min(np.abs(np.real(eig_val)))
    gap_index = np.where(np.real(eig_val)==gap)[0][0]
    return gap, gap_vector[:,gap_index]

def quadratic_gap(M):
    """Quadratic gap calculation"""
    _, quad_gap, gap_vector = sp.sparse.linalg.svds(M, k=1, which='SM', solver='arpack')
    return quad_gap[0], np.conj(gap_vector[0])

def calculate_gaps(args):
    """Calculate gaps for a single point (adapted for Dask)"""
    i, j, m, kappa, X, Y, H, fixed_dim_input = args
    x_diag = np.diag(X.toarray())
    y_diag = np.diag(Y.toarray())
    x_points = np.linspace(min(x_diag)-2, max(x_diag)+2, m)
    y_points = np.linspace(min(y_diag)-2, max(y_diag)+2, m)
    x, y = x_points[i], y_points[j]
    reE_fixed, imE_fixed = fixed_dim_input[2], fixed_dim_input[3]

    # Quadratic gap
    M_RQ = m_operator(kappa, x, y, complex(reE_fixed, imE_fixed), X, Y, H)
    quad_r_gap, _ = quadratic_gap(M_RQ)
    M_LQ = m_operator(kappa, x, y, complex(reE_fixed, -imE_fixed), X, Y, 
                    sp.sparse.lil_array(sp.sparse.lil_array.conj(H).T))
    quad_l_gap, _ = quadratic_gap(M_LQ)
    quad_gap = min(quad_r_gap, quad_l_gap)

    # Linear gap
    L = clifford_comp(kappa, x, y, complex(reE_fixed, imE_fixed), X, Y, H)
    linear_gap, _ = clifford_linear_gap(L)

    return i, j, quad_gap, linear_gap, np.abs(linear_gap - quad_gap)

def parallel_gap_calculation(kappa, fixed_dim_input, X, Y, H, m=100):
    """Dask-parallelized computation"""
    # Create indices for all grid points
    indices = [(i, j, m, kappa, X, Y, H, fixed_dim_input) 
              for i in range(m) for j in range(m)]
    
    # Convert to Dask delayed computations
    lazy_results = []
    for idx in indices:
        lazy_results.append(da.from_delayed(calculate_gaps(idx), shape=(), dtype=object))
    
    # Compute in parallel
    results = da.compute(*lazy_results)
    
    # Reconstruct result matrices
    quad_data = np.zeros((m, m))
    linear_data = np.zeros((m, m))
    diff_data = np.zeros((m, m))
    
    for i, j, q, l, d in results:
        quad_data[i, j] = q
        linear_data[i, j] = l
        diff_data[i, j] = d
        
    return quad_data, linear_data, diff_data

if __name__ == '__main__':
    # Start Dask client (auto-detects SLURM if available)
    # For local testing: client = Client(n_workers=4, threads_per_worker=1)
    client = Client()
    print(f"Dask Dashboard: {client.dashboard_link}")
    
    folder_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(folder_date, exist_ok=True)
    
    X, Y, H = generate_system()
    kappa = 0.5
    E_values = [0, 1, complex(0,1)]
    
    for E in E_values:
        x_min = [kappa, 0, 0, E.real, E.imag]
        fixed_dim_input = x_min[1:5]
        quad_data, linear_data, diff_data = parallel_gap_calculation(
            kappa, fixed_dim_input, X, Y, H, m=150
        )
        filename_base = f"{folder_date}/k{kappa}_reE{E.real:0.2f}_imE{E.imag:0.2f}"
        
        np.savez(f"{filename_base}.npz",
                quad_gap_data=quad_data,
                linear_gap_data=linear_data,
                gap_diff_data=diff_data)
    
    client.close()