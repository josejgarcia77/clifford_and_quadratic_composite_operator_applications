#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed Dask implementation with proper task scheduling
"""

import numpy as np
from haldane_model import haldane_hamiltonian, haldane_positions
import scipy as sp
import scipy.sparse.linalg
import datetime
import os
import dask
import dask.array as da
from dask.distributed import Client, LocalCluster

def calculate_gaps(args):
    """Calculate gaps for a single point (fixed)"""
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

    return (i, j, quad_gap, linear_gap, np.abs(linear_gap - quad_gap))

def parallel_gap_calculation(kappa, fixed_dim_input, X, Y, H, m=100):
    """Optimized Dask parallel computation"""
    indices = [(i, j, m, kappa, X, Y, H, fixed_dim_input) 
              for i in range(m) for j in range(m)]
    
    # Submit tasks efficiently
    futures = client.map(calculate_gaps, indices)
    
    # Process results as they complete
    quad_data = np.zeros((m, m))
    linear_data = np.zeros((m, m))
    diff_data = np.zeros((m, m))
    
    for future in dask.distributed.as_completed(futures):
        i, j, q, l, d = future.result()
        quad_data[i, j] = q
        linear_data[i, j] = l
        diff_data[i, j] = d
        
    return quad_data, linear_data, diff_data

if __name__ == '__main__':
    # Initialize Dask
    client = Client(n_workers=8, threads_per_worker=1)
    print(f"Dask Dashboard: {client.dashboard_link}")
    
    # Rest of your main code remains the same
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