#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 09:58:29 2023

@author: josejgarcia
"""

import scipy.sparse as sp
import numpy as np

def clifford_representations(d):
    """
    Generate a Clifford representation defined in the paper by Garcia, Cerjan,
    
    Loring (2024). In particular the elements in the Clifford group satisfy
    
    the following relations: $\Gamma_i^\dagger = \Gamma_i$, $\Gamma_i^2 = I$,
    
    $\Gamma_i\Gamma_k = -\Gamma_k\Gamma_i$ for $i\not=k$.
    

    Parameters
    ----------
    d : integer
        Positive integer at least 2 representing the number of non-trivial elements in
        the Clifford group.

    Returns
    -------
    clifford_matrices : list
        A list of d nontrivial Clifford representation matrices each being in
        the scipy.sparse.lil_array format.
        
    References
    ----------
    Garcia, J. J., Cerjan, A., & Loring, T. A. (2024). Clifford and quadratic composite operators with applications to non-Hermitian physics. 
    arXiv:2410.03880. https://arxiv.org/abs/2410.03880
    """
    
    #Generate pauli matrices sig_x, sig_y, and sig_z
    sig_x = sp.lil_array((2,2), dtype=complex)
    sig_x[0,1] = 1
    sig_x[1,0] = 1
    
    sig_y = sp.lil_array((2,2), dtype=complex)
    sig_y[0,1] = complex(0,-1)
    sig_y[1,0] = complex(0,1)
    
    sig_z = sp.lil_array((2,2), dtype=complex)
    sig_z[0,0] = 1
    sig_z[1,1] = -1
    
    
    if d == 2:
        
        return [sig_x, sig_y]
    
    elif d == 3:
        
        # Call d=2 case, append sig_z and return it
        return [sig_x, sig_y, sig_z]
    
    elif d % 2 == 0:
        # Generate the d-1 case and tensor it with sig_x
        clifford_matrices = clifford_representations(d-1)
        # compute dimension of d-1 matrices
        prev_matrix_dim = int(np.power(2,np.floor((d-1)/2)))
        
        for index, gamma in enumerate(clifford_matrices):
            clifford_matrices[index] = sp.kron(sig_x,gamma)
            
        clifford_matrices.append(\
            sp.kron(sig_y,sp.identity(prev_matrix_dim, dtype=complex))\
            )
            
        return clifford_matrices
            
    else:
        # Generate the d-1 case and tensor it with sig_x
        clifford_matrices = clifford_representations(d-1)
        # compute dimension of d-2 matrices
        prev_matrix_dim = int(np.power(2,np.floor((d-2)/2)))
        
        clifford_matrices.append(\
            sp.kron(sig_z,sp.identity(prev_matrix_dim, dtype=complex))\
            )
        
        return clifford_matrices  
    