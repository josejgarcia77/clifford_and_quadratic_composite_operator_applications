#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 11:58:09 2023

@author: josejgarcia
"""

from scipy.sparse import lil_array
import numpy as np
import matplotlib.pyplot as plt
                
def haldane_graphs(N_1):
    """
    Generate a graph representation of the Haldane Heterostructure used in
    
    both papers by Garcia, Cerjan, Loring (2024) and Cerjan, Koekenbier,
    
    Schulz-Baldes (2024).
    

    Parameters
    ----------
    N_1 : integer
        Longest number of hexagons from left to right. This tells us there
        will be $(N_1+1)^2$ cells

    Returns
    -------
    haldane_nn_graph : scipy.sparse.lil_array
        A graph representation of the nearest neighbor interactions.
    haldane_nnn_graphA : scipy.sparse.lil_array
        A graph representation of the next nearest neighbor interations
        between the A cells.
    haldane_nnn_graphB : scipy.sparse.lil_array
        A graph representation of the next nearest neibhbor interactions
        between the B cells.
        
    References
    ----------
    Garcia, J. J., Cerjan, A., & Loring, T. A. (2024). Clifford and quadratic composite operators with applications to non-Hermitian physics. 
    arXiv:2410.03880. https://arxiv.org/abs/2410.03880
    
    Cerjan, A., Koekenbier, L., & Schulz-Baldes, H. (2023). Spectral localizer for line-gapped non-Hermitian systems. 
    *Journal of Mathematical Physics*, *64*(8), 082102. https://doi.org/10.1063/5.0150995
    """
    # Use N_1 and N_2 odd.
    # N_1 is the longest number of hexagons from left to right
    # N_2 is the number of hexagon rows on the honeycomb lattice
    # This tells us that there will be N_1+1 * N_2+1 cells.
    N_2 = N_1
    haldane_nn_graph = {}
    haldane_nnn_graph_A = {}
    haldane_nnn_graph_B = {}
    
    # Construct nearest neighbor graph for model in dictionary format
    for n_2 in range(1,N_2 + 2):
        for n_1 in range(1,N_1 + 2):
            
            if n_2 == 1:
                if n_1 == 1:
                    haldane_nn_graph[(n_1,n_2,'A')] = \
                        [(n_1,n_2,'B'),(n_1,n_2 +1, 'B')]
                    
                    haldane_nn_graph[(n_1,n_2,'B')] = \
                        [(n_1,n_2,'A'),(n_1+1,n_2,'A')]
                        
                elif n_1 == N_2+1:
                    haldane_nn_graph[(n_1,n_2,'A')] = \
                        [(n_1 -1,n_2, 'B'),(n_1,n_2 +1, 'B')]
                        
                else:
                    haldane_nn_graph[(n_1,n_2,'A')] = \
                        [(n_1 -1,n_2, 'B'),(n_1,n_2,'B'),(n_1,n_2 +1, 'B')]
                    
                    haldane_nn_graph[(n_1,n_2,'B')] = \
                        [(n_1,n_2,'A'),(n_1+1,n_2,'A')]
            
            elif n_2 == N_2+1:
                if n_1 == 1:
                    haldane_nn_graph[(n_1,n_2,'B')] = \
                        [(n_1+1,n_2,'A'),(n_1, n_2 -1, 'A')]    
                    
                elif n_1 == N_2+1:
                    haldane_nn_graph[(n_1,n_2,'A')] = \
                        [(n_1 -1,n_2, 'B'),(n_1,n_2,'B')]
                    
                    haldane_nn_graph[(n_1,n_2,'B')] = \
                        [(n_1,n_2,'A'),(n_1 , n_2 -1, 'A')]    
                    
                else:
                    haldane_nn_graph[(n_1,n_2,'A')] = \
                        [(n_1 -1,n_2, 'B'),(n_1,n_2,'B')]
                    
                    haldane_nn_graph[(n_1,n_2,'B')] = \
                        [(n_1,n_2,'A'),(n_1+1,n_2,'A'),(n_1, n_2 -1, 'A')]    
                
            elif n_2 % 2 == 0:
                if n_1 == 1:
                    haldane_nn_graph[(n_1,n_2,'B')] = \
                        [(n_1+1,n_2,'A'),(n_1, n_2 -1, 'A')]
                        
                elif n_1 == N_2+1:
                    haldane_nn_graph[(n_1,n_2,'A')] = \
                        [(n_1 -1,n_2, 'B'),(n_1,n_2,'B'),(n_1-1,n_2 +1, 'B')]
                        
                    haldane_nn_graph[(n_1,n_2,'B')] = \
                        [(n_1,n_2,'A'),(n_1, n_2 -1, 'A')]
                
                else:
                    haldane_nn_graph[(n_1,n_2,'A')] = \
                        [(n_1 -1,n_2, 'B'),(n_1,n_2,'B'),(n_1-1,n_2 +1, 'B')]
                        
                    haldane_nn_graph[(n_1,n_2,'B')] = \
                        [(n_1,n_2,'A'),(n_1+1,n_2,'A'),(n_1, n_2 -1, 'A')]
                        
            else:
                if n_1 == 1:
                    haldane_nn_graph[(n_1,n_2,'A')] = \
                        [(n_1,n_2,'B'),(n_1,n_2 +1, 'B')]
                        
                    haldane_nn_graph[(n_1,n_2,'B')] = \
                        [(n_1,n_2,'A'),(n_1+1,n_2,'A'),(n_1 +1, n_2 -1, 'A')]
                        
                elif n_1 == N_2+1:
                    haldane_nn_graph[(n_1,n_2,'A')] = \
                        [(n_1 -1,n_2, 'B'),(n_1,n_2 +1, 'B')]
                        
                else:
                    haldane_nn_graph[(n_1,n_2,'A')] = \
                        [(n_1 -1,n_2, 'B'),(n_1,n_2,'B'),(n_1,n_2 +1, 'B')]
                        
                    haldane_nn_graph[(n_1,n_2,'B')] = \
                        [(n_1,n_2,'A'),(n_1+1,n_2,'A'),(n_1 +1, n_2 -1, 'A')]
    
    # construct next neareste neightbor graph in dictionary format for the
    # A sites
    for n_2 in range(1,N_2+2):
        for n_1 in range(1,N_1+2):
            if n_2 == 1:
                if n_1 == 1:
                    haldane_nnn_graph_A[(n_1,n_2,'A')] = \
                        [(n_1 +1, n_2+1,'A'),]
                        
                elif n_1 == N_1+1:
                    haldane_nnn_graph_A[(n_1,n_2,'A')] = \
                        [(n_1 -1, n_2, 'A'),]
                
                else:
                    haldane_nnn_graph_A[(n_1,n_2,'A')] = \
                        [(n_1 +1, n_2+1,'A'), (n_1 -1, n_2, 'A')]
                            
            elif n_2 == N_2 +1:
                if n_1 == 1:
                    pass
                
                elif n_1 == 2:
                    haldane_nnn_graph_A[(n_1,n_2,'A')] = \
                        [(n_1,n_2 -1, 'A'), ]
                        
                else:
                    haldane_nnn_graph_A[(n_1,n_2,'A')] = \
                        [(n_1,n_2 -1, 'A'), (n_1 -1, n_2, 'A') ]
                        
            elif n_2 % 2 == 0:
                if n_1 == 1:
                    pass
                elif n_1 == 2:
                    haldane_nnn_graph_A[(n_1,n_2,'A')] = \
                        [(n_1, n_2 +1,'A'), (n_1,n_2 -1, 'A')]
                else:
                    haldane_nnn_graph_A[(n_1,n_2,'A')] = \
                        [(n_1, n_2 +1,'A'),(n_1,n_2 -1, 'A'), \
                         (n_1 -1, n_2, 'A') ]
                
            else:
                if n_1 == 1:
                    haldane_nnn_graph_A[(n_1,n_2,'A')] = \
                        [(n_1 +1, n_2+1,'A'),(n_1 +1, n_2 -1, 'A')]
                            
                elif n_1 == N_1+1:
                    haldane_nnn_graph_A[(n_1,n_2,'A')] = \
                        [(n_1 -1, n_2, 'A'),]
                            
                else:
                    haldane_nnn_graph_A[(n_1,n_2,'A')] = \
                        [(n_1 +1, n_2+1,'A'),(n_1 +1, n_2 -1, 'A'), \
                         (n_1 -1, n_2, 'A')]
                            
    # Construct next nearest neighbor graph in dictionary format for the 
    # B sites.
    for n_2 in range(1, N_2 +2):
        for n_1 in range(1, N_1 +2):
            if n_2 == 1:
                if n_1 == N_1+1:
                    pass
                    
                elif n_1 == N_1:
                    haldane_nnn_graph_B[(n_1,n_2,'B')] = \
                        [(n_1, n_2 +1, 'B'), ]
                else:
                    haldane_nnn_graph_B[(n_1,n_2,'B')] = \
                        [(n_1 +1, n_2, 'B'), (n_1, n_2 +1, 'B') ]
                
            elif n_2 == N_2+1:
                if n_1 == 1:
                    haldane_nnn_graph_B[(n_1,n_2,'B')] = \
                        [(n_1 +1, n_2, 'B'), ]
                        
                elif n_1 == N_1 + 1:
                    haldane_nnn_graph_B[(n_1,n_2,'B')] = \
                        [(n_1 -1, n_2 -1, 'B'),]
                        
                else:
                    haldane_nnn_graph_B[(n_1,n_2,'B')] = \
                        [(n_1 +1, n_2, 'B'), (n_1 -1, n_2 -1, 'B')]
                            
            elif n_2 % 2 == 0:
                if n_1 == 1:
                    haldane_nnn_graph_B[(n_1,n_2,'B')] = \
                        [(n_1 +1, n_2, 'B'), ]
                        
                elif n_1 == N_1+1:
                    haldane_nnn_graph_B[(n_1,n_2,'B')] = \
                        [(n_1 -1, n_2 +1, 'B'), (n_1 -1, n_2 -1, 'B')]
                            
                else:
                    haldane_nnn_graph_B[(n_1,n_2,'B')] = \
                        [(n_1 +1, n_2, 'B'),(n_1 -1, n_2 +1, 'B'), \
                         (n_1 -1, n_2 -1, 'B')]
            else:
                if n_1 == N_1+1:
                    pass
                
                elif n_1 == N_1:
                    haldane_nnn_graph_B[(n_1,n_2,'B')] = \
                        [(n_1, n_2 +1, 'B'), (n_1, n_2 -1, 'B')]
                            
                else:
                    haldane_nnn_graph_B[(n_1,n_2,'B')] = \
                        [(n_1 +1, n_2, 'B'), (n_1, n_2 +1, 'B'),\
                         (n_1, n_2 -1, 'B')]
        
    return haldane_nn_graph, haldane_nnn_graph_A, haldane_nnn_graph_B


def haldane_hamiltonian(N_1, M, t, t_c, phi, mu):
    """
    Generate the Hamiltonian of the Haldane heterostructure used in
    
    both papers by Garcia, Cerjan, Loring (2024) and Cerjan, Koekenbier,
    
    Schulz-Baldes (2024). The Hamiltonian is as follows
    
    $$ H &= \sum_{n_A,n_B} \left( (M-i\mu)|n_A\rangle \langle n_A| - (M+i\mu)|n_B\rangle \langle n_B| \right) - t\sum_{\langle n_A,m_B \rangle} \left( |n_A\rangle \langle m_B| - |m_B\rangle \langle n_A| \right) - t_c\sum_{\alpha=A,B}\sum_{\langle\langle n_\alpha,m_\alpha \rangle\rangle} \left( e^{i \phi(n_\alpha,m_\alpha)}| n_\alpha\rangle \langle m_\alpha | + e^{-i \phi(n_\alpha,m_\alpha)}| m_\alpha\rangle \langle n_\alpha |\right).$$
    
    WARNING: The function currently only works for N_1 = 21. 
    
    Parameters
    ----------
    N_1 : integer
        Longest number of hexagons from left to right. This tells us there
        will be $(N_1+1)^2$ cells.
    M : list or tuple
        Three floats for the Real portion of onsite energy for each material.
    t : list or tuple
        Three floats for the Nearest neighbor hoping constant for each
        material.
    t_c : list or tuple
        Three floats for the Phase constant for next nearest neighbor hoping
        constant in each of the three materials.
    phi : list or tuple
        Three floats representing the Angle for phase constant in next nearest
        neighbor interactions for each of the three materials.
    mu : list or tuple
        Three floats for the Complex portion of imaginary energy, ie gain/loss.

    Returns
    -------
    H : scipy.sparse.lil_array
        The Hamiltonian for the Haldane heterostructure
        
    References
    ----------
    Garcia, J. J., Cerjan, A., & Loring, T. A. (2024). Clifford and quadratic composite operators with applications to non-Hermitian physics. 
    arXiv:2410.03880. https://arxiv.org/abs/2410.03880
    
    Cerjan, A., Koekenbier, L., & Schulz-Baldes, H. (2023). Spectral localizer for line-gapped non-Hermitian systems. 
    *Journal of Mathematical Physics*, *64*(8), 082102. https://doi.org/10.1063/5.0150995
    """
    # The paramaters will be passed in as a list so that each material has it's
    # own constant for each with the exception of N_1 which is just a scalar.
    # N_1 is the number of hexagons from left to right
    # N_2 is the number of hexagons from bottom to top
    # Returns Hamiltonian in lil sparce matrix format
    
    N_2 = N_1
    
    # nearest neighbor, and next nearest neighbor graphs
    nn, nnnA, nnnB = haldane_graphs(N_1)
    matrix_dimension = len(nn)
    H = lil_array((matrix_dimension, matrix_dimension),dtype=complex)
        
    # manually generate ordering of sites in the clockwise manner starting
    # from the lower left part of the graph
    site_enumeration_dictionary = {}
    site_number_item = 0
    labels = ['A','B']
    for n_2 in range(1,N_2+2):
        if n_2 <= 14 and n_2 >=9:
            if n_2 % 2 == 0:
                for n_1 in range(1,N_1+2):
                    if n_1<= 14 and n_1 >= 9:
                        #inner material
                        material = 0
                    elif (n_1<=18 and n_1 > 14) or (n_1<9 and n_1 >= 5):
                        material = 1
                    else:
                        material = 2
                    
                    if n_1 == 1:
                        labels = ['B',]
                    else:
                        labels = ['A','B']
                    
                    for label in labels:
                        site_enumeration_dictionary[(n_1,n_2,label)] = (site_number_item, material)
                        site_number_item += 1
            else:
                for n_1 in range(1,N_1+2):
                    if n_1<= 14 and n_1 >= 9:
                        #inner material
                        material = 0
                    elif (n_1<=18 and n_1 > 14) or (n_1<9 and n_1 >= 5):
                        material = 1
                    else:
                        material = 2
                        
                    if n_1 == N_1+1:
                        labels = ['A',]
                    else:
                        labels = ['A', 'B']
                        
                    for label in labels:
                        site_enumeration_dictionary[(n_1,n_2,label)] = (site_number_item,material)
                        site_number_item += 1
        elif n_2 <= 18 and n_2 >=5:
            if n_2 % 2 == 0:
                for n_1 in range(1,N_1+2):
                    if (n_1<=18 and n_1 >= 5):
                        material = 1
                    else:
                        material = 2
                    
                    if n_1 == 1:
                        labels = ['B',]
                    else:
                        labels = ['A','B']
                    
                    for label in labels:
                        site_enumeration_dictionary[(n_1,n_2,label)] = (site_number_item,material)
                        site_number_item += 1
            else:
                for n_1 in range(1,N_1+2):
                    if (n_1<=18 and n_1 >= 5):
                        material = 1
                    else:
                        material = 2
                        
                    if n_1 == N_1+1:
                        labels = ['A',]
                    else:
                        labels = ['A', 'B']
                        
                    for label in labels:
                        
                        site_enumeration_dictionary[(n_1,n_2,label)] = (site_number_item,material)
                        site_number_item += 1
                        
        else:
            material = 2
            if n_2 % 2 == 0:
                for n_1 in range(1,N_1+2):
                    if n_1 == 1:
                        labels = ['B',]
                    else:
                        labels = ['A','B']
                    
                    for label in labels:
                        site_enumeration_dictionary[(n_1,n_2,label)] = (site_number_item,material)
                        site_number_item += 1
            else:
                for n_1 in range(1,N_1+2):
                    if n_1 == N_1+1:
                        labels = ['A',]
                    else:
                        labels = ['A', 'B']
                        
                    for label in labels:
                        
                        site_enumeration_dictionary[(n_1,n_2,label)] = (site_number_item,material)
                        site_number_item += 1
    
    # Sum runs over all sites in the lattice and is a staggered potential
    # Giving A and B lattices  opposite on-site energies M and -M
    # We also make A and B lattice sites at the edje lossy by having on-site
    # energies M -i*mu and M +imu respectively.
    for site, site_number_material in site_enumeration_dictionary.items():
        coupling_constant = 0
        site_number = site_number_material[0]
        material = site_number_material[1]
        if site_number % 2 == 0:
            coupling_constant = M[material] - complex(0,mu[material])
        else:
            coupling_constant = -M[material] - complex(0,mu[material])
        H[site_number,site_number] = coupling_constant
    
    # Sum over kinetic energy with nearest neighbor coupling coefficient t
    for site, nearest_neighbors in nn.items():
        site_number, material = site_enumeration_dictionary[site]
        for nearest_neighbor in nearest_neighbors:
            nearest_neighbor_number, nearest_neighbor_material = \
                site_enumeration_dictionary[nearest_neighbor]
            H[nearest_neighbor_number, site_number] = -t[material]

    # Sum over next nearest neighbor pairs for sites A with a
    # direction-dependent phase factor that breaks time reversal symmetry.
    
    H_nnn_A = lil_array((matrix_dimension,matrix_dimension),dtype=complex)
    
    for site, next_nearest_neighbors in nnnA.items():
        site_number, material = site_enumeration_dictionary[site]
        for next_nearest_neighbor in next_nearest_neighbors:
            next_nearest_neighbor_number,next_nearest_neighbor_material = \
                site_enumeration_dictionary[next_nearest_neighbor]
            
            if material == next_nearest_neighbor_material:
                H_nnn_A[next_nearest_neighbor_number, site_number] = -t_c[material] \
                * complex(np.round(np.cos(phi[material]),15),np.round(np.sin(phi[material]),15))
    
    H += H_nnn_A + H_nnn_A.getH()

    # Sum over next nearest neighbor pairs for sites B with a
    # direction-dependent phase factor that breaks time reversal symmetry.
    H_nnn_B = lil_array((matrix_dimension,matrix_dimension),dtype=complex)
    
    for site, next_nearest_neighbors in nnnB.items():
        site_number, material = site_enumeration_dictionary[site]
        for next_nearest_neighbor in next_nearest_neighbors:
            next_nearest_neighbor_number, next_nearest_neighbor_material = \
                site_enumeration_dictionary[next_nearest_neighbor]
            
            if material == next_nearest_neighbor_material:
                H_nnn_B[next_nearest_neighbor_number, site_number] = -t_c[material] \
                * complex(np.round(np.cos(phi[material]),15),np.round(np.sin(phi[material]),15))
    H += H_nnn_B + H_nnn_B.getH()
    
    return H

def haldane_positions(N_1, d):
    """
    Generate the X,Y Positions of the Haldane heterostructure used in
    
    both papers by Garcia, Cerjan, Loring (2024) and Cerjan, Koekenbier,
    
    Schulz-Baldes (2024).
    

    Parameters
    ----------
    N_1 : integer
        Longest number of hexagons from left to right. This tells us there
        will be $(N_1+1)^2$ cells.
    d : float
        Positive number representing spacing between cells.

    Returns
    -------
    X : scipy.sparse.lil_array
        The X Position observable for the Haldane heterostructure.
    Y : scipy.sparse.lil_array
        The Y Position observable for the Haldane heterostructure.
        
    References
    ----------
    Garcia, J. J., Cerjan, A., & Loring, T. A. (2024). Clifford and quadratic composite operators with applications to non-Hermitian physics. 
    arXiv:2410.03880. https://arxiv.org/abs/2410.03880
    
    Cerjan, A., Koekenbier, L., & Schulz-Baldes, H. (2023). Spectral localizer for line-gapped non-Hermitian systems. 
    *Journal of Mathematical Physics*, *64*(8), 082102. https://doi.org/10.1063/5.0150995
    """
    # N_1 is the number of hexagons from left to right
    # N_2 is the number of hexagons from bottom to top
    # Returns Hamiltonian in lil sparce matrix format
    
    N_2 = N_1
    
    matrix_dimension = int((2*N_1 +1)*(N_1+1))
    
    X_1 = lil_array((matrix_dimension, matrix_dimension),dtype=float)
    X_2 = lil_array((matrix_dimension, matrix_dimension),dtype=float)
    
    # 
    site_number = 0
    x_1_step = (d*np.sqrt(3))/2
    x_2_step = d/2
    x_1 = 1
    x_2 = 1 + d/2
    
    for n_2 in range(1,N_2+2):
        if n_2 % 2 == 0:
            for n_1 in range(1,2*N_1 + 2):
                if n_1 == 1:
                    pass
                elif n_1 % 2 == 0:
                    x_1 = x_1 + x_1_step
                    x_2 = x_2 + x_2_step
                else:
                    x_1 = x_1 + x_1_step
                    x_2 = x_2 - x_2_step
                    
                X_1[site_number,site_number] = x_1
                X_2[site_number,site_number] = x_2
                site_number += 1
                
            x_1 = 1
            x_2 = x_2 + 2*d
        else:
            for n_1 in range(1, 2*N_1 + 2):
                if n_1 == 1:
                    pass
                elif n_1 % 2 ==0:
                    x_1 = x_1 + x_1_step
                    x_2 = x_2 - x_2_step
                else:
                    x_1 = x_1 + x_1_step
                    x_2 = x_2 + x_2_step
                X_1[site_number,site_number] = x_1
                X_2[site_number,site_number] = x_2
                site_number += 1
            x_1 = 1
            x_2 = x_2 + d
            
    return X_1, X_2
    
if __name__ == '__main__':
    
    H = haldane_hamiltonian(21, [0,0.3*np.sqrt(3),0.5*np.sqrt(3)], [1,1,1], [0.5,0,0], [np.pi/2,0,0], [0,0,0.2])
    H_array = H.toarray()
    eigvals = np.linalg.eigvals(H_array)
    
    plt.scatter(eigvals.real, eigvals.imag)
    
    