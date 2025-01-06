 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 09:58:40 2023

@author: josejgarcia
"""
import numpy as np
import scipy as sp
import os
from matplotlib import pyplot as plt
import datetime

def generate_hamiltonian(dg_1,dg_2, c):
    """
    Generate a shifted two-level system Hamiltonian matrix with the given
    
    parameters. The Hamiltonian $ H $ is defined as:

    $$H = \begin{bmatrix} dg_1 + i dg_2 & c \\ c & 0 \end{bmatrix}$$

    Parameters
    ----------
    dg_1 : float
        Real part of site 1 onsite energy.
    dg_2 : float
        Imaginary part of site 1 onsite energy (loss or gain).
    c : float
        Hopping term between two sites.

    Returns
    -------
    H : numpy.ndarray
        Hamiltonian matrix with the provided parameters.
    """
    
    H = np.zeros((2,2) ,dtype = 'complex')
    H[0,0] = complex(dg_1,dg_2)
    H[0,1] = c
    H[1,0] = c
    H[1,1] = 0
    
    return H

def generate_position():
    """"
    Generate a two-level system 1d position matrix with the given parameters.

    The Position $ X $ is defined as:

    $$X = \begin{bmatrix} -1 & 0 \\ 1 & 0 \end{bmatrix}$$

    Parameters
    ----------
    None

    Returns
    -------
    X : numpy.ndarray
        Position Matrix with one site at x=-1 and the second at x=1.
    """
    X = np.zeros((2,2) ,dtype = 'complex')
    X[0,0] = -1
    X[1,1] = 1
    
    return X

def test_twolevel_system(dg_1,dg_2,c):
    """
    Code to test the validity of the generated Hamiltonian in the function
    :func: 'generate_hamiltonian()'

    Parameters
    ----------
    dg_1 : float
        Real part of site 1 onsite energy.
    dg_2 : float
        Imaginary part of site 1 onsite energy (loss or gain).
    c : float
        Hopping term between two sites.

    Returns
    -------
    Success: boolean
        Prints in a pyplot the real and imaginary part of eigenvalues and saves
        the plot to a file in the local directory.
        
        It also saves a plot of the eigenvectors and saves them as bar charts.
    """

    H = generate_hamiltonian(dg_1, dg_2, c)
    
    eigvals, eigvects = sp.linalg.eig(H,overwrite_a=True,check_finite=False)
    
    plt.xlabel('Real Part of Eigenvalue')
    plt.ylabel('Imaginary Part of Eigenvalue')
    plt.scatter(eigvals.real, eigvals.imag)
    plt.savefig(f'ev_dg_1-{dg_1}_dg_2-{dg_2}_c-{c}.png')
    
    plt.clf()
    
    plt.xlabel('Index of Eigenvector')
    plt.ylabel('Output of Eigenvector')
    for k in range(0,2):
        evector = eigvects[:,k]
        plt.bar([1,2],evector, align='center')
        plt.savefig(f'evect_dg_1-{dg_1}_dg_2-{dg_2}_c-{c}.png')
        plt.clf()
        
    return True

def gen_system(dg_1,dg_2, c):
    """
    Generate a shifted two-level system Hamiltonian matrix and position
    
    obervable with the given parameters.

    The Hamiltonian $ H $ is defined as:

    $$H = \begin{bmatrix} dg_1 + i dg_2 & c \\ c & 0 \end{bmatrix}$$
    
    The position observable is:
        
    $$X = \begin{bmatrix} -1 & 0 \\ 0 & 1 \end{bmatrix}$$
    
    Note that this is a wrapper for calling both functions
    
    :func: 'generate_hamiltonian()' and :func: 'generate_position()'

    Parameters
    ----------
    dg_1 : float
        Real part of site 1 onsite energy.
    dg_2 : float
        Imaginary part of site 1 onsite energy (loss or gain).
    c : float
        Hopping term between two sites.

    Returns
    -------
    X : numpy.ndarray
        Position Matrix with one site at x=-1 and the second at x=1.
    H : numpy.ndarray
        Hamiltonian matrix with the provided parameters.
    """
    
    H = generate_hamiltonian(dg_1, dg_2, c)
    X = generate_position()
    
    return X,H
        
def clifford_comp(kappa,x,E,X,H):
    """
    Generate the Clifford composite operator with tuning parameter kappa,
    
    position probe x, energy probe E, position observable X and non-Hermitian
    
    Hamiltonian H.
    
    Note: in this operator is refered to as the non-Hermitian spectral
    
    localizer in both the papers Cerjan, Koekenbier, Shulz-Baldes (2023) and
    
    Cerjan, Loring (2024). While in the paper by Garcia, Cerjan, Loring(2024)
    
    it is known as the Clifford composite operator.

    Parameters
    ----------
    kappa : float
        Tuning parameter for Clifford composite operator
    x : float
        Position probe site.
    E : complex
        Complex 'energy' probe site.
    X : numpy.ndarray
        X Position observable.
    H : numpy.ndarray
        Hamiltonian for two level system.

    Returns
    -------
    L : numpy.ndarray
        Clifford composite operator
        
    References
    ----------
    Garcia, J. J., Cerjan, A., & Loring, T. A. (2024). Clifford and quadratic composite operators with applications to non-Hermitian physics. 
    arXiv:2410.03880. https://arxiv.org/abs/2410.03880
    
    Cerjan, A., & Loring, T. A. (2024). Classifying photonic topology using the spectral localizer and numerical K-theory. 
    *APL Photonics*, *9*(11). https://doi.org/10.1063/5.0239018
    
    Cerjan, A., Koekenbier, L., & Schulz-Baldes, H. (2023). Spectral localizer for line-gapped non-Hermitian systems. 
    *Journal of Mathematical Physics*, *64*(8), 082102. https://doi.org/10.1063/5.0150995
    """
    L = np.zeros((4,4), dtype='complex')
    L[0:2,0:2] = (H-E*np.identity(2))
    L[2:4,0:2] = kappa*(X-x*np.identity(2))
    L[0:2,2:4] = kappa*(X-x*np.identity(2))
    L[2:4,2:4] = -np.transpose(np.conj(H-E*np.identity(2)))
    
    return L

def m_operator(kappa,x,E,X,H):
    """
    Generate the M operator defined in the paper by Garcia, Cerjan and Loring
    
    with tuning parameter kappa, position probe x, energy probe E, position
    
    observable X and non-Hermitian Hamiltonian H. The M operator is as follows:
        
        $$ M = \begin{bmatrix} \kappa(X-x) \\ \kappa(H-E) \end{bmatrix} $$

    Parameters
    ----------
    kappa : float
        Tuning parameter for Clifford composite operator.
    x : float
        Position probe site.
    E : complex
        Complex 'energy' probe site.
    X : numpy.ndarray
        Imaginary part of site 2 onsite energy (loss or gain).
    H : numpy.ndarray
        Hopping term between two sites.

    Returns
    -------
    M : numpy.ndarray
        M opertor as as mentioned above.
        
    References
    ----------
    Garcia, J. J., Cerjan, A., & Loring, T. A. (2024). Clifford and quadratic composite operators with applications to non-Hermitian physics. 
    arXiv:2410.03880. https://arxiv.org/abs/2410.03880
    
    """
    
    M = np.zeros((4,2),dtype='complex')
    M[0:2,0:2] = kappa*(X-x*np.identity(2))
    M[2:4,0:2] = (H-E*np.identity(2))
    
    return M
    M = np.zeros((4,2),dtype='complex')
    M[0:2,0:2] = kappa*(X-x*np.identity(2))
    M[2:4,0:2] = (H-E*np.identity(2))
    
    return M

def clifford_gap(L,quantity=1):
    """
    Compute the Clifford linear or radial gap of the provided Clifford
    
    composite operator. It can also provide computation of the smallest eigenvalue
    
    and vector of the Clifford composite operator. In order the following are
    
    computed with Spec being the spectrum of L and $\sigma_min$ being the
    
    smallest singular value of L:
        
    quantity = 1 , $ \min | \Re(\text{Spec}(L)) |$
    
    quantity = 2 , $ \min | \text{Spec}(L) |$
    
    quantity = 3 , $ \sigma_\min(L)$.
    
    
    Note: when the input parameter quantity is 1, the return value is the
    
    non-Hermitian spectral localizer gap in both the papers Cerjan, Koekenbier,
    
    Shulz-Baldes (2023) and Cerjan, Loring (2024). It is known as the Clifford
    
    linear gap in the paper by Garcia, Cerjan, Loring(2024)
    
    where the clifford radial gap (quanity=3) is defined.

    Parameters
    ----------
    L : numpy.ndarray
        Clifford composite operator.
    quantity : int, optional
        Quantity of interest, quantity = 1 to compute the Clifford radial gap
        and corresponding vector, quantity = 2 to compute the smallest
        eigenvalue in absolute value and corresponding vector, and quantity =3
        to compute the Clifford radial gap with corresponding vector.

    Returns
    -------
    gap : float
        Positive float.
    gap_vector : numpy.ndarray
        eigenvector associated with the gap value.
        
    References
    ----------
    Garcia, J. J., Cerjan, A., & Loring, T. A. (2024). Clifford and quadratic composite operators with applications to non-Hermitian physics. 
    arXiv:2410.03880. https://arxiv.org/abs/2410.03880
    
    Cerjan, A., & Loring, T. A. (2024). Classifying photonic topology using the spectral localizer and numerical K-theory. 
    *APL Photonics*, *9*(11). https://doi.org/10.1063/5.0239018
    
    Cerjan, A., Koekenbier, L., & Schulz-Baldes, H. (2023). Spectral localizer for line-gapped non-Hermitian systems. 
    *Journal of Mathematical Physics*, *64*(8), 082102. https://doi.org/10.1063/5.0150995
    """
    
    if quantity == 1 or quantity == 2:
        eigvals, vects = sp.linalg.eig(L,overwrite_a=False,check_finite=False)
        if quantity == 1:
            gap = min(np.abs(eigvals[0].real),np.abs(eigvals[1].real))
            np_list = list(np.abs(eigvals.real))
        else:
            gap = min(np.abs(eigvals[0]),np.abs(eigvals[1]))
            np_list = list(np.abs(eigvals))
        gap_index = np_list.index(gap)
        gap_vector = vects[gap_index,:]
    else:
        _, singvals, vects = sp.linalg.svd(L,full_matrices=False,)
        gap = min(singvals[0],singvals[1])
        np_list = list(singvals)
        gap_index = np_list.index(gap)
        gap_vector = np.conj(vects[gap_index,:])

    return gap, gap_vector

def quadratic_gap(M):
    """
    Compute the quadratic gap of the physical system given the M operator
    
    instead of the fully constructed Quadratic composite operator.
    
    The following is computed:
        
        $ \sigma_\min(M)$.
    
    
    Note: In the paper by Cerjan, Loring, Vides (2023) the quadratic operator
    
    is defined for Hermitian systems while in the paper by Garcia, Cerjan,
    
    Loring(2024) the quadratic operator is defined for non-Hermitian physical
    
    systems. In both scenarios the corresponding M operator, the construction
    
    is the same.

    Parameters
    ----------
    M : numpy.ndarray
        M operator for given physical system.

    Returns
    -------
    gap : float
        Positive float.
    gap_vector : numpy.ndarray
        eigenvector associated with the gap value.
        
    References
    ----------
    Garcia, J. J., Cerjan, A., & Loring, T. A. (2024). Clifford and quadratic composite operators with applications to non-Hermitian physics. 
    arXiv:2410.03880. https://arxiv.org/abs/2410.03880
    
    Cerjan, A., Loring, T. A., & Vides, F. (2023). Quadratic pseudospectrum for identifying localized states. 
    *Journal of Mathematical Physics*, *64*(2), 023501. https://doi.org/10.1063/5.0098336
    """
    
    left_singvects, singvals, right_singvects = sp.linalg.svd(M,full_matrices=False,overwrite_a=False,check_finite=False)
    
    gap = min(singvals)
    np_list = list(singvals)
    gap_index = np_list.index(gap)
    gap_vector = np.conj(right_singvects[gap_index,:])
    
    return gap, gap_vector

def plot_gaps(kappa,fixed_dim_input,X,H,filenames,output_coord = (2,3), m=100):
    """
    Compute and plot the various gap functions and their differences, the plots
    
    are saved to provided filenames.
    
    Note: This is the main function used to generate the two level system gap
    
    plots in the paper by Garcia, Cerjan, Loring(2024).

    Parameters
    ----------
    kappa : float
        Tuning parameter for Clifford composite operator and Quadratic
        composite operator
    fixed_dim_input : list or tuple
        Fixed probe site not including kappa.
    X : numpy.ndarray
        X Position observable.
    H : numpy.ndarray
        Hamiltonian for system.
    filenames: dictionary
        Dictionary that contains as keys the plots to be generated, and the
        value will be the corresponding filename. Possible keys are
        'l' for linear gap plot, 'r' for radial gap plot, 'q' for quadratic gap
        plot, 'lr' for the plot of the difference in linear and radial gap,
        'lq' for the plot of the difference in linear and quadratic gap, and
        'rq' for the plot of the difference in radial and quadratic gap.
    output_coord: tuple, optional
        The tuple determines what the x and y axis will be on the plot. In
        particular (1,2), (1,3), (2,3) are the only options. The integer 1
        represents position, 2 represents real energy part, 3 represents the
        imaginary energy part. For example in Garcia, Cerjan, Loring (2024)
        (2,3) is used and is the default in the function.
    m : integer, optional
        Positive integer to designate the number of points in both x and y axis
        grid. Default is m=100.
    
    Returns
    -------
    success : boolean
        Sucess of fuction as well as saves the generated plots to the filenames
        given, provided the file path exists.
        
    References
    ----------
    Garcia, J. J., Cerjan, A., & Loring, T. A. (2024). Clifford and quadratic composite operators with applications to non-Hermitian physics. 
    arXiv:2410.03880. https://arxiv.org/abs/2410.03880
    
    Cerjan, A., & Loring, T. A. (2024). Classifying photonic topology using the spectral localizer and numerical K-theory. 
    *APL Photonics*, *9*(11). https://doi.org/10.1063/5.0239018
    
    Cerjan, A., Koekenbier, L., & Schulz-Baldes, H. (2023). Spectral localizer for line-gapped non-Hermitian systems. 
    *Journal of Mathematical Physics*, *64*(8), 082102. https://doi.org/10.1063/5.0150995
    """
    
    outputs = tuple(filenames.keys())
    
    if 'lq' in outputs:
        lq_diff_data_matrix = np.zeros((m,m))
    if 'lr' in outputs:
        lr_diff_data_matrix = np.zeros((m,m))
    if 'rq' in outputs:
        rq_diff_data_matrix = np.zeros((m,m))
    if 'q' in outputs:
        q_data_matrix = np.zeros((m,m))
    if 'l' in outputs:
        l_data_matrix = np.zeros((m,m))
    if 'r' in outputs:
        r_data_matrix = np.zeros((m,m))
    
    x_fixed = fixed_dim_input[0]
    reE_fixed = fixed_dim_input[1]
    imE_fixed = fixed_dim_input[2]
    
    x_input = np.linspace(-1,1,num=m,endpoint=True)
    reE_input = np.linspace(-10,10,num=m,endpoint=True)
    imE_input = np.linspace(-10,10,num=m,endpoint=True)
    
    plt.rcParams['text.usetex'] = True
    plt.rcParams['xtick.labelsize']=16
    plt.rcParams['ytick.labelsize']=16
    
    if output_coord == (1,3):
        #plot position and imaginary energy
        for x_ind, x in enumerate(x_input):
            for y_ind, imE in enumerate(imE_input):
                if ('q' in outputs) or ('lq'in outputs) or ('rq' in outputs):
                    M = m_operator(kappa, x , complex(reE_fixed,imE), X, H)
                    quad_gap, _ = quadratic_gap(M)
                
                if ('l' in outputs) or ('lq'in outputs) or ('lr' in outputs):
                    L = clifford_comp(kappa, x, complex(reE_fixed,imE), X, H)
                    linear_gap, _ = clifford_gap(L,1)
                
                if ('r' in outputs) or ('lr'in outputs) or ('rq' in outputs):
                    L = clifford_comp(kappa, x, complex(reE_fixed,imE), X, H)
                    radial_gap, _ = clifford_gap(L,3)
                
                if 'lq' in outputs:
                    lq_diff_data_matrix[y_ind,x_ind] = np.abs(linear_gap - quad_gap)
                if 'lr' in outputs:
                    lr_diff_data_matrix[y_ind,x_ind] = np.abs(linear_gap - radial_gap)
                if 'rq' in outputs:
                    rq_diff_data_matrix[y_ind,x_ind] = np.abs(quad_gap - radial_gap)
                if 'q' in outputs:
                    q_data_matrix[y_ind,x_ind] = quad_gap
                if 'l' in outputs:
                    l_data_matrix[y_ind,x_ind] = linear_gap
                if 'r' in outputs:
                    r_data_matrix[y_ind,x_ind] = radial_gap
                    
        single_max = max(np.max(q_data_matrix),np.max(l_data_matrix),np.max(r_data_matrix))
        diff_max = max(np.max(lq_diff_data_matrix),np.max(lr_diff_data_matrix),np.max(rq_diff_data_matrix))
        
        if 'lq' in outputs:
            extents = (0,3, imE_fixed-0.25,imE_fixed+0.25)
            plt.xlabel('$x$ Position',fontdict={'fontsize': 18, 'fontweight': 'medium'})
            plt.ylabel('Im$(E)$ Energy',fontdict={'fontsize': 18, 'fontweight': 'medium'})
            plt.imshow(lq_diff_data_matrix,cmap='viridis',origin='lower',extent=extents,aspect='auto')
            plt.clim(0,diff_max)
            plt.colorbar()
            plt.savefig(filenames['lq'],bbox_inches='tight')
            plt.clf()
        if 'lr' in outputs:
            extents = (0,3, imE_fixed-0.25,imE_fixed+0.25)
            plt.xlabel('$x$ Position',fontdict={'fontsize': 18, 'fontweight': 'medium'})
            plt.ylabel('Im$(E)$ Energy',fontdict={'fontsize': 18, 'fontweight': 'medium'})
            plt.imshow(lr_diff_data_matrix,cmap='viridis',origin='lower',extent=extents,aspect='auto')
            plt.clim(0,diff_max)
            plt.colorbar()
            plt.savefig(filenames['lr'],bbox_inches='tight')
            plt.clf()
        if 'rq' in outputs:
            extents = (0,3, imE_fixed-0.25,imE_fixed+0.25)
            plt.xlabel('$x$ Position',fontdict={'fontsize': 18, 'fontweight': 'medium'})
            plt.ylabel('Im$(E)$ Energy',fontdict={'fontsize': 18, 'fontweight': 'medium'})
            plt.imshow(rq_diff_data_matrix,cmap='viridis',origin='lower',extent=extents,aspect='auto')
            plt.clim(0,diff_max)
            plt.colorbar()
            plt.savefig(filenames['rq'],bbox_inches='tight')
            plt.clf()
        if 'q' in outputs:
            extents = (0,3, imE_fixed-0.25,imE_fixed+0.25)
            plt.xlabel('$x$ Position',fontdict={'fontsize': 18, 'fontweight': 'medium'})
            plt.ylabel('Im$(E)$ Energy',fontdict={'fontsize': 18, 'fontweight': 'medium'})
            plt.imshow(q_data_matrix,cmap='viridis',origin='lower',extent=extents,aspect='auto')
            plt.clim(0,single_max)
            plt.colorbar()
            plt.savefig(filenames['q'],bbox_inches='tight')
            plt.clf()
        if 'l' in outputs:
            extents = (0,3, imE_fixed-0.25,imE_fixed+0.25)
            plt.xlabel('$x$ Position',fontdict={'fontsize': 18, 'fontweight': 'medium'})
            plt.ylabel('Im$(E)$ Energy',fontdict={'fontsize': 18, 'fontweight': 'medium'})
            plt.imshow(l_data_matrix,cmap='viridis',origin='lower',extent=extents,aspect='auto')
            plt.clim(0,single_max)
            plt.colorbar()
            plt.savefig(filenames['l'],bbox_inches='tight')
            plt.clf()
        if 'r' in outputs:
            extents = (0,3, imE_fixed-0.25,imE_fixed+0.25)
            plt.xlabel('$x$ Position',fontdict={'fontsize': 18, 'fontweight': 'medium'})
            plt.ylabel('Im$(E)$ Energy',fontdict={'fontsize': 18, 'fontweight': 'medium'})
            plt.imshow(r_data_matrix,cmap='viridis',origin='lower',extent=extents,aspect='auto')
            plt.clim(0,single_max)
            plt.colorbar()
            plt.savefig(filenames['r'],bbox_inches='tight')
            plt.clf()

    
    elif output_coord == (1,2):
        #plot position and real energy
        for x_ind, x in enumerate(x_input):
            for y_ind, reE in enumerate(reE_input):
                if ('q' in outputs) or ('lq'in outputs) or ('rq' in outputs):
                    M = m_operator(kappa, x , complex(reE,imE_fixed), X, H)
                    quad_gap, _ = quadratic_gap(M)
                
                if ('l' in outputs) or ('lq'in outputs) or ('lr' in outputs):
                    L = clifford_comp(kappa, x, complex(reE,imE_fixed), X, H)
                    linear_gap, _ = clifford_gap(L,1)
                
                if ('r' in outputs) or ('lr'in outputs) or ('rq' in outputs):
                    L = clifford_comp(kappa, x, complex(reE,imE_fixed), X, H)
                    radial_gap, _ = clifford_gap(L,3)
                    
                if 'lq' in outputs:
                    lq_diff_data_matrix[y_ind,x_ind] = np.abs(linear_gap - quad_gap)
                if 'lr' in outputs:
                    lr_diff_data_matrix[y_ind,x_ind] = np.abs(linear_gap - radial_gap)
                if 'rq' in outputs:
                    rq_diff_data_matrix[y_ind,x_ind] = np.abs(quad_gap - radial_gap)
                if 'q' in outputs:
                    q_data_matrix[y_ind,x_ind] = quad_gap
                if 'l' in outputs:
                    l_data_matrix[y_ind,x_ind] = linear_gap
                if 'r' in outputs:
                    r_data_matrix[y_ind,x_ind] = radial_gap
        
        single_max = max(np.max(q_data_matrix),np.max(l_data_matrix),np.max(r_data_matrix))
        diff_max = max(np.max(lq_diff_data_matrix),np.max(lr_diff_data_matrix),np.max(rq_diff_data_matrix))
        
        if 'lq' in outputs:
            extents = (0,3, reE_fixed-0.25,reE_fixed+0.25)
            plt.xlabel('$x$ Position',fontdict={'fontsize': 18, 'fontweight': 'medium'})
            plt.ylabel('$\Re E$ Energy',fontdict={'fontsize': 18, 'fontweight': 'medium'})
            plt.imshow(lq_diff_data_matrix,cmap='viridis',origin='lower',extent=extents,aspect='auto')
            plt.clim(0,diff_max)
            plt.colorbar()
            plt.savefig(filenames['lq'],bbox_inches='tight')
            plt.clf()
        if 'lr' in outputs:
            extents = (0,3, reE_fixed-0.25,reE_fixed+0.25)
            plt.xlabel('$x$ Position',fontdict={'fontsize': 18, 'fontweight': 'medium'})
            plt.ylabel('$\Re E$ Energy',fontdict={'fontsize': 18, 'fontweight': 'medium'})
            plt.imshow(lr_diff_data_matrix,cmap='viridis',origin='lower',extent=extents,aspect='auto')
            plt.clim(0,diff_max)
            plt.colorbar()
            plt.savefig(filenames['lr'],bbox_inches='tight')
            plt.clf()
        if 'rq' in outputs:
            extents = (0,3, reE_fixed-0.25,reE_fixed+0.25)
            plt.xlabel('$x$ Position',fontdict={'fontsize': 18, 'fontweight': 'medium'})
            plt.ylabel('$\Re E$ Energy',fontdict={'fontsize': 18, 'fontweight': 'medium'})
            plt.imshow(rq_diff_data_matrix,cmap='viridis',origin='lower',extent=extents,aspect='auto')
            plt.clim(0,diff_max)
            plt.colorbar()
            plt.savefig(filenames['rq'],bbox_inches='tight')
            plt.clf()
        if 'q' in outputs:
            extents = (0,3, reE_fixed-0.25,reE_fixed+0.25)
            plt.xlabel('$x$ Position',fontdict={'fontsize': 18, 'fontweight': 'medium'})
            plt.ylabel('$\Re E$ Energy',fontdict={'fontsize': 18, 'fontweight': 'medium'})
            plt.imshow(q_data_matrix,cmap='viridis',origin='lower',extent=extents,aspect='auto')
            plt.clim(0,single_max)
            plt.colorbar()
            plt.savefig(filenames['q'],bbox_inches='tight')
            plt.clf()
        if 'l' in outputs:
            extents = (0,3, reE_fixed-0.25,reE_fixed+0.25)
            plt.xlabel('$x$ Position',fontdict={'fontsize': 18, 'fontweight': 'medium'})
            plt.ylabel('$\Re E$ Energy',fontdict={'fontsize': 18, 'fontweight': 'medium'})
            plt.imshow(l_data_matrix,cmap='viridis',origin='lower',extent=extents,aspect='auto')
            plt.clim(0,single_max)
            plt.colorbar()
            plt.savefig(filenames['l'],bbox_inches='tight')
            plt.clf()
        if 'r' in outputs:
            extents = (0,3, reE_fixed-0.25,reE_fixed+0.25)
            plt.xlabel('$x$ Position',fontdict={'fontsize': 18, 'fontweight': 'medium'})
            plt.ylabel('$\Re E$ Energy',fontdict={'fontsize': 18, 'fontweight': 'medium'})
            plt.imshow(r_data_matrix,cmap='viridis',origin='lower',extent=extents,aspect='auto')
            plt.clim(0,single_max)
            plt.colorbar()
            plt.savefig(filenames['r'],bbox_inches='tight')
            plt.clf()
        
    elif output_coord == (2,3):
        #plot real energy and imaginary energy
        for x_ind, reE in enumerate(reE_input):
            for y_ind, imE in enumerate(imE_input):
                if ('q' in outputs) or ('lq'in outputs) or ('rq' in outputs):
                    M = m_operator(kappa, x_fixed , complex(reE,imE), X, H)
                    quad_gap, _ = quadratic_gap(M)
                
                if ('l' in outputs) or ('lq'in outputs) or ('lr' in outputs):
                    L = clifford_comp(kappa, x_fixed , complex(reE,imE), X, H)
                    linear_gap, _ = clifford_gap(L,1)
                
                if ('r' in outputs) or ('lr'in outputs) or ('rq' in outputs):
                    L = clifford_comp(kappa, x_fixed , complex(reE,imE), X, H)
                    radial_gap, _ = clifford_gap(L,3)
                    
                if 'lq' in outputs:
                    lq_diff_data_matrix[y_ind,x_ind] = np.abs(linear_gap - quad_gap)
                if 'lr' in outputs:
                    lr_diff_data_matrix[y_ind,x_ind] = np.abs(linear_gap - radial_gap)
                if 'rq' in outputs:
                    rq_diff_data_matrix[y_ind,x_ind] = np.abs(quad_gap - radial_gap)
                if 'q' in outputs:
                    q_data_matrix[y_ind,x_ind] = quad_gap
                if 'l' in outputs:
                    l_data_matrix[y_ind,x_ind] = linear_gap
                if 'r' in outputs:
                    r_data_matrix[y_ind,x_ind] = radial_gap
        
        single_max = max(np.max(q_data_matrix),np.max(l_data_matrix),np.max(r_data_matrix))
        diff_max = max(np.max(lq_diff_data_matrix),np.max(lr_diff_data_matrix),np.max(rq_diff_data_matrix))
        
        if 'lq' in outputs:
            extents = (-10,10, -10,10)
            plt.xlabel('$\Re E$ Energy',fontdict={'fontsize': 18, 'fontweight': 'medium'})
            plt.ylabel('Im$(E)$ Energy',fontdict={'fontsize': 18, 'fontweight': 'medium'})
            plt.imshow(lq_diff_data_matrix,cmap='viridis',origin='lower',extent=extents,aspect='auto')
            plt.clim(0,diff_max)
            plt.colorbar()
            plt.savefig(filenames['lq'],bbox_inches='tight')
            plt.clf()
        if 'lr' in outputs:
            extents = (-10,10, -10,10)
            plt.xlabel('$\Re E$ Energy',fontdict={'fontsize': 18, 'fontweight': 'medium'})
            plt.ylabel('Im$(E)$ Energy',fontdict={'fontsize': 18, 'fontweight': 'medium'})
            plt.imshow(lr_diff_data_matrix,cmap='viridis',origin='lower',extent=extents,aspect='auto')
            plt.clim(0,diff_max)
            plt.colorbar()
            plt.savefig(filenames['lr'],bbox_inches='tight')
            plt.clf()
        if 'rq' in outputs:
            extents = (-10,10, -10,10)
            plt.xlabel('$\Re E$ Energy',fontdict={'fontsize': 18, 'fontweight': 'medium'})
            plt.ylabel('Im$(E)$ Energy',fontdict={'fontsize': 18, 'fontweight': 'medium'})
            plt.imshow(rq_diff_data_matrix,cmap='viridis',origin='lower',extent=extents,aspect='auto')
            plt.clim(0,diff_max)
            plt.colorbar()
            plt.savefig(filenames['rq'],bbox_inches='tight')
            plt.clf()
        if 'q' in outputs:
            extents = (-10,10, -10,10)
            plt.xlabel('$\Re E$ Energy',fontdict={'fontsize': 18, 'fontweight': 'medium'})
            plt.ylabel('Im$(E)$ Energy',fontdict={'fontsize': 18, 'fontweight': 'medium'})
            plt.imshow(q_data_matrix,cmap='viridis',origin='lower',extent=extents,aspect='auto')
            plt.clim(0,single_max)
            plt.colorbar()
            plt.savefig(filenames['q'],bbox_inches='tight')
            plt.clf()
        if 'l' in outputs:
            extents = (-10,10, -10,10)
            plt.xlabel('$\Re E$ Energy',fontdict={'fontsize': 18, 'fontweight': 'medium'})
            plt.ylabel('Im$(E)$ Energy',fontdict={'fontsize': 18, 'fontweight': 'medium'})
            plt.imshow(l_data_matrix,cmap='viridis',origin='lower',extent=extents,aspect='auto')
            plt.clim(0,single_max)
            plt.colorbar()
            plt.savefig(filenames['l'],bbox_inches='tight')
            plt.clf()
        if 'r' in outputs:
            extents = (-10,10, -10,10)
            plt.xlabel('$\Re E$ Energy',fontdict={'fontsize': 18, 'fontweight': 'medium'})
            plt.ylabel('Im$(E)$ Energy',fontdict={'fontsize': 18, 'fontweight': 'medium'})
            plt.imshow(r_data_matrix,cmap='viridis',origin='lower',extent=extents,aspect='auto')
            plt.clim(0,single_max)
            plt.colorbar()
            plt.savefig(filenames['r'],bbox_inches='tight')
            plt.clf()
            
    return True    

def plot_tls_lattice(filename=None):
    """
    Compute the quadratic gap of the physical system given the M operator
    
    instead of the fully constructed Quadratic composite operator.
    
    The following is computed:
        
        $ \sigma_\min(M)$.
    
    
    Note: In the paper by Cerjan, Loring, Vides (2023) the quadratic operator
    
    is defined for Hermitian systems while in the paper by Garcia, Cerjan,
    
    Loring(2024) the quadratic operator is defined for non-Hermitian physical
    
    systems. In both scenarios the corresponding M operator, the construction
    
    is the same.

    Parameters
    ----------
    filename : string
        Default is None, but otherwise the filename as a string should be
        provided with .png or other image file format ending.

    Returns
    -------
    success : boolean
        Returns termination of function, saves the figure to the provided file
        name or locally to "tls_lattice.png" is None is provided.
    """
    if filename == None:
        filename = 'tls_lattice.png'
        
    plt.figure(figsize=[5,2],dpi=150)
    plt.title('Two Level System Lattice')
    plt.yticks(ticks=[])
    plt.xticks(ticks=[-1,1])
    plt.xlim(-1.5,1.5)
    plt.ylim(-0.1,0.1)
    plt.plot([-1,1],[0,0],scalex=True, scaley=True)
    plt.scatter([-1,1],[0,0])
    plt.xlabel("$x$ Position")
    plt.savefig(filename,dpi=300,format='png')
    
    return True
  
if __name__ == '__main__':
    folder_date = str(datetime.datetime.now()).replace(" ", "_")
    folder_date = folder_date.replace(":","-")
    folder_date, _ , _ = folder_date.partition('.')
    newpath = f'{folder_date}' 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    #plot_tls_lattice()
    #test_twolevel_system(dg_1, dg_2, c)
    c = 1
    dg_1 = 0
    dg_2 = (2)*c
    X,H = gen_system(dg_1, dg_2, c)
    
    for position in [0]:
        print(f'Fixing Position at x={position}')
        probe_site = [1,position,0,0]
            
        filename_q = f'{newpath}/quadratic_gap_plot_c_{c}-dg2_{dg_2}-x_{probe_site[1]:0.2f}-reE_var-imE_var.png'
        filename_l = f'{newpath}/clifford_linear_gap_plot_c_{c}-dg2_{dg_2}-x_{probe_site[1]:0.2f}-reE_var-imE_var.png'
        filename_r = f'{newpath}/clifford_radial_gap_plot_c_{c}-dg2_{dg_2}-x_{probe_site[1]:0.2f}-reE_var-imE_var.png'
        filename_lq = f'{newpath}/lq_diff_plot_c_{c}-dg2_{dg_2}-x_{probe_site[1]:0.2f}-reE_var-imE_var.png'
        filename_lr = f'{newpath}/lr_diff_plot_c_{c}-dg2_{dg_2}-x_{probe_site[1]:0.2f}-reE_var-imE_var.png'
        filename_rq = f'{newpath}/rq_diff_plot_c_{c}-dg2_{dg_2}-x_{probe_site[1]:0.2f}-reE_var-imE_var.png'
            
        filenames = {'q': filename_q, 'l': filename_l, 'r': filename_r, 'lq': filename_lq, 'lr': filename_lr, 'rq': filename_rq }
        
        start_plot = datetime.datetime.now()
            
        plot_gaps(probe_site[0],probe_site[1:4],X,H, filenames, output_coord=(2,3),m=200)

        end_plot = datetime.datetime.now()

    print(f"Plot runtime is: {end_plot-start_plot}")