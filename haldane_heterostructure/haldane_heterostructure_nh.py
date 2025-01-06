#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat June 29 2024

@author: josejgarcia
"""

from haldane_model import haldane_hamiltonian, haldane_positions
import numpy as np
import scipy as sp
import datetime
import matplotlib.pyplot as plt
import os

f_mach_prec = np.finfo(float).eps

def generate_system():
    """
    Generate a shifted two-level system Hamiltonian matrix and position
    
    obervable with the given parameters.

    The Hamiltonian $ H $ is defined as:

    $$ H &= \sum_{n_A,n_B} \left( (M-i\mu)|n_A\rangle \langle n_A| - (M+i\mu)|n_B\rangle \langle n_B| \right) - t\sum_{\langle n_A,m_B \rangle} \left( |n_A\rangle \langle m_B| - |m_B\rangle \langle n_A| \right) - t_c\sum_{\alpha=A,B}\sum_{\langle\langle n_\alpha,m_\alpha \rangle\rangle} \left( e^{i \phi(n_\alpha,m_\alpha)}| n_\alpha\rangle \langle m_\alpha | + e^{-i \phi(n_\alpha,m_\alpha)}| m_\alpha\rangle \langle n_\alpha |\right).$$
    
    Note: that this is a wrapper for calling both functions
    
    :func: 'haldante_model.haldane_hamiltonian()' and
    
    :func: 'haldane_model.haldane_positions()'

    Parameters
    ----------
    None

    Returns
    -------
    X : scipy.sparse.lil_array
        X Position Matrix.
    Y : scipy.sparse.lil_array
        Y Position Matrix.
    H : scipy.sparse.lil_array
        Hamiltonian matrix with the provided parameters.
    """
    
    H = haldane_hamiltonian(21, [0,0.3*np.sqrt(3),0.5*np.sqrt(3)], [1,1,1], [0.5,0,0], [np.pi/2,0,0], [0,0,0.2])
    X, Y = haldane_positions(21,1)
    
    return (X, Y, H)

def clifford_comp(kappa,x,y,E,X,Y,H):
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
        X Position probe site.
    y : float
        Y Position probe site. 
    E : complex
        Complex 'energy' probe site.
    X : scipy.sparse.lil_array
        X Position observable.
    Y : scipy.sparse.lil_array
        Y Position observable.
    H : scipy.sparse.lil_array
        Hamiltonian for Haldane heterostructure.

    Returns
    -------
    L : scipy.sparse.lil_array
        Clifford composite operator.
        
    References
    ----------
    Garcia, J. J., Cerjan, A., & Loring, T. A. (2024). Clifford and quadratic composite operators with applications to non-Hermitian physics. 
    arXiv:2410.03880. https://arxiv.org/abs/2410.03880
    
    Cerjan, A., & Loring, T. A. (2024). Classifying photonic topology using the spectral localizer and numerical K-theory. 
    *APL Photonics*, *9*(11). https://doi.org/10.1063/5.0239018
    
    Cerjan, A., Koekenbier, L., & Schulz-Baldes, H. (2023). Spectral localizer for line-gapped non-Hermitian systems. 
    *Journal of Mathematical Physics*, *64*(8), 082102. https://doi.org/10.1063/5.0150995
    """
    
    N = len(X.toarray())
    clifford_comp_arr = np.zeros((2*N,2*N),dtype='complex')
    id_matrix_arr = np.identity(N,dtype='complex')
    
    x_diff = X-x*id_matrix_arr
    y_diff = Y-y*id_matrix_arr
    E_diff = H-E*id_matrix_arr
    
    clifford_comp_arr[0:N,0:N] = (E_diff)
    clifford_comp_arr[N:2*N,N:2*N] = -(np.transpose(np.conj(E_diff)))
    clifford_comp_arr[0:N,N:2*N] = kappa*(x_diff - complex(0,1)*y_diff)
    clifford_comp_arr[N:2*N,0:N] = kappa*(x_diff + complex(0,1)*y_diff)
    
    clifford_comp = sp.sparse.lil_array(clifford_comp_arr)
    
    return clifford_comp

def m_operator(kappa,x,y,E,X,Y,H):
    """
    Generate the M operator defined in the paper by Garcia, Cerjan and Loring
    
    with tuning parameter kappa, position probe x, energy probe E, position
    
    observable X and non-Hermitian Hamiltonian H. The M operator is as follows:
        
        $$ M = \begin{bmatrix} \kappa(X-x) \\ \kappa(Y-y) \\ (H-E)\end{bmatrix} $$

    Parameters
    ----------
    kappa : float
        Tuning parameter for Clifford composite operator
    x : float
        X Position probe site.
    y : float
        Y Position probe site. 
    E : complex
        Complex 'energy' probe site.
    X : scipy.sparse.lil_array
        X Position observable.
    Y : scipy.sparse.lil_array
        Y Position observable.
    H : scipy.sparse.lil_array
        Hamiltonian for Haldane heterostructure.

    Returns
    -------
    M : numpy.ndarray
        M opertor as as mentioned above.
        
    References
    ----------
    Garcia, J. J., Cerjan, A., & Loring, T. A. (2024). Clifford and quadratic composite operators with applications to non-Hermitian physics. 
    arXiv:2410.03880. https://arxiv.org/abs/2410.03880
    
    """
    
    N = len(X.toarray())
    m_operator_arr = np.zeros((3*N,N),dtype='complex')
    id_matrix_arr = np.identity(N,dtype='complex')
    
    x_diff = X-x*id_matrix_arr
    y_diff = Y-y*id_matrix_arr
    E_diff = H-E*id_matrix_arr
    
    m_operator_arr[0:N,0:N] = kappa*kappa*x_diff
    m_operator_arr[N:2*N,0:N] = kappa*kappa*y_diff
    m_operator_arr[2*N:3*N,0:N] = E_diff
    
    m_operator = sp.sparse.lil_array(m_operator_arr)
    
    return m_operator
    
def clifford_linear_gap(L):
    """
    Compute the Clifford linear Clifford composite operator. The following is
    
    computed with Spec being the spectrum of L: $ \min | \Re(\text{Spec}(L))|$.
    
    
    Note: The return value is the non-Hermitian spectral localizer gap in both
    
    the papers Cerjan, Koekenbier, Shulz-Baldes (2023) and
    
    Cerjan, Loring (2024). It is known as the Clifford
    
    linear gap in the paper by Garcia, Cerjan, Loring(2024).

    Parameters
    ----------
    L : numpy.ndarray
        Clifford composite operator.

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
    # Following code in comments can be used as an approximation but it is not
    # necessarily reliable.
    # we will assume that min|Re(lambda(L))| approx min|lambda(L)| for computation speed
    # eig_val, gap_vector = sp.sparse.linalg.eigs(L,k=1,which='SM')
    # gap = np.abs(eig_val[0].real)
    
    N_half = int(np.ceil(len(H.toarray())/2))
    eig_val, gap_vector = sp.sparse.linalg.eigs(L,k=N_half, which='LR')
    gap = np.abs(eig_val[0].real)
    
    return gap, gap_vector[:,0]

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
    
    # we use the smallest singular value characterization
    _, quad_gap, gap_vector = sp.sparse.linalg.svds(M,k=1,which='SM',solver='arpack')
    
    return quad_gap[0], np.conj(gap_vector[0])
    
def plot_gap_and_diff(kappa,fixed_dim_input, X,Y,H,filenames,output_coord = (1,2), m=100):
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
        X Position observables.
    Y : numpy.ndarray
        Y Position observables.
    H : numpy.ndarray
        Hamiltonian of system.
    filenames: dictionary
        Dictionary that contains as keys the plots to be generated, and the
        value will be the corresponding filename. Possible keys are
        'l' for linear gap plot, 'q' for quadratic gap plot,
        'lq' for the plot of the difference in linear and quadratic gap,, and
        lastly 'save' is for a file name that saves the data of each plot in
        an npz file, it should end in '.npz'.
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
    
    # Plot the gap as a function of two variables (x,y), (x,reE), (x,imE), (y,reE),
    # (y,imE), or (reE,imE).
    
    quad_data_matrix = np.zeros((m,m))
    linear_data_matrix = np.zeros((m,m))
    diff_data_matrix = np.zeros((m,m))
    vector_data = {}
    
    x_diag = np.diag(X.toarray())
    y_diag = np.diag(Y.toarray())
    x_min = min(x_diag)-2
    x_max = max(x_diag)+2
    y_min = min(y_diag)-2
    y_max = max(y_diag)+2
    
    percentage_complete = 0
    previous_percent = 0
    
    x_fixed = fixed_dim_input[0]
    y_fixed = fixed_dim_input[1]
    reE_fixed = fixed_dim_input[2]
    imE_fixed = fixed_dim_input[3]
    
    if output_coord == (1,2):
        print("Starting Computation of Data Points")
        #plot position and output quad gap
        for x_ind, x in enumerate(np.linspace(x_min,x_max,num=m,endpoint=True)):
            for y_ind, y in enumerate(np.linspace(y_min,y_max,num=m,endpoint=True)):
                M = m_operator(kappa, x, y, complex(reE_fixed,imE_fixed), X, Y, H)
                quad_gap, _ = quadratic_gap(M)
                quad_data_matrix[y_ind,x_ind] = quad_gap
                
                L = clifford_comp(kappa, x, y, complex(reE_fixed,imE_fixed), X, Y, H)
                linear_gap, _ = clifford_linear_gap(L)
                linear_data_matrix[y_ind,x_ind] = linear_gap
                
                diff_data_matrix[y_ind,x_ind] = np.abs(linear_gap-quad_gap)
                percentage_complete += 1/(m*m)
            
                if percentage_complete > previous_percent:
                    print(f"Computation of Plot Data Points: {percentage_complete:0.0%} Complete")
                    previous_percent += 0.1
        
        np.savez(filenames['save'],quad_gap_data=quad_data_matrix, linear_gap_data=linear_data_matrix, gap_diff_data=diff_data_matrix)
        
        max_linear_scale = np.nanmax(linear_data_matrix)
        max_quad_scale = np.nanmax(quad_data_matrix)
        max_clim = max(max_linear_scale,max_quad_scale)
        
        min_linear_scale = np.nanmin(linear_data_matrix)
        min_quad_scale = np.nanmin(quad_data_matrix)
        min_clim = min(min_linear_scale,min_quad_scale)
        
        plt.rcParams['text.usetex'] = True
        plt.rcParams['xtick.labelsize']=16
        plt.rcParams['ytick.labelsize']=16
        plt.xlabel('$x$ Position',fontdict={'fontsize': 18, 'fontweight': 'medium'})
        plt.ylabel('$y$ Position',fontdict={'fontsize': 18, 'fontweight': 'medium'})
        plt.imshow(quad_data_matrix,cmap='viridis',origin='lower',extent=(x_min,x_max,y_min,y_max))
        plt.colorbar()
        plt.clim(min_clim,max_clim)
        print(f"Saving Plot to {filenames['q']}")
        plt.savefig(filenames['q'],dpi=300,bbox_inches='tight')
        print(f"Saved Plot to {filenames['q']}")
        plt.clf()
        
        plt.rcParams['text.usetex'] = True
        plt.rcParams['xtick.labelsize']=16
        plt.rcParams['ytick.labelsize']=16
        plt.xlabel('$x$ Position',fontdict={'fontsize': 18, 'fontweight': 'medium'})
        plt.ylabel('$y$ Position',fontdict={'fontsize': 18, 'fontweight': 'medium'})
        plt.imshow(linear_data_matrix,cmap='viridis',origin='lower',extent=(x_min,x_max,y_min,y_max))
        plt.colorbar()
        plt.clim(min_clim,max_clim)
        print(f"Saving Plot to {filenames['l']}")
        plt.savefig(filenames['l'],dpi=300,bbox_inches='tight')
        print(f"Saved Plot to {filenames['l']}")
        plt.clf()
        
        plt.rcParams['text.usetex'] = True
        plt.rcParams['xtick.labelsize']=16
        plt.rcParams['ytick.labelsize']=16
        plt.xlabel('$x$ Position',fontdict={'fontsize': 18, 'fontweight': 'medium'})
        plt.ylabel('$y$ Position',fontdict={'fontsize': 18, 'fontweight': 'medium'})
        plt.imshow(diff_data_matrix,cmap='viridis',origin='lower',extent=(x_min,x_max,y_min,y_max))
        plt.colorbar()
        plt.clim(min_clim,max_clim)
        print(f"Saving Plot to {filenames['lq']}")
        plt.savefig(filenames['lq'],dpi=300,bbox_inches='tight')
        print(f"Saved Plot to {filenames['lq']}")
        plt.close()
    
    return True
    
    if output_coord == (3,4):
        print("Starting Computation of Plot")
        #plot complex energy with real along the x axis and imaginary along
        # the y axis.
        for x_ind, reE in enumerate(np.linspace(-10,10,num=m,endpoint=True)):
            for y_ind, imE in enumerate(np.linspace(-1,1,num=m,endpoint=True)):
                M = m_operator(kappa, x_fixed, y_fixed, complex(reE,imE), X, Y, H)
                gap, gap_vector = quadratic_gap(M)
                quad_data_matrix[y_ind,x_ind] = gap
                vector_data[(x_ind,y_ind)] = gap_vector
            
                percentage_complete += 1/(m*m)
            
                if percentage_complete > previous_percent:
                    print(f"Plot {percentage_complete:0.0%} Complete")
                    previous_percent += 0.1
        
        plt.xlabel('$\Re E$ Energy',fontdict={'fontsize': 18, 'fontweight': 'medium'})
        plt.ylabel('$\Im E$ Energy',fontdict={'fontsize': 18, 'fontweight': 'medium'})
        
        plt.imshow(quad_data_matrix,cmap='viridis',origin='lower',extent=(-10,10,-1,1),aspect='equal')
        plt.colorbar()
        
        print(f"Saving Plot to {filenames}")
        plt.savefig(filenames,dpi=300,bbox_inches='tight')
        plt.clf()
        
        return True
    
if __name__ == '__main__':
    
    folder_date = str(datetime.datetime.now()).replace(" ", "_")
    folder_date = folder_date.replace(":","-")
    folder_date, _ , _ = folder_date.partition('.')
    newpath = f'{folder_date}' 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    X,Y,H = generate_system()
    
    kappa = 0.5
    for E in [complex(-1,0),complex(0,-1)]:
        x_min = [kappa,0,0,E.real,E.imag]
    
        start_plot = datetime.datetime.now()
    
        print(f'Fixing Complex Energy at {x_min[3:5]}')
    
        filename_quad = f'{newpath}/quad_k{kappa}_x_var-y_var-reE{x_min[3]:0.2f}imE{x_min[4]:0.2f}.png'
        filename_linear = f'{newpath}/linear_k{kappa}_x_var-y_var-reE{x_min[3]:0.2f}imE{x_min[4]:0.2f}.png'
        filename_diff = f'{newpath}/diff_k{kappa}_x_var-y_var-reE{x_min[3]:0.2f}imE{x_min[4]:0.2f}.png'
        filename_matrix_save = f'{newpath}/k{kappa}_x_var-y_var-reE{x_min[3]:0.2f}imE{x_min[4]:0.2f}.npz'
        filenames = {'q': filename_quad, 'l':filename_linear, 'lq': filename_diff, 'save': filename_matrix_save}
        plot_gap_and_diff(kappa, x_min[1:5], X, Y, H,filenames=filenames,output_coord=(1,2),m=150)

    
