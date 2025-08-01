#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 14:41:08 2024

Code used to generate the results in

    Garcia, J. J., Cerjan, A., & Loring, T. A. (2024). Clifford and quadratic composite operators with applications to non-Hermitian physics. 
    arXiv:2410.03880. https://arxiv.org/abs/2410.03880

Authors of Paper: Jose J. Garcia, Alexander Cerjan, and Terry Loring.

Original code on github:
    
    https://github.com/josejgarcia77/clifford_and_quadratic_composite_operator_applications

@author: josejgarcia
"""

import numpy as np
from matplotlib import pyplot as plt
from haldane_model import haldane_positions

if __name__ == '__main__':
    # Routine to replot already computed plots with different display features
    # and properties.
    X, Y = haldane_positions(21,1)
    x_diag = np.diag(X.toarray())
    y_diag = np.diag(Y.toarray())
    x_min = min(x_diag)-2
    x_max = max(x_diag)+2
    y_min = min(y_diag)-2
    y_max = max(y_diag)+2
    
    folder_dir = "2025-08-01_10-14-59/"
    base = "k0.5_reE0.00_imE0.00"
    
    npz_filename = folder_dir + base + ".npz"
    
    #npzfile = np.load("2024-06-29_10-10-15/k0.5_x_var-y_var-reE0.00imE0.00.npz")
    npzfile = np.load(npz_filename)
    
    #filename_quad = "2024-06-29_10-10-15/quad_k0.5_x_var-y_var-reE0.00imE0.00.png"
    #filename_linear = "2024-06-29_10-10-15/linear_k0.5_x_var-y_var-reE0.00imE0.00.png"
    #filename_diff = "2024-06-29_10-10-15/diff_k0.5_x_var-y_var-reE0.00imE0.00.png"
    
    filename_quad = folder_dir + "quad_" + base + ".png"
    filename_linear = folder_dir + "linear_" + base + ".png"
    filename_diff = folder_dir + "diff_" + base + ".png"
    
    quad_data_matrix = npzfile['quad_gap_data']
    linear_data_matrix = npzfile['linear_gap_data']
    diff_data_matrix = npzfile['gap_diff_data']
    
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
    print(f"Saving Plot to {filename_quad}")
    plt.savefig(filename_quad,dpi=300,bbox_inches='tight')
    print(f"Saved Plot to {filename_quad}")
    plt.clf()
    
    plt.rcParams['text.usetex'] = True
    plt.rcParams['xtick.labelsize']=16
    plt.rcParams['ytick.labelsize']=16
    plt.xlabel('$x$ Position',fontdict={'fontsize': 18, 'fontweight': 'medium'})
    plt.ylabel('$y$ Position',fontdict={'fontsize': 18, 'fontweight': 'medium'})
    plt.imshow(linear_data_matrix,cmap='viridis',origin='lower',extent=(x_min,x_max,y_min,y_max))
    plt.colorbar()
    plt.clim(min_clim,max_clim)
    print(f"Saving Plot to {filename_linear}")
    plt.savefig(filename_linear,dpi=300,bbox_inches='tight')
    print(f"Saved Plot to {filename_linear}")
    plt.clf()
    
    plt.rcParams['text.usetex'] = True
    plt.rcParams['xtick.labelsize']=16
    plt.rcParams['ytick.labelsize']=16
    plt.xlabel('$x$ Position',fontdict={'fontsize': 18, 'fontweight': 'medium'})
    plt.ylabel('$y$ Position',fontdict={'fontsize': 18, 'fontweight': 'medium'})
    plt.imshow(diff_data_matrix,cmap='viridis',origin='lower',extent=(x_min,x_max,y_min,y_max))
    plt.colorbar()
    print(f"Saving Plot to {filename_diff}")
    plt.savefig(filename_diff,dpi=300,bbox_inches='tight')
    print(f"Saved Plot to {filename_diff}")
    plt.close()
    