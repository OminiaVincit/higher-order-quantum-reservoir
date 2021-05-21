#!/usr/bin/env python
"""
Separate spiral data in nonlinear map
"""

import sys
import numpy as np
import os
import scipy
import argparse
import multiprocessing
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import ticker

import time
import datetime
import re
import plot_utils as putils
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset)

RES_MNIST_DIR = "../results/rs_mnist"
MNIST_SIZE="10x10"

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--spins', type=int, default=5, help='Number of the spins')
    parser.add_argument('--dynamic', type=str, default='full_random',\
        help='full_random,half_random,full_const_trans,full_const_coeff,ion_trap')
    
    parser.add_argument('--nqrs', type=int, default=1, help='Number of reservoirs')
    parser.add_argument('--ntrials', type=int, default=1)
    parser.add_argument('--virtuals', type=int, default=1)
    parser.add_argument('--strengths', type=str, default='0.0,0.1,0.5,0.9', help='Connection strengths')
    parser.add_argument('--taudeltas', type=str, default='-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7')
    parser.add_argument('--tmin', type=float, default=0.0, help='tmin in plot inset')
    
    parser.add_argument('--linear_reg', type=int, default=0)
    parser.add_argument('--use_corr', type=int, default=0)
    parser.add_argument('--full', type=int, default=0)
    parser.add_argument('--label1', type=int, default=3)
    parser.add_argument('--label2', type=int, default=6)
    
    parser.add_argument('--savedir', type=str, default=RES_MNIST_DIR)
    parser.add_argument('--mnist_size', type=str, default=MNIST_SIZE)
    parser.add_argument('--nproc', type=int, default=100)
    parser.add_argument('--rseed', type=int, default=0)
    parser.add_argument('--inset', type=int, default=0)
    args = parser.parse_args()
    print(args)

    n_qrs, n_spins, rseed = args.nqrs, args.spins, args.rseed
    V, tmin, inset = args.virtuals, args.tmin, args.inset
    
    linear_reg, use_corr = args.linear_reg, args.use_corr
    full_mnist, label1, label2 = args.full, args.label1, args.label2
    dynamic, savedir = args.dynamic, args.savedir
    mnist_size, nproc = args.mnist_size, args.nproc

    taudeltas = [float(x) for x in args.taudeltas.split(',')]
    #taudeltas = list(np.arange(-7, 7.1, args.interval))
    taudeltas = [2**x for x in taudeltas]
    strengths = [float(x) for x in args.strengths.split(',')]

    figdir = os.path.join(savedir, 'figs')
    if os.path.isdir(figdir) == False:
        os.mkdir(figdir)

    logdir = os.path.join(savedir, 'log')
    basename = 'join_{}_{}_linear_{}_nqrs_{}_corr_{}_nspins_{}_V_{}'.format(\
        mnist_size, dynamic, linear_reg, n_qrs, use_corr, n_spins, V)
    if full_mnist <= 0:
        basename = '{}_lb_{}_{}'.format(basename, label1, label2)
    logfile = os.path.join(logdir, '{}.log'.format(basename))
    
    accs = dict()
    for alpha in strengths:
        accs[alpha] = dict()
        accs[alpha]['taus'] = []
        accs[alpha]['test_acc'] = []
        
    with open(logfile, 'r') as rf:
        lines = rf.readlines()
        for line in lines:
            if 'Test acc=' in line:
                tau = (float)(re.search('tau=([0-9.]*)', line).group(1))
                alpha = (float)(re.search('alpha=([0-9.]*)', line).group(1))
                if alpha not in strengths:
                    continue
                acc = (float)(re.search('Test acc=([0-9.]*)', line).group(1))
                accs[alpha]['taus'].append(tau)
                accs[alpha]['test_acc'].append(acc)
    
    # Plot the accuracy
    putils.setPlot(fontsize=24, labelsize=24)
    fig, axs = plt.subplots(1, 1, figsize=(24, 10), squeeze=False, dpi=600)
    axs = axs.ravel()
    ax1 = axs[0]

    if inset > 0:
        # Create a set of inset Axes: these should fill the bounding box allocated to them.
        ax2 = plt.axes([0,0,1,1])
        # Manually set the position and relative size of the inset axes within ax1
        # left, bottom, width, height
        ip = InsetPosition(ax1, [0.50,0.10,0.45,0.57])
        ax2.set_axes_locator(ip)
        # Mark the region corresponding to the inset axes on ax1 and draw lines
        # in grey linking the two axes.
        # mark_inset(ax1, ax2, loc1=2, loc2=4, fc="none", ec='0.5')

    
    colors = putils.cycle
    nc = 0
    for alpha in accs.keys():
        color = colors[nc % len(colors)]
        ts = np.array(accs[alpha]['taus'])
        avs = np.array(accs[alpha]['test_acc'])
        ax1.plot(ts, avs, 's-', label='$\\alpha=${}'.format(alpha), linewidth=4, \
            alpha=0.8, markersize=12, mec='k', mew=0.5, color=color)
        if inset > 0:
            ax2.plot(ts[ts >= 2**tmin], avs[ts >= 2**tmin], 's-', label='$\\alpha=${}'.format(alpha), linewidth=4, \
                alpha=0.8, markersize=12, mec='k', mew=0.5, color=color)
        
        nc += 1
    
    ax1.set_title('{}'.format(basename), fontsize=16)
    ax1.set_xscale("log", basex=2)
    ax1.set_xticks([2**x for x in np.arange(-7,7.01,1.0)])
    ax1.set_xlim([2**(-7), 2**7])
    #ax1.set_ylim([0.95, 0.99])
    ax1.legend()
    ax1.grid(True, which="both", ls="-", color='0.65')
    ax1.set_xlabel('$\\tau$', fontsize=32)
    ax1.set_ylabel('Accuracy', fontsize=28)
    if inset > 0:
        ax2.set_xscale("log", basex=2)
        ax2.tick_params('both', length=6, width=1, which='major', labelsize=24)
        #ax2.grid(True, which="both", ls="-", color='0.65')
        #ax2.set_xlim([2**(tmin), 2**7])

    for ax in axs:
        ax.tick_params('both', length=8, width=1, which='major', labelsize=28)
    
    outbase = os.path.join(figdir, basename)
    for ftype in ['png', 'svg']:
        plt.savefig('{}_acc.{}'.format(outbase, ftype), bbox_inches='tight', dpi=600)
    plt.show()
    
    
                

                
                