import sys
import numpy as np
import os
import scipy
import argparse
import multiprocessing
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
import utils

virtuals = [5*n for n in range(1, 6)]
virtuals.insert(0, 1)

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--basename', type=str, required=True)
    parser.add_argument('--label', type=str, default='nq')
    parser.add_argument('--layers', type=str, default='1,2,3,4,5')
    args = parser.parse_args()
    print(args)

    basename, lb = args.basename, args.label
    layers = [int(x) for x in args.layers.split(',')]
    outbase = basename.replace('.txt', '')
    outbase = outbase.replace('_{}_1_'.format(lb), '_')

    cmap = plt.get_cmap("viridis")
    plt.figure(figsize=(8,8))
    #plt.style.use('seaborn-colorblind')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams['font.size']=20

    for layer in layers:
        filename = basename.replace('_{}_1_'.format(lb), '_{}_{}_'.format(lb, layer))
        if os.path.isfile(filename) == False:
            continue
        
        rsarr = np.loadtxt(filename)
    
        print(rsarr)
        print(rsarr.shape)

        # plot the result
        xs = virtuals
        avg_tests, std_tests = rsarr[:, 4], rsarr[:, 6]

        plt.errorbar(xs, avg_tests, yerr=std_tests, elinewidth=2, linewidth=2, markersize=12, \
            label='N_{}={}'.format(lb, layer))
    
    plt.ylim([1e-5, 1e-2])
    
    plt.xlabel('$V$', fontsize=32)
    plt.ylabel('NMSE', fontsize=32)
    plt.yscale('log')

    plt.legend()
    plt.title(outbase, fontsize=10)
    plt.grid(True, which="both", ls="-", color='0.65')
    plt.show()
    for ftype in ['png', 'pdf']:
        plt.savefig('{}_NMSE.{}'.format(outbase, ftype), bbox_inches='tight')
        
