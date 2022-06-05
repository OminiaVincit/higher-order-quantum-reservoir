import sys
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
import plot_utils as putils

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='../../../data/hqrc/dynamics_rand/esp')
    parser.add_argument('--dynamic', type=str, default='phase_trans')
    parser.add_argument('--nqrc', type=int, default=5)
    parser.add_argument('--virtuals', type=int, default=1)
    parser.add_argument('--length', type=int, default=11000)
    parser.add_argument('--gamma', type=float, default=0.0)

    args = parser.parse_args()
    print(args)
    folder, dynamic, nqr, V, length = args.folder, args.dynamic, args.nqrc, args.virtuals, args.length
    gamma = args.gamma

    prefix  = '{}_nqr_{}_V_{}'.format(dynamic, nqr, V)
    
    fig, axs = plt.subplots(1, 1, figsize=(8, 4), squeeze=False)
    ax = axs[0, 0]
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params('both', length=8, width=1, which='major', labelsize=12, direction = "out")
    ax.tick_params('both', length=4, width=1, which='minor', direction = "out")
    ax.set_xlim(10**(-2.02), 10**2.1)
    ax.set_ylim(10**(-15), 10**1)

    pattern = '{}/{}_*_gam_{}_*_len_{}.binaryfile'.format(folder, prefix, gamma, length)
    filenames = glob.glob(pattern)
    if length >= 1e+5:
        col = putils.VERMILLION
    else:
        col = putils.BLUE
    esp_ls = []
    for filename in filenames:
        #print(filename)
        with open(filename, "rb") as rfile:
            data = pickle.load(rfile)
            Ws = sorted(data.keys())
            ys = [data[W] for W in Ws]
            xs = [10**W for W in Ws]
            #ax.plot(xs, ys, linewidth=0.7, color='#464646', alpha=0.8)
            ax.plot(xs, ys, linewidth=0.7, color=col, alpha=0.3)
            
            esp_ls.append(ys)
    esp_ls = np.array(esp_ls)
    ys = np.median(esp_ls, axis=0)
    ax.plot(xs, ys, linewidth=2, color=col, alpha=0.8)

    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.tight_layout()

    fig_folder = os.path.join(folder, 'figs')
    if os.path.isdir(fig_folder) == False:
        os.mkdir(fig_folder)
    outbase = os.path.join(fig_folder, 'esp_dim_prefix_{}_gam_{}_len_{}'.format(prefix, gamma, length))
    for ftype in ['png', 'svg']:
        plt.savefig('{}.{}'.format(outbase, ftype), bbox_inches='tight', dpi=600)
    plt.show()



