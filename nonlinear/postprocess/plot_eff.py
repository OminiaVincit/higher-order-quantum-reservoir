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
    parser.add_argument('--folder', type=str, default='res_high_eff_tau')
    parser.add_argument('--prefix', type=str, default='full_random_2021-05-0')
    parser.add_argument('--posfix', type=str, default='layers_5_eff_ntrials_10_eff')
    parser.add_argument('--sparsity', type=float, default=1.0)
    parser.add_argument('--virtuals', type=int, default=1)
    parser.add_argument('--vmin', type=float, default=1.0)
    parser.add_argument('--vmax', type=float, default=1.0)
    
    args = parser.parse_args()
    print(args)
    folder, prefix, posfix = args.folder, args.prefix, args.posfix
    posfix = 'sparsity_{}_V_{}_{}'.format(args.sparsity, args.virtuals, posfix)

    pstates = np.linspace(0.0, 1.0, 101)
    eff_ls = []

    tmin, tmax = -7, 7
    for p in pstates:
        for filename in glob.glob('{}/{}*_strength_{:.2f}_{}.txt'.format(folder, prefix, p, posfix)):
            rs_arr = np.loadtxt(filename)
            print(p, rs_arr.shape)
            t1 = np.array(rs_arr[:,3]).ravel()
            t1 = t1[rs_arr[:,1] >= (2**tmin)]
            t1 = t1[t1 <= (2**tmax)]
            eff_ls.append(t1)

    eff_ls = np.array(eff_ls)
    print(eff_ls.shape)
    (nu, nt) = eff_ls.shape
    # Plot file
    putils.setPlot(fontsize=24, labelsize=24)
    cmap = plt.get_cmap('nipy_spectral')
    cmap = plt.get_cmap('gist_rainbow_r')
    cmap = plt.get_cmap('gist_ncar')
    cmap = plt.get_cmap('gist_stern')

    fig, axs = plt.subplots(1, 1, figsize=(24, 8), squeeze=False, dpi=600)
    axs = axs.ravel()
    ax1 = axs[0]
    title = '{}'.format(os.path.basename(filename))
    
    vmin, vmax = args.vmin, args.vmax
    if vmin >= vmax:
        vmin, vmax = 1.0, np.max(eff_ls)
    #im1 = ax1.imshow(eff_ls, origin='lower', vmin=vmin, vmax=vmax, extent=extent, cmap='RdBu_r')
    im1 = putils.plotContour(fig, ax1, eff_ls, title, fontsize=14, vmin=vmin, vmax=vmax, cmap=cmap)
    ax1.set_ylabel('$\\alpha$', fontsize=32)
    ax1.set_xlabel('$\\tau$', fontsize=32)
    
    yticks = np.linspace(0, nu-1, 6)
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(labels=['{:.1f}'.format(t/(nu-1)) for t in yticks])

    xticks = np.linspace(0, nt-1, int(tmax-tmin)+1)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(labels=['{:d}'.format(int(t*(tmax-tmin)/(nt-1) + tmin)) for t in xticks])

    for ax in axs:
        #ax.minorticks_on()
        ax.tick_params('both', length=6, width=1, which='major', labelsize=28)
        #ax.tick_params('both', length=3, width=1, which='minor')

    fig.colorbar(im1, ax=[ax1], orientation="vertical", format='%.2f')

    figfolder = os.path.join(folder, 'figs')
    if os.path.isdir(figfolder) == False:
        os.mkdir(figfolder)
    outbase = os.path.join(figfolder, '{}_strength_{}_{:.2f}_{:.2f}_{}'.format(prefix, pstates[0], pstates[-1], len(pstates), posfix))
    for ftype in ['png', 'svg']:
        plt.savefig('{}_{}_{}_heat.{}'.format(outbase, tmin, tmax, ftype), bbox_inches='tight', dpi=600)
    plt.show()



