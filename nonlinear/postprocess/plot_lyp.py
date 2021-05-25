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
    parser.add_argument('--folder', type=str, default='res_lyp_tau')
    parser.add_argument('--prefix', type=str, default='full_random_nqrc_5')
    parser.add_argument('--posfix', type=str, default='lyp_trials_10_dist_1e-05_lyp')
    parser.add_argument('--sparsity', type=float, default=1.0)
    parser.add_argument('--virtuals', type=int, default=1)
    parser.add_argument('--vmin', type=float, default=1.0)
    parser.add_argument('--vmax', type=float, default=1.0)
    
    args = parser.parse_args()
    print(args)
    folder, prefix, posfix, V = args.folder, args.prefix, args.posfix, args.virtuals
    prefix = '{}_V_{}'.format(prefix, V)
    posfix = 'sparse_{}_{}'.format(args.sparsity, posfix)

    #pstates = np.linspace(0.0, 1.0, 101)
    #pstates = pstates[:-1]
    pstates = [0.1, 0.3, 0.5, 0.7, 0.9]
    eff_ls = []

    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams["font.size"] = 20 # 全体のフォントサイズが変更されます
    plt.rcParams['xtick.labelsize'] = 24 # 軸だけ変更されます
    plt.rcParams['ytick.labelsize'] = 24 # 軸だけ変更されます

    fig, axs = plt.subplots(1, 1, figsize=(24, 10), squeeze=False)
    axs = axs.ravel()
    ax1 = axs[0]
    colors = putils.cycle

    tmin, tmax = -7, 7
    for i in range(len(pstates)):
        p = pstates[i]
        color = colors[i]
        for filename in glob.glob('{}/{}_a_{:.2f}*_{}.txt'.format(folder, prefix, p, posfix)):
            rs_arr = np.loadtxt(filename)
            print(p, rs_arr.shape)
            tauls =  np.array(rs_arr[:,2]).ravel()
            ids = (tauls >= (2**tmin)) * (tauls <= (2**tmax))
            tauls = tauls[ids]

            avg_lyp = np.array(rs_arr[:,-2]).ravel()
            avg_lyp = avg_lyp[ids]
        
            std_lyp = np.array(rs_arr[:,-1]).ravel()
            std_lyp = std_lyp[ids]

            ax1.plot(tauls, avg_lyp, label='$\\alpha$={:.1f}'.format(p), linewidth=3, alpha=0.8, color=color)
            ax1.fill_between(tauls, avg_lyp - std_lyp, avg_lyp + std_lyp, \
                facecolor=color, alpha=0.1)

            eff_ls.append(avg_lyp)

    title = '{}'.format(os.path.basename(filename))
    ax1.set_title('{}'.format(title), fontsize=16)
    ax1.set_xscale("log", basex=2)
    ax1.set_xticks([2**x for x in np.arange(-7,7.01,1.0)])
    ax1.set_xlim([2**tmin,2**tmax])

    ax1.legend()
    ax1.grid(True, which="both", ls="-", color='0.65')
    ax1.axhline(y=0.0, color='gray', linestyle='-', linewidth=5)

    if False:
        eff_ls = np.array(eff_ls)
        print(eff_ls.shape)
        (nu, nt) = eff_ls.shape
        # Plot file
        putils.setPlot(fontsize=24, labelsize=24)
        #cmap = plt.get_cmap('nipy_spectral')
        cmap = plt.get_cmap('RdBu_r')
        #cmap = plt.get_cmap('gist_rainbow_r')
        #cmap = plt.get_cmap('gist_ncar')
        #cmap = plt.get_cmap('gist_stern')

        
        vmin, vmax = args.vmin, args.vmax
        if vmin >= vmax:
            vmin, vmax = np.min(eff_ls), np.max(eff_ls)
        #im1 = ax1.imshow(eff_ls, origin='lower', vmin=vmin, vmax=vmax, extent=extent, cmap='RdBu_r')
        im1 = putils.plotContour(fig, ax2, eff_ls, title, fontsize=14, vmin=vmin, vmax=vmax, cmap=cmap)
        ax2.set_ylabel('$\\alpha$', fontsize=32)
        ax2.set_xlabel('$\\tau$', fontsize=32)
        
        yticks = np.linspace(0, nu-1, 6)
        ax2.set_yticks(yticks)
        ax2.set_yticklabels(labels=['{:.1f}'.format(t/(nu-1)) for t in yticks])

        xticks = np.linspace(0, nt-1, int(tmax-tmin)+1)
        ax2.set_xticks(xticks)
        ax2.set_xticklabels(labels=['{:d}'.format(int(t*(tmax-tmin)/(nt-1) + tmin)) for t in xticks])
        fig.colorbar(im1, ax=[ax2], orientation="vertical", format='%.2f')

    for ax in axs:
        #ax.minorticks_on()
        ax.tick_params('both', length=8, width=1, which='major', labelsize=28)
        #ax.tick_params('both', length=3, width=1, which='minor')

    
    figfolder = os.path.join(folder, 'figs')
    if os.path.isdir(figfolder) == False:
        os.mkdir(figfolder)
    outbase = os.path.join(figfolder, '{}_a_{}_{:.2f}_{:.2f}_{}'.format(prefix, pstates[0], pstates[-1], len(pstates), posfix))
    for ftype in ['png', 'svg']:
        plt.savefig('{}_{}_{}_heat.{}'.format(outbase, tmin, tmax, ftype), bbox_inches='tight', dpi=600)
    plt.show()



