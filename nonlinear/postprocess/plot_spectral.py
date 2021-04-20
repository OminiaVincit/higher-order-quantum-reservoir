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
    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--nspins', type=int, default=5, help='Number of spins')
    parser.add_argument('--tmin', type=float, default=-7.0, help='Minimum of tau')
    parser.add_argument('--tmax', type=float, default=5.0, help='Maximum of tau')
    parser.add_argument('--ntaus', type=int, default=121, help='Number of taus')
    parser.add_argument('--basename', type=str, default='spec')
    
    args = parser.parse_args()
    print(args)
    Nspins, tmin, tmax, ntaus = args.nspins, args.tmin, args.tmax, args.ntaus
    basename = '{}_nspins_{}_log2_tmin_{:.2f}_tmax_{:.2f}_ntaus_{}'.format(args.basename, Nspins, tmin, tmax, ntaus)
    folder    = args.folder
    bindir = os.path.join(folder, 'binary')

    ild2, ld23 = [], []
    tx = list(np.linspace(tmin, tmax, ntaus))
    tauls = [2**x for x in tx]

    for seed in range(10):
        local_ild2, local_ld23 = [], []
        for pstate in np.linspace(0.0, 1.0, 101):
            filename = os.path.join(bindir, '{}_eig_pstate_{:.2f}_seed_{}.binary'.format(basename, pstate, seed))
            if os.path.isfile(filename) == False:
                print('Not found {}'.format(filename))
                continue
            
            t1, t2 = [], []
            with open(filename, 'rb') as rrs:
                z = pickle.load(rrs)
            #print(z.keys())
            for taub in z.keys():
                egvals = z[taub]
                egvals = sorted(egvals, key=abs, reverse=True)
                la = 1.0/np.abs(egvals[1])
                lb = np.abs(egvals[1])/np.abs(egvals[2])
                t1.append([la])
                t2.append([lb])
            t1, t2 = np.array(t1).ravel(), np.array(t2).ravel()
            local_ild2.append(t1)
            local_ld23.append(t2)
        local_ild2 = np.array(local_ild2)
        local_ld23 = np.array(local_ld23)
        ild2.append(local_ild2)
        ld23.append(local_ld23)
    ild2 = np.mean(np.array(ild2), axis=0)
    ld23 = np.mean(np.array(ld23), axis=0)
    print(ild2.shape, ld23.shape)    
    
    # Plot file
    cmap = plt.get_cmap('nipy_spectral')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams["font.size"] = 20 # 全体のフォントサイズが変更されます
    plt.rcParams['xtick.labelsize'] = 24 # 軸だけ変更されます
    plt.rcParams['ytick.labelsize'] = 24 # 軸だけ変更されます
    fig = plt.figure(figsize=(24, 6), dpi=600)
    #fig.suptitle(basename, fontsize=16, horizontalalignment='left')
    
    # Plot Nspins largest eigenvectors
    ax1 = plt.subplot2grid((1,1), (0,0), colspan=1, rowspan=1)
    im1 = putils.plotContour(fig, ax1, ild2, '$1/|\lambda_2|$ - {}'.format(basename), fontsize=16, vmin=None, vmax=None, cmap=cmap)
    #im1 = ax1.imshow(ild2, origin='lower', vmin=vmin, vmax=vmax, extent=extent)
    ax1.set_ylabel('$u$', fontsize=24)
    ax1.set_xlabel('$\\tau$', fontsize=24)
    #ax1.set_xticks(list(range(extent[0], extent[1] + 1)))
    
    plt.tight_layout()
    # call subplot adjust after tight_layout
    #plt.subplots_adjust(hspace=0.0)
    fig.colorbar(im1, ax=[ax1], orientation="vertical", format='%.2f')
    
    figsave = os.path.join(folder, 'figs')
    if os.path.isdir(figsave) == False:
        os.mkdir(figsave)

    outbase = os.path.join(figsave, basename)

    for ftype in ['png']:
        plt.savefig('{}_heat.{}'.format(outbase, ftype), bbox_inches='tight', dpi=600)
    plt.show()



