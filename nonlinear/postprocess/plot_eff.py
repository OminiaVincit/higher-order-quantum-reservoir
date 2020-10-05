import sys
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='res_high_eff_tau')
    parser.add_argument('--prefix', type=str, default='qrc_eff_2020-10-0')
    parser.add_argument('--posfix', type=str, default='V_1_layers_5_eff_ntrials_10_eff')
    
    args = parser.parse_args()
    print(args)
    folder, prefix, posfix = args.folder, args.prefix, args.posfix
    
    pstates = [i/100 for i in range(101)]
    ild2, ld23 = [], []
    eff_ls = []
    #qrc_eff_2020-10-05-06-06-53_strength_0.78_V_1_layers_5_eff_ntrials_10_eff


    lexp1, lexp2 = -4, 5
    for p in pstates:
        for filename in glob.glob('{}/{}*_strength_{:.2f}_{}.txt'.format(folder, prefix, p, posfix)):
            rs_arr = np.loadtxt(filename)
            print(p, rs_arr.shape)
            t1 = np.array(rs_arr[:,3]).ravel()
            t1 = t1[rs_arr[:,1] >= (2**lexp1)]
            eff_ls.append(t1)

    eff_ls = np.array(eff_ls)
    
    # Plot file
    plt.rc('font', family='serif', size=14)
    plt.rc('mathtext', fontset='cm')
    fig = plt.figure(figsize=(24, 6), dpi=600)
    vmin, vmax = 1.0, np.max(eff_ls)
    extent = [lexp1, lexp2, 0, 3]
    # Plot Nspins largest eigenvectors
    ax1 = plt.subplot2grid((2,1), (0,0), colspan=1, rowspan=1)
    ax1.set_title('{}'.format(os.path.basename(filename)), size=8)
    im1 = ax1.imshow(eff_ls, origin='lower', vmin=vmin, vmax=vmax, extent=extent, cmap='RdBu_r')
    ax1.set_ylabel('$\\alpha$', fontsize=16)
    ax1.set_xticks(list(range(extent[0], extent[1] + 1)))
    
    urange = np.linspace(extent[2], extent[3], 6)
    vrange = ['{:.1f}'.format(x) for x in np.linspace(0, 1, 6)]
    ax1.set_yticks(urange)
    ax1.set_yticklabels(vrange)
    ax1.set_xlabel('$\log_2(\\tau)$', fontsize=16)
    fig.colorbar(im1)

    outbase = filename.replace('.txt', '')
    for ftype in ['png', 'svg']:
        plt.savefig('{}_{}_{}_heat.{}'.format(outbase, lexp1, lexp2, ftype), bbox_inches='tight', dpi=600)
    plt.show()



