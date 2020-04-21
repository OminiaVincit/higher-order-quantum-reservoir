import sys
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--prefix', type=str, default='qrc_narma_2020-04')
    parser.add_argument('--posfix', type=str, default='NMSE')
    parser.add_argument('--strength', type=float, default=0.0)
    args = parser.parse_args()
    print(args)
    folder, prefix, posfix = args.folder, args.prefix, args.posfix
    alpha = args.strength

    orders = [5,10,15,20]
    slims  = [[2e-8, 5e-3], [2e-6, 5e-3], [3e-5, 5e-3], [2e-4, 5e-3]]
    N = len(orders)
    cmap = plt.get_cmap("viridis")
    fig, axs = plt.subplots(N, 1, figsize=(3*N, 16))
    axs = axs.ravel()
    #plt.style.use('seaborn-colorblind')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams['font.size']=14

    ntitle = ''
    for i in range(N):
        order = orders[i]
        rsarr = []
        for rfile in glob.glob('{}/{}*_strength_{}_tdt_2.0_narma_{}_*_{}.txt'.format(folder, prefix, alpha, order, posfix)):
            print(rfile)
            ntitle = os.path.basename(rfile)
            nidx = ntitle.find('strength_')
            ntitle = ntitle[nidx:]
            ntitle = ntitle.replace('.txt', '')
            tmp = np.loadtxt(rfile)
            print(tmp.shape)
            rsarr.append(tmp)
        if len(rsarr) > 0:
            rsarr = np.concatenate(rsarr, axis=0)
            print('narma={}'.format(order), rsarr.shape)
            xs, avg_tests, std_tests = rsarr[:, 1], rsarr[:, 4], rsarr[:, 6]

            ax = axs[i]
            for nqrc in [1,2,3,4,5]:
                ids = (rsarr[:, 0] == nqrc)
                xa, ya, za = xs[ids], avg_tests[ids], std_tests[ids]
                sids = np.argsort(xa)

                #ax.scatter(xs[ids], avg_tests[ids], label='Layers={}'.format(nqrc))
                #ax.errorbar(xa[sids], ya[sids], yerr=za[sids], elinewidth=2, linewidth=2, markersize=12, \
                #    label='Layers={}'.format(nqrc))
                ax.plot(xa[sids], ya[sids], 'o--', alpha = 0.8, linewidth=1.5, markersize=8, mec='k', mew=0.5, label='Layers={}'.format(nqrc))
            ax.set_xlabel('$V$', fontsize=14)
            ax.set_ylabel('NMSE', fontsize=14)
            ax.set_yscale('log', basey=10)
            #ax.set_xscale('log', basex=2)
            #ax.set_ylim([np.min(avg_tests)/2, 2*np.max(avg_tests)])
            ax.set_ylim(slims[i])
            ax.set_title('NARMA{}'.format(order))
            ax.grid(True, which="both", ls="-", color='0.65')
            if i == 0:
                ax.legend()
    outbase = '{}\{}'.format(folder, ntitle)
    plt.suptitle(outbase, fontsize=12)
    for ftype in ['pdf', 'svg']:
        plt.savefig('{}_nmse.{}'.format(outbase, ftype), bbox_inches='tight')
    plt.show()
    