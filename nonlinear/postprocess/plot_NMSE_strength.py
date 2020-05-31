import sys
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--folders', type=str, default='resnarma_deep_strength,resnarma_high_strength')
    parser.add_argument('--prefix', type=str, default='qrc_narma_2020-0')
    parser.add_argument('--posfix', type=str, default='NMSE')
    args = parser.parse_args()
    print(args)
    folders, prefix, posfix = args.folders, args.prefix, args.posfix
    folders = [str(x) for x in args.folders.split(',')]
    orders = [5,10,15,20]
    N = len(orders)
    cmap = plt.get_cmap("viridis")
    fig, axs = plt.subplots(1, N, figsize=(4*N, 10))
    axs = axs.ravel()
    #plt.style.use('seaborn-colorblind')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams['font.size']=14
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    ntitle = ''
    lstype = ['--','o-']

    for i in range(N):
        order = orders[i]
        ax = axs[i]
        for j in range(len(folders)):
            folder = folders[j]
            rsarr = []
            for rfile in glob.glob('{}/{}*_narma_{}_*_{}.txt'.format(folder, prefix, order, posfix)):
                print(rfile)
                ntitle = os.path.basename(rfile)
                nidx = ntitle.find('tdelta_')
                ntitle = ntitle[nidx:]
                tmp = np.loadtxt(rfile)
                print(tmp.shape)
                rsarr.append(tmp)

            if len(rsarr) > 0:
                rsarr = np.concatenate(rsarr, axis=0)
                print('narma={}'.format(order), rsarr.shape)
                xs, avg_tests, std_tests = rsarr[:, 3], rsarr[:, -3], rsarr[:, -1]

                id1 = (xs < 0.95)
                for nqrc in [1,2,3,4,5]:
                    ids = (rsarr[:, 0] == nqrc)
                    ids = id1 * ids
                    if nqrc == 1:
                        avg_tests[ids] = np.mean(avg_tests[ids])
                    xa, ya, za = xs[ids], avg_tests[ids], std_tests[ids]
                    sids = np.argsort(xa)

                    #ax.scatter(xs[ids], avg_tests[ids], label='Layers={}'.format(nqrc))
                    if False:
                        ax.errorbar(xa[sids], ya[sids], yerr=za[sids], elinewidth=2, linewidth=2, markersize=12, \
                            label='{}'.format(nqrc))
                    else:
                        ax.plot(xa[sids], ya[sids], lstype[j], color=colors[nqrc-1], alpha = 0.8, linewidth=2.0, mec='k', mew=0.5, markersize=8, label='{}'.format(nqrc))
            
            
            #ax.set_xlabel('Strength', fontsize=14)
            #ax.set_ylabel('NMSE', fontsize=14)
            ax.set_yscale('log', basey=10)
            ax.set_xticks(np.arange(0, 0.9001, step=0.2))
            #ax.set_ylim([np.min(avg_tests[id1])/1.2, 1.2*np.max(avg_tests[id1])])
            ax.set_title('NARMA{}'.format(order))
            ax.grid(True, which="both", ls="-", color='0.65')
            if i == N-1:
                ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    outbase = '{}\{}'.format(folders[-1], ntitle)
    plt.suptitle(outbase, fontsize=14)
    for ftype in ['pdf', 'svg']:
        plt.savefig('{}_nmse2.{}'.format(outbase, ftype), bbox_inches='tight')
    plt.show()
    