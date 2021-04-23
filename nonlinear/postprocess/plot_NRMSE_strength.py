import sys
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import plot_utils as putils

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--folders', type=str, default='resnarma_deep_strength,resnarma_high_strength')
    parser.add_argument('--prefix', type=str, default='qrc_narma_ridge_pinv_2021-0')
    parser.add_argument('--posfix', type=str, default='NRMSE')
    parser.add_argument('--tau', type=float, default=2.0)
    args = parser.parse_args()
    print(args)
    folders, prefix, posfix, tau = args.folders, args.prefix, args.posfix, args.tau
    folders = [str(x) for x in args.folders.split(',')]
    orders = [5,10,15,20]
    N = len(orders)
    cmap = plt.get_cmap("viridis")
    fig, axs = plt.subplots(1, N, figsize=(4*N, 10))
    axs = axs.ravel()
    #plt.style.use('seaborn-colorblind')
    cmap = plt.get_cmap('nipy_spectral')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams["font.size"] = 20 # 全体のフォントサイズが変更されます
    plt.rcParams['xtick.labelsize'] = 24 # 軸だけ変更されます
    plt.rcParams['ytick.labelsize'] = 24 # 軸だけ変更されます
    #colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = putils.cycle
    
    ntitle = ''
    lstype = ['--','o-']

    for i in range(N):
        order = orders[i]
        ax = axs[i]
        for j in range(len(folders)):
            folder = folders[j]
            for rfile in glob.glob('{}/{}*_narma_{}_*_{}.txt'.format(folder, prefix, order, posfix)):
                print(rfile)
                ntitle = os.path.basename(rfile)
                nidx = ntitle.find('units_')
                ntitle = ntitle[nidx:]
                rsarr = np.genfromtxt(rfile)
                rsarr = np.nan_to_num(rsarr)
                print('narma={}'.format(order), rsarr.shape)
                id1 = (rsarr[:, 2] == tau)
                if (np.sum(id1) == 0):
                    continue
                xs, avg_tests, std_tests = rsarr[:, 3], rsarr[:, -3], rsarr[:, -1]
                id2 = (xs < 0.95)
                id1 = id1 * id2
                #print(id1)
                for nqrc in [1,2,3,4,5]:
                    ids = (rsarr[:, 1] == nqrc)   
                    ids = id1 * ids
                    #if nqrc == 1:
                    #    avg_tests[ids] = np.mean(avg_tests[ids])
                    xa, ya, za = xs[ids], avg_tests[ids], std_tests[ids]
                    sids = np.argsort(xa)
                    #print(nqrc, sids)
                    #ax.scatter(xs[ids], avg_tests[ids], label='Layers={}'.format(nqrc))
                    if False:
                        ax.errorbar(xa[sids], ya[sids], yerr=za[sids], elinewidth=2, linewidth=2, markersize=12, \
                            label='{}'.format(nqrc))
                    else:
                        ax.plot(xa[sids], ya[sids], lstype[j], color=colors[nqrc-1], alpha = 0.8, linewidth=2.0, mec='k', mew=0.5, markersize=8, label='{}'.format(nqrc))
                        #ax.fill_between(xa[sids], ya[sids] - za[sids], ya[sids] + za[sids], facecolor=colors[nqrc-1], alpha=0.2)
                    
            
            ax.set_xlabel('$\\alpha$', fontsize=20)
            #ax.set_ylabel('NMSE', fontsize=14)
            ax.set_yscale('log', base=10)
            ax.set_xticks(np.arange(0, 0.9001, step=0.2))
            #ax.set_ylim([np.min(avg_tests[id1])/1.2, 1.2*np.max(avg_tests[id1])])
            ax.set_title('NARMA{}'.format(order))
            ax.grid(True, which="both", ls="-", color='0.65')
            if i == N-1:
                ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    
    for ax in axs:
        #ax.minorticks_on()
        ax.tick_params('both', length=6, width=1, which='major', labelsize=16)
        #ax.tick_params('both', length=4, width=1, which='minor')

    figsave = os.path.join(os.path.dirname(rfile), 'figs')
    if os.path.isdir(figsave) == False:
        os.mkdir(figsave)
    outbase = os.path.join(figsave, '{}_tau_{}'.format(ntitle, tau))
    #plt.suptitle(outbase, fontsize=14)
    plt.tight_layout()
    for ftype in ['pdf', 'png', 'svg']:
        plt.savefig('{}_nrmse.{}'.format(outbase, ftype), bbox_inches='tight')
    plt.show()
    