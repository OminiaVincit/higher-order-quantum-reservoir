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
    parser.add_argument('--folder', type=str, default='resnarma_strength')
    parser.add_argument('--prefix', type=str, default='full_random_linear_pinv_2021-0')
    parser.add_argument('--posfix', type=str, default='nl_0_sig_1.0_bn_0_NRMSE')
    parser.add_argument('--tau', type=float, default=8.0)
    parser.add_argument('--virtuals', type=int, default=5)
    parser.add_argument('--ntrials', type=int, default=10)
    
    args = parser.parse_args()
    print(args)
    folder, prefix, posfix, tau, V, Ntrials = args.folder, args.prefix, args.posfix, args.tau, args.virtuals, args.ntrials
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
    lstype = ['o-', '--']

    for i in range(N):
        order = orders[i]
        ax = axs[i]
        for deep in [0]:
            for rfile in glob.glob('{}/{}*_V_{}_*narma_{}_deep_{}_ntrials_{}_*_{}.txt'.format(folder, prefix, V, order, deep, Ntrials, posfix)):
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
                id2 = (xs <= 1.0)
                id1 = id1 * id2
                #print(id1)
                for nqrc in [2,3,4,5]:
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
                        ax.plot(xa[sids], ya[sids], lstype[deep], color=colors[nqrc-2], alpha = 0.8, linewidth=2.0, mec='k', mew=0.5, markersize=8, label='{}'.format(nqrc))
                        ax.fill_between(xa[sids], ya[sids] - za[sids], ya[sids] + za[sids], facecolor=colors[nqrc-2], alpha=0.2)
                    
            
            ax.set_xlabel('$\\alpha$', fontsize=20)
            #ax.set_ylabel('NMSE', fontsize=14)
            #ax.set_xscale('log', base=10)
            
            ax.set_xticks(np.arange(0, 1.01, step=0.2))
            ax.set_xlim([-0.02, 1.01])
            #ax.set_ylim([1e-2, 1.0])
            #ax.set_ylim([np.min(avg_tests[id1])/1.2, 1.0])
            ax.set_ylim([np.min(avg_tests[id1])/1.1, np.max(avg_tests[id1])*1.1])
            
            ax.set_title('NARMA{}'.format(order))
            ax.grid(True, which="both", ls="-", color='0.65')
            if i == 0:
                ax.legend(loc='best')
            ax.set_yscale('log', base=10)
            
    
    for ax in axs:
        #ax.minorticks_on()
        ax.tick_params('both', length=6, width=1, which='major', labelsize=16)
        #ax.tick_params('both', length=4, width=1, which='minor')

    figsave = os.path.join(folder, 'figs')
    if os.path.isdir(figsave) == False:
        os.mkdir(figsave)
    outbase = os.path.join(figsave, '{}_{}_tau_{}'.format(prefix, ntitle.replace('.txt', ''), tau))
    #plt.suptitle(outbase, fontsize=14)
    plt.tight_layout()
    if ntitle != '':
        for ftype in ['png']:
            plt.savefig('{}_nrmse.{}'.format(outbase, ftype), bbox_inches='tight')
    plt.show()
    