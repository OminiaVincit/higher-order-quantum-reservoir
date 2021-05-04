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
    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--prefix', type=str, default='full_random_ridge_pinv_2021-04')
    parser.add_argument('--posfix', type=str, default='NRMSE')
    parser.add_argument('--strengths', type=str, default='0.0,0.5,0.9')
    parser.add_argument('--virtuals', type=int, default=5)
    args = parser.parse_args()
    print(args)
    folder, prefix, posfix = args.folder, args.prefix, args.posfix
    strengths = [float(x) for x in args.strengths.split(',')]
    V = args.virtuals

    orders = [5,10,15,20]
    slims  = [[1e-3, 3], [1e-2, 2], [5e-2, 3], [1e-1, 3]]
    N, M = len(orders), len(strengths)
    cmap = plt.get_cmap("viridis")
    fig, axs = plt.subplots(N, 1, figsize=(24, 6*N))
    axs = axs.ravel()
    #plt.style.use('seaborn-colorblind')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams["font.size"] = 20 # 全体のフォントサイズが変更されます
    plt.rcParams['xtick.labelsize'] = 24 # 軸だけ変更されます
    plt.rcParams['ytick.labelsize'] = 24 # 軸だけ変更されます
    
    ntitle = ''
    for i in range(N):
        order = orders[i]
        ax = axs[i]
        dcl = 0
        for j in range(M):
            alpha = strengths[j]
            for rfile in glob.glob('{}/{}*_V_{}_*narma_{}_*_{}.txt'.format(folder, prefix, V, order, posfix)):
                print(rfile)
                rsarr = []
                if 'deep_1' in rfile:
                    lb_conn = 'Forward'
                else:
                    lb_conn = 'Mutual'
                if alpha == 0.0 and lb_conn == 'Forward':
                    continue
                if alpha == 0.0:
                    lb_conn = 'Spatial'
                
                if lb_conn == 'Forward':
                    linepat = 'dashed'
                    dcl = 2*j-1
                else:
                    linepat = 'solid'
                    dcl = 2*j
                
                
                ntitle = os.path.basename(rfile)
                nidx = ntitle.find('V_{}'.format(V))
                ntitle = ntitle[nidx:]
                ntitle = ntitle.replace('.txt', '')
                rsarr = np.loadtxt(rfile)
                ids1 = (rsarr[:,3]==alpha)
                if np.sum(ids1) == 0:
                    continue
                print('narma={}'.format(order), rsarr.shape)
                
                for nqrc in [5]:
                    ids2 = (rsarr[:, 1] == nqrc)
                    ids = ids1 * ids2
                    xa, ya, za = rsarr[ids, 2], rsarr[ids, -3], rsarr[ids, -1]
                    sids = np.argsort(xa)

                    #ax.scatter(xs[ids], avg_tests[ids], label='Layers={}'.format(nqrc))
                    #ax.errorbar(xa[sids], ya[sids], yerr=za[sids], elinewidth=2, linewidth=2, markersize=12, \
                    #    label='Layers={}'.format(nqrc))
                    ax.plot(xa[sids], ya[sids], linestyle=linepat, alpha = 0.8, linewidth=3, markersize=0, mec='k', mew=0.5, \
                        color=putils.cycle[dcl], label='{}, $\\alpha$={}'.format(lb_conn, alpha))
                    ax.fill_between(xa[sids], ya[sids] - za[sids], ya[sids] + za[sids], facecolor=putils.cycle[dcl], alpha=0.2)
                    
        #ax.set_xlabel('$\\tau$', fontsize=24)
        ax.set_ylabel('NRMSE', fontsize=28)
        ax.set_yscale('log', base=10)
        ax.set_xscale('log', base=2)
        #ax.set_ylim([np.min(avg_tests)/2, 2*np.max(avg_tests)])
        #ax.set_ylim(slims[i])
        #ax.set_xticklabels(labels='')
        #ax.set_yticklabels(labels='')
        ax.set_xticks([2**x for x in np.arange(-5,7.01, 1)])
        ax.set_xlim([2**(-5), 2**7])
        
        ax.set_title('NARMA{}'.format(order))
        #ax.grid(True, which="both", ls="-", color='0.65')
        #if i == 0:
        if True:
            ax.legend()
        if i == N-1:
            ax.set_xlabel('$\\tau$', fontsize=32)
        
    for ax in axs:
        ax.minorticks_on()
        ax.tick_params('both', length=8, width=1, which='major', labelsize=28)
        ax.tick_params('both', length=4, width=1, which='minor')
    figfolder = os.path.join(folder, 'figs')
    if os.path.isdir(figfolder) == False:
        os.mkdir(figfolder)
    outbase = os.path.join(figfolder, ntitle)
    #plt.suptitle(outbase, fontsize=12)
    plt.tight_layout()
    if ntitle != '':
        for ftype in ['png', 'pdf', 'svg']:
            plt.savefig('{}_nrmse.{}'.format(outbase, ftype), bbox_inches='tight')
    plt.show()
    