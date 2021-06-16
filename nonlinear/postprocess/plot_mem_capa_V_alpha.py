import sys
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import plot_utils as putils
import matplotlib as mpl
from pathlib import Path

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='rescapa_high_V')
    parser.add_argument('--prefix', type=str, default='full_random_qrc_stm_linear_pinv_2021-0')
    parser.add_argument('--posfix', type=str, default='layers_5_capa_ntrials_10_capacity')
    parser.add_argument('--ymin', type=float, default='0.0')
    parser.add_argument('--ymax', type=float, default='60.0')
    parser.add_argument('--taus', type=str, default='-4,-3,-2,-1,0,1,2,3,4,5,6,7')
    
    args = parser.parse_args()
    print(args)

    folder, prefix, posfix = args.folder, args.prefix, args.posfix
    ymin, ymax = args.ymin, args.ymax
    
    tstr = args.taus.replace('\'','')
    taus_log = [float(x) for x in tstr.split(',')]
    taus = [2**x for x in taus_log]

    Vs = [1, 5, 15, 25]
    cmap = plt.get_cmap("viridis")
    fig, axs = plt.subplots(1, 1, figsize=(12, 6), squeeze=False)
    axs = axs.ravel()
    #plt.style.use('seaborn-colorblind')
    mpl.rc('font', family='serif')
    mpl.rc('mathtext', fontset='cm')
    
    mpl.rcParams["font.size"] = 20 # 全体のフォントサイズが変更されます
    mpl.rcParams['xtick.labelsize'] = 24 # 軸だけ変更されます
    mpl.rcParams['ytick.labelsize'] = 24 # 軸だけ変更されます

    ntitle = ''
    xs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    ax = axs[0]
    dcl = 0
    for tau in taus:
        for V in reversed(Vs):
            rsarr = []
            for rfile in glob.glob('{}/{}*_tau_{}*V_*{}*_{}.txt'.format(folder, prefix, tau, V, posfix)):
                print(rfile)
                ntitle = os.path.basename(rfile)
                nidx = ntitle.find('layers_')
                ntitle = ntitle[nidx:]
                ntitle = ntitle.replace('.txt', '')
                rsarr.append(np.loadtxt(rfile))
            if len(rsarr) > 0:
                rsarr = np.concatenate(rsarr, axis=0)
            else:
                continue
            print(V, tau, dcl, rsarr.shape)
            ids = (rsarr[:, 2] == V)
            xs, avg_tests, std_tests = rsarr[ids, 1], rsarr[ids, -2], rsarr[ids, -1]
            sids = np.argsort(xs)
            xs, avg_tests, std_tests = xs[sids], avg_tests[sids], std_tests[sids]

            #if dcl == 0:
            #    color = 'gray'
            #else:
            color=putils.cycle[dcl]
            #ax.errorbar(xs, avg_tests, yerr=std_tests, alpha = 0.8, elinewidth=2, linewidth=2, markersize=12, \
            #    label='$V=${}'.format(V))
            ax.plot(xs, avg_tests, 's-', alpha = 0.8, linewidth=3, markersize=8, mec='k', mew=0.5, \
                        color=color, label='$V=${}, $\\tau=${:.3f}'.format(V, tau))
            ax.fill_between(xs, avg_tests - std_tests, avg_tests + std_tests, \
                facecolor=color, alpha=0.2)
            dcl += 1

            ax.set_xlabel('$\\alpha$', fontsize=24)
            #ax.set_xscale('log',base=10)
            ax.set_ylabel('MC', fontsize=24)
            #ax.set_ylim([np.min(avg_tests)/2, 2*np.max(avg_tests)])
            ax.set_ylim([ymin, ymax])
            ax.set_xticks(np.linspace(0.0, 1.0, 11))
            ax.set_xlim([0.0, 1.01])
            #ax.set_xticklabels(labels='')
            #ax.set_yticklabels(labels='')
            ax.grid(True, which="both", ls="-", color='0.65')
            ax.legend()
            #ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=14)

    for ax in axs:
        #ax.minorticks_on()
        ax.tick_params('both', length=8, width=1, which='major', labelsize=20)
        ax.tick_params('both', length=4, width=1, which='minor')

    figsave = os.path.join(folder, 'figs')
    if os.path.isdir(figsave) == False:
        os.mkdir(figsave)

    outbase = os.path.join(figsave, '{}_{}'.format(prefix, ntitle))
    #plt.suptitle(outbase, fontsize=12)
    plt.tight_layout()
    if ntitle != '':
        for ftype in ['pdf', 'svg', 'png']:
            plt.savefig('{}_tau_{}_capa_alpha.{}'.format(outbase, tau, ftype), bbox_inches='tight')
        plt.show()
    