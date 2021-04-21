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
    parser.add_argument('--folder', type=str, default='rescapa_high_V')
    parser.add_argument('--prefix', type=str, default='qrc_stm_ridge_pinv_2021-0')
    parser.add_argument('--posfix', type=str, default='layers_5_capa_ntrials_10_capacity')
    parser.add_argument('--ymin', type=float, default='0.0')
    parser.add_argument('--ymax', type=float, default='40.0')

    args = parser.parse_args()
    print(args)

    folder, prefix, posfix = args.folder, args.prefix, args.posfix
    ymin, ymax = args.ymin, args.ymax
    Vs = [1, 5, 15, 25]
    cmap = plt.get_cmap("viridis")
    fig, axs = plt.subplots(1, 1, figsize=(12, 6), squeeze=False)
    axs = axs.ravel()
    #plt.style.use('seaborn-colorblind')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams["font.size"] = 20 # 全体のフォントサイズが変更されます
    plt.rcParams['xtick.labelsize'] = 24 # 軸だけ変更されます
    plt.rcParams['ytick.labelsize'] = 24 # 軸だけ変更されます

    ntitle = ''
    xs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    ax = axs[0]
    dcl = 0
    for V in Vs:
        for rfile in glob.glob('{}/{}*_V_*{}*_{}.txt'.format(folder, prefix, V, posfix)):
            print(rfile)
            ntitle = os.path.basename(rfile)
            nidx = ntitle.find('layers_')
            ntitle = ntitle[nidx:]
            ntitle = ntitle.replace('.txt', '')
            rsarr = np.loadtxt(rfile)
            print(V, dcl, rsarr.shape)
            ids = (rsarr[:, 2] == V)
            xs, avg_tests, std_tests = rsarr[ids, 1], rsarr[ids, -2], rsarr[ids, -1]
            
            #ax.errorbar(xs, avg_tests, yerr=std_tests, alpha = 0.8, elinewidth=2, linewidth=2, markersize=12, \
            #    label='$V=${}'.format(V))
            ax.plot(xs, avg_tests, 'o-', alpha = 0.8, linewidth=3, markersize=0, mec='k', mew=0.5, \
                        color=putils.cycle[dcl], label='$V=${}'.format(V))
            ax.fill_between(xs, avg_tests - std_tests, avg_tests + std_tests, \
                facecolor=putils.cycle[dcl], alpha=0.2)
            dcl += 1

            ax.set_xlabel('$\\alpha$', fontsize=24)
            ax.set_ylabel('MC', fontsize=24)
            #ax.set_ylim([np.min(avg_tests)/2, 2*np.max(avg_tests)])
            ax.set_ylim([ymin, ymax])
            ax.set_xticks(xs)
            ax.set_xlim([xs[0], xs[-1]])
            #ax.set_xticklabels(labels='')
            #ax.set_yticklabels(labels='')
            ax.grid(True, which="both", ls="-", color='0.65')
            ax.legend()
            #ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=14)
    for ax in axs:
        #ax.minorticks_on()
        ax.tick_params('both', length=8, width=1, which='major', labelsize=24)
        ax.tick_params('both', length=4, width=1, which='minor')

    figsave = os.path.join(folder, 'figs')
    if os.path.isdir(figsave) == False:
        os.mkdir(figsave)

    outbase = os.path.join(figsave, ntitle)
    #plt.suptitle(outbase, fontsize=12)
    plt.tight_layout()
    if ntitle != '':
        for ftype in ['pdf', 'svg', 'png']:
            plt.savefig('{}_capa_alpha.{}'.format(outbase, ftype), bbox_inches='tight')
        plt.show()
    