import sys
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import ticker, cm
import plot_utils as putils

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='../../../data/hqrc/qesp_E_1000')
    parser.add_argument('--prefix', type=str, default='qesp_full_random')
    parser.add_argument('--posfix', type=str, default='esp_trials_10_10_esp')
    parser.add_argument('--strengths', type=str, default='0.0,0.5,0.9')
    parser.add_argument('--eval', type=int, default=1000)
    parser.add_argument('--Ts', type=str, default='1000,5000,10000')
    args = parser.parse_args()
    print(args)
    folder, prefix, posfix = args.folder, args.prefix, args.posfix
    Ts = [int(x) for x in args.Ts.split(',')]
    strengths = [float(x) for x in args.strengths.split(',')]

    cmap = plt.get_cmap("Spectral")
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams["font.size"] = 20 # 全体のフォントサイズが変更されます
    plt.rcParams['xtick.labelsize'] = 24 # 軸だけ変更されます
    plt.rcParams['ytick.labelsize'] = 24 # 軸だけ変更されます

    fig, axs = plt.subplots(2, 1, figsize=(24, 15), squeeze=False)
    axs = axs.ravel()
    ax1, ax2 = axs[0], axs[1]
    colors = putils.cycle

    # Plot diff curve with T
    ntitle = ''
    esp_bt = []
    lstyles = ['solid', 'dashed', 'dotted']
    for i in range(len(Ts)):
        T = Ts[i]
        color = colors[i]
        for j in range(len(strengths)):
            strength = strengths[j]
            lstyle = lstyles[j]
            for rfile in glob.glob('{}/{}*_strength_{}_*T_{}_{}*{}.txt'.format(folder, prefix, \
                strength, T, T+args.eval, posfix)):
                if os.path.isfile(rfile) == False:
                    continue
                print(rfile)
                ntitle = os.path.basename(rfile)
                nidx = ntitle.find('strength_')
                ntitle = ntitle[nidx:]
                ntitle = ntitle.replace('.txt', '')
                tmp = np.loadtxt(rfile)
                print(tmp.shape)
                ts, avs, stds = tmp[:, 2], tmp[:, -2], tmp[:, -1]
                ax1.plot(ts, avs, linestyle=lstyle, label='$T=${}$,\\alpha=${}'.format(T,strength), linewidth=3, alpha=0.8, color=color)
                #ax1.fill_between(ts, avs - stds, avs + stds, facecolor='k', alpha=0.2)
                        
                esp_bt.append(np.array(avs))
    esp_bt = np.array(esp_bt)

    ax1.set_title('{}'.format(ntitle), fontsize=16)
    ax1.set_xscale("log", basex=2)
    ax1.set_yscale("log", basey=10)
    ax1.set_xticks([2**x for x in np.arange(-7,7.01,1.0)])
    ax1.set_xlim([2**(-7),2**5])
    
    #ax.set_xticks([2**x for x in np.arange(0,14.1,1.0)])
    ax1.legend()
    
    # Plot diff heatmap
    im1 = putils.plotContour(fig, ax2, np.log10(esp_bt), 'QESP index', fontsize=16, vmin=None, vmax=None, cmap=cmap)

    for ax in axs:
        ax.minorticks_on()
        ax.tick_params('both', length=6, width=1, which='major', labelsize=28)
        ax.tick_params('both', length=3, width=1, which='minor')
    
    plt.tight_layout()
    fig.colorbar(im1, ax=[ax2], orientation="horizontal", format='%.2f')
    figfolder = os.path.join(folder, 'figs')
    if os.path.isdir(figfolder) == False:
        os.mkdir(figfolder)
    if ntitle != '':
        outbase = os.path.join(figfolder, ntitle)
        for ftype in ['png', 'svg']:
            plt.savefig('{}_v1.{}'.format(outbase, ftype), bbox_inches='tight')
        plt.show()
    