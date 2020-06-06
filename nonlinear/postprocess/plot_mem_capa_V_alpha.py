import sys
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='rescapa_high_V')
    parser.add_argument('--prefix', type=str, default='qrc_stm_2020-0')
    parser.add_argument('--posfix', type=str, default='layers_5_capa_ntrials_10_capacity')
    parser.add_argument('--ymin', type=float, default='0.0')
    parser.add_argument('--ymax', type=float, default='90.0')

    args = parser.parse_args()
    print(args)

    folder, prefix, posfix = args.folder, args.prefix, args.posfix
    ymin, ymax = args.ymin, args.ymax
    Vs = [1, 5, 10, 15, 25]
    cmap = plt.get_cmap("viridis")
    fig, axs = plt.subplots(1, 3, figsize=(6*3, 3))
    axs = axs.ravel()
    #plt.style.use('seaborn-colorblind')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams['font.size']=16

    ntitle = ''
    xs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    ax = axs[0]
    for V in Vs:
        rsarr = []
        for rfile in glob.glob('{}/{}*_V_{}_{}.txt'.format(folder, prefix, V, posfix)):
            print(rfile)
            ntitle = os.path.basename(rfile)
            nidx = ntitle.find('layers_')
            ntitle = ntitle[nidx:]
            ntitle = ntitle.replace('.txt', '')
            tmp = np.loadtxt(rfile)
            print(tmp.shape)
            rsarr.append(tmp)
        if len(rsarr) > 0:
            rsarr = np.concatenate(rsarr, axis=0)
            print(rsarr.shape)
            avg_tests, std_tests = rsarr[:, -2], rsarr[:, -1]
            
            ax.errorbar(xs, avg_tests, yerr=std_tests, alpha = 0.8, elinewidth=2, linewidth=2, markersize=12, \
                label='$V=${}'.format(V))
            ax.set_xlabel('$\\alpha$', fontsize=14)
            ax.set_ylabel('MC', fontsize=14)
            #ax.set_ylim([np.min(avg_tests)/2, 2*np.max(avg_tests)])
            ax.set_ylim([ymin, ymax])
            #ax.set_xticks([2**n for n in range(-4, 8)])
            #ax.set_xticklabels(labels='')
            #ax.set_yticklabels(labels='')
            ax.grid(True, which="both", ls="-", color='0.65')
            ax.legend()
            #ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=14)
    
    outbase = '{}\{}'.format(folder, ntitle)
    plt.suptitle(outbase, fontsize=12)
    
    for ftype in ['pdf', 'svg', 'png']:
        plt.savefig('{}_capa_alpha.{}'.format(outbase, ftype), bbox_inches='tight')
    plt.show()
    