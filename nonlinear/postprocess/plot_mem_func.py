import sys
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Plot for 2020-05-01
filenames = ['qrc_stm_2020-04-21-15-46-27_strength_0.0_V_1_layers_1_mem_ntrials_10_memfunc.npz',\
    'qrc_stm_2020-04-22-02-30-21_strength_0.0_V_1_layers_5_mem_ntrials_10_memfunc.npz',\
    'qrc_stm_2020-04-22-10-27-17_strength_0.9_V_1_layers_5_mem_ntrials_10_memfunc.npz']

MEM_FUNC_DATA='rescapa_highfunc_stm'

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default=MEM_FUNC_DATA)
    parser.add_argument('--ymin', type=float, default='0.0')
    parser.add_argument('--ymax', type=float, default='1.0')

    args = parser.parse_args()
    print(args)

    folder = args.folder
    M = len(filenames)
    ymin, ymax = args.ymin, args.ymax
    taudeltas = [2**float(x) for x in [-4,-3,-2,-1,0,1,2,3,4,5,6,7]]

    cmap = plt.get_cmap("viridis")
    fig, axs = plt.subplots(1, M, figsize=(6*M, 3))
    axs = axs.ravel()
    #plt.style.use('seaborn-colorblind')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams['font.size']=16

    ntitle = ''
    for j in range(M):
        rfile = os.path.join(folder, filenames[j])
        print(rfile)

        ntitle = os.path.basename(rfile)
        nidx = ntitle.find('V_')
        ntitle = ntitle[nidx:]
        ntitle = ntitle.replace('.txt', '')
        rsarr = np.load(rfile)

        ax = axs[j]
        for tau_delta in taudeltas:
            tarr = rsarr[str(tau_delta)]
            xs, ys, zs = tarr[:, 0], tarr[:, 1], tarr[:, 2]
            ax.plot(xs, ys, linewidth=2, markersize=12, \
                label='$\\tau$ = {}'.format(tau_delta))
        
        ax.set_xlabel('$d$', fontsize=14)
        ax.set_ylabel('$C(d)$', fontsize=14)
        #ax.set_ylim([np.min(avg_tests)/2, 2*np.max(avg_tests)])
        ax.set_ylim([ymin, ymax])
        #ax.set_xticks([2**n for n in range(-4, 8)])
        #ax.set_xticklabels(labels='')
        #ax.set_yticklabels(labels='')
        #ax.set_title('$\\alpha$={}'.format(alpha), fontsize=16)
        ax.grid(True, which="both", ls="-", color='0.65')
        #ax.legend()
        if j == M-1:
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    
    outbase = '{}\{}'.format(folder, ntitle)
    plt.suptitle(outbase, fontsize=12)
    
    for ftype in ['pdf', 'svg']:
        plt.savefig('{}_func.{}'.format(outbase, ftype), bbox_inches='tight')
    plt.show()
    