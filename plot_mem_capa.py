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
    parser.add_argument('--prefix', type=str, default='qrc_stm_2020-04')
    parser.add_argument('--posfix', type=str, default='capacity_ntrials_10')
    parser.add_argument('--strengths', type=str, default='0.0,0.5,0.9')
    parser.add_argument('--ymin', type=float, default='0.0')
    parser.add_argument('--ymax', type=float, default='90.0')

    args = parser.parse_args()
    print(args)

    folder, prefix, posfix = args.folder, args.prefix, args.posfix
    strengths = [float(x) for x in args.strengths.split(',')]
    M = len(strengths)
    ymin, ymax = args.ymin, args.ymax

    cmap = plt.get_cmap("viridis")
    fig, axs = plt.subplots(1, M, figsize=(6*M, 3))
    axs = axs.ravel()
    #plt.style.use('seaborn-colorblind')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams['font.size']=16

    ntitle = ''
    for j in range(M):
        alpha = strengths[j]
        rsarr = []
        for rfile in glob.glob('{}/{}*_strength_{}_V_15*_{}.txt'.format(folder, prefix, alpha, posfix)):
            print(rfile)
            ntitle = os.path.basename(rfile)
            nidx = ntitle.find('V_15')
            ntitle = ntitle[nidx:]
            ntitle = ntitle.replace('.txt', '')
            tmp = np.loadtxt(rfile)
            print(tmp.shape)
            rsarr.append(tmp)
        if len(rsarr) > 0:
            rsarr = np.concatenate(rsarr, axis=0)
            print('strength={}'.format(alpha), rsarr.shape)
            xs, avg_tests, std_tests = rsarr[:, 1], rsarr[:, 2], rsarr[:, 3]

            ax = axs[j]
            for nqrc in [1,2,3,4,5]:
                ids = (rsarr[:, 0] == nqrc)
                xa, ya, za = xs[ids], avg_tests[ids], std_tests[ids]
                sids = np.argsort(xa)

                #ax.scatter(xs[ids], avg_tests[ids], label='Layers={}'.format(nqrc))
                ax.errorbar(xa[sids], ya[sids], yerr=za[sids], alpha = 0.8, elinewidth=2, linewidth=2, markersize=12, \
                    label='$N_r$={}'.format(nqrc))
                #ax.plot(xa[sids], ya[sids], alpha = 0.8, linewidth=3.0, markersize=6, mec='k', mew=0.5, label='$N_r$={}'.format(nqrc))
            ax.set_xlabel('$\\tau$', fontsize=14)
            ax.set_ylabel('$C$', fontsize=14)
            ax.set_xscale('log', basex=2)
            #ax.set_ylim([np.min(avg_tests)/2, 2*np.max(avg_tests)])
            ax.set_ylim([ymin, ymax])
            #ax.set_xticks([2**n for n in range(-4, 8)])
            #ax.set_xticklabels(labels='')
            #ax.set_yticklabels(labels='')
            ax.set_title('$\\alpha$={}'.format(alpha), fontsize=16)
            ax.grid(True, which="both", ls="-", color='0.65')
            #ax.legend()
            if j == M-1:
                ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=14)
    
    outbase = '{}\{}'.format(folder, ntitle)
    plt.suptitle(outbase, fontsize=12)
    
    for ftype in ['pdf', 'svg']:
        plt.savefig('{}_capa2.{}'.format(outbase, ftype), bbox_inches='tight')
    plt.show()
    