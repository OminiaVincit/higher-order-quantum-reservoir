import sys
import os
import glob
import argparse
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='res_high_echo3')
    parser.add_argument('--prefix', type=str, default='qrc_echo_2020-04')
    parser.add_argument('--posfix', type=str, default='esp')
    parser.add_argument('--strengths', type=str, default='0.0,0.9')
    args = parser.parse_args()
    print(args)
    folder, prefix, posfix = args.folder, args.prefix, args.posfix
    strengths = [float(x) for x in args.strengths.split(',')]

    orders = [5,10,15,20]
    Js = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    Vs = [1, 15]
    slims  = [[1e-8, 2e-1], [2e-6, 1e-1], [2e-5, 1e-1], [2e-4, 1.2]]
    M, N = len(Vs), len(strengths)
    cmap = plt.get_cmap("viridis")
    fig, axs = plt.subplots(1, N*M, figsize=(5*M*N, 4))
    axs = axs.ravel()
    #plt.style.use('seaborn-colorblind')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams['font.size']=14

    ntitle = ''
    for j in range(M):
        V = Vs[j]
        for i in range(N):
            alpha = strengths[i]
            ax = axs[i*M+j]
            rsarr = []
            xs = []
            for J in Js:
                for rfile in glob.glob('{}/{}*_J_{}_strength_{}_V_{}_layers_5_*_{}.txt'.format(folder, prefix, J, alpha, V, posfix)):
                    print(rfile)
                    ntitle = os.path.basename(rfile)
                    nidx = ntitle.find('layers')
                    ntitle = ntitle[nidx:]
                    ntitle = ntitle.replace('.txt', '')
                    tmp = np.loadtxt(rfile)
                    print(tmp.shape)
                    rsarr.append(tmp)
                    xs.append(J)
            nx = len(rsarr)
            if len(rsarr) > 0:
                rsarr = np.concatenate(rsarr, axis=0)
                print('V={},strength={}'.format(V, alpha), rsarr.shape)
                ys, avg_esp, std_esp = rsarr[:, 2], rsarr[:, 4], rsarr[:, 5]
                
                arr_2d = avg_esp.reshape((nx, -1))
                ny = arr_2d.shape[1]
                print('nx={}, ny={}'.format(nx, ny))
                arr_2d = np.log10(arr_2d)
                df = pd.DataFrame(data=arr_2d, \
                    index=['{:.0f}'.format(np.log2(J)) for J in xs], \
                    columns=['{:.0f}'.format(np.log2(t)) for t in ys[:ny]])
            
                sns.heatmap(data=df, ax=ax, cmap='viridis')
                # print('xs shape ', len(xs), 'ys shape ', len(ys))
                # xs = xs.reshape((9, 15))
                # ys = ys.reshape((9, 15))
                # pcm = ax.pcolormesh(xs, ys, arr_2d,
                #     #    norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                #     #                           vmin=-5.0, vmax=1.0),
                #        cmap='RdBu_r', vmin=np.min(arr_2d), vmax=np.max(arr_2d))
                #ax.plot(xs, avg_esp, 'o-', alpha = 0.8, linewidth=1.5, markersize=6, mec='k', mew=0.5, label='$J$={}'.format(J))
                #ax.errorbar(xs, avg_esp, yerr=std_esp, elinewidth=2, linewidth=2, markersize=12, \
                #    label='J={}'.format(J))
            # ax.set_yscale('log', basey=10)
            # ax.set_xscale('log', basex=2)
            # #ax.set_ylim(slims[i])
            # if i == N-1 and j == M-1:
            #     ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    
    outbase = '{}\{}'.format(folder, ntitle)
    plt.suptitle(outbase, fontsize=12)
    
    #for ftype in ['pdf', 'svg']:
    #    plt.savefig('{}_nmse.{}'.format(outbase, ftype), bbox_inches='tight')
    plt.show()
    