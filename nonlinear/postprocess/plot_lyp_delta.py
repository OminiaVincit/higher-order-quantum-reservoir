import sys
import os
import glob
import argparse
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import ticker, cm

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='res_high_lyp')
    parser.add_argument('--prefix', type=str, default='qrc_lyp_2020-05')
    parser.add_argument('--posfix', type=str, default='lyp')
    parser.add_argument('--virtuals', type=str, default='1')
    parser.add_argument('--strengths', type=str, default='0.1,0.5,0.9')
    parser.add_argument('--vmin', type=int, default=-5)
    parser.add_argument('--vmax', type=int, default=5)
    args = parser.parse_args()
    print(args)
    folder, prefix, posfix = args.folder, args.prefix, args.posfix
    strengths = [float(x) for x in args.strengths.split(',')]

    Js = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    Vs = [int(v) for v in args.virtuals.split(',')]
    slims  = [[1e-8, 2e-1], [2e-6, 1e-1], [2e-5, 1e-1], [2e-4, 1.2]]
    M, N = len(Vs), len(strengths)
    cmap = plt.get_cmap("BrBG")
    fig, axs = plt.subplots(1, N*M, figsize=(5*M*N, 4))
    axs = axs.ravel()
    #plt.style.use('seaborn-colorblind')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams['font.size']=14

    ntitle = ''

    for j in range(M):
        V = Vs[j]
        rsarr = []
        for J in Js:
            for rfile in glob.glob('{}/{}*_J_{}_*_V_{}_layers_5_*_{}.txt'.format(folder, prefix, J, V, posfix)):
                print(rfile)
                ntitle = os.path.basename(rfile)
                nidx = ntitle.find('layers')
                ntitle = ntitle[nidx:]
                ntitle = ntitle.replace('.txt', '')
                tmp = np.loadtxt(rfile)
                print(tmp.shape)
                rsarr.append(tmp[:, [2, 4, -2, -1]]) # tau, alpha, avg, std

        if len(rsarr) > 0:
            rsarr = np.concatenate(rsarr, axis=0)
            print('rsarr shape', rsarr.shape)

        for i in range(N):
            alpha = strengths[i]
            ax = axs[i*M+j]
            ids = (rsarr[:, 1] == alpha)
            arr = rsarr[ids,:]
            print('V={},strength={}'.format(V, alpha), arr.shape)
            ys, avg_esp, std_esp = arr[:, 0], arr[:, -2], arr[:, -1]
            
            arr_2d = avg_esp.reshape((len(Js), -1))
            nx, ny = arr_2d.shape
            print('nx={}, ny={}'.format(nx, ny))
            
            df = pd.DataFrame(data=arr_2d, \
                index=['{:.0f}'.format(np.log2(J)) for J in Js], \
                columns=['{:.0f}'.format(np.log2(t)) for t in ys[:ny]])
        
            sns.heatmap(data=df, ax=ax, cmap=cmap, vmin=args.vmin, vmax=args.vmax)

            ax.set_title('V={},$\\alpha$={}'.format(V,alpha), fontsize=16)
    
    outbase = '{}\{}'.format(folder, ntitle)
    plt.suptitle(outbase, fontsize=12)
    
    for ftype in ['pdf', 'svg']:
        plt.savefig('{}_lyp.{}'.format(outbase, ftype), bbox_inches='tight')
    plt.show()
    