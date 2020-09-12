import sys
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import ticker, cm

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='qesp_E_100')
    parser.add_argument('--prefix', type=str, default='qrc_echo2_2020-0')
    parser.add_argument('--posfix', type=str, default='esp')
    parser.add_argument('--strength', type=float, default=0.0)
    parser.add_argument('--eval', type=int, default=100)
    parser.add_argument('--Ts', type=str, default='10,20,50,100,200,500,1000,2000,5000,9000')
    args = parser.parse_args()
    print(args)
    folder, prefix, posfix, strength = args.folder, args.prefix, args.posfix, args.strength

    
    Ts = [int(x) for x in args.Ts.split(',')]

    cmap = plt.get_cmap("RdBu")
    fig, axs = plt.subplots(1, 1, figsize=(12, 4), squeeze=False)
    axs = axs.ravel()
    ax = axs[0]
    #plt.style.use('seaborn-colorblind')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams['font.size']=14

    ntitle = ''

    for T in Ts:
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
            ax.plot(ts, avs, label='T={}'.format(T))

    ax.set_title('{}'.format(ntitle), fontsize=16)
    ax.set_xscale("log", basex=2)
    ax.set_xlim([2**(-7),2**5])
    ax.set_yscale("log", basey=10)
    ax.set_xticks([2**x for x in np.arange(-7,5.01,1.0)])
    #ax.set_xticks([2**x for x in np.arange(0,14.1,1.0)])
    
    ax.minorticks_on()
    ax.tick_params('both', length=6, width=1, which='major')
    ax.tick_params('both', length=3, width=1, which='minor')
    ax.legend()
    #ax.set_xlim([2**(0), 2**(7)])
    outbase = os.path.join(folder, ntitle)
    for ftype in ['png', 'svg']:
        plt.savefig('{}_v1.{}'.format(outbase, ftype), bbox_inches='tight')
    plt.show()
    