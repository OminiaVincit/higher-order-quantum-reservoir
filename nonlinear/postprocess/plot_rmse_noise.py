import sys
import os
import glob
import argparse
import numpy as np
import re
import matplotlib.pyplot as plt

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='denoise_rmse')
    parser.add_argument('--prefix', type=str, default='qrc_innate_linear_pinv_2020-1')
    parser.add_argument('--posfix', type=str, default='sc_0.2_sd_0')
    args = parser.parse_args()
    print(args)

    folder, prefix, posfix = args.folder, args.prefix, args.posfix
    figdir = os.path.join(folder, 'figs')
    if os.path.isdir(figdir) == False:
        os.mkdir(figdir)
    outbase = os.path.join(figdir, '{}_{}'.format(prefix, posfix))

    
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(1, 1, 1)
    #plt.style.use('seaborn-colorblind')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams['font.size']=16
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    noise_ls = []
    for i in [-6, -5, -4]:
        for j in range(1, 10):
            noise_ls.append(j * (10**i))
    noise_ls.append(1e-3)
    taus = [1.0, 2.0, 4.0]

    nmse = dict()
    for rate in ['0.0', '10.0']:
        xs, ys = [], []
        for noise in noise_ls:
            for rfile in glob.glob('{}/log/{}*_narma_10_*_no_{}_r_{}_{}.log'.format(folder, prefix, noise, rate, posfix)):
                with open(rfile, 'r') as rf:
                    lines = rf.readlines()
                    local_loss = []
                    for line in lines:
                        line = line.strip()
                        if 'total loss avg' in line:
                            avg = (float)(re.search('avg=([0-9.]*)', line).group(1))
                            local_loss.append(avg)
                    xs.append(noise)
                    ys.append(np.mean(local_loss))
        nmse[rate] = ys
    rels = []
    for i in range(len(nmse['0.0'])):
        pre_nmse = nmse['0.0'][i]
        post_nmse = nmse['10.0'][i]
        rels.append( 100*(pre_nmse - post_nmse)/pre_nmse )
    #ax.plot(xs, rels, 'o--', linewidth=2, label='Relative')
    ax.plot(xs, nmse['0.0'], '*--', linewidth=2, label='pre', markersize=10)
    ax.plot(xs, nmse['10.0'], 'o-', linewidth=2, label='post', markersize=10)
    ax.set_xscale("log", basex=10)
    ax.set_yscale("log", basey=10)
    #ax.set_ylim([2*10**(-3), 10**(-1)])
    ax.grid(which='major',color='black',linestyle='-')
    #ax.grid(axis='y',which='minor',color='black',linestyle='-')
    ax.tick_params('both', length=15, width=1, which='major')
    ax.tick_params('both', length=10, width=1, which='minor')
    ax.legend()
    ax.set_ylabel('RNMSE')
    ax.set_xlabel('Noise variance')

    for ftype in ['png', 'svg']:
        plt.savefig('{}.{}'.format(outbase, ftype), bbox_inches='tight', dpi=600)
    plt.show()
            
    