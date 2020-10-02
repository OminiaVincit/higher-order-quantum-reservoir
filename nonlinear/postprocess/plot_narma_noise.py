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
    parser.add_argument('--folder', type=str, default='../de-narma3')
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
    orders = [5, 10, 15, 20]
    for i in range(len(orders)):
        order = orders[i]
        cl = colors[i]
        nmse = dict()
        for rate in ['0.0', '10.0']:
            xs, ys = [], []
            for noise in noise_ls:
                for rfile in glob.glob('{}/log/{}*_narma_{}_*_no_{}_r_{}_{}.log'.format(folder, prefix, order, noise, rate, posfix)):
                    with open(rfile, 'r') as rf:
                        lines = rf.readlines()
                        for line in lines:
                            line = line.strip()
                            if 'total loss avg' in line:
                                avg = (float)(re.search('avg=([0-9.]*)', line).group(1))
                                xs.append(noise)
                                ys.append(avg)
                                break
            nmse[rate] = ys
        rels = []
        for i in range(len(nmse['0.0'])):
            pre_nmse = nmse['0.0'][i]
            post_nmse = nmse['10.0'][i]
            rels.append( 100*(pre_nmse - post_nmse)/pre_nmse )
        #ax.plot(xs, rels, 'o--', linewidth=2, label='{}'.format(order))
        ax.plot(xs, nmse['0.0'], 'o--', color=cl, linewidth=2, label='pre-{}'.format(order))
        ax.plot(xs, nmse['10.0'], '*--', color=cl, linewidth=2, label='post-{}'.format(order))
    ax.set_xscale("log", basex=10)
    ax.grid(which='major',color='black',linestyle='-')
    #ax.grid(which='minor',color='black',linestyle='-')
    ax.legend()
    ax.set_ylabel('NMSE')
    ax.set_xlabel('Noise variance')

    for ftype in ['png']:
        plt.savefig('{}.{}'.format(outbase, ftype), bbox_inches='tight', dpi=600)
    plt.show()
            
    