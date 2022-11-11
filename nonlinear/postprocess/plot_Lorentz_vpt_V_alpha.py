import sys
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import plot_utils as putils
import matplotlib as mpl
from pathlib import Path
import re

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='lorentz')
    parser.add_argument('--prefix', type=str, default='lorentz_phase_trans')
    parser.add_argument('--posfix', type=str, default='T_20_100_100_seed_0')
    parser.add_argument('--ymin', type=float, default='0.0')
    parser.add_argument('--ymax', type=float, default='5.0')
    parser.add_argument('--tau', type=float, default='10.0')
    parser.add_argument('--noise', type=float, default='0.1')
    parser.add_argument('--virtuals', type=str, default='5,10,15')
    parser.add_argument('--Ntrials', type=int, default=10)

    args = parser.parse_args()
    print(args)

    folder, prefix, posfix = args.folder, args.prefix, args.posfix
    ymin, ymax = args.ymin, args.ymax
    tau, noise, Ntrials = args.tau, args.noise, args.Ntrials
    
    Vs = [int(x) for x in args.virtuals.split(',')]
    cmap = plt.get_cmap("viridis")
    fig, axs = plt.subplots(1, 1, figsize=(12, 6), squeeze=False)
    axs = axs.ravel()
    putils.setPlot(fontsize=24, labelsize=24)
    #colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = putils.cycle

    ntitle = ''
    #alpha_ls = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    alpha_ls = np.linspace(0.0, 1.0, 21)

    ax = axs[0]
    dcl = 0

    for V in Vs:
        for cb_input in [1, 0]:
            avg_rs, std_rs, xs = [], [], []
            for alpha in alpha_ls:
                local_vpt = []
                for rfile in glob.glob('{}/log/{}*V_{}*_tau_{}*_alpha_{:.3f}_cb_{}*noise_{:.3f}*{}.log'.format(\
                    folder, prefix, V, tau, alpha, cb_input, noise, posfix)):
                    print(rfile)
                    ntitle = os.path.basename(rfile)
                    nidx = ntitle.find('cb_')
                    ntitle = ntitle[nidx:]
                    ntitle = ntitle.replace('.log', '')
                    with open(rfile, 'r') as rf:
                        lines = rf.readlines()
                        for line in lines:
                            if 'INFO' in line and 'pred_time_05' in line:
                                pred_time_05 = (float)(re.search('pred_time_05=([0-9.]*)', line).group(1))
                                local_vpt.append(pred_time_05)
                if len(local_vpt) > 0:
                    avg_vpt, std_vpt = np.mean(local_vpt), np.std(local_vpt)
                    avg_rs.append(avg_vpt)
                    std_rs.append(std_vpt)
                    xs.append(alpha)

            avg_rs, std_rs = np.array(avg_rs), np.array(std_rs)
            color = colors[dcl]
            if len(avg_rs) > 0:
                #ax.errorbar(xs, avg_tests, yerr=std_tests, alpha = 0.8, color=color, elinewidth=2, linewidth=2, markersize=12)
                ax.plot(xs, avg_rs, 's-', alpha = 0.8, linewidth=3, markersize=8, mec='k', mew=0.5, \
                            color=color, label='$V=${}, cb={}, $\\tau=${}'.format(V, cb_input, tau))
                ax.fill_between(xs, avg_rs - std_rs, avg_rs + std_rs, facecolor=color, alpha=0.2)
                dcl += 1

    ax.set_xlabel('$\\alpha$', fontsize=24)
    ax.set_xticks(np.linspace(0.0, 1.0, 11))
    ax.set_xlim([0.0, 1.01])

    #ax.set_xscale('log',base=10)
    ax.set_ylabel('VPT($\\varepsilon=0.05$)', fontsize=24)
    ax.set_ylim([ymin, ymax])
    #ax.set_xticklabels(labels='')
    #ax.set_yticklabels(labels='')
    ax.grid(True, which="both", ls="-", color='0.65')
    ax.legend(fontsize=12)
    ax.set_title(ntitle, fontsize=12)
    #ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=14)

    for ax in axs:
        #ax.minorticks_on()
        ax.tick_params('both', length=8, width=1, which='major', labelsize=20)
        ax.tick_params('both', length=4, width=1, which='minor')

    figsave = os.path.join(folder, 'resfigs')
    os.makedirs(figsave, exist_ok=True)

    outbase = os.path.join(figsave, 'trials_{}_tau_{}_{}'.format(Ntrials, tau, ntitle))
    plt.tight_layout()
    if ntitle != '':
        for ftype in ['png']:
            plt.savefig('{}.{}'.format(outbase, ftype), bbox_inches='tight')
        plt.show()
    