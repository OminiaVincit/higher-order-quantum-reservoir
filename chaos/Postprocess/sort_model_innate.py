#!/usr/bin/env python
# # -*- coding: utf-8 -*-
#!/usr/bin/env python
import pickle as pickle
import glob, os
import numpy as np
import argparse
import time
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys,inspect
import utils
from utils import *

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--sysname", help="type of chaotic system", type=str, default='Lorenz3D')
    parser.add_argument("--posfix", help="posfix", type=str, default='NICS_11')
    parser.add_argument('--tag', help="filter the models", type=str, default='N_used_10000-')
    
    args = parser.parse_args()
    sysname, posfix, tagstr = args.sysname, args.posfix, args.tag

    current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    results_dir = os.path.dirname(current_dir) + "/Results"
    print(results_dir)
    eval_path = os.path.join(results_dir, '{}/Evaluation_Data'.format(sysname))
    print(eval_path)
    fig_path = os.path.join(results_dir, '{}/Eval_Figures'.format(sysname))
    if os.path.isdir(fig_path) == False:
        os.mkdir(fig_path)
    
    maxLyp = 1.0
    dt = 0.01
    if sysname == 'Lorenz3D':
        maxLyp = 0.9056
    elif 'Lorenz96_F10' in sysname:
        maxLyp = 2.27
    elif 'Lorenz96_F8' in sysname:
        maxLyp = 1.68
    elif 'KuramotoSivashinskyGP64' in sysname:
        maxLyp = 20
        dt = 0.25
    
    noise_ls = []
    for i in [-6]:
        for j in range(1, 10):
            noise_ls.append(j * (10**i))
    #noise_ls.append(1e-5)


    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 1, 1)
    #plt.style.use('seaborn-colorblind')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams['font.size']=14
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    tags = [tagstr]
    if tagstr == 'all':
        tags = ['N_used_{}-'.format(x) for x in [10000, 20000, 50000]]

    for j in range(len(tags)):
        tag = tags[j]
        # list of models
        drmse, dnoise = dict(), dict()
        pre_tag, post_tag = 'INR_0.0-INL_1', 'INR_10.0-INL_10'
        drmse[pre_tag], dnoise[pre_tag] = [], []
        drmse[post_tag], dnoise[post_tag] = [], []

        #hqrc_innate_pinv-RDIM_1-N_used_1000-DL_100-Nqr_5-A_0.0-J_2.0-fJ_0-V_1-NL_1-
        #ON_5e-06-INR_0.0-INL_1-REG_1e-09-NICS_100
        for noise in noise_ls:
            rmse_ls = []
            for lb in [pre_tag, post_tag]:
                for rfolder in glob.glob('{}/hqrc_innate_pinv*ON_{}-{}-*{}'.format(eval_path, noise, lb, posfix)):
                    rbase = os.path.basename(rfolder)
                    if os.path.isdir(rfolder) and (tag in rbase):
                        fname = os.path.join(rfolder, 'results.pickle')
                        if os.path.isfile(fname):
                            with open(fname, 'rb') as rfile:
                                try:
                                    rs = pickle.load(rfile)
                                except:
                                    continue
                                #print(rs.keys())
                                #print(rfolder)
                                rmse = rs['rmnse_avg_TEST']
                                # qs = QResults()
                                # qs.rmnse_avg_test = rs['rmnse_avg_TEST']
                                # qs.rmnse_avg_train = rs['rmnse_avg_TRAIN']
                                # qs.n_pred_005_avg_test = rs['num_accurate_pred_005_avg_TEST']
                                # qs.n_pred_005_avg_train = rs['num_accurate_pred_005_avg_TRAIN']
                                # qs.n_pred_050_avg_test = rs['num_accurate_pred_050_avg_TEST']
                                # qs.n_pred_050_avg_train = rs['num_accurate_pred_050_avg_TRAIN']
                                # qs.model_name = rs['model_name']
                                if rmse == np.inf:
                                    continue
                                rmse_ls.append(rmse)
                        else:
                            print('Not found {}'.format(fname))
                    
                if len(rmse_ls) > 0:
                    #rmse = np.min(rmse_ls)
                    rmse = np.min(rmse_ls)
                    drmse[lb].append(rmse)
                    dnoise[lb].append(noise)
        rels = []
        for i in range(len(dnoise[pre_tag])):
            rel = (drmse[pre_tag][i] - drmse[post_tag][i]) / drmse[pre_tag][i]
            rels.append(rel)
        #ax.plot(dnoise[pre_tag], rels, 'o--', linewidth=2, label='Rel-{}'.format(tag), markersize=10)
        ax.plot(dnoise[pre_tag], drmse[pre_tag], '*--', color=colors[j], linewidth=2, label='pre-{}'.format(tag), markersize=10)
        ax.plot(dnoise[post_tag], drmse[post_tag], 'o-', color=colors[j], linewidth=2, label='post-{}'.format(tag), markersize=10)
    ax.set_xscale("log", basex=10)
    #ax.set_yscale("log", basey=10)
    ax.set_ylim([0.86, 0.942])
    ax.set_yticks([0.86, 0.90, 0.94])
    ax.grid(which='major',color='black',linestyle='-')
    ax.grid(axis='x',which='minor',color='black',linestyle='-')
    ax.tick_params('both', length=15, width=1, which='major')
    ax.tick_params('both', length=10, width=1, which='minor')
    ax.legend()
    ax.set_ylabel('RNMSE')
    ax.set_xlabel('Noise variance')

    outbase = os.path.join(fig_path, 'rmse_compare_{}_{}'.format(tagstr, posfix))
    ax.set_title(outbase)
    for ftype in ['png', 'svg']:
        plt.savefig('{}.{}'.format(outbase, ftype), bbox_inches='tight', dpi=600)
    plt.show()


    
    