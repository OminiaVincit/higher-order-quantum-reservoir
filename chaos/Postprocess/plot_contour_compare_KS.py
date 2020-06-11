#!/usr/bin/env python
# # -*- coding: utf-8 -*-
import pickle as pickle
import glob, os
import numpy as np
import argparse
import time
import seaborn as sns
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import sys,inspect
import utils
from utils import *

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--sysname", help="type of chaotic system", type=str, default='KuramotoSivashinskyGP64')
    parser.add_argument("--posfix", help="posfix", type=str, default='NICS_10')
    parser.add_argument('--tidx', type=int, default=4)

    args = parser.parse_args()
    sysname, posfix, tidx = args.sysname, args.posfix, args.tidx

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

    # list of models
    models = [\
        ['hqrc_pr_pinv-N_used_10000-NG_32-Nq_10-DL_2000-GIL_4-GS_2-A_0.0-J_2.0-fJ_1-V_10-T_4.0-NL_1-IPL_400-REG_1e-07-AU_1-NICS_10', 'HQR,V=10'],
        ['hqrc_pr_pinv-N_used_10000-NG_32-Nq_10-DL_2000-GIL_4-GS_2-A_0.0-J_2.0-fJ_1-V_15-T_4.0-NL_1-IPL_400-REG_1e-07-AU_1-NICS_10', 'HQR,V=15'],
        ['hqrc_pr_pinv-N_used_10000-NG_32-Nq_10-DL_2000-GIL_4-GS_2-A_0.0-J_2.0-fJ_1-V_20-T_4.0-NL_1-IPL_400-REG_1e-09-AU_1-NICS_10', 'HQR,V=20'],
        ['hqrc_pr_pinv-N_used_10000-NG_32-Nq_10-DL_2000-GIL_4-GS_2-A_0.0-J_2.0-fJ_1-V_25-T_4.0-NL_1-IPL_400-REG_1e-09-AU_1-NICS_10', 'HQR,V=25'],
        ['RNN-esn-PARALLEL-NG_32-RDIM_64-N_used_10000-SIZE_500-D_10.0-RADIUS_0.6-SIGMA_1.0-DL_2000-NL_1-IPL_400-REG_1e-05-NICS_10-GS_2-GIL_4', 'ESN-500'],
        ['RNN-esn-PL-NG_32-RDIM_64-N_used_10000-SIZE_1000-D_10.0-RADIUS_0.6-SIGMA_1.0-DL_2000-NL_1-IPL_400-REG_1e-07-NICS_100-GS_2-GIL_4', 'ESN-1000'],
        ['lstm-PL-NG_32-RDIM_64-N_used_10000-NUM-LAY_2-SIZE-LAY_150-ACT_tanh-ISH_statefull-SL_4-PL_4-LR_0.001-DKP_1.0-ZKP_0.998-HSPL_500-IPL_400-NL_1-NICS_10-GS_2-GIL_4', 'LSTM-150'],
        ['lstm-PL-NG_32-RDIM_64-N_used_10000-NUM-LAY_1-SIZE-LAY_500-ACT_tanh-ISH_statefull-SL_4-PL_4-LR_0.001-DKP_1.0-ZKP_0.998-HSPL_500-IPL_400-NL_1-NICS_10-GS_2-GIL_4', 'LSTM-500'],
        ['gru-PL-NG_32-RDIM_64-N_used_10000-NUM-LAY_1-SIZE-LAY_500-ACT_tanh-ISH_statefull-SL_4-PL_4-LR_0.001-DKP_1.0-ZKP_0.998-HSPL_500-IPL_400-NL_1-NICS_10-GS_2-GIL_4', 'GRU-500'],
        ['RNN-gru-PL-NG_32-RDIM_64-N_used_10000-NUM-LAY_1-SIZE-LAY_1000-SL_4-PL_4-LR_0.001-DKP_1.0-ZKP_0.998-HSPL_500-IPL_400-NL_5-NICS_10-GS_2-GIL_4', 'GRU-1000']
    ]
    models = [[os.path.join(eval_path, m[0]), m[1]] for m in models]
    dfdict = dict()
    targets = dict()
    outputs = dict()
    errors = dict()
    vmin, vmax, vmax_error, vmin_error = 10000, -10000, -10000, 0.0

    for i in range(len(models)):
        rfolder, label = models[i][0], models[i][1]
        fname = os.path.join(rfolder, 'results.pickle')
        if os.path.isfile(fname):
            with open(fname, 'rb') as rfile:
                try:
                    rs = pickle.load(rfile)
                except:
                    continue
                #print(rs.keys())
                qs = QResults()
                qs.rmnse_avg_test = rs['rmnse_avg_TEST']
                qs.rmnse_avg_train = rs['rmnse_avg_TRAIN']
                qs.n_pred_005_avg_test = rs['num_accurate_pred_005_avg_TEST']
                qs.n_pred_005_avg_train = rs['num_accurate_pred_005_avg_TRAIN']
                qs.n_pred_050_avg_test = rs['num_accurate_pred_050_avg_TEST']
                qs.n_pred_050_avg_train = rs['num_accurate_pred_050_avg_TRAIN']
                qs.model_name = rs['model_name']
                #if qs.rmnse_avg_test != np.inf and qs.rmnse_avg_train != np.inf:
                    #print(rs.keys())
                #print(qs.model_name)
                #print('train={}, test={}'.format(qs.rmnse_avg_train, qs.rmnse_avg_test))
                #qs.info()

                pred_test = rs['predictions_all_TEST']
                truth_test = rs['truths_all_TEST']
                M = len(pred_test)
                print('Number of test', M)
                rmsels = []
                for j in range(M):
                     rmsels.append(calNRMSE(pred_test[j], truth_test[j]))

                dfdict[label] = np.mean(np.array(rmsels), axis=0)
                print(dfdict[label].shape)

                targets[label] = truth_test[tidx]
                outputs[label] = pred_test[tidx]
                errors[label] = np.abs(targets[label]-outputs[label])
                vmin = min(vmin, targets[label].min())
                vmax = max(vmax, targets[label].max())
                vmax_error = max(vmax_error, targets[label].max())

        else:
            print('Not found {}'.format(fname))
    
    # PLOTTING
    cmap = plt.get_cmap("RdBu")
    ecmap = plt.get_cmap("summer_r")
    #plt.style.use('seaborn-colorblind')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams['font.size']=10
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, axs = plt.subplots(1, 6, figsize=(20, 4), sharey=True)
    fig.subplots_adjust(hspace=0.4, wspace = 0.4)
    
    axs = axs.ravel()


    # Plot NRMSE curves
    ax = axs[-1]
    for k in dfdict.keys():
        rmse = dfdict[k]
        ts = np.array(range(len(rmse))) * dt / maxLyp
        ax.plot(rmse, ts, label=k)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    #ax.grid()
    #ax.set_xlim([0, 2])
    #ax.set_ylabel('$tLambda$')
    
    # Plotting the contour plot
    fsize = 8
    #axs[0].set_ylabel(r"$tLambda$", fontsize=fsize)
    mp0 = createContour_(fig, axs[0], targets['HQR,V=10'], "Target", fsize, vmin, vmax, cmap, dt, maxLyp)
    mp1 = createContour_(fig, axs[1], outputs['HQR,V=10'], "HQR,V=10", fsize, vmin, vmax, cmap, dt, maxLyp)
    mp2 = createContour_(fig, axs[2], errors['HQR,V=10'], "Error", fsize, vmin_error, vmax_error, ecmap, dt, maxLyp)
    mp3 = createContour_(fig, axs[3], outputs['ESN-1000'], "ESN-1000", fsize, vmin, vmax, cmap, dt, maxLyp)
    mp4 = createContour_(fig, axs[4], errors['ESN-1000'], "Error", fsize, vmin_error, vmax_error, ecmap, dt, maxLyp)

    # # Basic violinplot
    # 
    # df = pd.DataFrame(dfdict)
    # ax = sns.stripplot(data=df, jitter=True, linewidth=1, alpha=0.9, marker=mk, size=7)
    # sns.violinplot(data=df, color=".8", ax = ax, inner=None)
    
    #ax.set_ylim([-1, 6])
    #ax.set_xticklabels([])
    #ax.set_yticklabels([])
    #g = sns.factorplot(data=dfdict)
    for ax in axs:
        ax.tick_params(direction='out', length=6, width=1)
    
    outbase = 'tidx_{}_{}'.format(tidx, posfix)
    outbase = os.path.join(fig_path, outbase)
    for ftype in ['pdf', 'svg', 'png']:
        plt.savefig('{}_{}_rs5.{}'.format(outbase, sysname, ftype), bbox_inches='tight', transparent=True, dpi=600)
    plt.show()


