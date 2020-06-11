#!/usr/bin/env python
# # -*- coding: utf-8 -*-
#!/usr/bin/env python
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
    parser.add_argument("--sysname", help="type of chaotic system", type=str, default='Lorenz3D')
    parser.add_argument('--tidx', type=int, default=2)
    parser.add_argument('--used', type=int, default=0)
    parser.add_argument('--plot', type=int, default=0, help="0: RMSE, 1: spectral")

    args = parser.parse_args()
    sysname, tidx = args.sysname, args.tidx

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
    models_1000 = [\
        ['hqrc_pinv-RDIM_1-N_used_1000-DL_200-Nqr_5-A_0.1-J_2.0-fJ_1-V_10-NL_1-IPL_500-IUL_0-REG_1e-11-AU_0-NICS_100', 'HQR,V=10'],
        ['hqrc_pinv-RDIM_1-N_used_1000-DL_200-Nqr_5-A_0.1-J_2.0-fJ_1-V_15-NL_1-IPL_500-IUL_0-REG_1e-07-AU_0-NICS_100', 'HQR,V=15'],
        ['hqrc_pinv-RDIM_1-N_used_1000-DL_200-Nqr_5-A_0.0-J_2.0-fJ_1-V_20-NL_1-IPL_500-IUL_0-REG_1e-09-AU_0-NICS_100', 'HQR,V=20'],
        #['hqrc_pinv-RDIM_1-N_used_1000-DL_200-Nqr_5-A_0.0-J_2.0-fJ_1-V_25-NL_1-IPL_500-IUL_0-REG_1e-09-AU_0-NICS_100', 'HQR,V=25'],
        ['ESN_pinv-RDIM_1-N_used_1000-SIZE_80-D_10.0-RADIUS_0.9-SIGMA_1.0-DL_200-NL_1-IPL_500-REG_1e-09-NICS_100', 'ESN-80'],
        ['ESN_pinv-RDIM_1-N_used_1000-SIZE_120-D_10.0-RADIUS_0.9-SIGMA_1.0-DL_200-NL_1-IPL_500-REG_1e-09-NICS_100', 'ESN-120'],
        ['ESN_pinv-RDIM_1-N_used_1000-SIZE_150-D_10.0-RADIUS_0.9-SIGMA_1.0-DL_200-NL_1-IPL_500-REG_1e-09-NICS_100', 'ESN-150'],
        #['ESN_pinv-RDIM_1-N_used_1000-SIZE_5000-D_10.0-RADIUS_0.9-SIGMA_1.0-DL_200-NL_1-IPL_500-REG_1e-09-NICS_100', 'ESN-5000'],
        ['RNN-lstm-RDIM_1-N_used_1000-NLAY_1-SLAY_150-ISH_statefull-SL_16-PL_4-LR_0.001-DKP_1.0-ZKP_1.0-HSPL_100-IPL_500-NL_1-NICS_100', 'LSTM-150'],
        ['RNN-lstm-RDIM_1-N_used_1000-NLAY_2-SLAY_1000-ISH_statefull-SL_16-PL_4-LR_0.001-DKP_1.0-ZKP_1.0-HSPL_100-IPL_500-NL_1-NICS_100', 'LSTM-1000(2)'],
        ['RNN-lstm-RDIM_1-N_used_1000-NLAY_1-SLAY_3000-ISH_statefull-SL_16-PL_4-LR_0.001-DKP_1.0-ZKP_1.0-HSPL_100-IPL_500-NL_1-NICS_100', 'LSTM-3000'],
        ['RNN-gru-RDIM_1-N_used_1000-NLAY_1-SLAY_80-ISH_statefull-SL_16-PL_16-LR_0.001-DKP_1.0-ZKP_1.0-HSPL_100-IPL_500-NL_1-NICS_100', 'GRU-80'],
        ['RNN-gru-RDIM_1-N_used_1000-NLAY_3-SLAY_120-ISH_statefull-SL_16-PL_16-LR_0.001-DKP_1.0-ZKP_1.0-HSPL_100-IPL_500-NL_1-NICS_100', 'GRU-120(3)'],
        ['RNN-gru-RDIM_1-N_used_1000-NLAY_3-SLAY_3000-ISH_statefull-SL_16-PL_16-LR_0.001-DKP_1.0-ZKP_1.0-HSPL_100-IPL_500-NL_1-NICS_100', 'GRU-3000(3)'],
    ]

    models_10000 = [\
        ['hqrc_pinv-RDIM_1-N_used_10000-DL_2000-Nqr_5-A_0.0-J_2.0-fJ_1-V_10-TAU_4.0-NL_1-IPL_1000-IUL_0-REG_1e-09-AU_0-NICS_100', 'HQR,V=10'],
        ['hqrc_pinv-RDIM_1-N_used_10000-DL_2000-Nqr_5-A_0.0-J_2.0-fJ_1-V_15-TAU_4.0-NL_1-IPL_1000-IUL_0-REG_1e-07-AU_0-NICS_100', 'HQR,V=15'],
        ['hqrc_pinv-RDIM_1-N_used_10000-DL_2000-Nqr_5-A_0.0-J_2.0-fJ_1-V_20-TAU_4.0-NL_1-IPL_1000-IUL_0-REG_1e-07-AU_0-NICS_100', 'HQR,V=20'],
        ['ESN_pinv-RDIM_1-N_used_10000-SIZE_150-D_10.0-RADIUS_0.9-SIGMA_1.0-DL_2000-NL_1-IPL_1000-REG_1e-09-NICS_100', 'ESN-150'],
        ['ESN_pinv-RDIM_1-N_used_10000-SIZE_500-D_10.0-RADIUS_0.9-SIGMA_1.0-DL_2000-NL_1-IPL_500-REG_1e-09-NICS_100', 'ESN-500'],
        ['ESN_pinv-RDIM_1-N_used_10000-SIZE_1000-D_10.0-RADIUS_0.9-SIGMA_1.0-DL_2000-NL_1-IPL_500-REG_1e-09-NICS_100', 'ESN-1000'],
        ['RNN-lstm-RDIM_1-N_used_10000-NLAY_3-SLAY_100-ISH_statefull-SL_16-PL_4-LR_0.001-DKP_1.0-ZKP_1.0-HSPL_1000-IPL_1000-NL_1-NICS_100', 'LSTM-150(3)'],
        ['RNN-lstm-RDIM_1-N_used_10000-NLAY_1-SLAY_500-ISH_statefull-SL_16-PL_4-LR_0.001-DKP_1.0-ZKP_1.0-HSPL_1000-IPL_500-NL_1-NICS_100', 'LSTM-500'],
        ['RNN-lstm-RDIM_1-N_used_10000-NLAY_1-SLAY_1000-ISH_statefull-SL_16-PL_4-LR_0.001-DKP_1.0-ZKP_1.0-HSPL_1000-IPL_500-NL_1-NICS_100', 'LSTM-1000'],
        ['RNN-gru-RDIM_1-N_used_10000-NLAY_1-SLAY_100-ISH_statefull-SL_16-PL_16-LR_0.001-DKP_1.0-ZKP_1.0-HSPL_1000-IPL_500-NL_1-NICS_100', 'GRU-100'],
        ['RNN-gru-RDIM_1-N_used_10000-NLAY_1-SLAY_1000-ISH_statefull-SL_16-PL_16-LR_0.001-DKP_1.0-ZKP_1.0-HSPL_1000-IPL_500-NL_1-NICS_100', 'GRU-1000'],
        ['RNN-gru-RDIM_1-N_used_10000-NLAY_1-SLAY_3000-ISH_statefull-SL_16-PL_16-LR_0.001-DKP_1.0-ZKP_1.0-HSPL_1000-IPL_1000-NL_1-NICS_100', 'GRU-3000'],
    ]


    models_100000 = [\
        ['hqrc_pinv-RDIM_1-N_used_100000-DL_2000-Nqr_5-A_0.0-J_2.0-fJ_1-V_10-NL_1-IPL_2000-IUL_0-REG_1e-07-AU_0-NICS_100', 'HQR,V=10'],
        ['hqrc_pinv-RDIM_1-N_used_100000-DL_2000-Nqr_5-A_0.0-J_2.0-fJ_1-V_15-NL_1-IPL_2000-IUL_0-REG_1e-09-AU_0-NICS_100', 'HQR,V=15'],
        ['hqrc_pinv-RDIM_1-N_used_100000-DL_2000-Nqr_5-A_0.0-J_2.0-fJ_1-V_20-NL_1-IPL_2000-IUL_0-REG_1e-09-AU_0-NICS_100', 'HQR,V=20'],
        ['ESN_pinv-RDIM_1-N_used_100000-SIZE_500-D_10.0-RADIUS_0.9-SIGMA_1.0-DL_2000-NL_1-IPL_2000-REG_1e-07-NICS_100', 'ESN-500'],
        ['ESN_pinv-RDIM_1-N_used_100000-SIZE_1000-D_10.0-RADIUS_0.9-SIGMA_1.0-DL_2000-NL_1-IPL_2000-REG_1e-07-NICS_100', 'ESN-1000'],
        ['ESN_pinv-RDIM_1-N_used_100000-SIZE_1500-D_10.0-RADIUS_0.9-SIGMA_1.0-DL_2000-NL_1-IPL_2000-REG_1e-07-NICS_100', 'ESN-1500'],
        ['RNN-lstm-RDIM_1-N_used_100000-NLAY_2-SLAY_80-ISH_statefull-SL_16-PL_4-LR_0.001-DKP_1.0-ZKP_1.0-HSPL_1000-IPL_1000-NL_1-NICS_100', 'LSTM-80(2)'],
        ['RNN-lstm-RDIM_1-N_used_100000-NLAY_2-SLAY_100-ISH_statefull-SL_16-PL_4-LR_0.001-DKP_1.0-ZKP_1.0-HSPL_1000-IPL_1000-NL_1-NICS_100', 'LSTM-100(2)'],
        ['RNN-lstm-RDIM_1-N_used_100000-NLAY_1-SLAY_150-ISH_statefull-SL_16-PL_4-LR_0.001-DKP_1.0-ZKP_1.0-HSPL_2000-IPL_2000-NL_1-NICS_100','LSTM-150'],
        ['RNN-gru-RDIM_1-N_used_100000-NLAY_3-SLAY_150-ISH_statefull-SL_16-PL_16-LR_0.001-DKP_1.0-ZKP_1.0-HSPL_2000-IPL_2000-NL_1-NICS_100', 'GRU-150(3)'],
        ['RNN-gru-RDIM_1-N_used_100000-NLAY_1-SLAY_80-ISH_statefull-SL_16-PL_16-LR_0.001-DKP_1.0-ZKP_1.0-HSPL_2000-IPL_2000-NL_1-NICS_100', 'GRU-80'],
        ['RNN-gru-RDIM_1-N_used_100000-NLAY_1-SLAY_1500-ISH_statefull-SL_16-PL_16-LR_0.001-DKP_1.0-ZKP_1.0-HSPL_2000-IPL_2000-NL_1-NICS_100', 'GRU-1500']
    ]

    models_compare = [\
        ['hqrc_pinv-RDIM_1-N_used_1000-DL_200-Nqr_5-A_0.1-J_2.0-fJ_1-V_15-NL_1-IPL_500-IUL_0-REG_1e-07-AU_0-NICS_100', 'HQR,10^3'],
        ['ESN_pinv-RDIM_1-N_used_1000-SIZE_80-D_10.0-RADIUS_0.9-SIGMA_1.0-DL_200-NL_1-IPL_500-REG_1e-09-NICS_100', 'ESN,10^3'],
        ['RNN-lstm-RDIM_1-N_used_1000-NLAY_1-SLAY_3000-ISH_statefull-SL_16-PL_4-LR_0.001-DKP_1.0-ZKP_1.0-HSPL_100-IPL_500-NL_1-NICS_100', 'LSTM,10^3'],
        ['RNN-gru-RDIM_1-N_used_1000-NLAY_3-SLAY_3000-ISH_statefull-SL_16-PL_16-LR_0.001-DKP_1.0-ZKP_1.0-HSPL_100-IPL_500-NL_1-NICS_100', 'GRU,10^3'],
        ['hqrc_pinv-RDIM_1-N_used_10000-DL_2000-Nqr_5-A_0.0-J_2.0-fJ_1-V_15-TAU_4.0-NL_1-IPL_1000-IUL_0-REG_1e-07-AU_0-NICS_100', 'HQR,10^4'],
        ['ESN_pinv-RDIM_1-N_used_10000-SIZE_150-D_10.0-RADIUS_0.9-SIGMA_1.0-DL_2000-NL_1-IPL_1000-REG_1e-09-NICS_100', 'ESN,10^4'],
        ['RNN-lstm-RDIM_1-N_used_10000-NLAY_3-SLAY_100-ISH_statefull-SL_16-PL_4-LR_0.001-DKP_1.0-ZKP_1.0-HSPL_1000-IPL_1000-NL_1-NICS_100', 'LSTM,10^4'],
        ['RNN-gru-RDIM_1-N_used_10000-NLAY_1-SLAY_1000-ISH_statefull-SL_16-PL_16-LR_0.001-DKP_1.0-ZKP_1.0-HSPL_1000-IPL_500-NL_1-NICS_100', 'GRU,10^4'],
        ['hqrc_pinv-RDIM_1-N_used_100000-DL_2000-Nqr_5-A_0.0-J_2.0-fJ_1-V_15-NL_1-IPL_2000-IUL_0-REG_1e-09-AU_0-NICS_100', 'HQR,10^5'],
        ['ESN_pinv-RDIM_1-N_used_100000-SIZE_500-D_10.0-RADIUS_0.9-SIGMA_1.0-DL_2000-NL_1-IPL_2000-REG_1e-07-NICS_100', 'ESN,10^5'],
        ['RNN-lstm-RDIM_1-N_used_100000-NLAY_2-SLAY_80-ISH_statefull-SL_16-PL_4-LR_0.001-DKP_1.0-ZKP_1.0-HSPL_1000-IPL_1000-NL_1-NICS_100', 'LSTM,10^5'],
        ['RNN-gru-RDIM_1-N_used_100000-NLAY_3-SLAY_150-ISH_statefull-SL_16-PL_16-LR_0.001-DKP_1.0-ZKP_1.0-HSPL_2000-IPL_2000-NL_1-NICS_100', 'GRU,10^5'],
    ]

    models_alpha = [\
        ['hqrc_pinv-RDIM_1-N_used_10000-DL_2000-Nqr_5-A_0.0-J_2.0-fJ_1-V_15-TAU_4.0-NL_1-IPL_1000-IUL_0-REG_1e-07-AU_0-NICS_100', 'alpha=0.0'],
        ['hqrc_pinv-RDIM_1-N_used_10000-DL_2000-Nqr_5-A_0.1-J_2.0-fJ_1-V_15-TAU_4.0-NL_1-IPL_1000-IUL_0-REG_1e-07-AU_0-NICS_100', 'alpha=0.1'],
        ['hqrc_pinv-RDIM_1-N_used_10000-DL_2000-Nqr_5-A_0.3-J_2.0-fJ_1-V_15-TAU_4.0-NL_1-IPL_1000-IUL_0-REG_1e-07-AU_0-NICS_100', 'alpha=0.3'],
        ['hqrc_pinv-RDIM_1-N_used_10000-DL_2000-Nqr_5-A_0.5-J_2.0-fJ_1-V_15-TAU_4.0-NL_1-IPL_1000-IUL_0-REG_1e-09-AU_0-NICS_100', 'alpha=0.5'],
        ['hqrc_pinv-RDIM_1-N_used_10000-DL_2000-Nqr_5-A_0.7-J_2.0-fJ_1-V_15-TAU_4.0-NL_1-IPL_1000-IUL_0-REG_1e-09-AU_0-NICS_100', 'alpha=0.7'],
        ['hqrc_pinv-RDIM_1-N_used_10000-DL_2000-Nqr_5-A_0.9-J_2.0-fJ_1-V_15-TAU_4.0-NL_1-IPL_1000-IUL_0-REG_1e-11-AU_0-NICS_100', 'alpha=0.9'],
    ]

    if args.used == 4:
        models = [[os.path.join(eval_path, m[0]), m[1]] for m in models_alpha]
        samples = ['alpha=0.0', 'alpha=0.1', 'alpha=0.5', 'alpha=0.9']
        title = 'compare connection strength'
    elif args.used == 3:
        models = [[os.path.join(eval_path, m[0]), m[1]] for m in models_compare]
        samples = ['HQR,10^3', '']
        title = 'compare num samples'
    elif args.used == 2:
        models = [[os.path.join(eval_path, m[0]), m[1]] for m in models_100000]
        samples = ['HQR,V=15', 'ESN-500', 'LSTM-80(2)', 'GRU-150(3)']
        title = '10^5'
    elif args.used == 1:
        models = [[os.path.join(eval_path, m[0]), m[1]] for m in models_10000]
        samples = ['HQR,V=10', 'ESN-500', 'LSTM-500', 'GRU-1000']
        title = '10^4'
    else:
        models = [[os.path.join(eval_path, m[0]), m[1]] for m in models_1000]
        samples = ['HQR,V=10', 'ESN-150', 'LSTM-150', 'GRU-120(3)']
        title = '1000'
    
    rmse_dict = dict()
    vpt_dict = dict()
    targets = dict()
    outputs = dict()
    sp_outputs = dict()
    sp_targets = dict()

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
                print('{} Number of test'.format(qs.model_name), M)
                rmsels = []
                vpts = []
                for j in range(M):
                     rmsels.append(calNRMSE(pred_test[j], truth_test[j]))
                     vpts.append(calVPT(pred_test[j], truth_test[j], maxLyp=maxLyp))

                rmse_dict[label] = np.mean(np.array(rmsels), axis=0)
                print(rmse_dict[label].shape)

                vpt_dict[label] = np.array(vpts)

                targets[label] = truth_test[tidx]
                outputs[label] = pred_test[tidx]

                # For frequency
                sp_outputs[label] = rs['sp_pred_TEST']
                sp_targets[label] = rs['sp_true_TEST']
        else:
            print('Not found {}'.format(fname))
    
    # PLOTTING
    cmap = plt.get_cmap("RdBu")
    ecmap = plt.get_cmap("summer_r")
    #plt.style.use('seaborn-colorblind')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams['font.size']=9
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    #fig, axs = plt.subplots(1, 2, figsize=(16, 4))
    #fig.subplots_adjust(hspace=0.4, wspace = 0.4)
    #axs = axs.ravel()

    fig = plt.figure(figsize=(16, 6))
    fig.subplots_adjust(hspace=0.6, wspace = 0.2)

    # Plot box plot
    ax1 = plt.subplot2grid((3,4), (0,0), colspan=4, rowspan=2)
    df = pd.DataFrame(vpt_dict)
    #sns.stripplot(ax=ax, data=df, jitter=True, linewidth=1, alpha=0.9, marker='o', size=6)
    sns.boxplot(data=df, ax = ax1)
    ax1.set_ylim([0, 8.0])
    ax1.set_ylabel('VPT')

    # Plot test samples
    ns = len(samples)
    for i in range(ns):
        label = samples[i]
        if label not in targets.keys():
            continue
        ax = plt.subplot2grid((3, 4), (2,i), colspan=1, rowspan=1)
        ts = np.array(range(len(targets[label]))) * dt / maxLyp
        ax.plot(ts, targets[label], label='Target')
        ax.plot(ts, outputs[label], label='Prediction')
        ax.set_title(label)
        ax.set_xlim([0, 5])
    
    # ax2 = plt.subplot2grid((3,6), (0,4), colspan=2, rowspan=3)
    # if args.plot == 0:
    #     # Plot NRMSE curves
    #     for k in rmse_dict.keys():
    #         rmse = rmse_dict[k]
    #         ts = np.array(range(len(rmse))) * dt / maxLyp
    #         ax2.plot(ts, rmse, label=k)
    # else:
    #     # Plot spectral
    #     for i in range(4):
    #         label = samples[i]
    #         if label not in sp_targets.keys():
    #             continue
    #         sp_truth_test = sp_targets[label]
    #         sp_pred_test  = sp_outputs[label]

    #         M = int(len(sp_pred_test) / 5)
    #         fids = range(M)
    #         ax2.scatter(fids, sp_pred_test[fids], marker='o', facecolors='none', edgecolor=colors[i], label=label)
    #         #fx.plot(fids, sp_pred_test)
    #         if i == 0:
    #             ax2.plot(fids, sp_truth_test[fids], color='k', label='Target')


    # ax2.legend()
    # #ax1.set_ylim([0, 1.0])
    # #ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    # ax2.set_title('used {} samples'.format(title))


    # Plot frequency

    outbase = 'tidx_{}_{}_v2'.format(tidx, title)
    outbase = os.path.join(fig_path, outbase)
    for ftype in ['pdf', 'svg', 'png']:
        plt.savefig('{}_{}_rs3.{}'.format(outbase, sysname, ftype), bbox_inches='tight', transparent=True, dpi=600)
    
    plt.show()


