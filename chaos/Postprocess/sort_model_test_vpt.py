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
    parser.add_argument("--sysname", help="type of chaotic system", type=str, default='KuramotoSivashinskyGP64')
    parser.add_argument("--posfix", help="posfix", type=str, default='NICS_10')
    parser.add_argument('--tag', help="filter the models", type=str, default='N_used_10000-')

    args = parser.parse_args()
    sysname, posfix, tag = args.sysname, args.posfix, args.tag

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
    models = []

    for rfolder in glob.glob('{}/*{}*'.format(eval_path, posfix)):
        if os.path.isdir(rfolder) and (tag in os.path.basename(rfolder)):
            models.append(rfolder)

    qslist = []
    for rfolder in models:
        fname = os.path.join(rfolder, 'results.pickle')
        if os.path.isfile(fname):
            with open(fname, 'rb') as rfile:
                try:
                    rs = pickle.load(rfile)
                except:
                    continue
                #print(rs.keys())
                #print(rfolder)
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
                
                for lb in ['TEST', 'TRAIN']:
                    preds = rs['predictions_all_{}'.format(lb)]
                    truths = rs['truths_all_{}'.format(lb)]
                    M = len(preds)
                    vpts = []
                    for j in range(M):
                        vpts.append(calVPT(preds[j], truths[j], eps=0.5, dt=dt, maxLyp=maxLyp))
                    if lb == 'TEST':
                        qs.avg_test_vpt = np.mean(vpts)
                        qs.std_test_vpt = np.std(vpts)
                    else:
                        qs.avg_train_vpt = np.mean(vpts)
                        qs.std_train_vpt = np.std(vpts)
                #qs.info()
                qslist.append(qs)
        else:
            print('Not found {}'.format(fname))
    qslist.sort(key=lambda c: c.n_pred_050_avg_test, reverse=True)
    sorted_file = os.path.join(fig_path, 'sorted_test_050_{}.txt'.format(tag))
    print('Write to file {}'.format(sorted_file))
    out_qslist_tofile(qslist, sorted_file)
    
    