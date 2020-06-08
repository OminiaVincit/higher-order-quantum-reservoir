#!/usr/bin/env python
"""
Quoc Hoan Tran, Nakajima-Lab, The University of Tokyo
    View predicted time series for NARMA tasks
    See run_view_narma.sh for an example of the running script
"""

import sys
import numpy as np
import os
import scipy
import argparse
import multiprocessing
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
import time
import datetime
import hqrc as hqrc
import utils
from utils import *
import pickle

def predict_job(qparams, nqrc, alpha, buffer, train_input_seq, train_output_seq, \
        val_input_seq, val_output_seq, rseed, outbase):
    print('Start process strength={}, taudelta={}, virtual={}, Jdelta={}'.format(alpha, qparams.tau, qparams.virtual_nodes, qparams.max_energy))
    train_pred_seq, train_loss, val_pred_seq, val_loss = hqrc.get_loss(qparams, buffer, train_input_seq, train_output_seq, \
        val_input_seq, val_output_seq, nqrc, alpha, ranseed=rseed)
    results = {'val_input': val_input_seq[0], 'val_pred': val_pred_seq, \
        'val_out': val_output_seq, 'train_loss': train_loss, 'val_loss': val_loss}
    #pickle.dump()
    with open('{}_alpha_{}.binaryfile'.format(outbase, alpha), 'wb') as wrs:
        pickle.dump(results, wrs)
    print(outbase, alpha, train_loss, val_loss)
    print(val_input_seq.shape, val_output_seq.shape)

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--units', type=int, default=5, help='Number of the hidden units')
    parser.add_argument('--coupling', type=float, default=1.0, help='Maximum coupling energy')
    parser.add_argument('--rho', type=int, default=0, help='Flag for initializing the density matrix')
    parser.add_argument('--beta', type=float, default=1e-14, help='regularization term')
    parser.add_argument('--solver', type=str, default=LINEAR_PINV, \
        help='regression solver by linear_pinv,ridge_pinv,auto,svd,cholesky,lsqr,sparse_cg,sag')

    parser.add_argument('--trainlen', type=int, default=2000)
    parser.add_argument('--vallen', type=int, default=2000)
    parser.add_argument('--transient', type=int, default=2000, help='Transitient time steps')
    
    parser.add_argument('--ntrials', type=int, default=1)
    parser.add_argument('--virtuals', type=int, default=20)
    parser.add_argument('--taudelta', type=float, default=2.0, help='Interval between the inputs')
    parser.add_argument('--nqrc', type=int, default=5, help='Number of reservoirs')

    parser.add_argument('--orders', type=str, default='5,10,15,20')
    parser.add_argument('--basename', type=str, default='qrc_narma')
    parser.add_argument('--savedir', type=str, default='view_narma')
    parser.add_argument('--plot', type=int, default=0)
    args = parser.parse_args()
    print(args)

    n_units, max_energy, beta = args.units, args.coupling, args.beta
    train_len, val_len, buffer = args.trainlen, args.vallen, args.transient
    V, tau, nqrc = args.virtuals, args.taudelta, args.nqrc
    init_rho, solver = args.rho, args.solver

    basename, savedir = args.basename, args.savedir
    if os.path.isfile(savedir) == False and os.path.isdir(savedir) == False:
        os.mkdir(savedir)

    train_input_seq_ls, train_output_seq_ls = [], []
    val_input_seq_ls, val_output_seq_ls = [], []
    
    orders = [int(x) for x in args.orders.split(',')]
    N = len(orders)
    cmap = plt.get_cmap("viridis")
    #plt.style.use('seaborn-colorblind')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams['font.size']=14
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, axs = plt.subplots(N, 1, figsize=(6, 2.5*N))
    axs = axs.ravel()

    strengths = [0.0, 0.2, 0.4, 0.6, 0.8]
    for i in range(N):
        order = orders[i]
        outbase = os.path.join(savedir, '{}_{}_{}_{}_{}_NARMA_{}_V_{}_nqr_{}'.format(basename, \
                solver, train_len, val_len, buffer, order, V, nqrc))
        
        if args.plot <= 0:
            data, target = make_data_for_narma(train_len + val_len + buffer, orders = [order])

            train_input_seq_org = np.array(data[: buffer + train_len])
            train_input_seq_org = train_input_seq_org.reshape(1, train_input_seq_org.shape[0])
            train_output_seq = target[  : buffer + train_len] 

            val_input_seq_org =  np.array(data[buffer + train_len : buffer + train_len + val_len])
            val_input_seq_org = val_input_seq_org.reshape(1, val_input_seq_org.shape[0])
            val_output_seq = target[buffer + train_len : buffer + train_len + val_len]

            train_input_seq = np.tile(train_input_seq_org, (nqrc, 1))
            val_input_seq = np.tile(val_input_seq_org, (nqrc, 1))
        
            # Evaluation
            jobs, pipels = [], []

            for alpha in strengths:
                qparams = QRCParams(n_units=n_units, max_energy=max_energy, \
                    beta=beta, virtual_nodes=V, tau=tau, init_rho=init_rho, solver=solver)
                rseed = 0
                p = multiprocessing.Process(target=predict_job, args=(qparams, nqrc, alpha, buffer, train_input_seq, train_output_seq, \
                        val_input_seq, val_output_seq, rseed, outbase))
                jobs.append(p)
            
                    # Start the process
            for p in jobs:
                p.start()

            # Ensure all processes have finished execution
            for p in jobs:
                p.join()
        else:
            ax = axs[i]
            bg = int(val_len / 2)
            ed = bg + 100
            for j in range(len(strengths)):
                alpha = strengths[j]
                with open('{}_alpha_{}.binaryfile'.format(outbase, alpha), 'rb') as rrs:
                    results = pickle.load(rrs)
                    #val_input  = results['val_input'].ravel()
                    val_pred   = results['val_pred'].ravel()
                    val_out    = results['val_out'].ravel()
                    if alpha == 0:
                        #axs[0].plot(val_input[bg:ed], c='black')
                        ax.plot(val_out[bg:ed], c='black', label='target')
                    ax.plot(val_pred[bg:ed], c=colors[j], label='{}'.format(alpha))
            if i == 0:
                ax.legend()
            #if i < N-1:
            ax.set_xticklabels([])
    if args.plot > 0:
        for ftype in ['png']:
            outbase = os.path.join(savedir, '{}_{}_{}_{}_{}_V_{}_nqr_{}'.format(basename, \
                    solver, train_len, val_len, buffer, V, nqrc))
            plt.savefig('{}_narma.{}'.format(outbase, ftype), bbox_inches='tight')
        #plt.show()
            
        
                
    
    