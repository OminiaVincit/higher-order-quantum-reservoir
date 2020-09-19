#!/usr/bin/env python
"""
    Calculate NMSE for NARMA tasks with quantum innate learning
    See run_hqrc_innate_narma.sh for an example to run the script
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
import hqrc_innate as hqrc
import utils
from utils import *

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--units', type=int, default=5, help='Number of the hidden units')
    parser.add_argument('--coupling', type=float, default=1.0, help='Maximum coupling energy')
    parser.add_argument('--rho', type=int, default=0, help='Flag for initializing the density matrix')
    parser.add_argument('--beta', type=float, default=1e-14, help='reg term')
    parser.add_argument('--solver', type=str, default=LINEAR_PINV, \
        help='regression solver by linear_pinv,ridge_pinv,auto,svd,cholesky,lsqr,sparse_cg,sag')

    parser.add_argument('--trainlen', type=int, default=200)
    parser.add_argument('--vallen', type=int, default=200)
    parser.add_argument('--transient', type=int, default=200, help='Transitient time steps')
    
    parser.add_argument('--ntrials', type=int, default=1)
    parser.add_argument('--virtuals', type=int, default=1)
    parser.add_argument('--taudelta', type=float, default=2.0, help='Interval between the inputs')
    parser.add_argument('--strength', type=float, default=0.5, help='Connection strengths')
    parser.add_argument('--nqrc', type=int, default=5, help='Number of reservoirs')

    parser.add_argument('--orders', type=str, default='10')
    parser.add_argument('--basename', type=str, default='qrc_narma_innate')
    parser.add_argument('--savedir', type=str, default='resnarma_innate')
    parser.add_argument('--ranseed', type=int, default=1)
    parser.add_argument('--trainloops', type=int, default=1)
    args = parser.parse_args()
    print(args)

    n_units, max_energy, beta = args.units, args.coupling, args.beta
    train_len, val_len, buffer = args.trainlen, args.vallen, args.transient
    V = args.virtuals
    init_rho, solver = args.rho, args.solver

    Ntrials, ranseed, train_loops = args.ntrials, args.ranseed, args.trainloops

    basename, savedir = args.basename, args.savedir
    if os.path.isdir(savedir) == False:
        os.mkdir(savedir)
    tau, alpha, nqrc = args.taudelta, args.strength, args.nqrc
    orders = [int(x) for x in args.orders.split(',')]

    # Evaluation
    timestamp = int(time.time() * 1000.0)
    now = datetime.datetime.now()
    datestr = now.strftime('{0:%Y-%m-%d-%H-%M-%S}'.format(now))

    for order in orders:
        outbase = os.path.join(savedir, '{}_{}_{}_units_{}_V_{}_alpha_{}_QRs_{}_narma_{}_ntrials_{}_loops_{}'.format(\
            basename, solver, datestr, n_units, V, alpha, nqrc, order, Ntrials, train_loops))

            # save experiments setting
        with open('{}_setting.txt'.format(outbase), 'w') as sfile:
            sfile.write('train_len={}, val_len={}, buffer={}\n'.format(train_len, val_len, buffer))
            sfile.write('n_units={}\n'.format(n_units))
            sfile.write('max_energy={}\n'.format(max_energy))
            sfile.write('beta={}\n'.format(beta))
            sfile.write('taudelta={}\n'.format(tau))
            sfile.write('layers={}\n'.format(nqrc))
            sfile.write('V={}\n'.format(V))
            sfile.write('alpha={}, Ntrials={}\n'.format(alpha, Ntrials))

        np.random.seed(seed=0)
        # Create input - target
        data, target = make_data_for_narma(train_len + val_len + buffer, orders=[order])
        
        train_input_seq_org = np.array(data[: buffer + train_len])
        train_input_seq_org = train_input_seq_org.reshape(1, train_input_seq_org.shape[0])
        train_output_seq = target[  : buffer + train_len] 

        val_input_seq_org =  np.array(data[buffer + train_len : buffer + train_len + val_len])
        val_input_seq_org = val_input_seq_org.reshape(1, val_input_seq_org.shape[0])
        val_output_seq = target[buffer + train_len : buffer + train_len + val_len]

        train_input_seq = np.tile(train_input_seq_org, (nqrc, 1))
        val_input_seq = np.tile(val_input_seq_org, (nqrc, 1))

        # Create qparams and model
        qparams = QRCParams(n_units=n_units, max_energy=max_energy,\
            beta=beta, virtual_nodes=V, tau=tau, init_rho=False)
        model = hqrc.HQRC(nqrc, alpha)

        # Create innate target activity
        target_state_list = model.init_forward(qparams, train_input_seq, ranseed=ranseed)
        N_local = model.get_local_nodes()
        target_innate_seq = target_state_list[:, ::N_local]
        print('target_state_list={}, target_innate_seq={}'.format(target_state_list.shape, target_innate_seq.shape))

        # Training innate
        # qparams.init_rho = True
        #model.init_forward(qparams, train_input_seq, ranseed=ranseed)
        trained_state_list, dW_recurr_ls = model.innate_train(train_input_seq, target_innate_seq, buffer, \
            ranseed=ranseed+1000, learn_every=1, noise_amp=0.0, train_loops=train_loops)

        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(nqrc+2, 1, 1)
        ax.plot(train_input_seq[0, :])
        ax.set_ylabel('Input')
        ax.set_title(outbase)

        for i in range(nqrc):
            ax = fig.add_subplot(nqrc+2, 1, i+2)
            ax.plot(target_state_list[:, N_local * i])
            ax.plot(trained_state_list[:, N_local * i])
            #ax.plot(trained_state_list[:, N_local * i] - target_state_list[:, N_local * i])
            ax.set_ylabel('QR_{}'.format(i))
        
        ax = fig.add_subplot(nqrc+2, 1, nqrc+2)
        ax.plot(dW_recurr_ls)
        ax.set_ylabel('dW')
        ax.set_xlabel('Time step')
        for ftype in ['png']:
            plt.savefig('{}.{}'.format(outbase, ftype), bbox_inches='tight', dpi=600)
        plt.show()
        #print(np.sum(target_innate_seq[:, :] - test_state_list[:, ::N_local]))



