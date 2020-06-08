#!/usr/bin/env python
"""
Quoc Hoan Tran, Nakajima-Lab, The University of Tokyo
    Calculate NMSE for NARMA tasks
    See run_hqrc_nmse_narma.sh for an example to run the script
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

def nmse_job(qparams, nqrc, deep, alpha, buffer, train_input_seq, train_output_seq, \
        val_input_seq, val_output_seq, Ntrials, send_end, order):
    train_loss_ls, val_loss_ls = [], []
    print('Start process alpha={}, taudelta={}, virtual={}, Jdelta={}'.format(\
        alpha, qparams.tau, qparams.virtual_nodes, qparams.max_energy))
    for n in range(Ntrials):
        _, train_loss, _, val_loss = hqrc.get_loss(qparams, buffer, train_input_seq, train_output_seq, \
            val_input_seq, val_output_seq, nqrc, alpha, ranseed=n, deep=deep)
        train_loss_ls.append(train_loss)
        val_loss_ls.append(val_loss)

    mean_train, mean_val = np.mean(train_loss_ls), np.mean(val_loss_ls)
    std_train, std_val = np.std(train_loss_ls), np.std(val_loss_ls)
    #mean_train, mean_val = np.random.rand(), np.random.rand()

    rstr = '{} {} {} {} {} {} {} {}'.format(\
        order, nqrc, qparams.tau, alpha, \
            mean_train, mean_val, std_train, std_val)
    print('Finish process {}'.format(rstr))
    send_end.send(rstr)

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
    parser.add_argument('--taudelta', type=str, default='0.0,1.0,2.0', help='Interval between the inputs')
    parser.add_argument('--strengths', type=str, default='0.1,0.5,0.9', help='Connection strengths')
    parser.add_argument('--nqrc', type=str, default='5', help='Number of reservoirs')

    parser.add_argument('--deep', type=int, default=0, help='0: mutual connection, 1: forward connection')
    parser.add_argument('--orders', type=str, default='5,10,15,20')
    parser.add_argument('--basename', type=str, default='qrc_narma')
    parser.add_argument('--savedir', type=str, default='resnarma_hqrc')
    args = parser.parse_args()
    print(args)

    n_units, max_energy, beta = args.units, args.coupling, args.beta
    train_len, val_len, buffer = args.trainlen, args.vallen, args.transient
    V = args.virtuals
    init_rho, solver = args.rho, args.solver

    Ntrials = args.ntrials
    deep = False
    if args.deep > 0:
        deep = True

    basename, savedir = args.basename, args.savedir
    if os.path.isdir(savedir) == False:
        os.mkdir(savedir)

    taudeltas = [float(x) for x in args.taudelta.split(',')]
    taudeltas = [2**x for x in taudeltas]
    
    layers = [int(x) for x in args.nqrc.split(',')]
    strengths = [float(x) for x in args.strengths.split(',')]
    orders = [int(x) for x in args.orders.split(',')]

    # Evaluation
    timestamp = int(time.time() * 1000.0)
    now = datetime.datetime.now()
    datestr = now.strftime('{0:%Y-%m-%d-%H-%M-%S}'.format(now))

    for order in orders:
        outbase = os.path.join(savedir, '{}_{}_{}_units_{}_V_{}_alpha_{}_QRs_{}_narma_{}_ntrials_{}'.format(\
            basename, solver, datestr, n_units, V,\
            '_'.join([str(o) for o in strengths]), \
            '_'.join([str(o) for o in layers]), \
            order, Ntrials))

        jobs, pipels = [], []
        data, target = make_data_for_narma(train_len + val_len + buffer, orders=[order])

        train_input_seq_org = np.array(data[: buffer + train_len])
        train_input_seq_org = train_input_seq_org.reshape(1, train_input_seq_org.shape[0])
        
        train_output_seq = target[  : buffer + train_len] 

        val_input_seq_org =  np.array(data[buffer + train_len : buffer + train_len + val_len])
        val_input_seq_org = val_input_seq_org.reshape(1, val_input_seq_org.shape[0])
            
        val_output_seq = target[buffer + train_len : buffer + train_len + val_len]

        for nqrc in layers:
            train_input_seq = np.tile(train_input_seq_org, (nqrc, 1))
            val_input_seq = np.tile(val_input_seq_org, (nqrc, 1))
            for alpha in strengths:
                for tau in taudeltas:
                    recv_end, send_end = multiprocessing.Pipe(False)
                    qparams = QRCParams(n_units=n_units, max_energy=max_energy,\
                        beta=beta, virtual_nodes=V, tau=tau, init_rho=init_rho)
                    p = multiprocessing.Process(target=nmse_job, args=(qparams, nqrc, deep, alpha, buffer, train_input_seq, train_output_seq, \
                        val_input_seq, val_output_seq, Ntrials, send_end, order))
                    jobs.append(p)
                    pipels.append(recv_end)

        # Start the process
        for p in jobs:
            p.start()

        # Ensure all processes have finished execution
        for p in jobs:
            p.join()

        # Sleep 5s
        time.sleep(5)

        result_list = [np.array( [float(y) for y in x.recv().split(' ')]  ) for x in pipels]
        rsarr = np.array(result_list)
        # save the result
        np.savetxt('{}_NMSE.txt'.format(outbase), rsarr, delimiter=' ')

        # save experiments setting
        with open('{}_setting.txt'.format(outbase), 'w') as sfile:
            sfile.write('train_len={}, val_len={}, buffer={}\n'.format(train_len, val_len, buffer))
            sfile.write('n_units={}\n'.format(n_units))
            sfile.write('max_energy={}\n'.format(max_energy))
            sfile.write('beta={}\n'.format(beta))
            sfile.write('taudeltas={}\n'.format(' '.join([str(v) for v in taudeltas])))
            sfile.write('layers={}\n'.format(' '.join([str(l) for l in layers])))
            sfile.write('V={}\n'.format(V))
            sfile.write('deep={}\n'.format(deep))
            sfile.write('alpha={}, Ntrials={}\n'.format(alpha, Ntrials))


