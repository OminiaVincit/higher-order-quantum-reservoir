#!/usr/bin/env python
"""
    Calculate NRMSE for NARMA tasks
    See run_hqrc_nrmse_narma.sh for an example to run the script
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

def compute_job(qparams, nqrc, deep, alpha, buffer, train_input_seq, train_output_seq, \
        val_input_seq, val_output_seq, Ntrials, send_end, save_order, save_path, load_order, load_path, combine_input):
    train_loss_ls, val_loss_ls = [], []
    print('Start process alpha={}, taudelta={}, virtual={}, Jdelta={}'.format(\
        alpha, qparams.tau, qparams.virtual_nodes, qparams.max_energy))
    
    basename = 'train_{}_nqr_{}_deep_{}_tau_{:.3f}_V_{}'.format(\
        train_input_seq.shape[1], nqrc, deep, qparams.tau, qparams.virtual_nodes)

    if load_path != None:
        load_path = os.path.join(load_path, 'order_{}_{}'.format(load_order, basename))
    if save_path != None:
        save_path = os.path.join(save_path, 'order_{}_{}'.format(save_order, basename))

    for n in range(Ntrials):
        local_save_path, local_load_path = None, None
        if save_path != None:
            local_save_path = os.path.join(save_path, 'trial_{}'.format(n))
        if load_path != None:
            local_load_path = os.path.join(load_path, 'trial_{}'.format(n))
            
        _, train_loss, _, val_loss = hqrc.get_loss(qparams, buffer, train_input_seq, train_output_seq, \
            val_input_seq, val_output_seq, nqrc=nqrc, gamma=alpha, ranseed=n, deep=deep, saving_path=local_save_path, loading_path=local_load_path, combine_input=combine_input)
        train_loss_ls.append(train_loss)
        val_loss_ls.append(val_loss)
        print('trials={}, tau={},V={},alpha={}, train_loss={}, val_loss={}'.format(\
            n, qparams.tau, qparams.virtual_nodes, alpha, train_loss, val_loss))

    mean_train, mean_val = np.mean(train_loss_ls), np.mean(val_loss_ls)
    std_train, std_val = np.std(train_loss_ls), np.std(val_loss_ls)
    #mean_train, mean_val = np.random.rand(), np.random.rand()

    rstr = '{} {} {} {} {} {} {} {}'.format(\
        save_order, nqrc, qparams.tau, alpha, \
            mean_train, mean_val, std_train, std_val)
    print('Finish process {}'.format(rstr))
    send_end.send(rstr)

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--units', type=int, default=5, help='Number of the hidden units')
    parser.add_argument('--coupling', type=float, default=1.0, help='Maximum coupling energy')
    parser.add_argument('--rho', type=int, default=0, help='Flag for initializing the density matrix')
    parser.add_argument('--beta', type=float, default=1e-14, help='reg term')
    parser.add_argument('--solver', type=str, default=LINEAR_PINV, \
        help='regression solver by linear_pinv,ridge_pinv,auto,svd,cholesky,lsqr,sparse_cg,sag')
    parser.add_argument('--dynamic', type=str, default=DYNAMIC_FULL_RANDOM,\
        help='full_random,half_random,full_const_trans,full_const_coeff,ion_trap')

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
    parser.add_argument('--load_order', type=str, default='2')
    parser.add_argument('--savedir', type=str, default='resnarma_hqrc')
    parser.add_argument('--rseed', type=int, default=0)
    parser.add_argument('--save_model', type=int, default=0)
    parser.add_argument('--load_model', type=int, default=0)
    parser.add_argument('--combine_input', type=int, default=1)
    
    args = parser.parse_args()
    print(args)

    n_units, max_energy, beta = args.units, args.coupling, args.beta
    train_len, val_len, buffer = args.trainlen, args.vallen, args.transient
    V, rseed = args.virtuals, args.rseed
    init_rho, solver = args.rho, args.solver

    Ntrials = args.ntrials
    deep = args.deep

    dynamic, savedir = args.dynamic, args.savedir
    if os.path.isdir(savedir) == False:
        os.mkdir(savedir)

    save_model, load_model, load_order = args.save_model, args.load_model, args.load_order
    save_path, load_path = None, None
    combine_input = args.combine_input

    if save_model > 0:
        save_path = os.path.join(savedir, 'saved_model')
    if load_model > 0:
        load_path = os.path.join(savedir, 'saved_model')

    if args.taudelta == 'default':
        taudeltas = list(np.arange(-5, 7.1, 0.1))
    else:
        tstr = args.taudelta.replace('\'','')
        taudeltas = [float(x) for x in tstr.split(',')]
    taudeltas = [2**x for x in taudeltas]
    
    layers = [int(x) for x in args.nqrc.split(',')]
    strengths = [float(x) for x in args.strengths.split(',')]
    orders = [int(x) for x in args.orders.split(',')]

    # Evaluation
    timestamp = int(time.time() * 1000.0)
    now = datetime.datetime.now()
    datestr = now.strftime('{0:%Y-%m-%d-%H-%M-%S}'.format(now))

    for order in orders:
        outbase = os.path.join(savedir, '{}_{}_{}_units_{}_V_{}_QRs_{}_narma_{}_deep_{}_ntrials_{}_load_{}_od_{}_cb_{}'.format(\
            dynamic, solver, datestr, n_units, V,\
            #'_'.join([str(o) for o in strengths]), \
            '_'.join([str(o) for o in layers]), \
            order, deep, Ntrials, load_model, load_order, combine_input))
        #np.random.seed(seed=rseed + order*100)

        jobs, pipels = [], []
        data, target = make_data_for_narma(train_len + val_len + buffer, orders=[order], ranseed=rseed + order*100)

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
                    qparams = QRCParams(n_units=n_units-1, n_envs=1, max_energy=max_energy,\
                        beta=beta, virtual_nodes=V, tau=tau, init_rho=init_rho, solver=solver, dynamic=dynamic)
                    p = multiprocessing.Process(target=compute_job, args=(qparams, nqrc, deep, alpha, buffer, train_input_seq, train_output_seq, \
                        val_input_seq, val_output_seq, Ntrials, send_end, order, save_path, load_order, load_path, combine_input))
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
        np.savetxt('{}_NRMSE.txt'.format(outbase), rsarr, delimiter=' ')

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
            sfile.write('ranseed={}\n'.format(rseed))
            sfile.write('alpha={}, Ntrials={}\n'.format(alpha, Ntrials))


