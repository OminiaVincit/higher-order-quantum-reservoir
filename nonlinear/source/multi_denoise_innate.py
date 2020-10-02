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
from loginit import get_module_logger

def narma_with_pre_train_job(logdir, tmpbase, qparams, nqrc, \
    order, train_len, val_len, buffer, \
    alpha, Ntrials, noise_amp, learning_rate, scale_input, train_loops, sel):
        
    log_filename = os.path.join(logdir, '{}.log'.format(tmpbase))
    logger = get_module_logger(__name__, log_filename)

    logger.info('n_units={}'.format(qparams.n_units))
    logger.info('max_energy={}'.format(qparams.max_energy))
    logger.info('beta={}'.format(qparams.beta))
    logger.info('taudelta={}'.format(qparams.tau))
    logger.info('V={}'.format(qparams.virtual_nodes))
    logger.info('layers={}'.format(nqrc))

    logger.info('train_len={}, val_len={}, buffer={}'.format(train_len, val_len, buffer))
    logger.info('alpha={}, Ntrials={}'.format(alpha, Ntrials))
    logger.info('noise={}, learning rate={},scale_input={}'.format(noise_amp, learning_rate, scale_input))

    val_loss_ls = []
    for n in range(Ntrials):
        np.random.seed(seed=n)
        # Create input - target
        data, target = make_data_for_narma(train_len + val_len + buffer, orders=[order])
        #data = np.random.randint(2, size=train_len + val_len)

        model = hqrc.HQRC(nqrc, alpha)
        new_ranseed = ranseed + n * 100
        
        model.init_model(qparams, ranseed=new_ranseed)
        
        # Pre training
        if learning_rate > 0.0:
            # For PRE Training
            innate_buffer = buffer
            innate_train_len = train_len
            innate_val_len = val_len

            pre_input_seq_org = np.array(data[: innate_buffer + innate_train_len + innate_val_len])
            pre_input_seq_org = pre_input_seq_org.reshape(1, pre_input_seq_org.shape[0])
            pre_input_seq = np.tile(pre_input_seq_org, (nqrc, 1))
            
            # Create innate target activity
            target_state_list = model.init_forward(qparams, pre_input_seq, ranseed=new_ranseed, noise_amp=0.0, scale_input=scale_input)
            N_local = model.get_local_nodes()
            n_qubits = model.n_qubits

            target_innate_seq = target_state_list[:, sel::N_local]
            logger.info('innate_buffer={}, innate_train_len={}, innate_val_len={}'.format(innate_buffer, innate_train_len, innate_val_len))
            logger.info('target_state_list={}, target_innate_seq={}'.format(target_state_list.shape, target_innate_seq.shape))

            trained_state_list, dW_recurr_ls, loss_dict \
                = model.innate_train(pre_input_seq, target_innate_seq, innate_buffer, innate_train_len, \
                    ranseed=new_ranseed + 1000, learn_every=1, \
                    noise_amp=noise_amp, learning_rate=learning_rate, \
                    scale_input=scale_input, train_loops=train_loops)

        train_input_seq_org = np.array(data[: buffer + train_len])
        train_input_seq_org = train_input_seq_org.reshape(1, train_input_seq_org.shape[0])
        train_output_seq = target[  : buffer + train_len] 

        val_input_seq_org =  np.array(data[buffer + train_len : buffer + train_len + val_len])
        val_input_seq_org = val_input_seq_org.reshape(1, val_input_seq_org.shape[0])
        val_output_seq = target[buffer + train_len : buffer + train_len + val_len]

        train_input_seq = np.tile(train_input_seq_org, (nqrc, 1))
        val_input_seq = np.tile(val_input_seq_org, (nqrc, 1))

        model.train(train_input_seq, train_output_seq, buffer=buffer, beta=beta, \
            ranseed=new_ranseed + 2000, noise_amp=noise_amp, scale_input=scale_input)
        val_pred_seq, val_loss = model.predict(val_input_seq, val_output_seq, buffer=0, \
            noise_amp=noise_amp, scale_input=scale_input)
        logger.debug('n={}, val_loss={}'.format(n, val_loss))
        val_loss_ls.append(val_loss)
    logger.info('Ntrials={}, total loss avg={}, std={}'.format(Ntrials, np.mean(val_loss_ls), np.std(val_loss_ls)))

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--units', type=int, default=5, help='Number of the hidden units')
    parser.add_argument('--coupling', type=float, default=1.0, help='Maximum coupling energy')
    parser.add_argument('--rho', type=int, default=0, help='Flag for initializing the density matrix')
    parser.add_argument('--beta', type=float, default=1e-14, help='reg term')
    parser.add_argument('--solver', type=str, default=LINEAR_PINV, \
        help='regression solver by linear_pinv,ridge_pinv,auto,svd,cholesky,lsqr,sparse_cg,sag')

    parser.add_argument('--trainlen', type=int, default=2000)
    parser.add_argument('--vallen', type=int, default=2000)
    parser.add_argument('--transient', type=int, default=2000, help='Transitient time steps')
    
    parser.add_argument('--ntrials', type=int, default=1)
    parser.add_argument('--virtuals', type=int, default=1)
    parser.add_argument('--taudelta', type=float, default=2.0, help='Interval between the inputs')
    parser.add_argument('--strength', type=float, default=0.5, help='Connection strengths')
    parser.add_argument('--nqrc', type=int, default=5, help='Number of reservoirs')

    parser.add_argument('--orders', type=str, default='10')
    parser.add_argument('--basename', type=str, default='qrc_innate')
    parser.add_argument('--savedir', type=str, default='de-narma')
    parser.add_argument('--ranseed', type=int, default=1)
    parser.add_argument('--trainloops', type=int, default=1)
    parser.add_argument('--select_qubit', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=10.0)
    parser.add_argument('--scale_input', type=float, default=0.4)
    args = parser.parse_args()
    print(args)

    n_units, max_energy, beta = args.units, args.coupling, args.beta
    train_len, val_len, buffer = args.trainlen, args.vallen, args.transient
    V = args.virtuals
    init_rho, solver = args.rho, args.solver

    Ntrials, ranseed, train_loops = args.ntrials, args.ranseed, args.trainloops
    learning_rate, scale_input, sel = args.learning_rate, args.scale_input, args.select_qubit

    basename, savedir = args.basename, args.savedir
    if os.path.isdir(savedir) == False:
        os.mkdir(savedir)
    
    logdir = os.path.join(savedir, 'log')
    if os.path.isdir(logdir) == False:
        os.mkdir(logdir)

    tau, alpha, nqrc = args.taudelta, args.strength, args.nqrc
    orders = [int(x) for x in args.orders.split(',')]

    # Evaluation
    timestamp = int(time.time() * 1000.0)
    now = datetime.datetime.now()
    datestr = now.strftime('{0:%Y-%m-%d-%H-%M-%S}'.format(now))

    
    noise_ls = []
    for i in [-6, -5, -4, -3]:
        for j in range(1, 10):
            noise_ls.append(j * (10**i))
    noise_ls.append(1e-2)

    # Create qparams and model
    qparams = QRCParams(n_units=n_units, max_energy=max_energy,\
        beta=beta, virtual_nodes=V, tau=tau, init_rho=False)

    jobs = []
    for order in orders:
        for noise_amp in noise_ls:
            tmpbase = '{}_{}_{}_units_{}_V_{}_a_{}_QR_{}_narma_{}_n_{}_lo_{}_no_{}_r_{}_sc_{}_sd_{}'.format(\
                basename, solver, datestr, n_units, V, alpha, nqrc, order, Ntrials, train_loops, noise_amp, learning_rate, scale_input, ranseed)
            p = multiprocessing.Process(target=narma_with_pre_train_job, \
                args=(logdir, tmpbase, qparams, nqrc, \
                    order, train_len, val_len, buffer, \
                    alpha, Ntrials, noise_amp, learning_rate, scale_input, train_loops, sel))
            jobs.append(p)

    # Start the process
    for p in jobs:
        p.start()

    # Ensure all processes have finished execution
    for p in jobs:
        p.join()



