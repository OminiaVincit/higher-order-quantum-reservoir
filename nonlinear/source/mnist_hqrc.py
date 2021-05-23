#!/usr/bin/env python
"""
Separate spiral data in nonlinear map
"""

import sys
import numpy as np
import os
import argparse
import multiprocessing
import matplotlib.pyplot as plt
import time

import hqrc as hqrc
from loginit import get_module_logger

from utils import *
from mnist_utils import *

MNIST_DIR = "../mnist"
RES_MNIST_DIR = "../results/rs_mnist"
MNIST_SIZE="10x10"

def get_acc_from_series(y_preds, y_lbs):
    y_preds = np.array([softmax(a) for a in y_preds])
    imlength = int(y_preds.shape[0] / len(y_lbs))
    y_preds = group_avg(y_preds, imlength)
    acc = get_acc(y_preds, y_lbs)
    return acc

def training_reservoir_states(logger, qparams, nqrs, alpha, buffer, use_corr, train_seq, test_seq, ranseed):
    if linear_reg < 0:
        logger.debug('Linear regression')
    else:
        tau = qparams.tau
        logger.debug('ranseed={}, Start regression with QR by tau={}, alpha={}, shape train={}, test={}'.format(\
            ranseed, tau, alpha, train_seq['input'].shape, test_seq['input'].shape))
        #input_signals = np.array(Xs)
        #rstates = model.init_forward(qparams, input_signals, init_rs=init_rs, ranseed = ranseed)
        #_, rstates =  model.feed_forward(input_signals, predict=False, use_lastrho=use_lastrho)
        train_pred_seq, train_loss, test_pred_seq, val_loss = hqrc.get_loss(qparams, buffer, \
            train_seq['input'], train_seq['output'], \
            test_seq['input'], test_seq['output'], \
            nqrc=nqrs, gamma=alpha, ranseed=ranseed, deep=0, use_corr=use_corr)
        
        logger.debug('ranseed={}, tau={}, alpha={}, (not real) loss train={}, val={}'.format(\
            ranseed, tau, alpha, train_loss, val_loss))
        train_acc = get_acc_from_series(train_pred_seq, train_seq['label'])
        test_acc  = get_acc_from_series(test_pred_seq, test_seq['label'])

        logger.info('ranseed={}, Finish regression with QR by tau={}, alpha={}, train_acc={}, test_acc={}'.format(\
            ranseed, tau, alpha, train_acc, test_acc))
    #rstr = '{} {:.10f} {:.10f} {:.10f}'.format(alpha, qparams.tau, train_acc, test_acc)
    #send_end.send(rstr)

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--spins', type=int, default=5, help='Number of the spins')
    parser.add_argument('--coupling', type=float, default=1.0, help='Maximum coupling energy')
    parser.add_argument('--rho', type=int, default=0, help='Flag for initializing the density matrix')
    parser.add_argument('--beta', type=float, default=1e-14, help='reg term')
    parser.add_argument('--solver', type=str, default=LINEAR_PINV, \
        help='regression solver by linear_pinv,ridge_pinv,auto,svd,cholesky,lsqr,sparse_cg,sag')
    parser.add_argument('--dynamic', type=str, default=DYNAMIC_FULL_RANDOM,\
        help='full_random,half_random,full_const_trans,full_const_coeff,ion_trap')

    parser.add_argument('--transient', type=int, default=0, help='Transitient time steps')
    
    parser.add_argument('--nqrs', type=int, default=1, help='Number of reservoirs')
    parser.add_argument('--ntrials', type=int, default=1)
    parser.add_argument('--virtuals', type=int, default=1)
    parser.add_argument('--strengths', type=str, default='0.0', help='Connection strengths')
    parser.add_argument('--taudeltas', type=str, default='-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7')
    parser.add_argument('--interval', type=float, default=0.2, help='tau-interval')
    
    parser.add_argument('--linear_reg', type=int, default=0)
    parser.add_argument('--use_corr', type=int, default=0)
    parser.add_argument('--full', type=int, default=0)
    parser.add_argument('--label1', type=int, default=3)
    parser.add_argument('--label2', type=int, default=6)
    
    parser.add_argument('--savedir', type=str, default=RES_MNIST_DIR)
    parser.add_argument('--mnist_dir', type=str, default=MNIST_DIR)
    parser.add_argument('--mnist_size', type=str, default=MNIST_SIZE)
    parser.add_argument('--nproc', type=int, default=100)
    parser.add_argument('--rseed', type=int, default=0)
    parser.add_argument('--rate', type=float, default=1.0)
    
    args = parser.parse_args()
    print(args)

    n_qrs, n_spins, beta, rseed = args.nqrs, args.spins, args.beta, args.rseed
    J, init_rho, V, transient = args.coupling, args.rho, args.virtuals, args.transient
    
    solver, linear_reg, use_corr, transient = args.solver, args.linear_reg, args.use_corr, args.transient
    full_mnist, label1, label2 = args.full, args.label1, args.label2
    ntrials, dynamic, savedir = args.ntrials, args.dynamic, args.savedir
    mnist_dir, mnist_size, nproc, rate = args.mnist_dir, args.mnist_size, args.nproc, args.rate

    taudeltas = [float(x) for x in args.taudeltas.split(',')]
    #taudeltas = list(np.arange(-7, 7.1, args.interval))
    taudeltas = [2**x for x in taudeltas]
    strengths = [float(x) for x in args.strengths.split(',')]

    if os.path.isfile(savedir) == False and os.path.isdir(savedir) == False:
        os.mkdir(savedir)

    if os.path.isdir(savedir) == True:
        logdir = os.path.join(savedir, 'log')
        figdir = os.path.join(savedir, 'figs')
        bindir = os.path.join(savedir, 'binary')
    else:
        logdir = os.path.join(os.path.dirname(__file__), 'log')
        figdir = os.path.join(os.path.dirname(__file__), 'figs')
        bindir = os.path.join(os.path.dirname(__file__), 'binary')

    if os.path.isdir(logdir) == False:
        os.mkdir(logdir)
    
    if os.path.isdir(figdir) == False:
        os.mkdir(figdir)

    if os.path.isdir(bindir) == False:
        os.mkdir(bindir)

    basename = 'join_{}_{}_linear_{}_nqrs_{}_corr_{}_nspins_{}_V_{}_rate_{}_trials_{}'.format(\
        mnist_size, dynamic, linear_reg, n_qrs, use_corr, n_spins, V, rate, ntrials)
    
    x_train, y_train_lb, x_test, y_test_lb = gen_mnist_dataset_join_test(mnist_dir, mnist_size)
    imlength = int(x_train.shape[1] / n_qrs)
    train_seq = dict()
    test_seq  = dict()

    if full_mnist <= 0:
        basename = '{}_lb_{}_{}'.format(basename, label1, label2)
        # Train data
        train_ids = (y_train_lb == label1) | (y_train_lb == label2)
        x_train = x_train[train_ids, :]
        y_train_lb = y_train_lb[train_ids]
        y_train_lb[y_train_lb == label1] = 0
        y_train_lb[y_train_lb == label2] = 1

        # Test data 
        test_ids = (y_test_lb == label1) | (y_test_lb == label2)
        x_test = x_test[test_ids, :]
        y_test_lb = y_test_lb[test_ids]
        y_test_lb[y_test_lb == label1] = 0
        y_test_lb[y_test_lb == label2] = 1

    log_filename = os.path.join(logdir, '{}_softmax.log'.format(basename))
    logger = get_module_logger(__name__, log_filename)
    logger.info(log_filename)
    logger.info(args)
    logger.info('Original shape x_train={}, y_train={}, x_test={}, y_test={}'.format(\
        x_train.shape, y_train_lb.shape, x_test.shape, y_test_lb.shape))

    buffer = imlength * transient
    logger.info('Transient={}, buffer={}, train rate={}'.format(transient, buffer, rate))
    
    for n in range(ntrials):
        ranseed = rseed + 10000*n
        np.random.seed(seed=ranseed)
        # Permute data
        N_train, N_test = len(y_train_lb), len(y_test_lb)
        nm_train, nm_test = int(N_train*rate), int(N_test*rate)
        train_idx = np.random.permutation(N_train)[:nm_train]
        test_idx = np.random.permutation(N_test)
        
        x_train, y_train_lb = x_train[train_idx, :], y_train_lb[train_idx]
        x_test, y_test_lb  = x_test[test_idx, :], y_test_lb[test_idx]

        logger.info('trials={}, ranseed={}, reduce shape x_train={}, y_train={}, x_test={}, y_test={}'.format(\
            n, ranseed, x_train.shape, y_train_lb.shape, x_test.shape, y_test_lb.shape))

        # Make input_seq to put in hqrc
        train_seq['input'] = [x.reshape(n_qrs, -1) for x in x_train]
        train_seq['input'] = np.concatenate(train_seq['input'], axis=1)

        test_seq['input'] = [x.reshape(n_qrs, -1) for x in x_test]
        test_seq['input'] = np.concatenate(test_seq['input'], axis=1)

        if full_mnist > 0:
            numlb = 10
        else:
            numlb = 2
        
        train_seq['label']  = y_train_lb
        train_seq['output'] = np.identity(numlb)[y_train_lb]
        train_seq['output'] = np.repeat(train_seq['output'], imlength, axis=0)
        
        test_seq['label']  = y_test_lb
        test_seq['output']  = np.identity(numlb)[y_test_lb]
        test_seq['output'] = np.repeat(test_seq['output'], imlength, axis=0)
        
        jobs, pipels = [], [] 
        for alpha in strengths:
            for tau in taudeltas:
                #recv_end, send_end = multiprocessing.Pipe(False)
                # Create params and model
                qparams = QRCParams(n_units=n_spins-1, n_envs=1, max_energy=J,\
                    beta=beta, virtual_nodes=V, tau=tau, init_rho=init_rho, solver=solver, dynamic=dynamic)
                p = multiprocessing.Process(target=training_reservoir_states, \
                    args=(logger, qparams, n_qrs, alpha, buffer, use_corr, train_seq, test_seq, ranseed))
                jobs.append(p)
                #pipels.append(recv_end)

        # Start the process
        for p in jobs:
            p.start()

        # Ensure all processes have finished execution
        for p in jobs:
            p.join()

        # Sleep 5s
        time.sleep(5)
