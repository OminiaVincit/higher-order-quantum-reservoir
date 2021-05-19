#!/usr/bin/env python
"""
Separate spiral data in nonlinear map
"""

import sys
import numpy as np
import os
import scipy
import argparse
import multiprocessing
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import ticker
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
from scipy.sparse import linalg as splinalg
from scipy.linalg import pinv2 as scipypinv2

import time
import datetime
import pickle

import hqrc as hqrc
from loginit import get_module_logger

import utils
from utils import *
from mnist_utils import *

MNIST_DIR = "../mnist"
RES_MNIST_DIR = "../results/rs_mnist"
MNIST_SIZE="10x10"

def dump_reservoir_states(logger, datfile, Xs, qparams, model, init_rs, ranseed):
    if os.path.isfile(datfile) == True:
        logger.debug('File existed {}'.format(datfile))
    else:
        input_signals = np.array(Xs)
        rstates = model.init_forward(qparams, input_signals, init_rs=init_rs, ranseed = ranseed)
        #_, rstates =  model.feed_forward(input_signals, predict=False, use_lastrho=use_lastrho)
        rstates = np.array(rstates)
        with open(datfile, 'wb') as wrs:
            pickle.dump(rstates, wrs)
            logger.debug('Dumped file {}'.format(datfile))
    return datfile

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

    parser.add_argument('--transient', type=int, default=200, help='Transitient time steps')
    
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
    args = parser.parse_args()
    print(args)

    n_qrs, n_spins, beta, rseed = args.nqrs, args.spins, args.beta, args.rseed
    J, init_rho, V = args.coupling, args.rho, args.virtuals
    
    solver, linear_reg, use_corr, transient = args.solver, args.linear_reg, args.use_corr, args.transient
    full_mnist, label1, label2 = args.full, args.label1, args.label2
    ntrials, dynamic, savedir = args.ntrials, args.dynamic, args.savedir
    mnist_dir, mnist_size, nproc = args.mnist_dir, args.mnist_size, args.nproc

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

    basename = 'join_{}_{}_linear_{}_nqrs_{}_corr_{}_nspins_{}_V_{}'.format(\
        mnist_size, dynamic, linear_reg, n_qrs, use_corr, n_spins, V)
    
    x_train, y_train_lb, x_test, y_test_lb = gen_mnist_dataset_join_test(mnist_dir, mnist_size)
    imlength = int(x_train.shape[1] / n_qrs)

    if full_mnist > 0:
        Y_train_org = np.identity(10)[y_train_lb]
        Y_test_org  = np.identity(10)[y_test_lb]
    else:
        basename = '{}_lb_{}_{}'.format(basename, label1, label2)
        # Train data
        train_ids = (y_train_lb == label1) | (y_train_lb == label2)
        x_train = x_train[train_ids, :]
        y_train_lb = y_train_lb[train_ids]
        y_train_lb[y_train_lb == label1] = 0
        y_train_lb[y_train_lb == label2] = 1

        Y_train_org = np.identity(2)[y_train_lb]

        # Test data 
        test_ids = (y_test_lb == label1) | (y_test_lb == label2)
        x_test = x_test[test_ids, :]
        y_test_lb = y_test_lb[test_ids]
        y_test_lb[y_test_lb == label1] = 0
        y_test_lb[y_test_lb == label2] = 1

        Y_test_org  = np.identity(2)[y_test_lb]

    log_filename = os.path.join(logdir, '{}.log'.format(basename))
    logger = get_module_logger(__name__, log_filename)
    logger.info(log_filename)
    logger.info('shape x_train={}, y_train={},  x_test={}, y_test={}'.format(\
        x_train.shape, Y_train_org.shape, x_test.shape, Y_test_org.shape))

    datfs = dict()

    for datlb in ['train', 'test']:
        if linear_reg > 0:
            continue
        if datlb == 'train':
            Xt = x_train
        else:
            Xt  = x_test
        # Make Xs to put in hqrc
        Xs = [x.reshape(n_qrs, -1) for x in Xt]
        Xs = np.concatenate(Xs, axis=1)
        # Create training/test file
        (ns, ds) = Xt.shape
        logger.info('{} file in time series {}'.format(datlb, Xs.shape))
        jobs, pipels = [], [] 
        for alpha in strengths:
            for tau in taudeltas:
                # Create params and model
                qparams = QRCParams(n_units=n_spins-1, n_envs=1, max_energy=J,\
                    beta=beta, virtual_nodes=V, tau=tau, init_rho=init_rho, solver=solver, dynamic=dynamic)
                model = hqrc.HQRC(nqrc=n_qrs, gamma=alpha, sparsity=1.0, sigma_input=1.0, use_corr=use_corr)
                model.init_reservoir(qparams, ranseed=rseed)
                init_rs, rs_seed, use_lastrho = False, 0, False
                # create training and testing data
                datfile = '{}_{}_alpha_{:.2f}_tau_{:.4f}_shape_{}_{}.bin'.format(datlb, basename, alpha, tau, ns, ds)
                datfile = os.path.join(bindir, datfile)
                datfs['{}_alpha_{:.2f}_tau_{:.4f}'.format(datlb, alpha, tau)] = datfile
                p = multiprocessing.Process(target=dump_reservoir_states, \
                    args=(logger, datfile, Xs, qparams, model, init_rs, rs_seed))
                jobs.append(p)
        # Start the process
        for p in jobs:
            p.start()

        # Ensure all processes have finished execution
        for p in jobs:
            p.join()

        # Sleep 5s
        time.sleep(5)
    
    # perform regression
    for alpha in strengths:
        for tau in taudeltas:
            if linear_reg > 0:
                X_train = np.array(x_train)
                X_train = [x.reshape(n_qrs, -1) for x in X_train]
                X_train = np.concatenate(X_train, axis=1).T
                #if full_mnist == 0:
                #    X_train = X_train[train_ids, :]
            else:
                # Training
                with open(datfs['train_alpha_{:.2f}_tau_{:.4f}'.format(alpha, tau)], 'rb') as rrs:
                    X_train = pickle.load(rrs)
                    # if use_corr == 0:
                    #     ids = np.array(range(X_train.shape[1]))
                    #     ids = ids[ids % 15 < 5]
                    #     X_train = X_train[:, ids]
                    #if full_mnist == 0:
                    #    X_train = X_train[train_ids, :]
            Y_train = np.repeat(Y_train_org, imlength, axis=0)
            X_train = np.hstack( [X_train, np.ones([X_train.shape[0], 1]) ] )
            logger.info('Nqr={}, V={}, alpha={}, tau={}, X_train shape={}, Y_train shape={}'.format(\
                n_qrs, V, alpha, tau, X_train.shape, Y_train.shape))

            #XTX = X_train.T @ X_train
            #XTY = X_train.T @ Y_train
            #I = np.identity(np.shape(XTX)[1])	
            #pinv_ = scipypinv2(XTX + beta * I)
            #W_out = pinv_ @ XTY
            W_out = np.linalg.pinv(X_train, rcond = beta) @ Y_train
            logger.info('Wout shape={}'.format(W_out.shape))
            y_train_predict = group_avg(X_train @ W_out, imlength)
            logger.info('y_train_predict shape={}'.format(y_train_predict.shape))
            
            train_acc = get_acc(y_train_predict, y_train_lb)
            logger.info('Nqr={}, V={}, tau={}, alpha={}, Train acc={}'.format(n_qrs, V, tau, alpha, train_acc))

            # Testing
            if linear_reg > 0:
                X_test = np.array(x_test)
                X_test = [x.reshape(n_qrs, -1) for x in X_test]
                X_test = np.concatenate(X_test, axis=1).T
                #if full_mnist == 0:
                #    X_test = X_test[test_ids, :]
            else:
                with open(datfs['test_alpha_{:.2f}_tau_{:.4f}'.format(alpha, tau)], 'rb') as rrs:
                    X_test = pickle.load(rrs)
                    # if use_corr == 0:
                    #     ids = np.array(range(X_test.shape[1]))
                    #     ids = ids[ids % 15 < 5]
                    #     X_test = X_test[:, ids]
                    #if full_mnist == 0:
                    #    X_test = X_test[test_ids, :]
            Y_test = np.repeat(Y_test_org, imlength, axis=0)
            X_test = np.hstack( [X_test, np.ones([X_test.shape[0], 1]) ] )
            logger.info('Nqr={}, V={}, tau={}, alpha={}, X_test shape = {}'.format(n_qrs, V, tau, alpha, X_test.shape))
            y_test_predict = group_avg(X_test @ W_out, imlength)
            logger.info('y_test_predict shape={}'.format(y_test_predict.shape))
            test_acc = get_acc(y_test_predict, y_test_lb)
            logger.info('Nqr={}, V={}, tau={}, alpha={}, Test acc={}'.format(n_qrs, V, tau, alpha, test_acc))
