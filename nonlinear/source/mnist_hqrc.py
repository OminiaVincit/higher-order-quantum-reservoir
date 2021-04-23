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

def get_reservoir_states(tempfile, Xs, qparams, model, init_rs, ranseed):
    rstates = []
    for s in Xs:
        input_signals = np.array(s)
        output_signals = model.init_forward(qparams, input_signals, init_rs=init_rs, ranseed = ranseed)
        rstates.append(output_signals.ravel())
    rstates = np.array(rstates)
    np.save(tempfile, rstates)
    return tempfile

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--spins', type=int, default=5, help='Number of the spins')
    parser.add_argument('--coupling', type=float, default=1.0, help='Maximum coupling energy')
    parser.add_argument('--rho', type=int, default=0, help='Flag for initializing the density matrix')
    parser.add_argument('--beta', type=float, default=1e-14, help='reg term')
    parser.add_argument('--solver', type=str, default=RIDGE_PINV, \
        help='regression solver by linear_pinv,ridge_pinv,auto,svd,cholesky,lsqr,sparse_cg,sag')
    parser.add_argument('--transient', type=int, default=200, help='Transitient time steps')
    
    parser.add_argument('--nqrs', type=int, default=10, help='Number of reservoirs')
    parser.add_argument('--ntrials', type=int, default=1)
    parser.add_argument('--virtuals', type=int, default=10)
    parser.add_argument('--strengths', type=float, default=0.0, help='Connection strengths')
    parser.add_argument('--tau', type=float, default=4.0, help='Input interval')

    parser.add_argument('--linear_reg', type=int, default=0)
    parser.add_argument('--use_corr', type=int, default=0)
    parser.add_argument('--full', type=int, default=1)
    parser.add_argument('--label1', type=int, default=3)
    parser.add_argument('--label2', type=int, default=6)
    
    parser.add_argument('--basename', type=str, default='qrc')
    parser.add_argument('--savedir', type=str, default=RES_MNIST_DIR)
    parser.add_argument('--mnist_dir', type=str, default=MNIST_DIR)
    parser.add_argument('--mnist_size', type=str, default=MNIST_SIZE)
    parser.add_argument('--nproc', type=int, default=100)
    parser.add_argument('--rseed', type=int, default=0)
    args = parser.parse_args()
    print(args)

    n_qrs, n_spins, beta, rseed = args.nqrs, args.spins, args.beta, args.rseed
    J, init_rho, tau, V, alpha = args.coupling, args.rho, args.tau, args.virtuals, args.strengths
    
    solver, linear_reg, use_corr, transient = args.solver, args.linear_reg, args.use_corr, args.transient
    full_mnist, label1, label2 = args.full, args.label1, args.label2
    ntrials, basename, savedir = args.ntrials, args.basename, args.savedir
    mnist_dir, mnist_size, nproc = args.mnist_dir, args.mnist_size, args.nproc

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

    basename = '{}_{}_nqrs_{}_nspins_{}'.format(mnist_size, basename, n_qrs, n_spins)
    log_filename = os.path.join(logdir, '{}_V_{}.log'.format(basename, args.virtuals))
    logger = get_module_logger(__name__, log_filename)
    logger.info(log_filename)

    x_train, y_train_lb, x_test, y_test_lb, x_val, y_val = gen_mnist_dataset(mnist_dir, mnist_size)
    
    if full_mnist > 0:
        Y_train = np.identity(10)[y_train_lb]
        Y_test  = np.identity(10)[y_test_lb]
    else:
        train_ids = (y_train_lb == label1) | (y_train_lb == label2)
        x_train = x_train[train_ids, :]
        y_train_lb = y_train_lb[train_ids]
        y_train_lb[y_train_lb == label1] = 0
        y_train_lb[y_train_lb == label2] = 1

        Y_train = np.identity(2)[y_train_lb]

        test_ids = (y_test_lb == label1) | (y_test_lb == label2)
        x_test = x_test[test_ids, :]
        y_test_lb = y_test_lb[test_ids]
        y_test_lb[y_test_lb == label1] = 0
        y_test_lb[y_test_lb == label2] = 1

        Y_test  = np.identity(2)[y_test_lb]
    
    logger.info('shape y_train={},  y_test={}'.format(Y_train.shape, Y_test.shape))

    # Create params and model
    qparams = QRCParams(n_units=n_units, max_energy=J,\
        beta=beta, virtual_nodes=V, tau=tau, init_rho=init_rho)
    model = hqrc.HQRC(nqrc=n_qrs, alpha=alpha)
    model.init_reservoir(qparams, ranseed=rseed)
    # create training and testing data
    for datlb in ['train', 'test']:
        if datlb == 'train':
            Xs = x_train
        else:
            Xs  = x_test
        lst = np.array_split(Xs, nproc)
        print(len(lst), lst[0].shape)
        jobs, pipels = [], []
        for proc_ind in range(nproc):
            Xsmall = lst[proc_id]
            if Xsmall.size == 0:
                continue
            tempfile = os.path.join(bindir, '{}_{}'.format(datlb))
            p = multiprocessing.Process(target=get_reservoir_states, \
                args=(tempfile, Xsmall, qparams, model, init_rs, ranseed))
            jobs.append(p)
        
        # Start the process
        for p in jobs:
            p.start()

        # Ensure all processes have finiished execution
        for p in jobs:
            p.join()

        # Sleep 5s
        time.sleep(5)

    # perform regression
    if linear_reg > 0:
        X_train = np.array(x_train)
    else:
        # Training
        with open(train_file, 'rb') as rrs:
            X_train = pickle.load(rrs)
            # if use_corr == 0:
            #     ids = np.array(range(X_train.shape[1]))
            #     ids = ids[ids % 15 < 5]
            #     X_train = X_train[:, ids]
            if full_mnist == 0:
                X_train = X_train[train_ids, :]

    X_train = np.hstack( [X_train, np.ones([X_train.shape[0], 1]) ] )
    logger.info('V={}, tauB={}, flip={}, X_train shape={}'.format(V, tau_delta, flip, X_train.shape))

    XTX = X_train.T @ X_train
    XTY = X_train.T @ Y_train
    I = np.identity(np.shape(XTX)[1])	
    pinv_ = scipypinv2(XTX + beta * I)
    W_out = pinv_ @ XTY
    logger.info('Wout shape={}'.format(W_out.shape))
    train_acc = get_acc(X_train @ W_out, y_train_lb)
    logger.info('Train acc={}'.format(train_acc))

    # Testing
    if linear_reg > 0:
        X_test = np.array(x_test)
    else:
        with open(test_file, 'rb') as rrs:
            X_test = pickle.load(rrs)
            # if use_corr == 0:
            #     ids = np.array(range(X_test.shape[1]))
            #     ids = ids[ids % 15 < 5]
            #     X_test = X_test[:, ids]
            if full_mnist == 0:
                X_test = X_test[test_ids, :]

    X_test = np.hstack( [X_test, np.ones([X_test.shape[0], 1]) ] )
    logger.info('V={}, tau={}, X_test shape = {}'.format(V, tau_delta, X_test.shape))
    test_acc = get_acc(X_test @ W_out, y_test_lb)
    logger.info('Test acc={}'.format(test_acc))
