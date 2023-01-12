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

def dump_reservoir_states(tempfile, Xs, qparams, model, init_rs, ranseed):
    rstates = []
    for s in Xs:
        input_signals = np.array(s)
        output_signals = model.init_forward(qparams, input_signals, init_rs=init_rs, ranseed = ranseed)
        rstates.append(output_signals.ravel())
    rstates = np.array(rstates)
    with open(tempfile, 'wb') as wrs:
        pickle.dump(rstates, wrs)
    return tempfile

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
    parser.add_argument('--strengths', type=float, default=0.0, help='Connection strengths')
    parser.add_argument('--tau', type=float, default=4.0, help='Input interval')

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
    J, init_rho, tau, V, alpha = args.coupling, args.rho, args.tau, args.virtuals, args.strengths
    
    solver, linear_reg, use_corr, transient = args.solver, args.linear_reg, args.use_corr, args.transient
    full_mnist, label1, label2 = args.full, args.label1, args.label2
    ntrials, dynamic, savedir = args.ntrials, args.dynamic, args.savedir
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

    if os.path.isdir(bindir) == False:
        os.mkdir(bindir)

    basename = '{}_{}_linear_{}_nqrs_{}_corr_{}_nspins_{}_V_{}_tau_{}_alpha_{}'.format(\
        mnist_size, dynamic, linear_reg, n_qrs, use_corr, n_spins, V, tau, alpha)
    log_filename = os.path.join(logdir, '{}.log'.format(basename))
    logger = get_module_logger(__name__, log_filename)
    logger.info(log_filename)

    x_train, y_train_lb, x_test, y_test_lb, x_val, y_val_lb = gen_mnist_dataset(mnist_dir, mnist_size)
    
    if full_mnist > 0:
        Y_train = np.identity(10)[y_train_lb]
        Y_test  = np.identity(10)[y_test_lb]
        Y_val   = np.identity(10)[y_val_lb]
    else:
        # Train data
        train_ids = (y_train_lb == label1) | (y_train_lb == label2)
        # x_train = x_train[train_ids, :]
        y_train_lb = y_train_lb[train_ids]
        y_train_lb[y_train_lb == label1] = 0
        y_train_lb[y_train_lb == label2] = 1

        Y_train = np.identity(2)[y_train_lb]

        # Test data 
        test_ids = (y_test_lb == label1) | (y_test_lb == label2)
        # x_test = x_test[test_ids, :]
        y_test_lb = y_test_lb[test_ids]
        y_test_lb[y_test_lb == label1] = 0
        y_test_lb[y_test_lb == label2] = 1

        Y_test  = np.identity(2)[y_test_lb]

        # Val data
        val_ids = (y_val_lb == label1) | (y_val_lb == label2)
        # x_val = x_val[val_ids, :]
        y_val_lb = y_val_lb[val_ids]
        y_val_lb[y_val_lb == label1] = 0
        y_val_lb[y_val_lb == label2] = 1

        Y_val  = np.identity(2)[y_val_lb]

    
    logger.info('shape x_train={}, y_train={},  x_test={}, y_test={}, x_val={}, y_val={}'.format(\
        x_train.shape, Y_train.shape, x_test.shape, Y_test.shape, x_val.shape, Y_val.shape))

    # Create params and model
    qparams = QRCParams(n_units=n_spins-1, n_envs=1, max_energy=J,\
        beta=beta, virtual_nodes=V, tau=tau, init_rho=init_rho, solver=solver, dynamic=dynamic)
    model = hqrc.HQRC(nqrc=n_qrs, gamma=alpha, sparsity=1.0, sigma_input=1.0, use_corr=use_corr)
    model.init_reservoir(qparams, ranseed=rseed)
    init_rs, rs_seed = False, 0
    # create training and testing data
    if linear_reg <= 0:
        datfs = dict()
        for datlb in ['train', 'test', 'val']:
            if datlb == 'train':
                Xt = x_train
            else:
                Xt  = x_test
            # Make Xs to put in hqrc
            Xs = [x.reshape(n_qrs, -1) for x in Xt]
            # Create training/test file
            (ns, ds) = Xt.shape
            datfile = '{}_{}_{}_{}.bin'.format(datlb, basename, ns, ds)
            datfile = os.path.join(bindir, datfile)
            datfs[datlb] = datfile
            if os.path.isfile(datfile) == True:
                logger.debug('File existed {}'.format(datfile))
                continue

            lst = np.array_split(Xs, nproc)
            print(len(lst), lst[0].shape)
            jobs, pipels = [], []
            for proc_id in range(nproc):
                Xsmall = lst[proc_id]
                if Xsmall.size == 0:
                    continue
                tempfile = os.path.join(bindir, '{}_{}.bin'.format(datlb, proc_id))
                p = multiprocessing.Process(target=dump_reservoir_states, \
                    args=(tempfile, Xsmall, qparams, model, init_rs, rs_seed))
                jobs.append(p)
                pipels.append(tempfile)
            
            # Start the process
            for p in jobs:
                p.start()

            # Ensure all processes have finished execution
            for p in jobs:
                p.join()

            # Sleep 5s
            time.sleep(5)

            # join the training file
            X_states = []
            for filename in pipels:
                if os.path.isfile(filename) == True:
                    with open(filename, 'rb') as rrs:
                        X_states.append(pickle.load(rrs))
                    # Delete file
                    os.remove(filename)
            X_states = np.concatenate(X_states)
            logger.info('Concatenate {} file with shape {}'.format(datlb, X_states.shape))
            # dump to datfile
            with open(datfile, 'wb') as wrs:
                pickle.dump(X_states, wrs)
            

    # perform regression
    if linear_reg > 0:
        X_train = np.array(x_train)
        if full_mnist == 0:
            X_train = X_train[train_ids, :]

    else:
        # Training
        with open(datfs['train'], 'rb') as rrs:
            X_train = pickle.load(rrs)
            # if use_corr == 0:
            #     ids = np.array(range(X_train.shape[1]))
            #     ids = ids[ids % 15 < 5]
            #     X_train = X_train[:, ids]
            if full_mnist == 0:
                X_train = X_train[train_ids, :]

    X_train = np.hstack( [X_train, np.ones([X_train.shape[0], 1]) ] )
    logger.info('V={}, tauB={}, X_train shape={}'.format(V, tau, X_train.shape))

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
        if full_mnist == 0:
            X_test = X_test[test_ids, :]
    else:
        with open(datfs['test'], 'rb') as rrs:
            X_test = pickle.load(rrs)
            # if use_corr == 0:
            #     ids = np.array(range(X_test.shape[1]))
            #     ids = ids[ids % 15 < 5]
            #     X_test = X_test[:, ids]
            if full_mnist == 0:
                X_test = X_test[test_ids, :]

    X_test = np.hstack( [X_test, np.ones([X_test.shape[0], 1]) ] )
    logger.info('V={}, tau={}, alpha={}, X_test shape = {}'.format(V, tau, alpha, X_test.shape))
    test_acc = get_acc(X_test @ W_out, y_test_lb)
    logger.info('Test acc={}'.format(test_acc))
