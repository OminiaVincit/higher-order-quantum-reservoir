#!/usr/bin/env python
"""
Perform sequential MNIST
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

import gzip
import _pickle as cPickle

import time
import datetime
import hqrc as hqrc
from loginit import get_module_logger

import utils
from utils import *

MNIST_DIR = "/data/zoro/qrep/mnist"
SAVE_DIR  = "/data/zoro/qrep/mnist/results"

LABEL1 = 3
LABEL2 = 5

def gen_mnist_dataset(size=10):
    f = gzip.open(os.path.join(MNIST_DIR, 'mnist_{0}x{0}.pkl.gz'.format(size)),'rb')
    data = cPickle.load(f, encoding='latin1')
    f.close()
    print(type(data), len(data))
    train_set, valid_set, test_set = data

    xs_train, ys_train = train_set
    xs_test, ys_test = test_set
    xs_val, ys_val = valid_set

    print('Train', xs_train.shape, ys_train.shape)
    print('Test', xs_test.shape, ys_test.shape)

    # Focus on classifying digit 2 and digit 3
    X_train_a = xs_train[ys_train == LABEL1]
    X_train_b = xs_train[ys_train == LABEL2]
    X_train = np.vstack([X_train_a, X_train_b])
    X_train = X_train / 255.0

    Y_train = np.zeros(len(X_train), dtype=bool)
    Y_train[len(X_train_a):] = 1

    X_test_a = xs_test[ys_test == LABEL1]
    X_test_b = xs_test[ys_test == LABEL2]
    X_test = np.vstack([X_test_a, X_test_b])
    X_test = X_test / 255.0

    Y_test = np.zeros(len(X_test), dtype=bool)
    Y_test[len(X_test_a):] = 1

    return X_train, Y_train, X_test, Y_test

def get_reservoir_states(Xs, n_qrs, qparams, model, init_rs, ranseed):
    rstates = []
    for s in Xs:
        input_signals = np.array(s)
        input_signals = np.tile(input_signals, (n_qrs, 1))
        output_signals = model.init_forward(qparams, input_signals, init_rs=init_rs, ranseed = ranseed)
        rstates.append(output_signals.ravel())
    rstates = np.array(rstates)
    return rstates



def mnist_compute(qparams, n_qrs, Xs_train, Ys_train, Xs_test, Ys_test, total_train_len, total_test_len, log_filename, ranseed, use_corr):
    logger = get_module_logger(__name__, log_filename)
    logger.info(log_filename)

    model = hqrc.HQRC(nqrc=n_qrs, alpha=0.0, sparsity=1.0, sigma_input=1.0, type_input=0.0, use_corr=use_corr)
    model.init_reservoir(qparams, ranseed)
    np.random.seed(seed=ranseed)

    if total_train_len < 0:
        total_train_len = Xs_train.shape[0]

    if total_test_len < 0:
        total_test_len = Xs_test.shape[0]
    logger.info('Total length train={}, test={}'.format(total_train_len, total_test_len))

    en_states = []

    train_acc_ls, test_acc_ls = [], []
    for ieval in range(Neval):
        # Shuffle data
        shuffle_train = np.random.permutation(Xs_train.shape[0])
        xs_train = Xs_train[shuffle_train, :][:total_train_len, :]
        ys_train = Ys_train[shuffle_train][:total_train_len]
        N_train = xs_train.shape[0]
        Yt_train = np.vstack([np.ones(N_train), -1*np.ones(N_train)]).T
        Yt_train[ys_train > 0] *= -1
        
        shuffle_test = np.random.permutation(Xs_test.shape[0])
        xs_test = Xs_test[shuffle_test, :][:total_test_len, :]
        ys_test = Ys_test[shuffle_test][:total_test_len]
        N_test = xs_test.shape[0]
        Yt_test = np.vstack([np.ones(N_test), -1*np.ones(N_test)]).T
        Yt_test[ys_test > 0] *= -1

        # print(ieval, 'Train', xs_train.shape, ys_train.shape, Yt_train.shape)
        # print(ieval, 'Test', xs_test.shape, ys_test.shape, Yt_test.shape)

        # Training
        Xt = get_reservoir_states(xs_train, n_qrs, qparams, model, init_rs=False, ranseed=ranseed)
        #print(Xt.shape)
        XTX = Xt.T @ Xt
        XTY = Xt.T @ Yt_train
        I = np.identity(np.shape(XTX)[1])	
        pinv_ = scipypinv2(XTX + beta * I)
        W_out = pinv_ @ XTY
        train_acc = get_acc(Xt @ W_out, ys_train)
        train_acc_ls.append(train_acc)
        
        # Testing
        Xt  = get_reservoir_states(xs_test, n_qrs, qparams, model, init_rs=False, ranseed=ranseed)
        #print(W_out.shape)
        test_acc = get_acc(Xt @ W_out, ys_test)
        test_acc_ls.append(test_acc)

        avg_train_acc, avg_test_acc = np.mean(train_acc_ls), np.mean(test_acc_ls)
        logger.info('i_eval={}, Accuracy train={}, test={}, Avg train={}, test={}'.format(ieval, train_acc, test_acc, avg_train_acc, avg_test_acc))            
    
    logger.info('N_eval={}, Average accuracy train={}, test={}'.format(Neval, avg_train_acc, avg_test_acc))


if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--nqrs', type=int, default=1, help='Number of QRs')
    parser.add_argument('--spins', type=int, default=6, help='Number of spins')
    parser.add_argument('--rho', type=int, default=0, help='Flag for initializing the density matrix')
    parser.add_argument('--max_energy', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1e-14)

    parser.add_argument('--nproc', type=int, default=50)
    parser.add_argument('--ntrials', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--virtuals', type=str, default='1')
    parser.add_argument('--tauB', type=float, default=22.0, help='Input interval')
    parser.add_argument('--usecorr', type=int, default=0)

    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha of coupled strength, 0 for random coupling')
    parser.add_argument('--bcoef', type=float, default=0.42, help='bcoeff nonlinear term (non-diagonal term)')

    parser.add_argument('--dynamic', type=str, default='ion_trap', help='full_random,half_random,full_const_trans,full_const_coeff,ion_trap')
    parser.add_argument('--basename', type=str, default='qrc')
    parser.add_argument('--savedir', type=str, default='res_mnist')
    parser.add_argument('--neval', type=int, default=1)
    parser.add_argument('--trainlen', type=int, default=50)
    parser.add_argument('--testlen', type=int, default=50)

    args = parser.parse_args()
    print(args)

    n_qrs, n_units, max_energy, beta, g, init_rho = args.nqrs, args.units, args.coupling, args.beta, args.nondiag, args.rho
    dynamic = args.dynamic
    ranseed = args.seed
    tau_delta = args.tau
    use_corr = args.usecorr

    Neval = args.neval
    total_train_len = args.trainlen
    total_test_len = args.testlen

    ntrials, basename, savedir = args.ntrials, args.basename, args.savedir
    if os.path.isfile(savedir) == False and os.path.isdir(savedir) == False:
        os.mkdir(savedir)

    if os.path.isdir(savedir) == True:
        logdir = os.path.join(savedir, 'log')
        figdir = os.path.join(savedir, 'figs')
    else:
        logdir = os.path.join(os.path.dirname(__file__), 'log')
        figdir = os.path.join(os.path.dirname(__file__), 'figs')

    if os.path.isdir(logdir) == False:
        os.mkdir(logdir)
    
    if os.path.isdir(figdir) == False:
        os.mkdir(figdir)

    basename = '{}_{}_{}_train_{}_test_{}_cor_{}_seed_{}'.format(basename, dynamic, args.virtuals, total_train_len, total_test_len, use_corr, ranseed)
    log_filename = os.path.join(logdir, '{}.log'.format(basename))
    logger = get_module_logger(__name__, log_filename)
    logger.info(log_filename)

    virtuals = [int(x) for x in args.virtuals.split(',')]
    taudeltas = np.linspace(1.0, 50.0, 99)
    
    if os.path.isfile(savedir) == False:
        Xs_train, Ys_train, Xs_test, Ys_test = gen_mnist_dataset()
        logger.info('mnist shape train = {}, test = {}'.format(Xs_train.shape, Xs_test.shape))

        #fig = plt.figure(figsize=(20, 10), dpi=600)

        jobs, pipels = [], []
        for V in virtuals:
            for tau_delta in taudeltas:
                log_filename = os.path.join(logdir, 'tau_{:.3f}_V_{}_{}.log'.format(tau_delta, V, basename))
                outbase = os.path.join(figdir, '{}_tau_{:.3f}_V_{}'.format(basename, tau_delta, V))

                qparams = QRCParams(n_units=n_units, max_energy=max_energy, non_diag=g,\
                            beta=beta, virtual_nodes=V, tau=tau_delta, init_rho=init_rho, dynamic=dynamic)
                p = multiprocessing.Process(target=mnist_compute, \
                    args=(qparams, n_qrs, Xs_train, Ys_train, Xs_test, Ys_test, total_train_len, total_test_len, log_filename, ranseed, use_corr))
                jobs.append(p)

    
        # Start the process
        for p in jobs:
            p.start()

        # Ensure all processes have finished execution
        for p in jobs:
            p.join()

        # Sleep 5s
        time.sleep(5)
