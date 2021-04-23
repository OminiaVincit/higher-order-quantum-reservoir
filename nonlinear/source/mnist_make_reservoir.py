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

import gzip
import _pickle as cPickle
import pickle

import time
import datetime
import hqrc as hqrc
from loginit import get_module_logger

import utils
from utils import *

MNIST_DIR = "/data/zoro/qrep/mnist"
TAU_MNIST_DIR = "/data/zoro/qrep/mnist/taudata"
MNIST_SIZE="10x10"

def get_reservoir_states(Xs, n_qrs, qparams, model, init_rs, ranseed, savefile, flip, buffer=100):
    # Create sequence for the initial washout stage
    np.random.seed(seed=0)
    washout_signals = np.random.uniform(0, 1, buffer)
    washout_signals = np.tile(washout_signals, (n_qrs, 1))
    
    rstates = []
    for s in Xs:
        # Initial washout stage
        _ = model.init_forward(qparams, washout_signals, init_rs=init_rs, ranseed = ranseed)
        if flip == True:
            input_signals = np.array(np.flip(s))
        else:
            input_signals = np.array(s)
        input_signals = np.tile(input_signals, (n_qrs, 1))
        _, output_signals = model.feed_forward(input_signals, predict=False, use_lastrho=True)
        rstates.append(output_signals.ravel())
    
    rstates = np.array(rstates)
    with open(savefile, 'wb') as wrs:
            pickle.dump(rstates, wrs)


if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--nqrs', type=int, default=1, help='Number of QRs')
    parser.add_argument('--spins', type=int, default=5, help='Number of the hidden spins')
    parser.add_argument('--rho', type=int, default=0, help='Flag for initializing the density matrix')
    parser.add_argument('--max_energy', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1e-14)

    parser.add_argument('--nproc', type=int, default=50)
    parser.add_argument('--virtual', type=int, default=1)
    parser.add_argument('--taus', type=str, default='', help='Input interval')

    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha of coupled strength, 0 for random coupling')
    parser.add_argument('--bcoef', type=float, default=1.0, help='bcoeff nonlinear term (non-diagonal term)')

    parser.add_argument('--dynamic', type=str, default='ion_trap', help='full_random,half_random,full_const_trans,full_const_coeff,ion_trap')
    parser.add_argument('--basename', type=str, default='qrc')
    parser.add_argument('--savedir', type=str, default=TAU_MNIST_DIR)
    parser.add_argument('--mnist_dir', type=str, default=MNIST_DIR)
    parser.add_argument('--mnist_size', type=str, default=MNIST_SIZE)

    parser.add_argument('--tmax', type=float, default=25, help='Maximum of tauB')
    parser.add_argument('--tmin', type=float, default=0, help='Minimum of tauB')
    parser.add_argument('--ntaus', type=int, default=125, help='Number of tausB')

    parser.add_argument('--buffer', type=int, default=100, help='Buffer length')

    args = parser.parse_args()
    print(args)

    n_spins, n_qrs, max_energy, beta, alpha, bcoef, init_rho = args.spins, args.nqrs, args.max_energy, args.beta, args.alpha, args.bcoef, args.rho
    tmin, tmax, ntaus = args.tmin, args.tmax, args.ntaus
    
    dynamic, buffer = args.dynamic, args.buffer
    V = args.virtual

    basename, savedir, mnist_dir, mnist_size = args.basename, args.savedir, args.mnist_dir, args.mnist_size
    if os.path.isfile(savedir) == False and os.path.isdir(savedir) == False:
        os.mkdir(savedir)

    if os.path.isdir(savedir) == True:
        logdir = os.path.join(savedir, 'log')
    else:
        logdir = os.path.join(os.path.dirname(__file__), 'log')

    if os.path.isdir(logdir) == False:
        os.mkdir(logdir)

    B = max_energy / bcoef
    taudeltas = args.taus
    if taudeltas == '':
            taudeltas = list(np.linspace(tmin, tmax, ntaus + 1))
            taudeltas = taudeltas[1:]
    else:
        taudeltas = [float(x) for x in args.taus.split(',')]
    
    Xs_train, Ys_train, Xs_test, Ys_test, Xs_val, Ys_val = gen_mnist_dataset(mnist_dir, mnist_size)
    
    for tau_delta in taudeltas:
        basename = '{}_{}_{}_nqrs_{}_nspins_{}_a_{}_bc_{}_tauB_{:.3f}_V_{}_buf_{}'.format(mnist_size, \
            args.basename, dynamic, n_qrs, n_spins, alpha, bcoef, tau_delta, V, buffer)
        log_filename = os.path.join(logdir, '{}.log'.format(basename))
        logger = get_module_logger(__name__, log_filename)
        logger.info(log_filename)
        logger.info('tau_delta={}, mnist shape train = {}, test = {}, val={}'.format(tau_delta, Xs_train.shape, Xs_test.shape, Xs_val.shape))

        if os.path.isfile(savedir) == False:
            qparams = QRCParams(n_units=n_spins-1, n_envs=1, max_energy=max_energy, non_diag=bcoef, alpha=alpha,\
                beta=beta, virtual_nodes=V, tau=tau_delta/B, init_rho=init_rho, dynamic=dynamic)
            model = hqrc.HQRC(nqrc=n_qrs, gamma=0.0, sparsity=1.0, sigma_input=1.0, type_input=0.0, use_corr=0)
            model.init_reservoir(qparams, ranseed=0)

            for flip in [False, True]:
                for lb in ['train', 'test', 'val']:
                    if lb == 'train':
                        X = Xs_train
                    elif lb == 'test':
                        X = Xs_test
                    else:
                        X = Xs_val
                    
                    # get reservoir states
                    tx = list(range(X.shape[0]))
                    #tx = list(range(10))
                    
                    nproc = min(len(tx), args.nproc)
                    lst = np.array_split(tx, nproc)

                if True:
                    jobs, pipels = [], []
                    tmp_files = []
                    for pid in range(nproc):
                        xs = lst[pid]
                        init_rs = False
                        ranseed = 0
                        savefile = os.path.join(savedir, 'temp_{}_{}_{}.binaryfile'.format(lb, basename, pid))
                        tmp_files.append(savefile)

                        p = multiprocessing.Process(target=get_reservoir_states, args=(X[xs], n_qrs, qparams, model, init_rs, ranseed, savefile, flip, buffer))
                        jobs.append(p)

                    # Start the process
                    for p in jobs:
                        p.start()

                    # Ensure all processes have finished execution
                    for p in jobs:
                        p.join()

                    # Sleep 5s
                    time.sleep(5)

                    # Joint dumped temp data file
                    zarr = []
                    for filename in tmp_files:
                        with open(filename, 'rb') as rrs:
                            tmp = pickle.load(rrs)
                            for arr in tmp:
                                zarr.append(arr)
                        
                        # Delete file
                        os.remove(filename)
                    zarr = np.array(zarr)
                    logger.info('Flip_{}_{}_{}'.format(flip, lb, zarr.shape))

                    filename = os.path.join(savedir, '{}_{}_flip_{}.binaryfile'.format(lb, basename, flip))
                    with open(filename, 'wb') as wrs:
                        pickle.dump(zarr, wrs)
