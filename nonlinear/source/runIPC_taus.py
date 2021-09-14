#!/usr/bin/env python
"""
    Calculate information processing capacity (IPC) for higher-order quantum reservoir
    See run_hqrc_ipc.sh for an example to run the script
"""

import sys
import numpy as np
import os
import scipy
import argparse
import multiprocessing
import time
import datetime
import hqrc as hqrc
from loginit import get_module_logger

import utils
from utils import *
from IPC import IPCParams

def IPC_compute(qparams, ipcparams, length, ntrials, ranseed, log_filename, savedir, posfix, nqrc, tauls, alpha, mask_input, combine_input, qr_input):
    logger = get_module_logger(__name__, log_filename)
    logger.info(log_filename)
    
    for tau in tauls:
        qparams.tau = tau
        hqrc.get_IPC(qparams, ipcparams, length, nqrc=nqrc, gamma=alpha, logger=logger, ranseed=ranseed, Ntrials=ntrials, \
            nonlinear=0, dim_input=qr_input, mask_input=mask_input, combine_input=combine_input,\
            savedir=savedir, posfix='capa_tau_{:.6f}_{}'.format(qparams.tau, posfix), \
            type_input=1, label='tau_{:.6f}'.format(tau))
    
if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--spins', type=int, default=5, help='Number of the hidden spins')
    parser.add_argument('--rho', type=int, default=0, help='Flag for initializing the density matrix')
    parser.add_argument('--max_energy', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=1e-14)
    parser.add_argument('--nqrc', type=int, default=5)

    parser.add_argument('--length', type=int, default=3000)
    parser.add_argument('--max_delay', type=int, default=100)
    parser.add_argument('--max_deg', type=int, default=2)
    parser.add_argument('--max_num_var', type=int, default=1)
    parser.add_argument('--max_window', type=int, default=50)
    parser.add_argument('--thres', type=float, default=0.0)
    parser.add_argument('--chunk', type=int, default=1000)

    parser.add_argument('--nproc', type=int, default=101)
    parser.add_argument('--ntrials', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--deg_delays', type=str, default='0,100,0,10,0,5', help='delays by degree')
    parser.add_argument('--writedelay', type=bool, default=False)
    parser.add_argument('--virtuals', type=str, default='1')
    parser.add_argument('--taus', type=str, default='8.0')
    parser.add_argument('--alpha', type=float, default='0.0')

    parser.add_argument('--mask_input', type=int, default=0)
    parser.add_argument('--combine_input', type=int, default=1)
    parser.add_argument('--qr_input', type=int, default=1, help='Number of QRs using for input')

    parser.add_argument('--dynamic', type=str, default='full_random', help='full_random,half_random,full_const_trans,full_const_coeff,ion_trap,phase_trans')
    parser.add_argument('--basename', type=str, default='hqrc_IPC')
    parser.add_argument('--savedir', type=str, default='IPC_hqrc')
    parser.add_argument('--solver', type=str, default=LINEAR_PINV, \
        help='regression solver by linear_pinv,ridge_pinv,auto,svd,cholesky,lsqr,sparse_cg,sag')

    args = parser.parse_args()
    print(args)

    nqrc, n_spins, max_energy, beta, init_rho = args.nqrc, args.spins, args.max_energy, args.beta, args.rho
    max_delay, max_deg, max_num_var, thres, chunk = args.max_delay, args.max_deg, args.max_num_var, args.thres, args.chunk
    max_window, writedelay = args.max_window, args.writedelay
    deg_delays = [int(x) for x in args.deg_delays.split(',')]
    mask_input, combine_input, qr_input = args.mask_input, args.combine_input, args.qr_input

    solver, dynamic = args.solver, args.dynamic
    length, ranseed = args.length, args.seed
    
    ntrials, basename, savedir = args.ntrials, args.basename, args.savedir
    if os.path.isfile(savedir) == False and os.path.isdir(savedir) == False:
        os.mkdir(savedir)

    if os.path.isdir(savedir) == True:
        logdir = os.path.join(savedir, 'log')
    else:
        logdir = os.path.join(os.path.dirname(__file__), 'log')
    if os.path.isdir(logdir) == False:
        os.mkdir(logdir)
    
    basename = '{}_{}_{}_nqrc_{}_nspins_{}_seed_{}_mdeg_{}_mvar_{}_thres_{}_delays_{}_T_{}_mask_{}_cb_{}_qrin_{}'.format(\
        basename, dynamic, solver, nqrc, n_spins, ranseed, max_deg, max_num_var, thres, args.deg_delays, length,\
        mask_input, combine_input, qr_input)
    log_filename = os.path.join(logdir, '{}.log'.format(basename))
    logger = get_module_logger(__name__, log_filename)
    logger.info(log_filename)

    # Make IPCparams
    ipcparams = IPCParams(max_delay=max_delay, max_deg=max_deg, max_num_var=max_num_var, \
        max_window=max_window, thres=thres, deg_delays=deg_delays)
    
    
    virtuals = [int(x) for x in args.virtuals.split(',')]
    alpha = args.alpha
    #taus = [float(x) for x in args.taus.split(',')]
    taus = list(np.arange(-7, 7.1, 0.05))
    taus = [float(2**x) for x in taus]
    nproc = min(len(taus), args.nproc)
    lst = np.array_split(taus, nproc)

    if os.path.isfile(savedir) == False:
        for V in virtuals:
            posfix = 'alpha_{:.6f}_V_{}_{}'.format(alpha, V, basename)
            jobs, pipels = [], []
            for pid in range(nproc):
                tBs = lst[pid]
                log_filename = os.path.join(logdir, 'taus_{:.6f}_{}.log'.format(tBs[0], posfix))
                
                # check file
                # degfile = os.path.join(savedir, 'degree_{}.txt'.format(posfix))
                # if os.path.isfile(degfile) == True:
                #     continue
                qparams = QRCParams(n_units=n_spins-1, n_envs=1, max_energy=max_energy, \
                            beta=beta, virtual_nodes=V, tau=1.0, init_rho=init_rho, solver=solver, dynamic=dynamic)
                p = multiprocessing.Process(target=IPC_compute, \
                    args=(qparams, ipcparams, length, ntrials, ranseed, log_filename, savedir, posfix, nqrc, tBs, alpha,\
                        mask_input, combine_input, qr_input))
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
