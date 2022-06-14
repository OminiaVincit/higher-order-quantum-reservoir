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
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
import time
import datetime
import hqrc as hqrc
import logging
from loginit import get_module_logger

import utils
from utils import *
from IPC import IPCParams
import copy

def IPC_compute(qparams, ipcparams, length, ntrials, ranseed, log_filename, savedir, posfix, \
    log_Ws, log_gams, nqrc, combine_input, type_input, type_op, input_file):
    logger = get_module_logger(__name__, log_filename)
    logger.setLevel(logging.INFO)
    logger.info(log_filename)
    
    for log_gam in log_gams:
        gamma = 10**log_gam
        for log_W in log_Ws:
            qparams.non_diag_var = 10**log_W
            label = 'log_g_{:.3f}_W_{:.3f}'.format(log_gam, log_W)
            hqrc.get_IPC(qparams, ipcparams, length, nqrc=nqrc, gamma=gamma, logger=logger, ranseed=ranseed, Ntrials=ntrials, \
                savedir=savedir, posfix='capa_{}_{}'.format(label, posfix), feed_nothing=False, nonlinear=0, dim_input=1,\
                type_op=type_op, type_input=type_input, combine_input=combine_input, label=label, input_file=input_file)
    
if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--spins', type=int, default=6, help='Number of the hidden spins')
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

    parser.add_argument('--nproc', type=int, default=111)
    parser.add_argument('--ntrials', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--combine_input', type=int, default=1)
    parser.add_argument('--type_input', type=int, default=0)
    parser.add_argument('--type_op', type=str, default='Z')
    parser.add_argument('--non_diag_const', type=float, default=2.0, help='The nondiag const')

    parser.add_argument('--deg_delays', type=str, default='0,100,0,10,0,5', help='delays by degree')
    parser.add_argument('--writedelay', type=bool, default=False)
    parser.add_argument('--virtuals', type=str, default='1')
    parser.add_argument('--taus', type=str, default='10.0')

    parser.add_argument('--dynamic', type=str, default=DYNAMIC_PHASE_TRANS, help='full_random,half_random,full_const_trans,full_const_coeff,ion_trap,phase_trans')
    parser.add_argument('--basename', type=str, default='hqrc_IPC')
    parser.add_argument('--savedir', type=str, default='IPC_hqrc')
    parser.add_argument('--solver', type=str, default=LINEAR_PINV, \
        help='regression solver by linear_pinv,ridge_pinv,auto,svd,cholesky,lsqr,sparse_cg,sag')
    
    parser.add_argument('--w_interval', type=float, default=0.05, help='interval for logW')
    parser.add_argument('--w_min', type=float, default=-2.0, help='minimum of parameter range for logW')
    parser.add_argument('--w_max', type=float, default=2.1, help='maximum of parameter range for logW')
    
    parser.add_argument('--g_interval', type=float, default=0.05, help='interval for gamma')
    parser.add_argument('--g_min', type=float, default=-2.0, help='minimum of parameter range for log_gamma')
    parser.add_argument('--g_max', type=float, default=2.1, help='maximum of parameter range for log_gamma')

    parser.add_argument('--input_file', type=str, default='') # Use input from file or uniform random input

    args = parser.parse_args()
    print(args)

    nqrc, n_spins, max_energy, beta, init_rho, combine_input = args.nqrc, args.spins, args.max_energy, args.beta, args.rho, args.combine_input
    w_interval, w_min, w_max = args.w_interval, args.w_min, args.w_max
    g_interval, g_min, g_max = args.g_interval, args.g_min, args.g_max

    max_delay, max_deg, max_num_var, thres, chunk = args.max_delay, args.max_deg, args.max_num_var, args.thres, args.chunk
    max_window, writedelay = args.max_window, args.writedelay
    deg_delays = [int(x) for x in args.deg_delays.split(',')]

    solver, dynamic = args.solver, args.dynamic
    length, ranseed = args.length, args.seed
    type_input, type_op, non_diag_const = args.type_input, args.type_op, args.non_diag_const

    ntrials, basename, savedir, input_file = args.ntrials, args.basename, args.savedir, args.input_file
    if os.path.isfile(savedir) == False and os.path.isdir(savedir) == False:
        os.makedirs(savedir)

    if os.path.isdir(savedir) == True:
        logdir = os.path.join(savedir, 'log')
    else:
        logdir = os.path.join(os.path.dirname(__file__), 'log')
    if os.path.isdir(logdir) == False:
        os.mkdir(logdir)
    
    basename = '{}_{}_{}_nqrc_{}_nspins_{}_seed_{}_mdeg_{}_mvar_{}_thres_{}_delays_{}_T_{}_cb_{}_op_{}_tp_{}'.format(\
        basename, dynamic, solver, nqrc, n_spins, ranseed, max_deg, max_num_var, thres, \
        args.deg_delays, length, combine_input, type_op, type_input)
    
    log_filename = os.path.join(logdir, '{}.log'.format(basename))
    logger = get_module_logger(__name__, log_filename)
    logger.info(log_filename)

    # Make IPCparams
    ipcparams = IPCParams(max_delay=max_delay, max_deg=max_deg, max_num_var=max_num_var, \
        max_window=max_window, thres=thres, deg_delays=deg_delays)
    
    w_vals =  list(np.arange(w_min, w_max, w_interval))
    g_vals =  list(np.arange(g_min, g_max, g_interval))
    
    nproc = min(len(g_vals), args.nproc)
    lst = np.array_split(g_vals, nproc)

    virtuals = [int(x) for x in args.virtuals.split(',')]
    taus = [float(x) for x in args.taus.split(',')]

    if os.path.isfile(savedir) == False:
        for V in virtuals:
            for tau in taus:
                qparams = QRCParams(n_units=n_spins-1, n_envs=1, max_energy=max_energy, \
                        non_diag_const=non_diag_const, non_diag_var=0.0,\
                        beta=beta, virtual_nodes=V, tau=tau, init_rho=init_rho, solver=solver, dynamic=dynamic)
                jobs, pipels = [], []
                for pid in range(nproc):
                    log_gams = lst[pid]
                    posfix = 'tau_{}_V_{}_{}'.format(tau, V, basename)
                    sub_log_filename = os.path.join(logdir, 'log_gam_{:.3f}_{:.3f}_{}.log'.format(log_gams[0], log_gams[-1], posfix))
                    
                    # check file
                    # degfile = os.path.join(savedir, 'degree_{}.txt'.format(posfix))
                    # if os.path.isfile(degfile) == True:
                    #     continue
                    qparam_p = copy.copy(qparams)

                    p = multiprocessing.Process(target=IPC_compute, \
                        args=(qparam_p, ipcparams, length, ntrials, ranseed, sub_log_filename, \
                            savedir, posfix, w_vals, log_gams, nqrc, combine_input, type_input, type_op, input_file))
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
