#!/usr/bin/env python
"""
Implementation of learning to learn on HQRC

"""

import sys
from macpath import basename
import numpy as np
import os
import argparse
import multiprocessing, math
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import ticker
import time
import hqrc as hqrc
import utils as utils
import gendata as gdata
import pickle
from loginit import get_module_logger

def fitness(qparams, W_feed, xs, Ys, transient, train_len, val_len, type_op, type_input):
    train_input_seq_org = np.array(xs[: transient + train_len])
    train_input_seq_org = train_input_seq_org.reshape(1, train_input_seq_org.shape[0])
    train_output_seq = Ys[  : transient + train_len] 

    val_input_seq_org =  np.array(xs[transient + train_len : transient + train_len + val_len])
    val_input_seq_org = val_input_seq_org.reshape(1, val_input_seq_org.shape[0])
    val_output_seq = Ys[transient + train_len : transient + train_len + val_len]

    train_input_seq = np.tile(train_input_seq_org, (nqrc, 1))
    val_input_seq = np.tile(val_input_seq_org, (nqrc, 1))

    _, train_loss, _, val_loss = hqrc.get_loss(qparams, transient, train_input_seq, train_output_seq, \
            val_input_seq, val_output_seq, nqrc=nqrc, gamma=1.0, ranseed=0, type_input=type_input, type_op=type_op, W_feed=W_feed)
    return train_loss

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--taskname', type=str, default='narma')

    parser.add_argument('--units', type=int, default=5, help='Number of the hidden units')
    parser.add_argument('--max_energy', type=float, default=1.0, help='Maximum coupling energy')
    parser.add_argument('--rho', type=int, default=0, help='Flag for initializing the density matrix')
    parser.add_argument('--reg', type=float, default=1e-14, help='reg term')
    parser.add_argument('--solver', type=str, default=utils.LINEAR_PINV, \
        help='regression solver by linear_pinv,ridge_pinv,auto,svd,cholesky,lsqr,sparse_cg,sag')
    parser.add_argument('--dynamic', type=str, default=utils.DYNAMIC_PHASE_TRANS,\
        help='full_random,half_random,full_const_trans,full_const_coeff,ion_trap')


    parser.add_argument('--nqrc', type=int, default=3, help='Number of reservoirs')
    parser.add_argument('--virtuals', type=int, default=1)
    parser.add_argument('--tau', type=float, default=10.0, help='Tau')
    parser.add_argument('--gamma', type=float, default=0.0, help='Feedback strength')
    parser.add_argument('--non_diag_var', type=float, default=1.0, help='non_diag_var for phase_trans dynamic')
    parser.add_argument('--non_diag_const', type=float, default=2.0, help='non_diag_const for phase_trans dynamic')


    parser.add_argument('--savedir', type=str, default='../results/leanr2learn')
    parser.add_argument('--rseed', type=int, default=0)
    parser.add_argument('--type_input', type=int, default=5)
    parser.add_argument('--type_op', type=str, default='X')
    
    # For data
    parser.add_argument('--train_len', type=int, default=1000)
    parser.add_argument('--val_len', type=int, default=1000)
    parser.add_argument('--transient', type=int, default=1000, help='Transitient time steps')

    # For SPSA optimizers
    parser.add_argument('--n_iters', type=int, default=100)
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--decay', type=float, default=0.0)
    parser.add_argument('--gdecay', type=float, default=0.0)

    args = parser.parse_args()
    taskname = args.taskname
    n_units, nqrc, dynamic, max_energy, V = args.units, args.nqrc, args.dynamic, args.max_energy, args.virtuals
    solver, beta, init_rho = args.solver, args.reg, args.rho

    gamma, type_op, type_input, tau = args.gamma, args.type_op, args.type_input, args.tau
    non_diag_const, non_diag_var = args.non_diag_const, args.non_diag_var
    randseed, savedir  = args.rseed, args.savedir
    train_len, val_len, transient = args.train_len, args.val_len, args.transient
    
    # For optimizer
    n_iters, eps, lr, decay, gdecay = args.n_iters, args.eps, args.lr, args.decay, args.gdecay
    # Make folder
    os.makedirs(savedir, exist_ok=True)
    save_fig = os.path.join(savedir, 'figs')
    os.makedirs(save_fig, exist_ok=True)
    logdir = os.path.join(savedir, 'log')
    os.makedirs(logdir, exist_ok=True)
    datdir = os.path.join(savedir, 'data')
    os.makedirs(datdir, exist_ok=True)
    resdir = os.path.join(savedir, 'res')
    os.makedirs(resdir, exist_ok=True)

    basename = '{}_V_{}_tau_{}_nqr_{}_op_{}_tp_{}_len_{}_{}_{}_es_{}_lr_{}_ac_{}_{}_seed_{}_it_{}'.format(\
        taskname, V, tau, nqrc, type_op, type_input, train_len, val_len, transient, \
        eps, lr, decay, gdecay, args.rseed, n_iters)
    
    log_filename = os.path.join(logdir, '{}.log'.format(basename))
    logger = get_module_logger(__name__, log_filename)
    logger.info(log_filename)
    logger.info(args)

    # Generate data (NARMA)
    np.random.seed(seed=args.rseed)
        
    xs, Ys = utils.make_data_for_narma(train_len + val_len + transient, orders = [2,5])
    outer_xs, outer_Ys = utils.make_data_for_narma(train_len + val_len + transient, orders = [10])

    # Outer loop
    qparams = utils.QRCParams(n_units=n_units-1, n_envs=1, max_energy=max_energy, non_diag_const=non_diag_const, non_diag_var=non_diag_var,\
                            beta=beta, virtual_nodes=V, tau=tau, init_rho=init_rho, solver=solver, dynamic=dynamic)
    
    W_feed = np.random.uniform(0.0, 1.0, size=(n_units * V * nqrc, nqrc))
    pre_outer_cost = fitness(qparams, W_feed, outer_xs, outer_Ys, transient, train_len, val_len, type_op=type_op, type_input=type_input)
    logger.info('Outer cost without pretraining: {}'.format(pre_outer_cost))

    n_coeffs = W_feed.shape[0]  * W_feed.shape[1]
    logger.info('Feedback dim {}'.format(W_feed.shape))
    # Using SPSA algorithm
    
    opt = utils.Optimizer(label='SPSA', schedule='decay', lr=lr, decay=decay, gdecay=gdecay)
    for iter in range(n_iters + 1):
        opt.schedule_lr(iter)
        delta = np.zeros(n_coeffs)
        perts = utils.bernoulli_pert(dim=n_coeffs)
        current_weights = W_feed.copy()

        pertubed_cost = []
        diff_delta = []
        
        for drift in [eps, -eps]:
            diff_pert = perts * drift / math.pow((iter + opt.A), opt.gdecay)
            W_feed = current_weights + diff_pert.reshape(W_feed.shape)
            outer_cost = fitness(qparams, W_feed, xs, Ys, transient, train_len, val_len, type_op=type_op, type_input=type_input)
            pertubed_cost.append(outer_cost)
            diff_delta.append(diff_pert)
        
        # Update gradient
        diff = diff_delta[1] - diff_delta[0]
        diff_cost = pertubed_cost[1] - pertubed_cost[0]
        for k in range(n_coeffs):
            delta[k] = diff_cost / diff[k]
        
        # Revert the weight
        W_feed = current_weights

        # Update the weight
        update_term = opt.update_term(delta)
        W_feed = W_feed - update_term.reshape(W_feed.shape)

        avg_cost = (pertubed_cost[1] + pertubed_cost[0]) / 2.0
        if iter % 10 == 0:
            logger.info('Iter={}, lr={:5f}, average cost={:5f}, max gradient={:.5f}'.format(iter, opt.lr, avg_cost, np.max(np.abs(delta)) ))
    
    outer_cost = fitness(qparams, W_feed, outer_xs, outer_Ys, transient, train_len, val_len, type_op=type_op, type_input=type_input)
    logger.info('Outer cost with pretraining: {}'.format(outer_cost))

    



    