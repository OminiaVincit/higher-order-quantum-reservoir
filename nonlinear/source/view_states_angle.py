#!/usr/bin/env python
"""
    View bifurcation diagrams and dynamics for higher-order quantum reservoir
    See run_view_states_angle.sh for an example of the running script
    Update version 2022-06-06 for new encoding method
"""
import sys
import numpy as np
import os
import scipy
import argparse
import multiprocessing
import matplotlib
#matplotlib.use("cairo")
import matplotlib.pyplot as plt
from matplotlib import ticker
import time
import datetime
import hqrc as hqrc
import utils
from utils import *
from loginit import get_module_logger
import pickle
from scipy import sparse
from sklearn.metrics.pairwise import pairwise_distances
from collections import defaultdict

UNITS=6
BETA=1e-14
INIT_RHO=0
INTERVAL=0.05
V=1

def dumpstates_job(savedir, dynamic, input_seq, nqrc, type_input, type_op, gamma,\
    non_diag_const, tau, xs, idx, bg, ed, interval, send_end, randseed):
    """
    Dump raw data of states
    """
    print('Start pid={} with size {} (from {} to {})'.format(idx, len(xs), xs[0], xs[-1]))
    results = dict()
    basename = '{}_nqr_{}_V_{}_tau_{}_nondiag_{}_gam_{}_op_{}_tp_{}_interval_{}_rtd_{}'.format(\
        dynamic, nqrc, V, tau, non_diag_const, gamma, type_op, type_input, interval, randseed)

    save_figdir = os.path.join(savedir, 'figs')
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(save_figdir, exist_ok=True)
    for x in xs:
        non_diag_var = 10**x
        qparams = QRCParams(n_units=UNITS-1, n_envs=1, max_energy=1.0, \
            non_diag_const=non_diag_const, non_diag_var=non_diag_var,
            beta=BETA, virtual_nodes=V, tau=tau, init_rho=INIT_RHO, solver=LINEAR_PINV, dynamic=dynamic)
        model = hqrc.HQRC(nqrc=nqrc, gamma=gamma, sparsity=1.0, sigma_input=1.0, nonlinear=0, type_input=type_input, type_op=type_op)
        state_list, feed_list = model.init_forward(qparams, input_seq, init_rs = True, ranseed = randseed)
        results[x] = state_list

        # Draw state and feedback
        cmap = plt.get_cmap("viridis")
        plt.style.use('seaborn-colorblind')
        plt.rc('font', family='serif')
        plt.rc('mathtext', fontset='cm')
        plt.rcParams['font.size'] = 12
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
        fig, axs = plt.subplots(nqrc, 2, figsize=(18, 3*nqrc), squeeze=False)

        n_local_nodes = int(state_list.shape[1] / nqrc)
        nobs = int(n_local_nodes / V)
        ts = list(range(bg, ed))
        vmin1, vmax1 = np.amin(input_seq[:,bg:ed]), np.amax(input_seq[:, bg:ed])
        vmin2, vmax2 = np.amin(state_list[bg:ed, :]), np.amax(state_list[bg:ed, :])
        
        if randseed == 5:
            outfile = 'nondiag_var_log_{:.3f}_{}'.format(x, basename)
            for i in range(nqrc):
                ax1, ax2 = axs[i, 0], axs[i, 1]
                ax1.plot(ts, input_seq[i, bg:ed], c='k', label='Input')
                ax1.plot(ts, feed_list[bg:ed, i], '--', c='gray', alpha=0.7, label='Feedback')

                for j in range(nobs):
                    #print(nobs, n_local_nodes, state_list.shape, len(colors))
                    ax2.plot(ts, state_list[bg:ed, i*n_local_nodes + j], c=colors[j], label='QR{}-{}'.format(i+1,j+1))
                
                ax1.legend()
                ax2.legend()
                if i == 0:
                    ax2.set_title('{}'.format(outfile))    
                #ax1.set_ylim([vmin1, vmax1])
                #ax2.set_ylim([vmin2, vmax2])
            for ftype in ['png']:
                figfile = os.path.join(save_figdir, '{}.{}'.format(outfile, ftype))
                plt.savefig(figfile, bbox_inches='tight')

    outbase = os.path.join(savedir, basename)
    filename = '{}_states_id_{}.binaryfile'.format(outbase, idx)
    with open(filename, 'wb') as wrs:
        pickle.dump(results, wrs)
    send_end.send(filename)
    print('Finish pid={} with size {} (from {} to {})'.format(idx, len(xs), xs[0], xs[-1]))

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--length', type=int, default=2000)
    parser.add_argument('--bg', type=int, default=1000, help='start index to view dynamics')
    parser.add_argument('--ed', type=int, default=2000, help='end index to view dynamics')
    parser.add_argument('--nqrc', type=int, default=1, help='Number of reservoirs')
    parser.add_argument('--non_diag_const', type=float, default=2.0, help='The nondiag const')
    parser.add_argument('--tau', type=float, default=10.0, help='Tau')
    parser.add_argument('--gamma', type=float, default=0.0, help='Feedback strength')
    
    parser.add_argument('--type_input', type=int, default=0)
    parser.add_argument('--type_op', type=str, default='Z')
    parser.add_argument('--randseed', type=int, default=0)
    
    parser.add_argument('--nproc', type=int, default=50)
    parser.add_argument('--dynamic', type=str, default=DYNAMIC_PHASE_TRANS,\
        help='full_random,half_random,full_const_trans,full_const_coeff,ion_trap')

    parser.add_argument('--interval', type=float, default=INTERVAL, help='tau-interval')
    parser.add_argument('--savedir', type=str, default='res_topo')
    parser.add_argument('--input_file', type=str, default='../data/random_binary_input.txt')
    
    args = parser.parse_args()
    print(args)

    length, nqrc, nproc, dynamic = args.length, args.nqrc, args.nproc, args.dynamic
    bg, ed, type_input, non_diag_const, tau = args.bg, args.ed, args.type_input, args.non_diag_const, args.tau
    type_op, gamma = args.type_op, args.gamma

    randseed = args.randseed
    interval = args.interval

    savedir = args.savedir
    if os.path.isfile(savedir) == False and os.path.isdir(savedir) == False:
        os.mkdir(savedir)
    
    # KEEP CONSTANT interval = 0.05
    tx = list(np.arange(-2, 2.1, interval))
    nproc = min(len(tx), nproc)
    lst = np.array_split(tx, nproc)

    if os.path.isfile(savedir) == False:
        # prepare the data
        if os.path.isfile(args.input_file):
            # Read from file: 
            data = np.loadtxt(args.input_file)[:length,1]
        else:
            print('File not found {}'.format(args.input_file))
            exit(1)
        # if type_input == 0 or type_input == 2:
        #     data = np.random.rand(length)
        # elif type_input == 1:
        #     # Read from file: random_binary_input
        #     data = np.loadtxt(args.input_file)[:length,1]

        input_seq = np.array(data)
        us_seq = input_seq[bg:ed]
        print(input_seq.shape)
        input_seq = np.tile(input_seq, (nqrc, 1))

        jobs, pipels = [], []
        for pid in range(nproc):
            xs = lst[pid]
            recv_end, send_end = multiprocessing.Pipe(False)
            p = multiprocessing.Process(target=dumpstates_job, args=(savedir, dynamic, input_seq, nqrc, type_input, type_op, gamma,\
                non_diag_const, tau, xs, pid, bg, ed, interval, send_end, randseed))

            jobs.append(p)
            pipels.append(recv_end)
        # Start the process
        for p in jobs:
            p.start()

        # Ensure all processes have finished execution
        for p in jobs:
            p.join()
        
        # Join dumpled pickle files
        z = dict()
        for px in pipels:
            filename = px.recv()
            with open(filename, 'rb') as rrs:
                tmp = pickle.load(rrs)
                z = dict(list(z.items()) + list(tmp.items()))
            # Delete file
            os.remove(filename)
            print('zlen={}, Deleted {}'.format(len(z), filename))

        filename = filename.replace('.binaryfile', '_len_{}.binaryfile'.format(length))
        with open(filename, 'wb') as wrs:
            pickle.dump(z, wrs)
    else:
        filename = savedir
        with open(filename, 'rb') as rrs:
            z = pickle.load(rrs)
    
    # Plot file
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams["font.size"] = 14 # 全体のフォントサイズが変更されます
    plt.rcParams['xtick.labelsize'] = 16 # 軸だけ変更されます
    plt.rcParams['ytick.labelsize'] = 16 # 軸だけ変更されます

    #fig, axs = plt.subplots(1, 1, figsize=(8, 6), squeeze=False, dpi=600)
    #ax1 = axs.ravel()[0]
    #ax.plot(ts, rs, ls="", marker=",")

    ids = np.array([0, 20 , 30, 40, 50, 60, 80]) * int(INTERVAL / interval)
    fig, axs = plt.subplots(len(ids), 7, figsize=(36, 24), squeeze=False, gridspec_kw={'width_ratios': [6, 1, 1, 1, 1, 1, 1]})

    for i in range(len(ids)):
        ax = axs[i, 0]
        x = tx[int(ids[i])]
        state_list = z[x]
        print(state_list.shape)
        for j in range(UNITS):
            ys = state_list[bg:ed, j].ravel()
            bs = range(bg,ed)
            # Plot time series of reservoir states
            ax.plot(bs, ys, label='spin-{}'.format(j+1), linewidth=2)
            # Plot input vs states
            axs[i, j+1].plot(us_seq, ys)

        ax.set_title('W=10^{:.2f}'.format(x))
        #ax2.set_yticklabels([])
        #ax2.set_xticklabels([])
        ax.legend()

    outbase = filename.replace('.binaryfile', '_bg_{}_ed_{}'.format(bg, ed))
    for ftype in ['png']:
        plt.savefig('{}_v1.{}'.format(outbase, ftype), bbox_inches='tight', dpi=600)
    plt.show()
