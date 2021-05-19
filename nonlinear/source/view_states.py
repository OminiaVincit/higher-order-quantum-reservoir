#!/usr/bin/env python
"""
    View bifurcation diagrams and dynamics for higher-order quantum reservoir
    See run_view_states.sh for an example of the running script
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

UNITS=5
BETA=1e-14
INIT_RHO=0
V=1
INTERVAL=0.05

def dumpstates_job(savedir, dynamic, input_seq, nqrc, layer_strength, nonlinear, sigma_input, sparsity,\
    xs, idx, send_end):
    """
    Dump raw data of states
    """
    print('Start pid={} with size {} (from {} to {})'.format(idx, len(xs), xs[0], xs[-1]))
    results = dict()
    for x in xs:
        tau = 2**x
        qparams = QRCParams(n_units=UNITS-1, n_envs=1, max_energy=1.0,\
            beta=BETA, virtual_nodes=V, tau=tau, init_rho=INIT_RHO, solver=LINEAR_PINV, dynamic=dynamic)
        model = hqrc.HQRC(nqrc=nqrc, gamma=layer_strength, sparsity=sparsity, sigma_input=sigma_input, nonlinear=nonlinear)
        state_list = model.init_forward(qparams, input_seq, init_rs = True, ranseed = 0)
        results[x] = state_list*2.0-1.0
    
    outbase = os.path.join(savedir, '{}_layers_{}_V_{}_nonlinear_{}_strength_{}_sigma_{}_sparse_{}'.format(dynamic, \
        nqrc, V, nonlinear, layer_strength, sigma_input, sparsity))
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
    parser.add_argument('--const', type=int, default=0, help='flag to set constant input')
    parser.add_argument('--nqrc', type=int, default=5, help='Number of reservoirs')
    parser.add_argument('--strength', type=float, default=0.5, help='The connection strength')
    parser.add_argument('--sparsity', type=float, default=1.0, help='The sparsity of the connection strength')
    parser.add_argument('--sigma_input', type=float, default=1.0, help='The sigma input for the feedback')
    parser.add_argument('--nonlinear', type=int, default=0, help='The nonlinear of feedback matrix')
    parser.add_argument('--nproc', type=int, default=50)
    parser.add_argument('--dynamic', type=str, default=DYNAMIC_FULL_RANDOM,\
        help='full_random,half_random,full_const_trans,full_const_coeff,ion_trap')

    parser.add_argument('--interval', type=float, default=INTERVAL, help='tau-interval')
    parser.add_argument('--savedir', type=str, default='res_states')
    args = parser.parse_args()
    print(args)

    length, nqrc, nproc, dynamic = args.length, args.nqrc, args.nproc, args.dynamic
    bg, ed = args.bg, args.ed
    layer_strength, nonlinear, sparsity, sigma_input = args.strength, args.nonlinear, args.sparsity, args.sigma_input
    const_input = args.const
    basename = '{}_nqrc_{}_V_{}_sm_{}_a_{}_sigma_{}_sparse_{}'.format(dynamic, \
        nqrc, V, nonlinear, layer_strength, sigma_input, sparsity)

    savedir = args.savedir
    if os.path.isfile(savedir) == False and os.path.isdir(savedir) == False:
        os.mkdir(savedir)
    
    # KEEP CONSTANT interval = 0.05
    tx = list(np.arange(-7, 7.1, args.interval))
    nproc = min(len(tx), nproc)
    lst = np.array_split(tx, nproc)

    if os.path.isfile(savedir) == False:
        # prepare the data
        if const_input == 0:
            np.random.seed(seed=1000)
            data = np.random.rand(length)
        else:
            data = np.zeros(length)
        input_seq = np.array(data)
        input_seq = np.tile(input_seq, (nqrc, 1))
        
        jobs, pipels = [], []
        for pid in range(nproc):
            xs = lst[pid]
            recv_end, send_end = multiprocessing.Pipe(False)
            p = multiprocessing.Process(target=dumpstates_job, args=(savedir, dynamic, input_seq, \
                nqrc, layer_strength, nonlinear, sigma_input, sparsity, xs, pid, send_end))
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

        filename = filename.replace('.binaryfile', 'const_{}_len_{}.binaryfile'.format(const_input, length))
        with open(filename, 'wb') as wrs:
            pickle.dump(z, wrs)
    else:
        filename = savedir
        with open(filename, 'rb') as rrs:
            z = pickle.load(rrs)
    
    # Plot file
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams["font.size"] = 8 # 全体のフォントサイズが変更されます
    plt.rcParams['xtick.labelsize'] = 8 # 軸だけ変更されます
    plt.rcParams['ytick.labelsize'] = 8 # 軸だけ変更されます

    #fig, axs = plt.subplots(1, 1, figsize=(8, 6), squeeze=False, dpi=600)
    #ax1 = axs.ravel()[0]
    #ax.plot(ts, rs, ls="", marker=",")

    fig = plt.figure(figsize=(8, 10), dpi=600)
    for i in range(nqrc):
        sbg = 1 + i*UNITS
        sed = (i+1)*UNITS
        rs, ts = [], []
        for x in tx:
            state_list = z[x]
            ys = state_list[bg:ed, sbg:sed].ravel()
            rs.extend(ys)
            ts.extend([2**x] * len(ys))
        ts = np.array(ts).ravel()
        rs = np.array(rs).ravel()

        ax1 = plt.subplot2grid((nqrc,3), (i,0), colspan=2, rowspan=1)
        
        if False:
            # Very slow to run density plot
            xy = np.vstack([ts, rs])
            z = gaussian_kde(xy)(xy)
            ax1.scatter(ts, rs, c=z, s=(12*72./fig.dpi)**2, marker='o', cmap='brg', lw=0, rasterized=True)
        else:
            ax1.scatter(ts, rs, s=(12*72./fig.dpi)**2, marker='o', lw=0, rasterized=True)
        
        if i == 0:
            ax1.set_title('{}_QR_{}'.format(basename, i+1), fontsize=8)
        ax1.set_xscale("log", base=2)
        ax1.set_yscale("symlog", base=10, linthresh=1e-5)
        ax1.grid(alpha=0.8, axis='x')
        ax1.set_xticks([2**x for x in np.arange(-7,7.1,1.0)])
        ax1.minorticks_on()
        ax1.tick_params('both', length=6, width=1, which='major')
        ax1.tick_params('both', length=3, width=1, which='minor')
        ax1.set_xlim([2**tx[0], 2**tx[-1]])

    ids = [20, 60, 80, 180]
    for i in range(len(ids)):
        ax2 = plt.subplot2grid((nqrc,3), (i,2))
        x = tx[ids[i]]
        state_list = z[x]
        for j in range(1, UNITS):
            ys = state_list[bg:ed, j].ravel()
            ax2.plot(ys)
        ax2.set_title('2^{:.1f}'.format(x))
        ax2.set_yticklabels([])
        ax2.set_xticklabels([])
        
    outbase = filename.replace('.binaryfile', '_bg_{}_ed_{}'.format(bg, ed))
    for ftype in ['png', 'svg']:
        plt.savefig('{}_v4.{}'.format(outbase, ftype), bbox_inches='tight', dpi=600)
    plt.show()
