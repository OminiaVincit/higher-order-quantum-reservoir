#!/usr/bin/env python
"""
    Calculate NMSE for NARMA tasks with quantum innate learning
    See run_hqrc_innate_narma.sh for an example to run the script
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
from scipy.special import legendre
import hqrc
import utils
from utils import *

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--units', type=int, default=5, help='Number of the hidden units')
    parser.add_argument('--coupling', type=float, default=1.0, help='Maximum coupling energy')
    parser.add_argument('--rho', type=int, default=0, help='Flag for initializing the density matrix')
    parser.add_argument('--beta', type=float, default=1e-14, help='reg term')
    parser.add_argument('--solver', type=str, default=LINEAR_PINV, \
        help='regression solver by linear_pinv,ridge_pinv,auto,svd,cholesky,lsqr,sparse_cg,sag')

    parser.add_argument('--length', type=int, default=2000)
    
    parser.add_argument('--virtuals', type=int, default=20)
    parser.add_argument('--taudelta', type=float, default=2.0, help='Interval between the inputs')
    parser.add_argument('--strength', type=float, default=0.0, help='Connection strengths')
    parser.add_argument('--nqrc', type=int, default=5, help='Number of reservoirs')

    parser.add_argument('--orders', type=str, default='10')
    parser.add_argument('--basename', type=str, default='dynamics')
    parser.add_argument('--savedir', type=str, default='view_dynamic')
    parser.add_argument('--ranseed', type=int, default=1)
    args = parser.parse_args()
    print(args)

    n_units, max_energy, beta, length = args.units, args.coupling, args.beta, args.length
    V = args.virtuals
    init_rho, solver = args.rho, args.solver
    ranseed = args.ranseed
    basename, savedir = args.basename, args.savedir
    if os.path.isdir(savedir) == False:
        os.mkdir(savedir)
    tau, alpha, nqrc = args.taudelta, args.strength, args.nqrc
    orders = [int(x) for x in args.orders.split(',')]

    # Evaluation
    timestamp = int(time.time() * 1000.0)
    now = datetime.datetime.now()
    datestr = now.strftime('{0:%Y-%m-%d-%H-%M-%S}'.format(now))

    outbase = os.path.join(savedir, '{}_{}_{}_units_{}_V_{}_a_{}_QRs_{}_sd_{}'.format(\
        basename, solver, datestr, n_units, V, alpha, nqrc, ranseed))

    # save experiments setting
    with open('{}_setting.txt'.format(outbase), 'w') as sfile:
        sfile.write('length={}\n'.format(length))
        sfile.write('n_units={}\n'.format(n_units))
        sfile.write('max_energy={}\n'.format(max_energy))
        sfile.write('beta={}\n'.format(beta))
        sfile.write('taudelta={}\n'.format(tau))
        sfile.write('layers={}\n'.format(nqrc))
        sfile.write('V={}, alpha={}\n'.format(V, alpha))

    np.random.seed(seed=ranseed)    
    # Create input - target
    data = np.random.rand(length)
    #data = np.random.randint(2, size=train_len + val_len)

    # Create qparams and model
    qparams = QRCParams(n_units=n_units, max_energy=max_energy,\
        beta=beta, virtual_nodes=V, tau=tau, init_rho=False)
    model = hqrc.HQRC(nqrc, alpha)

    pre_input_seq_org = np.array(data)
    pre_input_seq_org = pre_input_seq_org.reshape(1, pre_input_seq_org.shape[0])
    pre_input_seq = np.tile(pre_input_seq_org, (nqrc, 1))
    
    state_list = model.init_forward(qparams, pre_input_seq, ranseed=ranseed + 100, init_rs=True)
    L, D = state_list.shape
    # L = Length of time series
    # D = nqrc x V x Number of qubits
    buffer = length//2
    bg, ed = buffer, buffer+5
    ts = range(buffer, buffer+5)
    n_qubits = model.n_qubits
    n_local = V * n_qubits

    xs = []
    
    for q in range(n_qubits):
        series = []
        for t in ts:
            for v in range(V):
                series.append(state_list[t, q + v*n_qubits])
        xs.append(series)

    cmap = plt.get_cmap("viridis")
    plt.style.use('seaborn-colorblind')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams['font.size']=20
    fig = plt.figure(figsize=(20, 8))
    
    plt.subplots_adjust(wspace=0.4, hspace=0.5)
    
    # Plot input
    # ax = fig.add_subplot(2, 1, 1)
    # ax.plot(pre_input_seq[0, bg:ed], 'o--')
    # ax.set_ylabel('Input')
    # ax.set_title(outbase)

    # Plot states
    ax = fig.add_subplot(1, 1, 1)
    for i in range(1,len(xs)):
        x = xs[i]
        ax.plot(range(len(x)), x, label='qubit-{}'.format(i+1), linewidth=2)

    ax.legend()

    for ftype in ['png']:
        plt.savefig('{}.{}'.format(outbase, ftype), bbox_inches='tight', dpi=600)
    plt.show()