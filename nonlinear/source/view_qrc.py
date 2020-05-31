import sys
import numpy as np
import os
import scipy
import argparse
import multiprocessing
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
import tqdm
import time
import datetime
import qrc
import gendata as gen
import utils

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--units', type=int, default=5)
    parser.add_argument('--coupling', type=float, default=1.0)
    parser.add_argument('--trotter', type=int, default=10)
    parser.add_argument('--beta', type=float, default=1e-14)
    parser.add_argument('--virtuals', type=int, default=10)
    parser.add_argument('--tdelta', type=float, default=10)

    parser.add_argument('--basename', type=str, default='series')

    parser.add_argument('--vallen', type=int, default=500)
    parser.add_argument('--buffer', type=int, default=500)
    parser.add_argument('--savedir', type=str, default='viewqrc')
    args = parser.parse_args()
    print(args)

    hidden_unit_count, max_coupling_energy, trotter_step, beta =\
        args.units, args.coupling, args.trotter, args.beta
    val_len, buffer = args.vallen, args.buffer
    basename, savedir = args.basename, args.savedir
    V, tdelta = args.virtuals, args.tdelta

    if os.path.isfile(savedir) == False and os.path.isdir(savedir) == False:
        os.mkdir(savedir)

    data = np.random.rand(buffer + val_len)
    input_seq_ls = np.array(  [ data[: buffer + val_len] ] )
    
    # Evaluation
    model = qrc.QuantumReservoirComputing()
    qparams = qrc.QRCParams(hidden_unit_count=hidden_unit_count, max_coupling_energy=max_coupling_energy,\
            trotter_step=trotter_step, beta=beta, virtual_nodes=V, tau_delta=tdelta, init_rho=0)
    state_list = model.init_forward(qparams, input_seq_ls, ranseed=-1, init_rs=True)
    N, L, D = state_list.shape
    # N = Number of input list
    # L = Length of time series
    # D = Number of virtual nodes x Number of qubits
    xs = []
    ts = range(buffer, buffer+5)
    nQ = args.units

    for q in range(args.units):
        series = []
        for t in ts:
            for v in range(V):
                series.append(state_list[0, t, q + v*nQ])
        xs.append(series)

    # Plot
    timestamp = int(time.time() * 1000.0)
    now = datetime.datetime.now()
    datestr = now.strftime('{0:%Y-%m-%d-%H-%M-%S}'.format(now))
    outbase = os.path.join(savedir, '{}_{}'.format(basename, datestr))
    
    if True:
        cmap = plt.get_cmap("viridis")
        plt.figure(figsize=(16,8))
        plt.style.use('seaborn-colorblind')
        plt.rc('font', family='serif')
        plt.rc('mathtext', fontset='cm')
        plt.rcParams['font.size']=20
        for i in range(len(xs)):
            x = xs[i]
            plt.plot(range(len(x)), x, 'o--', label='qubit-{}'.format(i+1))
        

        plt.xlabel('time', fontsize=32)
        plt.ylabel('x', fontsize=32)
        plt.legend()
        #plt.show()
        for ftype in ['pdf']:
            plt.savefig('{}.{}'.format(outbase, ftype), bbox_inches='tight')
        
