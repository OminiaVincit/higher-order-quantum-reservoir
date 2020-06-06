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
from utils import QRCParams
from loginit import get_module_logger
from scipy.stats import gaussian_kde
import pickle

UNITS=5
J=1.0
BETA=1e-14
INIT_RHO=0
V=1

def dumpstates_job(savedir, basename, input_seq, nqrc, layer_strength, xs, idx, send_end):
    """
    Dump raw data of states
    """
    print('Start pid={} with size {} (from {} to {})'.format(idx, len(xs), xs[0], xs[-1]))
    results = dict()
    for x in xs:
        tau_delta = 2**x
        qparams = QRCParams(hidden_unit_count=UNITS, max_coupling_energy=J,\
            beta=BETA, virtual_nodes=V, tau_delta=tau_delta, init_rho=INIT_RHO)
        model = hqrc.HighorderQuantumReservoirComputing(nqrc, layer_strength)
        state_list = model.init_forward(qparams, input_seq, init_rs = True, ranseed = 0)
        results[x] = state_list
    
    outbase = os.path.join(savedir, '{}_layers_{}_V_{}_strength_{}'.format(basename, \
        nqrc, V, layer_strength))
    filename = '{}_states_id_{}.binaryfile'.format(outbase, idx)
    with open(filename, 'wb') as wrs:
        pickle.dump(results, wrs)
    send_end.send(filename)
    print('Finish pid={} with size {} (from {} to {})'.format(idx, len(xs), xs[0], xs[-1]))

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--length', type=int, default=2000)
    parser.add_argument('--bg', type=int, default=1000)
    parser.add_argument('--ed', type=int, default=2000)
    parser.add_argument('--const', type=int, default=0, help='constant input')
    parser.add_argument('--nqrc', type=int, default=5)
    parser.add_argument('--strength', type=float, default=0.5)
    parser.add_argument('--nproc', type=int, default=50)

    parser.add_argument('--interval', type=float, default=0.05, help='tau-interval')
    parser.add_argument('--basename', type=str, default='qrc_dyn')
    parser.add_argument('--savedir', type=str, default='res_states')
    args = parser.parse_args()
    print(args)

    length, nqrc, nproc = args.length, args.nqrc, args.nproc
    bg, ed = args.bg, args.ed
    layer_strength = args.strength
    const_input = args.const

    basename = '{}_const_input_{}'.format(args.basename, args.const)
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
            p = multiprocessing.Process(target=dumpstates_job, args=(savedir, basename, input_seq, \
                nqrc, layer_strength, xs, pid, send_end))
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

        filename = filename.replace('.binaryfile', 'len_{}.binaryfile'.format(length))
        with open(filename, 'wb') as wrs:
            pickle.dump(z, wrs)
    else:
        filename = savedir
        with open(filename, 'rb') as rrs:
            z = pickle.load(rrs)
    
    rs, ts = [], []
    for x in tx:
        state_list = z[x]
        ys = state_list[bg:ed, 1:UNITS].ravel()
        rs.extend(ys)
        ts.extend([2**x] * len(ys))
    ts = np.array(ts).ravel()
    rs = np.array(rs).ravel()

    # Plot file
    plt.rc('font', family='serif', size=8)
    plt.rc('mathtext', fontset='cm')

    #fig, axs = plt.subplots(1, 1, figsize=(8, 6), squeeze=False, dpi=600)
    #ax1 = axs.ravel()[0]
    
    #ax.plot(ts, rs, ls="", marker=",")

    fig = plt.figure(figsize=(8, 6), dpi=600)
    ax1 = plt.subplot2grid((4,3), (0,0), colspan=2, rowspan=2)

    if False:
        # Very slow to run density plot
        xy = np.vstack([ts, rs])
        z = gaussian_kde(xy)(xy)
        ax1.scatter(ts, rs, c=z, s=(12*72./fig.dpi)**2, marker='o', cmap='brg', lw=0, rasterized=True)
    else:
        ax1.scatter(ts, rs, s=(12*72./fig.dpi)**2, marker='o', lw=0, rasterized=True)
    
    ax1.set_title('{}'.format(os.path.basename(filename)))
    ax1.set_xscale("log", basex=2)
    ax1.set_yscale("symlog", basey=10, linthreshy=1e-5)
    ax1.grid(alpha=0.8,axis='x')
    ax1.set_xticks([2**x for x in np.arange(-7,7.1,1.0)])
    ax1.minorticks_on()
    ax1.tick_params('both', length=6, width=1, which='major')
    ax1.tick_params('both', length=3, width=1, which='minor')
    #ax1.set_xlim([2**(-2), 2**(0)])
    ids = [20, 60, 80, 180]
    for i in range(len(ids)):
        ax2 = plt.subplot2grid((4,3), (i,2))
        x = tx[ids[i]]
        state_list = z[x]
        for j in range(1, UNITS):
            ys = state_list[bg:ed, j].ravel()
            ax2.plot(ys)
        ax2.set_title('2^{:.1f}'.format(x))
        ax2.set_yticklabels([])
        ax2.set_xticklabels([])
        
    outbase = filename.replace('.binaryfile', '_bg_{}_ed_{}'.format(bg, ed))
    for ftype in ['png','svg']:
        plt.savefig('{}_v3.{}'.format(outbase, ftype), bbox_inches='tight', dpi=600)
    plt.show()
