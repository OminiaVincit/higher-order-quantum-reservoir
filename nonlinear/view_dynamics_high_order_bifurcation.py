import sys
import numpy as np
import os
import scipy
import argparse
import multiprocessing
import matplotlib
matplotlib.use("Qt5Cairo")
import matplotlib.pyplot as plt
from matplotlib import ticker
import tqdm
import time
import datetime
import highorder_qrc as hqrc
import qrc
import gendata as gen
import utils
from loginit import get_module_logger
from scipy.stats import gaussian_kde

UNITS=5
J=1.0
BETA=1e-14
INIT_RHO=0
TROTTER_STEP=10
V=1

def bifurcation_job(savedir, basename, input_seq, buffer, length, nqrc, layer_strength):
    plt.rc('font', family='serif', size=8)
    plt.rc('mathtext', fontset='cm')
    fig, axs = plt.subplots(1, 1, figsize=(8, 6), squeeze=False, dpi=600)
    ax = axs.ravel()[0]
    
    bg, ed = buffer, length
    tx = np.arange(-7,7.1,0.01)
    #tx = np.arange(-4,7.1,1)
    taudeltas = [2**x for x in tx]
    nqrc = args.nqrc

    # plot the scatter (bifurcation diagram)
    
    ts, rs = [], []
    for i in range(len(taudeltas)):
        tau_delta = taudeltas[i]
        qparams = qrc.QRCParams(hidden_unit_count=UNITS, max_coupling_energy=J,\
            trotter_step=TROTTER_STEP, beta=BETA, virtual_nodes=V, tau_delta=tau_delta, init_rho=INIT_RHO)
    
        model = hqrc.HighorderQuantumReservoirComputing(nqrc, layer_strength)
        x0_state_list = model.init_forward(qparams, input_seq, init_rs = True, ranseed = 0)
        if i % 10 == 0:
            print(i, layer_strength, tau_delta, x0_state_list.shape)
        ys = x0_state_list[bg:ed, 1:UNITS].ravel()
        #xs = np.array([tau_delta] * len(ys)).ravel()
        rs.extend(ys)
        ts.extend([tau_delta] * len(ys))
    
        #xy = np.vstack([xs, ys])
        #z = gaussian_kde(xy)(xy)
        #ax.scatter(xs, ys, c=z, s=(8*72./fig.dpi)**2, marker='o', cmap='brg', lw=0, rasterized=True)
    
    #ax.plot(ts, rs, ls="", marker=",")
    ts = np.array(ts).ravel()
    rs = np.array(rs).ravel()
    ax.scatter(ts, rs, s=(12*72./fig.dpi)**2, marker='o', lw=0, rasterized=True)
    
    ax.set_title('$\\alpha$ = {}'.format(layer_strength))
    ax.set_xscale("log", basex=2)
    ax.set_yscale("symlog", basey=10, linthreshy=1e-5)
    outbase = os.path.join(savedir, '{}_layers_{}_V_{}_strength_{}'.format(basename, \
        nqrc, V, layer_strength))
    for ftype in ['png', 'svg']:
        plt.savefig('{}_bif.{}'.format(outbase, ftype), bbox_inches='tight', dpi=600)

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--length', type=int, default=2000)
    parser.add_argument('--buffer', type=int, default=1000)

    parser.add_argument('--nqrc', type=int, default=5)
    parser.add_argument('--strengths', type=str, default='0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0')
    parser.add_argument('--virtuals', type=int, default=1)

    parser.add_argument('--basename', type=str, default='qrc_dyn')
    parser.add_argument('--savedir', type=str, default='res_bifurcation')
    args = parser.parse_args()
    print(args)

    length, buffer, nqrc = args.length, args.buffer, args.nqrc
    strengths = [float(x) for x in args.strengths.split(',')]

    basename, savedir = args.basename, args.savedir
    if os.path.isfile(savedir) == False and os.path.isdir(savedir) == False:
        os.mkdir(savedir)
    
    # Evaluation
    timestamp = int(time.time() * 1000.0)
    now = datetime.datetime.now()
    datestr = now.strftime('{0:%Y-%m-%d-%H-%M-%S}'.format(now))
    
    # prepare the data
    np.random.seed(seed=1000)
    data = np.random.rand(length)
    input_seq = np.array(data)
    input_seq = np.tile(input_seq, (nqrc, 1))

    jobs = []
    for layer_strength in strengths:
        p = multiprocessing.Process(target=bifurcation_job, args=(savedir, basename, input_seq, buffer, length, nqrc, layer_strength))
        jobs.append(p)
    # Start the process
    for p in jobs:
        p.start()

    # Ensure all processes have finished execution
    for p in jobs:
        p.join()