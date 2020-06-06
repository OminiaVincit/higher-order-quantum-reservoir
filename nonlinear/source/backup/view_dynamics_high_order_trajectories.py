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
import hqrc as hqrc
import utils
from utils import QRCParams
from loginit import get_module_logger

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--units', type=int, default=5)
    parser.add_argument('--coupling', type=float, default=1.0)
    parser.add_argument('--trotter', type=int, default=10)
    parser.add_argument('--rho', type=int, default=0)
    parser.add_argument('--beta', type=float, default=1e-14)

    parser.add_argument('--length', type=int, default=2000)
    parser.add_argument('--buffer', type=int, default=1000)

    parser.add_argument('--taudeltas', type=str, default='-6,2')
    parser.add_argument('--nqrc', type=int, default=5)
    parser.add_argument('--strengths', type=str, default='0.0,0.5,1.0')
    parser.add_argument('--virtuals', type=int, default=1)

    parser.add_argument('--basename', type=str, default='qrc_dyn')
    parser.add_argument('--savedir', type=str, default='res_dynamics2')
    args = parser.parse_args()
    print(args)

    hidden_unit_count, max_coupling_energy, trotter_step, beta =\
        args.units, args.coupling, args.trotter, args.beta
    length, buffer = args.length, args.buffer
    V = args.virtuals
    init_rho = args.rho

    basename, savedir = args.basename, args.savedir
    if os.path.isfile(savedir) == False and os.path.isdir(savedir) == False:
        os.mkdir(savedir)

    taudeltas = [float(x) for x in args.taudeltas.split(',')]
    taudeltas = [2**x for x in taudeltas]
    strengths = [float(x) for x in args.strengths.split(',')]
    nqrc = args.nqrc
    
    # Evaluation
    timestamp = int(time.time() * 1000.0)
    now = datetime.datetime.now()
    datestr = now.strftime('{0:%Y-%m-%d-%H-%M-%S}'.format(now))
    
    # prepare the data
    np.random.seed(seed=1000)
    data = np.random.rand(length)
    input_seq = np.array(data)
    input_seq = np.tile(input_seq, (nqrc, 1))

    plt.rc('font', family='serif', size=8)
    plt.rc('mathtext', fontset='cm')
    fig, axs = plt.subplots(1, 1, figsize=(4, 4), squeeze=False)
    axs = axs.ravel()
    ax = axs[0]

    bg = int((buffer + length) / 2)
    ed = bg + 100

    # plot the signals
    for layer_strength in strengths:
        ams = []
        for tau_delta in taudeltas:
            qparams = QRCParams(hidden_unit_count=hidden_unit_count, max_coupling_energy=max_coupling_energy,\
                trotter_step=trotter_step, beta=beta, virtual_nodes=V, tau_delta=tau_delta, init_rho=init_rho)
        
            model = hqrc.HighorderQuantumReservoirComputing(nqrc, layer_strength)
            x0_state_list = model.init_forward(qparams, input_seq, init_rs = True, ranseed = 0)
            print(layer_strength, tau_delta, x0_state_list.shape)
            ys = x0_state_list[bg:ed, 1:hidden_unit_count].ravel()
            ams.append(np.average(ys))
        print(layer_strength, ams)
        ax.plot(taudeltas, ams, label='{}'.format(layer_strength))
    #ax.set_title('$\\alpha$ = {}, $\\tau$ = {}'.format(layer_strength, tau_delta))
    ax.legend()
    ax.set_xscale("log", basex=2)
    ax.set_yscale("symlog", basey=10, linthreshy=1e-5)
    ax.grid()
    outbase = os.path.join(savedir, '{}_layers_{}_V_{}_rho_{}'.format(basename, \
        nqrc, V, init_rho))
    for ftype in ['png']:
        plt.savefig('{}_interval.{}'.format(outbase, ftype), bbox_inches='tight')