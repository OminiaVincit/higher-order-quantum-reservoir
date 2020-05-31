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
import highorder_qrc as hqrc
import qrc
import gendata as gen
import utils
from loginit import get_module_logger

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--units', type=int, default=5)
    parser.add_argument('--coupling', type=float, default=1.0)
    parser.add_argument('--trotter', type=int, default=10)
    parser.add_argument('--rho', type=int, default=0)
    parser.add_argument('--beta', type=float, default=1e-14)

    parser.add_argument('--length', type=int, default=500)
    parser.add_argument('--buffer', type=int, default=100)

    parser.add_argument('--taudeltas', type=str, default='-3,1,4')
    parser.add_argument('--nqrc', type=int, default=5)
    parser.add_argument('--strength', type=float, default=0.0)
    parser.add_argument('--virtuals', type=int, default=50)

    parser.add_argument('--basename', type=str, default='qrc_dyn')
    parser.add_argument('--savedir', type=str, default='res_dynamics')
    args = parser.parse_args()
    print(args)

    hidden_unit_count, max_coupling_energy, trotter_step, beta =\
        args.units, args.coupling, args.trotter, args.beta
    length, buffer = args.length, args.buffer
    layer_strength, V = args.strength, args.virtuals
    init_rho = args.rho

    basename, savedir = args.basename, args.savedir
    if os.path.isfile(savedir) == False and os.path.isdir(savedir) == False:
        os.mkdir(savedir)

    taudeltas = [float(x) for x in args.taudeltas.split(',')]
    taudeltas = [2**x for x in taudeltas]
    
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

    N = len(taudeltas)

    plt.rc('font', family='serif', size=14)
    plt.rc('mathtext', fontset='cm')
    fig, axs = plt.subplots(1, N, figsize=(4*N, 3.6), squeeze=False)
    axs = axs.ravel()
    
    bg = int((buffer + length) / 2)
    ed = bg + 2

    # plot the signals
    for i in range(N):
        ax = axs[i]
        tau_delta = taudeltas[i]

        qparams = qrc.QRCParams(hidden_unit_count=hidden_unit_count, max_coupling_energy=max_coupling_energy,\
            trotter_step=trotter_step, beta=beta, virtual_nodes=V, tau_delta=tau_delta, init_rho=init_rho)
    
        model = hqrc.HighorderQuantumReservoirComputing(nqrc, layer_strength)
        x0_state_list = model.init_forward(qparams, input_seq, init_rs = True, ranseed = 0)
        print(x0_state_list.shape)

        #ts = list(range(bg, ed))
        Ntot = hidden_unit_count * V
        # State of the first QR
        tmp_state_list = x0_state_list[:, 0:Ntot]
        for j in range(hidden_unit_count):
            ys = tmp_state_list[bg:ed, j::V].ravel()
            lst = "solid"
            if j == 0:
                lst = "dashed"
            else:
                ax.plot(ys, linestyle=lst)
        ax.set_title('$\\alpha$ = {}, $\\tau$ = {}'.format(layer_strength, tau_delta))
        ax.set_xticks([])
        # if i != 0:
        #     ax.set_ylim([-0.03, 0.04])
    outbase = os.path.join(savedir, '{}_layers_{}_V_{}_strength_{}_rho_{}'.format(basename, \
        nqrc, V, layer_strength, init_rho))
    for ftype in ['png']:
        plt.savefig('{}_interval2.{}'.format(outbase, ftype), bbox_inches='tight')