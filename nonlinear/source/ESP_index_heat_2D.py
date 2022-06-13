#!/usr/bin/env python
"""
    Calculate ESP index for different transverse field parameters
    See run_esp_index.sh for an example of the running script
"""
import sys
import numpy as np
import os
import argparse
import multiprocessing
import matplotlib
#matplotlib.use("cairo")
import matplotlib.pyplot as plt
from matplotlib import ticker
import hqrc as hqrc
from utils import *
from loginit import get_module_logger
import pickle
from collections import defaultdict
from matplotlib.colors import LogNorm, SymLogNorm

UNITS=6
BETA=1e-14
INIT_RHO=1
V=1

def dum_esp_index_job(savedir, dynamic, input_seq, nqrc, type_input, type_op,\
    non_diag_const, tau, log_Ws, log_gams, idx, buffer, send_end, num_trials, randseed):
    """
    Dump raw data of states
    """
    print('Start pid={} with size {} (from {} to {})'.format(idx, len(log_gams), log_gams[0], log_gams[-1]))
    results = dict()

    basename = 'heat_{}_nqr_{}_V_{}_tau_{}_nondiag_{}_op_{}_tp_{}_trials_{}_rsd_{}'.format(\
        dynamic, nqrc, V, tau, non_diag_const, type_op, type_input, num_trials, randseed)
    os.makedirs(savedir, exist_ok=True)
    
    for log_gam in log_gams:
        gamma = 10**log_gam
        for log_W in log_Ws:
            non_diag_var = 10**log_W
            qparams = QRCParams(n_units=UNITS-1, n_envs=1, max_energy=1.0, \
                non_diag_const=non_diag_const, non_diag_var=non_diag_var,
                beta=BETA, virtual_nodes=V, tau=tau, init_rho=INIT_RHO, solver=LINEAR_PINV, dynamic=dynamic)
            model = hqrc.HQRC(nqrc=nqrc, gamma=gamma, sparsity=1.0, sigma_input=1.0, nonlinear=0, type_input=type_input, type_op=type_op)
            
            x0_state_list, _ = model.init_forward(qparams, input_seq, init_rs = True, ranseed = randseed)
            # Compute esp index and esp_lambda
            dP = []

            for i in range(num_trials):
                # Initialzie the reservoir to a random initial state
                # Keep same coupling configuration
                model.gen_rand_rhos(ranseed = i + 300000)
                z0_state_list, _ = model.init_forward(qparams, input_seq, init_rs = False, ranseed = i + 200000)
                L, D = z0_state_list.shape
                # L = Length of time series
                # D = Number of layers x Number of virtual nodes x Number of qubits
                # print('i={}, State shape'.format(i), z0_state_list.shape)
                local_diff = 0
                # prev, current = None, None
                for t in range(buffer, L):
                    diff_state = x0_state_list[t, :] - z0_state_list[t, :]
                    diff = np.sqrt(np.power(diff_state, 2).sum())
                    local_diff += diff
                local_diff = local_diff / (L-buffer)
                dP.append(local_diff)

            esp_index = np.mean(dP)
            key_str = 'log_W_{:.3f}_gam_{:.3f}'.format(log_W, log_gam)
            results[key_str] = esp_index

    outbase = os.path.join(savedir, basename)
    filename = '{}_esp_id_{}.binaryfile'.format(outbase, idx)
    with open(filename, 'wb') as wrs:
        pickle.dump(results, wrs)
    send_end.send(filename)
    print('Finish pid={} with size {} (from {} to {})'.format(idx, len(log_gams), log_gams[0], log_gams[-1]))

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--length', type=int, default=2000)
    parser.add_argument('--buffer', type=int, default=1000, help='start index to calculate ESP index')
    parser.add_argument('--trials', type=int, default=1, help='Number of trials')

    parser.add_argument('--nqrc', type=int, default=1, help='Number of reservoirs')
    parser.add_argument('--non_diag_const', type=float, default=2.0, help='The nondiag const')
    parser.add_argument('--tau', type=float, default=10.0, help='Tau')

    parser.add_argument('--type_input', type=int, default=0)
    parser.add_argument('--type_op', type=str, default='Z')
    parser.add_argument('--randseed', type=int, default=0)
    
    parser.add_argument('--nproc', type=int, default=100)
    parser.add_argument('--dynamic', type=str, default=DYNAMIC_PHASE_TRANS,\
        help='full_random,half_random,full_const_trans,full_const_coeff,ion_trap')

    parser.add_argument('--interval', type=float, default=0.05, help='interval for logW')
    parser.add_argument('--vmin', type=float, default=-2.0, help='minimum of parameter range for logW')
    parser.add_argument('--vmax', type=float, default=2.1, help='maximum of parameter range for logW')
    
    parser.add_argument('--ginterval', type=float, default=0.05, help='interval for gamma')
    parser.add_argument('--gvmin', type=float, default=-2.0, help='minimum of parameter range for log_gamma')
    parser.add_argument('--gvmax', type=float, default=2.1, help='maximum of parameter range for log_gamma')

    parser.add_argument('--savedir', type=str, default='res_esp_index')
    parser.add_argument('--input_file', type=str, default='../data/sin_input_T_50.txt')
    
    args = parser.parse_args()
    print(args)

    length, nqrc, nproc, dynamic, num_trials = args.length, args.nqrc, args.nproc, args.dynamic, args.trials
    buffer, type_input, non_diag_const, tau = args.buffer, args.type_input, args.non_diag_const, args.tau
    type_op, randseed = args.type_op, args.randseed
    vmin, vmax, interval = args.vmin, args.vmax, args.interval
    gvmin, gvmax, ginterval = args.gvmin, args.gvmax, args.ginterval

    savedir = args.savedir
    if os.path.isfile(savedir) == False and os.path.isdir(savedir) == False:
        os.mkdir(savedir)
    
    vx = list(np.arange(vmin, vmax, interval))
    gvx = list(np.arange(gvmin, gvmax, ginterval))
    nproc = min(len(gvx), nproc)
    lst = np.array_split(gvx, nproc)

    if os.path.isfile(savedir) == False:
        # prepare the data
        data = np.loadtxt(args.input_file)[:length,1]
        input_seq = np.array(data)
        print(input_seq.shape)
        input_seq = np.tile(input_seq, (nqrc, 1))

        jobs, pipels = [], []
        for pid in range(nproc):
            val_ls = lst[pid]
            recv_end, send_end = multiprocessing.Pipe(False)
            p = multiprocessing.Process(target=dum_esp_index_job, args=(savedir, dynamic, input_seq, nqrc, type_input, type_op,\
                non_diag_const, tau, vx, val_ls, pid, buffer, send_end, num_trials, randseed))

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

    fig, axs = plt.subplots(1, 1, figsize=(12, 8), squeeze=False)
    ax = axs[0, 0]
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    ax.tick_params('both', length=8, width=1, which='major', labelsize=10, direction = "out")
    ax.tick_params('both', length=4, width=1, which='minor', direction = "out")
    
    esp_ind_arr = []
    vx, gvx = np.array(vx), np.array(gvx)

    log_Ws   = vx[vx <= 2.0001]
    log_gams = gvx[gvx <= 2.0001]

    print(log_gams)
    for log_W in log_Ws:
        for log_gam in log_gams:
            key_str = 'log_W_{:.3f}_gam_{:.3f}'.format(log_W, log_gam)
            esp_index = z[key_str]
            esp_ind_arr.append(esp_index)
    esp_ind_arr = np.array(esp_ind_arr).reshape(len(log_Ws), len(log_gams))

    
    ax.set_xlabel('(log) Feedback strength')
    ax.set_ylabel('(log) Disorder strength')
    ax.set_yticks(np.linspace(log_Ws[0], log_Ws[-1], 11))
    ax.set_xticks(np.linspace(log_gams[0], log_gams[-1], 11))

    dx, = np.diff([log_gams[0],log_gams[-1]])/(esp_ind_arr.shape[1]-1)
    dy, = -np.diff([log_Ws[0], log_Ws[-1]])/(esp_ind_arr.shape[0]-1)
    extent = [log_gams[0]-dx/2, log_gams[-1]+dx/2, log_Ws[0]+dy/2, log_Ws[-1]-dy/2]
    im1 = ax.imshow(esp_ind_arr, origin='lower', norm=SymLogNorm(linthresh=1e-15), \
        extent=extent, aspect='auto', interpolation='nearest')
    #ax.grid()
    plt.colorbar(im1)

    outbase = filename.replace('.binaryfile', '')
    ax.set_title('ESP index {}'.format(os.path.basename(outbase)), fontsize=8)
    for ftype in ['png']:
        plt.savefig('{}_v1.{}'.format(outbase, ftype), bbox_inches='tight', dpi=600)
    plt.show()
