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

def dum_feedback_job(savedir, dynamic, input_seq, nqrc, type_input, type_op,\
    non_diag_const, tau, log_Ws, log_gams, idx, buffer, send_end, randseed):
    """
    Dump raw data of states
    """
    print('Start pid={} with size {} (from {} to {})'.format(idx, len(log_gams), log_gams[0], log_gams[-1]))
    results = dict()

    basename = 'heat_{}_nqr_{}_V_{}_tau_{}_nondiag_{}_op_{}_tp_{}_rsd_{}'.format(\
        dynamic, nqrc, V, tau, non_diag_const, type_op, type_input, randseed)
    os.makedirs(savedir, exist_ok=True)
    
    for log_gam in log_gams:
        gamma = 10**log_gam
        for log_W in log_Ws:
            non_diag_var = 10**log_W
            qparams = QRCParams(n_units=UNITS-1, n_envs=1, max_energy=1.0, \
                non_diag_const=non_diag_const, non_diag_var=non_diag_var,
                beta=BETA, virtual_nodes=V, tau=tau, init_rho=INIT_RHO, solver=LINEAR_PINV, dynamic=dynamic)
            model = hqrc.HQRC(nqrc=nqrc, gamma=gamma, sparsity=1.0, sigma_input=1.0, nonlinear=0, type_input=type_input, type_op=type_op)
            
            _, feed_state_list = model.init_forward(qparams, input_seq, init_rs = True, ranseed = randseed)
            selected_feed = np.array(feed_state_list[buffer:, 0])
            
            f_min  = np.min(selected_feed)
            f_max  = np.max(selected_feed)
            f_mean = np.mean(selected_feed)
            f_std  = np.std(selected_feed)

            key_str = 'log_W_{:.3f}_gam_{:.3f}'.format(log_W, log_gam)
            results[key_str] = np.array([f_min, f_max, f_mean, f_std])

    outbase = os.path.join(savedir, basename)
    filename = '{}_feed_id_{}.binaryfile'.format(outbase, idx)
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
            p = multiprocessing.Process(target=dum_feedback_job, args=(savedir, dynamic, input_seq, nqrc, type_input, type_op,\
                non_diag_const, tau, vx, val_ls, pid, buffer, send_end, randseed))

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
    plt.rcParams["font.size"] = 16 # 全体のフォントサイズが変更されます
    plt.rcParams['xtick.labelsize'] = 16 # 軸だけ変更されます
    plt.rcParams['ytick.labelsize'] = 16 # 軸だけ変更されます

    fig, axs = plt.subplots(2, 2, figsize=(24, 16), squeeze=False)
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    
    
    v_min_arr, v_max_arr, v_mean_arr, v_std_arr = [], [], [], []
    vx, gvx = np.array(vx), np.array(gvx)

    log_Ws   = vx[vx <= 2.0001]
    log_gams = gvx[gvx <= 2.0001]

    print(log_gams)
    for log_W in log_Ws:
        for log_gam in log_gams:
            key_str = 'log_W_{:.3f}_gam_{:.3f}'.format(log_W, log_gam)
            val_arr = z[key_str] * (10**log_gam)
            v_min_arr.append(val_arr[0])
            v_max_arr.append(val_arr[1])
            v_mean_arr.append(val_arr[2])
            v_std_arr.append(val_arr[3])

    v_min_arr = np.array(v_min_arr).reshape(len(log_Ws), len(log_gams))
    v_max_arr = np.array(v_max_arr).reshape(len(log_Ws), len(log_gams))
    v_mean_arr = np.array(v_mean_arr).reshape(len(log_Ws), len(log_gams))
    v_std_arr = np.array(v_std_arr).reshape(len(log_Ws), len(log_gams))

    for ax in axs.ravel():
        ax.tick_params('both', length=8, width=1, which='major', labelsize=14, direction = "out")
        ax.tick_params('both', length=4, width=1, which='minor', direction = "out")
        ax.set_yticks(np.linspace(log_Ws[0], log_Ws[-1], 11))
        ax.set_xticks(np.linspace(log_gams[0], log_gams[-1], 11))
        
    for ax in [axs[0, 0], axs[1, 0]]:
        ax.set_ylabel('(log) Disorder strength')
    for ax in [axs[1, 0], axs[1, 1]]:
        ax.set_xlabel('(log) Feedback strength')
        

    dx, = np.diff([log_gams[0],log_gams[-1]])/(v_min_arr.shape[1]-1)
    dy, = -np.diff([log_Ws[0], log_Ws[-1]])/(v_min_arr.shape[0]-1)
    extent = [log_gams[0]-dx/2, log_gams[-1]+dx/2, log_Ws[0]+dy/2, log_Ws[-1]-dy/2]

    im1 = axs[0, 0].imshow(v_min_arr, origin='lower', extent=extent, norm=SymLogNorm(linthresh=1e-15), aspect='auto', interpolation='nearest', cmap="viridis")
    axs[0, 0].set_title('Min feed')

    im2 = axs[0, 1].imshow(v_max_arr, origin='lower', extent=extent, norm=SymLogNorm(linthresh=1e-15), aspect='auto', interpolation='nearest', cmap="cividis")
    axs[0, 1].set_title('Max feed')

    im3 = axs[1, 0].imshow(v_mean_arr, origin='lower', extent=extent, norm=SymLogNorm(linthresh=1e-15), aspect='auto', interpolation='nearest', cmap="inferno")
    axs[1, 0].set_title('Avg. feed')

    im4 = axs[1, 1].imshow(v_std_arr, origin='lower', extent=extent, norm=SymLogNorm(linthresh=1e-15), aspect='auto', interpolation='nearest', cmap="Spectral")
    axs[1, 1].set_title('Std. feed')

    #ax.grid()
    fig.colorbar(im1, ax=axs[0, 0])
    fig.colorbar(im2, ax=axs[0, 1])
    fig.colorbar(im3, ax=axs[1, 0])
    fig.colorbar(im4, ax=axs[1, 1])

    outbase = filename.replace('.binaryfile', '')
    plt.tight_layout()

    for ftype in ['png']:
        plt.savefig('{}_v1.{}'.format(outbase, ftype), bbox_inches='tight', dpi=600)
    plt.show()
