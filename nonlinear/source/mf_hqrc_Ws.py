#!/usr/bin/env python
"""
    Calculate memory function for higher-order quantum reservoir
    following by the variation of parameters W
    See run_hqrc_mem_func.sh for an example to run the script
    
    Version: 2022-06-06
    - Note: new encoding method (feedback to phase)
"""

import sys
import numpy as np
import os
import argparse
import multiprocessing
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
import time
import datetime
import hqrc as hqrc
from loginit import get_module_logger
import utils
from utils import *

def memory_func(taskname, qparams, nqrc, gamma, mask_input, combine_input, non_linear, sigma_input, type_input, type_op,\
        train_len, val_len, buffer, dlist, ranseed, pid, send_end):
    btime = int(time.time() * 1000.0)
    rsarr = hqrc.memory_function(taskname, qparams, train_len=train_len, val_len=val_len, buffer=buffer, \
        dlist=dlist, nqrc=nqrc, gamma=gamma, ranseed=ranseed, sparsity=1.0, type_input=type_input, type_op=type_op,\
        sigma_input=sigma_input, mask_input=mask_input, combine_input=combine_input, nonlinear=non_linear)
    
    # obtain the memory
    rslist = []
    for i in range(rsarr.shape[0]):
        rslist.append('{:f},{:f}'.format(rsarr[i, 0], rsarr[i, 1]))

    etime = int(time.time() * 1000.0)
    now = datetime.datetime.now()
    datestr = now.strftime('{0:%Y-%m-%d-%H-%M-%S}'.format(now))
    print('{} Finished process {} in {} s with non_diag_var={}, nqrc={}, gamma={}, V={}, taudelta={}, dmin={}, dmax={}'.format(\
        datestr, pid, etime-btime, qparams.non_diag_var, nqrc, gamma, qparams.virtual_nodes, qparams.tau, dlist[0], dlist[-1]))
    send_end.send('{}'.format(','.join([str(c) for c in rslist])))

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--units', type=int, default=6, help='Number of the hidden units')
    parser.add_argument('--coupling', type=float, default=1.0, help='Maximum coupling energy')
    parser.add_argument('--rho', type=int, default=0, help='Flag for initializing the density matrix')
    parser.add_argument('--beta', type=float, default=1e-14, help='reg term')
    parser.add_argument('--solver', type=str, default=LINEAR_PINV, \
        help='regression solver by linear_pinv,ridge_pinv,auto,svd,cholesky,lsqr,sparse_cg,sag')
    parser.add_argument('--dynamic', type=str, default=DYNAMIC_PHASE_TRANS,\
        help='full_random,half_random,full_const_trans,full_const_coeff,ion_trap')

    parser.add_argument('--trainlen', type=int, default=3000)
    parser.add_argument('--vallen', type=int, default=1000)
    parser.add_argument('--buffer', type=int, default=1000)
    
    parser.add_argument('--mind', type=int, default=0)
    parser.add_argument('--maxd', type=int, default=10)
    parser.add_argument('--interval', type=int, default=1)

    parser.add_argument('--nproc', type=int, default=101)
    parser.add_argument('--ntrials', type=int, default=1)
    parser.add_argument('--non_linear', type=int, default=0)
    parser.add_argument('--sigma_input', type=float, default=1.0)
    parser.add_argument('--mask_input', type=int, default=0)
    parser.add_argument('--combine_input', type=int, default=1)
    parser.add_argument('--type_input', type=int, default=5)
    parser.add_argument('--type_op', type=str, default='X')

    parser.add_argument('--non_diag_const', type=float, default=2.0, help='The nondiag const')
    parser.add_argument('--tau', type=float, default=10.0, help='Tau')
    parser.add_argument('--strength', type=float, default=0.0, help='Feedback strength')

    parser.add_argument('--nqrc', type=int, default=5, help='Number of reservoirs')
    parser.add_argument('--virtuals', type=int, default=1, help='Number of virtual nodes')
    parser.add_argument('--log_Ws', type=str, default='-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0', \
        help='List of disorder strengths in log')

    parser.add_argument('--task', type=str, default='qrc_stm') # Use _stm or _pc or input from file
    parser.add_argument('--savedir', type=str, default='rescapa_highfunc_stm')
    
    args = parser.parse_args()
    print(args)

    n_units, max_energy, beta = args.units, args.coupling, args.beta
    train_len, val_len, buffer = args.trainlen, args.vallen, args.buffer
    nproc, V = args.nproc, args.virtuals
    init_rho, dynamic = args.rho, args.dynamic
    minD, maxD, interval, Ntrials = args.mind, args.maxd, args.interval, args.ntrials
    non_linear, sigma_input, type_input, type_op = args.non_linear, args.sigma_input, args.type_input, args.type_op
    mask_input, combine_input = args.mask_input, args.combine_input
    non_diag_const, tau, gamma = args.non_diag_const, args.tau, args.strength

    dlist = list(range(minD, maxD + 1, interval))
    nproc = min(nproc, len(dlist))
    nqrc  = args.nqrc
    print('Divided into {} processes'.format(nproc))

    task, savedir, solver = args.task, args.savedir, args.solver
    taskname = os.path.basename(task)
    if os.path.isfile(savedir) == False and os.path.isdir(savedir) == False:
        os.mkdir(savedir)

    tstr = args.log_Ws.replace('\'','')
    log_Ws = [float(x) for x in tstr.split(',')]
    
    # Evaluation
    timestamp = int(time.time() * 1000.0)
    now = datetime.datetime.now()
    datestr = now.strftime('{0:%Y-%m-%d-%H-%M-%S}'.format(now))
    
    stmp = '{}_gamma_{}_V_{}_nqr_{}_mask_{}_cb_{}_sm_{}_sg_{}_op_{}_tp_{}_mem_{}_{}_{}_ntrials_{}'.format(\
        taskname, gamma, V, nqrc, mask_input, \
        combine_input, non_linear, sigma_input, type_op, type_input, train_len, val_len, buffer, Ntrials)
    outbase = os.path.join(savedir, stmp)

    rsarr = dict()
    if os.path.isfile(savedir) == False:
        logdir = os.path.join(savedir, 'log')
        if os.path.isdir(logdir) == False:
            os.mkdir(logdir)
        log_filename = os.path.join(logdir, '{}.log'.format(stmp))
        logger = get_module_logger(__name__, log_filename)
        logger.info(log_filename)

        for log_W in log_Ws:
            non_diag_var = 10**log_W
            qparams = QRCParams(n_units=n_units-1, n_envs=1, max_energy=max_energy, \
                non_diag_const=non_diag_const, non_diag_var=non_diag_var,\
                beta=beta, virtual_nodes=V, tau=tau, init_rho=init_rho, solver=solver, dynamic=dynamic)
            local_sum = []
            for n in range(Ntrials):
                # Multi process
                lst = np.array_split(dlist, nproc)
                jobs, pipels = [], []
                for proc_id in range(nproc):
                    dsmall = lst[proc_id]
                    if dsmall.size == 0:
                        continue
                    print('dlist: ', dsmall)
                    recv_end, send_end = multiprocessing.Pipe(False)
                    p = multiprocessing.Process(target=memory_func, \
                        args=(task, qparams, nqrc, gamma, mask_input, combine_input, non_linear, sigma_input, type_input, type_op,\
                            train_len, val_len, buffer, dsmall, n, proc_id, send_end))
                    jobs.append(p)
                    pipels.append(recv_end)
        
                # Start the process
                for p in jobs:
                    p.start()
        
                # Ensure all processes have finiished execution
                for p in jobs:
                    p.join()

                # Sleep 5s
                time.sleep(5)

                # Get the result
                local_rsarr = ','.join([x.recv() for x in pipels])
                local_rsarr = np.fromstring(local_rsarr, sep=',')
                local_rsarr = local_rsarr.reshape(-1, 2)
                local_sum.append(local_rsarr)
            local_sum = np.array(local_sum)
            local_avg, local_std = np.mean(local_sum, axis=0), np.std(local_sum, axis=0)
            local_arr = np.hstack([local_avg, local_std[:,1].reshape(-1,1)])
            print('local_arr', local_arr.shape)
            rsarr['{:.6f}'.format(log_W)] = local_arr
            #logger.debug('nqr={},V={},log_W={:.6f},mem_func={}'.format(nqrc, V, log_W, local_arr))
            np.savetxt('{}_logW_{:.6f}_memfunc.txt'.format(outbase, log_W), local_arr)
        # save multi files
        np.savez('{}_memfunc.npz'.format(outbase), **rsarr)
        
        # save experiments setting
        with open('{}_setting.txt'.format(outbase), 'w') as sfile:
            sfile.write('train_len={}, val_len={}, buffer={}\n'.format(train_len, val_len, buffer))
            sfile.write('n_units={}\n'.format(n_units))
            sfile.write('max_energy={}\n'.format(max_energy))
            sfile.write('beta={}\n'.format(beta))
            sfile.write('log_Ws={}\n'.format(' '.join([str(v) for v in log_Ws])))
            sfile.write('nqr={}\n'.format(nqrc))
            sfile.write('V={}\n'.format(V))
            sfile.write('minD={}, maxD={}, interval={}\n'.format(minD, maxD, interval))
            sfile.write('gamma={}, Ntrials={}\n'.format(gamma, Ntrials))
    else:
        # Read the result
        rsarr = np.load(savedir)
        outbase = savedir.replace('.npz', '')

    # plot the result
    cmap = plt.get_cmap("viridis")
    fig, axs = plt.subplots(1, 1, figsize=(16, 8), squeeze=False)
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams['font.size']=20

    ax = axs[0, 0]
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params('both', length=8, width=1, which='major', labelsize=20, direction = "out")
    ax.tick_params('both', length=4, width=1, which='minor', direction = "out")

    for log_W in log_Ws:
        tarr = rsarr['{:.6f}'.format(log_W)]
        xs, ys, zs = tarr[:, 0], tarr[:, 1], tarr[:, 2]
        MC = np.sum(ys)
        #plt.errorbar(xs, ys, yerr=zs, elinewidth=2, linewidth=2, markersize=12, \
        #    label='$\\tau\Delta$={}'.format(tau)) #markerfacecolor='None',
        ax.plot(xs, ys, linewidth=2, markersize=12, marker='s',  \
            label='$W$={}, MC={:.2f}'.format(10**log_W, MC), alpha=0.7)
    
    ax.set_xlim([0, 50])    
    ax.set_ylim([0, 1.0])
    ax.set_xlabel('Delay', fontsize=28)
    ax.set_ylabel(taskname, fontsize=28)
    ax.set_title(outbase, fontsize=12)
    ax.grid(True, which="both", ls="-", color='0.65')
    ax.legend()

    for ftype in ['png', 'svg']:
        plt.savefig('{}.{}'.format(outbase, ftype), bbox_inches='tight')
    #plt.show()
    
