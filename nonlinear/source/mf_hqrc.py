#!/usr/bin/env python
"""
    Calculate memory function for higher-order quantum reservoir
    See run_hqrc_mem_func.sh for an example to run the script
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
import hqrc as hqrc
from loginit import get_module_logger
import utils
from utils import *

def memory_func(taskname, qparams, nqrc, alpha, mask_input, combine_input, non_linear, sigma_input, type_input,\
        train_len, val_len, buffer, dlist, ranseed, pid, send_end):
    btime = int(time.time() * 1000.0)
    rsarr = hqrc.memory_function(taskname, qparams, train_len=train_len, val_len=val_len, buffer=buffer, \
        dlist=dlist, nqrc=nqrc, gamma=alpha, ranseed=ranseed, sparsity=1.0, type_input=type_input,\
        sigma_input=sigma_input, mask_input=mask_input, combine_input=combine_input, nonlinear=non_linear)
    
    # obtain the memory
    rslist = []
    for i in range(rsarr.shape[0]):
        rslist.append('{:f},{:f}'.format(rsarr[i, 0], rsarr[i, 1]))

    etime = int(time.time() * 1000.0)
    now = datetime.datetime.now()
    datestr = now.strftime('{0:%Y-%m-%d-%H-%M-%S}'.format(now))
    print('{} Finished process {} in {} s with nqrc={}, alpha={}, V={}, taudelta={}, dmin={}, dmax={}'.format(\
        datestr, pid, etime-btime, nqrc, alpha, qparams.virtual_nodes, qparams.tau, dlist[0], dlist[-1]))
    send_end.send('{}'.format(','.join([str(c) for c in rslist])))

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--units', type=int, default=5, help='Number of the hidden units')
    parser.add_argument('--coupling', type=float, default=1.0, help='Maximum coupling energy')
    parser.add_argument('--rho', type=int, default=0, help='Flag for initializing the density matrix')
    parser.add_argument('--beta', type=float, default=1e-14, help='reg term')
    parser.add_argument('--solver', type=str, default=LINEAR_PINV, \
        help='regression solver by linear_pinv,ridge_pinv,auto,svd,cholesky,lsqr,sparse_cg,sag')
    parser.add_argument('--dynamic', type=str, default=DYNAMIC_FULL_RANDOM,\
        help='full_random,half_random,full_const_trans,full_const_coeff,ion_trap')

    parser.add_argument('--trainlen', type=int, default=3000)
    parser.add_argument('--vallen', type=int, default=1000)
    parser.add_argument('--buffer', type=int, default=1000)
    
    parser.add_argument('--mind', type=int, default=0)
    parser.add_argument('--maxd', type=int, default=10)
    parser.add_argument('--interval', type=int, default=1)

    parser.add_argument('--nproc', type=int, default=50)
    parser.add_argument('--ntrials', type=int, default=1)
    parser.add_argument('--non_linear', type=int, default=0)
    parser.add_argument('--sigma_input', type=float, default=1.0)
    parser.add_argument('--mask_input', type=int, default=0)
    parser.add_argument('--combine_input', type=int, default=1)
    parser.add_argument('--type_input', type=int, default=0)

    parser.add_argument('--taudeltas', type=str, default='-4,-3,-2,-1,0,1,2,3,4,5,6,7', \
        help='Interval between the inputs')
    parser.add_argument('--nqrc', type=int, default=1, help='Number of reservoirs')
    parser.add_argument('--strength', type=float, default=0.0, help='The connection strength')
    parser.add_argument('--virtuals', type=int, default=1, help='Number of virtual nodes')

    parser.add_argument('--taskname', type=str, default='qrc_stm') # Use _stm or _pc
    parser.add_argument('--savedir', type=str, default='rescapa_highfunc_stm')
    args = parser.parse_args()
    print(args)

    n_units, max_energy, beta = args.units, args.coupling, args.beta
    train_len, val_len, buffer = args.trainlen, args.vallen, args.buffer
    nproc, alpha, V = args.nproc, args.strength, args.virtuals
    init_rho, dynamic = args.rho, args.dynamic
    minD, maxD, interval, Ntrials = args.mind, args.maxd, args.interval, args.ntrials
    non_linear, sigma_input, type_input = args.non_linear, args.sigma_input, args.type_input
    mask_input, combine_input = args.mask_input, args.combine_input
    

    dlist = list(range(minD, maxD + 1, interval))
    nproc = min(nproc, len(dlist))
    nqrc  = args.nqrc
    print('Divided into {} processes'.format(nproc))

    taskname, savedir, solver = args.taskname, args.savedir, args.solver
    if os.path.isfile(savedir) == False and os.path.isdir(savedir) == False:
        os.mkdir(savedir)

    tstr = args.taudeltas.replace('\'','')
    taudeltas = [float(x) for x in tstr.split(',')]
    taudeltas = [2**x for x in taudeltas]

    # Evaluation
    timestamp = int(time.time() * 1000.0)
    now = datetime.datetime.now()
    datestr = now.strftime('{0:%Y-%m-%d-%H-%M-%S}'.format(now))
    
    stmp = '{}_{}_strength_{}_V_{}_layers_{}_mask_{}_cb_{}_sm_{}_sg_{}_tp_{}_mem_ntrials_{}'.format(\
        taskname, datestr, alpha, V, nqrc, mask_input, \
        combine_input, non_linear, sigma_input, type_input, Ntrials)
    outbase = os.path.join(savedir, stmp)
    
    rsarr = dict()
    if os.path.isfile(savedir) == False:
        logdir = os.path.join(savedir, 'log')
        if os.path.isdir(logdir) == False:
            os.mkdir(logdir)
        log_filename = os.path.join(logdir, '{}.log'.format(stmp))
        logger = get_module_logger(__name__, log_filename)
        logger.info(log_filename)

        for tau in taudeltas:
            qparams = QRCParams(n_units=n_units-1, n_envs=1, max_energy=max_energy, \
                beta=beta, virtual_nodes=V, tau=tau, init_rho=init_rho, solver=solver,dynamic=dynamic)
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
                        args=(taskname, qparams, nqrc, alpha, mask_input, combine_input, non_linear, sigma_input, type_input,\
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
            rsarr[str(tau)] = local_arr
            logger.debug('layers={},V={},taudelta={},mem_func={}'.format(nqrc, V, tau, local_arr))
        # save multi files
        np.savez('{}_memfunc.npz'.format(outbase), **rsarr)
        
        # save experiments setting
        with open('{}_setting.txt'.format(outbase), 'w') as sfile:
            sfile.write('train_len={}, val_len={}, buffer={}\n'.format(train_len, val_len, buffer))
            sfile.write('n_units={}\n'.format(n_units))
            sfile.write('max_energy={}\n'.format(max_energy))
            sfile.write('beta={}\n'.format(beta))
            sfile.write('taudeltas={}\n'.format(' '.join([str(v) for v in taudeltas])))
            sfile.write('layers={}\n'.format(nqrc))
            sfile.write('V={}\n'.format(V))
            sfile.write('minD={}, maxD={}, interval={}\n'.format(minD, maxD, interval))
            sfile.write('alpha={}, Ntrials={}\n'.format(alpha, Ntrials))

    else:
        # Read the result
        rsarr = np.load(savedir)
        outbase = savedir.replace('.npz', '')

    # plot the result
    cmap = plt.get_cmap("viridis")
    plt.figure(figsize=(16,8))
    #plt.style.use('seaborn-colorblind')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams['font.size']=20

    for tau in taudeltas:
        tarr = rsarr[str(tau)]
        xs, ys, zs = tarr[:, 0], tarr[:, 1], tarr[:, 2]
        #plt.errorbar(xs, ys, yerr=zs, elinewidth=2, linewidth=2, markersize=12, \
        #    label='$\\tau\Delta$={}'.format(tau))
        plt.plot(xs, ys, linewidth=2, markersize=12, \
            label='$\\tau\Delta$={}'.format(tau))
    #plt.xlim([1e-3, 1024])    
    plt.ylim([0, 1.0])
    plt.xlabel('Delay', fontsize=28)
    plt.ylabel('STM', fontsize=28)
    
    plt.legend()
    plt.title(outbase, fontsize=12)
    plt.grid(True, which="both", ls="-", color='0.65')
    for ftype in ['png']:
        plt.savefig('{}.{}'.format(outbase, ftype), bbox_inches='tight')
    #plt.show()
    
