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

tdeltas = [2**n for n in range(11)]
tdeltas.insert(0, 0.5)

def memory_compute(taskname, qparams, train_len, val_len, buffer, dlist, ranseed, pid, send_end):
    rsarr = qrc.memory_function(taskname, qparams, train_len=train_len, val_len=val_len, buffer=buffer, \
        dlist=dlist, ranseed=ranseed)
    print(rsarr)
    C = np.sum(rsarr[:, 1])
    print('Finished process {} with dmax={}, capacity={}'.format(pid, dlist[-1], C))
    send_end.send('{}'.format(C))

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--units', type=int, default=5)
    parser.add_argument('--coupling', type=float, default=1.0)
    parser.add_argument('--trotter', type=int, default=10)
    parser.add_argument('--rho', type=int, default=0)
    parser.add_argument('--beta', type=float, default=1e-14)

    parser.add_argument('--trainlen', type=int, default=3000)
    parser.add_argument('--vallen', type=int, default=1000)
    parser.add_argument('--buffer', type=int, default=1000)
    
    parser.add_argument('--mind', type=int, default=0)
    parser.add_argument('--maxd', type=int, default=10)
    parser.add_argument('--interval', type=int, default=1)
    parser.add_argument('--seed', type=int, default=-1)
    
    parser.add_argument('--ntrials', type=int, default=1)
    parser.add_argument('--virtuals', type=int, default=10)

    parser.add_argument('--basename', type=str, default='qrc_stm')
    parser.add_argument('--savedir', type=str, default='resmem')
    args = parser.parse_args()
    print(args)

    hidden_unit_count, max_coupling_energy, trotter_step, beta =\
        args.units, args.coupling, args.trotter, args.beta
    train_len, val_len, buffer = args.trainlen, args.vallen, args.buffer
    V = args.virtuals
    init_rho = args.rho
    minD, maxD, interval, N = args.mind, args.maxd, args.interval, args.ntrials
    dlist = list(range(minD, maxD + 1, interval))
    basename = args.basename
    savedir = args.savedir
    ranseed = args.seed

    if os.path.isdir(savedir) == False:
        os.mkdir(savedir)
    timestamp = int(time.time() * 1000.0)
    now = datetime.datetime.now()
    datestr = now.strftime('{0:%Y-%m-%d-%H-%M-%S}'.format(now))

    avg_rs, std_rs = [], []
    
    for tau_delta in tdeltas:
        qparams = qrc.QRCParams(hidden_unit_count=hidden_unit_count, max_coupling_energy=max_coupling_energy,\
            trotter_step=trotter_step, beta=beta, virtual_nodes=V, tau_delta=tau_delta, init_rho=init_rho)

        # Multi-process
        jobs, pipels = [], []
        for proc_id in range(N):
            recv_end, send_end = multiprocessing.Pipe(False)
            p = multiprocessing.Process(target=memory_compute, \
                    args=(basename, qparams, train_len, val_len, buffer, dlist, proc_id, proc_id, send_end))
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
        rsarr = [float(x.recv()) for x in pipels]
        local_avg, local_std = np.mean(rsarr), np.std(rsarr)
        avg_rs.append(local_avg) 
        std_rs.append(local_std)

        print(tau_delta, local_avg, local_std)
