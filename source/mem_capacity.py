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

tdeltas = [2**n for n in range(-5, 11)]

def memory_compute(taskname, qparams, train_len, val_len, buffer, dlist, ranseed, pid, send_end):
    rsarr = qrc.memory_function(taskname, qparams, train_len=train_len, val_len=val_len, buffer=buffer, \
        dlist=dlist, ranseed=ranseed)
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
    parser.add_argument('--nproc', type=int, default=50)

    parser.add_argument('--basename', type=str, default='qrc_stm')
    parser.add_argument('--savedir', type=str, default='rescapacity')
    args = parser.parse_args()
    print(args)

    hidden_unit_count, max_coupling_energy, trotter_step, beta =\
        args.units, args.coupling, args.trotter, args.beta
    train_len, val_len, buffer = args.trainlen, args.vallen, args.buffer
    V, nproc = args.virtuals, args.nproc
    init_rho = args.rho
    minD, maxD, interval, N = args.mind, args.maxd, args.interval, args.ntrials
    dlist = list(range(minD, maxD + 1, interval))
    nproc = min(nproc, len(dlist))
    print('Divided into {} processes'.format(nproc))

    basename = args.basename
    savedir = args.savedir
    ranseed = args.seed
    
    if os.path.isdir(savedir) == False:
        os.mkdir(savedir)
    timestamp = int(time.time() * 1000.0)
    now = datetime.datetime.now()
    datestr = now.strftime('{0:%Y-%m-%d-%H-%M-%S}'.format(now))
    outbase = os.path.join(savedir, '{}_{}_V_{}'.format(basename, datestr, V))

    global_rs = []
    
    for tau_delta in tdeltas:
        qparams = qrc.QRCParams(hidden_unit_count=hidden_unit_count, max_coupling_energy=max_coupling_energy,\
            trotter_step=trotter_step, beta=beta, virtual_nodes=V, tau_delta=tau_delta, init_rho=init_rho)
        local_sum = []
        for n in range(N):
            # Multi-process
            lst = np.array_split(dlist, nproc)
            jobs, pipels = [], []
            for proc_id in range(nproc):
                dsmall = lst[proc_id]
                if dsmall.size == 0:
                    continue
                print('dlist: ', dsmall)
                recv_end, send_end = multiprocessing.Pipe(False)
                p = multiprocessing.Process(target=memory_compute, \
                    args=(basename, qparams, train_len, val_len, buffer, dsmall, n, proc_id, send_end))
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
            local_sum.append(np.sum(rsarr))
        local_avg, local_std = np.mean(local_sum), np.std(local_sum)
        global_rs.append([tau_delta, local_avg, local_std])
        print(tau_delta, local_avg, local_std)
    global_rs = np.array(global_rs)
    np.savetxt('{}_capacity_tau.txt'.format(outbase), global_rs, delimiter=' ')

    # save experiments setting
    with open('{}_setting.txt'.format(outbase), 'w') as sfile:
        sfile.write('train_len={}, val_len={}, buffer={}\n'.format(train_len, val_len, buffer))
        sfile.write('hidden_unit_count={}\n'.format(qparams.hidden_unit_count))
        sfile.write('max_coupling_energy={}\n'.format(qparams.max_coupling_energy))
        sfile.write('trotter_step={}\n'.format(qparams.trotter_step))
        sfile.write('beta={}\n'.format(qparams.beta))
        sfile.write('virtual nodes={}\n'.format(V))
        sfile.write('tau_delta={}\n'.format(' '.join([str(t) for t in tdeltas])))
        sfile.write('minD={}, maxD={}, interval={}\n'.format(minD, maxD, interval))
        sfile.write('Ntrials={}, seed={}\n'.format(N, ranseed))
