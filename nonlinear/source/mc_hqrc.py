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
from utils import QRCParams

def memory_compute(taskname, qparams, nqrc, layer_strength,\
        train_len, val_len, buffer, dlist, ranseed, pid, send_end):
    btime = int(time.time() * 1000.0)
    rsarr = hqrc.memory_function(taskname, qparams, train_len=train_len, val_len=val_len, buffer=buffer, \
        dlist=dlist, nqrc=nqrc, layer_strength=layer_strength, ranseed=ranseed)
    C = np.sum(rsarr[:, 1])
    etime = int(time.time() * 1000.0)
    now = datetime.datetime.now()
    datestr = now.strftime('{0:%Y-%m-%d-%H-%M-%S}'.format(now))
    print('{} Finished process {} in {} s with J={}, taudelta={}, V={}, layers={}, strength={}, dmin={}, dmax={}, capacity={}'.format(\
        datestr, pid, etime-btime, \
        qparams.max_energy, qparams.tau, qparams.virtual_nodes, nqrc, layer_strength, dlist[0], dlist[-1], C))
    send_end.send('{}'.format(C))

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--units', type=int, default=5)
    parser.add_argument('--init_rho', type=int, default=0)
    parser.add_argument('--beta', type=float, default=1e-14)

    parser.add_argument('--trainlen', type=int, default=3000)
    parser.add_argument('--vallen', type=int, default=1000)
    parser.add_argument('--buffer', type=int, default=1000)
    
    parser.add_argument('--mind', type=int, default=0)
    parser.add_argument('--maxd', type=int, default=10)
    parser.add_argument('--interval', type=int, default=1)

    parser.add_argument('--nproc', type=int, default=50)
    parser.add_argument('--ntrials', type=int, default=1)

    parser.add_argument('--couplings', type=str, default='1.0')
    parser.add_argument('--taudeltas', type=str, default='-4,-3,-2,-1,0,1,2,3,4,5,6,7')
    parser.add_argument('--layers', type=str, default='5')
    parser.add_argument('--strengths', type=str, default='0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9')
    parser.add_argument('--virtuals', type=str, default='1')

    parser.add_argument('--taskname', type=str, default='qrc_stm') # Use _stm or _pc
    parser.add_argument('--savedir', type=str, default='rescapa_high_alpha')
    args = parser.parse_args()
    print(args)

    n_units, beta = args.units, args.beta
    train_len, val_len, buffer = args.trainlen, args.vallen, args.buffer
    nproc, init_rho = args.nproc, args.init_rho
    minD, maxD, interval, Ntrials = args.mind, args.maxd, args.interval, args.ntrials
    dlist = list(range(minD, maxD + 1, interval))
    nproc = min(nproc, len(dlist))
    print('Divided into {} processes'.format(nproc))
    
    taskname, savedir = args.taskname, args.savedir
    if os.path.isdir(savedir) == False:
        os.mkdir(savedir)

    tstr = args.taudeltas.replace('\'','')
    taudeltas = [float(x) for x in tstr.split(',')]
    taudeltas = [2**x for x in taudeltas]
    
    couplings = [float(x) for x in args.couplings.split(',')]
    layers = [int(x) for x in args.layers.split(',')]
    strengths = [float(x) for x in args.strengths.split(',')]
    virtuals = [int(x) for x in args.virtuals.split(',')]

    # Evaluation
    timestamp = int(time.time() * 1000.0)
    now = datetime.datetime.now()
    datestr = now.strftime('{0:%Y-%m-%d-%H-%M-%S}'.format(now))
    outbase = os.path.join(savedir, '{}_{}_J_{}_strength_{}_V_{}_layers_{}_capa_ntrials_{}'.format(\
        taskname, datestr, \
        '_'.join([str(o) for o in couplings]), \
        '_'.join([str(o) for o in strengths]), \
        '_'.join([str(o) for o in virtuals]), \
        '_'.join([str(o) for o in layers]), Ntrials))
    
    log_filename = '{}.log'.format(outbase)
    logger = get_module_logger(__name__, log_filename)
    logger.info(log_filename)

    global_rs = []
    for max_energy in couplings:
        for tau in taudeltas:
            for V in virtuals:
                qparams = QRCParams(n_units=n_units, max_energy=max_energy,\
                    beta=beta, virtual_nodes=V, tau=tau, init_rho=init_rho)
                for layer_strength in strengths:
                    for nqrc in layers:
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
                                p = multiprocessing.Process(target=memory_compute, \
                                    args=(taskname, qparams, nqrc, layer_strength, train_len, val_len, buffer, dsmall, n, proc_id, send_end))
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
                            local_rsarr = [float(x.recv()) for x in pipels]
                            local_sum.append(np.sum(local_rsarr))
                        local_avg, local_std = np.mean(local_sum), np.std(local_sum)
                        global_rs.append([nqrc, tau, max_energy, local_avg, local_std])
                        logger.debug('J={},tau={},V={},alpha={},layers={},capa_avg={},capa_std={}'.format(\
                            max_energy, tau, V, layer_strength, nqrc, local_avg, local_std))
    rsarr = np.array(global_rs)
    np.savetxt('{}_capacity.txt'.format(outbase), rsarr, delimiter=' ')
    
    # save experiments setting
    with open('{}_setting.txt'.format(outbase), 'w') as sfile:
        sfile.write('train_len={}, val_len={}, buffer={}\n'.format(train_len, val_len, buffer))
        sfile.write('beta={}, Ntrials={}\n'.format(beta, Ntrials))
        sfile.write('n_units={}\n'.format(n_units))
        sfile.write('max_energy={}\n'.format(' '.join([str(v) for v in couplings])))
        sfile.write('taudeltas={}\n'.format(' '.join([str(v) for v in taudeltas])))
        sfile.write('layers={}\n'.format(' '.join([str(l) for l in layers])))
        sfile.write('V={}\n'.format(' '.join([str(v) for v in virtuals])))
        sfile.write('minD={}, maxD={}, interval={}\n'.format(minD, maxD, interval))
        sfile.write('layer_strength={}\n'.format(' '.join([str(v) for v in strengths])))