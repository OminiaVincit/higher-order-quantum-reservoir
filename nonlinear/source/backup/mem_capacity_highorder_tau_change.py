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

# virtuals = [5*n for n in range(1, 6)]
# virtuals.insert(0, 1)

# layers = [n for n in range(1, 6)]
# strengths = [0.0 0.1 0.3 0.5 0.7 0.9]

def memory_compute(taskname, qparams, nqrc, layer_strength,\
        train_len, val_len, buffer, dlist, ranseed, pid, send_end):
    btime = int(time.time() * 1000.0)
    rsarr = hqrc.memory_function(taskname, qparams, train_len=train_len, val_len=val_len, buffer=buffer, \
        dlist=dlist, nqrc=nqrc, layer_strength=layer_strength, ranseed=ranseed)
    C = np.sum(rsarr[:, 1])
    etime = int(time.time() * 1000.0)
    now = datetime.datetime.now()
    datestr = now.strftime('{0:%Y-%m-%d-%H-%M-%S}'.format(now))
    print('{} Finished process {} in {} s with nqrc={}, strength={}, V={}, taudelta={}, dmin={}, dmax={}, capacity={}'.format(\
        datestr, pid, etime-btime, nqrc, layer_strength, qparams.virtual_nodes, qparams.tau_delta, dlist[0], dlist[-1], C))
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

    parser.add_argument('--nproc', type=int, default=50)
    parser.add_argument('--ntrials', type=int, default=1)
    parser.add_argument('--taudeltas', type=str, default='-4,-3,-2,-1,0,1,2,3,4,5,6,7')
    parser.add_argument('--layers', type=str, default='5')
    parser.add_argument('--strength', type=float, default=0.0)
    parser.add_argument('--virtuals', type=int, default=1)

    parser.add_argument('--taskname', type=str, default='qrc_stm') # Use _stm or _pc
    parser.add_argument('--savedir', type=str, default='rescapa_high_stm')
    parser.add_argument('--plot', type=int, default=0)
    args = parser.parse_args()
    print(args)

    n_units, max_energy, trotter_step, beta =\
        args.units, args.coupling, args.trotter, args.beta
    train_len, val_len, buffer = args.trainlen, args.vallen, args.buffer
    nproc, layer_strength, V = args.nproc, args.strength, args.virtuals
    init_rho = args.rho
    minD, maxD, interval, Ntrials = args.mind, args.maxd, args.interval, args.ntrials
    dlist = list(range(minD, maxD + 1, interval))
    nproc = min(nproc, len(dlist))
    print('Divided into {} processes'.format(nproc))
    
    taskname, savedir = args.taskname, args.savedir
    if os.path.isfile(savedir) == False and os.path.isdir(savedir) == False:
        os.mkdir(savedir)

    taudeltas = [float(x) for x in args.taudeltas.split(',')]
    taudeltas = [2**x for x in taudeltas]
    layers = [int(x) for x in args.layers.split(',')]
    
    # Evaluation
    timestamp = int(time.time() * 1000.0)
    now = datetime.datetime.now()
    datestr = now.strftime('{0:%Y-%m-%d-%H-%M-%S}'.format(now))
    outbase = os.path.join(savedir, '{}_{}_J_{}_strength_{}_V_{}_layers_{}_capacity_ntrials_{}'.format(\
        taskname, datestr, max_energy, layer_strength, V, '_'.join([str(o) for o in layers]), Ntrials))
    
    if os.path.isfile(savedir) == False:
        log_filename = '{}.log'.format(outbase)
        logger = get_module_logger(__name__, log_filename)
        logger.info(log_filename)

        global_rs = []
        for nqrc in layers:
            for tau_delta in taudeltas:
                qparams = qrc.QRCParams(n_units=n_units, max_energy=max_energy,\
            trotter_step=trotter_step, beta=beta, virtual_nodes=V, tau_delta=tau_delta, init_rho=init_rho)
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
                global_rs.append([nqrc, tau_delta, max_energy, local_avg, local_std])
                logger.debug('layers={},taudelta={},J={},capa_avg={},capa_std={}'.format(nqrc, tau_delta, max_energy, local_avg, local_std))
        rsarr = np.array(global_rs)
        np.savetxt('{}_capacity.txt'.format(outbase), rsarr, delimiter=' ')
        
        # save experiments setting
        with open('{}_setting.txt'.format(outbase), 'w') as sfile:
            sfile.write('train_len={}, val_len={}, buffer={}\n'.format(train_len, val_len, buffer))
            sfile.write('n_units={}\n'.format(n_units))
            sfile.write('max_energy={}\n'.format(max_energy))
            sfile.write('trotter_step={}\n'.format(trotter_step))
            sfile.write('beta={}\n'.format(beta))
            sfile.write('taudeltas={}\n'.format(' '.join([str(v) for v in taudeltas])))
            sfile.write('layers={}\n'.format(' '.join([str(l) for l in layers])))
            sfile.write('V={}\n'.format(V))
            sfile.write('minD={}, maxD={}, interval={}\n'.format(minD, maxD, interval))
            sfile.write('layer_strength={}, Ntrials={}\n'.format(layer_strength, Ntrials))

    else:
        # Read the result
        rsarr = np.loadtxt(savedir)
        outbase = savedir.replace('.txt', '')

    print(rsarr)
    print(rsarr.shape)

    # plot the result
    if args.plot > 0:
        xs = taudeltas
        avg_capa, std_capa = rsarr[:, -2], rsarr[:, -1]

        cmap = plt.get_cmap("viridis")
        plt.figure(figsize=(16,8))
        #plt.style.use('seaborn-colorblind')
        plt.rc('font', family='serif')
        plt.rc('mathtext', fontset='cm')
        plt.rcParams['font.size']=20

        for nqrc in layers:
            ids = (rsarr[:, 0] == nqrc)
            plt.errorbar(xs, avg_capa[ids], yerr=std_capa[ids], elinewidth=2, linewidth=2, markersize=12, \
                label='Layers={}'.format(nqrc))
        #plt.xlim([1e-3, 1024])    
        plt.ylim([0, 80])
        plt.xlabel('$\\tau\Delta$', fontsize=28)
        plt.ylabel('Capacity', fontsize=28)
        plt.xscale('log', basex=2)

        plt.legend()
        plt.title(outbase, fontsize=12)
        plt.grid(True, which="both", ls="-", color='0.65')
        #plt.show()
        for ftype in ['png']:
            plt.savefig('{}_capacity.{}'.format(outbase, ftype), bbox_inches='tight')
 
