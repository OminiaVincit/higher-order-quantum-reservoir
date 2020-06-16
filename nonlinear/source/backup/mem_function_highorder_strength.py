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
import highorder_qrc as hqrc
from loginit import get_module_logger
import qrc
import utils

def memory_func(taskname, qparams, nqrc, deep, layer_strength,\
        train_len, val_len, buffer, dlist, ranseed, pid, send_end):
    btime = int(time.time() * 1000.0)
    rsarr = hqrc.memory_function(taskname, qparams, train_len=train_len, val_len=val_len, buffer=buffer, \
        dlist=dlist, nqrc=nqrc, layer_strength=layer_strength, ranseed=ranseed, deep=deep)
    
    # obtain the memory
    rslist = []
    for i in range(rsarr.shape[0]):
        rslist.append('{:f},{:f}'.format(rsarr[i, 0], rsarr[i, 1]))

    etime = int(time.time() * 1000.0)
    now = datetime.datetime.now()
    datestr = now.strftime('{0:%Y-%m-%d-%H-%M-%S}'.format(now))
    print('{} Finished process {} in {} s with nqrc={}, strength={}, V={}, taudelta={}, dmin={}, dmax={}'.format(\
        datestr, pid, etime-btime, nqrc, layer_strength, qparams.virtual_nodes, qparams.tau_delta, dlist[0], dlist[-1]))
    send_end.send('{}'.format(','.join([str(c) for c in rslist])))

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
    parser.add_argument('--strengths', type=str, default='0.0,0.1,0.3,0.5,0.7,0.9')
    parser.add_argument('--nqrc', type=int, default='1')
    parser.add_argument('--taudelta', type=float, default=0.25)
    parser.add_argument('--virtuals', type=int, default=1)

    parser.add_argument('--deep', type=int, default=0)
    parser.add_argument('--taskname', type=str, default='qrc_stm') # Use _stm or _pc
    parser.add_argument('--savedir', type=str, default='res_highfunc_stm2')
    args = parser.parse_args()
    print(args)

    n_units, max_energy, trotter_step, beta =\
        args.units, args.coupling, args.trotter, args.beta
    train_len, val_len, buffer = args.trainlen, args.vallen, args.buffer
    nproc, tau_delta, V = args.nproc, args.taudelta, args.virtuals
    init_rho = args.rho
    minD, maxD, interval, Ntrials = args.mind, args.maxd, args.interval, args.ntrials
    dlist = list(range(minD, maxD + 1, interval))
    nproc = min(nproc, len(dlist))
    nqrc  = args.nqrc
    print('Divided into {} processes'.format(nproc))
    deep = False
    if args.deep > 0:
        deep =True

    taskname, savedir = args.taskname, args.savedir
    if os.path.isfile(savedir) == False and os.path.isdir(savedir) == False:
        os.mkdir(savedir)

    strengths = [float(x) for x in args.strengths.split(',')]
    # Evaluation
    timestamp = int(time.time() * 1000.0)
    now = datetime.datetime.now()
    datestr = now.strftime('{0:%Y-%m-%d-%H-%M-%S}'.format(now))
    
    stmp = '{}_{}_deep_{}_tdt_{}_V_{}_layers_{}_mem_ntrials_{}'.format(\
        taskname, datestr, deep, tau_delta, V, nqrc, Ntrials)
    outbase = os.path.join(savedir, stmp)
    
    rsarr = dict()
    if os.path.isfile(savedir) == False:
        logdir = os.path.join(savedir, 'log')
        if os.path.isdir(logdir) == False:
            os.mkdir(logdir)
        log_filename = '{}.log'.format(stmp)
        log_filename = os.path.join(logdir, stmp)
        logger = get_module_logger(__name__, log_filename)
        logger.info(log_filename)

        for layer_strength in strengths:
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
                    p = multiprocessing.Process(target=memory_func, \
                        args=(taskname, qparams, nqrc, deep, layer_strength, train_len, val_len, buffer, dsmall, n, proc_id, send_end))
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
            rsarr[str(layer_strength)] = local_arr
            logger.debug('layers={},V={},taudelta={},strength={},mem_func={}'.format(nqrc, V, tau_delta, layer_strength,local_arr))
        # save multi files
        np.savez('{}_memfunc.npz'.format(outbase), **rsarr)
        
        # save experiments setting
        with open('{}_setting.txt'.format(outbase), 'w') as sfile:
            sfile.write('train_len={}, val_len={}, buffer={}\n'.format(train_len, val_len, buffer))
            sfile.write('n_units={}\n'.format(n_units))
            sfile.write('max_energy={}\n'.format(max_energy))
            sfile.write('trotter_step={}\n'.format(trotter_step))
            sfile.write('beta={}\n'.format(beta))
            sfile.write('strengths={}\n'.format(' '.join([str(v) for v in strengths])))
            sfile.write('layers={}\n'.format(nqrc))
            sfile.write('V={}\n'.format(V))
            sfile.write('deep={}\n'.format(deep))
            sfile.write('minD={}, maxD={}, interval={}\n'.format(minD, maxD, interval))
            sfile.write('taudelta={}, Ntrials={}\n'.format(tau_delta, Ntrials))

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

    for layer_strength in strengths:
        tarr = rsarr[str(layer_strength)]
        xs, ys, zs = tarr[:, 0], tarr[:, 1], tarr[:, 2]
        #plt.errorbar(xs, ys, yerr=zs, elinewidth=2, linewidth=2, markersize=12, \
        #    label='$\\tau\Delta$={}'.format(tau_delta))
        plt.plot(xs, ys, linewidth=2, markersize=12, \
            label='$\\alpha$={}'.format(layer_strength))
    #plt.xlim([1e-3, 1024])    
    plt.ylim([0, 1.0])
    plt.xlabel('Delay', fontsize=28)
    plt.ylabel('STM', fontsize=28)
    
    plt.legend()
    plt.title(outbase, fontsize=12)
    plt.grid(True, which="both", ls="-", color='0.65')
    #plt.show()
    for ftype in ['png', 'pdf']:
        plt.savefig('{}.{}'.format(outbase, ftype), bbox_inches='tight')
 
