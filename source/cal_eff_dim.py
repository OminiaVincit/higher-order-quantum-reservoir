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

def eff_job(qparams, P, T, input_seq_ls, Ntrials, send_end):
    esp_ls = []
    print('Start process taudelta={}, virtual={}, Jdelta={}'.format(qparams.tau_delta, qparams.virtual_nodes, qparams.max_coupling_energy))
    for n in range(Ntrials):
         esp_val = qrc.effective_dim(qparams, P, T, input_seq_ls)
         esp_ls.append(esp_val)

    mean_esp = np.mean(esp_ls)

    rstr = '{} {} {} {}'.format(\
        qparams.tau_delta, qparams.virtual_nodes, qparams.max_coupling_energy, mean_esp)
    print('Finish process {}'.format(rstr))
    send_end.send(rstr)

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--units', type=int, default=5)
    parser.add_argument('--coupling', type=float, default=1.0)
    parser.add_argument('--trotter', type=int, default=10)
    parser.add_argument('--beta', type=float, default=1e-14)

    parser.add_argument('--vallen', type=int, default=500)
    parser.add_argument('--buffer', type=int, default=500)
    parser.add_argument('--pindex', type=int, default=10)
    parser.add_argument('--ntrials', type=int, default=1)
    parser.add_argument('--nums', type=int, default=1)
    
    parser.add_argument('--resolution', type=int, default=64)

    parser.add_argument('--orders', type=str, default='10')
    parser.add_argument('--basename', type=str, default='qrc_effective')
    parser.add_argument('--savedir', type=str, default='reseff')
    args = parser.parse_args()
    print(args)

    hidden_unit_count, max_coupling_energy, trotter_step, beta =\
        args.units, args.coupling, args.trotter, args.beta
    val_len, pindex, buffer = args.vallen, args.pindex, args.buffer
    Ntrials = args.ntrials
    nums = args.nums

    basename, savedir = args.basename, args.savedir
    if os.path.isfile(savedir) == False and os.path.isdir(savedir) == False:
        os.mkdir(savedir)

    tdeltas = [2**n for n in range(nums)]
    tdeltas.insert(0, 0.5)

    virtuals = [5*n for n in range(1, nums+1)]
    virtuals.insert(0, 1)

    orders = [int(x) for x in args.orders.split(',')]
    data, target = gen.make_data_for_narma(buffer + val_len, orders=orders)

    input_seq_ls = np.array(  [ data[: buffer + val_len] ] )
    
    # Evaluation
    timestamp = int(time.time() * 1000.0)
    now = datetime.datetime.now()
    datestr = now.strftime('{0:%Y-%m-%d-%H-%M-%S}'.format(now))
    outbase = os.path.join(savedir, '{}_{}_order_{}'.format(basename, datestr, '_'.join([str(o) for o in orders])))

    if True:
        if os.path.isfile(savedir) == False:
            jobs, pipels = [], []

            for tdelta in tdeltas:
                for V in virtuals:
                    recv_end, send_end = multiprocessing.Pipe(False)
                    qparams = qrc.QRCParams(hidden_unit_count=hidden_unit_count, max_coupling_energy=max_coupling_energy,\
                        trotter_step=trotter_step, beta=beta, virtual_nodes=V, tau_delta=tdelta, init_rho=0)
                    
                    p = multiprocessing.Process(target=eff_job, args=(qparams, pindex, buffer, input_seq_ls, Ntrials, send_end))
                    jobs.append(p)
                    pipels.append(recv_end)

            # Start the process
            for p in jobs:
                p.start()
    
            # Ensure all processes have finished execution
            for p in jobs:
                p.join()

            # Sleep 5s
            time.sleep(5)

            result_list = [np.array( [float(y) for y in x.recv().split(' ')]  ) for x in pipels]
            rsarr = np.array(result_list)
            # save the result
            np.savetxt('{}_EFF.txt'.format(outbase), rsarr, delimiter=' ')
        else:
            # Read the result
            rsarr = np.loadtxt(savedir)
            outbase = savedir.replace('.txt', '')

        print(rsarr)
        print(rsarr.shape)

        # plot the result
        xs, ys = tdeltas, virtuals
        zs = rsarr[:, 3].reshape(len(xs), len(ys))
        # zs = np.random.rand(len(xs), len(ys))
        # print(zs.shape)

        cmap = plt.get_cmap("viridis")
        plt.figure(figsize=(11,8))
        plt.style.use('seaborn-colorblind')
        plt.rc('font', family='serif')
        plt.rc('mathtext', fontset='cm')
        plt.rcParams['font.size']=20

        #plt.subplot(1, 2, i+1)
        #zs[i] = np.random.rand(len(xs), len(ys))
        plt.contourf(xs, ys, zs, args.resolution, cmap=cmap)
        plt.xlabel('$\\tau\Delta$', fontsize=32)
        plt.ylabel('$V$', fontsize=32)
        plt.xscale('log')
        cb = plt.colorbar()
        cb.set_label('Effective dim')
        #plt.show()
        for ftype in ['png', 'pdf']:
            plt.savefig('{}_EFF.{}'.format(outbase, ftype), bbox_inches='tight')
        
