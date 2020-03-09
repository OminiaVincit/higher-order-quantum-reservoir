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

N = 10 
tdeltas = [2**n for n in range(N)]
tdeltas.insert(0, 0.5)

virtuals = [5*n for n in range(1, N+1)]
virtuals.insert(0, 1)

def nmse_job(qparams, train_input_seq_ls, train_output_seq_ls, val_input_seq_ls, val_output_seq_ls, Ntrials, send_end):
    train_loss_ls, val_loss_ls = [], []
    print('Start process taudelta={}, virtual={}, Jdelta={}'.format(qparams.tau_delta, qparams.virtual_nodes, qparams.max_coupling_energy))
    for n in range(Ntrials):
         _, train_loss, _, val_loss = qrc.get_loss(qparams, train_input_seq_ls, train_output_seq_ls, val_input_seq_ls, val_output_seq_ls)
         train_loss_ls.append(train_loss)
         val_loss_ls.append(val_loss)

    mean_train, mean_val = np.mean(train_loss_ls), np.mean(val_loss_ls)
    #mean_train, mean_val = np.random.rand(), np.random.rand()

    rstr = '{} {} {} {} {}'.format(\
        qparams.tau_delta, qparams.virtual_nodes, qparams.max_coupling_energy, mean_train, mean_val)
    print('Finish process {}'.format(rstr))
    send_end.send(rstr)

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--units', type=int, default=5)
    parser.add_argument('--coupling', type=float, default=1.0)
    parser.add_argument('--trotter', type=int, default=10)
    parser.add_argument('--rho', type=int, default=0)
    parser.add_argument('--beta', type=float, default=1e-14)

    parser.add_argument('--trainlen', type=int, default=2000)
    parser.add_argument('--vallen', type=int, default=2000)
    parser.add_argument('--buffer', type=int, default=2000)
    
    parser.add_argument('--nproc', type=int, default=50)
    parser.add_argument('--ntrials', type=int, default=1)
    parser.add_argument('--virtuals', type=int, default=10)
    parser.add_argument('--taudelta', type=float, default=1.0)

    parser.add_argument('--orders', type=str, default='2,5,10,15,20')
    parser.add_argument('--basename', type=str, default='qrc_narma')
    parser.add_argument('--savedir', type=str, default='results')
    parser.add_argument('--eval', type=int, default=1)
    args = parser.parse_args()
    print(args)

    hidden_unit_count, max_coupling_energy, trotter_step, beta =\
        args.units, args.coupling, args.trotter, args.beta
    train_len, val_len, buffer = args.trainlen, args.vallen, args.buffer
    nproc, virtual_nodes, tau_delta = args.nproc, args.virtuals, args.taudelta
    init_rho = args.rho
    Ntrials = args.ntrials

    basename, savedir = args.basename, args.savedir
    if os.path.isfile(savedir) == False and os.path.isdir(savedir) == False:
        os.mkdir(savedir)

    train_input_seq_ls, train_output_seq_ls = [], []
    val_input_seq_ls, val_output_seq_ls = [], []
    
    orders = [int(x) for x in args.orders.split(',')]
    data, target = gen.make_data_for_narma(train_len + val_len + buffer, orders=orders)

    train_input_seq_ls.append(  data[buffer  : buffer + train_len] )
    train_output_seq_ls.append( target[buffer  : buffer + train_len] )

    val_input_seq_ls.append(  data[buffer + train_len : buffer + train_len + val_len] )
    val_output_seq_ls.append( target[buffer + train_len : buffer + train_len + val_len] )

    # Evaluation
    timestamp = int(time.time() * 1000.0)
    now = datetime.datetime.now()
    datestr = now.strftime('{0:%Y-%m-%d-%H-%M-%S}'.format(now))
    outbase = os.path.join(savedir, '{}_{}_order_{}'.format(basename, datestr, '_'.join([str(o) for o in orders])))

    if args.eval == 0:
        qparams = qrc.QRCParams(hidden_unit_count=hidden_unit_count, max_coupling_energy=max_coupling_energy,\
                trotter_step=trotter_step, beta=beta, virtual_nodes=virtual_nodes, tau_delta=tau_delta, init_rho=init_rho)
        qrc.evaluation(outbase, qparams, train_input_seq_ls, train_output_seq_ls, val_input_seq_ls, val_output_seq_ls)
    
    if args.eval == 1:
        if os.path.isfile(savedir) == False:
            jobs, pipels = [], []

            for tdelta in tdeltas:
                for V in virtuals:
                    recv_end, send_end = multiprocessing.Pipe(False)
                    qparams = qrc.QRCParams(hidden_unit_count=hidden_unit_count, max_coupling_energy=max_coupling_energy,\
                        trotter_step=trotter_step, beta=beta, virtual_nodes=V, tau_delta=tdelta, init_rho=init_rho)
                    p = multiprocessing.Process(target=nmse_job, args=(qparams, train_input_seq_ls, train_output_seq_ls, \
                        val_input_seq_ls, val_output_seq_ls, Ntrials, send_end))
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
            np.savetxt('{}_NMSE.txt'.format(outbase), rsarr, delimiter=' ')
        else:
            # Read the result
            rsarr = np.loadtxt(savedir)
            outbase = savedir.replace('.txt', '')

        print(rsarr)
        print(rsarr.shape)

        # plot the result
        xs, ys = tdeltas, virtuals
        zs = dict()
        zs[0] = np.log10(rsarr[:, 3].reshape(len(xs), len(ys)))
        zs[1] = np.log10(rsarr[:, 4].reshape(len(xs), len(ys)))
        # zs = np.random.rand(len(xs), len(ys))
        # print(zs.shape)

        cmap = plt.get_cmap("viridis")
        labels = ['train_NMSE', 'val_NMSE']
        plt.figure(figsize=(22,8))
        plt.style.use('seaborn-colorblind')
        plt.rc('font', family='serif')
        plt.rc('mathtext', fontset='cm')
        plt.rcParams['font.size']=20

        for i in range(2):
            plt.subplot(1, 2, i+1)
            #zs[i] = np.random.rand(len(xs), len(ys))
            plt.contourf(xs, ys, zs[i], 64, cmap=cmap)
            plt.xlabel('$\\tau\Delta$', fontsize=32)
            plt.ylabel('$V$', fontsize=32)
            plt.xscale('log')
            cb = plt.colorbar()
            cb.set_label('log_{}'.format(labels[i]))
        plt.show()
        for ftype in ['png', 'pdf']:
            plt.savefig('{}_NMSE.{}'.format(outbase, ftype), bbox_inches='tight')
        
