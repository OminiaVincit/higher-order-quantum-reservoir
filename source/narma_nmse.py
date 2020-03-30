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

virtuals = [5*n for n in range(1, 11)]
virtuals.insert(0, 1)

def nmse_job(qparams, buffer, train_input_seq_ls, train_output_seq_ls, val_input_seq_ls, val_output_seq_ls, Ntrials, send_end):
    train_loss_ls, val_loss_ls = [], []
    print('Start process taudelta={}, virtual={}, Jdelta={}'.format(qparams.tau_delta, qparams.virtual_nodes, qparams.max_coupling_energy))
    for n in range(Ntrials):
         _, train_loss, _, val_loss = qrc.get_loss(qparams, buffer, train_input_seq_ls, train_output_seq_ls, val_input_seq_ls, val_output_seq_ls, ranseed=n)
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
    parser.add_argument('--taudelta', type=float, default=2.0)

    parser.add_argument('--orders', type=str, default='10')
    parser.add_argument('--basename', type=str, default='qrc_narma')
    parser.add_argument('--savedir', type=str, default='resnarma_nmse')
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

    train_input_seq_ls.append(  data[: buffer + train_len] )
    train_output_seq_ls.append( target[  : buffer + train_len] )

    val_input_seq_ls.append(  data[buffer + train_len : buffer + train_len + val_len] )
    val_output_seq_ls.append( target[buffer + train_len : buffer + train_len + val_len] )

    # Evaluation
    timestamp = int(time.time() * 1000.0)
    now = datetime.datetime.now()
    datestr = now.strftime('{0:%Y-%m-%d-%H-%M-%S}'.format(now))
    outbase = os.path.join(savedir, '{}_{}_tdt_{}_order_{}'.format(basename, datestr, tau_delta, '_'.join([str(o) for o in orders])))

    if os.path.isfile(savedir) == False:
        jobs, pipels = [], []

        for V in virtuals:
            recv_end, send_end = multiprocessing.Pipe(False)
            qparams = qrc.QRCParams(hidden_unit_count=hidden_unit_count, max_coupling_energy=max_coupling_energy,\
                trotter_step=trotter_step, beta=beta, virtual_nodes=V, tau_delta=tau_delta, init_rho=init_rho)
            p = multiprocessing.Process(target=nmse_job, args=(qparams, buffer, train_input_seq_ls, train_output_seq_ls, \
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
    xs = virtuals
    zs = dict()
    zs[0] = rsarr[:, 3]
    zs[1] = rsarr[:, 4]
    # zs = np.random.rand(len(xs), len(ys))
    # print(zs.shape)

    cmap = plt.get_cmap("viridis")
    labels = ['train_NMSE', 'val_NMSE']
    plt.figure(figsize=(22,8))
    plt.style.use('seaborn-colorblind')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams['font.size']=20

    plt.plot(xs, zs[0], 'o--', label='Train loss')
    plt.plot(xs, zs[1], 'o--', label='Test loss')
    plt.ylim([1e-4, 1e-2])
    plt.xlabel('$V$', fontsize=32)
    plt.ylabel('NMSE', fontsize=32)
    plt.yscale('log')

    plt.legend()
    plt.title(outbase)
    plt.grid(True, which="both", ls="-", color='0.65')
    plt.show()
    for ftype in ['png', 'pdf']:
        plt.savefig('{}_NMSE.{}'.format(outbase, ftype), bbox_inches='tight')
        
