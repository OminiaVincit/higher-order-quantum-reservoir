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
import highorder_qrc as hqrc
import qrc
import gendata as gen
import utils

def nmse_job(qparams, nqrc, deep, layer_strength, buffer, train_input_seq, train_output_seq, \
        val_input_seq, val_output_seq, Ntrials, send_end, order):
    train_loss_ls, val_loss_ls = [], []
    print('Start process taudelta={}, virtual={}, Jdelta={}'.format(qparams.tau_delta, qparams.virtual_nodes, qparams.max_coupling_energy))
    for n in range(Ntrials):
        _, train_loss, _, val_loss = hqrc.get_loss(qparams, buffer, train_input_seq, train_output_seq, \
            val_input_seq, val_output_seq, nqrc, layer_strength, ranseed=n, deep=deep)
        train_loss_ls.append(train_loss)
        val_loss_ls.append(val_loss)

    mean_train, mean_val = np.mean(train_loss_ls), np.mean(val_loss_ls)
    std_train, std_val = np.std(train_loss_ls), np.std(val_loss_ls)
    #mean_train, mean_val = np.random.rand(), np.random.rand()

    rstr = '{} {} {} {} {} {} {} {}'.format(\
        order, nqrc, qparams.tau_delta, layer_strength, \
            mean_train, mean_val, std_train, std_val)
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
    parser.add_argument('--ntrials', type=int, default=10)
    parser.add_argument('--taudeltas', type=str, default='1.0')
    parser.add_argument('--layers', type=str, default='5')
    parser.add_argument('--strength', type=float, default=0.0)
    parser.add_argument('--virtuals', type=int, default=20)

    parser.add_argument('--deep', type=int, default=0)
    parser.add_argument('--orders', type=str, default='5,10,15,20')
    parser.add_argument('--basename', type=str, default='qrc_narma')
    parser.add_argument('--savedir', type=str, default='resnarma_high_orders')
    args = parser.parse_args()
    print(args)

    hidden_unit_count, max_coupling_energy, trotter_step, beta =\
        args.units, args.coupling, args.trotter, args.beta
    train_len, val_len, buffer = args.trainlen, args.vallen, args.buffer
    nproc, layer_strength, V = args.nproc, args.strength, args.virtuals
    init_rho = args.rho
    Ntrials = args.ntrials
    deep = False
    if args.deep > 0:
        deep = True

    basename, savedir = args.basename, args.savedir
    if os.path.isfile(savedir) == False and os.path.isdir(savedir) == False:
        os.mkdir(savedir)

    taudeltas = [float(x) for x in args.taudeltas.split(',')]
    taudeltas = [2**x for x in taudeltas]
    
    layers = [int(x) for x in args.layers.split(',')]
    orders = [int(x) for x in args.orders.split(',')]

    # Evaluation
    timestamp = int(time.time() * 1000.0)
    now = datetime.datetime.now()
    datestr = now.strftime('{0:%Y-%m-%d-%H-%M-%S}'.format(now))
    outbase = os.path.join(savedir, '{}_{}_units_{}_strength_{}_V_{}_layers_{}_narma_{}_ntrials_{}'.format(\
        basename, datestr, hidden_unit_count, layer_strength, V, '_'.join([str(o) for o in layers]), \
        '_'.join([str(o) for o in orders]), Ntrials))

    if os.path.isfile(savedir) == False:
        jobs, pipels = [], []
        for order in orders:
            data, target = gen.make_data_for_narma(train_len + val_len + buffer, orders=[order])

            train_input_seq_org = np.array(data[: buffer + train_len])
            train_input_seq_org = train_input_seq_org.reshape(1, train_input_seq_org.shape[0])
            
            train_output_seq = target[  : buffer + train_len] 

            val_input_seq_org =  np.array(data[buffer + train_len : buffer + train_len + val_len])
            val_input_seq_org = val_input_seq_org.reshape(1, val_input_seq_org.shape[0])
                
            val_output_seq = target[buffer + train_len : buffer + train_len + val_len]

            for nqrc in layers:
                train_input_seq = np.tile(train_input_seq_org, (nqrc, 1))
                val_input_seq = np.tile(val_input_seq_org, (nqrc, 1))
                for tau_delta in taudeltas:
                    recv_end, send_end = multiprocessing.Pipe(False)
                    qparams = qrc.QRCParams(hidden_unit_count=hidden_unit_count, max_coupling_energy=max_coupling_energy,\
                        trotter_step=trotter_step, beta=beta, virtual_nodes=V, tau_delta=tau_delta, init_rho=init_rho)
                    p = multiprocessing.Process(target=nmse_job, args=(qparams, nqrc, deep, layer_strength, buffer, train_input_seq, train_output_seq, \
                        val_input_seq, val_output_seq, Ntrials, send_end, order))
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

        # save experiments setting
        with open('{}_setting.txt'.format(outbase), 'w') as sfile:
            sfile.write('train_len={}, val_len={}, buffer={}\n'.format(train_len, val_len, buffer))
            sfile.write('hidden_unit_count={}\n'.format(hidden_unit_count))
            sfile.write('max_coupling_energy={}\n'.format(max_coupling_energy))
            sfile.write('trotter_step={}\n'.format(trotter_step))
            sfile.write('beta={}\n'.format(beta))
            sfile.write('taudeltas={}\n'.format(' '.join([str(v) for v in taudeltas])))
            sfile.write('layers={}\n'.format(' '.join([str(l) for l in layers])))
            sfile.write('V={}\n'.format(V))
            sfile.write('deep={}\n'.format(deep))
            sfile.write('layer_strength={}, Ntrials={}\n'.format(layer_strength, Ntrials))

    
