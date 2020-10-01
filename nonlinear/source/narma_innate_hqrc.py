#!/usr/bin/env python
"""
    Calculate NMSE for NARMA tasks with quantum innate learning
    See run_hqrc_innate_narma.sh for an example to run the script
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
from scipy.special import legendre
import hqrc_innate as hqrc
import utils
from utils import *

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--units', type=int, default=5, help='Number of the hidden units')
    parser.add_argument('--coupling', type=float, default=1.0, help='Maximum coupling energy')
    parser.add_argument('--rho', type=int, default=0, help='Flag for initializing the density matrix')
    parser.add_argument('--beta', type=float, default=1e-14, help='reg term')
    parser.add_argument('--solver', type=str, default=LINEAR_PINV, \
        help='regression solver by linear_pinv,ridge_pinv,auto,svd,cholesky,lsqr,sparse_cg,sag')

    parser.add_argument('--trainlen', type=int, default=2000)
    parser.add_argument('--vallen', type=int, default=2000)
    parser.add_argument('--transient', type=int, default=2000, help='Transitient time steps')
    
    parser.add_argument('--ntrials', type=int, default=1)
    parser.add_argument('--virtuals', type=int, default=1)
    parser.add_argument('--taudelta', type=float, default=2.0, help='Interval between the inputs')
    parser.add_argument('--strength', type=float, default=0.5, help='Connection strengths')
    parser.add_argument('--nqrc', type=int, default=5, help='Number of reservoirs')

    parser.add_argument('--orders', type=str, default='10')
    parser.add_argument('--basename', type=str, default='qrc_narma_innate')
    parser.add_argument('--savedir', type=str, default='resnarma_innate')
    parser.add_argument('--ranseed', type=int, default=1)
    parser.add_argument('--trainloops', type=int, default=1)
    parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, default=10.0)
    parser.add_argument('--scale_input', type=float, default=0.4)
    args = parser.parse_args()
    print(args)

    n_units, max_energy, beta = args.units, args.coupling, args.beta
    train_len, val_len, buffer = args.trainlen, args.vallen, args.transient
    V = args.virtuals
    init_rho, solver = args.rho, args.solver

    Ntrials, ranseed, train_loops, noise_amp = args.ntrials, args.ranseed, args.trainloops, args.noise
    learning_rate, scale_input = args.learning_rate, args.scale_input

    basename, savedir = args.basename, args.savedir
    if os.path.isdir(savedir) == False:
        os.mkdir(savedir)
    tau, alpha, nqrc = args.taudelta, args.strength, args.nqrc
    orders = [int(x) for x in args.orders.split(',')]

    # Evaluation
    timestamp = int(time.time() * 1000.0)
    now = datetime.datetime.now()
    datestr = now.strftime('{0:%Y-%m-%d-%H-%M-%S}'.format(now))

    sel = 0
    bg = 50
    for order in orders:
        outbase = os.path.join(savedir, '{}_{}_{}_units_{}_V_{}_a_{}_QRs_{}_narma_{}_n_{}_loops_{}_noise_{}_r_{}_sc_{}_sd_{}'.format(\
            basename, solver, datestr, n_units, V, alpha, nqrc, order, Ntrials, train_loops, noise_amp, learning_rate, scale_input, ranseed))

            # save experiments setting
        with open('{}_setting.txt'.format(outbase), 'w') as sfile:
            sfile.write('train_len={}, val_len={}, buffer={}\n'.format(train_len, val_len, buffer))
            sfile.write('n_units={}\n'.format(n_units))
            sfile.write('max_energy={}\n'.format(max_energy))
            sfile.write('beta={}\n'.format(beta))
            sfile.write('taudelta={}\n'.format(tau))
            sfile.write('layers={}\n'.format(nqrc))
            sfile.write('V={}\n'.format(V))
            sfile.write('alpha={}, Ntrials={}\n'.format(alpha, Ntrials))
            sfile.write('noise={}, learning rate={},scale_input={}\n'.format(noise_amp, learning_rate, scale_input))

        val_loss_ls = []
        for n in range(Ntrials):
            np.random.seed(seed=n)
            # Create input - target
            data, target = make_data_for_narma(train_len + val_len + buffer, orders=[order])
            #data = np.random.randint(2, size=train_len + val_len)

            # Create qparams and model
            qparams = QRCParams(n_units=n_units, max_energy=max_energy,\
                beta=beta, virtual_nodes=V, tau=tau, init_rho=False)
            model = hqrc.HQRC(nqrc, alpha)
            new_ranseed = ranseed + n * 100

            # For PRE Training
            innate_buffer = 100
            innate_train_len = 200
            innate_val_len = 100

            pre_input_seq_org = np.array(data[: innate_buffer + innate_train_len + innate_val_len])
            pre_input_seq_org = pre_input_seq_org.reshape(1, pre_input_seq_org.shape[0])
            pre_input_seq = np.tile(pre_input_seq_org, (nqrc, 1))
            
            # Create innate target activity
            state_list = model.init_forward(qparams, pre_input_seq, ranseed=new_ranseed, noise_amp=0.0, scale_input=scale_input)
            N_local = model.get_local_nodes()
            n_qubits = model.n_qubits
            innate_seq = state_list[:, sel::N_local]

            # Create target activity from legendre polynomial
            target_innate_seq = []
            for d in range(nqrc):
                input_lg = legendre(d+1)(pre_input_seq_org.reshape(-1)) 
                input_lg = (input_lg - np.min(input_lg)) / (np.max(input_lg) - np.min(input_lg))
                mi, ma = np.min(innate_seq[:, d]), np.max(innate_seq[:, d])
                input_lg = input_lg * (ma - mi) + mi
                target_innate_seq.append(input_lg)
            target_innate_seq = np.array(target_innate_seq).T

            # Pre training
            trained_state_list, dW_recurr_ls, loss_dict \
                = model.innate_train(pre_input_seq, target_innate_seq, innate_buffer, innate_train_len, \
                    ranseed=new_ranseed + 1000, learn_every=1, \
                    noise_amp=noise_amp, learning_rate=learning_rate, \
                    scale_input=scale_input, train_loops=train_loops)

            fig = plt.figure(figsize=(16, 16))
            plt.subplots_adjust(wspace=0.4, hspace=0.5)
            ax = fig.add_subplot(nqrc+3, 1, 1)
            ax.plot(pre_input_seq[0, bg:])
            ax.set_ylabel('Input')
            ax.set_title(outbase)

            for i in range(nqrc):
                ax = fig.add_subplot(nqrc+3, 1, i+2)
                ax.plot(target_innate_seq[bg:, i], label='target')
                ax.plot(trained_state_list[bg:, N_local * i], label='trained')
                diff_state = target_innate_seq[(innate_buffer + innate_train_len):, i] - trained_state_list[(innate_buffer + innate_train_len):, sel + N_local * i]
                target_state = target_innate_seq[(innate_buffer + innate_train_len):, i] 
                #nmse = np.mean(diff_state**2) / np.mean(target_state**2)
                loss = np.sqrt(np.mean(diff_state**2))
                print('QR {}, Loss={}'.format(i, loss))
                ax.set_title('Val loss={}'.format(loss))
                #ax.plot(trained_state_list[:, N_local * i] - target_state_list[:, N_local * i])
                ax.set_ylabel('QR_{}'.format(i))
                ax.legend()
            
            ax = fig.add_subplot(nqrc+3, 1, nqrc+2)
            ax.plot(dW_recurr_ls[bg:])
            ax.set_ylabel('dW')
            ax.set_xlabel('Time step')

            # Plot for NMSE of Pre-training
            ax = fig.add_subplot(nqrc+3, 1, nqrc+3)
            for i in range(nqrc):
                ax.plot(loss_dict[i], 'o--', label='QR-{}'.format(i))
            ax.set_ylabel('Train loss')
            ax.set_xlabel('Train loops')
            ax.legend()

            for ftype in ['png']:
                plt.savefig('{}.{}'.format(outbase, ftype), bbox_inches='tight', dpi=600)
            plt.show()
            #print(np.sum(target_innate_seq[:, :] - test_state_list[:, ::N_local]))

            # Training NARMA
            if False:
                train_input_seq_org = np.array(data[: buffer + train_len])
                train_input_seq_org = train_input_seq_org.reshape(1, train_input_seq_org.shape[0])
                train_output_seq = target[  : buffer + train_len] 

                val_input_seq_org =  np.array(data[buffer + train_len : buffer + train_len + val_len])
                val_input_seq_org = val_input_seq_org.reshape(1, val_input_seq_org.shape[0])
                val_output_seq = target[buffer + train_len : buffer + train_len + val_len]

                train_input_seq = np.tile(train_input_seq_org, (nqrc, 1))
                val_input_seq = np.tile(val_input_seq_org, (nqrc, 1))

                model.train(train_input_seq, train_output_seq, buffer=buffer, beta=beta, \
                    ranseed=new_ranseed + 2000, noise_amp=noise_amp, scale_input=scale_input)
                val_pred_seq, val_loss = model.predict(val_input_seq, val_output_seq, buffer=0, \
                    noise_amp=noise_amp, scale_input=scale_input)
                print('n={}, val_loss={}'.format(n, val_loss))
                val_loss_ls.append(val_loss)
        #print('Ntrials={}, avg loss={}'.format(Ntrials, np.mean(val_loss_ls)))