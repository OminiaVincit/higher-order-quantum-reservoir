import sys
import numpy as np
import os
import scipy
import argparse
from multiprocessing import Process
import matplotlib
import matplotlib.pyplot as plt
import tqdm
import time
import datetime
import qrc
import gendata as gen
import utils

def evaluation(outbase, qrcparams, train_input_seq_ls, train_output_seq_ls, val_input_seq_ls, val_output_seq_ls):
    
    model = qrc.QuantumReservoirComputing()
    
    model.train_to_predict(train_input_seq_ls, train_output_seq_ls, qrcparams)

    train_pred_seq_ls, train_loss = model.predict(train_input_seq_ls, train_output_seq_ls)
    print("train_loss={}".format(train_loss))
    print(train_pred_seq_ls.shape)
    
    
    # Test phase
    val_input_seq_ls = np.array(val_input_seq_ls)
    val_output_seq_ls = np.array(val_output_seq_ls)
    val_pred_seq_ls, val_loss = model.predict(val_input_seq_ls, val_output_seq_ls)
    print("val_loss={}".format(val_loss))
    print(val_pred_seq_ls.shape)

    # save experiments setting
    with open('{}_results.txt'.format(outbase), 'w') as sfile:
        sfile.write('train_loss={}\n'.format(train_loss))
        sfile.write('val_loss={}\n'.format(val_loss))
        sfile.write('hidden_unit_count={}\n'.format(qrcparams.hidden_unit_count))
        sfile.write('max_coupling_energy={}\n'.format(qrcparams.max_coupling_energy))
        sfile.write('trotter_step={}\n'.format(qrcparams.trotter_step))
        sfile.write('beta={}\n'.format(qparams.beta))
        sfile.write('virtual nodes={}\n'.format(qparams.virtual_nodes))
        sfile.write('tau_delta={}\n'.format(qparams.tau_delta))
        sfile.write('init_rho={}\n'.format(qparams.init_rho))
    
    rstrls = []
    rstrls.append('train_loss={}'.format(train_loss))
    rstrls.append('val_loss={}'.format(val_loss))
    rstrls.append('hidden_unit={},virtual={}'.format(qrcparams.hidden_unit_count, qparams.virtual_nodes))
    rstrls.append('Jdelta={},tau_delta={}'.format(qrcparams.max_coupling_energy, qparams.tau_delta))
    #rstrls.append('trotter_step={}'.format(qrcparams.trotter_step))
    #rstrls.append('beta={}'.format(qparams.beta))
    #rstrls.append('init_rho={}'.format(qparams.init_rho))

    rstr = '\n'.join(rstrls)
    utils.plot_predict_multi('{}_train'.format(outbase), rstr, train_input_seq_ls[0], \
        train_output_seq_ls[0].T, train_pred_seq_ls[0].T)

    utils.plot_predict_multi('{}_val'.format(outbase), rstr, val_input_seq_ls[0], \
        val_output_seq_ls[0].T, val_pred_seq_ls[0].T)
    
    
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
    parser.add_argument('--virtuals', type=int, default=10)
    parser.add_argument('--taudelta', type=float, default=1.0)

    parser.add_argument('--basename', type=str, default='qrc_narma')
    parser.add_argument('--savedir', type=str, default='results')
    args = parser.parse_args()
    print(args)

    hidden_unit_count, max_coupling_energy, trotter_step, beta =\
        args.units, args.coupling, args.trotter, args.beta
    train_len, val_len, buffer = args.trainlen, args.vallen, args.buffer
    nproc, virtual_nodes, tau_delta = args.nproc, args.virtuals, args.taudelta
    init_rho = args.rho

    basename, savedir = args.basename, args.savedir
    
    train_input_seq_ls, train_output_seq_ls = [], []
    val_input_seq_ls, val_output_seq_ls = [], []
    
    data, target = gen.make_data_for_narma(train_len + val_len + buffer, orders=[2, 5, 10, 15, 20])

    train_input_seq_ls.append(  data[buffer  : buffer + train_len] )
    train_output_seq_ls.append( target[buffer  : buffer + train_len] )

    val_input_seq_ls.append(  data[buffer + train_len : buffer + train_len + val_len] )
    val_output_seq_ls.append( target[buffer + train_len : buffer + train_len + val_len] )

    train_input_seq_ls = np.array(train_input_seq_ls)
    train_output_seq_ls = np.array(train_output_seq_ls)

    # Evaluation
    timestamp = int(time.time() * 1000.0)
    now = datetime.datetime.now()
    datestr = now.strftime('{0:%Y-%m-%d-%H-%M-%S}'.format(now))
    outbase = os.path.join(savedir, '{}_{}'.format(basename, datestr))

    qparams = qrc.QRCParams(hidden_unit_count=hidden_unit_count, max_coupling_energy=max_coupling_energy,\
            trotter_step=trotter_step, beta=beta, virtual_nodes=virtual_nodes, tau_delta=tau_delta, init_rho=init_rho)
    evaluation(outbase, qparams, train_input_seq_ls, train_output_seq_ls, val_input_seq_ls, val_output_seq_ls)