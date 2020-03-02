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

train_len = 2000
val_len = 2000
buffer = 2000

hidden_unit_count = 5
max_coupling_energy = 1.0
trotter_step = 10
beta = 1e-14

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--units', type=int, default=5)
    parser.add_argument('--coupling', type=float, default=1.0)
    parser.add_argument('--trotter', type=int, default=10)
    parser.add_argument('--beta', type=float, default=1e-14)

    parser.add_argument('--trainlen', type=int, default=2000)
    parser.add_argument('--vallen', type=int, default=2000)
    parser.add_argument('--buffer', type=int, default=2000)
    
    parser.add_argument('--nproc', type=int, default=50)

    parser.add_argument('--basename', type=str, default='qrc_narma')
    parser.add_argument('--savedir', type=str, default='results')
    args = parser.parse_args()
    print(args)

    hidden_unit_count, max_coupling_energy, trotter_step, beta =\
        args.units, args.coupling, args.trotter, args.beta
    train_len, val_len, buffer = args.trainlen, args.vallen, args.buffer
    nproc = args.nproc

    basename, savedir = args.basename, args.savedir
    model = qrc.QuantumReservoirComputing()
    
    train_input_seq_ls, train_output_seq_ls = [], []
    val_input_seq_ls, val_output_seq_ls = [], []
    
    for order in [2, 5, 10, 15, 20]:
        data, target = gen.make_data_for_narma(train_len + val_len + buffer, order)

        train_input_seq_ls.append(  data[buffer  : buffer + train_len] )
        train_output_seq_ls.append( target[buffer  : buffer + train_len] )

        val_input_seq_ls.append(  data[buffer + train_len : buffer + train_len + val_len] )
        val_output_seq_ls.append( target[buffer + train_len : buffer + train_len + val_len] )

    train_input_seq_ls = np.array(train_input_seq_ls)
    train_output_seq_ls = np.array(train_output_seq_ls)

    model.train(train_input_seq_ls, train_output_seq_ls, hidden_unit_count, \
        max_coupling_energy, trotter_step, beta)
    
    train_pred_seq_ls, train_loss = model.predict(train_input_seq_ls, train_output_seq_ls)
    print("train_loss={}".format(train_loss))
    utils.plot_predict(train_input_seq_ls, train_output_seq_ls, train_pred_seq_ls)

    # Test phase
    #val_input_seq_ls = np.array(val_input_seq_ls)
    #val_output_seq_ls = np.array(val_output_seq_ls)
    #val_pred_seq_ls, val_loss = model.predict(val_input_seq_ls, val_output_seq_ls)