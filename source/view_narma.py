import sys
import numpy as np
import os
import scipy
import argparse
import multiprocessing
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
import tqdm
import time
import datetime
import qrc
import highorder_qrc as hqrc
import gendata as gen
import utils
import pickle

def predict_job(qparams, nqrc, layer_strength, buffer, train_input_seq, train_output_seq, \
        val_input_seq, val_output_seq, rseed, outbase):
    print('Start process strength={}, taudelta={}, virtual={}, Jdelta={}'.format(layer_strength, qparams.tau_delta, qparams.virtual_nodes, qparams.max_coupling_energy))
    train_pred_seq, train_loss, val_pred_seq, val_loss = hqrc.get_loss(qparams, buffer, train_input_seq, train_output_seq, \
        val_input_seq, val_output_seq, nqrc, layer_strength, ranseed=rseed)
    results = {'val_input': val_input_seq[0], 'val_pred': val_pred_seq, \
        'val_out': val_output_seq, 'train_loss': train_loss, 'val_loss': val_loss}
    #pickle.dump()
    with open('{}_alpha_{}.binaryfile'.format(outbase, layer_strength), 'wb') as wrs:
        pickle.dump(results, wrs)
    print(outbase, layer_strength, train_loss, val_loss)
    print(val_input_seq.shape, val_output_seq.shape)

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
    
    parser.add_argument('--ntrials', type=int, default=1)
    parser.add_argument('--virtuals', type=int, default=20)
    parser.add_argument('--taudelta', type=float, default=2.0)
    parser.add_argument('--layers', type=int, default=5)

    parser.add_argument('--orders', type=str, default='5,10,15,20')
    parser.add_argument('--basename', type=str, default='qrc_narma')
    parser.add_argument('--savedir', type=str, default='view_narma')
    parser.add_argument('--plot', type=int, default=0)
    args = parser.parse_args()
    print(args)

    hidden_unit_count, max_coupling_energy, trotter_step, beta =\
        args.units, args.coupling, args.trotter, args.beta
    train_len, val_len, buffer = args.trainlen, args.vallen, args.buffer
    V, tau_delta, nqrc = args.virtuals, args.taudelta, args.layers
    init_rho = args.rho

    basename, savedir = args.basename, args.savedir
    if os.path.isfile(savedir) == False and os.path.isdir(savedir) == False:
        os.mkdir(savedir)

    train_input_seq_ls, train_output_seq_ls = [], []
    val_input_seq_ls, val_output_seq_ls = [], []
    
    orders = [int(x) for x in args.orders.split(',')]
    N = len(orders)
    cmap = plt.get_cmap("viridis")
    #plt.style.use('seaborn-colorblind')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams['font.size']=14
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, axs = plt.subplots(N, 1, figsize=(6, 2.5*N))
    axs = axs.ravel()

    strengths = [0.0, 0.2, 0.4, 0.6, 0.8]
    for i in range(N):
        order = orders[i]
        outbase = os.path.join(savedir, '{}_{}_{}_{}_order_{}_V_{}_nqr_{}'.format(basename, \
                train_len, val_len, buffer, order, V, nqrc))
        
        if args.plot <= 0:
            data, target = gen.make_data_for_narma(train_len + val_len + buffer, orders = [order])

            train_input_seq_org = np.array(data[: buffer + train_len])
            train_input_seq_org = train_input_seq_org.reshape(1, train_input_seq_org.shape[0])
            train_output_seq = target[  : buffer + train_len] 

            val_input_seq_org =  np.array(data[buffer + train_len : buffer + train_len + val_len])
            val_input_seq_org = val_input_seq_org.reshape(1, val_input_seq_org.shape[0])
            val_output_seq = target[buffer + train_len : buffer + train_len + val_len]

            train_input_seq = np.tile(train_input_seq_org, (nqrc, 1))
            val_input_seq = np.tile(val_input_seq_org, (nqrc, 1))
        
            # Evaluation
            jobs, pipels = [], []

            for layer_strength in strengths:
                qparams = qrc.QRCParams(hidden_unit_count=hidden_unit_count, max_coupling_energy=max_coupling_energy,\
                            trotter_step=trotter_step, beta=beta, virtual_nodes=V, tau_delta=tau_delta, init_rho=init_rho)
                rseed = 0
                p = multiprocessing.Process(target=predict_job, args=(qparams, nqrc, layer_strength, buffer, train_input_seq, train_output_seq, \
                        val_input_seq, val_output_seq, rseed, outbase))
                jobs.append(p)
            
                    # Start the process
            for p in jobs:
                p.start()

            # Ensure all processes have finished execution
            for p in jobs:
                p.join()
        else:
            ax = axs[i]
            bg = int(val_len / 2)
            ed = bg + 100
            for j in range(len(strengths)):
                layer_strength = strengths[j]
                with open('{}_alpha_{}.binaryfile'.format(outbase, layer_strength), 'rb') as rrs:
                    results = pickle.load(rrs)
                    #val_input  = results['val_input'].ravel()
                    val_pred   = results['val_pred'].ravel()
                    val_out    = results['val_out'].ravel()
                    if layer_strength == 0:
                        #axs[0].plot(val_input[bg:ed], c='black')
                        ax.plot(val_out[bg:ed], c='black', label='target')
                    ax.plot(val_pred[bg:ed], c=colors[j], label='{}'.format(layer_strength))
            if i == 0:
                ax.legend()
            #if i < N-1:
            ax.set_xticklabels([])
    if args.plot > 0:
        for ftype in ['pdf', 'svg']:
            outbase = os.path.join(savedir, '{}_{}_{}_{}_V_{}_nqr_{}'.format(basename, \
                    train_len, val_len, buffer, V, nqrc))
            plt.savefig('{}_narma.{}'.format(outbase, ftype), bbox_inches='tight')
        plt.show()
            
        
                
    
    