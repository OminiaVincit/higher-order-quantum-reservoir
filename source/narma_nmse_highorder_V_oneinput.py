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
import highorder_qrc_oneinput as hqrc
import qrc
import gendata as gen
import utils

# virtuals = [5*n for n in range(1, 6)]
# virtuals.insert(0, 1)

# layers = [n for n in range(1, 6)]
# strengths = [0.0 0.1 0.3 0.5 0.7 0.9 1.0]

def nmse_job(qparams, nqrc, layer_strength, buffer, train_input_seq, train_output_seq, \
        val_input_seq, val_output_seq, Ntrials, send_end):
    train_loss_ls, val_loss_ls = [], []
    print('Start process taudelta={}, virtual={}, Jdelta={}'.format(qparams.tau_delta, qparams.virtual_nodes, qparams.max_coupling_energy))
    for n in range(Ntrials):
        _, train_loss, _, val_loss = hqrc.get_loss(qparams, buffer, train_input_seq, train_output_seq, val_input_seq, val_output_seq, nqrc, layer_strength)
        train_loss_ls.append(train_loss)
        val_loss_ls.append(val_loss)

    mean_train, mean_val = np.mean(train_loss_ls), np.mean(val_loss_ls)
    std_train, std_val = np.std(train_loss_ls), np.std(val_loss_ls)
    #mean_train, mean_val = np.random.rand(), np.random.rand()

    rstr = '{} {} {} {} {} {} {}'.format(\
        nqrc, qparams.virtual_nodes, layer_strength, \
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
    parser.add_argument('--ntrials', type=int, default=1)
    parser.add_argument('--virtuals', type=str, default='1,5,10,15,20,25')
    parser.add_argument('--layers', type=str, default='1,2,3,4,5')
    parser.add_argument('--strength', type=float, default=0.0)
    parser.add_argument('--taudelta', type=float, default=2.0)

    parser.add_argument('--orders', type=str, default='10')
    parser.add_argument('--basename', type=str, default='qrc_narma')
    parser.add_argument('--savedir', type=str, default='resnarma_high_oneinput')
    args = parser.parse_args()
    print(args)

    hidden_unit_count, max_coupling_energy, trotter_step, beta =\
        args.units, args.coupling, args.trotter, args.beta
    train_len, val_len, buffer = args.trainlen, args.vallen, args.buffer
    nproc, layer_strength, tau_delta = args.nproc, args.strength, args.taudelta
    init_rho = args.rho
    Ntrials = args.ntrials

    basename, savedir = args.basename, args.savedir
    if os.path.isfile(savedir) == False and os.path.isdir(savedir) == False:
        os.mkdir(savedir)

    virtuals = [int(x) for x in args.virtuals.split(',')]
    layers = [int(x) for x in args.layers.split(',')]
    orders = [int(x) for x in args.orders.split(',')]

    data, target = gen.make_data_for_narma(train_len + val_len + buffer, orders=orders)

    train_input_seq_org = np.array(data[: buffer + train_len])
    train_input_seq_org = train_input_seq_org.reshape(1, train_input_seq_org.shape[0])
    
    train_output_seq = target[  : buffer + train_len] 

    val_input_seq_org =  np.array(data[buffer + train_len : buffer + train_len + val_len])
    val_input_seq_org = val_input_seq_org.reshape(1, val_input_seq_org.shape[0])
        
    val_output_seq = target[buffer + train_len : buffer + train_len + val_len]

    # Evaluation
    timestamp = int(time.time() * 1000.0)
    now = datetime.datetime.now()
    datestr = now.strftime('{0:%Y-%m-%d-%H-%M-%S}'.format(now))
    outbase = os.path.join(savedir, '{}_{}_strength_{}_tdt_{}_narma_{}_ntrials_{}'.format(\
        basename, datestr, layer_strength, tau_delta, '_'.join([str(o) for o in orders]), Ntrials))

    if os.path.isfile(savedir) == False:
        jobs, pipels = [], []
        for nqrc in layers:
            train_input_seq = np.tile(train_input_seq_org, (nqrc, 1))
            val_input_seq = np.tile(val_input_seq_org, (nqrc, 1))
            for V in virtuals:
                recv_end, send_end = multiprocessing.Pipe(False)
                qparams = qrc.QRCParams(hidden_unit_count=hidden_unit_count, max_coupling_energy=max_coupling_energy,\
                    trotter_step=trotter_step, beta=beta, virtual_nodes=V, tau_delta=tau_delta, init_rho=init_rho)
                p = multiprocessing.Process(target=nmse_job, args=(qparams, nqrc, layer_strength, buffer, train_input_seq, train_output_seq, \
                    val_input_seq, val_output_seq, Ntrials, send_end))
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
            sfile.write('virtual nodes={}\n'.format(' '.join([str(v) for v in virtuals])))
            sfile.write('layers={}\n'.format(' '.join([str(l) for l in layers])))
            sfile.write('tau_delta={}\n'.format(tau_delta))
            sfile.write('layer_strength={}, Ntrials={}\n'.format(layer_strength, Ntrials))

    else:
        # Read the result
        rsarr = np.loadtxt(savedir)
        outbase = savedir.replace('.txt', '')

    print(rsarr)
    print(rsarr.shape)

    # plot the result
    xs = virtuals
    avg_trains, std_trains = rsarr[:, 3], rsarr[:, 5]
    avg_tests, std_tests = rsarr[:, 4], rsarr[:, 6]
    
    cmap = plt.get_cmap("viridis")
    plt.figure(figsize=(8,8))
    #plt.style.use('seaborn-colorblind')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams['font.size']=20

    for nqrc in layers:
        ids = (rsarr[:, 0] == nqrc)
        #plt.plot(xs, avg_trains, 'o--',  linewidth=2, markersize=12, \
        #    label='Train layers={}'.format(nqrc))
        #plt.plot(xs, avg_tests, 'o--',  linewidth=2, markersize=12, \
        #    label='Test layers={}'.format(nqrc))

        # plt.errorbar(xs, avg_trains, yerr=std_trains, elinewidth=2, linewidth=2, markersize=12, \
        #     label='Train layers={}'.format(nqrc))
        plt.errorbar(xs, avg_tests[ids], yerr=std_tests[ids], elinewidth=2, linewidth=2, markersize=12, \
            label='Layers={}'.format(nqrc))
        
    plt.ylim([1e-6, 1e-2])
    plt.xlabel('$V$', fontsize=28)
    plt.ylabel('NMSE', fontsize=28)
    plt.yscale('log')

    plt.legend()
    plt.title(outbase, fontsize=12)
    plt.grid(True, which="both", ls="-", color='0.65')
    plt.show()
    for ftype in ['png', 'pdf']:
        plt.savefig('{}_NMSE.{}'.format(outbase, ftype), bbox_inches='tight')
 
