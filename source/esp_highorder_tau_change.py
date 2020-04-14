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
from loginit import get_module_logger

# virtuals = [5*n for n in range(1, 6)]
# virtuals.insert(0, 1)

# layers = [n for n in range(1, 6)]
# strengths = [0.0 0.1 0.3 0.5 0.7 0.9 1.0]

def esp_job(qparams, nqrc, layer_strength, buffer, input_seq, state_trials, net_trials, send_end):
    print('Start process layer={}, taudelta={}, virtual={}, Jdelta={}'.format(nqrc, qparams.tau_delta, qparams.virtual_nodes, qparams.max_coupling_energy))
    btime = int(time.time() * 1000.0)
    dPs, ldas = [], []
    for n in range(net_trials):
        dP, lda = hqrc.esp_index(qparams, buffer, input_seq, nqrc, layer_strength, ranseed=n, state_trials=state_trials)
        dPs.append(dP)
        ldas.append(lda)

    mean_dp, std_dp = np.mean(dPs), np.std(dPs)
    max_lda, mean_lda, std_lda = np.max(ldas), np.mean(ldas), np.std(ldas)
    
    rstr = '{} {} {} {} {} {} {} {}'.format(\
        nqrc, qparams.tau_delta, layer_strength, mean_dp, std_dp, max_lda, mean_lda, std_lda)
    etime = int(time.time() * 1000.0)
    now = datetime.datetime.now()
    datestr = now.strftime('{0:%Y-%m-%d-%H-%M-%S}'.format(now))
    print('{} Finish process {} in {}s'.format(datestr, rstr, etime-btime))
    send_end.send(rstr)

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--units', type=int, default=5)
    parser.add_argument('--coupling', type=float, default=1.0)
    parser.add_argument('--trotter', type=int, default=10)
    parser.add_argument('--rho', type=int, default=0)
    parser.add_argument('--beta', type=float, default=1e-14)

    parser.add_argument('--length', type=int, default=4000)
    parser.add_argument('--buffer', type=int, default=2000)
    
    parser.add_argument('--ntrials', type=int, default=1)
    parser.add_argument('--strials', type=int, default=1)

    parser.add_argument('--taudeltas', type=str, default='-4,-3,-2,-1,0,1,2,3,4,5,6,7')
    parser.add_argument('--layers', type=str, default='1,2,3,4,5')
    parser.add_argument('--strength', type=float, default=0.0)
    parser.add_argument('--virtuals', type=int, default=15)

    parser.add_argument('--orders', type=str, default='10')
    parser.add_argument('--basename', type=str, default='qrc_narma_echo')
    parser.add_argument('--savedir', type=str, default='res_high_echo_tau')
    args = parser.parse_args()
    print(args)

    hidden_unit_count, max_coupling_energy, trotter_step, beta =\
        args.units, args.coupling, args.trotter, args.beta
    length, buffer = args.length, args.buffer
    layer_strength, V = args.strength, args.virtuals
    init_rho = args.rho
    net_trials, state_trials = args.ntrials, args.strials

    basename, savedir = args.basename, args.savedir
    if os.path.isfile(savedir) == False and os.path.isdir(savedir) == False:
        os.mkdir(savedir)

    taudeltas = [float(x) for x in args.taudeltas.split(',')]
    taudeltas = [2**x for x in taudeltas]
    
    layers = [int(x) for x in args.layers.split(',')]
    
    orders = [int(x) for x in args.orders.split(',')]
    data, target = gen.make_data_for_narma(length, orders=orders)
    input_seq_org = np.array(data)
    input_seq_org = input_seq_org.reshape(1, input_seq_org.shape[0])
    

    # Evaluation
    timestamp = int(time.time() * 1000.0)
    now = datetime.datetime.now()
    datestr = now.strftime('{0:%Y-%m-%d-%H-%M-%S}'.format(now))
    outbase = os.path.join(savedir, '{}_{}_strength_{}_V_{}_layers_{}_narma_{}_esp_trials_{}_{}'.format(\
        basename, datestr, layer_strength, V, '_'.join([str(l) for l in layers]), \
            '_'.join([str(o) for o in orders]), net_trials, state_trials))
    
    if os.path.isfile(savedir) == False:
        jobs, pipels = [], []
        for nqrc in layers:
            input_seq = np.tile(input_seq_org, (nqrc, 1))
            for tau_delta in taudeltas:
                recv_end, send_end = multiprocessing.Pipe(False)
                qparams = qrc.QRCParams(hidden_unit_count=hidden_unit_count, max_coupling_energy=max_coupling_energy,\
                    trotter_step=trotter_step, beta=beta, virtual_nodes=V, tau_delta=tau_delta, init_rho=init_rho)
                p = multiprocessing.Process(target=esp_job, \
                    args=(qparams, nqrc, layer_strength, buffer, input_seq, net_trials, state_trials, send_end))
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
        np.savetxt('{}_esp.txt'.format(outbase), rsarr, delimiter=' ')

        # save experiments setting
        with open('{}_setting.txt'.format(outbase), 'w') as sfile:
            sfile.write('length={}, buffer={}\n'.format(length, buffer))
            sfile.write('hidden_unit_count={}\n'.format(hidden_unit_count))
            sfile.write('max_coupling_energy={}\n'.format(max_coupling_energy))
            sfile.write('trotter_step={}\n'.format(trotter_step))
            sfile.write('beta={}\n'.format(beta))
            sfile.write('taudeltas={}\n'.format(' '.join([str(v) for v in taudeltas])))
            sfile.write('layers={}\n'.format(' '.join([str(l) for l in layers])))
            sfile.write('V={}\n'.format(V))
            sfile.write('layer_strength={}, net_trials={}, state_trials={}\n'.format(layer_strength, net_trials, state_trials))

    else:
        # Read the result
        rsarr = np.loadtxt(savedir)
        outbase = savedir.replace('.txt', '')

    print(rsarr)
    print(rsarr.shape)

    # plot the result
    xs = taudeltas
    avg_effs, std_effs = rsarr[:, 3], rsarr[:, 4]
    max_ldas = rsarr[:, 5]
    avg_ldas, std_ldas = rsarr[:, 6], rsarr[:, 7]

    cmap = plt.get_cmap("viridis")
    plt.figure(figsize=(16,8))
    #plt.style.use('seaborn-colorblind')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams['font.size']=20

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('$\\tau\Delta$', fontsize=28)
    ax1.set_ylabel('Esp index', fontsize=28)
    ax1.set_xscale('log', basex=2)
    # ax2 = ax1.twinx()
    # ax2.set_ylabel('Max $\lambda$', fontsize=28)
    
    for nqrc in layers:
        ids = (rsarr[:, 0] == nqrc)
        ax1.errorbar(xs, avg_effs[ids], yerr=std_effs[ids], elinewidth=2, linewidth=2, markersize=12, \
            label='Esp-Layers={}'.format(nqrc))
        #ax2.plot(xs, avg_ldas[ids], 'o--',linewidth=2, markersize=10, label='$\lambda$-Layers={}'.format(nqrc))
    #plt.xlim([1e-3, 1024])    
    #plt.ylim([1e-6, 1e-2])
    ax1.set_yscale('log', basey=10)
    ax1.legend(fontsize=14)
    #ax2.legend(fontsize=14)
    plt.title(os.path.basename(outbase), fontsize=12)
    ax1.grid(True, which="both", ls="-", color='0.65')
    #plt.show()
    for ftype in ['png', 'pdf']:
        plt.savefig('{}_esp.{}'.format(outbase, ftype), bbox_inches='tight')
 
