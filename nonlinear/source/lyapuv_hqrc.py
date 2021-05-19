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
import hqrc as hqrc
import utils
from utils import *
from loginit import get_module_logger

def lyp_job(qparams, nqrc, layer_strength, buffer, length, sparsity, sigma_input, nonlinear, net_trials, initial_distance, send_end):
    print('Start process layer={}, taudelta={}, virtual={}, strength={}'.format(\
        nqrc, qparams.tau, qparams.virtual_nodes, layer_strength))
    btime = int(time.time() * 1000.0)
    dPs = []
    for n in range(net_trials):
        dP = hqrc.lyapunov_exp(qparams, buffer, length, nqrc=nqrc, \
            gamma=layer_strength, sparsity=sparsity, sigma_input=sigma_input, nonlinear=nonlinear,\
            ranseed=n, initial_distance=initial_distance)
        dPs.append(dP)

    mean_dp, std_dp = np.mean(dPs), np.std(dPs)
    
    rstr = '{} {} {} {} {} {}'.format(\
        nqrc, qparams.virtual_nodes, qparams.tau, layer_strength, mean_dp, std_dp)
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
    parser.add_argument('--rho', type=int, default=0)
    parser.add_argument('--beta', type=float, default=1e-14)
    parser.add_argument('--sparsity', type=float, default=1.0, help='The sparsity of the connection strength')
    parser.add_argument('--sigma_input', type=float, default=1.0, help='The sigma input for the feedback')
    parser.add_argument('--nonlinear', type=int, default=0, help='The nonlinear of feedback matrix')
    
    parser.add_argument('--length', type=int, default=1500)
    parser.add_argument('--buffer', type=int, default=1000)
    parser.add_argument('--solver', type=str, default=LINEAR_PINV, \
        help='regression solver by linear_pinv,ridge_pinv,auto,svd,cholesky,lsqr,sparse_cg,sag')
    parser.add_argument('--dynamic', type=str, default=DYNAMIC_FULL_RANDOM,\
        help='full_random,half_random,full_const_trans,full_const_coeff,ion_trap')


    parser.add_argument('--ntrials', type=int, default=1)
    parser.add_argument('--initial_distance', type=float, default=1e-8)

    parser.add_argument('--taudeltas', type=str, default='-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7')
    parser.add_argument('--layers', type=str, default='5')
    parser.add_argument('--strength', type=float, default=0.0)
    parser.add_argument('--virtuals', type=str, default='1')
    parser.add_argument('--interval', type=float, default=0.1, help='tau-interval')
    
    parser.add_argument('--savedir', type=str, default='res_high_lyp')
    parser.add_argument('--plot', type=int, default=1)
    args = parser.parse_args()
    print(args)

    n_units, max_energy, beta = args.units, args.coupling, args.beta
    length, buffer = args.length, args.buffer
    init_rho = args.rho
    net_trials, initial_distance = args.ntrials, args.initial_distance
    sparsity, sigma_input, nonlinear = args.sparsity, args.sigma_input, args.nonlinear
    dynamic, solver, savedir = args.dynamic, args.solver, args.savedir
    if os.path.isfile(savedir) == False and os.path.isdir(savedir) == False:
        os.mkdir(savedir)

    tstr = args.taudeltas.replace('\'','')
    #taudeltas = [float(x) for x in tstr.split(',')]
    #taudeltas = [2**x for x in taudeltas]
    tx = list(np.arange(-7, 7.1, args.interval))
    taudeltas = [2**x for x in tx]

    virtuals = [int(x) for x in args.virtuals.split(',')]
    layers = [int(x) for x in args.layers.split(',')]
    layer_strength = args.strength

    # Evaluation
    timestamp = int(time.time() * 1000.0)
    now = datetime.datetime.now()
    datestr = now.strftime('{0:%Y-%m-%d-%H-%M-%S}'.format(now))
    outfile = os.path.join(savedir, '{}_nqrc_{}_V_{}_a_{:.2f}_sm_{}_sigma_{}_sparse_{}_lyp_trials_{}_dist_{}_lyp.txt'.format(\
        dynamic, \
            '_'.join([str(l) for l in layers]), \
            '_'.join([str(v) for v in virtuals]), \
            layer_strength, nonlinear, sigma_input, sparsity, net_trials, initial_distance))
    
    if os.path.isfile(outfile) == False:
        jobs, pipels = [], []
        for nqrc in layers:
            for V in virtuals:
                for tau_delta in taudeltas:
                    recv_end, send_end = multiprocessing.Pipe(False)
                    qparams = QRCParams(n_units=n_units-1, n_envs=1, max_energy=max_energy,\
                        beta=beta, virtual_nodes=V, tau=tau_delta, init_rho=init_rho, solver=solver, dynamic=dynamic)
                    p = multiprocessing.Process(target=lyp_job, \
                        args=(qparams, nqrc, layer_strength, buffer, length, sparsity, sigma_input, nonlinear, net_trials, initial_distance, send_end))
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
        np.savetxt(outfile, rsarr, delimiter=' ')

        # save experiments setting
        with open('{}_setting.txt'.format(outbase), 'w') as sfile:
            sfile.write('length={}, buffer={}\n'.format(length, buffer))
            sfile.write('n_units={}\n'.format(n_units))
            sfile.write('max_energy={},nonlinear={},sigma_input={},sparsity={}\n'.format(max_energy, nonlinear, sigma_input, sparsity))
            sfile.write('beta={}\n'.format(beta))
            sfile.write('taudeltas={}\n'.format(' '.join([str(v) for v in taudeltas])))
            sfile.write('layers={}\n'.format(' '.join([str(l) for l in layers])))
            sfile.write('Vs={}\n'.format(' '.join([str(v) for v in virtuals])))
            sfile.write('strength={}\n'.format(layer_strength))
            sfile.write('net_trials={}, initial_distance={}\n'.format(net_trials, initial_distance))
