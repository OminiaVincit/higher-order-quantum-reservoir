#!/usr/bin/env python
"""
    View bifurcation diagrams and dynamics for higher-order quantum reservoir
    See run_view_states.sh for an example of the running script
"""
import sys
import numpy as np
import os
import scipy
import argparse
import multiprocessing
import matplotlib
#matplotlib.use("cairo")
import matplotlib.pyplot as plt
from matplotlib import ticker
import time
import datetime
import hqrc as hqrc
import utils
from utils import *
from loginit import get_module_logger
import pickle
import umap
import umap.plot
from scipy import sparse
from sklearn.metrics.pairwise import pairwise_distances
import nolds # to calculate lyapunov eff
from collections import defaultdict

UNITS=5
BETA=1e-14
INIT_RHO=0
V=1
INTERVAL=0.05

def makeSparseDM(X, thresh):
    N = X.shape[0]
    D = pairwise_distances(X, metric='euclidean')
    [I, J] = np.meshgrid(np.arange(N), np.arange(N))
    I = I[D <= thresh]
    J = J[D <= thresh]
    V = D[D <= thresh]
    return sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()

def reduce_states_dimension(arr, n_neighbors=15, min_dist=0.1, n_components=2, norm=False):
    D = makeSparseDM(arr, thresh=100)
    if norm:
        D = D/np.max(D)

    fit = umap.UMAP(
        random_state=42,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components
    )
    mapper = fit.fit(D)
    return mapper


def dumpstates_job(savedir, dynamic, input_seq, nqrc, layer_strength, mask_input, combine_input, \
    nonlinear, sigma_input, type_input, sparsity, xs, idx, bg, ed, send_end):
    """
    Dump raw data of states
    """
    print('Start pid={} with size {} (from {} to {})'.format(idx, len(xs), xs[0], xs[-1]))
    results = dict()
    basename = '{}_nqr_{}_V_{}_sm_{}_alpha_{}_sg_{}_sparse_{}_ms_{}_cb_{}_tp_{}'.format(dynamic, \
        nqrc, V, nonlinear, layer_strength, sigma_input, sparsity, mask_input, combine_input, type_input)
    save_figdir = os.path.join(savedir, 'figs')
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(save_figdir, exist_ok=True)
    bg1 = int(max(bg/2 - 100, bg/4))
    ed1 = int(min(bg/2 + 100, ed))
    for x in xs:
        tau = 2**x
        qparams = QRCParams(n_units=UNITS-1, n_envs=1, max_energy=1.0,\
            beta=BETA, virtual_nodes=V, tau=tau, init_rho=INIT_RHO, solver=LINEAR_PINV, dynamic=dynamic)
        model = hqrc.HQRC(nqrc=nqrc, gamma=layer_strength, sparsity=sparsity, sigma_input=sigma_input, \
            mask_input=mask_input, combine_input=combine_input, nonlinear=nonlinear, type_input=type_input, feed_trials = int(bg/2))
        state_list, feed_list = model.init_forward(qparams, input_seq, init_rs = True, ranseed = 0)
        state_list = state_list*2.0-1.0 
        results[x] = state_list

        # Draw state and feedback
        cmap = plt.get_cmap("viridis")
        plt.style.use('seaborn-colorblind')
        plt.rc('font', family='serif')
        plt.rc('mathtext', fontset='cm')
        plt.rcParams['font.size'] = 12
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
        fig, axs = plt.subplots(nqrc, 2, figsize=(18, 3*nqrc), squeeze=False)

        n_local_nodes = int(state_list.shape[1] / nqrc)
        xs = list(range(bg1, ed1))
        vmin1, vmax1 = np.amin(input_seq[:,bg1:ed1]), np.amax(input_seq[:, bg1:ed1])
        vmin2, vmax2 = np.amin(state_list[bg1:ed1, :]), np.amax(state_list[bg1:ed1, :])
        if len(feed_list) > 0:
            vmin1 = min(vmin1, np.amin(feed_list[bg1:ed1, :]))
            vmax1 = max(vmax1, np.amax(feed_list[bg1:ed1, :]))
        vmin1, vmax1 = 0.0, 1.0

        outfile = 'tau_{:.3f}_{}'.format(tau, basename)
        for i in range(nqrc):
            ax1, ax2 = axs[i, 0], axs[i, 1]
            ax1.plot(xs, input_seq[i, bg1:ed1], c='gray', label='Input')
            ax1.plot(xs, (1.0-layer_strength)*input_seq[i, bg1:ed1], c='b', label='Scale-in', alpha=0.5)
            
            #print('Feedback list', feed_list[-1])
            if len(feed_list) > 0:
                ax1.plot(xs, feed_list[bg1:ed1, i], c='k', label='Feedback', linestyle='dashed')
                combine_input_seq =  input_seq[i, bg1:ed1] * (1.0-layer_strength) + feed_list[bg1:ed1, i] * layer_strength
                ax1.plot(xs, combine_input_seq, c='r', label='Combine', linestyle='dashed', alpha=0.8)
                
            for j in range(n_local_nodes):
                ax2.plot(xs, state_list[bg1:ed1, i*n_local_nodes + j], c=colors[j], label='QR{}-{}'.format(i+1,j+1))
            ax1.legend()
            ax2.legend()
            if i == 0:
                ax2.set_title('{}'.format(outfile))
            ax1.set_title('Min={:.3f}, Max={:.3f}, Steps trials={}'.format(model.feed_min[i], model.feed_max[i], model.feed_trials))
                
            ax1.set_ylim([vmin1, vmax1])
            ax2.set_ylim([vmin2, vmax2])
        for ftype in ['png']:
            figfile = os.path.join(save_figdir, '{}.{}'.format(outfile, ftype))
            plt.savefig(figfile, bbox_inches='tight')

    outbase = os.path.join(savedir, basename)
    filename = '{}_states_id_{}.binaryfile'.format(outbase, idx)
    with open(filename, 'wb') as wrs:
        pickle.dump(results, wrs)
    send_end.send(filename)
    print('Finish pid={} with size {} (from {} to {})'.format(idx, len(xs), xs[0], xs[-1]))

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--length', type=int, default=2000)
    parser.add_argument('--bg', type=int, default=1000, help='start index to view dynamics')
    parser.add_argument('--ed', type=int, default=2000, help='end index to view dynamics')
    parser.add_argument('--const', type=int, default=0, help='flag to set constant input')
    parser.add_argument('--nqrc', type=int, default=5, help='Number of reservoirs')
    parser.add_argument('--strength', type=float, default=0.5, help='The connection strength')
    parser.add_argument('--sparsity', type=float, default=1.0, help='The sparsity of the connection strength')
    parser.add_argument('--sigma_input', type=float, default=1.0, help='The sigma input for the feedback')
    parser.add_argument('--mask_input', type=int, default=0, help='Mask input')
    parser.add_argument('--combine_input', type=int, default=0, help='Combine input')
    parser.add_argument('--type_input', type=int, default=0)
    parser.add_argument('--nonlinear', type=int, default=0, help='The nonlinear of feedback matrix')
    parser.add_argument('--scale_input', type=float, default=1.0)
    parser.add_argument('--trans_input', type=float, default=0.0)
    
    parser.add_argument('--nproc', type=int, default=50)
    parser.add_argument('--dynamic', type=str, default=DYNAMIC_FULL_RANDOM,\
        help='full_random,half_random,full_const_trans,full_const_coeff,ion_trap')

    parser.add_argument('--interval', type=float, default=INTERVAL, help='tau-interval')
    parser.add_argument('--savedir', type=str, default='res_states')
    parser.add_argument('--lyap', type=int, default=0)
    
    args = parser.parse_args()
    print(args)

    length, nqrc, nproc, dynamic = args.length, args.nqrc, args.nproc, args.dynamic
    bg, ed, lyap = args.bg, args.ed, args.lyap
    layer_strength, nonlinear, sparsity, sigma_input = args.strength, args.nonlinear, args.sparsity, args.sigma_input
    const_input, mask_input, combine_input, type_input = args.const, args.mask_input, args.combine_input, args.type_input
    scale_input, trans_input = args.scale_input, args.trans_input
    basename = '{}_nqr_{}_V_{}_sm_{}_a_{}_sg_{}_sparse_{}_ms_{}_cb_{}_tp_{}'.format(dynamic, \
        nqrc, V, nonlinear, layer_strength, sigma_input, sparsity, mask_input, combine_input, type_input)

    savedir = args.savedir
    if os.path.isfile(savedir) == False and os.path.isdir(savedir) == False:
        os.mkdir(savedir)
    
    # KEEP CONSTANT interval = 0.05
    tx = list(np.arange(-7, 7.1, args.interval))
    stx = [2**x for x in tx]
    nproc = min(len(tx), nproc)
    lst = np.array_split(tx, nproc)

    if os.path.isfile(savedir) == False:
        # prepare the data
        if const_input == 0:
            np.random.seed(seed=1000)
            data = np.random.rand(length)
        else:
            data = np.zeros(length)
        data = data * scale_input
        input_seq = np.array(data)
        input_seq = np.tile(input_seq, (nqrc, 1))
        
        jobs, pipels = [], []
        for pid in range(nproc):
            xs = lst[pid]
            recv_end, send_end = multiprocessing.Pipe(False)
            p = multiprocessing.Process(target=dumpstates_job, args=(savedir, dynamic, input_seq, \
                nqrc, layer_strength, mask_input, combine_input, nonlinear, sigma_input, type_input,\
                sparsity, xs, pid, bg, ed, send_end))

            jobs.append(p)
            pipels.append(recv_end)
        # Start the process
        for p in jobs:
            p.start()

        # Ensure all processes have finished execution
        for p in jobs:
            p.join()
        
        # Join dumpled pickle files
        z = dict()
        for px in pipels:
            filename = px.recv()
            with open(filename, 'rb') as rrs:
                tmp = pickle.load(rrs)
                z = dict(list(z.items()) + list(tmp.items()))
            # Delete file
            os.remove(filename)
            print('zlen={}, Deleted {}'.format(len(z), filename))

        filename = filename.replace('.binaryfile', 'const_{}_scale_{}_trans_{}_len_{}.binaryfile'.format(const_input, scale_input, trans_input, length))
        with open(filename, 'wb') as wrs:
            pickle.dump(z, wrs)
    else:
        filename = savedir
        with open(filename, 'rb') as rrs:
            z = pickle.load(rrs)
    
    # Plot file
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams["font.size"] = 8 # 全体のフォントサイズが変更されます
    plt.rcParams['xtick.labelsize'] = 8 # 軸だけ変更されます
    plt.rcParams['ytick.labelsize'] = 8 # 軸だけ変更されます

    #fig, axs = plt.subplots(1, 1, figsize=(8, 6), squeeze=False, dpi=600)
    #ax1 = axs.ravel()[0]
    #ax.plot(ts, rs, ls="", marker=",")

    fig = plt.figure(figsize=(16, 16), dpi=600)
    for i in range(nqrc):
        ax1 = plt.subplot2grid((nqrc,4), (i,0), colspan=2, rowspan=1)
        sbg = 1 + i*UNITS
        sed = (i+1)*UNITS
        
        if lyap > 0:
            # # Very slow to run density plot
            # xy = np.vstack([ts, rs])
            # z = gaussian_kde(xy)(xy)
            # ax1.scatter(ts, rs, c=z, s=(12*72./fig.dpi)**2, marker='o', cmap='brg', lw=0, rasterized=True)
            rs_units = defaultdict(list)
            for x in tx:
                state_list = z[x]
                for k in range(1, 2):
                    ys = state_list[bg:ed, sbg+k].ravel()
                    # Calculate maximum lyapunov
                    lypval = nolds.lyap_r(ys, tau=1)
                    rs_units[k].append(lypval)
            
            for k in range(1, 2):
                #print(rs_units[k], len(rs_units[k]), np.min(rs_units[k]), np.max(rs_units[k]))
                ax1.plot(stx, rs_units[k], linewidth=2)
        else:
            rs, ts = [], []
            for x in tx:
                state_list = z[x]
                ys = state_list[bg:ed, sbg:sed].ravel()
                rs.extend(ys)
                ts.extend([2**x] * len(ys))
            ts = np.array(ts).ravel()
            rs = np.array(rs).ravel()

            ax1.scatter(ts, rs, s=(12*72./fig.dpi)**2, marker='o', lw=0, rasterized=True)
            ax1.set_yscale("symlog", base=10, linthresh=1e-5)
        
        # Calculate maximum lyapunov exponent
        lya_maxs = []
        for x in tx:
            state_list = z[x]
        

        if i == 0:
            ax1.set_title('{}_QR_{}'.format(os.path.basename(filename), i+1), fontsize=8)
        ax1.set_xscale("log", base=2)
        
        ax1.grid(alpha=0.8, axis='x')
        ax1.set_xticks([2**x for x in np.arange(-7,7.1,1.0)])
        ax1.minorticks_on()
        ax1.tick_params('both', length=6, width=1, which='major')
        ax1.tick_params('both', length=3, width=1, which='minor')
        ax1.set_xlim([2**tx[0], 2**tx[-1]])

    ids = [20, 60, 80, 100, 180]
    N = ed - bg
    colors = plt.cm.viridis(np.linspace(0, 1, N))
    for i in range(len(ids)):
        ax2 = plt.subplot2grid((nqrc,4), (i,2))
        x = tx[ids[i]]
        state_list = z[x]
        for j in range(1, UNITS):
            ys = state_list[bg:ed, j].ravel()
            ax2.plot(ys)
        ax2.set_title('2^{:.1f}'.format(x))
        #ax2.set_yticklabels([])
        #ax2.set_xticklabels([])

        ax3 = plt.subplot2grid((nqrc,4), (i,3))
        states = state_list[bg:ed, :UNITS]
        mapper = reduce_states_dimension(states)
        umap.plot.points(mapper, ax=ax3, theme='fire')

        

    outbase = filename.replace('.binaryfile', '_bg_{}_ed_{}_lyap_{}'.format(bg, ed, lyap))
    for ftype in ['png']:
        plt.savefig('{}_v4.{}'.format(outbase, ftype), bbox_inches='tight', dpi=600)
    plt.show()
