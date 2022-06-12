import sys
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
import plot_utils as putils
from collections import defaultdict

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=str, required=True)
    parser.add_argument('--folders', type=str, required=True)
    parser.add_argument('--dynamic', type=str, default='phase_trans')
    parser.add_argument('--virtuals', type=str, default='1', help='Number of virtual nodes')
    parser.add_argument('--taus', type=str, default='10.0', help='Taus')
    parser.add_argument('--nqrc', type=int, default=5)

    parser.add_argument('--thres', type=float, default=0.0)
    parser.add_argument('--width', type=float, default=0.1)
    parser.add_argument('--max_capa', type=float, default=4.0)
    parser.add_argument('--max_mc', type=float, default=2.0)
    
    parser.add_argument('--nspins', type=int, default=6, help='Number of spins')
    parser.add_argument('--max_energy', type=float, default=1.0)
    
    parser.add_argument('--prefix', type=str, default='ipc_capa')
    parser.add_argument('--T', type=int, default=200000, help='Number of time steps')
    parser.add_argument('--keystr', type=str, default='mdeg_4_mvar_4')
    parser.add_argument('--exp', type=int, default=0, help='Use exponent for alpha')
    parser.add_argument('--solver', type=str, default='', help='linear_pinv_,ridge_pinv_')
    parser.add_argument('--interval', type=float, default=0.05, help='logW-interval')

    args = parser.parse_args()
    print(args)
    parent, folders, dynamic, prefix, keystr, thres, width = args.parent, args.folders, args.dynamic, args.prefix, args.keystr, args.thres, args.width
    V, tau, nspins, max_energy = args.virtuals, args.taus, args.nspins, args.max_energy
    T, nqrc, solver = args.T, args.nqrc, args.solver

    #ipc_capa_logW_-0.300_tau_10.0_V_1_hqrc_IPC_phase_trans_linear_pinv_nqrc_2_
    #nspins_6_seed_0_mdeg_4_mvar_4_thres_0.0_delays_0,100,50,50,20_T_2000_cb_1_gam_0.0_op_X_tp_5

    posfix  = 'tau_{}_V_{}_hqrc_IPC_{}_{}nqrc_{}_nspins_{}'.format(tau, V, dynamic, solver, nqrc, nspins)
    print(posfix)
    log_Ws =  list(np.arange(-2.0, 2.1, args.interval))

    folders = [str(x) for x in folders.split(',')]
    degcapa_ls, mem_delays_ls = [], []
    local_mem_ls = defaultdict(list)
    avg_mem, std_mem = defaultdict(list), defaultdict(list)
    key1 = 'delay=0'
    key2 = 'delay=1'
    key3 = '2<=delay<=10'
    key4 = '11<=delay<=100'
    keys = [key1, key2, key3, key4]

    for folder in folders:
        folder = os.path.join(parent, folder)
        print('Folder={}'.format(folder))
        if os.path.isdir(folder) == False:
            continue
        dfolder = os.path.join(folder, 'ipc')
        degcapa, mem_delays, xs = [], [], []
        local_mem = defaultdict(list)
        for logW in log_Ws:
            tarr = []
            pattern1 = '{}/{}_log*_{:.3f}_{}*{}*T_{}*.pickle'.format(dfolder, prefix, logW, posfix, keystr, T)
            filenames = glob.glob(pattern1)
            #print(pattern1)
            for filename in filenames:
                #print(filename)
                with open(filename, "rb") as rfile:
                    data = pickle.load(rfile)
                    ipc_arr = data['ipc_arr']
                    for deg in sorted(ipc_arr.keys()):
                        darr = ipc_arr[deg].ravel()
                        darr[darr < thres] = 0.0
                        tarr.append( np.sum(darr) )
                        if deg == 1:
                            mem_delays.append(darr)
                            local_mem[key1].append(np.sum(darr[0]))
                            local_mem[key2].append(np.sum(darr[1]))
                            local_mem[key3].append(np.sum(darr[2:11]))
                            local_mem[key4].append(np.sum(darr[11:]))
                #print(deg_arr.shape)
                degcapa.append(np.array(tarr).ravel())
                xs.append(10**logW)
                break
            
        for key in keys:
            local_mem[key] = np.array(local_mem[key])
            local_mem_ls[key].append(local_mem[key])

        if len(degcapa) == 0:
            continue
        degcapa = np.array(degcapa).T
        mem_delays = np.array(mem_delays).T

        degcapa_ls.append(degcapa)
        mem_delays_ls.append(mem_delays)
    degcapa_ls = np.array(degcapa_ls)
    mem_delays_ls = np.array(mem_delays_ls)

    for key in keys:
        local_mem_ls[key] = np.array(local_mem_ls[key])
        print('Shape ', local_mem_ls[key].shape)
        avg_mem[key] = np.mean(local_mem_ls[key], axis=0)
        std_mem[key] = np.std(local_mem_ls[key], axis=0)
        

    print(degcapa_ls.shape, len(xs))
    degcapa_mean = np.mean(degcapa_ls, axis=0)
    degcapa_std  = np.std(degcapa_ls, axis=0)

    mem_delays_mean = np.mean(mem_delays_ls, axis=0)
    mem_delays_std  = np.std(mem_delays_ls, axis=0)
    

    print(degcapa_mean.shape)

    sum_by_cols = np.sum(degcapa_mean, axis=0)
    
    fig = plt.figure(figsize=(20, 16), dpi=600)
    cmap = plt.get_cmap('viridis')
    putils.setPlot(fontsize=20, labelsize=24)
    colors = putils.cycle
    d_colors = putils.d_colors

    N = min(len(d_colors), degcapa_mean.shape[0])

    ax1 = plt.subplot2grid((2,1), (0,0), colspan=1, rowspan=1)
    #ax1.set_title('THRES_{}_{}'.format(thres, posfix), size=14)

    ax2 = plt.subplot2grid((2,1), (1,0), colspan=1, rowspan=1)
    #ax2.set_title('THRES_{}_{}'.format(thres, posfix), size=14)

    logx = np.logspace(log_Ws[0], log_Ws[-1], num=len(log_Ws)+1)

    #ax1.bar(xs, degcapa_mean[0], width=width, color=d_colors[0], edgecolor='gray', label='deg-0')
    #ax2.bar(xs, degcapa[0] / sum_by_cols,  width=width, color=d_colors[0], edgecolor='k', label='deg-0')

    for i in range(1, N):
        bt = degcapa_mean[:i].reshape(i, -1)
        bt = np.sum(bt, axis=0).ravel()
        ax1.bar(xs, degcapa_mean[i], bottom=bt, width=np.diff(logx), label='deg-{}'.format(i), color=d_colors[i], edgecolor='gray')
        #ax2.bar(xs, degcapa[i] / sum_by_cols, bottom=bt/sum_by_cols, width=width, label='deg-{}'.format(i), color=d_colors[i], edgecolor='k')
    m_avg, m_std = dict(), dict()
    m_avg[0], m_std[0] = degcapa_mean[1], degcapa_std[1]
    m_avg[1], m_std[1] = np.sum(degcapa_mean[2:], axis=0), np.sum(degcapa_std[2:], axis=0)
    #m_avg[2], m_std[2] = degcapa_mean[3] + degcapa_mean[4], degcapa_std[3] + degcapa_std[4]
    
    # for i in range(2):
    #     #xs = 1.0 - np.array(xs)
    #     ax2.plot(xs, m_avg[i], 's-',linewidth=3, markersize=0, label='deg-{}'.format(i+1), color=colors[i], alpha=0.8)
    #     ax2.fill_between(xs, m_avg[i] - m_std[i], m_avg[i] + m_std[i], facecolor=colors[i], alpha=0.2)
    
    # ax1.set_xlabel('$log(W)$', size=32)
    # ax1.set_ylabel('Total IPC', fontsize=28)
    
    # ax2.set_xlabel('$log(W)$', size=32)
    # ax2.set_ylabel('$C(d)$', size=32)
    # #ax2.set_xscale('log',basex=10)
    # #ax2.set_ylim([0.0, np.max(m_avg[0]+m_std[0])])
    # ax2.set_ylim([0.0, args.max_capa])
    # #ax2.set_xticks(np.linspace(amin, amax, 6))
    # ax2.legend()

    #ax1.set_ylim([0, 4.0])
    #ax1.set_xscale('log', basex=2)
    ax1.axhline(y=args.max_capa, color='k', linestyle='-')
    ax1.set_ylabel('Total IPC', fontsize=28)
    ax1.grid(True, axis='y', which="both", ls="-", color='0.65')
    
    colors = putils.cycle
    dcl = 0
    for key in keys:
        color = colors[dcl]
        ax2.plot(xs, avg_mem[key], 's-', alpha = 0.8, linewidth=3, markersize=10, mec='k', mew=0.5, \
            color=color, label=key)
        ax2.fill_between(xs, avg_mem[key] - std_mem[key], avg_mem[key] + std_mem[key], \
            facecolor=color, alpha=0.2)
        dcl += 1
    ax2.set_ylabel('Degree 1', fontsize=32)
    ax2.grid(True, which="both", ls="-", color='0.65')
    ax2.set_ylim([0, args.max_mc])
    for ax in [ax1, ax2]:
        ax.set_xscale('log')
        ax.set_xlabel('Param', fontsize=28)
        ax.tick_params('both', length=12, width=1, which='major', labelsize=28)
        ax.tick_params('both', length=8, width=1, which='minor')

        ax.set_xlim([xs[0], xs[-1]])
        ax.legend()

    plt.tight_layout()

    fig_folder = os.path.join(folder, 'figs')
    if os.path.isdir(fig_folder) == False:
        os.mkdir(fig_folder)
    outbase = os.path.join(fig_folder, 'fig_thres_{}_{}_T_{}'.format(thres, posfix, T))
    for ftype in ['png', 'svg']:
        plt.savefig('{}.{}'.format(outbase, ftype), bbox_inches='tight', dpi=600)
    plt.show()



