import sys
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
import plot_utils as putils
from collections import defaultdict
from matplotlib.colors import LogNorm, SymLogNorm

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
    
    parser.add_argument('--nspins', type=int, default=6, help='Number of spins')
    parser.add_argument('--max_energy', type=float, default=1.0)
    
    parser.add_argument('--prefix', type=str, default='ipc_capa')
    parser.add_argument('--T', type=int, default=200000, help='Number of time steps')
    parser.add_argument('--keystr', type=str, default='mdeg_4_mvar_4')
    parser.add_argument('--exp', type=int, default=0, help='Use exponent for alpha')
    parser.add_argument('--solver', type=str, default='', help='linear_pinv_,ridge_pinv_')

    parser.add_argument('--w_interval', type=float, default=0.05, help='interval for logW')
    parser.add_argument('--w_min', type=float, default=-2.0, help='minimum of parameter range for logW')
    parser.add_argument('--w_max', type=float, default=2.1, help='maximum of parameter range for logW')
    
    parser.add_argument('--g_interval', type=float, default=0.05, help='interval for gamma')
    parser.add_argument('--g_min', type=float, default=-2.0, help='minimum of parameter range for log_gamma')
    parser.add_argument('--g_max', type=float, default=2.1, help='maximum of parameter range for log_gamma')

    args = parser.parse_args()
    print(args)
    parent, folders, dynamic, prefix, keystr, thres, width = args.parent, args.folders, args.dynamic, args.prefix, args.keystr, args.thres, args.width
    V, tau, nspins, max_energy = args.virtuals, args.taus, args.nspins, args.max_energy
    T, nqrc, solver = args.T, args.nqrc, args.solver

    #ipc_capa_logW_-0.300_tau_10.0_V_1_hqrc_IPC_phase_trans_linear_pinv_nqrc_2_
    #nspins_6_seed_0_mdeg_4_mvar_4_thres_0.0_delays_0,100,50,50,20_T_2000_cb_1_gam_0.0_op_X_tp_5

    posfix  = 'tau_{}_V_{}_hqrc_IPC_{}_{}nqrc_{}_nspins_{}'.format(tau, V, dynamic, solver, nqrc, nspins)
    print(posfix)

    w_interval, w_min, w_max = args.w_interval, args.w_min, args.w_max
    g_interval, g_min, g_max = args.g_interval, args.g_min, args.g_max
    
    log_Ws =  list(np.arange(w_min, w_max, w_interval))
    log_Gs =  list(np.arange(g_min, g_max, g_interval))
    w_len, g_len = len(log_Ws), len(log_Gs)

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
        degcapa, mem_delays = [], []
        local_mem = defaultdict(list)
        for log_W in log_Ws:
            for log_g in log_Gs:
                tarr = []
                pattern1 = '{}/{}_log_g_{:.3f}_W_{:.3f}_{}*{}*T_{}*.pickle'.format(dfolder, prefix, log_g, log_W, posfix, keystr, T)
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
                    break
            
        for key in keys:
            local_mem[key] = np.array(local_mem[key]).reshape(w_len, g_len)
            local_mem_ls[key].append(local_mem[key])

        if len(degcapa) == 0:
            continue
        degcapa = np.array(degcapa).reshape(w_len, g_len, -1)
        mem_delays = np.array(mem_delays).reshape(w_len, g_len, -1)

        degcapa_ls.append(degcapa)
        mem_delays_ls.append(mem_delays)
    degcapa_ls = np.array(degcapa_ls)
    mem_delays_ls = np.array(mem_delays_ls)

    for key in keys:
        local_mem_ls[key] = np.array(local_mem_ls[key])
        print('Shape ', local_mem_ls[key].shape)
        avg_mem[key] = np.mean(local_mem_ls[key], axis=0)
        std_mem[key] = np.std(local_mem_ls[key], axis=0)
        

    degcapa_mean = np.mean(degcapa_ls, axis=0)
    degcapa_std  = np.std(degcapa_ls, axis=0)

    mem_delays_mean = np.mean(mem_delays_ls, axis=0)
    mem_delays_std  = np.std(mem_delays_ls, axis=0)
    
    print(degcapa_mean.shape)
    capa_tot = np.sum(degcapa_mean[:, :, :], axis=2)
    capa_1 = degcapa_mean[:, :, 1]
    capa_2 = degcapa_mean[:, :, 2]
    capa_3 = degcapa_mean[:, :, 3]
    capa_4 = degcapa_mean[:, :, 4]
    capa_5 = np.sum(degcapa_mean[:, :, 5:], axis=2)

    fig = plt.figure(figsize=(36, 16), dpi=600)
    cmap = plt.get_cmap('viridis')
    putils.setPlot(fontsize=20, labelsize=24)

    ax0 = plt.subplot2grid((2,3), (0,0), colspan=1, rowspan=1)
    ax0.set_title('THRES_{}_tot_capa'.format(thres), size=10)

    ax1 = plt.subplot2grid((2,3), (0,1), colspan=1, rowspan=1)
    ax1.set_title('THRES_{}_capa_1'.format(thres), size=10)

    ax2 = plt.subplot2grid((2,3), (0,2), colspan=1, rowspan=1)
    ax2.set_title('THRES_{}_capa_2'.format(thres), size=10)

    ax3 = plt.subplot2grid((2,3), (1,0), colspan=1, rowspan=1)
    ax3.set_title('THRES_{}_capa_3'.format(thres), size=10)

    ax4 = plt.subplot2grid((2,3), (1,1), colspan=1, rowspan=1)
    ax4.set_title('THRES_{}_capa_4'.format(thres), size=10)

    ax5 = plt.subplot2grid((2,3), (1,2), colspan=1, rowspan=1)
    ax5.set_title('THRES_{}_capa_rest'.format(thres), size=10)

    log_Ws, log_Gs = np.array(log_Ws), np.array(log_Gs)
    log_Ws   = log_Ws[log_Ws <= 2.0001]
    log_Gs   = log_Gs[log_Gs <= 2.0001]

    for ax in [ax0, ax1, ax2, ax3, ax4, ax5]:
        ax.set_xlabel('(log) Feedback strength')
        ax.set_ylabel('(log) Disorder strength')
        ax.set_yticks(np.linspace(log_Ws[0], log_Ws[-1], 11))
        ax.set_xticks(np.linspace(log_Gs[0], log_Gs[-1], 11))
    dx, = np.diff([log_Gs[0],log_Gs[-1]])/(capa_1.shape[1]-1)
    dy, = -np.diff([log_Ws[0], log_Ws[-1]])/(capa_1.shape[0]-1)
    extent = [log_Gs[0]-dx/2, log_Gs[-1]+dx/2, log_Ws[0]+dy/2, log_Ws[-1]-dy/2]

    im0 = ax0.imshow(capa_tot, origin='lower', extent=extent, aspect='auto', interpolation='nearest', cmap="Spectral")
    im1 = ax1.imshow(capa_1, origin='lower', extent=extent, aspect='auto', interpolation='nearest', cmap="viridis")
    im2 = ax2.imshow(capa_2, origin='lower', extent=extent, aspect='auto', interpolation='nearest', cmap="cividis")
    im3 = ax3.imshow(capa_3, origin='lower', extent=extent, aspect='auto', interpolation='nearest', cmap="plasma")
    im4 = ax4.imshow(capa_4, origin='lower', extent=extent, aspect='auto', interpolation='nearest', cmap="inferno")
    im5 = ax5.imshow(capa_5, origin='lower', extent=extent, aspect='auto', interpolation='nearest', cmap="magma")
    
    fig.colorbar(im0, ax=ax0)
    fig.colorbar(im1, ax=ax1)
    fig.colorbar(im2, ax=ax2)
    fig.colorbar(im3, ax=ax3)
    fig.colorbar(im4, ax=ax4)
    fig.colorbar(im5, ax=ax5)

    plt.tight_layout()

    fig_folder = os.path.join(folder, 'figs')
    if os.path.isdir(fig_folder) == False:
        os.mkdir(fig_folder)
    outbase = os.path.join(fig_folder, 'heat_thres_{}_{}_T_{}'.format(thres, posfix, T))
    for ftype in ['png', 'svg']:
        plt.savefig('{}.{}'.format(outbase, ftype), bbox_inches='tight', dpi=600)
    plt.show()



