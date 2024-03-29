import sys
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
import plot_utils as putils

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=str, required=True)
    parser.add_argument('--folders', type=str, required=True)
    parser.add_argument('--dynamic', type=str, default='full_random')
    parser.add_argument('--virtuals', type=str, default='1', help='Number of virtual nodes')
    parser.add_argument('--taus', type=str, default='8.0', help='Taus')
    parser.add_argument('--nqrc', type=int, default=5)

    parser.add_argument('--thres', type=float, default=0.0)
    parser.add_argument('--width', type=float, default=0.1)
    parser.add_argument('--max_capa', type=float, default=4.0)
    
    parser.add_argument('--nspins', type=int, default=5, help='Number of spins')
    parser.add_argument('--max_energy', type=float, default=1.0)
    parser.add_argument('--amax', type=float, default=1.0, help='Maximum of alpha')
    parser.add_argument('--amin', type=float, default=0.0, help='Minimum of alpha')
    parser.add_argument('--nas', type=int, default=100, help='Number of alpha')
    
    parser.add_argument('--prefix', type=str, default='ipc_capa')
    parser.add_argument('--T', type=int, default=200000, help='Number of time steps')
    parser.add_argument('--keystr', type=str, default='mdeg_4_mvar_4')
    parser.add_argument('--exp', type=int, default=0, help='Use exponent for alpha')
    parser.add_argument('--solver', type=str, default='', help='linear_pinv_,ridge_pinv_')
 
    args = parser.parse_args()
    print(args)
    parent, folders, dynamic, prefix, keystr, thres, width = args.parent, args.folders, args.dynamic, args.prefix, args.keystr, args.thres, args.width
    V, tau, nspins, max_energy, amin, amax, nas = args.virtuals, args.taus, args.nspins, args.max_energy, args.amin, args.amax, args.nas
    T, nqrc, explb, solver = args.T, args.nqrc, args.exp, args.solver

    posfix  = 'tau_{}_V_{}_hqrc_IPC_{}_{}nqrc_{}_nspins_{}_amax_{}_amin_{}_nas_{}'.format(\
        tau, V, dynamic, solver, nqrc, nspins, amax, amin, nas)
    if explb > 0:
        posfix  = 'tau_{}_V_{}_hqrc_IPC_exp_{}_{}_{}nqrc_{}_nspins_{}_amax_{}_amin_{}_nas_{}'.format(\
        tau, V, explb, dynamic, solver, nqrc, nspins, amax, amin, nas)
    print(posfix)
    txBs = list(np.linspace(amin, amax, nas + 1))

    folders = [str(x) for x in folders.split(',')]
    degcapa_ls = []
    for folder in folders:
        folder = os.path.join(parent, folder)
        print('Folder={}'.format(folder))
        if os.path.isdir(folder) == False:
            continue
        dfolder = os.path.join(folder, 'ipc')
        degcapa, xs = [], []
        for tB in txBs:
            tarr = []
            pattern1 = '{}/{}_alpha_{:.3f}_{}*{}*T_{}*.pickle'.format(dfolder, prefix, tB, posfix, keystr, T)
            #if explb > 0:
            pattern2 = '{}/{}_alpha_{}_{}*{}*T_{}*.pickle'.format(dfolder, prefix, tB, posfix, keystr, T)
            filenames = glob.glob(pattern1)
            filenames.extend(glob.glob(pattern2))
            #print(pattern1)
            for filename in filenames:
                #print(filename)
                with open(filename, "rb") as rfile:
                    data = pickle.load(rfile)
                    ipc_arr = data['ipc_arr']
                    for deg in sorted(ipc_arr.keys()):
                        darr = ipc_arr[deg].ravel()
                        tarr.append( np.sum(darr[darr >= thres]) )    
                #print(deg_arr.shape)
                degcapa.append(np.array(tarr).ravel())
                if explb > 0:
                    xs.append(1.0-10**tB)
                else:
                    xs.append(tB)
                break
        if len(degcapa) == 0:
            continue
        degcapa = np.array(degcapa).T
        degcapa_ls.append(degcapa)
    
    degcapa_ls = np.array(degcapa_ls)
    print(degcapa_ls.shape, len(xs))
    degcapa_mean = np.mean(degcapa_ls, axis=0)
    degcapa_std  = np.std(degcapa_ls, axis=0)

    print(degcapa_mean.shape)

    sum_by_cols = np.sum(degcapa_mean, axis=0)
    
    fig = plt.figure(figsize=(16, 10), dpi=600)
    cmap = plt.get_cmap('nipy_spectral')
    putils.setPlot(fontsize=20, labelsize=24)
    colors = putils.cycle
    d_colors = putils.d_colors

    N = min(len(d_colors), degcapa_mean.shape[0])

    ax1 = plt.subplot2grid((1,2), (0,0), colspan=1, rowspan=1)
    #ax1.set_title('THRES_{}_{}'.format(thres, posfix), size=14)

    ax2 = plt.subplot2grid((1,2), (0,1), colspan=1, rowspan=1)
    #ax2.set_title('THRES_{}_{}'.format(thres, posfix), size=14)

    ax1.bar(xs, degcapa_mean[0], width=width, color=d_colors[0], edgecolor='gray', label='deg-0')
    #ax2.bar(xs, degcapa[0] / sum_by_cols,  width=width, color=d_colors[0], edgecolor='k', label='deg-0')
    for i in range(1, N):
        bt = degcapa_mean[:i].reshape(i, -1)
        bt = np.sum(bt, axis=0).ravel()
        ax1.bar(xs, degcapa_mean[i], bottom=bt, width=width, label='deg-{}'.format(i), color=d_colors[i], edgecolor='gray')
        #ax2.bar(xs, degcapa[i] / sum_by_cols, bottom=bt/sum_by_cols, width=width, label='deg-{}'.format(i), color=d_colors[i], edgecolor='k')
    m_avg, m_std = dict(), dict()
    m_avg[0], m_std[0] = degcapa_mean[1], degcapa_std[1]
    m_avg[1], m_std[1] = np.sum(degcapa_mean[2:], axis=0), np.sum(degcapa_std[2:], axis=0)
    #m_avg[2], m_std[2] = degcapa_mean[3] + degcapa_mean[4], degcapa_std[3] + degcapa_std[4]
    
    for i in range(2):
        #xs = 1.0 - np.array(xs)
        ax2.plot(xs, m_avg[i], 's-',linewidth=3, markersize=0, label='deg-{}'.format(i+1), color=colors[i], alpha=0.8)
        ax2.fill_between(xs, m_avg[i] - m_std[i], m_avg[i] + m_std[i], facecolor=colors[i], alpha=0.2)
    
    ax1.set_xlabel('$\\alpha$', size=32)
    ax1.set_ylabel('Total IPC', fontsize=28)
    ax1.set_xlim([amin, amax])
    ax1.set_xticks(np.linspace(amin, amax, 6))
    #ax1.set_xticks(list(range(int(amin), int(amax)+1)))

    ax2.set_xlabel('$\\alpha$', size=32)
    ax2.set_ylabel('$C(d)$', size=32)
    ax2.set_xlim([amin, amax])
    #ax2.set_xscale('log',basex=10)
    #ax2.set_ylim([0.0, np.max(m_avg[0]+m_std[0])])
    ax2.set_ylim([0.0, args.max_capa])
    ax2.set_xticks(np.linspace(amin, amax, 6))
    
    #ax1.set_ylim([0, 4.0])
    #ax1.set_xscale('log', basex=2)
    ax1.axhline(y=args.max_capa, color='k', linestyle='-')

    ax1.legend()
    ax2.legend()
    
    for ax in [ax1, ax2]:
        #ax.minorticks_on()
        ax.tick_params('both', length=8, width=1, which='major', labelsize=28)
        #ax.tick_params('both', length=4, width=1, which='minor')

    plt.tight_layout()

    fig_folder = os.path.join(folder, 'figs')
    if os.path.isdir(fig_folder) == False:
        os.mkdir(fig_folder)
    outbase = os.path.join(fig_folder, 'fig_thres_{}_{}'.format(thres, posfix))
    for ftype in ['png', 'svg']:
        plt.savefig('{}.{}'.format(outbase, ftype), bbox_inches='tight', dpi=600)
    plt.show()



