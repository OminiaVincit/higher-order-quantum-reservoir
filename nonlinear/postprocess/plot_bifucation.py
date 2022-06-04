import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib.gridspec as gridspec


if __name__  == '__main__':
        # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='../../../data/hqrc/dynamics_sinwave')
    parser.add_argument('--prefix', type=str, default='phase_trans_nqr_5_V_1_tau_10.0_nondiag_2.0')
    parser.add_argument('--posfix', type=str, default='op_X_tp_5_interval_0.01_rtd_0_states_id_99_len_10000')
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--bg', type=int, default=9000)
    parser.add_argument('--ed', type=int, default=10000)
    parser.add_argument('--nspins', type=int, default=6)

    args = parser.parse_args()
    print(args)
    folder, prefix, posfix, gamma = args.folder, args.prefix, args.posfix, args.gamma
    bg, ed, Nspins = args.bg, args.ed, args.nspins
    
    filename = '{}_gam_{}_{}'.format(prefix, gamma, posfix)
    with open('{}/{}.binaryfile'.format(folder, filename), 'rb') as rrs:
        z = pickle.load(rrs)
    
        plt.rcParams['font.size'] = 14
        fig = plt.figure(figsize=(8, 16))
        #Nspins = list(z.values())[0].shape[1]
        gs = gridspec.GridSpec(Nspins, 1, wspace=0.4, hspace=0.4)
        axs = [plt.subplot(gs[i, 0]) for i in range(Nspins)]
        for i in range(Nspins):
            ax = axs[i]
            ax.set_xlabel('W')
            ax.set_ylabel('Spin {}'.format(i+1))
            ax.set_xlim(10**(-2.02), 10**2.1)
            ax.set_ylim(-1.1, 1.1)
            ax.grid()
            ax.set_xscale('log')
            #removing top and right borders
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params('both', length=8, width=1, which='major', labelsize=12, direction = "out")
            ax.tick_params('both', length=4, width=1, which='minor', direction = "out")

        Ws = sorted(z.keys())
        for W in Ws:
            state_list = z[W] #Reservoir state, DIM = 𝑁𝑡𝑖𝑚𝑒 × 𝑁𝑠𝑝𝑖𝑛𝑠
            selected_states = state_list[bg:ed, :]
            #print(W, selected_states.shape, np.min(selected_states), np.max(selected_states))
            for i in range(Nspins):
                ys = selected_states[:, i]
                axs[i].scatter([10**W]*len(ys), ys, color='#464646', s=(3.6*72./fig.dpi)**2, marker='o', facecolors='None', lw=0.3, alpha=0.8, rasterized=True)

        for ftype in ['png']:
            plt.savefig('{}/bif_{}_bg_{}_ed_{}.{}'.format(folder, filename, bg, ed, ftype), bbox_inches='tight', dpi=600)