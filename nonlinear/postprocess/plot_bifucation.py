import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
import matplotlib.gridspec as gridspec
import glob, os

if __name__  == '__main__':
        # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='../../../data/hqrc/dynamics_sinwave/states')
    parser.add_argument('--prefix', type=str, default='phase_trans')
    parser.add_argument('--posfix', type=str, default='states_id')
    parser.add_argument('--gamma', type=float, default=-1.0)
    parser.add_argument('--logW', type=float, default=0.0)
    
    parser.add_argument('--bg', type=int, default=9000)
    parser.add_argument('--ed', type=int, default=10000)
    parser.add_argument('--nspins', type=int, default=6)
    parser.add_argument('--nqrc', type=int, default=1)
    parser.add_argument('--nproc', type=int, default=100)
    parser.add_argument('--length', type=int, default=1000)
    parser.add_argument('--randseed', type=int, default=0)
    parser.add_argument('--type_op', type=str, default='X')
    parser.add_argument('--type_input', type=int, default=5)

    args = parser.parse_args()
    print(args)
    folder, prefix, posfix, gamma, logW = args.folder, args.prefix, args.posfix, args.gamma, args.logW
    bg, ed, Nspins, nqrc, nproc, length = args.bg, args.ed, args.nspins, args.nqrc, args.nproc, args.length
    rtd, type_input, type_op = args.randseed, args.type_input, args.type_op

    prefix = '{}_nqr_{}_V_1_tau_10.0_nondiag_2.0'.format(prefix, nqrc)
    posfix = '{}_{}_len_{}'.format(posfix, nproc-1, length)

    if gamma > 0:
        label = 'Disorder strength'
        pattern = '{}_*gam_{}_op_{}_tp_{}_*rtd_{}_*{}'.format(prefix, gamma, type_op, type_input, rtd, posfix)
    else:
        label = 'Feedback strength'
        pattern = '{}_*logW_{:.3f}_op_{}_tp_{}_*rtd_{}_*{}'.format(prefix, logW, type_op, type_input, rtd, posfix)

    fls = glob.glob('{}/{}.binaryfile'.format(folder, pattern))
    
    for filepath in fls:
        with open(filepath, 'rb') as rrs:
            z = pickle.load(rrs)
        
            plt.rcParams['font.size'] = 14
            fig = plt.figure(figsize=(8, 16))
            #Nspins = list(z.values())[0].shape[1]
            gs = gridspec.GridSpec(Nspins, 1, wspace=0.4, hspace=0.4)
            axs = [plt.subplot(gs[i, 0]) for i in range(Nspins)]
            for i in range(Nspins):
                ax = axs[i]
                ax.set_xlabel(label)
                ax.set_ylabel('Spin {}'.format(i+1))
                ax.set_xlim(10**(-2.02), 10**2.1)
                #ax.set_ylim(-1.1, 1.1)
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
                plt.savefig('{}/bif_{}_bg_{}_ed_{}.{}'.format(folder, os.path.basename(filepath), bg, ed, ftype), bbox_inches='tight', dpi=600)