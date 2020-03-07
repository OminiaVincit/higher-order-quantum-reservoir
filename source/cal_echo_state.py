import sys
import numpy as np
import os
import scipy
import argparse
from multiprocessing import Process
import matplotlib
import matplotlib.pyplot as plt
import tqdm
import time
import datetime
import qrc
import gendata as gen
import utils

train_len = 2000
val_len = 2000
buffer = 2000

hidden_unit_count = 5
max_coupling_energy = 1.0
trotter_step = 10
beta = 1e-14

def memory_compute(outlist, tmpdir, qparams, train_len, val_len, buffer, maxD, pid):
    for idx in outlist:
        rsarr = utils.memory_function(qparams, train_len=train_len, val_len=val_len, buffer=buffer, L=maxD)
        np.savetxt(os.path.join(tmpdir, 'mem_idx_{}.txt'.format(idx)), rsarr, delimiter='\t')

    print('Finished process {} with bg={}, ed={}'.format(pid, outlist[0], outlist[-1]))

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--units', type=int, default=5)
    parser.add_argument('--coupling', type=float, default=1.0)
    parser.add_argument('--trotter', type=int, default=10)
    parser.add_argument('--beta', type=float, default=1e-14)

    parser.add_argument('--trainlen', type=int, default=2000)
    parser.add_argument('--vallen', type=int, default=2000)
    parser.add_argument('--buffer', type=int, default=2000)
    
    parser.add_argument('--maxD', type=int, default=200)
    parser.add_argument('--ntrials', type=int, default=50)
    parser.add_argument('--virtuals', type=int, default=10)
    parser.add_argument('--nproc', type=int, default=50)
    parser.add_argument('--taudelta', type=float, default=1.0)

    parser.add_argument('--basename', type=str, default='echo')
    parser.add_argument('--savedir', type=str, default='results')
    args = parser.parse_args()
    print(args)

    hidden_unit_count, max_coupling_energy, trotter_step, beta =\
        args.units, args.coupling, args.trotter, args.beta
    train_len, val_len, buffer = args.trainlen, args.vallen, args.buffer
    maxD, N, nproc = args.maxD, args.ntrials, args.nproc
    virtual_nodes, tau_delta = args.virtuals, args.taudelta

    basename = args.basename
    savedir = args.savedir

    if os.path.isfile(savedir):
        # Load result file and plot
        rsarr = np.loadtxt(savedir)
        figbase = savedir.replace('.txt', '')
    else:
        if os.path.isdir(savedir) == False:
            os.mkdir(savedir)

        qparams = qrc.QRCParams(hidden_unit_count=hidden_unit_count, max_coupling_energy=max_coupling_energy,\
            trotter_step=trotter_step, beta=beta)

        timestamp = int(time.time() * 1000.0)
        now = datetime.datetime.now()
        datestr = now.strftime('{0:%Y-%m-%d-%H-%M-%S}'.format(now))
        # Multi-process
        tmpdir = os.path.join(savedir, '{}_{}'.format(basename, timestamp))
        os.mkdir(tmpdir)

        processes = []
        lst = np.array_split(range(N), nproc)
        for proc_id in range(nproc):
            outlist = lst[proc_id]
            if outlist.size == 0:
                continue
            print(outlist)
            p = Process(target=memory_compute, args=(outlist, tmpdir, qparams, train_len, val_len, buffer, maxD, proc_id))
            processes.append(p)
    
        # Start the process
        for p in processes:
            p.start()
    
        # Ensure all processes have finiished execution
        for p in processes:
            p.join()

        # Sleep 5s
        time.sleep(5)

        # Average results and remove tmpdir
        rsarr = []
        for idx in range(N):
            filename = os.path.join(tmpdir, 'mem_idx_{}.txt'.format(idx))
            if os.path.isfile(filename):
                arr = np.loadtxt(filename)
                rsarr.append(arr)
        rsarr = np.mean(rsarr, axis=0)

        # remove tmpdir
        import shutil
        shutil.rmtree(tmpdir)
        
        # # Save results
        # outbase = os.path.join(savedir, '{}_{}'.format(basename, datestr))
        # np.savetxt('{}_mem.txt'.format(outbase), rsarr, delimiter='\t')
    
        # # save experiments setting
        # with open('{}_setting.txt'.format(outbase), 'w') as sfile:
        #     sfile.write('train_len={}, val_len={}, buffer={}, maxD={}, numtrials={}\n'.format(train_len, val_len, buffer, maxD, N))
        #     sfile.write('hidden_unit_count={}\n'.format(qparams.hidden_unit_count))
        #     sfile.write('max_coupling_energy={}\n'.format(qparams.max_coupling_energy))
        #     sfile.write('trotter_step={}\n'.format(qparams.trotter_step))
        #     sfile.write('beta={}\n'.format(qparams.beta))
        # figbase = '{}_mem'.format(outbase)

    # save MF plot
    # plt.style.use('seaborn-colorblind')
    # plt.rc('font', family='serif')
    # plt.rc('mathtext', fontset='cm')

    # fig, ax = plt.subplots()
    # ax.set_xlabel(r'Delay $d$', fontsize=16)
    # ax.set_ylabel(r'$MF_d$', fontsize=16)
    # dlist, MFlist = rsarr[:, 0], rsarr[:, 1]
    # ax.set_yscale('log')
    # ax.scatter(dlist, MFlist, label='MF_d')
    # for ftype in ['png', 'pdf', 'svg']:
    #     plt.savefig('{}.{}'.format(figbase, ftype), bbox_inches='tight')
    
