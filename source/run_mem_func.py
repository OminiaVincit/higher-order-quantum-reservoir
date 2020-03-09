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

def memory_compute(taskname, outlist, tmpdir, qparams, train_len, val_len, buffer, maxD, pid):
    for idx in outlist:
        rsarr = qrc.memory_function(taskname, qparams, train_len=train_len, val_len=val_len, buffer=buffer, maxD=maxD, ranseed=0)
        np.savetxt(os.path.join(tmpdir, 'mem_idx_{}.txt'.format(idx)), rsarr, delimiter='\t')

    print('Finished process {} with bg={}, ed={}'.format(pid, outlist[0], outlist[-1]))

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
    
    parser.add_argument('--maxD', type=int, default=10)
    parser.add_argument('--ntrials', type=int, default=20)
    parser.add_argument('--nproc', type=int, default=50)
    parser.add_argument('--taudelta', type=float, default=1.0)

    parser.add_argument('--basename', type=str, default='qrc_stm')
    parser.add_argument('--savedir', type=str, default='results')
    args = parser.parse_args()
    print(args)

    hidden_unit_count, max_coupling_energy, trotter_step, beta =\
        args.units, args.coupling, args.trotter, args.beta
    train_len, val_len, buffer = args.trainlen, args.vallen, args.buffer
    nproc, tau_delta = args.nproc, args.taudelta
    init_rho = args.rho
    maxD, N = args.maxD, args.ntrials

    basename = args.basename
    savedir = args.savedir
    Vs = [1, 2, 5, 10, 25, 50]
    #Vs = [1]
    Varrs = []
    
    if os.path.isfile(savedir):
        # Load setting file
        figbase = savedir.replace('_setting.txt', '')
        for V in Vs:
            rsarr = np.loadtxt('{}_V_{}_mem.txt'.format(figbase, V))
            Varrs.append(rsarr)
    else:
        if os.path.isdir(savedir) == False:
            os.mkdir(savedir)
        timestamp = int(time.time() * 1000.0)
        now = datetime.datetime.now()
        datestr = now.strftime('{0:%Y-%m-%d-%H-%M-%S}'.format(now))

        for V in Vs:
            qparams = qrc.QRCParams(hidden_unit_count=hidden_unit_count, max_coupling_energy=max_coupling_energy,\
            trotter_step=trotter_step, beta=beta, virtual_nodes=V, tau_delta=tau_delta, init_rho=init_rho)

            # Multi-process
            tmpdir = os.path.join(savedir, '{}_V_{}_{}'.format(basename, V, timestamp))
            os.mkdir(tmpdir)

            processes = []
            lst = np.array_split(range(N), nproc)
            for proc_id in range(nproc):
                outlist = lst[proc_id]
                if outlist.size == 0:
                    continue
                print(outlist)
                p = Process(target=memory_compute, args=(basename, outlist, tmpdir, qparams, train_len, val_len, buffer, maxD, proc_id))
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
            Varrs.append(rsarr)

            # remove tmpdir
            import shutil
            shutil.rmtree(tmpdir)
            
            # Save results
            outbase = os.path.join(savedir, '{}_{}_V_{}'.format(basename, datestr, V))
            np.savetxt('{}_mem.txt'.format(outbase), rsarr, delimiter='\t')
    
        # save experiments setting
        outbase = os.path.join(savedir, '{}_{}'.format(basename, datestr, V))
        with open('{}_setting.txt'.format(outbase), 'w') as sfile:
            sfile.write('train_len={}, val_len={}, buffer={}, maxD={}, numtrials={}\n'.format(train_len, val_len, buffer, maxD, N))
            sfile.write('hidden_unit_count={}\n'.format(qparams.hidden_unit_count))
            sfile.write('max_coupling_energy={}\n'.format(qparams.max_coupling_energy))
            sfile.write('trotter_step={}\n'.format(qparams.trotter_step))
            sfile.write('beta={}\n'.format(qparams.beta))
            sfile.write('virtual nodes={}\n'.format(' '.join([str(v) for v in Vs])))
            sfile.write('tau_delta={}\n'.format(qparams.tau_delta))
            sfile.write('maxD={}\n'.format(maxD))
            sfile.write('Ntrials={}\n'.format(N))
        figbase = '{}_mem'.format(outbase)

    # save MF plot
    plt.style.use('seaborn-colorblind')
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams['font.size']=16

    fig, ax = plt.subplots()
    ax.set_xlabel(r'Delay $d$')
    ax.set_ylabel(r'$MF_d$')
    for i in range(len(Vs)):
        rsarr = Varrs[i]
        dlist, MFlist = rsarr[:, 0], rsarr[:, 1]

        #ax.set_yscale('log')
        ax.plot(dlist, MFlist, 'o--', label='V={}'.format(Vs[i]))
    plt.legend()
    # plt.show()
    for ftype in ['png', 'pdf', 'svg']:
        plt.savefig('{}.{}'.format(figbase, ftype), bbox_inches='tight')
    
