# Run export OMP_NUM_THREADS=1
from qutip import *
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from loginit import get_module_logger
import multiprocessing
from collections import defaultdict
import os
from utils import *

def getSci(sc, i, Nspins):
    iop = identity(2)
    sci = iop
    if i == 0:
        sci = sc
    for j in range(1, Nspins):
        tmp = iop
        if j == i:
            tmp = sc
        sci = tensor(sci, tmp)
    return sci

def dump_eigenval_job(savedir, tempfile, tauls, L, s_prep):
    """
    Dump eigenvalues of superoperator
    """
    print('Start pid={} with size {} (from {} to {})'.format(tempfile, len(tauls), tauls[0], tauls[-1]))
    results = dict()
    for tau in tauls:
        S = (tau*L).expm()
        ts = tensor_contract(S, (0, Nspins)) * s_prep
        # # if tau == 1.0:
        # print(tau, ts.shape, ts.iscp, ts.istp, ts.iscptp, S.shape)
        # exit(1)
        egvals = ts.eigenstates()[0] # Eigenvalues sorted from low to high (magnitude)
        results[tau] = egvals
    with open(tempfile, 'wb') as wrs:
        pickle.dump(results, wrs)
    print('Finished pid={} with size {} (from {} to {})'.format(tempfile, len(tauls), tauls[0], tauls[-1]))
    
if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--nspins', type=int, default=5, help='Number of spins')
    
    parser.add_argument('--tmin', type=float, default=-7.0, help='Minimum of tau')
    parser.add_argument('--tmax', type=float, default=5.0, help='Maximum of tau')
    parser.add_argument('--ntaus', type=int, default=121, help='Number of taus')

    parser.add_argument('--nproc', type=int, default=121)
    parser.add_argument('--savedir', type=str, default='spectral_random')
    parser.add_argument('--basename', type=str, default='spec')
    args = parser.parse_args()
    print(args)

    Nspins, tmin, tmax, ntaus = args.nspins, args.tmin, args.tmax, args.ntaus
    basename = '{}_nspins_{}_log2_tmin_{:.2f}_tmax_{:.2f}_ntaus_{}'.format(args.basename, Nspins, tmin, tmax, ntaus)
    savedir = args.savedir
    if os.path.isdir(savedir) == False:
        os.mkdir(savedir)

    bindir = os.path.join(savedir, 'binary')
    if os.path.isdir(bindir) == False:
        os.mkdir(bindir)

    logdir = os.path.join(savedir, 'log')
    if os.path.isdir(logdir) == False:
        os.mkdir(logdir)
    log_filename = os.path.join(logdir, '{}.log'.format(basename))
    logger = get_module_logger(__name__, log_filename)
    logger.info(log_filename)
    logger.info('Nspins={},log2 tmin={},tmax={},ntaus={}'.format(Nspins, tmin, tmax, ntaus))

    X = sigmax()
    Z = sigmaz()
    I = identity(2)

    #tauls = list(np.linspace(0.0, tmax, ntaus))
    tx = list(np.linspace(tmin, tmax, ntaus))
    tauls = [2**x for x in tx]

    nproc = min(len(tauls), args.nproc)
    lst = np.array_split(tauls, nproc)

    for seed in range(10):
        np.random.seed(seed=seed)

        # Create Hamiltonian
        H0 = getSci(I, 0, Nspins) * 0.0
        H1 = getSci(I, 0, Nspins) * 0.0
        for i in range(Nspins):
            Szi = getSci(Z, i, Nspins)
            H0 = H0 + (np.random.rand()-0.5) * 2 * Szi # Hamiltonian for the magnetic field
            for j in range(i+1, Nspins):
                Sxi = getSci(X, i, Nspins)
                Sxj = getSci(X, j, Nspins)
                H1 = H1 + (np.random.rand()-0.5) * 2 * Sxi * Sxj # Interaction Hamiltonian
        H = H0 + H1 # Total Hamiltonian
        L = liouvillian(H, [])

        # swap with environment
        q0 = getSci(basis(2, 0), 0, Nspins)
        q1 = getSci(basis(2, 1), 0, Nspins)
        
        for pstate in np.linspace(0.0, 1.0, 101):
            binfile = os.path.join(bindir, '{}_eig_pstate_{:.2f}_seed_{}.binary'.format(basename, pstate, seed))
            if os.path.isfile(binfile) == True:
                continue
            s_prep = pstate * sprepost(q0, q0.dag()) + (1.0-pstate) * sprepost(q1, q1.dag())

            jobs, lfiles = [], []
            for pid in range(nproc):
                ts = lst[pid]
                tempfile = '{}_{}'.format(binfile, pid)
                lfiles.append(tempfile)
                p = multiprocessing.Process(target=dump_eigenval_job, \
                    args=(savedir, tempfile, ts, L, s_prep))
                jobs.append(p)

            # Start the process
            for p in jobs:
                p.start()

            # Ensure all processes have finished execution
            for p in jobs:
                p.join()

            # Join dumbpled pickle files
            z = dict()
            for filename in lfiles:
                with open(filename, 'rb') as rrs:
                    tmp = pickle.load(rrs)
                    z = dict(list(z.items()) + list(tmp.items()))
                # Delete file
                os.remove(filename)
                print('zlen={}, Deleted {}'.format(len(z), filename))

            with open(binfile, 'wb') as wrs:
                pickle.dump(z, wrs)

