import sys
import os
import glob
import argparse
import numpy as np

if __name__  == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True)
    parser.add_argument('--prefix', type=str, default='qrc_eff_2020-04-15')
    parser.add_argument('--posfix', type=str, default='ntrials_10_eff')
    args = parser.parse_args()
    print(args)

    folder, prefix, posfix = args.folder, args.prefix, args.posfix
    for alpha in [0.9, 0.5, 0.0]:
        rsarr = []
        for rfile in glob.glob('./{}/{}*_strength_{}_*_{}.txt'.format(folder, prefix, alpha, posfix)):
            print(rfile)
            tmp = np.loadtxt(rfile)
            print(tmp.shape)
            rsarr.append(tmp)
        if len(rsarr) > 0:
            rsarr = np.concatenate(rsarr, axis=0)
            print(rsarr.shape)
            np.savetxt('{}\{}_strength_{}_{}.txt'.format(folder, prefix, alpha, posfix), rsarr, delimiter=' ')

