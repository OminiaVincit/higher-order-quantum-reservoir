import time
import numpy as np
from multiprocessing import Pool, Process
import argparse

import os
import mkl

def foo(x, pid=1):
    #mkl.set_num_threads(24)
    np.linalg.pinv(x)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nproc', type=int, default=1)
    args = parser.parse_args()
    print(args)
    nproc = args.nproc
    #os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
    t = time.time()
    r = np.random.rand(1000, 1000)

    # # Running fine with Pool on windows
    # p = Pool(nproc)
    # p.map(foo, [r.copy() for i in range(8)])
    # print('Pool map Finished in', time.time() - t, 'sec')

    # t = time.time()
    processes = []
    for proc_id in range(nproc):
        p = Process(target=foo, args=([r.copy() for i in range(32)], proc_id))
    processes.append(p)
    # Start the process
    for p in processes:
        p.start()

    # Ensure all processes have finiished execution
    for p in processes:
        p.join()
    print('Process Finished in', time.time() - t, 'sec')