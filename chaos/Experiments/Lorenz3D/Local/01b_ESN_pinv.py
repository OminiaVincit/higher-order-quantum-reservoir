#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by:  Quoc Hoan Tran,
                Nakajima-lab, The University of Tokyo
"""
#!/usr/bin/env python

import numpy as np 
import os 
import subprocess
import multiprocessing

# os.environ["OMP_NUM_THREADS"] = "15" # export OMP_NUM_THREADS=4
# os.environ["OPENBLAS_NUM_THREADS"] = "15" # export OPENBLAS_NUM_THREADS=4 
# os.environ["MKL_NUM_THREADS"] = "15" # export MKL_NUM_THREADS=6
# os.environ["VECLIB_MAXIMUM_THREADS"] = "15" # export VECLIB_MAXIMUM_THREADS=4
# os.environ["NUMEXPR_NUM_THREADS"] = "15" # export NUMEXPR_NUM_THREADS=6


def execute_job(name, reservoir_size, regularization, numics):
    cmd = 'cd ../../../Methods && python3 RUN.py esn \
            --mode all \
            --display_output 1 \
            --write_to_log 1 \
            --N 100000 \
            --N_used 1000 \
            --RDIM 1 \
            --noise_level 1 \
            --scaler Standard \
            --approx_reservoir_size {0} \
            --degree 10 \
            --radius 0.6 \
            --sigma_input 1 \
            --regularization {1} \
            --solver pinv \
            --dynamics_length 200 \
            --iterative_prediction_length 500 \
            --number_of_epochs 1000000 \
            --learning_rate 0.001 \
            --reference_train_time 10 \
            --buffer_train_time 0.5 \
            --num_test_ICS {2} \
            --system_name {3} '.format(reservoir_size, regularization, numics, name)
    #cmd = 'cd ../../../Methods'
    os.system(cmd)
    
if __name__ == '__main__':
    # Fix parameter
    NAME = "Lorenz3D"
    numics = 100

    jobs = []
    for units in [80, 120, 150]:
        for r in [1e-5, 1e-7, 1e-9]:
            p = multiprocessing.Process(target=execute_job, args=(NAME, units, r, numics))
            jobs.append(p)

    # Start the process
    for p in jobs:
        p.start()

    # Ensure all processes have finished execution
    for p in jobs:
        p.join()

    
