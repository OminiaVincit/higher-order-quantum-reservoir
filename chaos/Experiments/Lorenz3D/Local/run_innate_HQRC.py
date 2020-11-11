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

class QParams():
    def __init__(self, nqrc, hidden_unit_count, max_coupling_energy, \
        virtual_nodes, tau_delta, layer_strength, scale_input):
        self.nqrc = nqrc
        self.hidden_unit_count = hidden_unit_count
        self.max_coupling_energy = max_coupling_energy
        self.virtual_nodes = virtual_nodes
        self.tau_delta = tau_delta
        self.layer_strength = layer_strength
        self.scale_input = scale_input

class DParams():
    def __init__(self, N_length, N_used, RDIM, noise_level, scaler,\
        trans, ratio, dynamics_length, iterative_prediction_length, num_test_ICS):
        self.N_length = N_length
        self.N_used = N_used
        self.RDIM = RDIM
        self.noise_level = noise_level
        self.scaler = scaler
        self.trans = trans
        self.ratio = ratio
        self.dynamics_length = dynamics_length
        self.iterative_prediction_length = iterative_prediction_length
        self.num_test_ICS = num_test_ICS

class OParams():
    def __init__(self, solver, regularization):
        self.solver = solver
        self.regularization = regularization


class IParams():
    def __init__(self, output_noise, innate_learning_rate, innate_learning_loops):
        self.output_noise = output_noise
        self.innate_learning_rate = innate_learning_rate
        self.innate_learning_loops = innate_learning_loops

def execute_job(name, qparams, dparams, oparams, iparams):
    print('Start process with output_noise={}, scale={}, r={}'.format(\
        iparams.output_noise, qparams.scale_input, oparams.regularization))
    cmd = 'cd ../../../Methods && python3 RUN.py hqrc_innate \
            --mode all \
            --augment 0 \
            --display_output 1 \
            --write_to_log 1 \
            --N {0} \
            --N_used {1} \
            --RDIM {2} \
            --noise_level {3} \
            --scaler {4} \
            --trans {5} \
            --ratio {6} \
            --nqrc {7} \
            --alpha {8} \
            --scale_input {9} \
            --max_energy {10} \
	        --fix_coupling 0 \
            --virtual_nodes {11} \
            --tau {12} \
            --n_units {13} \
            --reg {14} \
            --solver {15} \
            --dynamics_length {16} \
            --it_pred_length {17} \
            --n_tests {18} \
            --system_name {19} \
            --output_noise {20}\
            --innate_learning_rate {21}\
            --innate_learning_loops {22}'.format(\
            dparams.N_length, dparams.N_used, dparams.RDIM, \
            dparams.noise_level, dparams.scaler, dparams.trans, dparams.ratio,\
            qparams.nqrc, qparams.layer_strength, qparams.scale_input, qparams.max_coupling_energy,\
            qparams.virtual_nodes, qparams.tau_delta, qparams.hidden_unit_count,\
            oparams.regularization, oparams.solver,\
            dparams.dynamics_length, dparams.iterative_prediction_length, dparams.num_test_ICS,\
            name, iparams.output_noise, iparams.innate_learning_rate, iparams.innate_learning_loops)
    #cmd = 'cd ../../../Methods'
    os.system(cmd)
    print('Finish process with scale={}, r={}'.format(qparams.scale_input, oparams.regularization))
    
if __name__ == '__main__':
    # Fix parameter
    NAME = "Lorenz3D"

    nqrc, V, units = 5, 1, 6
    J, tau = 2.0, 4.0
    
    noise = 1
    trans, ratio = 0, 0
    scaler = 'MinMaxZeroOne'

    N, N_used = 100000, 50000
    RDIM, num_test_ICS = 1, 11
    dynamics_length, prediction_length = 1000, 1000
    innate_learning_loops = 10
    innate_learning_rate = 10.0
    solver = 'pinv'

    #noise_ls = [1e-6, 2e-6, 5e-6, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3]
    noise_ls = []
    for i in [-6]:
        for j in range(1, 11):
            noise_ls.append(j * (10**i))
    #noise_ls.append(1e-5)
    
    for innate_learning_rate in [0.0]:
        if innate_learning_rate == 0.0:
            innate_learning_loops = 1
        else:
            innate_learning_loops = 10
        
        jobs = []
        for N_used in [10000, 20000, 50000, 100000]:
            for V in [1]:
                for alpha in [0.0]:
                    for scale_input in [0.1, 0.2]:
                        for onoise in noise_ls:
                            for r in [1e-7, 1e-9]:
                                qparams = QParams(nqrc=nqrc, hidden_unit_count=units, max_coupling_energy=J,\
                                    virtual_nodes=V, tau_delta=tau, layer_strength=alpha, scale_input=scale_input)
                                dparams = DParams(N_length=N, N_used=N_used, RDIM=RDIM, noise_level=noise, scaler=scaler,\
                                    trans=trans, ratio=ratio, dynamics_length=dynamics_length, \
                                    iterative_prediction_length=prediction_length, num_test_ICS=num_test_ICS)
                                oparams = OParams(solver=solver, regularization=r)
                                iparams = IParams(output_noise=onoise, innate_learning_loops=innate_learning_loops, \
                                    innate_learning_rate=innate_learning_rate)
                                p = multiprocessing.Process(target=execute_job, args=(NAME, qparams, dparams, oparams, iparams))
                                jobs.append(p)

        # Start the process
        for p in jobs:
            p.start()

        # Ensure all processes have finished execution
        for p in jobs:
            p.join()