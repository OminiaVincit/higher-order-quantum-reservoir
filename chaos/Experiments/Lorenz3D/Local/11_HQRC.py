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

def execute_job(name, qparams, dparams, oparams):
    print('Start process with layer_strength={}, scale={}, r={}'.format(\
        qparams.layer_strength, qparams.scale_input, oparams.regularization))
    cmd = 'cd ../../../Methods && python3 RUN.py hqrc \
            --mode all \
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
            --layer_strength {8} \
            --scale_input {9} \
            --max_coupling_energy {10} \
	    --fix_coupling 1 \
            --virtual_nodes {11} \
            --tau_delta {12} \
            --hidden_unit_count {13} \
            --regularization {14} \
            --solver {15} \
            --dynamics_length {16} \
            --iterative_prediction_length {17} \
            --num_test_ICS {18} \
            --system_name {19} '.format(\
            dparams.N_length, dparams.N_used, dparams.RDIM, \
            dparams.noise_level, dparams.scaler, dparams.trans, dparams.ratio,\
            qparams.nqrc, qparams.layer_strength, qparams.scale_input, qparams.max_coupling_energy,\
            qparams.virtual_nodes, qparams.tau_delta, qparams.hidden_unit_count,\
            oparams.regularization, oparams.solver,\
            dparams.dynamics_length, dparams.iterative_prediction_length, dparams.num_test_ICS,\
            name)
    #cmd = 'cd ../../../Methods'
    os.system(cmd)
    print('Finish process with scale={}, r={}'.format(qparams.scale_input, oparams.regularization))
    
if __name__ == '__main__':
    # Fix parameter
    NAME = "Lorenz3D"

    nqrc, V, units = 5, 20, 6
    J, tau = 2.0, 4.0
    
    noise = 1
    trans, ratio = 0, 0
    scaler = 'MinMaxZeroOne'

    N, N_used = 100000, 10000
    RDIM, num_test_ICS = 1, 100
    dynamics_length, prediction_length = 2000, 500

    solver = 'pinv'

    jobs = []
    for V in [10, 15, 20]:
        for alpha in [0.0, 0.1]:
            for scale_input in [1.0]:
                for r in [1e-7, 1e-9]:
                    qparams = QParams(nqrc=nqrc, hidden_unit_count=units, max_coupling_energy=J,\
                        virtual_nodes=V, tau_delta=tau, layer_strength=alpha, scale_input=scale_input)
                    dparams = DParams(N_length=N, N_used=N_used, RDIM=RDIM, noise_level=noise, scaler=scaler,\
                        trans=trans, ratio=ratio, dynamics_length=dynamics_length, \
                        iterative_prediction_length=prediction_length, num_test_ICS=num_test_ICS)
                    oparams = OParams(solver=solver, regularization=r)
                    p = multiprocessing.Process(target=execute_job, args=(NAME, qparams, dparams, oparams))
                    jobs.append(p)

    # Start the process
    for p in jobs:
        p.start()

    # Ensure all processes have finished execution
    for p in jobs:
        p.join()

    
