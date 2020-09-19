"""
Perform timing task for quantum innate learning
"""
import sys
import numpy as np
import os
import scipy
import scipy.io as sio # For loading matlab file
from scipy.stats import norm
import argparse
import multiprocessing
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
import time
import datetime
import hqrc as hqrc
import utils
from utils import *

numInputs = 1   # number of input units
numOut = 1      # number of output units
# Parameters and hyperparameters

# Input parameters
input_pulse_value = 1.0
start_pulse = 200   # (ms)
reset_duration = 50 # (ms)

# Training
interval = 1000
learn_every = 2 # skip time points
start_train = start_pulse + reset_duration
end_train = start_train + interval + 150
n_learn_loops_recu = 20     # number of training loops (recurrent)
n_learn_loops_read = 10     # number of training loops (readout)
n_test_loops = 10

# For numeric
dt = 1.0
tmax = end_train + 200
time_axis = np.arange(0, tmax, dt)
n_steps = len(time_axis)

# For output function
ready_level = 0.2
peak_level = 1
peak_width = 30
peak_time = start_train + interval

# For training and testting
# savefile_trained = 'DAC_timing_hqrc.innate'

if __name__ == '__main__':
    # Check for command line arguments
    parser = argparse.ArgumentParser()
    # parser.add_argument('--matfile', type=str, default='../data/DAC_timing_recurr800_p0.1_g1.5.mat')
    args = parser.parse_args()
    print(args)

    # # generate innate activity and input, output
    # matfile = sio.loadmat(args.matfile)
    # target_innate_X = np.array(matfile['Target_innate_X']).astype(np.float32)
    # target_out = np.array(matfile['target_Out']).astype(np.float32)
    
    # input_pattern = np.array(matfile['input_pattern']).astype(np.float32)
    # start_train = int(matfile['start_train'])
    # end_train = int(matfile['end_train'])
    # print(target_innate_X.shape)

    # Input
    start_pulse_n = int(start_pulse/dt)
    reset_duration_n = int(reset_duration/dt)
    start_train_n = int(start_train/dt)
    end_train_n = int(end_train/dt)
    input_seq = np.zeros(numInputs, n_steps)
    input_seq[0, start_pulse_n:(start_pulse_n+reset_duration_n)] = input_pulse_value

    # Output target
    bell = norm.pdf(time_axis, loc=peak_time, scale=peak_width)
    bell_max = max(bell)
    target_seq = ready_level + ((peak_level-ready_level)/bell_max)*bell

    # Creat innate trajectory for target
