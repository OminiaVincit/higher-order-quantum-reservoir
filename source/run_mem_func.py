import sys
import numpy as np
import os
import scipy
import matplotlib
import matplotlib.pyplot as plt
import tqdm
import time
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

qparams = qrc.QRCParams(hidden_unit_count=hidden_unit_count, max_coupling_energy=max_coupling_energy,\
    trotter_step=trotter_step, beta=beta)


MFlist = utils.memory_function('results', 'qrc', qparams, \
    train_len=train_len, val_len=val_len, buffer=buffer, \
    L=200, N=100)

#data, target = gen.make_data_for_narma(train_len + val_len, buffer=buffer, order=20)

# model = qrc.QuantumReservoirComputing()

# input_seq, output_seq = data[buffer:trainlen + buffer], target[:trainlen]
# input_sequence_list = [input_seq]
# input_sequence_list = np.array(input_sequence_list)

# output_sequence_list = [output_seq]
# output_sequence_list = np.array(output_sequence_list)

# model.train(input_sequence_list, output_sequence_list, hidden_unit_count, max_coupling_energy, trotter_step, beta)

# prediction_sequence_list, loss = model.predict(input_sequence_list,output_sequence_list)
# print("loss=%f"%loss)
# utils.plot(input_sequence_list, output_sequence_list, prediction_sequence_list)

# test_input_seq, test_output_seq  = data[trainlen + buffer:], target[trainlen:]
# test_input_sequence_list = [test_input_seq]
# test_input_sequence_list = np.array(test_input_sequence_list)

# test_output_sequence_list  = [test_output_seq]
# test_output_sequence_list = np.array(test_output_sequence_list)

# test_prediction_sequence_list, test_loss = model.predict(test_input_sequence_list,test_output_sequence_list)
# print("test_loss=%f"%test_loss)
# utils.plot(test_input_sequence_list, test_output_sequence_list, test_prediction_sequence_list)

# # Calculate memori capacity
# for d in range(150):
#     input_list = np.array([data[buffer:trainlen + buffer]]])

#mem_cap_list = []
#for k in range(len(test_input_sequence_list)):
#    prediction = test_prediction_sequence_list[k]
#    mem_cap = utils.memory_capacity(hidden_unit_count, trainlen, buffer, data, prediction)
#    print(k, mem_cap)
