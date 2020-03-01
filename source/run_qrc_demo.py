import sys
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import tqdm
import time
import qrc
import gendata as gen
import utils

sequence_count = 100
sequence_length = 400
delay = 5
input_sequence_list, output_sequence_list = \
    gen.generate_data(sequence_count, sequence_length, delay)


# この値は量子ビットに相当するため、大きくすると容易にOut of Memoryを起こすので注意
hidden_unit_count = 5
max_coupling_energy = 1.0
trotter_step = 10
beta = 1e-14
model = qrc.QuantumReservoirComputing()
model.train(input_sequence_list, output_sequence_list, hidden_unit_count, max_coupling_energy, trotter_step, beta)

prediction_sequence_list, loss = model.predict(input_sequence_list, output_sequence_list)
print("loss=%f"%loss)
utils.plot(input_sequence_list, output_sequence_list, prediction_sequence_list)

test_input_sequence_list, test_output_sequence_list  = \
    gen.generate_data(sequence_count = sequence_count, sequence_length = sequence_length, delay = delay)
test_prediction_sequence_list, test_loss = model.predict(test_input_sequence_list,test_output_sequence_list)
print("test_loss=%f"%test_loss)
utils.plot(test_input_sequence_list, test_output_sequence_list, test_prediction_sequence_list)