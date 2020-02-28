import sys
import numpy as np
import scipy as sp
from scipy import linalg
import tqdm
import time

class QuantumReservoirComputing(object):
    def __feed_forward(self, input_sequence_list):
        sequence_count, sequence_length = input_sequence_list.shape
        predict_sequence_list = []
        state_list = []
        dim = 2**self.qubit_count
        sequence_range = tqdm.trange(sequence_count)
        for sequence_index in sequence_range:
            rho = np.zeros( [dim,dim] )
            rho[0,0]=1
            state = []
            for time_step in range(sequence_length):
                rho = self.P0op @ rho @ self.P0op + self.Xop[0] @ self.P1op @ rho @ self.P1op @ self.Xop[0]
                # (1 + u Z)/2 = (1+u)/2 |0><0| + (1-u)/2 |1><1|
                value = input_sequence_list[sequence_index, time_step]
                rho = (1+value)/2 * rho + (1-value)/2 *self.Xop[0] @ rho @ self.Xop[0]
                rho = self.Uop @ rho @ self.Uop.T.conj()

                current_state = []
                for qubit_index in range(1,self.qubit_count):
                    expectation_value = np.real(np.trace(self.Zop[qubit_index] @ rho))
                    current_state.append(expectation_value)
                state.append(current_state)
            state = np.array(state)
            state_list.append(state)

            stacked_state = np.hstack( [state, np.ones([sequence_length,1])])
            predict_sequence = stacked_state @ self.W_out
            predict_sequence = np.squeeze(predict_sequence, axis=1)
            predict_sequence_list.append(predict_sequence)
        predict_sequence_list = np.array(predict_sequence_list)
        state_list = np.array(state_list)
        return predict_sequence_list, state_list

    def train(self, input_sequence_list, output_sequence_list, hidden_unit_count, max_coupling_energy, trotter_step, beta):
        assert(input_sequence_list.shape == output_sequence_list.shape)
        self.hidden_unit_count = hidden_unit_count
        self.trotter_step = trotter_step

        self.sequence_count, self.sequence_length = input_sequence_list.shape
        self.hidden_unit_count = hidden_unit_count

        self.W_out = np.random.rand(self.hidden_unit_count+1,1)

        I = [[1,0],[0,1]]
        Z = [[1,0],[0,-1]]
        X = [[0,1],[1,0]]
        P0 = [[1,0],[0,0]]
        P1 = [[0,0],[0,1]]
        self.qubit_count = self.hidden_unit_count+1
        self.dim = 2**self.qubit_count
        self.Zop = [1]*self.qubit_count
        self.Xop = [1]*self.qubit_count
        self.P0op = [1]
        self.P1op = [1]

        for cursor_index in range(self.qubit_count):
            for qubit_index in range(self.qubit_count):
                if cursor_index == qubit_index:
                    self.Xop[qubit_index] = np.kron(self.Xop[qubit_index],X)
                    self.Zop[qubit_index] = np.kron(self.Zop[qubit_index],Z)
                else:
                    self.Xop[qubit_index] = np.kron(self.Xop[qubit_index],I)
                    self.Zop[qubit_index] = np.kron(self.Zop[qubit_index],I)

            if cursor_index == 0:
                self.P0op = np.kron(self.P0op, P0)
                self.P1op = np.kron(self.P1op, P1)
            else:
                self.P0op = np.kron(self.P0op, I)
                self.P1op = np.kron(self.P1op, I)

        self.hamiltonian = np.zeros( (self.dim,self.dim) )
        for qubit_index in range(self.qubit_count):
            coef = (np.random.rand()-0.5) * 2 * max_coupling_energy
            self.hamiltonian += coef * self.Zop[qubit_index]
        for qubit_index1 in range(self.qubit_count):
            for qubit_index2 in range(qubit_index1+1, self.qubit_count):
                coef = (np.random.rand()-0.5) * 2 * max_coupling_energy
                self.hamiltonian += coef * self.Xop[qubit_index1] @ self.Xop[qubit_index2]
        self.Uop = sp.linalg.expm(1.j * self.hamiltonian)

        _, state_list = self.__feed_forward(input_sequence_list)

        state_list = np.array(state_list)
        V = np.reshape(state_list, [-1, hidden_unit_count])
        V = np.hstack( [V, np.ones([V.shape[0], 1]) ] )
        S = np.reshape(output_sequence_list, [-1])
        self.W_out = np.linalg.pinv(V, rcond = beta) @ S
        self.W_out = np.expand_dims(self.W_out,axis=1)

    def predict(self, input_sequence_list,output_sequence_list):
        prediction_sequence_list, _ = self.__feed_forward(input_sequence_list)
        loss = np.sum((prediction_sequence_list-output_sequence_list)**2)/2
        loss /= prediction_sequence_list.shape[0]
        return prediction_sequence_list, loss