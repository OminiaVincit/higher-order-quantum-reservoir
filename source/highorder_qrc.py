import sys
import numpy as np
import scipy as sp 
import utils
import qrc
from scipy.special import softmax

def solfmax_layer(states):
    states = np.array(states)
    return softmax(states)

def linear_combine(u, states, coeffs):
    assert(len(coeffs) == len(states))
    v = 1.0 - np.sum(coeffs)
    assert(v <= 1.00001 and v >= -0.00001)
    v = max(v, 0.0)
    v = min(v, 1.0)
    total = v * u
    total += np.dot(np.array(states).flatten(), np.array(coeffs).flatten())
    return total

def softmax_linear_combine(u, states, coeffs):
    states = solfmax_layer(states)
    return linear_combine(u, states, coeffs)

class HighorderQuantumReservoirComputing(object):
    def __init_reservoir(self, qparams, nqrc, layer_strength):
        I = [[1,0],[0,1]]
        Z = [[1,0],[0,-1]]
        X = [[0,1],[1,0]]
        P0 = [[1,0],[0,0]]
        P1 = [[0,0],[0,1]]
        self.hidden_unit_count = qparams.hidden_unit_count
        self.trotter_step = qparams.trotter_step
        self.virtual_nodes = qparams.virtual_nodes
        self.tau_delta = qparams.tau_delta
        self.nqrc = nqrc
        self.qubit_count = self.hidden_unit_count
        self.dim = 2**self.qubit_count
        self.Zop = [1]*self.qubit_count
        self.Xop = [1]*self.qubit_count
        self.P0op = [1]
        self.P1op = [1]
        self.layer_strength = layer_strength

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

        # initialize connection to layer i
        connections = []
        N_local_states = self.hidden_unit_count * self.virtual_nodes
        if nqrc > 1:
            for i in range(nqrc):
                local_cs = []
                for j in range(nqrc):
                    cs = [0] * N_local_states
                    if j != i:
                        cs = np.random.rand(N_local_states)
                    local_cs.append(cs)
                local_cs = np.array(local_cs).flatten()

                alpha = self.layer_strength
                if alpha < 0 or alpha > 1:
                    alpha = np.random.rand()
                local_cs = alpha * local_cs / np.sum(local_cs)
                connections.append(local_cs)
        self.coeffs = connections

        # initialize current states
        self.previous_states = [None] * nqrc
        self.current_states  = [None] * nqrc

        # Intialize evolution operators
        tmp_uops = []
        tmp_rhos = []
        for i in range(nqrc):
            # initialize density matrix
            rho = np.zeros( [self.dim, self.dim] )
            rho[0, 0]=1
            if qparams.init_rho != 0:
                # initialize random density matrix
                rho = gen_random.random_density_matrix(self.dim)
            tmp_rhos.append(rho)

            # generate hamiltonian
            hamiltonian = np.zeros( (self.dim,self.dim) )

            # include input qubit for computation
            for qubit_index in range(self.qubit_count):
                coef = (np.random.rand()-0.5) * 2 * qparams.max_coupling_energy
                hamiltonian += coef * self.Zop[qubit_index]
            for qubit_index1 in range(self.qubit_count):
                for qubit_index2 in range(qubit_index1+1, self.qubit_count):
                    coef = (np.random.rand()-0.5) * 2 * qparams.max_coupling_energy
                    hamiltonian += coef * self.Xop[qubit_index1] @ self.Xop[qubit_index2]
                    
            ratio = float(self.tau_delta) / float(self.virtual_nodes)        
            Uop = sp.linalg.expm(1.j * hamiltonian * ratio)
            tmp_uops.append(Uop)
        
        self.init_rhos = tmp_rhos
        self.last_rhos = tmp_rhos
        self.Uops = tmp_uops


    def __feed_forward(self, input_sequence, predict=True, use_lastrho=False):
        input_dim, input_length = input_sequence.shape
        assert(input_dim == self.nqrc)
        
        dim = 2**self.qubit_count
        predict_sequence = None
        local_rhos = self.last_rhos
        nqrc = self.nqrc

        state_list = []
        for time_step in range(0, input_length):
            local_prev_states = []
            for i in reversed(range(nqrc)):
                Uop = self.Uops[i]
                if use_lastrho == True :
                    #print('Use last density matrix')
                    rho = self.last_rhos[i]
                else:
                    rho = self.init_rhos[i]
                # Obtain value from the input
                value = input_sequence[i, time_step]
                # Obtain values from previous layer
                previous_states = self.previous_states
                if nqrc > 1 and previous_states[0] is not None:
                    value = softmax_linear_combine(value, previous_states, self.coeffs[i])

                # Replace the density matrix
                rho = self.P0op @ rho @ self.P0op + self.Xop[0] @ self.P1op @ rho @ self.P1op @ self.Xop[0]
                # (1 + u Z)/2 = (1+u)/2 |0><0| + (1-u)/2 |1><1|
            
                # for input in [-1, 1]
                # rho = (1+value)/2 * rho + (1-value)/2 *self.Xop[0] @ rho @ self.Xop[0]
                
                # for input in [0, 1]
                rho = (1 - value) * rho + value * self.Xop[0] @ rho @ self.Xop[0]
                current_state = []
                for v in range(self.virtual_nodes):
                    # Time evolution of density matrix
                    rho = Uop @ rho @ Uop.T.conj()
                    for qubit_index in range(0, self.qubit_count):
                        expectation_value = np.real(np.trace(self.Zop[qubit_index] @ rho))
                        current_state.append(expectation_value)
                # Size of current_state is Nqubits x Nvirtuals
                local_prev_states.append(self.current_states[i])
                self.current_states[i] = np.array(current_state)
                local_rhos[i] = rho

            # only use state of the last qrc to train
            state = np.array(self.current_states)
            state_list.append(state.flatten())

            # update previous states
            self.previous_states = np.array(local_prev_states).flatten()
        state_list = np.array(state_list)
        self.last_rhos = local_rhos

        if predict:
            stacked_state = np.hstack( [state_list, np.ones([input_length, 1])])
            print('stacked state {}; Wout {}'.format(stacked_state.shape, self.W_out.shape))
            predict_sequence = stacked_state @ self.W_out
        
        return predict_sequence, state_list


    def __train(self, input_sequence, output_sequence, buffer, beta):
        print('shape', input_sequence.shape, output_sequence.shape)
        assert(input_sequence.shape[1] == output_sequence.shape[0])
        Nout = output_sequence.shape[1]
        self.W_out = np.random.rand(self.hidden_unit_count * self.virtual_nodes * self.nqrc + 1, Nout)

        _, state_list = self.__feed_forward(input_sequence, predict=False)

        state_list = np.array(state_list)
        print('before washingout state list shape', state_list.shape)
        
        state_list = state_list[buffer:, :]
        print('after washingout state list shape', state_list.shape)

        # discard the transitient state for training
        V = np.reshape(state_list, [-1, self.hidden_unit_count * self.virtual_nodes * self.nqrc])
        V = np.hstack( [state_list, np.ones([V.shape[0], 1]) ] )

        print('output seq', output_sequence.shape)
        discard_output = output_sequence[buffer:, :]
        print('discard output seq', discard_output.shape)
        #S = np.reshape(output_sequence_list, [-1])
        S = np.reshape(discard_output, [discard_output.shape[0]*discard_output.shape[1], -1])
        print('V S', V.shape, S.shape)
        self.W_out = np.linalg.pinv(V, rcond = beta) @ S
        print('bf Wout', self.W_out.shape)
        
    def train_to_predict(self, input_sequence, output_sequence, buffer, qparams, nqrc, layer_strength):
        self.__init_reservoir(qparams, nqrc, layer_strength)
        self.__train(input_sequence, output_sequence, buffer, qparams.beta)

    def predict(self, input_sequence, output_sequence, buffer, use_lastrho):
        prediction_sequence, _ = self.__feed_forward(input_sequence, \
            predict=True, use_lastrho=use_lastrho)
        pred = prediction_sequence[buffer:, :]
        out  = output_sequence[buffer:, :]
        loss = np.sum((pred - out)**2)/np.sum(pred**2)
        return prediction_sequence, loss

    def init_forward(self, qparams, input_seq, nqrc, layer_strength, init_rs):
        if init_rs == True:
            self.__init_reservoir(qparams, nqrc, layer_strength)
        _, state_list =  self.__feed_forward(input_seq, predict=False)
        return state_list

def get_loss(qrcparams, buffer, train_input_seq, train_output_seq, \
    val_input_seq, val_output_seq, nqrc, layer_strength):
    model = HighorderQuantumReservoirComputing()

    train_input_seq = np.array(train_input_seq)
    train_output_seq = np.array(train_output_seq)
    model.train_to_predict(train_input_seq, train_output_seq, buffer, qrcparams, nqrc, layer_strength)

    train_pred_seq, train_loss = model.predict(train_input_seq, train_output_seq, buffer=buffer, use_lastrho=False)
    #print("train_loss={}, shape".format(train_loss), train_pred_seq_ls.shape)
    
    
    # Test phase
    val_input_seq = np.array(val_input_seq)
    val_output_seq = np.array(val_output_seq)
    val_pred_seq, val_loss = model.predict(val_input_seq, val_output_seq, buffer=0, use_lastrho=True)
    #print("val_loss={}, shape".format(val_loss), val_pred_seq_ls.shape)

    return train_pred_seq, train_loss, val_pred_seq, val_loss

