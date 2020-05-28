import sys
import numpy as np
import scipy as sp 
import utils
import qrc
import gen_random
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

def scale_linear_combine(u, states, coeffs, bias):
    states = (states + bias) / (2.0 * bias)
    return linear_combine(u, states, coeffs)

class HighorderQuantumReservoirComputing(object):
    def __init__(self, nqrc, layer_strength, one_input=False, deep=False, bias=1.0):
        self.nqrc = nqrc
        self.layer_strength = layer_strength
        self.one_input = one_input
        self.bias = bias
        self.deep = deep

    def __init_reservoir(self, qparams, ranseed):
        if ranseed >= 0:
            np.random.seed(seed=ranseed)
        
        I = [[1,0],[0,1]]
        Z = [[1,0],[0,-1]]
        X = [[0,1],[1,0]]
        P0 = [[1,0],[0,0]]
        P1 = [[0,0],[0,1]]
        self.hidden_unit_count = qparams.hidden_unit_count
        self.trotter_step = qparams.trotter_step
        self.virtual_nodes = qparams.virtual_nodes
        self.tau_delta = qparams.tau_delta
        self.qubit_count = self.hidden_unit_count
        self.max_coupling_energy = qparams.max_coupling_energy
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

        # initialize connection to layer i
        connections = []
        N_local_states = self.hidden_unit_count * self.virtual_nodes
        nqrc = self.nqrc
        if nqrc > 1:
            for i in range(nqrc):
                local_cs = []
                for j in range(nqrc):
                    cs = [0] * N_local_states
                    if self.deep == False:
                        if j != i:
                            cs = np.random.rand(N_local_states)
                    else:
                        if j == i-1:
                            cs = np.random.rand(N_local_states)
                    local_cs.append(cs)
                
                local_cs = np.array(local_cs).flatten()
                total = np.sum(local_cs)
                if total > 0:
                    local_cs = local_cs / total
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
                coef = (np.random.rand()-0.5) * 2 * self.max_coupling_energy
                hamiltonian += coef * self.Zop[qubit_index]
            for qubit_index1 in range(self.qubit_count):
                for qubit_index2 in range(qubit_index1+1, self.qubit_count):
                    coef = (np.random.rand()-0.5) * 2 * self.max_coupling_energy
                    hamiltonian += coef * self.Xop[qubit_index1] @ self.Xop[qubit_index2]
                    
            ratio = float(self.tau_delta) / float(self.virtual_nodes)        
            Uop = sp.linalg.expm(-1.j * hamiltonian * ratio)
            tmp_uops.append(Uop)
        
        self.init_rhos = tmp_rhos.copy()
        self.last_rhos = tmp_rhos.copy()
        self.Uops = tmp_uops.copy()

    def __reset_states(self):
        self.previous_states = [None] * self.nqrc
        self.current_states  = [None] * self.nqrc

    def gen_rand_rhos(self, ranseed):
        if ranseed >= 0:
            np.random.seed(seed=ranseed)
        tmp_rhos = []
        for i in range(self.nqrc):
            rho = gen_random.random_density_matrix(self.dim)
            tmp_rhos.append(rho)
        self.init_rhos = tmp_rhos.copy()

    def step_forward(self, local_rhos, input_val):
        nqrc = self.nqrc
        local_prev_states = []
        for i in range(nqrc):
            Uop = self.Uops[i]
            rho = local_rhos[i]
            # Obtain value from the input
            value = 0
            if self.one_input <= 0 or i == 0:
                value = input_val[i]
            prev_states = self.previous_states
            if nqrc > 1 and prev_states[0] is not None:
                scaled_coeffs = self.coeffs[i] * self.layer_strength
                value = scale_linear_combine(value, prev_states, scaled_coeffs, self.bias)
            #print(i, input_val[i], value)
            # Replace the density matrix
            rho = self.P0op @ rho @ self.P0op + self.Xop[0] @ self.P1op @ rho @ self.P1op @ self.Xop[0]
            # (1 + u Z)/2 = (1+u)/2 |0><0| + (1-u)/2 |1><1|
        
            # for input in [-1, 1]
            # rho = (1+value)/2 * rho + (1-value)/2 *self.Xop[0] @ rho @ self.Xop[0]

            rho = (1 - value) * rho + value * self.Xop[0] @ rho @ self.Xop[0]
            current_state = []
            for v in range(self.virtual_nodes):
                # Time evolution of density matrix
                rho = Uop @ rho @ Uop.T.conj()
                for qubit_index in range(0, self.qubit_count):
                    expectation_value = np.real(np.trace(self.Zop[qubit_index] @ rho))
                    current_state.append(expectation_value)
            # Size of current_state is Nqubits x Nvirtuals
            tmp = np.array(current_state, dtype=np.float64)
            local_prev_states.append(tmp)
            self.current_states[i] = tmp.copy()
            local_rhos[i] = rho
        # update previous states
        if any(x is None for x in local_prev_states) == False:
            self.previous_states = np.array(local_prev_states, dtype=np.float64).flatten()
        return local_rhos

    def __feed_forward(self, input_sequence, predict, use_lastrho):
        input_dim, input_length = input_sequence.shape
        assert(input_dim == self.nqrc)
        
        predict_sequence = None
        local_rhos = self.init_rhos.copy()
        if use_lastrho == True :
            #print('Use last density matrix')
            local_rhos = self.last_rhos.copy()
        nqrc = self.nqrc

        state_list = []
        for time_step in range(0, input_length):
            input_val = input_sequence[:, time_step].ravel()
            local_rhos = self.step_forward(local_rhos, input_val)

            state = np.array(self.current_states.copy(), dtype=np.float64)
            state_list.append(state.flatten())

        state_list = np.array(state_list)
        self.last_rhos = local_rhos.copy()

        if predict:
            stacked_state = np.hstack( [state_list, np.ones([input_length, 1])])
            #print('stacked state {}; Wout {}'.format(stacked_state.shape, self.W_out.shape))
            predict_sequence = stacked_state @ self.W_out
        
        return predict_sequence, state_list


    def __train(self, input_sequence, output_sequence, buffer, beta):
        #print('shape', input_sequence.shape, output_sequence.shape)
        assert(input_sequence.shape[1] == output_sequence.shape[0])
        Nout = output_sequence.shape[1]
        self.W_out = np.random.rand(self.hidden_unit_count * self.virtual_nodes * self.nqrc + 1, Nout)

        _, state_list = self.__feed_forward(input_sequence, predict=False, use_lastrho=False)

        state_list = np.array(state_list)
        #print('before washingout state list shape', state_list.shape)
        
        state_list = state_list[buffer:, :]
        #print('after washingout state list shape', state_list.shape)

        # discard the transitient state for training
        V = np.reshape(state_list, [-1, self.hidden_unit_count * self.virtual_nodes * self.nqrc])
        V = np.hstack( [state_list, np.ones([V.shape[0], 1]) ] )

        #print('output seq', output_sequence.shape)
        discard_output = output_sequence[buffer:, :]
        #print('discard output seq', discard_output.shape)
        #S = np.reshape(output_sequence_list, [-1])
        S = np.reshape(discard_output, [discard_output.shape[0], discard_output.shape[1]])
        #print('V S', V.shape, S.shape)
        self.W_out = np.linalg.pinv(V, rcond = beta) @ S
        #print('bf Wout', self.W_out.shape)
        
    def train_to_predict(self, input_sequence, output_sequence, buffer, qparams, ranseed):
        self.__init_reservoir(qparams, ranseed)
        self.__train(input_sequence, output_sequence, buffer, qparams.beta)

    def predict(self, input_sequence, output_sequence, buffer, use_lastrho):
        prediction_sequence, _ = self.__feed_forward(input_sequence, \
            predict=True, use_lastrho=use_lastrho)
        pred = prediction_sequence[buffer:, :]
        out  = output_sequence[buffer:, :]
        loss = np.sum((pred - out)**2)/np.sum(pred**2)
        return prediction_sequence, loss

    def init_forward(self, qparams, input_seq, init_rs, ranseed):
        self.__reset_states()
        if init_rs == True:
            self.__init_reservoir(qparams, ranseed)
        _, state_list =  self.__feed_forward(input_seq, predict=False, use_lastrho=False)
        return state_list

def get_loss(qparams, buffer, train_input_seq, train_output_seq, \
    val_input_seq, val_output_seq, nqrc, layer_strength, ranseed, one_input=False, deep=False):
    model = HighorderQuantumReservoirComputing(nqrc, layer_strength, one_input, deep)

    train_input_seq = np.array(train_input_seq)
    train_output_seq = np.array(train_output_seq)
    model.train_to_predict(train_input_seq, train_output_seq, buffer, qparams, ranseed)

    train_pred_seq, train_loss = model.predict(train_input_seq, train_output_seq, buffer=buffer, use_lastrho=False)
    #print("train_loss={}, shape".format(train_loss), train_pred_seq_ls.shape)
    
    # Test phase
    val_input_seq = np.array(val_input_seq)
    val_output_seq = np.array(val_output_seq)
    val_pred_seq, val_loss = model.predict(val_input_seq, val_output_seq, buffer=0, use_lastrho=True)
    #print("val_loss={}, shape".format(val_loss), val_pred_seq_ls.shape)

    return train_pred_seq, train_loss, val_pred_seq, val_loss

def memory_function(taskname, qparams, train_len, val_len, buffer, dlist, \
        nqrc, layer_strength, ranseed=-1, Ntrials=1, one_input=False, deep=False):    
    MFlist = []
    MFstds = []
    train_list, val_list = [], []
    length = buffer + train_len + val_len
    # generate data
    if '_stm' not in taskname and '_pc' not in taskname:
        raise ValueError('Not found taskname ={} to generate data'.format(taskname))

    if ranseed >= 0:
        np.random.seed(seed=ranseed)
    
    if '_pc' in taskname:
        print('Generate parity check data')
        data = np.random.randint(0, 2, length)
    else:
        print('Generate STM task data')
        data = np.random.rand(length)

    for d in dlist:
        train_input_seq = np.array(data[  : buffer + train_len])
        train_input_seq = np.tile(train_input_seq, (nqrc, 1))
        
        val_input_seq = np.array(data[buffer + train_len : length])
        val_input_seq = np.tile(val_input_seq, (nqrc, 1))
            
        train_out, val_out = [], []
        if '_pc' in taskname:
            for k in range(length):
                yk = 0
                if k >= d:
                    yk = np.sum(data[k-d : k+1]) % 2
                if k >= buffer + train_len:
                    val_out.append(yk)
                else:
                    train_out.append(yk)
        else:
            for k in range(length):
                yk = 0
                if k >= d:
                    yk = data[k-d]
                if k >= buffer + train_len:
                    val_out.append(yk)
                else:
                    train_out.append(yk)
        
        train_output_seq = np.array(train_out).reshape(len(train_out), 1)
        val_output_seq = np.array(val_out).reshape(len(val_out), 1)
        
        train_loss_ls, val_loss_ls, mfs = [], [], []
        for n in range(Ntrials):
            ranseed_net = ranseed
            if ranseed >= 0:
                ranseed_net = (ranseed + 10000) * (n + 1)
            #print('d={}, trial={}'.format(d, n))
            # Use the same ranseed the same trial
            train_pred_seq, train_loss, val_pred_seq, val_loss = \
                get_loss(qparams, buffer, train_input_seq, train_output_seq, \
                    val_input_seq, val_output_seq, nqrc, layer_strength, ranseed_net, one_input, deep)

            # Compute memory function
            val_out_seq, val_pred_seq = val_output_seq.flatten(), val_pred_seq.flatten()
            #print('cov', val_output_seq.shape, val_pred_seq.shape)
            cov_matrix = np.cov(np.array([val_out_seq, val_pred_seq]))
            MF_d = cov_matrix[0][1] ** 2
            MF_d = MF_d / (np.var(val_out_seq) * np.var(val_pred_seq))
            # print('d={}, n={}, MF={}'.format(d, n, MF_d))
            train_loss_ls.append(train_loss)
            val_loss_ls.append(val_loss)
            mfs.append(MF_d)

        avg_train, avg_val, avg_MFd, std_MFd = np.mean(train_loss_ls), np.mean(val_loss_ls), np.mean(mfs), np.std(mfs)
        #print("d={}, train_loss={}, val_loss={}, MF={}".format(d, avg_train, avg_val, avg_MFd))
        MFlist.append(avg_MFd)
        MFstds.append(std_MFd)
        train_list.append(avg_train)
        val_list.append(avg_val)
    
    return np.array(list(zip(dlist, MFlist, MFstds, train_list, val_list)))

def effective_dim(qparams, buffer, length, nqrc, layer_strength, ranseed, Ntrials, one_input=False, deep=False):
    # Calculate effective dimension for reservoir
    from numpy import linalg as LA
    
    if ranseed >= 0:
        np.random.seed(seed=ranseed)

    data = np.random.rand(length)
    input_seq = np.array(data)
    input_seq = np.tile(input_seq, (nqrc, 1))

    model = HighorderQuantumReservoirComputing(nqrc, layer_strength, one_input, deep)

    effdims = []
    for n in range(Ntrials):
        ranseed_net = ranseed
        if ranseed >= 0:
            ranseed_net = (ranseed + 11000) * (n + 1)
        
        state_list = model.init_forward(qparams, input_seq, init_rs=True, ranseed=ranseed_net)
        L, D = state_list.shape
        # L = Length of time series
        # D = Number of virtual nodes x Number of qubits
        locls = []
        for i in range(D):
            for j in range(D):
                ri = state_list[buffer:, i]
                rj = state_list[buffer:, j]
                locls.append(np.mean(ri*rj))
        locls = np.array(locls).reshape(D, D)
        w, v = LA.eig(locls)
        #print(w)
        w = np.abs(w) / np.abs(w).sum()
        effdims.append(1.0 / np.power(w, 2).sum())
    return np.mean(effdims), np.std(effdims)

def esp_index(qparams, buffer, length, nqrc, layer_strength, ranseed, state_trials, one_input=False, deep=False):
    if ranseed >= 0:
        np.random.seed(seed=ranseed)

    data = np.random.rand(length)
    input_seq = np.array(data)
    input_seq = np.tile(input_seq, (nqrc, 1))

    # Initialize the reservoir to zero state - density matrix
    model = HighorderQuantumReservoirComputing(nqrc, layer_strength, one_input, deep)
    x0_state_list = model.init_forward(qparams, input_seq, init_rs = True, ranseed = ranseed)
    # Compute esp index and esp_lambda
    dP = []
    for i in range(state_trials):
        # Initialzie the reservoir to a random initial state
        # Keep same coupling configuration
        model.gen_rand_rhos(ranseed = i + 300000)
        z0_state_list = model.init_forward(qparams, input_seq, init_rs = False, ranseed = i + 200000)
        L, D = z0_state_list.shape
        # L = Length of time series
        # D = Number of layers x Number of virtual nodes x Number of qubits
        # print('i={}, State shape'.format(i), z0_state_list.shape)
        local_diff = 0
        # prev, current = None, None
        for t in range(buffer, L):
            diff_state = x0_state_list[t, :] - z0_state_list[t, :]
            diff = np.sqrt(np.power(diff_state, 2).sum())
            #prev = current
            #current = diff
            #if prev is not None:
            #    tmp = np.log(diff / prev)
                #print('t={},diff={},lambda={},tau_delta={}'.format(t, diff, tmp, qparams.tau_delta))
                #lda = max(lda, tmp)
            #    local_lda += tmp
            local_diff += diff
        local_diff = local_diff / (L-buffer)
        #local_lda  = local_lda / (L-buffer)
        #print('i={}, avg Delta={}'.format(i, local_diff))
        dP.append(local_diff)
        #mlda.append(local_lda)
    return np.mean(dP)

def lyapunov_exp(qparams, buffer, length, nqrc, layer_strength, ranseed, \
    initial_distance, one_input=False, deep=False):
    if ranseed >= 0:
        np.random.seed(seed=ranseed)

    data = np.random.rand(length)
    input_seq = np.array(data)
    input_seq = np.tile(input_seq, (nqrc, 1))

    # Initialize the reservoir to zero state - density matrix
    model = HighorderQuantumReservoirComputing(nqrc, layer_strength, one_input, deep)
    states1 = model.init_forward(qparams, input_seq, init_rs = True, ranseed = -1)
    L, D = states1.shape
    # L = Length of time series
    # D = Number of layers x Number of virtual nodes x Number of qubits
    lyps = []
    for n in range(int(D / nqrc)):
        model.init_forward(qparams, input_seq[:buffer], init_rs = False, ranseed = -1)
        states2 = np.zeros((L, D))
        states2[buffer-1, :] = states1[buffer-1, :]
        states2[buffer-1, n] = states1[buffer-1, n] + initial_distance
        gamma_k_list = []
        local_rhos = model.last_rhos.copy()
        for k in range(buffer, L):
            # Update prev states
            model.previous_states = states2[k-1, :].copy()
            input_val = input_seq[:, k].ravel()
            local_rhos = model.step_forward(local_rhos, input_val)
            states2[k, :] = np.array(model.current_states, dtype=np.float64).flatten()
            # Add to gamma list and update states
            gamma_k = np.linalg.norm(states2[k, :] - states1[k, :])
            gamma_k_list.append(gamma_k / initial_distance)
            states2[k, :] = states1[k, :] + (initial_distance / gamma_k) * (states2[k, :] - states1[k, :])

        lyps.append(np.mean(np.log(gamma_k_list)))
    lyapunov_exp = np.mean(lyps)
    return lyapunov_exp