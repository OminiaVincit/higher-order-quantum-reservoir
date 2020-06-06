import sys
import numpy as np
import scipy as sp 
from scipy.sparse import linalg as splinalg
from scipy.linalg import pinv2 as scipypinv2

from utils import *

class HQRC(object):
    def __init__(self, nqrc, alpha, deep=False, bias=1.0):
        self.nqrc = nqrc
        self.alpha = alpha
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
        self.n_units = qparams.n_units
        self.virtual_nodes = qparams.virtual_nodes
        self.tau = qparams.tau
        self.max_energy = qparams.max_energy
        self.solver = qparams.solver

        self.n_qubits = self.n_units
        self.dim = 2**self.n_qubits
        self.Zop = [1]*self.n_qubits
        self.Xop = [1]*self.n_qubits
        self.P0op = [1]
        self.P1op = [1]
        
        # create operators from tensor product
        for cindex in range(self.n_qubits):
            for qindex in range(self.n_qubits):
                if cindex == qindex:
                    self.Xop[qindex] = np.kron(self.Xop[qindex],X)
                    self.Zop[qindex] = np.kron(self.Zop[qindex],Z)
                else:
                    self.Xop[qindex] = np.kron(self.Xop[qindex],I)
                    self.Zop[qindex] = np.kron(self.Zop[qindex],I)

            if cindex == 0:
                self.P0op = np.kron(self.P0op, P0)
                self.P1op = np.kron(self.P1op, P1)
            else:
                self.P0op = np.kron(self.P0op, I)
                self.P1op = np.kron(self.P1op, I)

        # initialize connection to layer i
        connections = []
        N_local_states = self.n_units * self.virtual_nodes
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
        self.prev_states = [None] * nqrc
        self.cur_states  = [None] * nqrc

        # Intialize evolution operators
        tmp_uops = []
        tmp_rhos = []
        for i in range(nqrc):
            # initialize density matrix
            rho = np.zeros( [self.dim, self.dim] )
            rho[0, 0] = 1
            if qparams.init_rho != 0:
                # initialize random density matrix
                rho = gen_random.random_density_matrix(self.dim)
            tmp_rhos.append(rho)

            # generate hamiltonian
            hamiltonian = np.zeros( (self.dim,self.dim) )

            # include input qubit for computation
            for qindex in range(self.n_qubits):
                coef = (np.random.rand()-0.5) * 2 * self.max_energy
                hamiltonian += coef * self.Zop[qindex]
            for qindex1 in range(self.n_qubits):
                for qindex2 in range(qindex1+1, self.n_qubits):
                    coef = (np.random.rand()-0.5) * 2 * self.max_energy
                    hamiltonian += coef * self.Xop[qindex1] @ self.Xop[qindex2]
                    
            ratio = float(self.tau) / float(self.virtual_nodes)        
            Uop = sp.linalg.expm(-1.j * hamiltonian * ratio)
            tmp_uops.append(Uop)
        
        self.init_rhos = tmp_rhos.copy()
        self.last_rhos = tmp_rhos.copy()
        self.Uops = tmp_uops.copy()

    def __get_comput_nodes(self):
        return self.n_units * self.virtual_nodes * self.nqrc
    
    def __reset_states(self):
        self.prev_states = [None] * self.nqrc
        self.cur_states  = [None] * self.nqrc

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
            value = input_val[i]
            prev_states = self.prev_states
            if nqrc > 1 and prev_states[0] is not None:
                scaled_coeffs = self.coeffs[i] * self.alpha
                value = scale_linear_combine(value, prev_states, scaled_coeffs, self.bias)
            
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
                for qindex in range(0, self.n_qubits):
                    expectation_value = np.real(np.trace(self.Zop[qindex] @ rho))
                    current_state.append(expectation_value)
            # Size of current_state is Nqubits x Nvirtuals
            tmp = np.array(current_state, dtype=np.float64)
            local_prev_states.append(tmp)
            self.cur_states[i] = tmp.copy()
            local_rhos[i] = rho
        # update previous states
        if any(x is None for x in local_prev_states) == False:
            self.prev_states = np.array(local_prev_states, dtype=np.float64).flatten()
        return local_rhos

    def __feed_forward(self, input_seq, predict, use_lastrho):
        input_dim, input_length = input_seq.shape
        nqrc = self.nqrc
        assert(input_dim == nqrc)
        
        predict_seq = None
        local_rhos = self.init_rhos.copy()
        if use_lastrho == True :
            local_rhos = self.last_rhos.copy()
        
        state_list = []
        for time_step in range(0, input_length):
            input_val = input_seq[:, time_step].ravel()
            local_rhos = self.step_forward(local_rhos, input_val)

            state = np.array(self.cur_states.copy(), dtype=np.float64)
            state_list.append(state.flatten())

        state_list = np.array(state_list)
        self.last_rhos = local_rhos.copy()

        if predict:
            stacked_state = np.hstack( [state_list, np.ones([input_length, 1])])
            predict_seq = stacked_state @ self.W_out
        
        return predict_seq, state_list


    def __train(self, input_seq, output_seq, buffer, beta):
        assert(input_seq.shape[1] == output_seq.shape[0])
        Nout = output_seq.shape[1]
        self.W_out = np.random.rand(self.__get_comput_nodes() + 1, Nout)

        _, state_list = self.__feed_forward(input_seq, predict=False, use_lastrho=False)

        state_list = np.array(state_list)
        state_list = state_list[buffer:, :]

        # discard the transitient state for training
        X = np.reshape(state_list, [-1, self.__get_comput_nodes()])
        X = np.hstack( [state_list, np.ones([X.shape[0], 1]) ] )

        discard_output = output_seq[buffer:, :]
        Y = np.reshape(discard_output, [discard_output.shape[0], -1])
        
        if self.solver == LINEAR_PINV:
            self.W_out = np.linalg.pinv(X, rcond = beta) @ Y
        else:
            XTX = X.T @ X
            XTY = X.T @ Y
            if self.solver == RIDGE_PINV:
                I = np.identity(np.shape(XTX)[1])	
                pinv_ = scipypinv2(XTX + self.regularization * I)
                W_out = pinv_ @ XTY
            elif self.solver in ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']:
                ridge = Ridge(alpha=self.beta, fit_intercept=False, normalize=False, copy_X=True, solver=self.solver)
                ridge.fit(XTX, XTY)
                W_out = np.array(ridge.coef_).reshape((-1, Nout))
            else:
                raise ValueError('Undefined solver')

    def train_to_predict(self, input_seq, output_seq, buffer, qparams, ranseed):
        self.__init_reservoir(qparams, ranseed)
        self.__train(input_seq, output_seq, buffer, qparams.beta)

    def predict(self, input_seq, output_seq, buffer, use_lastrho):
        prediction_seq, _ = self.__feed_forward(input_seq, \
            predict=True, use_lastrho=use_lastrho)
        pred = prediction_seq[buffer:, :]
        out  = output_seq[buffer:, :]
        loss = np.sum((pred - out)**2)/np.sum(pred**2)
        return prediction_seq, loss

    def init_forward(self, qparams, input_seq, init_rs, ranseed):
        self.__reset_states()
        if init_rs == True:
            self.__init_reservoir(qparams, ranseed)
        _, state_list =  self.__feed_forward(input_seq, predict=False, use_lastrho=False)
        return state_list

def get_loss(qparams, buffer, train_input_seq, train_output_seq, \
        val_input_seq, val_output_seq, nqrc, alpha, ranseed, deep=False):
    model = HQRC(nqrc, alpha, deep)

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
        nqrc, alpha, ranseed=-1, Ntrials=1, deep=False):    
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
            # Use the same ranseed the same trial
            train_pred_seq, train_loss, val_pred_seq, val_loss = \
                get_loss(qparams, buffer, train_input_seq, train_output_seq, \
                    val_input_seq, val_output_seq, nqrc, alpha, ranseed_net, deep)

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
        MFlist.append(avg_MFd)
        MFstds.append(std_MFd)
        train_list.append(avg_train)
        val_list.append(avg_val)
    
    return np.array(list(zip(dlist, MFlist, MFstds, train_list, val_list)))

def effective_dim(qparams, buffer, length, nqrc, alpha, ranseed, Ntrials, deep=False):
    # Calculate effective dimension for reservoir
    from numpy import linalg as LA
    
    if ranseed >= 0:
        np.random.seed(seed=ranseed)

    data = np.random.rand(length)
    input_seq = np.array(data)
    input_seq = np.tile(input_seq, (nqrc, 1))

    model = HQRC(nqrc, alpha, deep)

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

def esp_index(qparams, buffer, length, nqrc, alpha, ranseed, state_trials, deep=False):
    if ranseed >= 0:
        np.random.seed(seed=ranseed)

    data = np.random.rand(length)
    input_seq = np.array(data)
    input_seq = np.tile(input_seq, (nqrc, 1))

    # Initialize the reservoir to zero state - density matrix
    model = HQRC(nqrc, alpha, deep)
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
            local_diff += diff
        local_diff = local_diff / (L-buffer)
        dP.append(local_diff)
    return np.mean(dP)

def lyapunov_exp(qparams, buffer, length, nqrc, alpha, ranseed, initial_distance, deep=False):
    if ranseed >= 0:
        np.random.seed(seed=ranseed)

    data = np.random.rand(length)
    input_seq = np.array(data)
    input_seq = np.tile(input_seq, (nqrc, 1))

    # Initialize the reservoir to zero state - density matrix
    model = HQRC(nqrc, alpha, deep)
    states1 = model.init_forward(qparams, input_seq, init_rs = True, ranseed = -1)
    L, D = states1.shape
    # L = Length of time series
    # D = Number of layers x Number of virtual nodes x Number of qubits
    lyps = []
    for n in range(int(D / nqrc)):
        if n % qparams.n_units == 0:
            # Skip the input qubits
            continue
        model.init_forward(qparams, input_seq[:buffer], init_rs = False, ranseed = -1)
        states2 = np.zeros((L, D))
        states2[buffer-1, :] = states1[buffer-1, :]
        states2[buffer-1, n] = states1[buffer-1, n] + initial_distance
        gamma_k_list = []
        local_rhos = model.last_rhos.copy()
        for k in range(buffer, L):
            # Update prev states
            model.prev_states = states2[k-1, :].copy()
            input_val = input_seq[:, k].ravel()
            local_rhos = model.step_forward(local_rhos, input_val)
            states2[k, :] = np.array(model.cur_states, dtype=np.float64).flatten()
            # Add to gamma list and update states
            gamma_k = np.linalg.norm(states2[k, :] - states1[k, :])
            gamma_k_list.append(gamma_k / initial_distance)
            states2[k, :] = states1[k, :] + (initial_distance / gamma_k) * (states2[k, :] - states1[k, :])

        lyps.append(np.mean(np.log(gamma_k_list)))
    lyapunov_exp = np.mean(lyps)
    return lyapunov_exp