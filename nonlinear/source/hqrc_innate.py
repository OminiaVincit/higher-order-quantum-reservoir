#!/usr/bin/env python
"""
    Higher-order reservoir class
"""

import sys
import numpy as np
import scipy as sp 
from scipy.sparse import linalg as splinalg
from scipy.linalg import pinv2 as scipypinv2

from utils import *

class HQRC(object):
    def __init__(self, nqrc, alpha):
        self.nqrc = nqrc
        self.alpha = alpha

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

        # initialize connection W_feed
        nqrc  = self.nqrc
        N_local = self.n_units * self.virtual_nodes
        N_tot = nqrc * N_local
        W_feed = np.random.rand(nqrc, N_tot)
        for i in range(nqrc):
            bg = i * N_local
            # skip self-connection
            W_feed[i, bg:(bg + N_local)] = 0
            
            # normalize the row sum
            # rowsum = np.sum(W_feed[i, :])
            # if rowsum > 0:
            #     W_feed[i, :] = W_feed[i, :] / rowsum
        self.W_feed = W_feed

        # initialize current states
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
                rho = random_density_matrix(self.dim)
            tmp_rhos.append(rho)

            # generate hamiltonian
            hamiltonian = np.zeros( (self.dim,self.dim) )

            # include input qubit for computation
            for qindex in range(self.n_qubits):
                coef = (np.random.rand()-0.5) * 2 * self.max_energy
                hamiltonian += coef * self.Zop[qindex]
            for qindex1 in range(self.n_qubits):
                for qindex2 in range(self.n_qubits):
                    if qindex1 != qindex2:
                        coef = (np.random.rand()-0.5) * 2 * self.max_energy
                        hamiltonian += coef * self.Xop[qindex1] @ self.Xop[qindex2]
                    
            ratio = float(self.tau) / float(self.virtual_nodes)        
            Uop = sp.linalg.expm(-1.j * hamiltonian * ratio)
            tmp_uops.append(Uop)
        
        self.init_rhos = tmp_rhos.copy()
        self.last_rhos = tmp_rhos.copy()
        self.Uops = tmp_uops.copy()

    def __init_rhos(self, ranseed, rand_rho=False):
        np.random.seed(seed=ranseed)
        tmp_rhos = []
        for i in range(self.nqrc):
            # initialize density matrix
            rho = np.zeros( [self.dim, self.dim] )
            rho[0, 0] = 1
            if rand_rho == True:
                # initialize random density matrix
                rho = random_density_matrix(self.dim)
            tmp_rhos.append(rho)
        self.init_rhos = tmp_rhos.copy()
        self.last_rhos = tmp_rhos.copy()

    def __init_w_feed(self, ranseed):
        np.random.seed(seed=ranseed)
        nqrc  = self.nqrc
        N_local = self.n_units * self.virtual_nodes
        N_tot = nqrc * N_local
        W_feed = np.random.rand(nqrc, N_tot)
        for i in range(nqrc):
            bg = i * N_local
            # skip self-connection
            W_feed[i, bg:(bg + N_local)] = 0
            
            # normalize the row sum
            # rowsum = np.sum(W_feed[i, :])
            # if rowsum != 0:
            #     W_feed[i, :] = self.alpha * W_feed[i, :] / rowsum
        #W_feed = W_feed * (1/(N_tot - N_local))
        print('W_feed diff:', np.sum(np.abs(W_feed - self.W_feed)))
        self.W_feed = W_feed
        #self.W_feed = np.zeros(W_feed.shape)

    def get_comput_nodes(self):
        return self.n_units * self.virtual_nodes * self.nqrc
    
    def get_local_nodes(self):
        return self.n_units * self.virtual_nodes

    def __reset_states(self):
        self.cur_states  = [None] * self.nqrc

    def __step_forward(self, external_input, innate_train=False, out_train=False, \
            innate_target=None, out_target=None, noise_amp=0.0, noise_rate=10.0):
        nqrc = self.nqrc
        alpha = self.alpha
        N_local = self.get_local_nodes()
        N_tot = self.get_comput_nodes()
        X = self.cur_states
        dW_recurr_mag = 0
        learning_rate = 10.0
        if (X[0] is not None) and innate_train and (innate_target is not None):
            #print('X.shape={}'.format(X.shape))
            # Innate training
            for i in range(nqrc):
                error = X[i*N_local] - innate_target[i]
                #if i == 0:
                print('i={},Error={}'.format(i, error))
                if error != error:
                    continue 
                bg = i * N_local
                ed = bg + N_local

                X_pre_plastic = np.concatenate([X[:bg], X[ed:]])
                X_pre_plastic = X_pre_plastic.reshape(-1, 1)
                P_recurr_old = self.P_recurr[i]
                #print('X_pre_plastic.shape={}, P_recurr={}'.format(X_pre_plastic.shape, P_recurr_old.shape))
                P_recurr_old_X = P_recurr_old @ X_pre_plastic
                den_recurr = 1 + np.dot(X_pre_plastic.ravel(), P_recurr_old_X.ravel())
                #print('P_recurr_X={}, den_recurr={}'.format(P_recurr_old_X.shape, den_recurr))
                self.P_recurr[i] = P_recurr_old - (P_recurr_old_X @ P_recurr_old_X.T)/den_recurr
                # Update W_feed[i, j]
                dw_recurr = - error * (P_recurr_old_X.T/den_recurr) * learning_rate
                dw_recurr = dw_recurr.ravel()
                #print('dw_recurr shape={}'.format(dw_recurr.shape))

                self.W_feed[i, :bg] = self.W_feed[i, :bg] + dw_recurr[:bg]
                self.W_feed[i, ed:] = self.W_feed[i, ed:] + dw_recurr[bg:]

                dW_recurr_mag += np.sqrt(np.dot(dw_recurr, dw_recurr))
        
        # Update the input
        new_input = external_input
        if X[0] is not None:
            X_noise = X + noise_amp * (np.random.rand(len(X)) - 0.5) * 2
            feed_noise = self.W_feed @ X_noise
            feed_noise = feed_noise/len(feed_noise)
            #feed_noise = 1.0 / (1.0 + np.exp(-feed_noise * gamma))
            #print('noise, max={}, min={}'.format(np.max(feed_noise), np.min(feed_noise)))
            # feed_noise  = self.W_feed @ X + noise_amp * (np.random.rand(len(new_input)) - 0.5) * 2
            # To ensure that new_input in [0, 1]
            # print('shape Wfeed={}, cur states={}'.format(self.W_feed.shape, self.cur_states.shape))
            new_input = 0.2 * (external_input + feed_noise)
        #new_input[new_input < 0] = 0
        #new_input[new_input > 1] = 1
        new_input = new_input.flatten()
        print('max={}, min={}, diff={}'.format(np.max(new_input), np.min(new_input), np.sum(np.abs(new_input - external_input))))
        # Evolve the reservoir and update the current states
        cur_states = []
        for i in range(nqrc):
            Uop = self.Uops[i]
            rho = self.last_rhos[i]
            value = new_input[i]
            
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
                    expectation_value = (1.0 + np.real(np.trace(self.Zop[qindex] @ rho))) / 2.0
                    current_state.append(expectation_value)
            # Size of current_state is Nqubits x Nvirtuals
            tmp = np.array(current_state, dtype=np.float64)
            cur_states.append(tmp.copy())
            self.last_rhos[i] = rho

        self.cur_states = np.array(cur_states).ravel()
        return dW_recurr_mag

    def innate_train(self, input_seq, innate_seq, buffer, ranseed, \
            learn_every=1, noise_amp=0.0, noise_rate=10.0, train_loops=1):
        self.__init_w_feed(ranseed=ranseed)
        # Initialize P_recurr
        Plist = []
        nqrc = self.nqrc
        for i in range(nqrc):
            Plist.append(np.eye(self.get_local_nodes() * (nqrc-1)))
        self.P_recurr = Plist

        for k in range(train_loops):
            self.__reset_states()
            self.__init_rhos(ranseed=ranseed + k, rand_rho=True)

            # Stage 1: Transient regime
            _, state_list = self.__feed_forward(input_seq[:, :buffer], predict=False, noise_amp=noise_amp, noise_rate=noise_rate)
            dW_recurr_ls = [0] * buffer
            # Stage 2: Innate training for W_feed
            tmp_list = []
            for i in range(buffer, input_seq.shape[1]):
                if i % learn_every == 0:
                    innate_train = True
                else:
                    innate_train = False
                #innate_train = False
                dW_recurr_mag = self.__step_forward(input_seq[:, i], innate_train=innate_train, innate_target=innate_seq[i-1],\
                    noise_amp=noise_amp, noise_rate=noise_rate)
                dW_recurr_ls.append(dW_recurr_mag)
                state = np.array(self.cur_states.copy(), dtype=np.float64)
                tmp_list.append(state.flatten())
            tmp_list = np.array(tmp_list)
            state_list = np.concatenate([state_list, tmp_list])
        return state_list, dW_recurr_ls

    def __train_to_predict(self, input_seq, output_seq, innate_seq, \
        buffer, innate_end_index, train_end_index):
        assert(input_seq.shape[1] == output_seq.shape[0])



        # Stage 3: Readout training
        Nout = output_seq.shape[1]
        self.W_out = np.random.rand(self.get_comput_nodes() + 1, Nout)

        for i in range(innate_end_index, train_end_index):
            self.__step_forward(input_seq[:, i], out_train=True, out_target=output_seq[i]) 
        
        # Stage 4: Predict
        state_list = []
        for i in range(train_end_index, input_seq.shape[1]):
            self.__step_forward(input_seq[i])
            state = np.array(self.cur_states.copy(), dtype=np.float64)
            state_list.append(state.flatten())
        
        stacked_state = np.hstack( [state_list, np.ones([input_length, 1])])
        pred = stacked_state @ self.W_out
        out  = output_seq[train_end_index:, :]
        loss = np.sum((pred - out)**2)/np.sum(pred**2)
        print('Loss={}'.format(loss))
        return pred, loss

    def train_to_predict(self, input_seq, output_seq, innate_seq, \
        buffer, innate_end_index, train_end_index, qparams, ranseed):
        self.__init_reservoir(qparams, ranseed)
        self.__train_to_predict(input_seq, output_seq, innate_seq, \
            buffer, innate_end_index, train_end_index)

    def __feed_forward(self, input_seq, predict, noise_amp=0.0, noise_rate=10.0):
        input_dim, input_length = input_seq.shape
        nqrc = self.nqrc
        assert(input_dim == nqrc)
        
        predict_seq = None
        state_list = []
        for time_step in range(0, input_length):
            input_val = input_seq[:, time_step].ravel()
            self.__step_forward(input_val, noise_amp=noise_amp, noise_rate=noise_rate)
            state = np.array(self.cur_states.copy(), dtype=np.float64)
            state_list.append(state.flatten())

        state_list = np.array(state_list)

        if predict:
            stacked_state = np.hstack( [state_list, np.ones([input_length, 1])])
            predict_seq = stacked_state @ self.W_out
        
        return predict_seq, state_list


    def init_forward(self, qparams, input_seq, ranseed):
        self.__reset_states()
        self.__init_reservoir(qparams, ranseed)
        _, state_list =  self.__feed_forward(input_seq, predict=False)
        return state_list
