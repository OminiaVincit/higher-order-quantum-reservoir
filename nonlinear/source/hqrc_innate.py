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
        self.W_feed = None
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

        nqrc  = self.nqrc
        
        # initialize current states
        self.cur_states  = [None] * nqrc

        # initialize connection W_feed
        self.__init_w_feed(ranseed)
        
        # initialize density matrix
        self.__init_rhos(ranseed, qparams.init_rho)

        # Intialize evolution operators
        self.__init_spins(ranseed)

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

    def __init_w_feed(self, ranseed, reset_zero=False):
        nqrc  = self.nqrc
        N_local = self.n_units * self.virtual_nodes
        N_tot = nqrc * N_local
        if reset_zero == True:
            self.W_feed = np.zeros((nqrc, N_tot))
        else:
            np.random.seed(seed=ranseed)
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
            if self.W_feed is not None:
                print('W_feed diff:', np.sum(np.abs(W_feed - self.W_feed)))
            self.W_feed = W_feed
        
        # else:
        #     # normalize the row sum
        #     rowsum = np.sum(W_feed[i, :])
        #     if rowsum != 0:
        #         W_feed[i, :] = self.alpha * W_feed[i, :] / rowsum
        #    self.W_feed = np.zeros(W_feed.shape)
        

    def __init_spins(self, ranseed):
        np.random.seed(seed=ranseed)
        tmp_uops = []
        J = self.max_energy
        for i in range(self.nqrc):
            # generate hamiltonian
            hamiltonian = np.zeros( (self.dim,self.dim) )

            # include input qubit for computation
            for qindex in range(self.n_qubits):
                coef = (np.random.rand()-0.5) * 2 * J
                hamiltonian += coef * self.Zop[qindex]
            for qindex1 in range(self.n_qubits):
                for qindex2 in range(self.n_qubits):
                    if qindex1 != qindex2:
                        coef = (np.random.rand()-0.5) * 2 * J
                        hamiltonian += coef * self.Xop[qindex1] @ self.Xop[qindex2]
                    
            ratio = float(self.tau) / float(self.virtual_nodes)        
            Uop = sp.linalg.expm(-1.j * hamiltonian * ratio)
            tmp_uops.append(Uop)
        self.Uops = tmp_uops.copy()

    def get_comput_nodes(self):
        return self.n_units * self.virtual_nodes * self.nqrc
    
    def get_local_nodes(self):
        return self.n_units * self.virtual_nodes

    def __reset_states(self):
        self.cur_states  = [None] * self.nqrc

    def __step_forward(self, external_input, innate_train=False, out_train=False, \
            innate_target=None, out_target=None, noise_amp=0.0, learning_rate=10.0, scale_input=0.4, sel=0):
        nqrc = self.nqrc
        alpha = self.alpha
        N_local = self.get_local_nodes()
        N_tot = self.get_comput_nodes()
        X = self.cur_states
        dW_recurr_mag = 0
        if (X[0] is not None) and innate_train and (innate_target is not None) and learning_rate > 0.0:
            #print('X.shape={}'.format(X.shape))
            # Innate training
            for i in range(nqrc):
                error = (X[sel + i*N_local] - innate_target[i])
                # error = 0.0
                # for v in range(self.virtual_nodes):
                #     error += np.abs(X[i*N_local + v * self.n_qubits] - innate_target[i + v])
                # error = error / self.virtual_nodes

                #if i == 0:
                #print('i={},Error={}'.format(i, error))
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
                dw_recurr = error * (P_recurr_old_X.T/den_recurr) * learning_rate
                dw_recurr = dw_recurr.ravel()
                #print('dw_recurr shape={}'.format(dw_recurr.shape))

                self.W_feed[i, :bg] = self.W_feed[i, :bg] + dw_recurr[:bg]
                self.W_feed[i, ed:] = self.W_feed[i, ed:] + dw_recurr[bg:]

                dW_recurr_mag += np.sqrt(np.dot(dw_recurr, dw_recurr))
        
        # Update the input
        new_input = external_input
        if X[0] is not None:
            #noise = np.random.normal(loc=0.0, scale=noise_amp, size=len(X))
            #noise = (np.random.rand(len(X)) - 0.5) * 2.0 * noise_amp
            # print('Xmax={}, Xmin={}, noise max={}, min={}'.format(\
            #     np.max(X), np.min(X),\
            #     np.max(noise), np.min(noise)))

            #X_noise = X + noise_amp * noise
            #X_noise = X * np.random.rand(len(X)) * noise_amp
            #X_noise = X + noise
            X_noise = X
            feed_noise = self.W_feed @ X_noise
            feed_noise = feed_noise/(2*len(feed_noise))
            
            #feed_noise = 1.0 / (1.0 + np.exp(-feed_noise * gamma))
            #print('ext={}, noise={}'.format(external_input, feed_noise))
            # feed_noise  = self.W_feed @ X + noise_amp * (np.random.rand(len(new_input)) - 0.5) * 2
            # To ensure that new_input in [0, 1]
            # print('shape Wfeed={}, cur states={}'.format(self.W_feed.shape, self.cur_states.shape))
            # feed_noise = noise_amp
            new_input = scale_input * (external_input + feed_noise)
        #new_input[new_input < 0] = 0
        #new_input[new_input > 1] = 1
        new_input = new_input.flatten()
        maxval, minval = np.max(new_input), np.min(new_input)
        if maxval > 1.0 or minval < 0.0:
            print('max={}, min={}, diff={}'.format(maxval, minval, np.sum(np.abs(new_input - external_input))))
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
        
        noise = np.random.normal(loc=0.0, scale=np.sqrt(noise_amp), size=len(self.cur_states))
        #noise = (np.random.rand(len(self.cur_states)) - 0.5) * 2.0 * noise_amp
        self.cur_states += noise
        return dW_recurr_mag

    def __feed_forward(self, input_seq, predict, noise_amp, scale_input, sel=0):
        input_dim, input_length = input_seq.shape
        nqrc = self.nqrc
        assert(input_dim == nqrc)
        
        predict_seq = None
        state_list = []
        for time_step in range(0, input_length):
            input_val = input_seq[:, time_step].ravel()
            self.__step_forward(input_val, noise_amp=noise_amp, scale_input=scale_input, sel=sel)
            state = np.array(self.cur_states.copy(), dtype=np.float64)
            state_list.append(state.flatten())

        state_list = np.array(state_list)

        if predict:
            stacked_state = np.hstack( [state_list, np.ones([input_length, 1])])
            predict_seq = stacked_state @ self.W_out
        
        return predict_seq, state_list

    def innate_train(self, input_seq, innate_seq, buffer, train_len, ranseed, \
            learn_every=1, noise_amp=0.0, learning_rate=10.0, \
            scale_input=0.4, train_loops=1, sel=0):
        
        #reset_zero = (learning_rate > 0.0)
        self.__init_w_feed(ranseed=ranseed, reset_zero=False)

        # Initialize P_recurr
        Plist = []
        nqrc = self.nqrc
        for i in range(nqrc):
            Plist.append(np.eye(self.get_local_nodes() * (nqrc-1)))
        self.P_recurr = Plist
        N_local = self.get_local_nodes()

        loss_dict = dict()
        for i in range(self.nqrc):
            loss_dict[i] = []
        
        for k in range(train_loops):
            print('{} Train loop'.format(k))
            self.__reset_states()
            self.__init_rhos(ranseed=ranseed + k, rand_rho=True)

            # Stage 1: Transient regime
            _, state_list = self.__feed_forward(input_seq[:, :buffer], predict=False, \
                noise_amp=noise_amp, scale_input=scale_input, sel=sel)
            dW_recurr_ls = [0] * buffer

            # Stage 2: Innate training for W_feed
            tmp_list = []
            for i in range(buffer, buffer + train_len):
                if i % learn_every == 0:
                    innate_train = True
                else:
                    innate_train = False
                dW_recurr_mag = self.__step_forward(input_seq[:, i], innate_train=innate_train, innate_target=innate_seq[i-1],\
                    learning_rate=learning_rate, noise_amp=noise_amp, scale_input=scale_input, sel=sel)
                dW_recurr_ls.append(dW_recurr_mag)
                state = np.array(self.cur_states.copy(), dtype=np.float64)
                tmp_list.append(state.flatten())
            tmp_list = np.array(tmp_list)
            state_list = np.concatenate([state_list, tmp_list])

            # RMSE values
            for i in range(self.nqrc):
                target_state = innate_seq[buffer:(buffer + train_len), i]
                diff_state = state_list[buffer:(buffer + train_len), sel + N_local * i] - innate_seq[buffer:(buffer + train_len), i]
                #nmse = np.mean(diff_state**2) / np.mean(target_state**2)
                loss = np.sqrt(np.mean(diff_state**2))
                loss_dict[i].append(loss)
                print('Train={}, QR={}, Loss={}'.format(k, i, loss)) 
            
            # Evaluation stage
            _, eval_state_list = self.__feed_forward(input_seq[:, (buffer+train_len):], predict=False, \
                noise_amp=noise_amp, scale_input=scale_input, sel=sel)
            state_list = np.concatenate([state_list, eval_state_list])
            dW_recurr_ls.extend([0] * (input_seq.shape[1] - len(dW_recurr_ls)))

        return state_list, dW_recurr_ls, loss_dict

    def train(self, input_seq, output_seq, buffer, beta, ranseed, noise_amp, scale_input):
        assert(input_seq.shape[1] == output_seq.shape[0])

        self.__reset_states()
        self.__init_rhos(ranseed=ranseed, rand_rho=True)

        # Stage 3: Readout training
        Nout = output_seq.shape[1]
        self.W_out = np.random.rand(self.get_comput_nodes() + 1, Nout)

        _, state_list = self.__feed_forward(input_seq, predict=False, noise_amp=noise_amp, scale_input=scale_input)
        state_list = np.array(state_list)
        state_list = state_list[buffer:, :]

        # discard the transitient state for training
        X = np.reshape(state_list, [-1, self.get_comput_nodes()])
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
                pinv_ = scipypinv2(XTX + beta * I)
                self.W_out = pinv_ @ XTY
            elif self.solver in ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']:
                ridge = Ridge(alpha=beta, fit_intercept=False, normalize=False, copy_X=True, solver=self.solver)
                ridge.fit(XTX, XTY)
                self.W_out = np.array(ridge.coef_).reshape((-1, Nout))
            else:
                raise ValueError('Undefined solver')
    
    def predict(self, input_seq, output_seq, buffer, noise_amp, scale_input):
        prediction_seq, _ = self.__feed_forward(input_seq, predict=True, noise_amp=noise_amp, scale_input=scale_input)
        pred = prediction_seq[buffer:, :]
        out  = output_seq[buffer:, :]
        loss = np.sum((pred - out)**2)/np.sum(pred**2)
        return prediction_seq, loss

    def init_forward(self, qparams, input_seq, ranseed, noise_amp, scale_input):
        self.__reset_states()
        self.__init_reservoir(qparams, ranseed)
        _, state_list =  self.__feed_forward(input_seq, predict=False, noise_amp=noise_amp, scale_input=scale_input)
        return state_list
    
    def init_model(self, qparams, ranseed):
        self.__reset_states()
        self.__init_reservoir(qparams, ranseed)
