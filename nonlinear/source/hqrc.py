#!/usr/bin/env python
"""
    Higher-order reservoir class
"""

import sys
import numpy as np
import scipy as sp
from scipy import sparse
from scipy.sparse import linalg as splinalg
from scipy.linalg import pinv2 as scipypinv2
from IPC import *
from utils import *

import time
from datetime import timedelta
from scipy.special import expit
from sklearn.utils import shuffle

import pickle
# MEMORY TRACKING
import psutil

class HQRC(object):
    def __init__(self, nqrc, gamma, sparsity, sigma_input, dim_input=1,\
        type_input=0, use_corr=0, type_op='Z', type_connect=0, nonlinear=0, mask_input=0, combine_input=1, feed_trials=-1, feed_nothing=True):
        self.nqrc = nqrc
        self.gamma = gamma
        self.sparsity = sparsity
        self.sigma_input = sigma_input
        self.type_input = type_input
        self.dim_input  = dim_input
        self.use_corr = use_corr
        self.type_op = type_op
        self.type_connect = type_connect # type of connection: 0 (full higher-order), 
        # 1 (higher-order but only input at several qrs), 
        # 2 (deep, only input in first qr, feedback for only previous qr)
        self.nonlinear = nonlinear
        self.mask_input = mask_input # feedback between inputs
        self.combine_input = combine_input # combine input and feedback
        self.feed_trials = feed_trials
        self.feed_nothing = feed_nothing
        self.feed_mean = None
        self.feed_std  = None
        self.feed_max = None
        self.feed_min = None
        self.feed_max2 = None
        self.feed_min2 = None

    def __init_reservoir(self, qparams, ranseed, loading_path):
        I = [[1,0],[0,1]]
        X = [[0,1],[1,0]]
        Y = [[0,-1.j],[1.j,0]]
        Z = [[1,0],[0,-1]]

        P0 = [[1,0],[0,0]]
        P1 = [[0,0],[0,1]]
        self.init_ranseed = ranseed
        self.n_units = qparams.n_units
        self.n_envs = qparams.n_envs
        self.virtual_nodes = qparams.virtual_nodes
        self.tau = qparams.tau
        self.max_energy = qparams.max_energy
        self.alpha = qparams.alpha
        self.solver = qparams.solver
        self.dynamic = qparams.dynamic
        self.non_diag_var = qparams.non_diag_var
        self.non_diag_const = qparams.non_diag_const
        
        # Overwrite parameter data
        self.W_feed = np.array(None)
        self.Uops = []

        if loading_path != None and os.path.isdir(loading_path):
            data_path = loading_path + "/data.pickle"
            with open(data_path, "rb") as file:
                data = pickle.load(file)
                self.init_ranseed = data["init_ranseed"]
                self.n_units = data["n_units"]
                self.n_envs = data["n_envs"]
                self.virtual_nodes = data["virtual_nodes"]
                self.tau = data["tau"]
                self.max_energy = data["max_energy"]
                self.non_diag_var = data["non_diag_var"]
                self.non_diag_const = data["non_diag_const"]
                self.solver = data["solver"]
                self.dynamic = data["dynamic"]
                self.Uops = data["Uops"]

                # Create W_feed
                wout = data["W_out"]
                self.W_feed = np.tile(wout, (1, self.nqrc)) 
                print('Loaded Wfeed shape={}, min={}, max={}'.format( self.W_feed.shape, np.min(self.W_feed), np.max(self.W_feed) ))
                del data
        
        if self.init_ranseed >= 0:
            np.random.seed(seed=self.init_ranseed)
        
        self.n_qubits = self.n_units + self.n_envs
        self.dim = 2**self.n_qubits
        
        nqrc = self.nqrc
        Nspins = self.n_qubits
        
        # generate feedback matrix
        n_nodes = self.__get_comput_nodes()
        # skip self loop
        n_local_nodes = self.__get_qr_nodes()
        
        # Create W_feed or load from saved Wout
        if self.W_feed.shape == ():
            W_feed = np.zeros((n_nodes, nqrc))
            # if self.nonlinear == 6:
            #     for k in range(0, n_nodes, n_local_nodes):
            #         #qridx = int(k/n_local_nodes)
            #         W_feed[k, :] = 1.0 / nqrc
            
            if False:
                if nqrc > 1:
                    for i in range(0, nqrc):
                        if self.type_connect == 0:
                            smat = np.random.rand(n_nodes)
                            smat = np.ravel(smat)
                            bg = i * n_local_nodes
                            ed = bg + n_local_nodes 
                            smat[bg:ed] = 0
                            #if self.nonlinear == 0 or self.nonlinear == 3:
                            #    smat *= (self.sigma_input / (n_nodes - n_local_nodes))
                            
                            # else:
                            #     #print(np.std(smat), self.sigma_input)
                            #     valstd = np.std(smat)
                            #     if valstd > 0:
                            #         smat /= valstd
                            #     smat *= (self.sigma_input)
                            
                            W_feed[:len(smat), i] = smat.copy()
                        else:
                            if i > 1:
                                #smat = scipy.sparse.random(n_local_nodes, 1, density = self.sparsity).todense()
                                if self.nonlinear == 0 or self.nonlinear == 3:
                                    smat = np.random.rand(n_nodes)
                                else:
                                    #smat = np.random.randn(n_nodes) * self.sigma_input
                                    smat = np.random.normal(loc=0, scale=self.sigma_input, size=(n_nodes, 1))
                            
                                smat = smat.ravel()
                                #if self.nonlinear == 0 or self.nonlinear == 3:
                                #    smat *= (self.sigma_input / n_local_nodes)
                                
                                # else: 
                                #     smat /= np.std(smat)
                                #     smat *= (self.sigma_input)
                                bg = (i-1) * n_local_nodes
                                ed = bg + n_local_nodes
                                W_feed[bg:ed, i] = smat.copy()
            elif self.type_connect == 0:
                W_feed = np.random.normal(loc=0, scale=self.sigma_input, size=(n_nodes, nqrc))
            elif self.type_connect == 1:
                feed_mat = np.random.normal(loc=0, scale=self.sigma_input, size=(n_nodes, nqrc))
                for i in range(nqrc-1):
                    bg = (i+1) * n_local_nodes
                    ed = (i+2) * n_local_nodes
                    W_feed[bg:ed, i] = feed_mat[bg:ed, i]
            elif self.type_connect == 2 and nqrc > self.dim_input:
                feed_dim = nqrc - self.dim_input
                if self.nonlinear == 0 and self.feed_nothing == False:
                    feed_mat = scipy.sparse.random(n_nodes, feed_dim, density = self.sparsity).todense()
                    # normalize
                    for i in range(feed_dim):
                        colsum = np.sum(feed_mat[:,i].ravel())
                        if colsum > 0:
                            feed_mat[:, i] = feed_mat[:, i] / colsum
                else:
                    feed_mat = np.random.normal(loc=0, scale=self.sigma_input, size=(n_nodes, feed_dim))
                W_feed[:, self.dim_input:] = feed_mat

                #W_feed = scipy.sparse.random(n_nodes, nqrc, density = self.sparsity).todense() * self.sigma_input
                # if self.radius > 0:
                #     eigenvalues, eigvectors = splinalg.eigs(W_feed)
                #     eigenvalues = np.abs(eigenvalues)
                #     W_feed = (W_feed/np.max(eigenvalues))*self.radius
                #W_feed[self.__get_comput_nodes():, :] = 0
            
            self.W_feed = W_feed

        
        #  create operators from tensor product
        
        self.Xop = [1]*self.n_qubits
        self.Xop_corr = dict()

        self.Yop = [1]*self.n_qubits
        self.Yop_corr = dict()

        self.Zop = [1]*self.n_qubits
        self.Zop_corr = dict()
        
        self.P0op = [1]
        self.P1op = [1]

        for q1 in range(self.n_qubits):
            for q2 in range(q1+1, self.n_qubits):
                self.Xop_corr[(q1, q2)] = [1]
                self.Yop_corr[(q1, q2)] = [1]
                self.Zop_corr[(q1, q2)] = [1]

        for cindex in range(self.n_qubits):
            for qindex in range(self.n_qubits):
                if cindex == qindex:
                    self.Xop[qindex] = np.kron(self.Xop[qindex],X)
                    self.Yop[qindex] = np.kron(self.Yop[qindex],Y)
                    self.Zop[qindex] = np.kron(self.Zop[qindex],Z)
                else:
                    self.Xop[qindex] = np.kron(self.Xop[qindex],I)
                    self.Yop[qindex] = np.kron(self.Yop[qindex],I)
                    self.Zop[qindex] = np.kron(self.Zop[qindex],I)

            if cindex == 0:
                self.P0op = np.kron(self.P0op, P0)
                self.P1op = np.kron(self.P1op, P1)
            else:
                self.P0op = np.kron(self.P0op, I)
                self.P1op = np.kron(self.P1op, I)

        # generate correlatior operators
        if self.use_corr > 0:
            for q1 in range(self.n_qubits):
                for q2 in range(q1+1, self.n_qubits):
                    cindex = (q1, q2)
                    for qindex in range(self.n_qubits):
                        if qindex == q1 or qindex == q2:
                            self.Xop_corr[cindex] = np.kron(self.Xop_corr[cindex], X)
                            self.Yop_corr[cindex] = np.kron(self.Yop_corr[cindex], Y)
                            self.Zop_corr[cindex] = np.kron(self.Zop_corr[cindex], Z)
                        else:
                            self.Xop_corr[cindex] = np.kron(self.Xop_corr[cindex], I)
                            self.Yop_corr[cindex] = np.kron(self.Yop_corr[cindex], I)
                            self.Zop_corr[cindex] = np.kron(self.Zop_corr[cindex], I)
                            
        if self.type_op == 'X':
            self.Pauli_op = self.Xop
            self.Pauli_op_corr = self.Xop_corr
        elif self.type_op == 'Y':
            self.Pauli_op = self.Yop
            self.Pauli_op_corr = self.Yop_corr
        else:
            self.Pauli_op = self.Zop
            self.Pauli_op_corr = self.Zop_corr
        
        # initialize current states
        self.cur_states = [None] * nqrc
        # initialize feedback input
        self.feed_inputs = [0]  * nqrc

        if len(self.Uops) == 0:
            # create coupling strength for ion trap
            a = self.alpha
            bc = self.non_diag_const
            Nalpha = 0
            for qindex1 in range(Nspins):
                for qindex2 in range(qindex1+1, Nspins):
                    Jij = np.abs(qindex2-qindex1)**(-a)
                    Nalpha += Jij / (Nspins-1)
            if bc > 0:
                B = self.max_energy / bc # Magnetic field
            else:
                B = self.max_energy

            # Intialize evolution operators
            tmp_uops = []
            # generate hamiltonian
            for i in range(nqrc):
                hamiltonian = np.zeros( (self.dim,self.dim) )

                for qindex in range(Nspins):
                    if self.dynamic == DYNAMIC_FULL_RANDOM:
                        coef = (np.random.rand()-0.5) * 2 * self.max_energy
                    elif self.dynamic == DYNAMIC_PHASE_TRANS:
                        coef = (np.random.rand()-0.5) * self.non_diag_var + self.non_diag_const
                    else:
                        coef = B
                    hamiltonian -= coef * self.Zop[qindex]

                for qindex1 in range(Nspins):
                    for qindex2 in range(qindex1+1, Nspins):
                        if self.dynamic == DYNAMIC_FULL_CONST_COEFF:
                            coef =  self.max_energy
                        elif self.dynamic == DYNAMIC_ION_TRAP:
                            coef = np.abs(qindex2 - qindex1)**(-a) / Nalpha
                            coef = self.max_energy * coef
                        elif self.dynamic == DYNAMIC_PHASE_TRANS:
                            coef = (np.random.rand()-0.5) * self.max_energy
                        else:
                            coef = (np.random.rand()-0.5) * 2 * self.max_energy
                        hamiltonian -= coef * self.Xop[qindex1] @ self.Xop[qindex2]
                        
                ratio = float(self.tau) / float(self.virtual_nodes)        
                Uop = sp.linalg.expm(-1.j * hamiltonian * ratio)
                tmp_uops.append(Uop)
            self.Uops = tmp_uops.copy()
        
        # initialize density matrix
        tmp_rhos = []
        for i in range(nqrc):
            rho = np.zeros( [self.dim, self.dim] )
            rho[0, 0] = 1
            if qparams.init_rho != 0:
                # initialize random density matrix
                rho = random_density_matrix(self.dim)
            tmp_rhos.append(rho)

        self.init_rhos = tmp_rhos.copy()
        self.last_rhos = tmp_rhos.copy()
        
    def init_reservoir(self, qparams, ranseed, loading_path=None):
        self.__init_reservoir(qparams, ranseed, loading_path)

    def __get_qr_nodes(self):
        if self.use_corr > 0:
            qrnodes = int((self.n_qubits * (self.n_qubits + 1)) / 2)
        else:
            qrnodes = self.n_qubits
        qrnodes = qrnodes * self.virtual_nodes
        return qrnodes

    def __get_comput_nodes(self):
        return self.__get_qr_nodes() * self.nqrc
    
    def reset_states(self):
        self.cur_states = [None] * self.nqrc

    def gen_rand_rhos(self, ranseed):
        if ranseed >= 0:
            np.random.seed(seed=ranseed)
        tmp_rhos = []
        for i in range(self.nqrc):
            rho = random_density_matrix(self.dim)
            tmp_rhos.append(rho)
        self.init_rhos = tmp_rhos.copy()

    def step_forward(self, local_rhos, input_val, feedback_flag=1, scale_input=True):
        nqrc = self.nqrc
        original_input = input_val.copy().ravel()
        
        q0 = np.array([1, 0]).reshape((2, 1))
        q1 = np.array([0, 1]).reshape((2, 1))

        if self.cur_states[0] is None:
            if scale_input == True:
                update_input = (1.0-self.gamma) * original_input
            else:
                # Use in masking input (keep the original input)
                update_input = original_input
            self.feed_inputs = original_input * 0.0
        else:
            tmp_states = np.array(self.cur_states.copy(), dtype=np.float64).reshape(1, -1)
            tmp_states = (tmp_states + 1.0) / 2.0
            #tmp_states = np.hstack( [tmp_states, np.ones([1, 1])])
            #print(tmp_states.shape, self.W_feed.shape)
            #tmp_states = expit(tmp_states)
            tmp_states = tmp_states @ self.W_feed
            tmp_states = np.ravel(tmp_states)
            
            # if self.feed_min2 is not None:
            #     tmp_states = tmp_states - self.feed_min2
            #tmp_states = min_max_norm(tmp_states, self.feed_min2, self.feed_max2)
                
            #if self.feed_min is not None:
                #tmp_states = tmp_states - self.feed_mean
                #tmp_states = np.divide(tmp_states, self.feed_std)
                #tmp_states = np.multiply(tmp_states, self.feed_scale) + self.feed_trans
                
                #tmp_states = np.divide(tmp_states, np.divide(self.feed_max - self.feed_min, self.feed_std))
                
                #tmp_states = np.multiply(tmp_states, self.feed_scale) + self.feed_trans 
                
                #if self.feed_min2 is not None:
                #    tmp_states = tmp_states - self.feed_min2
                #    tmp_states = np.divide(tmp_states, self.feed_max2 - self.feed_min) 
                
                # if self.feed_trials < 0:
                #     tmp_states = (tmp_states - 0.2)*2.0

                #print(tmp_states, self.feed_max, self.feed_min, self.feed_mean, self.feed_std)
                #tmp_states[tmp_states < 0.0] = 0.0
                #tmp_states[tmp_states > 1.0] = 1.0
                #print(self.feed_mean, self.feed_std, tmp_states)
                #tmp_states = np.abs(tmp_states)
                
            #print('W_feed', np.min(self.W_feed), np.max(self.W_feed))
            #print('tmpstates', tmp_states, np.min(self.cur_states), np.max(self.cur_states))
            #tmp_states = np.exp(-1.0*tmp_states)
            if self.nonlinear == 1:
                tmp_states = expit(tmp_states)
            elif self.nonlinear == 2:
                # Min-max norm
                tmp_states = (tmp_states - np.min(tmp_states)) / (np.max(tmp_states) - np.min(tmp_states))
            elif self.nonlinear == 3:
                tmp_states = shuffle(tmp_states)
            elif self.nonlinear == 4:
                tmp_states = expit(tmp_states)
                tmp_states = shuffle(tmp_states)
            elif self.nonlinear == 5:
                # Min-max norm
                tmp_states = (tmp_states - np.min(tmp_states)) / (np.max(tmp_states) - np.min(tmp_states))
                tmp_states = shuffle(tmp_states)
            elif self.nonlinear == 6:
                tmp_states = [np.modf(x / (2*np.pi))[0] for x in tmp_states]
                # to make sure the nonegative number
                tmp_states = np.array([np.modf(x + 1.0)[0] for x in tmp_states])
            elif self.nonlinear == 7:
                tmp_states = [np.modf(x)[0] for x in tmp_states]
                tmp_states = np.array([np.modf(x + 1.0)[0] for x in tmp_states])
            else:
                tmp_states = min_max_norm(tmp_states, self.feed_min, self.feed_max)
            #print(tmp_states, self.feed_min, self.feed_max)
            self.feed_inputs = tmp_states.copy().ravel()
            
            tmp_states[tmp_states < 0.0] = 0.0
            tmp_states[tmp_states > 1.0] = 1.0
        #print('Original input',  original_input) 
        
            if self.feed_nothing == True:
                update_input = original_input
            elif feedback_flag > 0:
                if original_input[0] < -1.0:
                    # feedback between inputs
                    update_input = self.gamma * tmp_states
                    #print('Update input', update_input)
                else:
                    # combine input
                    if feedback_flag == 1:
                        update_input = self.gamma * tmp_states + (1.0 - self.gamma) * original_input
                    elif feedback_flag == 2:
                        if self.gamma > 0:
                            update_input = self.gamma * tmp_states
                        else:
                            update_input = original_input
                    #update_input = np.multiply(update_input, tmp_states)
                    #print('Combine input', self.gamma, tmp_states, original_input)
                            
                # if update_input[0] < -1.0 or update_input[0] > 1.0:
                #     # If the update_input goes out of range, just use the normal input
                #     update_input = original_input
            else:
                if scale_input == True:
                    update_input = (1.0-self.gamma) * original_input
                else:
                    # Use in masking input (keep the original input)
                    update_input = original_input
                
                #print('Update input', update_input)

                   
        if True:
            for i in range(nqrc):
                Uop = self.Uops[i]
                rho = local_rhos[i]
                # Obtain value from the input
                if i < self.dim_input and feedback_flag == 2:
                    value = original_input[i]
                else:
                    value = update_input[i]

                # Replace the density matrix
                # rho = self.P0op @ rho @ self.P0op + self.Xop[0] @ self.P1op @ rho @ self.P1op @ self.Xop[0]
                # (1 + u Z)/2 = (1+u)/2 |0><0| + (1-u)/2 |1><1|
                # inv1 = (self.affine[1] + self.value) / self.affine[0]
                # inv2 = (self.affine[1] - self.value) / self.affine[0]

                if self.type_input == 0:
                    rho = self.P0op @ rho @ self.P0op + self.Xop[0] @ self.P1op @ rho @ self.P1op @ self.Xop[0]
                    # for input in [0, 1]
                    value = clipping(value, minval=0.0, maxval=1.0)
                    rho = (1 - value) * rho + value * self.Xop[0] @ rho @ self.Xop[0]
                elif self.type_input == 1:
                    value = clipping(value, minval=-1.0, maxval=1.0)
                    rho = self.P0op @ rho @ self.P0op + self.Xop[0] @ self.P1op @ rho @ self.P1op @ self.Xop[0]
                    # for input in [-1, 1]
                    rho = ((1+value)/2) * rho + ((1-value)/2) *self.Xop[0] @ rho @ self.Xop[0]
                else:
                    value = clipping(value, minval=0.0, maxval=1.0)
                    par_rho = partial_trace(rho, keep=[1], dims=[2**self.n_envs, 2**self.n_units], optimize=False)
                    if self.type_input == 2:
                        input_state = np.sqrt(1-value) * q0 + np.sqrt(value) * q1
                    elif self.type_input == 3:
                        angle_val = 2*np.pi*value
                        input_state = np.cos(angle_val) * q0 + np.sin(angle_val) * q1
                    elif self.type_input == 4:
                        input_state = np.sqrt(1-value) * q0 + np.sqrt(value) * np.exp(1.j * 2*np.pi*value) * q1
                    else:
                        update_contrib = self.gamma * self.feed_inputs[i]
                        orig_contrib = original_input[i]
                        input_state = np.sqrt(1-orig_contrib) * q0 + np.sqrt(orig_contrib) * np.exp(1.j * 2*np.pi*update_contrib) * q1

                    input_state = input_state @ input_state.T.conj() 
                    rho = np.kron(input_state, par_rho)


                current_state = []
                for v in range(self.virtual_nodes):
                    # Time evolution of density matrix
                    rho = Uop @ rho @ Uop.T.conj()
                    for qindex in range(0, self.n_qubits):
                        rvstate = np.real(np.trace(self.Pauli_op[qindex] @ rho))
                        # if self.type_input != 1:
                        #     rvstate = (1.0 + expectation_value) / 2.0
                        #     #rvstate = expectation_value
                        # else:
                        #     rvstate = expectation_value
                        current_state.append(rvstate)
                    
                    if self.use_corr > 0:
                        for q1 in range(0, self.n_qubits):
                            for q2 in range(q1+1, self.n_qubits):
                                cindex = (q1, q2)
                                rvstate = np.real(np.trace(self.Pauli_op_corr[cindex] @ rho))
                                # if self.type_input != 1:
                                #     rvstate = (1.0 + expectation_value) / 2.0
                                # else:
                                #     rvstate = expectation_value
                                current_state.append(rvstate)

                # Size of current_state is Nqubits x Nvirtuals)
                self.cur_states[i] = np.array(current_state, dtype=np.float64)
                # if self.cur_states[i] is not None:
                #     self.cur_states[i] = 0.9*upstates + 0.1*self.cur_states[i]
                # else:
                #     self.cur_states[i] = upstates
                local_rhos[i] = rho
        return local_rhos

    def feed_forward(self, input_seq, predict, use_lastrho):
        input_dim, input_length = input_seq.shape
        nqrc = self.nqrc
        assert(input_dim == nqrc)
        
        predict_seq = None
        local_rhos = self.init_rhos.copy()
        if use_lastrho == True :
            local_rhos = self.last_rhos.copy()
        
        state_list, feed_list = [], []
        for time_step in range(0, input_length):
            input_val = np.ravel(input_seq[:, time_step])
            scale_input = True
            if self.mask_input > 0 and self.gamma > 0 and self.feed_nothing == False:
                scale_input = False
                # put the feedback between the inputs
                dummy_input = np.zeros(input_val.shape) - 100.0
                for i in range(self.mask_input):
                    local_rhos = self.step_forward(local_rhos, dummy_input, feedback_flag=1)
            
            local_rhos = self.step_forward(local_rhos, input_val, feedback_flag=self.combine_input, scale_input=scale_input)
            state = np.array(self.cur_states.copy(), dtype=np.float64)
            state_list.append(state.flatten())
            feed_list.append(self.feed_inputs)
            if self.feed_nothing == True and time_step == self.feed_trials:
                tmp_list = feed_list[(time_step//2):time_step]
                self.feed_mean = np.mean(tmp_list, axis=0)
                self.feed_std = np.sqrt(np.var(tmp_list, axis=0) + 1e-8)
                self.feed_max = np.max(tmp_list, axis=0)
                self.feed_min = np.min(tmp_list, axis=0)
                #print(self.feed_max, self.feed_min)
                self.feed_nothing = False
                
            # elif time_step == self.feed_trials + 101:
            #     tmp_list = feed_list[(time_step-100):time_step]
            #     self.feed_max2 = np.max(tmp_list, axis=0) * 1.1
            #     self.feed_min2 = np.min(tmp_list, axis=0) * 0.9

        state_list = np.array(state_list)
        feed_list  = np.array(feed_list)
        self.last_rhos = local_rhos.copy()

        if predict:
            stacked_state = np.hstack( [state_list, np.ones([input_length, 1])])
            predict_seq = stacked_state @ self.W_out
        
        return predict_seq, state_list, feed_list


    def __train(self, input_seq, output_seq, buffer, beta, \
        pertubed_gammas=[], pertubed_inputs=[], pertubed_outputs=[]):
        assert(input_seq.shape[1] == output_seq.shape[0])
        self.start_time = time.time()
        Nout = output_seq.shape[1]
        self.W_out = np.random.rand(self.__get_comput_nodes() + 1, Nout)

        true_gamma = self.gamma
        state_list_ls, output_list_ls = [], []
        for i in range(len(pertubed_inputs)):
            pgamma = pertubed_gammas[i]
            pinput = pertubed_inputs[i]
            poutput = pertubed_outputs[i].copy()
            poutput = poutput[buffer:, :]
            output_list_ls.append(poutput)

            self.gamma = pgamma
            #self.reset_states()
            _, p_state_list, _ = self.feed_forward(pinput, predict=False, use_lastrho=True)
            p_state_list = p_state_list[buffer:, :]
            state_list_ls.append(p_state_list)

        self.gamma = true_gamma
        if len(pertubed_inputs) == 0:
            _, p_state_list, _ = self.feed_forward(input_seq, predict=False, use_lastrho=False)
            
            # discard the transitient state for training
            p_state_list = p_state_list[buffer:, :]
            state_list_ls.append(p_state_list)
            discard_output = output_seq[buffer:, :]
            output_list_ls.append(discard_output)

        # Training
        state_list = np.concatenate(state_list_ls, axis=0)
        X = np.hstack( [state_list, np.ones([state_list.shape[0], 1]) ] )
        #print('shape', X.shape, state_list.shape)
    
        Y = np.concatenate(output_list_ls, axis=0)
        #Y = np.reshape(discard_output, [discard_output.shape[0], -1])

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

    def train_to_predict(self, input_seq, output_seq, buffer, qparams, ranseed, \
        saving_path=None, loading_path=None, pertubed_gammas=[], pertubed_inputs=[], pertubed_outputs=[]):
        self.__init_reservoir(qparams, ranseed, loading_path)
        self.__train(input_seq, output_seq, buffer, qparams.beta, \
            pertubed_gammas=pertubed_gammas, pertubed_inputs=pertubed_inputs, pertubed_outputs=pertubed_outputs)
        if saving_path != None:
            self.save_model(saving_path=saving_path)

    def save_model(self, saving_path):
        print("Recording time...")
        self.total_training_time = time.time() - self.start_time
        print("Total training time is {:}".format(self.total_training_time))

        print("MEMORY TRACKING IN MB...")
        process = psutil.Process(os.getpid())
        memory = process.memory_info().rss/1024/1024
        self.memory = memory
        print("Script used {:} MB".format(self.memory))
        print("SAVING MODEL...")

        data = {
            "memory": self.memory,
            "n_units": self.n_units,
            "n_envs": self.n_envs,
            "virtual_nodes": self.virtual_nodes,
            "tau": self.tau,
            "max_energy": self.max_energy,
            "non_diag_const": self.non_diag_const,
            "non_diag_var": self.non_diag_var,
            "alpha": self.alpha,
            "solver": self.solver,
            "dynamic": self.dynamic,
            "n_qubits": self.n_qubits,
            "nqrc": self.nqrc,
            "n_nodes": self.__get_comput_nodes(),
            "n_local_nodes": self.__get_qr_nodes(),
            "type_connect": self.type_connect,
            "nonlinear": self.nonlinear,
            "sigma_input": self.sigma_input,
            "use_corr": self.use_corr,
            "init_ranseed": self.init_ranseed,
            "total_training_time": self.total_training_time,
            "W_out": self.W_out,
            "W_feed": self.W_feed,
            "Uops": self.Uops,
        }
        os.makedirs(saving_path, exist_ok=True)
        data_path = saving_path + "/data.pickle"
        with open(data_path, "wb") as file:
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
            del data
        return 0

    def predict(self, input_seq, output_seq, buffer, use_lastrho):
        prediction_seq, _, _ = self.feed_forward(input_seq, predict=True, use_lastrho=use_lastrho)
        pred = prediction_seq[buffer:, :]
        out  = output_seq[buffer:, :]
        
        nrmse_loss = np.sqrt(np.mean((pred - out)**2)/(np.std(out)**2))
        #nmse_loss = np.sum((pred - out)**2) / np.sum(out**2)
        
        return prediction_seq, nrmse_loss
    
    def init_forward(self, qparams, input_seq, init_rs, ranseed, loading_path=None):
        self.reset_states()
        if init_rs == True:
            self.__init_reservoir(qparams, ranseed, loading_path)
        _, state_list, feed_list =  self.feed_forward(input_seq, predict=False, use_lastrho=False)
        return state_list, feed_list

def get_loss(qparams, buffer, train_input_seq, train_output_seq, val_input_seq, val_output_seq, \
        ranseed, nqrc, gamma=0.0, sparsity=1.0, sigma_input=1.0, type_input=0, type_op='Z', mask_input=0, combine_input=1,feed_nothing=False,\
        type_connect=0, use_corr=0, nonlinear=0, saving_path=None, loading_path=None, dim_input=0):

    model = HQRC(nqrc=nqrc, gamma=gamma, sparsity=sparsity, dim_input=dim_input,\
        sigma_input=sigma_input, type_input=type_input, type_op=type_op, mask_input=mask_input, combine_input=combine_input,feed_nothing=feed_nothing,\
        type_connect=type_connect, use_corr=use_corr, nonlinear=nonlinear, feed_trials=buffer//2)

    train_input_seq = np.array(train_input_seq)
    train_output_seq = np.array(train_output_seq)
    
    model.train_to_predict(train_input_seq, train_output_seq, buffer, qparams, ranseed, \
        saving_path=saving_path, loading_path=loading_path)
    model.reset_states()
    train_pred_seq, train_loss = model.predict(train_input_seq, train_output_seq, buffer=buffer, use_lastrho=False)
    #print("train_loss={}, shape".format(train_loss), train_pred_seq_ls.shape)
    
    # Test phase
    val_input_seq = np.array(val_input_seq)
    val_output_seq = np.array(val_output_seq)
    val_pred_seq, val_loss = model.predict(val_input_seq, val_output_seq, buffer=0, use_lastrho=True)
    #print("val_loss={}, shape".format(val_loss), val_pred_seq_ls.shape)

    return train_pred_seq, train_loss, val_pred_seq, val_loss

def closed_loop(qparams, buffer, train_input_seq, train_output_seq, valsteps, ranseed, nqrc, \
    gamma=0.0, sparsity=1.0, sigma_input=1.0, type_input=0, mask_input=0, combine_input=1, \
    feed_nothing=False, type_connect=0, use_corr=0, nonlinear=0, \
    pertubed_gammas=[], pertubed_inputs=[], pertubed_outputs=[], test_gammas=[]):

    model = HQRC(nqrc=nqrc, gamma=gamma, sparsity=sparsity, dim_input=0,\
        sigma_input=sigma_input, type_input=type_input, mask_input=mask_input, \
        combine_input=combine_input, feed_nothing=feed_nothing,\
        type_connect=type_connect, use_corr=use_corr, nonlinear=nonlinear, feed_trials=buffer//2)
    
    train_input_seq = np.array(train_input_seq)
    train_output_seq = np.array(train_output_seq)
    
    model.train_to_predict(train_input_seq, train_output_seq, buffer, qparams, ranseed, \
        pertubed_gammas=pertubed_gammas, pertubed_inputs=pertubed_inputs, pertubed_outputs=pertubed_outputs)
    
    # train_pred_seq = [0.0] * len(train_input_seq)
    # if len(pertubed_inputs) == 0:
    #     model.reset_states()
    #     train_pred_seq, _ = model.predict(train_input_seq, train_output_seq, \
    #         buffer=buffer, use_lastrho=False)
    #     current_input = train_pred_seq[-1].copy().ravel()
    # else:
    #     state = np.array(model.cur_states, dtype=np.float64)
    #     stacked_state = np.hstack( [state.reshape((1, -1)), np.ones([1, 1])])
    #     pred_vec = stacked_state @ model.W_out
    #     current_input = pred_vec.ravel()

    model.reset_states()
    train_pred_seq, _ = model.predict(train_input_seq, train_output_seq, \
        buffer=buffer, use_lastrho=False)
    current_input = train_pred_seq[-1].copy().ravel()

    val_pred_seq = []
    ndup = int(nqrc/len(current_input))
    for n in range(valsteps):
        #if n < 10:
        #    print(n, current_input, model.cur_states)
        if len(test_gammas) >= valsteps:
            model.gamma = test_gammas[n]
        current_input = np.tile(current_input, (ndup, 1)).ravel()
        local_rhos = model.last_rhos.copy()
        local_rhos = model.step_forward(local_rhos, current_input, feedback_flag=combine_input)
        model.last_rhos = local_rhos.copy()
        state = np.array(model.cur_states, dtype=np.float64)
        stacked_state = np.hstack( [state.reshape((1, -1)), np.ones([1, 1])])
        pred_vec = stacked_state @ model.W_out
        pred_vec = pred_vec.ravel()
        current_input = pred_vec.copy()
        val_pred_seq.append(pred_vec.copy())
        
    val_pred_seq = np.array(val_pred_seq)
    return train_pred_seq, val_pred_seq

def view_dynamic(qparams, input_seq, ranseed, nqrc, \
    gamma=0.0, sparsity=1.0, sigma_input=1.0, type_input=0, mask_input=0, combine_input=1,\
    type_connect=0, use_corr=0, nonlinear=0, loading_path=None):

    input_seq = np.array(input_seq)
    
    model = HQRC(nqrc=nqrc, gamma=gamma, sparsity=sparsity, \
        sigma_input=sigma_input, type_input=type_input, mask_input=mask_input, combine_input=combine_input,\
        type_connect=type_connect, use_corr=use_corr, nonlinear=nonlinear, feed_trials=input_seq.shape[1]//2)

    state_list, feed_list = model.init_forward(qparams, input_seq, init_rs=True, \
        ranseed=ranseed, loading_path=loading_path)
    return state_list, feed_list

def get_IPC(qparams, ipcparams, length, logger, nqrc=1, gamma=0.0, ranseed=-1, Ntrials=1, savedir=None, \
    posfix='capa', feed_nothing=False, sparsity=1.0, nonlinear = 0, type_op='Z',\
    sigma_input=1.0, type_input=1, mask_input=0, combine_input=1, label='', dim_input=1, input_file=''):
    start_time = time.monotonic()
    fname = '{}_{}'.format(label, sys._getframe().f_code.co_name)
    transient = length // 2

    if ranseed >= 0:
        np.random.seed(seed=ranseed)
    for n in range(Ntrials):
        if os.path.isfile(input_file):
            print('Data from input file {}'.format(input_file))
            # Read from file: 
            input_signals = np.loadtxt(input_file)[:length, 1]
        else:
            input_signals = np.random.uniform(0, 1, length) 
        
        rescale_input_signals = input_signals * 2.0 - 1.0
        input_signals = np.array(input_signals)
        input_signals = np.tile(input_signals, (nqrc, 1))

        ipc = IPC(ipcparams, log=logger, savedir=savedir, label=label)
        model = HQRC(nqrc=nqrc, gamma=gamma, sparsity=sparsity, feed_nothing=feed_nothing, nonlinear = nonlinear, sigma_input=sigma_input, dim_input=dim_input,\
            type_input=type_input, type_op=type_op, mask_input=mask_input, combine_input=combine_input, feed_trials=transient//2)
        output_signals, _ = model.init_forward(qparams, input_signals, init_rs=True, ranseed = n + ranseed)
        logger.debug('{}: n={} per {} trials, input shape = {}, output shape={}'.format(fname, n+1, Ntrials, input_signals.shape, output_signals.shape))
        
        ipc.run(rescale_input_signals[transient:], output_signals[transient:])
        ipc.write_results(posfix=posfix)
    end_time = time.monotonic()
    logger.info('{}: Executed time {}'.format(fname, timedelta(seconds=end_time - start_time)))

def memory_function(taskname, qparams, train_len, val_len, buffer, dlist, \
        nqrc, gamma, sparsity, sigma_input, mask_input=0, combine_input=1, feed_nothing=False,\
        ranseed=-1, Ntrials=1, type_input=0, type_op='Z', nonlinear=0, dim_input=1):    
    MFlist = []
    MFstds = []
    train_list, val_list = [], []
    length = buffer + train_len + val_len
    
    if ranseed >= 0:
        np.random.seed(seed=ranseed)
    
    if os.path.isfile(taskname):
        print('Data from input file {}'.format(taskname))
        # Read from file: 
        data = np.loadtxt(taskname)[:length,1]
    else:
        # generate data
        if '_stm' not in taskname and '_pc' not in taskname:
            raise ValueError('Not found taskname ={} to generate data'.format(taskname))

        if '_pc' in taskname:
            print('Generate parity check data')
            data = np.random.randint(0, 2, length)
        else:
            print('Generate STM task data')
            if type_input != 1:
                data = np.random.rand(length)
            else:
                data = 2.0*np.random.rand(length) - 1.0

    for d in dlist:
        train_input_seq = np.array(data[  : buffer + train_len])
        train_input_seq = np.tile(train_input_seq, (nqrc, 1))
        
        val_input_seq = np.array(data[buffer + train_len : length])
        val_input_seq = np.tile(val_input_seq, (nqrc, 1))
            
        train_out, val_out = [], []
        if '_pc' in taskname or 'binary' in taskname:
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
            get_loss(qparams, buffer, train_input_seq, train_output_seq, val_input_seq, val_output_seq, \
                nqrc=nqrc, gamma=gamma, sparsity=sparsity, sigma_input=sigma_input, ranseed=ranseed_net, feed_nothing=feed_nothing,\
                type_input=type_input, type_op=type_op, mask_input=mask_input, combine_input=combine_input, nonlinear=nonlinear, dim_input=dim_input)

            # Compute memory function
            val_out_seq, val_pred_seq = val_output_seq.flatten(), val_pred_seq.flatten()
            #print('cov', val_output_seq.shape, val_pred_seq.shape)
            cor_matrix = np.corrcoef(np.array([val_out_seq, val_pred_seq]))
            MF_d = cor_matrix[0][1] ** 2
            #MF_d = MF_d / (np.var(val_out_seq) * np.var(val_pred_seq))
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

def effective_dim(qparams, buffer, length, nqrc, gamma, sparsity, sigma_input, mask_input, nonlinear, ranseed, Ntrials, combine_input=1):
    # Calculate effective dimension for reservoir
    from numpy import linalg as LA
    
    if ranseed >= 0:
        np.random.seed(seed=ranseed)

    data = np.random.rand(length)

    input_seq = np.array(data)
    input_seq = np.tile(input_seq, (nqrc, 1))

    model = HQRC(nqrc, gamma, sparsity, sigma_input, nonlinear=nonlinear, mask_input=mask_input, combine_input=combine_input)

    effdims = []
    for n in range(Ntrials):
        ranseed_net = ranseed
        if ranseed >= 0:
            ranseed_net = (ranseed + 11000) * (n + 1)
        
        state_list, _ = model.init_forward(qparams, input_seq, init_rs=True, ranseed=ranseed_net)
        L, D = state_list.shape
        # L = Length of time series
        # D = Number of virtual nodes x Number of qubits
        locls = []
        for i in range(D):
            ri = state_list[buffer:, i] * 2.0 - 1.0
            mi = np.mean(ri)
            for j in range(D):
                rj = state_list[buffer:, j] * 2.0 - 1.0
                mj = np.mean(rj)
                locls.append(np.mean((ri-mi)*(rj-mj)))
        locls = np.array(locls).reshape(D, D)
        w, v = LA.eig(locls)
        #print(w)
        w = np.abs(w) / np.abs(w).sum()
        effdims.append(1.0 / np.power(w, 2).sum())
    return np.mean(effdims), np.std(effdims)

def esp_index(qparams, buffer, length, nqrc, gamma, sparsity, sigma_input, ranseed, state_trials):
    if ranseed >= 0:
        np.random.seed(seed=ranseed)

    data = np.random.rand(length)
    input_seq = np.array(data)
    input_seq = np.tile(input_seq, (nqrc, 1))

    # Initialize the reservoir to zero state - density matrix
    model = HQRC(nqrc=nqrc, gamma=gamma, sparsity=sparsity, sigma_input=sigma_input)
    x0_state_list, _ = model.init_forward(qparams, input_seq, init_rs = True, ranseed = ranseed)
    # Compute esp index and esp_lambda
    dP = []
    for i in range(state_trials):
        # Initialzie the reservoir to a random initial state
        # Keep same coupling configuration
        model.gen_rand_rhos(ranseed = i + 300000)
        z0_state_list, _ = model.init_forward(qparams, input_seq, init_rs = False, ranseed = i + 200000)
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

def lyapunov_exp(qparams, buffer, length, nqrc, gamma, sparsity, sigma_input, nonlinear, ranseed, initial_distance):
    if ranseed >= 0:
        np.random.seed(seed=ranseed)

    data = np.random.rand(length)
    input_seq = np.array(data)
    input_seq = np.tile(input_seq, (nqrc, 1))

    # Initialize the reservoir to zero state - density matrix
    model = HQRC(nqrc=nqrc, gamma=gamma, sparsity=sparsity, sigma_input=sigma_input, nonlinear=nonlinear)
    states1, _ = model.init_forward(qparams, input_seq, init_rs = True, ranseed = -1)
    L, D = states1.shape
    # L = Length of time series
    # D = Number of layers x Number of virtual nodes x Number of qubits
    lyps = []
    n_local = int(D / nqrc)
    for n in range(n_local):
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
            # Update cur states
            model.cur_states[0] = states2[k-1, :n_local].copy()
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