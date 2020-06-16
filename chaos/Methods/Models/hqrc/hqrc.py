#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Created by:  Anonymous authors to submit NeurIPS 2020

    Implemented in the framework created by Vlachas Pantelis, CSE-lab, ETH Zurich
        https://github.com/pvlachas/RNN-RC-Chaos
        [1] P.R. Vlachas, J. Pathak, B.R. Hunt et al., 
        Backpropagation algorithms and Reservoir Computing in Recurrent Neural Networks 
        for the forecasting of complex spatiotemporal dynamics. Neural Networks (2020), 
        doi: https://doi.org/10.1016/j.neunet.2020.02.016.
"""
#!/usr/bin/env python
import numpy as np
import pickle
import scipy as sp 
from scipy import sparse as sparse
from scipy.sparse import linalg as splinalg
from scipy.linalg import pinv2 as scipypinv2
# from scipy.linalg import lstsq as scipylstsq
# from numpy.linalg import lstsq as numpylstsq
import os
from plotting_utils import *
from global_utils import *
import pickle
import time
from functools import partial
print = partial(print, flush=True)

from sklearn.linear_model import Ridge

# MEMORY TRACKING
import psutil

class QRCParams():
    def __init__(self, n_units, max_energy, beta, virtual_nodes, tau):
        self.n_units = n_units
        self.max_energy = max_energy
        self.beta = beta
        self.virtual_nodes = virtual_nodes
        self.tau = tau
    
    def info(self):
        print('units={},Jdelta={},V={},taudelta={}'.format(\
            self.n_units, self.max_energy,
            self.virtual_nodes, self.tau))

def generate_list_rho(dim, n):
    rho = np.zeros( [dim, dim], dtype=np.float64 )
    rho[0, 0] = 1.0
    rhos = [rho] * n
    return rhos

def linear_combine(u, states, coeffs):
    #print('coeffs: ', coeffs.shape, states.shape)
    assert(len(coeffs) == len(states))
    v = 1.0 - np.sum(coeffs)
    assert(v <= 1.00001 and v >= -0.00001)
    v = max(v, 0.0)
    v = min(v, 1.0)
    total = v * u
    total += np.dot(np.array(states, dtype=np.float64).flatten(), np.array(coeffs, dtype=np.float64).flatten())
    return total

def scale_linear_combine(u, states, coeffs, bias):
    states = (states + bias) / (2.0 * bias)
    value = linear_combine(u, states, coeffs)
    #print(u.shape, 'scale linear combine', value)
    return value

class hqrc(object):
    def delete(self):
        return 0
    
    def __init__(self, params):
        self.display_output = params["display_output"]
        print("RANDOM SEED: {:}".format(params["worker_id"]))
        np.random.seed(params["worker_id"])
        
        self.worker_id = params["worker_id"]
        self.input_dim = params["RDIM"] # energy for SVD model
        self.N_used = params["N_used"]
        
        # Parameters for high-order model
        self.nqrc = params["nqrc"]
        self.alpha = params["alpha"]
        self.max_energy = params["max_energy"]
        self.fix_coupling = params["fix_coupling"]
        self.virtual_nodes = params["virtual_nodes"]
        self.tau = params["tau"]
        self.one_input = params["one_input"]
        self.scale_input = params["scale_input"]
        self.trans_input = params["trans_input"]
        self.bias = params["bias"]
        self.deep = params["deep"]
        self.n_units = params["n_units"]
        self.qubit_count = self.n_units
        self.dim = 2**self.qubit_count
        # Finish

        self.dynamics_length = params["dynamics_length"]
        self.it_pred_length = params["it_pred_length"]
        self.iterative_update_length = params["iterative_update_length"]
        self.n_tests = params["n_tests"]
        self.train_data_path = params["train_data_path"]
        self.test_data_path = params["test_data_path"]
        self.fig_dir = params["fig_dir"]
        self.model_dir = params["model_dir"]
        self.logfile_dir = params["logfile_dir"]
        self.write_to_log = params["write_to_log"]
        self.results_dir = params["results_dir"]
        self.saving_path = params["saving_path"]
        self.reg = params["reg"]
        self.scaler_tt = params["scaler"]
        self.scaler_trans = params["trans"]
        self.scaler_ratio = params["ratio"]

        # Parameters for optimizers
        self.learning_rate = params["learning_rate"]
        self.number_of_epochs = params["number_of_epochs"]
        self.solver = str(params["solver"])
        self.norm_every = params["norm_every"]
        self.augment =  params["augment"]

        ##########################################
        self.scaler = scaler(self.scaler_tt, self.scaler_trans, self.scaler_ratio)
        self.noise_level = params["noise_level"]
        self.model_name = self.createModelName(params)

        self.reference_train_time = 60*60*(params["reference_train_time"]-params["buffer_train_time"])
        print("Reference train time {:} seconds / {:} minutes / {:} hours.".format(self.reference_train_time, self.reference_train_time/60, self.reference_train_time/60/60))

        os.makedirs(self.saving_path + self.model_dir + self.model_name, exist_ok=True)
        os.makedirs(self.saving_path + self.fig_dir + self.model_name, exist_ok=True)
        os.makedirs(self.saving_path + self.results_dir + self.model_name, exist_ok=True)
        os.makedirs(self.saving_path + self.logfile_dir + self.model_name, exist_ok=True)

    def __init_reservoir(self):
        I = [[1,0],[0,1]]
        Z = [[1,0],[0,-1]]
        X = [[0,1],[1,0]]
        P0 = [[1,0],[0,0]]
        P1 = [[0,0],[0,1]]

        self.Zop = [1]*self.qubit_count
        self.Xop = [1]*self.qubit_count
        self.P0op = [1]
        self.P1op = [1]
        self.alpha = self.alpha

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
        self.Zop = np.array(self.Zop, dtype=np.float64)
        self.Xop = np.array(self.Xop, dtype=np.float64)
        self.P0op = np.array(self.P0op, dtype=np.float64)
        self.P1op = np.array(self.P1op, dtype=np.float64)
        # initialize density matrix
        self.last_rhos = generate_list_rho(self.dim, self.nqrc)

        # initialize connection to layer i
        connections = []
        N_local_states = self.n_units * self.virtual_nodes
        nqrc = self.nqrc
        if nqrc > 1:
            for i in range(nqrc):
                local_cs = []
                for j in range(nqrc):
                    cs = [0] * N_local_states
                    if self.deep <= 0:
                        if j != i:
                            cs = np.random.rand(N_local_states)
                    else:
                        if j == i-1:
                            cs = np.random.rand(N_local_states)
                    local_cs.append(cs)
                
                local_cs = np.array(local_cs, dtype=np.float64).flatten()
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
        for i in range(nqrc):
            # generate hamiltonian
            hamiltonian = np.zeros( (self.dim,self.dim), dtype=np.float64 )

            # include input qubit for computation
            for qubit_index in range(self.qubit_count):
                if self.fix_coupling > 0:
                    coef = 2 * self.max_energy
                else:
                    coef = (np.random.rand()-0.5) * 2 * self.max_energy
                hamiltonian += coef * self.Zop[qubit_index]
            for qubit_index1 in range(self.qubit_count):
                for qubit_index2 in range(qubit_index1+1, self.qubit_count):
                    coef = (np.random.rand()-0.5) * 2 * self.max_energy
                    hamiltonian += coef * self.Xop[qubit_index1] @ self.Xop[qubit_index2]
                    
            ratio = float(self.tau) / float(self.virtual_nodes)        
            Uop = sp.linalg.expm(-1.j * hamiltonian * ratio)
            tmp_uops.append(Uop)
        
        self.Uops = tmp_uops.copy()

    def __reset_states(self):
        self.previous_states = [None] * self.nqrc
        self.current_states  = [None] * self.nqrc

    def getKeysInModelName(self):
        keys = {
        'RDIM':'RDIM', 
        'N_used':'N_used', 
        'dynamics_length':'DL',
        'nqrc':'Nqr',
        'alpha':'A',
        #'trans':'sT',
        #'ratio':'sR',
        #'scale_input':'sI',
        'max_energy':'J',
        'fix_coupling':'fJ',
        'virtual_nodes':'V',
        #'tau':'TAU',
        #'n_units':'UNIT',
        #'bias':'B',
        'noise_level':'NL',
        'it_pred_length':'IPL',
        'iterative_update_length':'IUL',
        'reg':'REG',
        #'scaler':'SC',
        #'norm_every':'NE',
        'augment':'AU',
        'n_tests':'NICS',
        #'worker_id':'WID', 
        }
        return keys
    
    def createModelName(self, params):
        keys = self.getKeysInModelName()
        str_ = "hqrc_" + self.solver
        for key in keys:
            str_ += "-" + keys[key] + "_{:}".format(params[key])
        return str_

    def __step_forward(self, local_rhos, input_val):
        nqrc = self.nqrc
        local_prev_states = []
        for i in range(nqrc):
            Uop = self.Uops[i]
            rho = local_rhos[i]
            # Obtain value from the input
            value = 0
            if self.one_input <= 0 or i == 0:
                value = input_val[i] * self.scale_input
            # Obtain values from previous layer
            prev_states = self.previous_states
            #print('Size of prev states', len(prev_states))
            if nqrc > 1 and prev_states[0] is not None:
                #value = softmax_linear_combine(value, previous_states, self.coeffs[i])
                scaled_coeffs = self.coeffs[i] * self.alpha
                value = scale_linear_combine(value, prev_states, scaled_coeffs, self.bias)
            
            # Replace the density matrix
            rho = self.P0op @ rho @ self.P0op + self.Xop[0] @ self.P1op @ rho @ self.P1op @ self.Xop[0]
            # (1 + u Z)/2 = (1+u)/2 |0><0| + (1-u)/2 |1><1|
        
            # for input in [-1, 1]
            # rho = (1+value)/2 * rho + (1-value)/2 *self.Xop[0] @ rho @ self.Xop[0]
            
            # for input in [0, 1]
            # print(len(prev_states), value, rho.shape)
            #if np.isnan(value):
            #    print('input val', input_val)
            #    print('prev states', prev_states)
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
        input_length, input_dim = input_sequence.shape
        print('Input length={}, dim={}'.format(input_length, input_dim))
        
        assert(input_dim == self.nqrc)
        
        predict_sequence = None
        local_rhos = None
        if use_lastrho == True :
            #print('Use last density matrix')
            local_rhos = self.last_rhos.copy()
        else:
            local_rhos = generate_list_rho(self.dim, self.nqrc)
        
        state_list = []
        for time_step in range(0, input_length):
            input_val = input_sequence[time_step]
            local_rhos = self.__step_forward(local_rhos, input_val)

            state = np.array(self.current_states.copy(), dtype=np.float64)
            state_list.append(state.flatten())

        state_list = np.array(state_list, dtype=np.float64)
        self.last_rhos = local_rhos.copy()

        if predict:
            aug_state_list = state_list.copy()
            if self.augment > 0:
                print('Augment data')
                aug_state_list = self.augmentHiddenList(aug_state_list)
                aug_state_list = np.array(aug_state_list, dtype=np.float64)
            
            stacked_state = np.hstack( [aug_state_list, np.ones([input_length, 1])])
            #print('stacked state {}; Wout {}'.format(stacked_state.shape, self.W_out.shape))
            predict_sequence = stacked_state @ self.W_out
        
        return predict_sequence, state_list

    def __train(self, input_sequence, output_sequence):
        print('Training input, output shape', input_sequence.shape, output_sequence.shape)
        assert(input_sequence.shape[0] == output_sequence.shape[0])
        Nout = output_sequence.shape[1]
        self.W_out = np.random.rand(self.getReservoirSize() + 1, Nout)

        # After washing out, use last density matrix to update
        _, state_list = self.__feed_forward(input_sequence, predict=False, use_lastrho=True)

        state_list = np.array(state_list, dtype=np.float64)
        print('State list shape', state_list.shape)

        print("\nSOLVER used to find W_out: {:}. \n\n".format(self.solver))
        if self.solver == "pinv_naive":
            """
            Learn mapping  to S with Penrose Pseudo-Inverse
            No augment data
            """
            X = np.reshape(state_list, [-1, self.getReservoirSize()])
            X = np.hstack( [state_list, np.ones([X.shape[0], 1]) ] )
            Y = np.reshape(output_sequence, [output_sequence.shape[0], -1])
            print('TEACHER FORCING ENDED; direct mapping X Y shape', X.shape, Y.shape)
            W_out = np.linalg.pinv(X, rcond = self.reg) @ Y
        else:
            X, Y = [], []
            # Augment data and using batch normalization
            XTX = np.zeros((self.getReservoirSize() + 1, self.getReservoirSize() + 1))
            XTY = np.zeros((self.getReservoirSize() + 1, output_sequence.shape[1]))
            for t in range(state_list.shape[0]):
                h = state_list[t]
                h_aug = self.augmentHidden(h)
                X.append(h_aug)
                Y.append(output_sequence[t])
                if self.norm_every > 0 and (t+1) % self.norm_every == 0:
                    # Batched approach used in the pinv case
                    X = np.array(X)
                    X = np.hstack( [X, np.ones([X.shape[0], 1])] )
                    Y = np.array(Y)
                    
                    XTX += X.T @ X
                    XTY += X.T @ Y
                    X, Y = [], []
            
            if len(X) != 0:
                # ADDING THE REMAINING BATCH
                X = np.array(X)
                X = np.hstack( [X, np.ones([X.shape[0], 1])] )
                Y = np.array(Y)
                
                XTX += X.T @ X
                XTY += X.T @ Y

            print('TEACHER FORCING ENDED. shape XTX, XTY',  np.shape(XTX), np.shape(XTY))
                
            if self.solver == "pinv":
                I = np.identity(np.shape(XTX)[1])	
                pinv_ = scipypinv2(XTX + self.reg * I)
                W_out = pinv_ @ XTY
            elif self.solver in ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag"]:
                """
                Learns mapping V to S with Ridge Regression
                """
                ridge = Ridge(alpha=self.reg, fit_intercept=False, normalize=False, copy_X=True, solver=self.solver)
                ridge.fit(XTX, XTY) 
                # ridge.fit(A, B) -> A: n_samples x n_features, B: n_samples x n_targets
                # ridge.coef_ -> ndarray of shape (n_features,) or (n_targets, n_features)
                W_out = np.array(ridge.coef_).reshape((-1, Nout))    
            else:
                raise ValueError("Undefined solver.")

        print('Finalizing weights Wout', W_out.shape)
        self.W_out = W_out
        self.n_trainable_parameters = np.size(self.W_out)
        self.n_model_parameters = np.size(self.Uops[0]) * self.nqrc + np.size(self.W_out)
        print("Number of trainable parameters: {}".format(self.n_trainable_parameters))
        print("Total number of parameters: {}".format(self.n_model_parameters))
        print("SAVING MODEL...")
        self.saveModel()

    def train(self):
        self.start_time = time.time()
        dynamics_length = self.dynamics_length
        input_dim = self.input_dim
        N_used = self.N_used

        with open(self.train_data_path, "rb") as file:
            # Pickle the "data" dictionary using the highest protocol available.
            data = pickle.load(file)
            train_input_sequence = data["train_input_sequence"]
            print("Adding noise to the training data. {:} per mille ".format(self.noise_level))
            train_input_sequence = addNoise(train_input_sequence, self.noise_level)
            N_all, dim = np.shape(train_input_sequence)
            if input_dim > dim: raise ValueError("Requested input dimension is wrong.")
            train_input_sequence = train_input_sequence[:N_used, :input_dim]
            dt = data["dt"]
            del data
        print("##Using {:}/{:} dimensions and {:}/{:} samples ##".format(input_dim, dim, N_used, N_all))
        if N_used > N_all: raise ValueError("Not enough samples in the training data.")
        print("SCALING")
        
        # Initialize reservoir
        self.__init_reservoir()
        
        train_input_sequence = self.scaler.scaleData(train_input_sequence)
        N, input_dim = np.shape(train_input_sequence)
        print('Train input sequence shape', train_input_sequence.shape)
        # Replicate intput sequence to fed into high-order machines
        nqrc = self.nqrc
        if int(nqrc) % int(input_dim) != 0:
            ValueError("Number of qrc does not divide input's dimension.")
        K = int(nqrc / input_dim)
        rep_train_input_seq = np.tile(train_input_sequence, (1, K))
        print('Replicate train input sequence shape', train_input_sequence.shape)
        # TRAINING LENGTH
        tl = N - dynamics_length

        print("TRAINING: Dynamics prerun...with layer strength={}".format(self.alpha))
        self.__feed_forward(rep_train_input_seq[:dynamics_length], predict=False, use_lastrho=False)
        print("\n")

        print("TRAINING: Teacher forcing...")
        # Create output
        Y = []
        for t in range(tl - 1):
            target = np.reshape(train_input_sequence[t + dynamics_length + 1], (-1,1))
            Y.append(target[:, 0])
        train_output_sequence = np.array(Y, dtype=np.float64)
        out_length, out_dim = train_output_sequence.shape
        print("TRAINING: Output shape", train_output_sequence.shape)
        print("TEACHER FORCING ENDED.")
        print("\n")
        self.__train(rep_train_input_seq[dynamics_length:(dynamics_length + out_length)], train_output_sequence)

    def isWallTimeLimit(self):
        training_time = time.time() - self.start_time
        if training_time > self.reference_train_time:
            print("## Maximum train time reached. ##")
            return True
        else:
            return False

    def augmentHidden(self, h):
        h_aug = h.copy().ravel()
        h_aug = (1.0 + h_aug) / 2.0
        if self.augment > 0:
            h_aug[::2] = pow(h_aug[::2], 2.0)
        return h_aug

    def augmentHiddenList(self, hs):
        hs_aug = [self.augmentHidden(h) for h in hs]
        return hs_aug

    def getReservoirSize(self): 
        return self.n_units * self.virtual_nodes * self.nqrc
    
    def predictSequence(self, input_sequence):
        dynamics_length = self.dynamics_length
        it_pred_length = self.it_pred_length
        iterative_update_length = self.iterative_update_length

        N, input_dim = np.shape(input_sequence)
        print('Shape of predict sequence:', input_sequence.shape)
        # PREDICTION LENGTH
        if N != it_pred_length + dynamics_length: 
            raise ValueError("Error! N != it_pred_length + dynamics_length")
        
        nqrc = self.nqrc
        if int(nqrc) % int(input_dim) != 0:
            ValueError("Number of qrc does not divide input's dimension.")
        K = int(nqrc / input_dim)
        rep_train_input_seq = np.tile(input_sequence, (1, K))

        self.__reset_states()
        prediction_warm_up, state_list = \
            self.__feed_forward(rep_train_input_seq[:dynamics_length], predict=True, use_lastrho=False)
        print("\n")

        target = input_sequence[dynamics_length:]
        prediction = []

        if True:
            print('Closed loop to generate chaotic signals')
            local_rhos = self.last_rhos.copy()
            nqrc = self.nqrc
            for t in range(it_pred_length):
                state = np.array(self.current_states, dtype=np.float64)
                state_aug = self.augmentHidden(state).reshape((1, -1))
                stacked_state = np.hstack( [state_aug, np.ones([1, 1])])
                #print('PREDICT stage: stacked state {}; Wout {}'.format(stacked_state.shape, self.W_out.shape))
                out = stacked_state @ self.W_out
                prediction.append(out)
                out = out.reshape(1, -1)
                #if np.max(np.abs(out)) > 10:
                #    print('out signal, t={}/{}'.format(t, it_pred_length))
                #    print(out)
                # out[out < 0] = 0.0
                # out[out > 1.0] = 1.0
                # if np.isnan(out) == False:
                #     print('out', out)
                if iterative_update_length > 0 and (t+1) % iterative_update_length == 0:
                    intput_val = rep_train_input_seq[t]
                else:
                    input_val = np.tile(out, (1, K))[0]
                local_rhos = self.__step_forward(local_rhos, input_val)
            self.last_rhos = local_rhos.copy()
        else:
            # Because restart_alpha is set to 1.0
            # It means that, not need input signals
            print('Restart layer strength from {} to 1.0'.format(self.alpha))
            self.alpha = 1.0
            prediction, _ = \
                self.__feed_forward(rep_train_input_seq[dynamics_length:], predict=True, use_lastrho=True)
        print("\n")
        prediction = np.array(prediction, dtype=np.float64).reshape((it_pred_length,-1))
        print('Prediction shape', prediction.shape)
        prediction_warm_up = np.array(prediction_warm_up, dtype=np.float64)
        print('shape prediction and warm_up', prediction.shape, prediction_warm_up.shape)
        target_augment = input_sequence
        prediction_augment = np.concatenate((prediction_warm_up, prediction), axis=0)
        return prediction, target, prediction_augment, target_augment

    def testing(self):
        if self.loadModel() ==  0:
            self.testingOnTrainingSet()
            self.testingOnTestingSet()
            self.saveResults()
        return 0
    
    def testingOnTrainingSet(self):
        n_tests = self.n_tests
        with open(self.test_data_path, "rb") as file:
            data = pickle.load(file)
            testing_ic_indexes = data["testing_ic_indexes"]
            dt = data["dt"]
            del data

        with open(self.train_data_path, "rb") as file:
            data = pickle.load(file)
            train_input_sequence = data["train_input_sequence"][:, :self.input_dim]
            del data
            
        rmnse_avg, num_accurate_pred_005_avg, num_accurate_pred_050_avg, error_freq, predictions_all, truths_all, freq_pred, freq_true, sp_true, sp_pred = self.predictIndexes(train_input_sequence, testing_ic_indexes, dt, "TRAIN")
        
        for var_name in getNamesInterestingVars():
            exec("self.{:s}_TRAIN = {:s}".format(var_name, var_name))
        return 0

    def testingOnTestingSet(self):
        n_tests = self.n_tests
        with open(self.test_data_path, "rb") as file:
            data = pickle.load(file)
            testing_ic_indexes = data["testing_ic_indexes"]
            test_input_sequence = data["test_input_sequence"][:, :self.input_dim]
            dt = data["dt"]
            del data
            
        rmnse_avg, num_accurate_pred_005_avg, num_accurate_pred_050_avg, error_freq, predictions_all, truths_all, freq_pred, freq_true, sp_true, sp_pred = self.predictIndexes(test_input_sequence, testing_ic_indexes, dt, "TEST")
        
        for var_name in getNamesInterestingVars():
            exec("self.{:s}_TEST = {:s}".format(var_name, var_name))
        return 0

    def predictIndexes(self, input_sequence, ic_indexes, dt, set_name):
        n_tests = self.n_tests
        input_sequence = self.scaler.scaleData(input_sequence, reuse=1)
        predictions_all = []
        truths_all = []
        rmse_all = []
        rmnse_all = []
        num_accurate_pred_005_all = []
        num_accurate_pred_050_all = []
        for ic_num in range(n_tests):
            if self.display_output == True:
                print("IC {:}/{:}, {:2.3f}%".format(ic_num, n_tests, ic_num/n_tests*100))
            ic_idx = ic_indexes[ic_num]
            input_sequence_ic = input_sequence[ic_idx-self.dynamics_length:ic_idx+self.it_pred_length]
            prediction, target, prediction_augment, target_augment = self.predictSequence(input_sequence_ic)
            prediction = self.scaler.descaleData(prediction)
            target = self.scaler.descaleData(target)
            rmse, rmnse, num_accurate_pred_005, num_accurate_pred_050, abserror = computeErrors(target, prediction, self.scaler.data_std)
            predictions_all.append(prediction)
            truths_all.append(target)
            rmse_all.append(rmse)
            rmnse_all.append(rmnse)
            num_accurate_pred_005_all.append(num_accurate_pred_005)
            num_accurate_pred_050_all.append(num_accurate_pred_050)
            # PLOTTING ONLY THE FIRST THREE PREDICTIONS
            if ic_num < 3: plotIterativePrediction(self, set_name, target, prediction, rmse, rmnse, ic_idx, dt, target_augment, prediction_augment, self.dynamics_length)

        predictions_all = np.array(predictions_all, dtype=np.float64)
        truths_all = np.array(truths_all, dtype=np.float64)
        rmse_all = np.array(rmse_all, dtype=np.float64)
        rmnse_all = np.array(rmnse_all, dtype=np.float64)
        num_accurate_pred_005_all = np.array(num_accurate_pred_005_all, dtype=np.float64)
        num_accurate_pred_050_all = np.array(num_accurate_pred_050_all, dtype=np.float64)

        print("TRAJECTORIES SHAPES:")
        print(np.shape(truths_all))
        print(np.shape(predictions_all))
        rmnse_avg = np.mean(rmnse_all)
        print("AVERAGE RMNSE ERROR: {:}".format(rmnse_avg))
        num_accurate_pred_005_avg = np.mean(num_accurate_pred_005_all)
        print("AVG NUMBER OF ACCURATE 0.05 PREDICTIONS: {:}".format(num_accurate_pred_005_avg))
        num_accurate_pred_050_avg = np.mean(num_accurate_pred_050_all)
        print("AVG NUMBER OF ACCURATE 0.5 PREDICTIONS: {:}".format(num_accurate_pred_050_avg))
        freq_pred, freq_true, sp_true, sp_pred, error_freq = computeFrequencyError(predictions_all, truths_all, dt)
        print("FREQUENCY ERROR: {:}".format(error_freq))

        plotSpectrum(self, sp_true, sp_pred, freq_true, freq_pred, set_name)
        return rmnse_avg, num_accurate_pred_005_avg, num_accurate_pred_050_avg, error_freq, predictions_all, truths_all, freq_pred, freq_true, sp_true, sp_pred

    def saveResults(self):

        if self.write_to_log == 1:
            logfile_test = self.saving_path + self.logfile_dir + self.model_name  + "/test.txt"
            writeToTestLogFile(logfile_test, self)
            
        data = {}
        for var_name in getNamesInterestingVars():
            exec("data['{:s}_TEST'] = self.{:s}_TEST".format(var_name, var_name))
            exec("data['{:s}_TRAIN'] = self.{:s}_TRAIN".format(var_name, var_name))
        data["model_name"] = self.model_name
        data["n_tests"] = self.n_tests
        data_path = self.saving_path + self.results_dir + self.model_name + "/results.pickle"
        with open(data_path, "wb") as file:
            # Pickle the "data" dictionary using the highest protocol available.
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
            del data
        return 0

    def loadModel(self):
        data_path = self.saving_path + self.model_dir + self.model_name + "/data.pickle"
        try:
            with open(data_path, "rb") as file:
                data = pickle.load(file)
                self.W_out = data["W_out"]
                self.Zop = data["Zop"]
                self.Xop = data["Xop"]
                self.P0op = data["P0op"]
                self.P1op = data["P1op"]
                self.coeffs = data["coeffs"]
                self.Uops = data["Uops"]
                self.scaler = data["scaler"]
                del data
            return 0
        except:
            print("MODEL {:s} NOT FOUND.".format(data_path))
            return 1

    def saveModel(self):
        print("Recording time...")
        self.total_training_time = time.time() - self.start_time
        print("Total training time is {:}".format(self.total_training_time))

        print("MEMORY TRACKING IN MB...")
        process = psutil.Process(os.getpid())
        memory = process.memory_info().rss/1024/1024
        self.memory = memory
        print("Script used {:} MB".format(self.memory))
        print("SAVING MODEL...")

        if self.write_to_log == 1:
            logfile_train = self.saving_path + self.logfile_dir + self.model_name  + "/train.txt"
            writeToTrainLogFile(logfile_train, self)

        data = {
            "memory":self.memory,
            "n_trainable_parameters":self.n_trainable_parameters,
            "n_model_parameters":self.n_model_parameters,
            "total_training_time":self.total_training_time,
            "W_out":self.W_out,
            "Zop":self.Zop,
            "Xop":self.Xop,
            "P0op":self.P0op,
            "P1op":self.P1op,
            "coeffs":self.coeffs,
            "Uops":self.Uops,
            "scaler":self.scaler,
        }
        data_path = self.saving_path + self.model_dir + self.model_name + "/data.pickle"
        with open(data_path, "wb") as file:
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
            del data
        return 0


