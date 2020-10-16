#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
    Innate learning for higher-order quantum reservoir
    Created by:  Quoc Hoan Tran
    Project: Higher-order quantum reservoir computing by Quoc Hoan Tran
    
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
from qrc_utils import *
import pickle
import time
from functools import partial
print = partial(print, flush=True)

from sklearn.linear_model import Ridge

# MEMORY TRACKING
import psutil

class hqrc_innate(object):
    def delete(self):
        return 0
    
    def __init__(self, params):
        self.display_output = params["display_output"]
        self.ranseed = params["worker_id"]
        print("RANDOM SEED: {:}".format(self.ranseed))
        np.random.seed(self.ranseed)
        
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
        self.n_qubits = self.n_units
        self.dim = 2**self.n_qubits
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
        self.output_noise = params["output_noise"]
        self.select_qubit = params["select_qubit"]
        self.innate_learning_rate = params["innate_learning_rate"]
        self.innate_learning_loops = params["innate_learning_loops"]
        
        self.innate_flag = True
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

        self.Zop = [1]*self.n_qubits
        self.Xop = [1]*self.n_qubits
        self.P0op = [1]
        self.P1op = [1]
        self.alpha = self.alpha
        self.W_feed = None

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
        self.Zop = np.array(self.Zop, dtype=np.float64)
        self.Xop = np.array(self.Xop, dtype=np.float64)
        self.P0op = np.array(self.P0op, dtype=np.float64)
        self.P1op = np.array(self.P1op, dtype=np.float64)

        nqrc = self.nqrc

        # initialize current states
        self.current_states  = [None] * nqrc

        # initialize connection W_feed
        self.__init_w_feed(self.ranseed, reset_zero=False)
        
        
        # Intialize evolution operators
        self.__init_spins(self.ranseed)

        # initialize density matrix
        self.__init_rhos(ranseed=self.ranseed, rand_rho = False)
        

    def __init_w_feed(self, ranseed, reset_zero=False):
        nqrc  = self.nqrc
        N_local = self.get_local_nodes()
        N_tot = self.get_comput_nodes()

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
                rowsum = np.sum(W_feed[i, :])
                if rowsum != 0:
                    W_feed[i, :] = self.alpha * W_feed[i, :] / rowsum

            if self.W_feed is not None:
                print('W_feed diff:', np.sum(np.abs(W_feed - self.W_feed)))
            self.W_feed = W_feed

    def __init_spins(self, ranseed):
        np.random.seed(seed=ranseed)
        tmp_uops = []

        for i in range(self.nqrc):
            # generate hamiltonian
            hamiltonian = np.zeros( (self.dim,self.dim) )

            # include input qubit for computation
            for qindex in range(self.n_qubits):
                if self.fix_coupling > 0:
                    coef = 2 * self.max_energy
                else:
                    coef = (np.random.rand()-0.5) * 2 * self.max_energy
                hamiltonian += coef * self.Zop[qindex]
            for qubit_index1 in range(self.n_qubits):
                for qubit_index2 in range(qubit_index1+1, self.n_qubits):
                    coef = (np.random.rand()-0.5) * 2 * self.max_energy
                    hamiltonian += coef * self.Xop[qubit_index1] @ self.Xop[qubit_index2]
                    
            ratio = float(self.tau) / float(self.virtual_nodes)        
            Uop = sp.linalg.expm(-1.j * hamiltonian * ratio)
            tmp_uops.append(Uop)
        self.Uops = tmp_uops.copy()

    def __reset_states(self):
        self.current_states  = [None] * self.nqrc

    def __init_rhos(self, ranseed, rand_rho = False):
        self.last_rhos = generate_list_rho(self.dim, self.nqrc, ranseed, rand_rho)

    def getKeysInModelName(self):
        keys = {
        'RDIM':'RDIM', 
        'N_used':'N_used', 
        'dynamics_length':'DL',
        'nqrc':'Nqr',
        'alpha':'A',
        #'trans':'sT',
        #'ratio':'sR',
        'scale_input':'sI',
        'max_energy':'J',
        'fix_coupling':'fJ',
        'virtual_nodes':'V',
        #'tau':'TAU',
        #'n_units':'UNIT',
        #'bias':'B',
        'noise_level':'NL',
        'output_noise':'ON',
        'innate_learning_rate':'INR',
        'innate_learning_loops':'INL',
        'it_pred_length':'IPL',
        #'iterative_update_length':'IUL',
        'reg':'REG',
        #'scaler':'SC',
        #'norm_every':'NE',
        #'augment':'AU',
        'n_tests':'NICS',
        #'worker_id':'WID', 
        }
        return keys
    
    def createModelName(self, params):
        keys = self.getKeysInModelName()
        str_ = "hqrc_innate_" + self.solver
        for key in keys:
            str_ += "-" + keys[key] + "_{:}".format(params[key])
        return str_

    def get_comput_nodes(self):
        return self.n_units * self.virtual_nodes * self.nqrc
    
    def get_local_nodes(self):
        return self.n_units * self.virtual_nodes

    def __step_forward(self, local_rhos, external_input, innate_train=False, out_train=False, \
            innate_target=None, out_target=None):
        nqrc = self.nqrc
        alpha = self.alpha
        N_local = self.get_local_nodes()
        N_tot = self.get_comput_nodes()
        sel = self.select_qubit
        X = self.current_states
        dW_recurr_mag = 0
        learning_rate = self.innate_learning_rate
        noise_amp = self.output_noise

        # Innate training
        if (X[0] is not None) and innate_train and (innate_target is not None) and learning_rate > 0.0:
            for i in range(nqrc):
                error = (X[sel + i*N_local] - innate_target[i])
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
            feed_sigs = self.W_feed @ X
            feed_sigs = feed_sigs/(2*len(feed_sigs))
            maxval, minval = np.max(feed_sigs), np.min(feed_sigs)
            if maxval > 1.0 or minval < 0.0:
                print('feed_sigs={}, external_input={}'.format(feed_sigs, external_input))
            new_input = external_input + feed_sigs
        
        new_input = self.scale_input * new_input
        new_input = new_input.flatten()
        maxval, minval = np.max(new_input), np.min(new_input)
        if maxval > 1.0 or minval < 0.0:
            print('max={}, min={}, diff={}'.format(maxval, minval, np.sum(np.abs(new_input - external_input))))

        cur_states = []
        for i in range(nqrc):
            Uop = self.Uops[i]
            rho = local_rhos[i]
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
            local_rhos[i] = rho
        # update current states
        self.current_states = np.array(cur_states).ravel()
        noise = np.random.normal(loc=0.0, scale=np.sqrt(noise_amp), size=len(self.current_states))
        #noise = (np.random.rand(len(self.cur_states)) - 0.5) * 2.0 * noise_amp
        self.current_states += noise

        return local_rhos, dW_recurr_mag

    def __feed_forward(self, input_sequence, predict, use_lastrho, sel=0):
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
            local_rhos, _ = self.__step_forward(local_rhos, input_val)

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
        N_local = self.get_local_nodes()
        sel = self.select_qubit

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


        # Create output
        Y = []
        for t in range(tl - 1):
            target = np.reshape(train_input_sequence[t + dynamics_length + 1], (-1,1))
            Y.append(target[:, 0])
        train_output_sequence = np.array(Y, dtype=np.float64)
        out_length, out_dim = train_output_sequence.shape
        
        # Stage 1: Innate training with online learning
        if self.innate_learning_rate > 0.0 and self.innate_learning_loops > 0:
            print("Innate: Dynamics prerun...with layer strength={}".format(self.alpha))
            innate_length = dynamics_length + out_length
            _, innate_target_state_list = self.__feed_forward(rep_train_input_seq[:innate_length], predict=False, use_lastrho=False)
            print("\n")
            # Create innate target training
            target_innate_seq = innate_target_state_list[:, sel::N_local]
            self.innate_train(rep_train_input_seq[:innate_length], target_innate_seq, \
                buffer=dynamics_length, train_len=out_length)
        
        # Stage 2: Readout Training
        # Move dynamic again
        self.__reset_states()
        print("TRAINING: Dynamics prerun...with layer strength={}".format(self.alpha))
        self.__feed_forward(rep_train_input_seq[:dynamics_length], predict=False, use_lastrho=False)
        print("\n")

        print("TRAINING: Teacher forcing...")
        print("TRAINING: Output shape", train_output_sequence.shape)
        self.__train(rep_train_input_seq[dynamics_length:(dynamics_length + out_length)], train_output_sequence)
        print("TEACHER FORCING ENDED.")
        print("\n")

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
    
    def innate_train(self, input_seq, innate_seq, buffer, train_len):
        # reset w_feed
        self.__init_w_feed(ranseed=self.ranseed, reset_zero=False)

        # Initialize P_recurr
        Plist = []
        nqrc = self.nqrc
        sel = self.select_qubit
        for i in range(nqrc):
            Plist.append(np.eye(self.get_local_nodes() * (nqrc-1)))
        self.P_recurr = Plist
        N_local = self.get_local_nodes()

        loss_dict = dict()
        for i in range(self.nqrc):
            loss_dict[i] = []
        
        for k in range(self.innate_learning_loops):
            print('{} Innate Train loop, input_seq shape={}, innate_seq.shape={}'.format(\
                k, input_seq.shape, innate_seq.shape))
            
            self.__reset_states()
            self.__init_rhos(ranseed=self.ranseed + k, rand_rho=True)

            # Stage 1: Transient regime
            _, state_list = self.__feed_forward(input_seq[:buffer], predict=False, use_lastrho=False)
            dW_recurr_ls = [0] * buffer

            # Stage 2: Innate training for W_feed
            tmp_list = []
            local_rhos = self.last_rhos.copy()
            for i in range(buffer, buffer + train_len):
                local_rhos, dW_recurr_mag = self.__step_forward(local_rhos, input_seq[i], innate_train=True, \
                    innate_target=innate_seq[i-1])
                dW_recurr_ls.append(dW_recurr_mag)
                state = np.array(self.current_states.copy(), dtype=np.float64)
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
                print('Innate Train={}, QR={}, Loss={}'.format(k, i, loss)) 

        return state_list, dW_recurr_ls, loss_dict


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
        prediction_warm_up, innate_target_state_list = \
            self.__feed_forward(rep_train_input_seq[:dynamics_length], predict=False, use_lastrho=False)
        print("\n")

        # Move dynamic again
        self.__reset_states()
        prediction_warm_up, innate_target_state_list = \
            self.__feed_forward(rep_train_input_seq[:dynamics_length], predict=True, use_lastrho=False)
        print("\n")

        # Predict stage
        target = input_sequence[dynamics_length:]
        prediction = []

        if True:
            print('Closed loop to generate chaotic signals')
            local_rhos = self.last_rhos.copy()
            
            # Predict stage
            for t in range(it_pred_length):
                state = np.array(self.current_states, dtype=np.float64)
                state_aug = self.augmentHidden(state).reshape((1, -1))
                stacked_state = np.hstack( [state_aug, np.ones([1, 1])])
                #print('PREDICT stage: stacked state {}; Wout {}'.format(stacked_state.shape, self.W_out.shape))
                out = stacked_state @ self.W_out
                prediction.append(out)
                out = out.reshape(1, -1)
                
                if iterative_update_length > 0 and (t+1) % iterative_update_length == 0:
                    intput_val = rep_train_input_seq[t]
                else:
                    input_val = np.tile(out, (1, K))[0]
                local_rhos, _ = self.__step_forward(local_rhos, input_val)
            
            self.last_rhos = local_rhos.copy()

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
                self.W_feed = data["W_feed"]
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
            "W_feed":self.W_feed,
            "Uops":self.Uops,
            "scaler":self.scaler,
        }
        data_path = self.saving_path + self.model_dir + self.model_name + "/data.pickle"
        with open(data_path, "wb") as file:
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
            del data
        return 0


