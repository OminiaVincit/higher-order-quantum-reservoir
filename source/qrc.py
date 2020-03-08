import sys
import numpy as np
import scipy as sp
from scipy import linalg
import tqdm
import time
import gen_random
import utils

class QRCParams():
    def __init__(self, hidden_unit_count, max_coupling_energy, trotter_step, beta, virtual_nodes, tau_delta, init_rho):
        self.hidden_unit_count = hidden_unit_count
        self.max_coupling_energy = max_coupling_energy
        self.trotter_step = trotter_step
        self.beta = beta
        self.virtual_nodes = virtual_nodes
        self.tau_delta = tau_delta
        self.init_rho = init_rho

class QuantumReservoirComputing(object):
    def __init_reservoir(self, qparams):
        I = [[1,0],[0,1]]
        Z = [[1,0],[0,-1]]
        X = [[0,1],[1,0]]
        P0 = [[1,0],[0,0]]
        P1 = [[0,0],[0,1]]
        self.hidden_unit_count = qparams.hidden_unit_count
        self.trotter_step = qparams.trotter_step
        self.virtual_nodes = qparams.virtual_nodes
        self.tau_delta = qparams.tau_delta
        
        self.qubit_count = self.hidden_unit_count+1
        self.dim = 2**self.qubit_count
        self.Zop = [1]*self.qubit_count
        self.Xop = [1]*self.qubit_count
        self.P0op = [1]
        self.P1op = [1]
            
        # initialize density matrix
        rho = np.zeros( [self.dim, self.dim] )
        rho[0,0]=1
        if qparams.init_rho != 0:
            # initialize random density matrix
            rho = gen_random.random_density_matrix(self.dim)
        #print(rho)
        self.init_rho = rho

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
            coef = (np.random.rand()-0.5) * 2 * qparams.max_coupling_energy
            self.hamiltonian += coef * self.Zop[qubit_index]
        for qubit_index1 in range(self.qubit_count):
            for qubit_index2 in range(qubit_index1+1, self.qubit_count):
                coef = (np.random.rand()-0.5) * 2 * qparams.max_coupling_energy
                self.hamiltonian += coef * self.Xop[qubit_index1] @ self.Xop[qubit_index2]
                
        ratio = float(self.tau_delta) / float(self.virtual_nodes)        
        self.Uop = sp.linalg.expm(1.j * self.hamiltonian * ratio)

    def __feed_forward(self, input_sequence_list, predict=True):
        sequence_count, sequence_length = input_sequence_list.shape
        predict_sequence_list = []
        state_list = []
        dim = 2**self.qubit_count
        sequence_range = tqdm.trange(sequence_count)
        tau_delta = self.tau_delta
        for sequence_index in sequence_range:
            rho = self.init_rho
            state = []
            for time_step in range(0, sequence_length):
                rho = self.P0op @ rho @ self.P0op + self.Xop[0] @ self.P1op @ rho @ self.P1op @ self.Xop[0]
                # (1 + u Z)/2 = (1+u)/2 |0><0| + (1-u)/2 |1><1|
                value = input_sequence_list[sequence_index, time_step]
                rho = (1+value)/2 * rho + (1-value)/2 *self.Xop[0] @ rho @ self.Xop[0]
                
                # virtual nodes
                current_state = []
                for v in range(self.virtual_nodes):
                    rho = self.Uop @ rho @ self.Uop.T.conj()
                    for qubit_index in range(1,self.qubit_count):
                        expectation_value = np.real(np.trace(self.Zop[qubit_index] @ rho))
                        current_state.append(expectation_value)
                state.append(current_state)
            state = np.array(state)
            state_list.append(state)

            if predict:
                stacked_state = np.hstack( [state, np.ones([sequence_length, 1])])
                predict_sequence = stacked_state @ self.W_out
                # if predict_sequence.shape[1] == 1:
                #     predict_sequence = np.squeeze(predict_sequence, axis=1)
                predict_sequence_list.append(predict_sequence)
        predict_sequence_list = np.array(predict_sequence_list)
        state_list = np.array(state_list)
        return predict_sequence_list, state_list


    def __train(self, input_sequence_list, output_sequence_list, beta):
        #print('shape', input_sequence_list.shape, output_sequence_list.shape)
        assert(input_sequence_list.shape[0] == output_sequence_list.shape[0])
        assert(input_sequence_list.shape[1] == output_sequence_list.shape[1])
        self.sequence_count, self.sequence_length = input_sequence_list.shape
        Nout = output_sequence_list[0].shape[1]
        self.W_out = np.random.rand(self.hidden_unit_count * self.virtual_nodes + 1, Nout)

        _, state_list = self.__feed_forward(input_sequence_list)

        state_list = np.array(state_list)
        V = np.reshape(state_list, [-1, self.hidden_unit_count * self.virtual_nodes])
        V = np.hstack( [V, np.ones([V.shape[0], 1]) ] )
        #print('output seq list', output_sequence_list.shape)
        #S = np.reshape(output_sequence_list, [-1])
        (nx, ny, nz) = output_sequence_list.shape
        S = np.reshape(output_sequence_list, [nx*ny, nz])
        #print('V S', V.shape, S.shape)
        self.W_out = np.linalg.pinv(V, rcond = beta) @ S
        #print('bf Wout', self.W_out.shape)
        #self.W_out = np.expand_dims(self.W_out,axis=1)
        #print('af Wout', self.W_out.shape)

    def train_to_predict(self, input_sequence_list, output_sequence_list, qparams):
        self.__init_reservoir(qparams)
        self.__train(input_sequence_list, output_sequence_list, qparams.beta)

    def predict(self, input_sequence_list,output_sequence_list):
        prediction_sequence_list, _ = self.__feed_forward(input_sequence_list)
        N = prediction_sequence_list.shape[0]
        loss = 0
        for i in range(N):
            loss += np.sum((prediction_sequence_list[i] - output_sequence_list[i])**2)/np.sum(prediction_sequence_list[i]**2)
        loss /= N
        return prediction_sequence_list, loss

def get_loss(qrcparams, train_input_seq_ls, train_output_seq_ls, val_input_seq_ls, val_output_seq_ls):
    model = QuantumReservoirComputing()
    
    train_input_seq_ls = np.array(train_input_seq_ls)
    train_output_seq_ls = np.array(train_output_seq_ls)
    model.train_to_predict(train_input_seq_ls, train_output_seq_ls, qrcparams)

    train_pred_seq_ls, train_loss = model.predict(train_input_seq_ls, train_output_seq_ls)
    print("train_loss={}, shape".format(train_loss, train_pred_seq_ls.shape))
    
    
    # Test phase
    val_input_seq_ls = np.array(val_input_seq_ls)
    val_output_seq_ls = np.array(val_output_seq_ls)
    val_pred_seq_ls, val_loss = model.predict(val_input_seq_ls, val_output_seq_ls)

    print("val_loss={}, shape".format(val_loss), val_pred_seq_ls.shape)

    return train_pred_seq_ls, train_loss, val_pred_seq_ls, val_loss

def evaluation(outbase, qrcparams, train_input_seq_ls, train_output_seq_ls, val_input_seq_ls, val_output_seq_ls):
    
    train_input_seq_ls = np.array(train_input_seq_ls)
    train_output_seq_ls = np.array(train_output_seq_ls)

    val_input_seq_ls = np.array(val_input_seq_ls)
    val_output_seq_ls = np.array(val_output_seq_ls)

    train_pred_seq_ls, train_loss, val_pred_seq_ls, val_loss = \
        get_loss(qrcparams, train_input_seq_ls, train_output_seq_ls, val_input_seq_ls, val_output_seq_ls)
    # save experiments setting
    with open('{}_results.txt'.format(outbase), 'w') as sfile:
        sfile.write('train_loss={}\n'.format(train_loss))
        sfile.write('val_loss={}\n'.format(val_loss))
        sfile.write('hidden_unit_count={}\n'.format(qrcparams.hidden_unit_count))
        sfile.write('max_coupling_energy={}\n'.format(qrcparams.max_coupling_energy))
        sfile.write('trotter_step={}\n'.format(qrcparams.trotter_step))
        sfile.write('beta={}\n'.format(qrcparams.beta))
        sfile.write('virtual nodes={}\n'.format(qrcparams.virtual_nodes))
        sfile.write('tau_delta={}\n'.format(qrcparams.tau_delta))
        sfile.write('init_rho={}\n'.format(qrcparams.init_rho))
    
    rstrls = []
    rstrls.append('train_loss={}'.format(train_loss))
    rstrls.append('val_loss={}'.format(val_loss))
    rstrls.append('hidden_unit={},virtual={}'.format(qrcparams.hidden_unit_count, qrcparams.virtual_nodes))
    rstrls.append('Jdelta={},tau_delta={}'.format(qrcparams.max_coupling_energy, qrcparams.tau_delta))
    #rstrls.append('trotter_step={}'.format(qrcparams.trotter_step))
    #rstrls.append('beta={}'.format(qparams.beta))
    #rstrls.append('init_rho={}'.format(qparams.init_rho))
    rstr = '\n'.join(rstrls)
    print('shape val out and predict', val_output_seq_ls.shape, val_pred_seq_ls.shape)
    utils.plot_predict_multi('{}_train'.format(outbase), rstr, train_input_seq_ls[0], \
        train_output_seq_ls[0].T, train_pred_seq_ls[0].T)

    utils.plot_predict_multi('{}_val'.format(outbase), rstr, val_input_seq_ls[0], \
        val_output_seq_ls[0].T, val_pred_seq_ls[0].T)

def memory_function(taskname, qparams, train_len, val_len, buffer, maxD, Ntrials=1):    
    MFlist = []
    dlist = []
    train_list, val_list = [], []
    length = buffer + train_len + val_len
    # generate data
    if 'stm' not in taskname and 'pc' not in taskname:
        raise ValueError('Not found taskname ={} to generate data'.format(taskname))

    #data = np.random.randint(0, 2, length )
    data = np.random.rand(length)
    for d in range(maxD+1):
        train_input_seq_ls = np.array([ data[buffer  : buffer + train_len] ] )
        val_input_seq_ls = np.array([ data[buffer + train_len : length] ] )
        
        train_out, val_out = [], []
        if 'pc' in taskname:
            print('Generate parity check data')
            for k in range(buffer, length):
                yk = np.sum(data[k-d : k+1]) % 2
                if k >= buffer + train_len:
                    val_out.append(yk)
                else:
                    train_out.append(yk)
        else:
            print('Generate STM task data')
            train_out = data[buffer - d : buffer - d + train_len]
            val_out = data[buffer - d + train_len : length - d ] 
        
        train_output_seq_ls = np.array([ train_out ]).reshape(1, len(train_out), 1)
        val_output_seq_ls = np.array([ val_out ]).reshape(1, len(val_out), 1)
        
        train_loss_ls, val_loss_ls, mfs = [], [], []
        for n in range(Ntrials):
            #print('d={}, trial={}'.format(d, n))
            train_pred_seq_ls, train_loss, val_pred_seq_ls, val_loss = \
                get_loss(qparams, train_input_seq_ls, train_output_seq_ls, val_input_seq_ls, val_output_seq_ls)

            # Compute memory function
            val_output_seq, val_pred_seq = val_output_seq_ls[0].ravel(), val_pred_seq_ls[0].ravel()
            #print('cov', val_output_seq.shape, val_pred_seq.shape)
            cov_matrix = np.cov(np.array([val_output_seq, val_pred_seq]))
            MF_d = cov_matrix[0][1] ** 2
            MF_d = MF_d / (np.var(val_output_seq) * np.var(val_pred_seq))

            train_loss_ls.append(train_loss)
            val_loss_ls.append(val_loss)
            mfs.append(MF_d)

        avg_train, avg_val, avg_MFd = np.mean(train_loss_ls), np.mean(val_loss_ls), np.mean(mfs)
        #print("d={}, train_loss={}, val_loss={}, MF={}".format(d, avg_train, avg_val, avg_MFd))
        MFlist.append(avg_MFd)
        train_list.append(avg_train)
        val_list.append(avg_val)
        dlist.append(d)

    return np.array([dlist, MFlist, train_list, val_list]).T