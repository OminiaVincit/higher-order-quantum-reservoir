import sys
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import os 
import time

def memory_function(qparams, train_len, val_len, buffer, L=150, N=1):    
    MFlist = []
    dlist = []
    train_list, val_list = [], []
    # generate data
    data = np.random.rand(train_len + buffer + val_len)
    model = qparams.model
    
    for d in range(L):
        train_input_seq_ls = np.array([ data[buffer  : buffer + train_len] ] )
        train_output_seq_ls = np.array([ data[buffer - d : buffer - d + train_len] ])

        val_input_seq_ls = np.array([ data[buffer + train_len : buffer + train_len + val_len] ] )
        val_output_seq_ls = np.array([ data[buffer - d + train_len : buffer - d + train_len + val_len] ])
        
        train_loss_ls, val_loss_ls, mfs = [], [], []
        for n in range(N):
            #print('d={}, trial={}'.format(d, n))
            model.train(train_input_seq_ls, train_output_seq_ls, qparams.hidden_unit_count, \
                qparams.max_coupling_energy, qparams.trotter_step, qparams.beta)

            train_pred_seq_ls, train_loss = model.predict(train_input_seq_ls, train_output_seq_ls)

            # Test phase
            val_pred_seq_ls, val_loss = model.predict(val_input_seq_ls, val_output_seq_ls)

            # Compute memory function
            val_output_seq, val_pred_seq = val_output_seq_ls[0], val_pred_seq_ls[0]
            cov_matrix = np.cov(np.array([val_output_seq, val_pred_seq]))
            MF_d = cov_matrix[0][1] ** 2
            MF_d = MF_d / (np.var(val_output_seq) * np.var(val_pred_seq))

            train_loss_ls.append(train_loss)
            val_loss_ls.append(val_loss)
            mfs.append(MF_d)

        avg_train, avg_val, avg_MFd = np.mean(train_loss_ls), np.mean(val_loss_ls), np.mean(mfs)
        print("d={}, train_loss={}, val_loss={}, MF={}".format(d, avg_train, avg_val, avg_MFd))
        MFlist.append(avg_MFd)
        train_list.append(avg_train)
        val_list.append(avg_val)
        dlist.append(d)

    return np.array([dlist, MFlist, train_list, val_list]).T


def plot_predict(input_sequence_list, output_sequence_list, prediction_sequence_list):
    cmap = plt.get_cmap("tab10")
    N = len(input_sequence_list)
    plt.figure(figsize=(12,2*N))
    
    for index in range(N):
        plt.subplot(N,1,index+1)
        plt.plot(input_sequence_list[index],color=cmap(index),linestyle="--",label="input")
        plt.plot(output_sequence_list[index],color=cmap(index),linestyle=":",label="label")
        plt.plot(prediction_sequence_list[index],color=cmap(index),label="prediction")
        plt.legend()
    plt.show()
