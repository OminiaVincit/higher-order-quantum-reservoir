import sys
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
import os 
import time


def plot_predict(input_sequence_list, output_sequence_list, prediction_sequence_list):
    cmap = plt.get_cmap("tab10")
    N = len(input_sequence_list)
    plt.figure(figsize=(12,2*N))
    
    for index in range(N):
        plt.subplot(N,1,index+1)
        #plt.plot(input_sequence_list[index][1000:1100],color=cmap(index),linestyle="--",label="input")
        plt.plot(output_sequence_list[index][1000:1200],color=cmap(index),linestyle=":",label="label")
        plt.plot(prediction_sequence_list[index][1000:1200],color=cmap(index),label="prediction")
        plt.legend()
    plt.show()

def plot_predict_multi(savefile, rstr, input_sequence, output_sequence_multi, prediction_sequence_multi):
    cmap = plt.get_cmap("tab10")
    N = len(output_sequence_multi) + 1
    plt.figure(figsize=(12,2*N))
    bg, ed = 1000, 1200
    plt.subplot(N, 1, 1)
    plt.plot(input_sequence[bg:ed], color=cmap(0),linestyle="--",label=rstr)
    plt.legend()
    print('output multi', output_sequence_multi.shape)
    for index in range(N-1):
        plt.subplot(N,1,index+2)
        cl = cmap(index+1)
        plt.plot(output_sequence_multi[index][bg:ed],color=cl,linestyle=":",label="label")
        plt.plot(prediction_sequence_multi[index][bg:ed],color=cl,label="prediction")
        plt.legend()

    for ftype in ['png', 'pdf', 'svg']:
        plt.savefig('{}.{}'.format(savefile, ftype), bbox_inches='tight')
    #plt.show()
