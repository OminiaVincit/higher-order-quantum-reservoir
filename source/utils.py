import sys
import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt

def plot(input_sequence_list, output_sequence_list, prediction_sequence_list):
    cmap = plt.get_cmap("tab10")
    plt.figure(figsize=(12,10))
    for index in range(6):
        plt.subplot(6,1,index+1)
        plt.plot(input_sequence_list[index],color=cmap(index),linestyle="--",label="input")
        plt.plot(output_sequence_list[index],color=cmap(index),linestyle=":",label="label")
        plt.plot(prediction_sequence_list[index],color=cmap(index),label="prediction")
        plt.legend()
    plt.show()