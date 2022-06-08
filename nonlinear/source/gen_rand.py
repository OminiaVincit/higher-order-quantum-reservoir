import numpy as np

np.random.seed(seed=12345)

input_length = 2001000
input_seq = np.random.rand(input_length)
#input_seq = np.zeros(input_length) + 0.5

dat = np.vstack([range(input_length), input_seq]).T
print(dat.shape)
filename = '../data/rand_input_{}.txt'.format(input_length)
#filename = '../data/const_input_0.5_{}.txt'.format(input_length)

np.savetxt(filename, dat)