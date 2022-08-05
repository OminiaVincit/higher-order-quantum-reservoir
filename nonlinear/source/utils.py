#!/usr/bin/env python
"""
    Utils for higher-order quantum reservoir computing framework
"""
import sys
import os
import numpy as np
import math
from scipy.stats import unitary_group
from scipy.special import softmax

LINEAR_PINV = 'linear_pinv'
RIDGE_PINV  = 'ridge_pinv'
RIDGE_AUTO  = 'auto'
RIDGE_SVD   = 'svd'
RIDGE_CHOLESKY = 'cholesky'
RIDGE_LSQR = 'lsqr'
RIDGE_SPARSE = 'sparse_cg'
RIDGE_SAG = 'sag'

DYNAMIC_FULL_RANDOM = 'full_random'
DYNAMIC_HALF_RANDOM = 'half_random'
DYNAMIC_FULL_CONST_TRANS = 'full_const_trans'
DYNAMIC_FULL_CONST_COEFF = 'full_const_coeff'
DYNAMIC_ION_TRAP = 'ion_trap'
DYNAMIC_PHASE_TRANS = 'phase_trans'

class QRCParams():
    def __init__(self, n_units, n_envs, max_energy, virtual_nodes, tau, init_rho, \
        beta, solver, dynamic, non_diag_var, non_diag_const=4.0, alpha=1.0):
        self.n_units = n_units
        self.n_envs = n_envs
        self.max_energy = max_energy
        self.non_diag_var = non_diag_var
        self.non_diag_const = non_diag_const
        
        self.alpha = alpha
        self.beta = beta
        self.virtual_nodes = virtual_nodes
        self.tau = tau
        self.init_rho = init_rho
        self.solver = solver
        self.dynamic = dynamic

    def info(self):
        print('units={},n_envs={},J={},non_diag_var={},alpha={},V={},t={},init_rho={}'.format(\
            self.n_units, self.n_envs, self.max_energy, self.non_diag_var, self.alpha,
            self.virtual_nodes, self.tau, self.init_rho))

# class scaler(object):
# 	def __init__(self, tt, trans=0.0, ratio=1.0):
# 		self.tt = tt
# 		self.data_min = 0
# 		self.data_max = 0
# 		self.data_mean = 0
# 		self.data_std = 0
# 		self.trans  = trans
# 		self.ratio = ratio      

# 	def scaleData(self, input_sequence, reuse=None):
# 		# data_mean = np.mean(train_input_sequence,0)
# 		# data_std = np.std(train_input_sequence,0)
# 		# train_input_sequence = (train_input_sequence-data_mean)/data_std
# 		if reuse == None:
# 			self.data_mean = np.mean(input_sequence,0)
# 			self.data_std = np.std(input_sequence,0)
# 			self.data_min = np.min(input_sequence,0)
# 			self.data_max = np.max(input_sequence,0)
# 		if self.tt == "MinMaxZeroOne":
# 			input_sequence = np.array((input_sequence-self.data_min)/(self.data_max-self.data_min))
# 		elif self.tt == "Standard" or self.tt == "standard":
# 			input_sequence = np.array((input_sequence-self.data_mean)/self.data_std)
# 		elif self.tt == "Linear" or self.tt == "linear":
# 			input_sequence = np.array((input_sequence + self.trans) / self.ratio)
# 		elif self.tt != "no":
# 			raise ValueError("Scaler not implemented.")
# 		return input_sequence

# 	def descaleData(self, input_sequence):
# 		if self.tt == "MinMaxZeroOne":
# 			input_sequence = np.array(input_sequence*(self.data_max - self.data_min) + self.data_min)
# 		elif self.tt == "Standard" or self.tt == "standard":
# 			input_sequence = np.array(input_sequence*self.data_std.T + self.data_mean)
# 		elif self.tt == "Linear" or self.tt == "linear":
# 			input_sequence = np.array(input_sequence*self.ratio - self.trans)
# 		elif self.tt != "no":
# 			raise ValueError("Scaler not implemented.")
# 		return input_sequence

def replaceNaN(data):
    data[np.isnan(data)]=float('Inf')
    return data

def cal_NRMSE(pred, truth):
    assert(pred.shape == truth.shape)
    pred = replaceNaN(pred)

    N, M = pred.shape
    sigma = np.std(truth[:, :], axis=0)
    sigma2 = np.square(sigma)
    rs = []
    for i in range(N):
        diff = pred[i, :] - truth[i, :]
        diff2 = np.square(diff)
        mse  = np.mean(diff2 / sigma2)
        rmse = np.sqrt(mse)
        rs.append(rmse)
    return rs

def get_num_accurate_pred(nerror, thresh=0.05):
    nerror_bool = nerror < thresh
    n_max = np.shape(nerror)[0]
    n = 0
    while nerror_bool[n] == True:
        n += 1
        if n == n_max: break
    return n

def clipping(val, minval, maxval):
    return max(min(val, maxval), minval)

def add_noise(data, percent):
    std_data = np.std(data, axis=0)
    std_data = np.reshape(std_data, (1, -1))
    std_data = np.repeat(std_data, np.shape(data)[0], axis=0)
    noise = np.multiply(np.random.randn(*np.shape(data)), percent/1000.0*std_data)
    data += noise
    return data

def min_max_norm(tmp_arr, min_arr, max_arr):
    if min_arr is None or max_arr is None:
        return tmp_arr
    tmp_arr = tmp_arr - min_arr
    tmp_arr = np.divide(tmp_arr, max_arr - min_arr)
    return tmp_arr

def solfmax_layer(states):
    states = np.array(states)
    return softmax(states)

def softmax_linear_combine(u, states, coeffs):
    states = solfmax_layer(states)
    return linear_combine(u, states, coeffs)

def linear_combine(u, states, coeffs):
    assert(len(coeffs) == len(states))
    v = 1.0 - np.sum(coeffs)
    assert(v <= 1.00001 and v >= -0.00001)
    v = max(v, 0.0)
    v = min(v, 1.0)
    total = v * u
    total += np.dot(np.array(states).flatten(), np.array(coeffs).flatten())
    return total

def scale_linear_combine(u, states, coeffs, bias):
    states = (states + bias) / (2.0 * bias)
    return linear_combine(u, states, coeffs)

def make_data_for_narma(length, orders, ranseed=0):
    np.random.seed(seed=ranseed)
    xs = np.random.rand(length)
    x = xs * 0.2
    N = len(orders)
    Y = np.zeros((length, N))
    for j in range(N):
        order = orders[j]
        y = np.zeros(length)
        if order == 2:
            for i in range(length):
                y[i] = 0.4 * y[i-1] + 0.4 * y[i-1]*y[i-2] + 0.6 * (x[i]**3) + 0.1
        else:
            for i in range(length):
                if i < order:
                    y[i] = 0.3 * y[i - 1] + 0.05 * y[i - 1] * np.sum(np.hstack((y[i - order:], y[:i]))) + \
                        1.5 * x[i - order + 1] * x[i] + 0.1
                else:
                    y[i] = 0.3 * y[i - 1] + 0.05 * y[i - 1] * np.sum(np.hstack((y[i - order:i]))) + \
                        1.5 * x[i - order + 1] * x[i] + 0.1
        Y[:,j] = y
    return xs, Y

def partial_trace(rho, keep, dims, optimize=False):
    """
    Calculate the partial trace.
    Consider a joint state ρ on the Hilbert space :math:`H_a \otimes H_b`. We wish to trace out
    :math:`H_b`
    .. math::
        ρ_a = Tr_b(ρ)
    :param rho: 2D array, the matrix to trace.
    :param keep: An array of indices of the spaces to keep after being traced. For instance,
                 if the space is A x B x C x D and we want to trace out B and D, keep = [0, 2].
    :param dims: An array of the dimensions of each space. For example, if the space is
                 A x B x C x D, dims = [dim_A, dim_B, dim_C, dim_D].
    :param optimize: optimize argument in einsum
    :return:  ρ_a, a 2D array i.e. the traced matrix
    """
    # Code from
    # https://scicomp.stackexchange.com/questions/30052/calculate-partial-trace-of-an-outer-product-in-python
    keep = np.asarray(keep)
    dims = np.asarray(dims)
    Ndim = dims.size
    Nkeep = np.prod(dims[keep])

    idx1 = [i for i in range(Ndim)]
    idx2 = [Ndim + i if i in keep else i for i in range(Ndim)]
    rho_a = rho.reshape(np.tile(dims, 2))
    rho_a = np.einsum(rho_a, idx1 + idx2, optimize=optimize)
    return rho_a.reshape(Nkeep, Nkeep)


# Reference from
# https://qiskit.org/documentation/_modules/qiskit/quantum_info/random/utils.html

def random_state(dim, seed=None):
    """
    Return a random quantum state from the uniform (Haar) measure on
    state space.

    Args:
        dim (int): the dim of the state space
        seed (int): Optional. To set a random seed.

    Returns:
        ndarray:  state(2**num) a random quantum state.
    """
    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    rng = np.random.RandomState(seed)
    # Random array over interval (0, 1]
    x = rng.rand(dim)
    x += x == 0
    x = -np.log(x)
    sumx = sum(x)
    phases = rng.rand(dim) * 2.0 * np.pi
    return np.sqrt(x / sumx) * np.exp(1j * phases)

def random_density_matrix(length, rank=None, method='Hilbert-Schmidt', seed=None):
    """
    Generate a random density matrix rho.

    Args:
        length (int): the length of the density matrix.
        rank (int or None): the rank of the density matrix. The default
            value is full-rank.
        method (string): the method to use.
            'Hilbert-Schmidt': sample rho from the Hilbert-Schmidt metric.
            'Bures': sample rho from the Bures metric.
        seed (int): Optional. To set a random seed.
    Returns:
        ndarray: rho (length, length) a density matrix.
    Raises:
        QiskitError: if the method is not valid.
    """
    if method == 'Hilbert-Schmidt':
        return __random_density_hs(length, rank, seed)
    elif method == 'Bures':
        return __random_density_bures(length, rank, seed)
    else:
        raise ValueError('Error: unrecognized method {}'.format(method))


def __ginibre_matrix(nrow, ncol=None, seed=None):
    """
    Return a normally distributed complex random matrix.

    Args:
        nrow (int): number of rows in output matrix.
        ncol (int): number of columns in output matrix.
        seed (int): Optional. To set a random seed.
    Returns:
        ndarray: A complex rectangular matrix where each real and imaginary
            entry is sampled from the normal distribution.
    """
    if ncol is None:
        ncol = nrow
    rng = np.random.RandomState(seed)

    ginibre = rng.normal(size=(nrow, ncol)) + rng.normal(size=(nrow, ncol)) * 1j
    return ginibre


def __random_density_hs(length, rank=None, seed=None):
    """
    Generate a random density matrix from the Hilbert-Schmidt metric.

    Args:
        length (int): the length of the density matrix.
        rank (int or None): the rank of the density matrix. The default
            value is full-rank.
        seed (int): Optional. To set a random seed.
    Returns:
        ndarray: rho (N,N  a density matrix.
    """
    ginibre = __ginibre_matrix(length, rank, seed)
    ginibre = ginibre.dot(ginibre.conj().T)
    return ginibre / np.trace(ginibre)


def __random_density_bures(length, rank=None, seed=None):
    """
    Generate a random density matrix from the Bures metric.

    Args:
        length (int): the length of the density matrix.
        rank (int or None): the rank of the density matrix. The default
            value is full-rank.
        seed (int): Optional. To set a random seed.
    Returns:
        ndarray: rho (N,N) a density matrix.
    """
    density = np.eye(length) + random_unitary(length).data
    ginibre = density.dot(__ginibre_matrix(length, rank, seed))
    ginibre = ginibre.dot(ginibre.conj().T)
    return ginibre / np.trace(ginibre)

def plot_lorentz(target_seq, pred_seq, nrmse, buffer, train_len, val_len, outbase, n_title, \
    ftypes=['png'], pertubed_targets=[]):
    import matplotlib.pyplot as plt

    # Plot to file
    plt.rc('font', family='serif')
    plt.rc('mathtext', fontset='cm')
    plt.rcParams["font.size"] = 20
    plt.rcParams['xtick.labelsize'] = 24
    plt.rcParams['ytick.labelsize'] = 24
    #plt.rcParams['agg.path.chunksize'] = 100000

    fig = plt.figure(figsize=(20, 24))

    ax = plt.subplot2grid((4, 1), (0,0), projection='3d', colspan=1, rowspan=3)
    
    ax.plot3D(target_seq[train_len:, 0], target_seq[train_len:, 1], target_seq[train_len:, 2], label='Target', alpha=0.9, rasterized=True, linestyle='-')
    ax.plot3D(pred_seq[train_len:, 0], pred_seq[train_len:, 1], pred_seq[train_len:, 2], '.', label='Predict',alpha=0.8, rasterized=True, linestyle='None')
    for i in range(len(pertubed_targets)):
        pertubed_preds = pertubed_targets[i]
        ax.plot3D(pertubed_preds[train_len:, 0], pertubed_preds[train_len:, 1], pertubed_preds[train_len:, 2], \
            label='Target-{}'.format(i+1), alpha=0.6, rasterized=True, linestyle='-')
    
    #seqlen = len(target_seq)
    # s = 10
    # cmap_target = plt.cm.winter
    # cmap_pred = plt.cm.viridis
        
    # for i in range(train_len, seqlen-s, s):
    #     c = cmap_pred( (i-train_len) / (seqlen-train_len))
    #     ax.plot3D(pred_seq[i:i+s+1, 0], pred_seq[i:i+s+1, 1], pred_seq[i:i+s+1, 2], label='Predict', color=c)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.9))
    ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.9))
    ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.9))
    ax.grid(False)
    ax.legend()
    ax.set_title(n_title, fontsize=12)
    ax.set_xlim([-25, 25])
    ax.set_ylim([-25, 25])
    ax.set_zlim([0, 50])
    #ax.set_axis_off()
    #ax.set_facecolor('k')

    bx = plt.subplot2grid((4, 1), (3,0), colspan=1, rowspan=1)
    bx.plot(range(buffer + 1, buffer + train_len + val_len), nrmse, linewidth=2.0, rasterized=True, linestyle='-')
    bx.axvline(x=buffer, label='T-buffer', c='k')
    bx.axvline(x=buffer + train_len, label='T-train', c='r')
    bx.set_yscale('log')
    bx.tick_params('both', length=10, width=1, which='both', \
            labelsize=16, direction='in', grid_alpha=0.6)
    bx.legend()
    bx.set_title(os.path.basename(outbase), fontsize=12)
    bx.set_ylabel('NRMSE')
    bx.set_xlabel('Time steps')

    
    for ftype in ftypes:
        transparent = (ftype != 'png')
        figfile = '{}.{}'.format(outbase, ftype)
        plt.savefig(figfile, bbox_inches='tight', transparent=transparent, dpi=120)
    #plt.show()
    plt.clf()