#!/usr/bin/env python
"""
Quoc Hoan Tran, Nakajima-Lab, The University of Tokyo
    Utils for higher-order quantum reservoir computing framework
"""
import sys
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

class QRCParams():
    def __init__(self, n_units, max_energy, beta, virtual_nodes, tau, init_rho, solver=LINEAR_PINV):
        self.n_units = n_units
        self.max_energy = max_energy
        self.beta = beta
        self.virtual_nodes = virtual_nodes
        self.tau = tau
        self.init_rho = init_rho
        self.solver = solver

    def info(self):
        print('units={},Jdelta={},V={},taudelta={},init_rho={}'.format(\
            self.n_units, self.max_energy,
            self.virtual_nodes, self.tau, self.init_rho))

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

def make_data_for_narma(length, orders):
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