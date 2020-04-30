import sys
import numpy as np
import scipy as sp
from scipy import linalg
from numpy import linalg as LA
import qdistance as qd
import gen_random
import matplotlib.pyplot as plt

I = [[1,0],[0,1]]
Z = [[1,0],[0,-1]]
X = [[0,1],[1,0]]
P0 = [[1,0],[0,0]]
P1 = [[0,0],[0,1]]
qubit_count = 5
dim = 2**qubit_count
Zop = [1]*qubit_count
Xop = [1]*qubit_count
P0op = [1]
P1op = [1]

for cursor_index in range(qubit_count):
    for qubit_index in range(qubit_count):
        if cursor_index == qubit_index:
            Xop[qubit_index] = np.kron(Xop[qubit_index],X)
            Zop[qubit_index] = np.kron(Zop[qubit_index],Z)
        else:
            Xop[qubit_index] = np.kron(Xop[qubit_index],I)
            Zop[qubit_index] = np.kron(Zop[qubit_index],I)

    if cursor_index == 0:
        P0op = np.kron(P0op, P0)
        P1op = np.kron(P1op, P1)
    else:
        P0op = np.kron(P0op, I)
        P1op = np.kron(P1op, I)

K=1
ratio = 16.0

hamiltonian = np.zeros( (dim,dim) )
for qubit_index in range(qubit_count):
    coef = (np.random.rand()-0.5) * 2 * K
    hamiltonian += coef * Zop[qubit_index]
for qubit_index1 in range(qubit_count):
    for qubit_index2 in range(qubit_index1+1, qubit_count):
        coef = (np.random.rand()-0.5) * 2 * K
        hamiltonian += coef * Xop[qubit_index1] @ Xop[qubit_index2]
          
Uop = sp.linalg.expm(1.j * hamiltonian * ratio)

print('Uop Norm 2: ', LA.norm(Uop, 2))
print('Hamitonian Norm 2: ', LA.norm(hamiltonian, 2))

rho1 = gen_random.random_density_matrix(dim)
rho2 = gen_random.random_density_matrix(dim)
df = qd.trace_distance(rho1, rho2)

print('trace dist', df)

dfs = []
for k in range(2000):
   rho1 = Uop @ rho1 @ Uop.T.conj()
   rho2 = Uop @ rho2 @ Uop.T.conj()
   dfs.append(qd.trace_distance(rho1, rho2))

plt.figure()
plt.plot(range(len(dfs)), dfs)
plt.show()