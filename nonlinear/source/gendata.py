#!/usr/bin/env python
import numpy as np
import random as rand
from scipy.integrate import ode
import sys

def lorenz(t0, u0, sigma, rho, beta):
    dudt = np.zeros(np.shape(u0))
    dudt[0] = sigma * (u0[1]-u0[0])
    dudt[1] = u0[0] * (rho-u0[2]) - u0[1]
    dudt[2] = u0[0] * u0[1] - beta*u0[2]
    return dudt

def RK4(dudt, u0, t0, dt, sigma, rho, beta):
    k1 = dudt(t0, u0, sigma, rho, beta)
    k2 = dudt(t0+dt/2., u0+dt*k1/2., sigma, rho, beta)
    k3 = dudt(t0+dt/2., u0+dt*k2/2., sigma, rho, beta)
    k4 = dudt(t0+dt, u0+dt*k3, sigma, rho, beta)
    u = u0 + 1./6*(k1+2*k2+2*k3+k4)*dt
    return u

def Lorenz3D(sigma=10.0, rho=28, beta=8.0/3.0, dimensions=3, T1=1000, T2=2000, dt=0.01, random=False):
    # INTEGRATION
    u0 = np.ones((dimensions,1))
    if random == True:
        u0 = np.random.randn(dimensions, 1)
    t0 = 0

    r = ode(lorenz)
    r.set_initial_value(u0, t0).set_f_params(sigma, rho, beta)


    #print("Initial transients...")
    while r.successful() and r.t < T1:
        r.integrate(r.t+dt)
        #sys.stdout.write("\r Time={:.2f}".format(r.t)+ " " * 10)
        #sys.stdout.flush()

    u0 = r.y
    t0 = 0
    r.set_initial_value(u0, t0).set_f_params(sigma, rho, beta)

    u = np.zeros((dimensions, int(T2/dt)+1))
    u[:,0] = np.reshape(u0, (-1))

    #print("Integration...")
    i=1
    while r.successful() and r.t < T2 - dt:
        r.integrate(r.t + dt)
        u[:,i] = np.reshape(r.y, (-1))
        i = i+1
        #sys.stdout.write("\r Time={:.2f}".format(r.t)+ " " * 10)
        #sys.stdout.flush()
    
    u = np.transpose(u)

    data = {
        "sigma":sigma,
        "rho":rho,
        "beta":beta,
        "T1":T1,
        "T2":T2,
        "dt":dt,
        "u":u,
    }
    return data