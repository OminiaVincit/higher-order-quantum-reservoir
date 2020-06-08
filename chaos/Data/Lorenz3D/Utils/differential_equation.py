#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
"""
import numpy as np

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











