# This is a demo to generate a trajectory for a unit mass block
# Author : Avadesh Meduri
# Date : 29/04/2020

import torch
import matplotlib.pyplot as plt
from py_solvers.constrained.augmented_lagrangian.bound_constrained import BoundConstrainedFormulation

def dynamics(x, u, dt):
    return x[0] + dt*x[1], x[1] + dt*u

def ce11(x0):
    return lambda x: x[0] - x0[0]

def ce12(x0):
    return lambda x: x[1] - x0[1]

def ce1(nx, nu, t, dt, dynamics):
    n_t_1 = (nx + nu)*(t + 1)
    n_t = (nx + nu)*(t)
    return lambda x : (x[n_t_1] - dynamics(x[n_t:n_t+nx], x[n_t+nx:n_t+nx+nu], dt)[0])[0] 

def ce2(nx, nu, t, dt, dynamics):
    n_t_1 = (nx + nu)*(t + 1)
    n_t = (nx + nu)*(t)
    return lambda x : (x[n_t_1 + 1] - dynamics(x[n_t:n_t+nx], x[n_t+nx:n_t+nx+nu], dt)[1])[0] 

def ci1(nx, nu, u_lim, t):
    n_t = (nx + nu)*(t)
    return lambda x : x[n_t + nx] + u_lim

def ci2(nx, nu, u_lim, t):
    n_t = (nx + nu)*(t)
    return lambda x : u_lim - x[n_t + nx]


horizon = 5
dt = 0.5
nx = 2
nu = 1
n =  int(nx + (horizon/dt)*(nx) + (horizon/dt)*(nu))
u_lim = torch.tensor(2, dtype = float)
x0 = torch.tensor([[0], [0]], dtype=float)
xdes = torch.tensor([[5], [0]], dtype=float)

print(n)
ce = []
ci = []
ce.append(ce11(x0))
ce.append(ce12(x0))
for i in range(int(horizon/dt)):
    ce.append(ce1(nx, nu, i, dt, dynamics))
    ce.append(ce2(nx, nu, i, dt, dynamics))
    ci.append(ci1(nx, nu, u_lim, i))
    ci.append(ci2(nx, nu, u_lim, i))

def f(x):
    return (x[n -2] - 5)**2 + (x[n-1] - 0)**2

x0 = torch.zeros(n, 1, dtype = float)
# x0[1::3] = 2
# x0[2::3] = 3
# x0[-2] = 0
bcf = BoundConstrainedFormulation(maxit = 100)
x_opt = bcf.optimize(f, ce, ci, n, 0.001, 0.001, 5, x0 = x0, use_sr1 = True)

fig, axs = plt.subplots(2,1)
axs[0].plot(x_opt[0::3], label = "x")
axs[0].plot(x_opt[1::3], label = 'xd')
axs[0].legend()
axs[0].grid()

axs[1].plot(x_opt[2::3], label = "u")
axs[1].legend()
axs[1].grid()

plt.show()