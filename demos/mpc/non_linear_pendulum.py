# This file is a demo that generates a trajecory for a non linear inverted
# pendulum such that it minimizes velocity at the end of the step and also
# minimizes the distance between the center of mass and next step location
# Author : Avadesh Meduri
# Date : 1/05/2020

import torch
import matplotlib.pyplot as plt
from py_solvers.constrained.augmented_lagrangian.bound_constrained import BoundConstrainedFormulation

def dynamics(x, u, cop, dt):
    assert len(x) == 6
    x_next = torch.zeros(6, 1, dtype=float)
    x_next[0] = x[0] + dt*x[3]
    x_next[1] = x[1] + dt*x[4]
    x_next[2] = x[2] + dt*x[5]
    x_next[3] = x[3] + dt*((x[0] - cop[0])*(9.81 - u))/(x[2] - cop[2])
    x_next[4] = x[4] + dt*((x[1] - cop[1])*(9.81 - u))/(x[2] - cop[2])
    x_next[5] = x[5] + dt*(u - 9.81)

    return x_next

#initial state constraints
def ce11(x0):
    return lambda x : x[0] - x0[0]

def ce12(x0):
    return lambda x : x[1] - x0[1]

def ce13(x0):
    return lambda x : x[2] - x0[2]

def ce14(x0):
    return lambda x : x[3] - x0[3]

def ce15(x0):
    return lambda x : x[4] - x0[4]

def ce16(x0):
    return lambda x : x[5] - x0[5]

# dynamics constratins

def ce1(nx, nu, t, dt, cop, dynamics):
    n_t_1 = (nx + nu)*(t + 1)
    n_t = (nx + nu)*t
    return lambda x : x[n_t_1] - dynamics(x[n_t:n_t + nx], x[n_t+nx:n_t+nx+nu], cop, dt)[0]

def ce2(nx, nu, t, dt, cop, dynamics):
    n_t_1 = (nx + nu)*(t + 1)
    n_t = (nx + nu)*t
    return lambda x : x[n_t_1 + 1] - dynamics(x[n_t:n_t + nx], x[n_t+nx:n_t+nx+nu], cop, dt)[1]

def ce3(nx, nu, t, dt, cop, dynamics):
    n_t_1 = (nx + nu)*(t + 1)
    n_t = (nx + nu)*t
    return lambda x : x[n_t_1 + 2] - dynamics(x[n_t:n_t + nx], x[n_t+nx:n_t+nx+nu], cop, dt)[2]

def ce4(nx, nu, t, dt, cop, dynamics):
    n_t_1 = (nx + nu)*(t + 1)
    n_t = (nx + nu)*t
    return lambda x : x[n_t_1 + 3] - dynamics(x[n_t:n_t + nx], x[n_t+nx:n_t+nx+nu], cop, dt)[3]

def ce5(nx, nu, t, dt, cop, dynamics):
    n_t_1 = (nx + nu)*(t + 1)
    n_t = (nx + nu)*t
    return lambda x : x[n_t_1 + 4] - dynamics(x[n_t:n_t + nx], x[n_t+nx:n_t+nx+nu], cop, dt)[4]

def ce6(nx, nu, t, dt, cop, dynamics):
    n_t_1 = (nx + nu)*(t + 1)
    n_t = (nx + nu)*t
    return lambda x : x[n_t_1 + 5] - dynamics(x[n_t:n_t + nx], x[n_t+nx:n_t+nx+nu], cop, dt)[5]

# inequality constraints (bounds)

def ci1(nx, nu, t, u_lim):
    n_t = (nx + nu)*t
    return lambda x : u_lim - x[n_t + nx]

def ci2(nx, nu, t):
    n_t = (nx + nu)*t
    return lambda x : x[n_t + nx]

horizon = 0.4
dt = 0.1
nx = 6
nu = 1
n =  int(nx + (horizon/dt)*(nx) + (horizon/dt)*(nu))
print(n)
x0 = torch.tensor([[0], [0], [0.2], [1.0], [0], [0]], dtype=float)
cop = torch.tensor([[0.1], [0], [0]],  dtype=float)
cop_next = torch.tensor([[0.2], [0], [0.1]], dtype=float)
u_lim = 25.0 # max acceleration in the z direction
h_nom = 0.2 # prefered height above the ground after each step

ce = []
ci = []
ce.append(ce11(x0))
ce.append(ce12(x0))
ce.append(ce13(x0))
ce.append(ce14(x0))
ce.append(ce15(x0))
ce.append(ce16(x0))

for i in range(int(horizon/dt)):
    ce.append(ce1(nx, nu, i, dt, cop, dynamics))
    ce.append(ce2(nx, nu, i, dt, cop, dynamics))
    ce.append(ce3(nx, nu, i, dt, cop, dynamics))
    ce.append(ce4(nx, nu, i, dt, cop, dynamics))
    ce.append(ce5(nx, nu, i, dt, cop, dynamics))
    ce.append(ce6(nx, nu, i, dt, cop, dynamics))

    ci.append(ci1(nx, nu, i, u_lim))
    ci.append(ci2(nx, nu, i))
    # ci.append(ci3(nx, nu, i, cop, kin_lim))

def f(x):
    return (x[-4] - cop_next[2] - 0.2)**2 
    
x0 = torch.zeros(n, 1, dtype = float)

bcf = BoundConstrainedFormulation(maxit = 100)
x_opt = bcf.optimize(f, ce, ci, n, 0.001, 0.001, 20, use_sr1 = False)

fig, axs = plt.subplots(3,1)
axs[0].plot(x_opt[0::7], label = "x")
axs[0].plot(x_opt[1::7], label = "y")
axs[0].plot(x_opt[2::7], label = "z")
axs[0].legend()
axs[0].grid()

axs[1].plot(x_opt[3::7], label = "xd")
axs[1].plot(x_opt[4::7], label = "yd")
axs[1].plot(x_opt[5::7], label = "zd")
axs[1].legend()
axs[1].grid()

axs[2].plot(x_opt[6::7], label = "u")
axs[2].legend()
axs[2].grid()

plt.show()
