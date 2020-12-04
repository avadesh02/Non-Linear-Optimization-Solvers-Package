# This is a demo to generate a trajectory for a unit mass block using 
# first order methods
# Author : Avadesh Meduri
# Date : 4/12/2020

import os.path
import sys
from matplotlib import pyplot as plt 
import torch
from torch import matmul as mm
curdir = os.path.dirname(__file__)
cdir = os.path.abspath(os.path.join(curdir,'../../python/'))
sys.path.append(cdir)

from py_solvers.first_order.proximal_gradient import ProximalGradient


horizon = 1.0
dt = 0.1
nx = 2
nu = 1
T = int(horizon/dt)
n =  int(nx + (horizon/dt)*(nx) + (horizon/dt)*(nu))
x0 = torch.tensor([[0], [0]], dtype=float)
xdes = torch.tensor([[5], [0]], dtype=float)
w_t = torch.tensor([1e-5, 1e-5, 1e-5]) # running cost
w_T = torch.tensor([1e+3, 10.0]) #terminal tracking cost

rho = 1e+2
l = torch.ones((n, 1))
u = torch.ones((n, 1))

# dynamics constrataint matrix 
dyn_a = torch.tensor([[1.0, dt, 0, -1.0, 0],[0, 1, dt, 0, -1]])

A = torch.zeros((2*T + 2, n))
b = torch.zeros((2*T + 2, 1))

for i in range(T):
    A[2*i:2*i+2, 3*i:3*i+5] = dyn_a
A[-2,0] = 1.0
A[-1,1] = 1.0
b[-2:] = x0

# cost matrix 
Q = torch.zeros((n,n))
q = torch.zeros((1,n))

for i in range(T):
    Q[3*i,3*i] = w_t[0]
    Q[3*i+1,3*i+1] = w_t[1]
    Q[3*i+2,3*i+2] = w_t[2] 

Q[-2,-2] = w_T[0]
Q[-1,-1] = w_T[1]
q[0,-2] = -2*xdes[0]
q[0,-1] = -2*xdes[1]

def cost(x, Q, q, A, b, rho):
    return mm(mm(torch.transpose(x, 0, 1),Q), x) + 2*mm(q, x) + rho*torch.norm((mm(A,x) - b))**2

def const(x, u, l, L):
    #L is step length
    # return torch.min(torch.max(x,l), u)
    return x

f = lambda x : cost(x, Q, q, A, b, rho)
g = lambda x, L : const(x, u, l, L)
x_init = torch.ones((n,1))


prx_grad = ProximalGradient(0.01, 0.9, 1.1)
x_opt = prx_grad.optimize(f, g, x_init, 10, 0.001)
# print(x_opt)

fig, axs = plt.subplots(2,1)
axs[0].plot(x_opt[0::3], label = "x")
axs[0].plot(x_opt[1::3], label = 'xd')
axs[0].legend()
axs[0].grid()

axs[1].plot(x_opt[2::3], label = "u")
axs[1].legend()
axs[1].grid()

plt.show()