# This is a demo to generate a trajectory for a unit mass block using 
# first order methods (FISTA, MFISTA, Prox grad)
# Author : Avadesh Meduri
# Date : 4/12/2020

import os.path
import sys
from matplotlib import pyplot as plt 
import torch
from torch import matmul as mm
from torch import transpose as tp
curdir = os.path.dirname(__file__)
cdir = os.path.abspath(os.path.join(curdir,'../../python/'))
sys.path.append(cdir)

from py_solvers.first_order.proximal_gradient import ProximalGradient
from py_solvers.first_order.fista import FISTA 


horizon = 1.5
dt = 0.1
nx = 2
nu = 1
T = int(horizon/dt)
n =  int(nx + (horizon/dt)*(nx) + (horizon/dt)*(nu))
x0 = torch.tensor([[2], [0]], dtype=float)
xdes = torch.tensor([[5], [0]], dtype=float)
w_t = torch.tensor([1e-3, 1e-3, 1e-3]) # running cost
w_T = torch.tensor([1e+3, 1e+3]) #terminal tracking cost


rho = 5e+3
l = -1e+3*torch.ones((n, 1))
u = 1e+3*torch.ones((n, 1))
l[2::3] = -3.0
u[2::3] = 3.0

# dynamics constrataint matrix 
dyn_a = torch.tensor([[1.0, dt, 0, -1.0, 0],[0, 1.0, dt, 0, -1.0]])

A = torch.zeros((2*T + 2, n))
b = torch.zeros((2*T + 2, 1))

for i in range(T):
    A[2*i:2*i+2, 3*i:3*i+5] = dyn_a
A[-2,0] = 1.0
A[-1,1] = 1.0
b[-2:] = x0

# cost matrix 
Q = torch.zeros((n,n))
q = torch.zeros((n))

for i in range(T):
    Q[3*i,3*i] = w_t[0]
    Q[3*i+1,3*i+1] = w_t[1]
    Q[3*i+2,3*i+2] = w_t[2] 

Q[-2,-2] = w_T[0]
Q[-1,-1] = w_T[1]
q[-2] = -2*w_T[0]*xdes[0]
q[-1] = -2*w_T[1]*xdes[1]


def dyn_const_viol(x, A, b):
    return torch.norm((mm(A,x) - b))**2

# penalty method cost (dynamic constraints are in the cost)
def cost(x, Q, q, A, b, rho):
    return mm(mm(torch.transpose(x, 0, 1),Q), x) + mm(q, x) + rho*torch.norm((mm(A,x) - b))**2

def const(x, u, l, L):
    #L is step length
    return torch.min(torch.max(x,l), u)

# dynamic constraint are enforced using projections
def dyn_cost(x, Q, q):
    return mm(mm(torch.transpose(x, 0, 1),Q), x) + mm(q, x)

def const_proj(x, A, b, L):
    # projects on to dynamic constraint
    return x - mm(mm(tp(A, 0, 1), torch.inverse(mm(A, tp(A, 0, 1)))), (mm(A, x) - b))

f = lambda x : cost(x, Q, q, A, b, rho)
g = lambda x, L : const(x, u, l, L)

f_dyn = lambda x : dyn_cost(x, Q, q)
g_dyn = lambda x, L : const(const_proj(x, A, b, L), u, l, L)

x_init = 0.0*torch.ones((n,1))

# prx grad
# prx_grad = ProximalGradient(0.01, 0.8, 1.1)
# x_opt = prx_grad.optimize(f, g, x_init, 3000, 0.001, is_convex = True)
# prx_grad.stats()
# print(dyn_const_viol(x_opt, A, b))

# Fista
fista = FISTA(0.01, 1.1)
x_opt = fista.optimize(f_dyn, g_dyn, x_init, 3000, 0.001)
fista.stats()
print(dyn_const_viol(x_opt, A, b))

fig, axs = plt.subplots(2,1)
axs[0].plot(x_opt[0::3], label = "x")
axs[0].plot(x_opt[1::3], label = 'xd')
axs[0].legend()
axs[0].grid()

axs[1].plot(x_opt[2::3], label = "u")
axs[1].legend()
axs[1].grid()

plt.show()