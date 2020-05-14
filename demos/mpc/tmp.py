# This is a demo to generate a trajectory for a unit mass block
# Author : Avadesh Meduri
# Date : 29/04/2020

import torch
import matplotlib.pyplot as plt
from py_solvers.constrained.augmented_lagrangian.bound_constrained import BoundConstrainedFormulation

dt = 0.1
x0 = torch.tensor([[0], [1]], dtype = float)
xdes = torch.tensor([[5], [0]], dtype = float)
u = 10
def ce11(x):
    return x[0] - x0[0]

def ce12(x):
    return x[1] - x0[1]

def ce21(x):
    return x[3] - x[0] - x[1]*dt

def ce22(x):
    return x[4] - x[1] - x[2]*dt

def ce31(x):
    return x[6] - x[3] - x[4]*dt

def ce32(x):
    return x[7] - x[4] - x[5]*dt

def ci11(x):
    return x[2] + u

def ci12(x):
    return u - x[2]

def ci21(x):
    return x[5] + u 

def ci22(x):
    return u - x[5]

def f(x):
    return (x[6] - 5.0)**2 + (x[7] - 0)**2

ce = [ce11, ce12, ce21, ce22, ce31, ce32]
ci = [ci11, ci12, ci21, ci22]

x0 = torch.zeros(8, 1, dtype = float)
# x0[1::3] = 2
# x0[2::3] = 3
# x0[-2] = 0


bcf = BoundConstrainedFormulation(maxit = 100)
x_opt = bcf.optimize(f, ce, ci, 8, 0.001, 0.001, 10, x0 = x0, use_sr1 = False)

fig, axs = plt.subplots(2,1)
axs[0].plot(x_opt[0::3], label = "x")
axs[0].plot(x_opt[1::3], label = 'xd')
axs[0].legend()
axs[0].grid()

axs[1].plot(x_opt[2::3], label = "u")
axs[1].legend()
axs[1].grid()

plt.show()