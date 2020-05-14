# This is a demo of the Quasi Newton algorithms
# Author : Avadesh Meduri
# Date : 17/04/2020

import torch
from py_solvers.unconstrained.quasi_newton import BFGS

bfgs = BFGS(1e-4, 0.9, 0.5)

# def f(x):
#     return torch.pow(2.0, x[0]*x[1]) + 2

def f(x):
    return x[0]*x[0]*x[1]*x[1]

x0 = torch.tensor([[1.0], [2.0]])
x_k, f_opt = bfgs.optimize(f, x0, 100, 0.001, use_btls = False, alpha_max = 2.0)
bfgs.stats()

