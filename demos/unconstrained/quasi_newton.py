# This is a demo of the Quasi Newton algorithms
# Author : Avadesh Meduri
# Date : 17/04/2020

import torch
from py_solvers.unconstrained.quasi_newton import BFGS


def f(x):
    return x[0]*x[0]*x[1]*x[1] + (x[0] + x[1] + 2)**2


bfgs = BFGS(1e-4, 0.9, 0.5)

x0 = torch.tensor([[-5.0], [2.0]])
x_k, f_opt = bfgs.optimize(f, x0, 100, 0.001, use_btls = False, alpha_max = 2.0)
print(x_k)
bfgs.stats()


