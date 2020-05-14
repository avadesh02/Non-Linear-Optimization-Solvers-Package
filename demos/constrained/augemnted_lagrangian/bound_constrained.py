# This file contains a demo of the bound constrained formulation
# of the augment lagrangian method
# Author : Avadesh Meduri
# Date : 22/04/2020

import torch
from py_solvers.constrained.augmented_lagrangian.bound_constrained import BoundConstrainedFormulation

def f(x):
    return x[0]*x[1] + x[2]*x[1]

def ce_1(x):
    return x[0] + x[1]*x[2]- 1

def ce_2(x):
    return 2*x[0] + x[2] - 1

def ci_1(x):
    return x[0]

def ci_2(x):
    return x[1]

def ci_3(x):
    return x[2]

ce = [ce_1, ce_2]
ci = [ci_1, ci_2, ci_3]

l0 = torch.ones(len(ce) + len(ci), dtype= float)
u0 = 1.0

x0 = torch.tensor([[1.0], [2.0], [4.0]], dtype = float)

bcf = BoundConstrainedFormulation(maxit = 100)
x_opt = bcf.optimize(f, ce, ci, 3, 0.001, 0.001, 10, use_sr1 = False, x0 = x0)
print(x_opt)
# bcf.stats()