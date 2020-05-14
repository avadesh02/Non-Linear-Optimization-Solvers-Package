# This is a demo of the gradient descent algorithms
# Author : Avadesh Meduri
# Date : 14/04/2020

import torch
from py_solvers.unconstrained.gradient_descent import SteepestDescent

gd = SteepestDescent(1e-4, 0.9, 0.5)

def f(x):
    return x[0]*x[0]*x[1]*x[1] + (x[0] + x[1] + 2)**2

x0 = torch.tensor([[-5.0], [2.0]])
x_k, f_opt = gd.optimize(f, x0, 100, 0.001, False, alpha_max = 1.0)
print(x_k)
gd.stats()

