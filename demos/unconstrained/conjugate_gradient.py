# This is a demo of the conjugate gradient methods
# Author : Avadesh Meduri
# Date : 16/04/2020

import torch
from py_solvers.unconstrained.conjugate_gradient import FletcherReeves, PolakRibiere

def f(x):
    return x[0]*x[0]*x[1]*x[1] + (x[0] + x[1] + 2)**2

fr = FletcherReeves(use_pure_fr = False)
x0 = torch.tensor([[-5.0], [2.0]])
x_opt, f_opt = fr.optimize(f, x0, 100, 0.001, 10.0, use_btls = False)
print(x_opt)
fr.stats()

pr = PolakRibiere(use_pure_pr = False)
x0 = torch.tensor([[-5.0], [2.0]])
x_opt, f_opt = pr.optimize(f, x0, 100, 0.001, 10.0, use_btls = False)
print(x_opt)
pr.stats()
