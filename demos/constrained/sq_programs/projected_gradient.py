# This is a demo of the non linear gradient projection algorithm
# Author : Avadesh Meduri
# Date : 27/04/2020

import torch
from py_solvers.constrained.sq_programs.projected_gradient import NonLinearGradientProjection

def f(x):
    return x[0]*x[1]*x[2] + (x[0]*x[1] - 2)**2

l = torch.tensor([[-2.0], [-1.0], [-1.0]], dtype = float)
u = torch.tensor([[7.0], [5.0], [5.0]], dtype = float)

# For the above function warm starting with a good x0 is important
x0 = torch.tensor([[1.0], [2.0], [4.0]], dtype = float)

ngp = NonLinearGradientProjection()
x_opt, fun_opt = ngp.optimize(f, l, u, 100, 0.0001, x0 = x0, use_sr1 = False)
print("Optimal solution is:" + str(x_opt))
print("Optimal value of objecitve is:" + str(fun_opt))
ngp.stats()
