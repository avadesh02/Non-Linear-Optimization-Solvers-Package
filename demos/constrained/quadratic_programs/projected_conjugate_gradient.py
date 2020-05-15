# This is the demo for Projected conjugate Gradient algorithm
# Author : Avadesh Meduri
# Date : 20/04/2020
import numpy as np
import torch
from py_solvers.constrained.quadratic_programs.conjugate_gradient import ProjectedConjugateGradient

from cvxopt import solvers, matrix

# These matrices are obtained from Nocedal Example 16.2
G = torch.tensor([[6, 2, 1], [2, 5, 2], [1, 2, 4]], dtype=float)
c = torch.tensor([[-8], [-3], [-3]], dtype=float)
A = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype = float)
b = torch.tensor([[3], [0]], dtype = float)

pcg = ProjectedConjugateGradient(tol = 0.00001)
x_opt, f_opt = pcg.optimize(G, c, A, b, 100)
print(x_opt.T, f_opt)

pcg.stats()

Q = matrix(G.numpy())
p = matrix(c.numpy().T[0])
A1 = matrix(A.numpy())
b1 = matrix(b.numpy().T[0])

x_opt = solvers.qp(Q, p, A = A1, b = b1, G = None, h = None)
# print(x_opt)
print("solution from cvx_opt: " + str(x_opt['x'].T))
print("primal objective: " + str(x_opt['primal objective']))


