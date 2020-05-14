# This is a demo for the gradient projection algorithm based on 
# cauchy point
# Author : Avadesh Meduri
# Date : 21/04/2020

import numpy as np
import torch
from py_solvers.constrained.quadratic_programs.projected_gradient import GradientProjection

from cvxopt import solvers, matrix

G = torch.tensor([[6, 2, 1], [2, 5, 2], [1, 2, 4]], dtype=float)
c = torch.tensor([[10], [-5], [-4]], dtype=float)
l = torch.tensor([[-0.5], [-10.0], [0.0]], dtype= float)
u = torch.tensor([[0.5], [15.0], [0.0]], dtype=float)

gradproj = GradientProjection()
x_opt, f_opt = gradproj.optimize(G, c, l, u, 10)
print(x_opt.T)
print("primal objective value : " + str(f_opt.numpy()))
# gradproj.stats()

## setiing up problem in cvx to compare results
def form_gh_constr(l, u):
    i = np.eye((l.size()[0]))
    G = np.block([[ i],
                 [ -i]])
    h = np.block([u.numpy().T, -l.numpy().T])[0]
    return matrix(G), matrix(h)
    

Q = matrix(G.numpy())
p = matrix(c.numpy().T[0])
G , h = form_gh_constr(l, u)

solvers.options['show_progress'] = False
x_opt = solvers.qp(Q, p, A = None, b = None, G = G, h = h)
print("solution from cvx_opt: " + str(x_opt['x'].T))
print("primal objective: " + str(x_opt['primal objective']))

