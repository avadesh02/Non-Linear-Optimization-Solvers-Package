# This is a demo of the FISTA algorithm 
# Author : Avadesh Meduri
# Date : 5/12/2020

import os.path
import sys
from matplotlib import pyplot as plt 
import torch
curdir = os.path.dirname(__file__)
cdir = os.path.abspath(os.path.join(curdir,'../../python/'))
sys.path.append(cdir)

from py_solvers.first_order.fista import FISTA 

fista = FISTA(0.01, 1.1)

def f(x):
    return -x[0]*x[1]

def g_lu(x, u, l, L):
    #L is step length
    return torch.min(torch.max(x,l), u)

x0 = torch.tensor([[-5.0], [2.0]])
u = torch.tensor([[4.0], [3.0]])
l = torch.tensor([[2.0], [1.0]])

g = lambda x, L : g_lu(x, u, l, L)

x_opt = fista.optimize(f, g, x0, 10, 0.001)
print(x_opt)
fista.stats()
