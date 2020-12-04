# The file contains an implementation of proximal gradient method
# This algorithm is based on the Algorithm described in First Order Methods 
# by Amir Beck. ch10
# Author : Avadesh Meduri
# Date : 4/12/2020

import os.path
import sys
from matplotlib import pyplot as plt 
curdir = os.path.dirname(__file__)
cdir = os.path.abspath(os.path.join(curdir,'../../python/'))
sys.path.append(cdir)

import matplotlib.pyplot as plt
import torch

from py_autodiff.torch_autodiff import TorchAutoDiff

class ProximalGradient:
    
    def __init__(self, L0, gamma, beta, tol = 0.00001):
        '''
        This is an implementation of the proximal gradient method
        Input:
            L0 : initial step length (ideally greater than L/2 where L is lipschitz constant of f)
            gamma : suffecient decrease condition param
            beta : factor by which step length is increased
            tol : value check for equality constraint
        '''
        
        self.tad = TorchAutoDiff()
        self.tol = tol
        assert L0 > 0
        self.L0 = L0
        assert gamma < 1 and gamma > 0
        self.gamma = gamma 
        assert beta > 1
        self.beta = beta
        
        self.f_all = []
        
    def compute_step_length(self, x_k, f, g):
        '''
        computes step length using back tracking line search
        Input :
            x_k : current x
            f : objecive function
            g : objective function 2 (prox function)
        '''
        grad_k = self.tad.compute_gradient(f, x_k)
        L = self.L0
        while True:
            x_k_1 = g(x_k - grad_k/L, L)
            G_k_norm = torch.norm(L * (x_k - x_k_1)) # proximal gradient
            if f(x_k) - f(x_k_1) < (self.gamma/L)*G_k_norm**2:
                L = self.beta*L
            else:
                break
        return x_k_1, G_k_norm
            
    def optimize(self, f, g, x0, maxit, tol, is_convex = False):
        '''
        This function optimizes the given objective and returns optimal x and f(x) + g(x)
        it minimies the objective f(x) + g(x)
        Input:
            f : objective function to be minimized
            g : proximal operator
            x0 : starting point
            maxit : maximum number of iterations
            tol : tolerance of gradient to exit 
            is_convex : true if the objective function is convex
        '''
        x_k = x0
        for k in range(maxit):
            x_k_1, G_k_norm = self.compute_step_length(x_k, f, g)
            self.f_all.append(f(x_k_1))
            if G_k_norm < tol:
                break
            x_k = x_k_1
        
        return x_k
    
    def stats(self):
        '''
        This function returns stats and plots
        '''
        print("The algorithm has terminated after : " + str(len(self.f_all)) + " iterations")
        print("The optimal value of the objective funtion is : " + str(self.f_all[-1]))
        fig, axs = plt.subplots(1, 1, sharex=True)
        axs.plot(self.f_all, label="objective value")
        axs.set_ylabel("Value of objective function")
        axs.legend()
        axs.grid()
        plt.xlabel("iteration")
        plt.show()
        