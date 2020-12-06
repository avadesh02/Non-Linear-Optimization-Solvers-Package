# The file is an implementation of Fast Iterative Shrinkage Thresholding
# This algorithm is based on the Algorithm described in First Order Methods 
# by Amir Beck. ch10
# Author : Avadesh Meduri
# Date : 5/12/2020

import os.path
import sys
from matplotlib import pyplot as plt 
curdir = os.path.dirname(__file__)
cdir = os.path.abspath(os.path.join(curdir,'../../python/'))
sys.path.append(cdir)

import matplotlib.pyplot as plt
import torch

from py_autodiff.torch_autodiff import TorchAutoDiff

class FISTA:

    def __init__(self, L0, beta, tol = 0.00001):
        '''
        This is an implementation of Fast Iterative Shrinkage thresholding
        Algorithm
        Input:
            L0 : initial step length (ideally greater than L/2 where L is lipschitz constant of f)
            beta : factor by which step length is increased
            tol : value check for equality constraint
        '''

        self.tad = TorchAutoDiff()
        self.tol = tol
        assert L0 > 0
        self.L0 = L0
        assert beta > 1
        self.beta = beta
        
        self.f_all = []
        self.g_all = []

    def compute_step_length(self, y_k, f, g):
        '''
        computes step length using back tracking line search
        Note : the function f and g must be convex to get accelerated convervgence
        Input:
            y_k : current y
            f : objecive function
            g : objective function 2 (prox function)
        '''
        grad_k = self.tad.compute_gradient(f, y_k)
        L = self.L0
        while True:
            y_k_1 = g(y_k - grad_k/L, L)
            G_k_norm = torch.norm((y_k_1 - y_k)) # proximal gradient
            if f(y_k_1) > f(y_k) + torch.matmul(torch.transpose(grad_k, 0, 1), y_k_1 - y_k) + (L/2)*G_k_norm**2:
                L = self.beta*L
            else:
                break
        self.L0  = L

        return y_k_1, G_k_norm

    def optimize(self, f, g, x0, maxit, tol):
        '''
        This function optimizes the given objective and returns optimal x and f(x) + g(x)
        it minimies the objective f(x) + g(x)
        Input:
            f : objective function to be minimized
            g : proximal operator
            x0 : starting point
            maxit : maximum number of iterations
            tol : tolerance of gradient to exit 
        '''

        x_k = x0
        y_k = x_k
        t_k = torch.tensor([1.0])
        for k  in range(maxit):
            x_k_1, G_k_norm = self.compute_step_length(y_k, f, g)
            t_k_1 = torch.tensor([1.0]) + torch.sqrt(1 + 4*(t_k**2))/2.0
            y_k_1 = x_k_1 + ((t_k - 1)/t_k_1)*(x_k_1 - x_k)
            self.f_all.append(f(x_k_1))
            self.g_all.append(G_k_norm)
            # print("finished iteration {} and the cost is {}".format(k, float(self.f_all[-1])), end='\r')

            if G_k_norm < tol:
                break
            
            x_k = x_k_1
            y_k = y_k_1
            t_k = t_k_1   

        return x_k_1

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