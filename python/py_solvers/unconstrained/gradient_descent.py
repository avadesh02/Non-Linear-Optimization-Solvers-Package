# This file contains implementation of gradient descent method
# Author : Avadesh Meduri
# Date : 15/04/2020

import torch
import matplotlib.pyplot as plt

from py_solvers.unconstrained.line_search import LineSearch
from py_autodiff.torch_autodiff import TorchAutoDiff


class SteepestDescent:

    def __init__(self, c1, c2, beta, tol = 0.00001):
        '''
        Implementation of the steepest descent algorithm
        Input:
            c1 : parameter used for line search wolfe condition 1
            c2 : parameter used for line search wolfe condition 2
            beta : used in back tracking line search
            tol : value to check equality condition (refer to line search for details)
        '''
        self.line_search = LineSearch(c1, c2, beta, tol)
        self.tad = TorchAutoDiff()
        self.f_all = [] ## list of values of the objective 

    def optimize(self, fun, x0, maxit, tol, use_btls = False, alpha_max = 1.0):
        '''
        This function optimizes a given function and returns minima
        Input:
            fun : objective function to be minimized
            x0 : starting point 
            maxit : maxmimum number of iterations
            tol : tolerance of gradient to exit
            use_btls : uses back tracking line search to compute step length
            alpha_max : refer to btls function
        '''
        x_k = x0
        for k in range(maxit):
            self.f_all.append(fun(x_k).detach().numpy())
            grad_k = self.tad.compute_gradient(fun, x_k)

            if use_btls:
                alpha_k = self.line_search.btls(fun, x_k, -grad_k, grad_k)
            else:
                alpha_k = self.line_search.strong_line_search(fun, x_k, -grad_k, grad_k, alpha_max)
            x_k = x_k - alpha_k*grad_k
            if torch.norm(grad_k) < tol:
                self.f_all.append(fun(x_k).detach().numpy())
                break
        
        self.alpha0 = 1.0

        return x_k, fun(x_k)

    def stats(self):
        '''
        This function returns stats and plots
        '''
        print("The algorithm has terminated after : " + str(len(self.f_all)) + " iterations")
        print("The optimal value of the objective funtion is : " + str(self.f_all[-1]))
        plt.plot(self.f_all)
        plt.grid()
        plt.title("Value of the objective function vs iterations")
        plt.ylabel("Value of objective function")
        plt.xlabel("iteration")
        plt.show()
