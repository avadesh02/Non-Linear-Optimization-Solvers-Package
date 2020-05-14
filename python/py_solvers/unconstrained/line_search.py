# This file contains implementation of line search methods based on wolfe conditions
# Author : Avadesh Meduri
# Date : 14/04/2020

import torch
from py_autodiff.torch_autodiff import TorchAutoDiff

class LineSearch:

    def __init__(self, c1, c2, beta = None, tol = 0.0001):
        '''
        This the steepest descent algorithm.
        Input:
            c1 : parameter to check inequality
            c2 : paramter to ensure sufficient reduction in gradient
            beta : parameter by which alpha is reduced
            tol : tolerance to check equality conditions (Read details below)
        '''
        assert c1 < 0.5 # this ensures superlinear convergence (Thr - 3.6 Nocedal)
        assert c2 > c1 # Wolfe condition
        self.c1 = c1
        self.c2 = c2
        self.beta = beta
        self.tad = TorchAutoDiff()
        self.alpha0 = 1.0
        # this value checks for equality conditions. Direct == does not work due to machine 
        # precission issues for ex. 1e-38 and 1e-39 are not equal to the machine but for practical
        # pursposes it is equal to zero
        self.tol = tol

    def btls(self, fun, x_k, p_k, grad_k):
        '''
        This is the back tracking line search algorithm (Nocedal Alg - 3.1)
        Input:
            x_k : the value of x at which step is to be taken
            alpha0 : initial value of step size
            p_k : descent direction
            fun : function to be minimized
            grad_k : gradient of f at x_k
        '''

        alpha = self.alpha0
        while fun(x_k + alpha*p_k) > fun(x_k) + self.c1*alpha*torch.t(grad_k).matmul(p_k):
            alpha = self.beta*alpha
        # Computes alpha0 for next iteration (3.60 Nocedal)
        self.alpha0 = 2*(fun(x_k + alpha*p_k) - fun(x_k))/torch.t(grad_k).matmul(p_k)
        self.alpha0 = max(1, 1.01*self.alpha0)

        return alpha

    def step_zoom(self, fun, x_k, p_k, grad_k, alpha_l, alpha_h):
        '''
        This is an implementation find step length based on alg - 3.6 Noecdal
        This satisfies strong wolfe conditions
        Input:
            fun : function to be mininimized
            x_k : current x
            p_k : descent direction
            grad_k : gradient at x_k
            alpha_l : lower bound on step length
            alhpa_h : upper bound on step length
        '''
        alpha_low = alpha_l
        alpha_high = alpha_h
        while True:
            alpha_j = 0.5*(alpha_low + alpha_high)
            if fun(x_k + alpha_j*p_k) > fun(x_k) + self.c1*alpha_j*torch.t(grad_k).matmul(p_k) \
                or fun(x_k + alpha_j*p_k) > fun(x_k + alpha_low*p_k):
                alpha_high = alpha_j
            else:
                grad_j = self.tad.compute_gradient(fun, x_k + alpha_j*p_k)
                if torch.abs(torch.t(grad_j).matmul(p_k)) < -self.c2*torch.t(grad_k).matmul(p_k) or \
                torch.norm(torch.abs(torch.t(grad_j).matmul(p_k)) + self.c2*torch.t(grad_k).matmul(p_k)) < self.tol:
                    alpha_opt = alpha_j
                    break
                elif torch.t(grad_j).matmul(p_k)*(alpha_high - alpha_low) > 0 or \
                    torch.norm(torch.t(grad_j).matmul(p_k)*(alpha_high - alpha_low)) < self.tol:
                    alpha_high = alpha_low
                if alpha_high - alpha_j < self.tol and alpha_high - alpha_low < self.tol:
                    # This inequality checks if alpha_high and alpha_low and alpha_j have converged, 
                    # in which case there is no point in running any more iterations
                    print("Warning: Strong Wolfe Condition - 2 not satisfied. Line search terminated as alpha's have converged")
                    alpha_opt = alpha_j
                    break
                alpha_low = alpha_j
        return alpha_opt
            
    def strong_line_search(self, fun, x_k, p_k, grad_k, alpha_max):
        '''
        This line search satisfies strong wolfe conditions - Alg 3.5 Nocedal
        It is preferred to use this for Conjugate Gradient and Quasi Newton methods.
        '''
        alpha_old = 0
        alpha = 0.001*alpha_max
        while True:
            if fun(x_k + alpha*p_k) > fun(x_k) + self.c1*alpha*torch.t(grad_k).matmul(p_k):
                alpha_opt = self.step_zoom(fun, x_k, p_k, grad_k, alpha_old, alpha)
                break
            elif fun(x_k + alpha*p_k) > fun(x_k + alpha_old*p_k):
                alpha_opt = self.step_zoom(fun, x_k, p_k, grad_k, alpha_old, alpha)
                break
            else:
                grad_i = self.tad.compute_gradient(fun, x_k + alpha*p_k)
                if torch.abs(torch.t(grad_i).matmul(p_k)) < -self.c2*torch.t(grad_k).matmul(p_k) or\
                torch.norm(torch.abs(torch.t(grad_i).matmul(p_k)) + self.c2*torch.t(grad_k).matmul(p_k)) < self.tol:
                    alpha_opt = alpha
                    break
                elif torch.t(grad_i).matmul(p_k) > 0 or torch.norm(torch.t(grad_i).matmul(p_k)) < self.tol:
                    alpha_opt = self.step_zoom(fun, x_k, p_k, grad_k, alpha, alpha_old)
                    break
                elif alpha - alpha_old < self.tol:
                    # This inequality checks if alpha_high and alpha_low and alpha_j have converged, 
                    # in which case there is no point in running any more iterations
                    print("Warning: Strong Wolfe Condition - 2 not satisfied. Line search terminated as alpha's have converged")
                    alpha_opt = alpha
                    break
                alpha_old = alpha
                alpha = min(1.5*alpha, alpha_max)
        return alpha_opt
            
    def quad_interpolate(self, fun, x_k, p_k, grad_k, alpha):
        '''
        This function finds the step lenght that minimizes the quadratic approximation
        Input:
            fun : function to be minimized
            x_k : current x
            p_k : descent direction
            grad_k : gradient at current x
            alpha : step length
        '''

        numerator = -(torch.t(grad_k).matmul(p_k)*alpha*alpha)
        denominator = 2*(fun(x_k + alpha*p_k) - fun(x_k) - alpha*torch.t(grad_k).matmul(p_k)) 

        return numerator / denominator

    
        