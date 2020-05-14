# This file contains the implementation of conjugate gradient methods for
# both linear and non linear problems (Nocedal ch 5)
# Author : Avadesh Meduri
# Date : 16/04/2020

import matplotlib.pyplot as plt
import torch

from py_autodiff.torch_autodiff import TorchAutoDiff
from py_solvers.unconstrained.line_search import LineSearch


class FletcherReeves:

    def __init__(self, v = 0.1, c1 = 1e-4, c2 = 0.4, beta = 0.5, tol = 0.00001, use_pure_fr = False):
        '''
        This is an implementation of the fletcher reeves algorithm Al - 5.4 Nocedal
        Input:
            v : threshold test of orthogonality for restarting
            c1 : constant for Wolfe first constraint
            c2 : constant for Wolfe second constraint
            beta : reduction factor of alpha in btls
            tol : value to check equality condition (refer to line search for details)
            use_pure_fr : uses FR residual to compute next step. (Not the best option)
        Note : 0 < c1 < c2 < 0.5 ensures convergence (Lemma 5.6 Nocedal)
        '''
        assert c2 < 0.5
        assert c1 < c2
        self.v = v
        self.c1 = c1
        self.c2 = c2
        self.tad = TorchAutoDiff()
        self.line_search = LineSearch(c1, c2, beta, tol)
        self.use_pure_fr = use_pure_fr 
        self.f_all = [] # list of values of the objective values

    def compute_fr_residual(self, grad_k_1, grad_k):
        '''
        This computes the equivivalent residual for the non linear problem
        Input:
            grad_k_1 : gradient at x_k+1
            grad_k : gradient at x_k
        '''
        # checking for restart
        if torch.abs(torch.t(grad_k_1).matmul(grad_k))/(torch.norm(grad_k)*torch.norm(grad_k)) < self.v:
            numerator = torch.t(grad_k_1).matmul(grad_k_1)
            denominator = torch.t(grad_k).matmul(grad_k)
            return numerator / denominator
        else:
            return torch.tensor([0.0])

    def compute_residual(self, grad_k_1, grad_k, p_k):
        '''
        This computes equivivalent residual for the non linear problem (5.49 Nocedal)
        This has better convergence properties in practice
        Input:
            grad_k_1 : gradient at x_k+1
            grad_k : gradient at x_k
            p_k : descent direction at iteration k
        '''
         # checking for restart
        if torch.abs(torch.t(grad_k_1).matmul(grad_k))/(torch.norm(grad_k)*torch.norm(grad_k)) < self.v:
            numerator = torch.t(grad_k_1).matmul(grad_k_1)
            denominator = torch.t(grad_k_1 - grad_k).matmul(p_k)
            return numerator / denominator
        else:
            return torch.tensor([0.0])

    def optimize(self, fun, x0, maxit, tol, alpha_max, use_btls = False):
        '''
        This function optimizes a given function and returns optimal x and f(x)
        Input:
            fun : objective function to be minimized
            x0 : starting point
            maxit : maximum number of iterations
            tol : tolerance of gradient to exit 
            alpha_max : max step length
            use_btls : use back tracking line search to decide step length
        Note: It is prefered to use strong line search over btls as it garuntees 
            convergence for fletcher reeves (Lemma 5.6). It garuntees that the new
            p_k is a descent direction. This is not true for btls
        '''

        x_k = x0
        grad_k = self.tad.compute_gradient(fun, x_k)
        p_k = -grad_k
        for k in range(maxit):
            self.f_all.append(fun(x_k).detach().numpy())
            
            if use_btls:
                alpha_k = self.line_search.btls(fun, x_k, -grad_k, grad_k)
            else:
                alpha_k = self.line_search.strong_line_search(fun, x_k, p_k, grad_k, alpha_max)
                
            x_k_1 = x_k + alpha_k*p_k
            grad_k_1 = self.tad.compute_gradient(fun, x_k_1)
            if self.use_pure_fr:
                beta_k_1 = self.compute_fr_residual(grad_k_1, grad_k)
            else:
                beta_k_1 = self.compute_residual(grad_k_1, grad_k, p_k)
            p_k_1 = -grad_k_1 + beta_k_1*p_k

            x_k = x_k_1
            p_k = p_k_1
            grad_k = grad_k_1
            
            if torch.norm(grad_k) < tol:
                self.f_all.append(fun(x_k).detach().numpy())
                break

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

class PolakRibiere:

    def __init__(self, v = 0.1, c1 = 1e-4, c2 = 0.4, beta = 0.5, tol = 0.00001, use_pure_pr = False):
        '''
        This is an implementation of the Polak Ribiere algorithm Al - 5.4 Nocedal with
        pr and pr+ step
        Input:
            v : threshold test of orthogonality for restarting
            c1 : constant for Wolfe first constraint
            c2 : constant for Wolfe second constraint
            beta : reduction factor of alpha in btls
            tol : value to check equality condition (refer to line search for details)
            use_pure_pr : uses PR residual to compute next step. (Not the best option)
            Note : Pure pr may not converge even when strong wolfe condition is satisfied
        Note : 0 < c1 < c2 < 0.5 ensures convergence (Lemma 5.6 Nocedal) 
        '''
        assert c2 < 0.5
        assert c1 < c2
        self.v = v
        self.c1 = c1
        self.c2 = c2
        self.tad = TorchAutoDiff()
        self.line_search = LineSearch(c1, c2, beta, tol)
        self.use_pure_pr = use_pure_pr 
        self.f_all = [] # list of values of the objective values

    def compute_pr_residual(self, grad_k_1, grad_k):
        '''
        This computes the equivivalent residual for the non linear problem
        computes pr residual (5.44 Nocedal). This method has self correcting properties
        that fr does not have. However, global convergence is not garunteed.  
        Input:
            grad_k_1 : gradient at x_k+1
            grad_k : gradient at x_k
        '''
        # checking for restart
        numerator = torch.t(grad_k_1).matmul(grad_k_1 - grad_k)
        denominator = torch.t(grad_k).matmul(grad_k)
        return numerator / denominator

    def compute_pr_plus_residual(self, grad_k_1, grad_k):
        '''
        This computes the equivivalent residual for the non linear problem
        Computes Pr+ residual (5.45 Nocedal). It is suggested to use pr+ as 
        this has global convergence properties.
        Input:
            grad_k_1 : gradient at x_k+1
            grad_k : gradient at x_k
        '''
        beta_k_1 = self.compute_pr_residual(grad_k_1, grad_k)
        if beta_k_1 < 0:
            beta_k_1 = torch.tensor([0.0])
        return beta_k_1

    
    def optimize(self, fun, x0, maxit, tol, alpha_max, use_btls = False):
        '''
        This function optimizes a given function and returns optimal x and f(x)
        Input:
            fun : objective function to be minimized
            x0 : starting point
            maxit : maximum number of iterations
            tol : tolerance of gradient to exit 
            alpha_max : max step length
            use_btls : use back tracking line search to decide step length
        Note: It is prefered to use strong line search over btls as it garuntees 
            convergence for fletcher reeves (Lemma 5.6). It garuntees that the new
            p_k is a descent direction. This is not true for btls
        '''

        x_k = x0
        grad_k = self.tad.compute_gradient(fun, x_k)
        p_k = -grad_k
        for k in range(maxit):
            self.f_all.append(fun(x_k).detach().numpy())

            if use_btls:
                alpha_k = self.line_search.btls(fun, x_k, -grad_k, grad_k)
            else:
                alpha_k = self.line_search.strong_line_search(fun, x_k, p_k, grad_k, alpha_max)

            x_k_1 = x_k + alpha_k*p_k
            grad_k_1 = self.tad.compute_gradient(fun, x_k_1)
            if self.use_pure_pr:
                beta_k_1 = self.compute_pr_residual(grad_k_1, grad_k)
            else:
                beta_k_1 = self.compute_pr_plus_residual(grad_k_1, grad_k)
            p_k_1 = -grad_k_1 + beta_k_1*p_k

            x_k = x_k_1
            p_k = p_k_1
            grad_k = grad_k_1
            
            if torch.norm(grad_k) < tol:
                self.f_all.append(fun(x_k).detach().numpy())
                break

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



