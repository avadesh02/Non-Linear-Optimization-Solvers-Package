## This file contains implementation of quasi newton methods. BFGS and DPS. 
## Author : Avadesh Meduri
## Date : 17/04/2020

import torch
import matplotlib.pyplot as plt

from py_solvers.unconstrained.line_search import LineSearch
from py_autodiff.torch_autodiff import TorchAutoDiff

class BFGS:

    def __init__(self, c1, c2, beta = None, tol = 0.0001):
        '''
        This is the implementation of the BFGS algorithm. 
        (Alg - 6.1 Nocedal). 
        Input:
            c1 : parameter to check inequality
            c2 : paramter to ensure sufficient reduction in gradient
            beta : parameter by which alpha is reduced
            tol : tolerance to check equality conditions (check line search for details)
        '''
        assert c2 < 1
        assert c1 < c2

        self.line_search = LineSearch(c1, c2, beta, tol)
        self.tad = TorchAutoDiff()
        self.f_all = [] ## list of values of the objective 

    def compute_inverse_hessian(self, x_k, x_k_1, grad_k, grad_k_1, H_k):
        '''
        This function computes the H matrix (inverse of hessian) (6.17 Nocedal)
        Input:
            x_k : current x
            x_k_1 : x at after update
            grad_k : gradient at x
            grad_k_1 : gradient at x_k_1
            H_k : current inverse hessian approximation
        '''
        y_k = grad_k_1 - grad_k
        s_k = x_k_1 - x_k
        rho_k = 1.0/(torch.t(y_k).matmul(s_k))
        I = torch.eye(s_k.size()[0])
        H_k_1 = (I - rho_k*s_k.matmul(torch.t(y_k))).matmul(H_k).matmul((I - rho_k*s_k.matmul(torch.t(y_k)))) \
                    + rho_k*s_k.matmul(torch.t(s_k))

        return H_k_1

    def compute_hessian(self, x_k, x_k_1, grad_k, grad_k_1, B_k):
        '''
        This function computes the B matrix (Hessian approximation) (6.19 Nocedal)
        Input:
            x_k : current x
            x_k_1 : x at after update
            grad_k : gradient at x
            grad_k_1 : gradient at x_k_1
            H_k : current hessian approximation
        '''
        y_k = grad_k_1 - grad_k
        s_k = x_k_1 - x_k
        rho_k = 1.0/(torch.t(y_k).matmul(s_k))

        B_k_1 = B_k + (y_k.matmul(torch.t(y_k)))/(torch.t(y_k).matmul(s_k))
        B_k_1 -= (B_k.matmul(s_k).matmul(torch.t(s_k)).matmul(B_k))/ \
                        (torch.t(s_k).matmul(B_k).matmul(s_k))

        return B_k_1

    def optimize(self, fun, x0, maxit, tol, use_btls = False, alpha_max = 1.0):
        '''
        This function optimizes a given function and returns minima. It is prefered
        to use strong line search as it ensures that p_k is a descent direction
        Input:
            fun : objective function to be minimized
            x0 : starting point 
            maxit : maxmimum number of iterations
            tol : tolerance of gradient to exit
            use_btls : uses back tracking line search to compute step length
            alpha_max : refer to btls function
        '''
        x_k = x0
        H_k = torch.eye(x_k.size()[0])
        grad_k = self.tad.compute_gradient(fun, x_k)
        for k in range(maxit):
            self.f_all.append(fun(x_k).detach().numpy())
            p_k = -H_k.matmul(grad_k)

            if use_btls:
                alpha_k = self.line_search.btls(fun, x_k, p_k, grad_k)
            else:
                alpha_k = self.line_search.strong_line_search(fun, x_k, p_k, grad_k, alpha_max)
            x_k_1 = x_k + alpha_k*p_k
            grad_k_1 = self.tad.compute_gradient(fun, x_k_1)
            
            H_k = self.compute_inverse_hessian(x_k, x_k_1, grad_k, grad_k_1, H_k)

            x_k = x_k_1
            grad_k = grad_k_1

            if torch.norm(grad_k_1) < tol:
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

class SR1:

    def __init__(self, r, c1, c2, beta = None, tol = 0.0001):
        '''
        This is the implementation of the SR1 method. sec 6.2 Nocedal
        Line search is used to compute next step. To be Done : Trust region
        step selection.
        This method is advantageous in constrained optimization setting where
        the secant condition y_{k}^{T}s_{k} > 0 can not be satisfied. 
        Input:
            c1 : parameter to check inequality
            c2 : paramter to ensure sufficient reduction in gradient
            beta : parameter by which alpha is reduced
            tol : tolerance to check equality conditions (check line search for details)
        '''
        assert c2 < 1
        assert c1 < c2

        self.line_search = LineSearch(c1, c2, beta, tol)
        self.tad = TorchAutoDiff()
        self.f_all = [] ## list of values of the objective 

        self.r = r # This factor is used in the breakdown factor check (6.26 Nocedal)
        self.tol = tol

    def compute_inverse_hessian(self, x_k, x_k_1, grad_k, grad_k_1, H_k):
        '''
        This function computes the inverse hessian approximation using 
        SR1 update strategy. (6.24 Nocedal)
        Input:
            x_k : current x
            x_k_1 : x at after update
            grad_k : gradient at x
            grad_k_1 : gradient at x_k_1
            H_k : current inverse hessian approximation
        '''
        
        y_k = grad_k_1 - grad_k
        s_k = x_k_1 - x_k
        # This criteria prevents breakdown of SR1 method
        if torch.abs(torch.t(y_k).matmul(s_k - H_k.matmul(y_k))) > \
            self.r*torch.norm(y_k)*(s_k - H_k.matmul(y_k)) or \
            torch.abs(torch.t(y_k).matmul(s_k - H_k.matmul(y_k))) - \
            self.r*torch.norm(y_k)*(s_k - H_k.matmul(y_k)) < self.tol:

            H_k_1 = H_k
            H_k_1 += (s_k - H_k.matmul(y_k)).matmul(torch.t(s_k - H_k.matmul(y_k)))/\
                        torch.t(s_k - H_k.matmul(y_k)).matmul(y_k)
        else:
            H_k_1 = H_k
        return H_k_1
