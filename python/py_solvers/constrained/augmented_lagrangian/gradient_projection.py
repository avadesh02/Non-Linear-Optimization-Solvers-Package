# This is an implementation of the trust region gradient projection
# algorithm for the bound constrained Lagrange problem
# Author: Avadesh Meduri
# Date : 22/04/2020

import torch
import matplotlib.pyplot as plt
from py_autodiff.torch_autodiff import TorchAutoDiff

from py_solvers.constrained.quadratic_programs.projected_gradient import GradientProjection


class AugGradientProjection:

    def __init__(self, delta0 = 0.5, delta_max = 5, eta = 0.01, beta = 0.01, r = 1e-8, tol = 0.0001):
        '''
        Input : 
            delta0 : starting trust region radius
            delta_max : maximum trust region radius
            eta : lower bound on reduction ratio
            beta : value to initialise hessian
            r : lower bound on denominator in sr1 update
            tol : tolerance to check equality
        '''
        self.delta0 = delta0
        self.delta_max = delta_max
        self.eta = eta
        self.beta = beta
        self.r = r
        self.tol = tol
        self.gradproj = GradientProjection()
        self.tad = TorchAutoDiff()
        self.f_all = []

    def compute_gradients(self, x_k, lag, aug_lag):
        '''
        This function computes the derivative of the lagrange
        function and augmented lagrange function
        Input:
            x_k : value at which gradient to be taken
            lag : lagrange function
            aug_lag : augmented lagrange function
        '''
        lag_grad_k = self.tad.compute_gradient(lag, x_k)
        aug_lag_grad_k = self.tad.compute_gradient(aug_lag, x_k)

        return lag_grad_k, aug_lag_grad_k

    def compute_gradient_matrix(self, x_k, bc_arr):
        '''
        This function computes the matrix of constraint gradients
        required to form the hessian of the QP
        Input:
            x_k : value at which gradient to be taken
            bc_arr : function of bounded constraints
        '''
        for i in range(len(bc_arr)):
            if i == 0:
                A = torch.t(self.tad.compute_gradient(bc_arr[i], x_k))
            else:
                A = torch.cat((A, torch.t(self.tad.compute_gradient(
                                    bc_arr[i], x_k))))
        return A

    def compute_trust_region_bounds(self, x_k, l, u, delta):
        '''
        This function computes the bounds of the linearized problem
        where gradient projection is done,
        Input:
            x_k : value at which linearization is done
            l : lower bound of original bcl problem
            u : upper bound of original bcl problem
            delta : trust region
        '''
        l_trust = torch.clamp(l - x_k, -delta, delta)
        u_trust = torch.clamp(u - x_k, -delta, delta)
        
        return l_trust, u_trust

    def compute_dampled_bfgs_update(self, x_k, x_k_old, grad_k, grad_k_old, B_old):
        '''
        This function computes the Dampled BFGS update eq. 18.16 Nocedal
        Input:
            x_k : value at which hessian is to be computed
            x_k : previous value of x
            grad_k ; gradient at x_k
            grad_k : gradient at x_k_old
            B_old : old hessian
        '''
        s_k = x_k - x_k_old
        y_k = grad_k - grad_k_old
        denominator = torch.t(s_k).matmul(B_old).matmul(s_k)

        if torch.t(s_k).matmul(y_k) > 0.2*denominator:
            theta_k = 1
        else:
            theta_k = 0.8*denominator/(denominator - torch.t(s_k).matmul(y_k))
        
        r_k = theta_k*y_k + (1 - theta_k)*B_old.matmul(s_k)

        B_k = -((B_old.matmul(s_k).matmul(torch.t(s_k)).matmul(B_old)))/(denominator)
        B_k += (r_k.matmul(torch.t(r_k)))/(torch.t(s_k).matmul(r_k))
        B_k += B_old

        return B_k

    def compute_sr1_update(self, x_k, x_k_old, grad_k, grad_k_old, B_old):
        '''
        This function computes the SR1 update eq. 6.24 Nocedal
        Input:
            x_k : value at which hessian is to be computed
            x_k : previous value of x
            grad_k ; gradient at x_k
            grad_k : gradient at x_k_old
            B_old : old hessian
        '''
        s_k = x_k - x_k_old
        y_k = grad_k - grad_k_old
        denominator = torch.t(y_k - B_old.matmul(s_k)).matmul(s_k)
        if torch.norm(denominator) > self.r*torch.norm(s_k)*torch.norm(y_k - B_old.matmul(s_k)):
            numerator = (y_k - B_old.matmul(s_k)).matmul(torch.t(y_k - B_old.matmul(s_k)))
            B_k = B_old + (numerator/denominator)
        else:
            B_k = B_old
        
        return B_k

    def compute_hessian(self, x_k, x_k_old, grad_k, grad_k_old, A, B_old, u_k, use_sr1):
        '''
        This function computes the hessian of the augmented lagrangian
        Eq. 17.54 Nocedal
        Input:
            x_k : value at which hessian is to be computed
            x_k : previous value of x
            grad_k ; gradient at x_k
            grad_k : gradient at x_k_old
            A : gradient matrix
            B_old : old hessian
            u_k : penalty term on the constraints
            use_sr1 : uses sr1 update to compute hessian of lagrangian
        '''
        if use_sr1:
            B_k = self.compute_sr1_update(x_k, x_k_old, grad_k, grad_k_old, B_old)
        else:
            B_k = self.compute_dampled_bfgs_update(x_k, x_k_old, grad_k, grad_k_old, B_old)
        G = B_k + u_k*torch.t(A).matmul(A)
        return G

    def project_vector(self, x_k, l, u):
        '''
        This function projects a vector into the feasible space. eq 17.52 Nocedal
        Input : 
            x_k : current x
            l : lower bound constraint
            u : upper bound constraint
        '''
        for i in range(x_k.size()[0]):
            if x_k[i] < l[i]:
                x_k[i] = l[i]
            elif x_k[i] > u[i]:
                x_k[i] = u[i]
        return x_k

    def compute_KKT(self, x_k, aug_lag_grad_k, l, u):
        '''
        This the KKT condition for the augmented lagrangian problem 
        Eq. 17.51 Nocedal. This is an exit criteria for the trust region
        problem.
        Input:
            x_k : current x
            aug_lag_grad_k : current augmented lagrangian gradient
            l : lower bound constraint
            u : upper bound constraint
        '''
        x_proj = self.project_vector(x_k - aug_lag_grad_k, l, u)

        return torch.norm(x_k - x_proj)

    def optimize(self, lag, aug_lag, bc_arr, l, u, n, ni, u_k, omega_k, maxit, x0 = None, use_sr1 = False):
        '''
        This function optmizes the ubproblem using trust region
        method
        Input:
            lag : lagrangian funtion
            aug_lag : augmented lagrangian function
            bc_arr : list of bound constraint functions
            l : lower bound constraint
            u : upper bound constraint
            n : number of variables in x
            ni : number of inequality constraints
            u_k : current penalty on constraint violation
            omega_k : exit criteria 
            maxit : maximum number of iterations
            x0 : starting x
            use_sr1 : uses sr1 update to compute hessian of lagrangian
        '''
        if x0 != None:
            x_k = x0
        else:
            x_k = self.beta*torch.ones( n + ni, 1, dtype = float)
        delta_k = self.delta0
        G_k = self.beta*torch.eye(n + ni, dtype = float)
        lag_grad_k, aug_lag_grad_k = self.compute_gradients(x_k, lag, aug_lag)
        self.f_all.append(self.compute_KKT(x_k, aug_lag_grad_k, l, u))
        for k in range(maxit):
            l_trust_k, u_trust_k = self.compute_trust_region_bounds(x_k, l, u, delta_k)
            p_k, red_k = self.gradproj.optimize(G_k, aug_lag_grad_k, l_trust_k, u_trust_k, maxit)
            rho_k = (aug_lag(x_k + p_k) - aug_lag(x_k))/(red_k)
            if rho_k < 0.25 :
                delta_k = 0.25*delta_k
            else:
                if rho_k > 0.75 and torch.norm(p_k) - delta_k < self.tol:
                    if 2*delta_k < self.delta_max:
                        delta_k = 2*delta_k 
                    else:
                        delta_k = self.delta_max
                else:
                    delta_k = delta_k
            if rho_k > self.eta:
                x_k_new = x_k + p_k
            else:
                x_k_new = x_k

            A_k = self.compute_gradient_matrix(x_k_new, bc_arr)
            lag_grad_k_new, aug_lag_grad_k_new = self.compute_gradients(x_k_new, lag, aug_lag)
            G_k = self.compute_hessian(x_k_new, x_k, lag_grad_k_new, lag_grad_k, A_k, G_k, u_k, use_sr1)
            
            x_k = x_k_new
            lag_grad_k = lag_grad_k_new
            aug_lag_grad_k = aug_lag_grad_k_new
            
            self.f_all.append(self.compute_KKT(x_k, aug_lag_grad_k, l, u))
                
            if  self.f_all[-1] < omega_k:
                break
        
        return x_k, self.f_all[-1]

    def stats(self):
        '''
        This function returns stats and plots
        '''
        print("The algorithm has terminated after : " + str(len(self.f_all)) + " iterations")
        print("The optimal value of the objective funtion is : " + str(self.f_all[-1]))
        plt.plot(self.f_all)
        plt.grid()
        plt.title("Value of kkt violation vs iterations")
        plt.ylabel("Value of objective function")
        plt.xlabel("iteration")
        plt.show()
