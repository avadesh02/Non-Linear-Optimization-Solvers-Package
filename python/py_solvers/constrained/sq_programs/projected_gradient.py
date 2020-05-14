# This is an implementation of the Non linear gradient projection 
# algorithm with trust region 18.6 Nocedal.
# Author : Avadesh Meduri
# Date : 27/04/2020

import torch
import matplotlib.pyplot as plt
from py_autodiff.torch_autodiff import TorchAutoDiff

from py_solvers.constrained.quadratic_programs.projected_gradient import GradientProjection

class NonLinearGradientProjection:

    def __init__(self, delta0 = 0.05, delta_max = 5, eta = 0.01, beta = 0.01, r = 1e-8, tol = 0.0001):
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

    def compute_gradient(self, x_k, fun):
        '''
        This function computes the derivative of the objective 
        function.
        Input:
            x_k : value at which gradient to be taken
            fun : objective to be minimized
        '''
        grad_k = self.tad.compute_gradient(fun, x_k)
        
        return grad_k

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
        if torch.norm(s_k) > 0 and torch.norm(y_k) > 0:
            if torch.t(s_k).matmul(y_k) > 0.2*denominator:
                theta_k = 1
            else:
                theta_k = 0.8*denominator/(denominator - torch.t(s_k).matmul(y_k))
            
            r_k = theta_k*y_k + (1 - theta_k)*B_old.matmul(s_k)

            B_k = -((B_old.matmul(s_k).matmul(torch.t(s_k)).matmul(B_old)))/(denominator)
            B_k += (r_k.matmul(torch.t(r_k)))/(torch.t(s_k).matmul(r_k))
            B_k += B_old
        else:
            B_k = B_old

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
        if torch.norm(s_k) > 0 and torch.norm(y_k) > 0:
            if torch.norm(denominator) > self.r*torch.norm(s_k)*torch.norm(y_k - B_old.matmul(s_k)):
                numerator = (y_k - B_old.matmul(s_k)).matmul(torch.t(y_k - B_old.matmul(s_k)))
                B_k = B_old + (numerator/denominator)
            else:
                B_k = B_old
        else:
            B_k = B_old
        
        return B_k

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

    def compute_KKT(self, x_k, grad_k, l, u):
        '''
        This the KKT condition for the augmented lagrangian problem.
        This is the exit criteria 
        Eq. 17.51 Nocedal. This is an exit criteria for the trust region
        problem.
        Input:
            x_k : current x
            grad_k : current gradient of the objective function
            l : lower bound constraint
            u : upper bound constraint
        '''
        x_proj = self.project_vector(x_k - grad_k, l, u)

        return torch.norm(x_k - x_proj)

    def optimize(self, fun, l, u, maxit, tol, x0 = None, use_sr1 = False):
        '''
        This function optmizes the ubproblem using trust region
        method
        Input:
            fun : objective function to be minimized 
            l : lower bound constraint
            u : upper bound constraint
            maxit : maximum number of iterations
            tol : tolerance to exit
            x0 : starting x
            use_sr1 : uses sr1 update strategy to update the hessian
                    Works well when the function has indefinite hessian
        '''
        if x0 != None:
            x_k = x0
        else:
            x_k = 0.5*(l + u)

        delta_k = self.delta0
        G_k = self.beta*torch.eye(l.size()[0], dtype = float)
        grad_k = self.compute_gradient(x_k, fun)
        self.f_all.append(self.compute_KKT(x_k, grad_k, l, u))
        for k in range(maxit):
            l_trust_k, u_trust_k = self.compute_trust_region_bounds(x_k, l, u, delta_k)
            p_k, red_k = self.gradproj.optimize(G_k, grad_k, l_trust_k, u_trust_k, maxit)
            rho_k = (fun(x_k + p_k) - fun(x_k))/(red_k)
            if rho_k < 0.25 :
                delta_k = 0.25*delta_k
            else:
                if rho_k > 0.75: # and torch.norm(p_k) - delta_k < self.tol:
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
            grad_k_new = self.compute_gradient(x_k_new, fun)
            if use_sr1:
                G_k = self.compute_sr1_update(x_k_new, x_k, grad_k_new, grad_k, G_k)
            else:
                G_k = self.compute_dampled_bfgs_update(x_k_new, x_k, grad_k_new, grad_k, G_k)
            x_k = x_k_new
            grad_k = grad_k_new
            self.f_all.append(self.compute_KKT(x_k, grad_k, l, u))
                
            if  self.f_all[-1] < tol:
                break
        
        return x_k, self.f_all[-1]

    def stats(self):
        '''
        This function returns stats and plots
        '''
        print("The algorithm has terminated after : " + str(len(self.f_all)) + " iterations")
        print("The optimal kkt violation is : " + str(self.f_all[-1]))
        plt.plot(self.f_all)
        plt.grid()
        plt.title("Value of kkt violation vs iterations")
        plt.ylabel("Value of objective function")
        plt.xlabel("iteration")
        plt.show()

