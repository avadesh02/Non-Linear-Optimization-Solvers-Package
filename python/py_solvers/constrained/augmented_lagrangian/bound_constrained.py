# This is an implementation of the bound constrained formulation
# of the aigmented lagrangian problem. Alg - 17.4 Nocedal
# Author : Avadesh Meduri
# Date : 22/04/2020

import torch
import matplotlib.pyplot as plt
from py_autodiff.torch_autodiff import TorchAutoDiff

from py_solvers.constrained.augmented_lagrangian.gradient_projection import AugGradientProjection
from py_solvers.constrained.sq_programs.projected_gradient import NonLinearGradientProjection

class BoundConstrainedFormulation:

    def __init__(self, u0 = 10, delta_max = 5.0, tol = 0.00001, maxit = 100):
        '''
        Input:
            u0 : inital value of penalty on constraints
            detla_max : maximum trust region bound
            tol : tolerance for checking equality
            maxit : maximum iterations for the gradient projection method
        '''
        self.aug_gradproj = AugGradientProjection(delta_max=delta_max)
        self.ngp = NonLinearGradientProjection(delta_max=delta_max)
        self.u0 = torch.tensor(u0, dtype=float)
        self.tol = tol
        self.maxit = maxit
        self.f_all = []
        self.const_violation = []
    def convert_to_bound_constraint_form(self, ce, ci, n):
        '''
        This function converts the given optimization problem into 
        bound constrained form, ie,. transforms inequality constraints
        to equality constraints
        Input:
            ce : array of eauality constraint functions
            ci : array of inequality constraint functions
            n : number of variables in x before bound constraints are added
        '''

        bound_constraints_arr = ce
        # this function is neccessary to preven lambda to bind to the same
        # x in all the loops
        def create_lambda(fun, i, n):
            return lambda x : fun(x) - x[i + n]

        for i in range(len(ci)):
            bc_e = create_lambda(ci[i], i, n)
            bound_constraints_arr.append(bc_e)

        return bound_constraints_arr

    def compute_lagrangian(self, x_k, l_k, fun, bc_arr):
        '''
        This function computes the current value of the lagrangian 
        Input:
            x_k : current x
            l_k : current lagrange multipliers
            fun : function to be minimized
            bc_arr : array of bounded constraints
        '''
        lag_val = fun(x_k)
        for i in range(len(bc_arr)):
            lag_val -= l_k[i]*bc_arr[i](x_k)

        return lag_val

    def compute_aug_lagrangian(self, x_k, l_k, u_k, fun, bc_arr):
        '''
        This function computes the current value of the augmented lagrangian 
        Input:
            x_k : current x
            l_k : current lagrange multipliers
            u_k : penalty on the current constarint violation
            fun : function to be minimized
            bc_arr : array of bounded constraints
        '''
        aug_lag_val = fun(x_k)
        for i in range(len(bc_arr)):
            aug_lag_val -= l_k[i]*bc_arr[i](x_k)
            aug_lag_val += 0.5*u_k*((bc_arr[i](x_k))**2) 

        return aug_lag_val

    def compute_constraint_violation(self, x_k, bc_arr):
        '''
        This function computes the constraint violation norm
        Input:
            x_k : current x
            bc_arr : array of bounded constraints
        '''
        violation = 0
        for i in range(len(bc_arr)):
            violation += torch.norm(bc_arr[i](x_k))
        return violation

    def update_lagrangian_multipliers(self, x_k, lambda_k, bc_arr, u_k):
        '''
        This function updates the lagrange multipliers. Eq 17.39 Nocedal
        Input:
            x_k : current x
            lambda_k : current lagrange multipliers
            bc_arr : bound constraints
            u_k : current penalty on violation
        '''
        for i in range(len(bc_arr)):
            lambda_k[i] -= u_k*bc_arr[i](x_k)

        return lambda_k


    def compute_bounds(self, n, ni):
        '''
        This function computes the bound constraints
        Input:
            n : number of variables in x before bound constraints are added
            ni : number of inequality constraints
        '''
        l = -float("inf")*torch.ones(n + ni, 1, dtype=float)
        l[n:] = 0.0
        u = float("inf")*torch.ones(n + ni, 1, dtype=float)

        return l, u

    def optimize(self, fun, ce, ci, n, omega_opt, eta_opt, maxit, x0 = None, use_sr1 = False):
        '''
        This function optimized the given non linear problem
        Input:
            fun : objective function to be minimized
            ce : array of equality constraints
            ci : array of inequality constraints
            n : number of variables
            omega_opt : optimal exit critiria for Augmented lagrangian kkt condition
            eta_opt : optimal exit criteria for constraint violation
            maxit : maximum number of iterations
            x0 : warm starting x0
            use_sr1 : uses sr1 update to compute hessian of lagrangian
        '''    
        # Converting constraints to bound constraints
        ne = len(ce)
        ni = len(ci)
        bc_arr = self.convert_to_bound_constraint_form(ce, ci, n)
        # computeing bound constraints
        l, u = self.compute_bounds(n, ni)
        # initialising parameters
        u_k = self.u0
        omega_k = 1/u_k
        eta_k = 1/torch.pow(u_k, 0.1)
        # initialising lagrange multilpliers
        lambda_k = 0.0001*torch.ones(ne + ni, 1, dtype = float)
        if x0 != None:
            # Add things correctly here to warm start
            x_k = torch.cat((x0, torch.zeros(ni, 1, dtype = float)))
        else:
            x_k = torch.ones( n + ni, 1, dtype = float)
        
        # setting up things for solving the gradient projection
        lag_k = lambda x : self.compute_lagrangian(x, lambda_k, fun, bc_arr)
        aug_lag_k = lambda y : self.compute_aug_lagrangian(y, lambda_k, u_k, fun, bc_arr)
        for k in range(maxit):
            print("iter no. : " + str(k))
            x_k, kkt_violation = self.ngp.optimize(aug_lag_k, l, u, self.maxit, \
                                            omega_k, x0 = x_k, use_sr1=use_sr1)
            self.ngp.f_all = []
            print("kkt violation : "  + str(kkt_violation) )
            violation_k = self.compute_constraint_violation(x_k, bc_arr)
            self.f_all.append(fun(x_k))
            self.const_violation.append(violation_k)
            print("constraint violation : " + str(self.const_violation[-1]))
            if violation_k < eta_k + self.tol:
                if violation_k < eta_opt + self.tol and kkt_violation < omega_opt + self.tol:
                    break
                lambda_k = self.update_lagrangian_multipliers(x_k, lambda_k, bc_arr, u_k)
                eta_k = eta_k/torch.pow(u_k, 0.9)
                omega_k = omega_k/u_k
            else:
                u_k = 2.0*u_k
                eta_k = eta_k/torch.pow(u_k, 0.1)
                omega_k = omega_k/u_k
            print('penalty', u_k)

        return x_k[0:n]

    def stats(self):
        '''
        This function returns stats and plots
        '''
        print("The algorithm has terminated after : " + str(len(self.f_all)) + " iterations")
        print("The optimal value of the objective funtion is : " + str(self.f_all[-1]))
        fig, axs = plt.subplots(2, 1, sharex=True)
        axs[0].plot(self.f_all, label="objective value")
        axs[0].set_ylabel("Value of objective function")
        axs[0].legend()
        axs[0].grid()
        axs[1].plot(self.const_violation, label = "constraint_violation")
        axs[1].grid()
        axs[1].set_ylabel("Constraint violation")
        axs[1].legend()
        plt.xlabel("iteration")
        plt.show()
