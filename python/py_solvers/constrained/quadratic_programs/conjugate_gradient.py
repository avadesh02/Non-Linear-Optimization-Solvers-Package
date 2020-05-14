# This file contains implementation of the projected conjugate gradient Method
# that solves QP with only equality constraints. Alg. 16.2 Nocedal 
# Author : Avadesh Meduri
# Date : 20/04/2020

import matplotlib.pyplot as plt

import torch
from py_autodiff.torch_autodiff import TorchAutoDiff

class ProjectedConjugateGradient:

    def __init__(self, tol = 0.00001):
        '''
        This is the Projected Conjugate Gradient Method algorithm
        that solves QP with only equality constraints
        Algorithm 16.2 Nocedal
        Input:
            tol : tolerance to terminate
        '''
        self.tol = tol
        self.f_all = []

    def compute_feasible_start(self, A, b):
        '''
        This computes feasible starting point for the algorithm using
        normal equations approach. (INverse is computed using cholwsky)
        x0 = A^T(AA^T)^(-1)b
        Input:
            A : jacobian matrix of equality constraints
            b : values of equality constratins (Ax = b)
        '''

        L = torch.cholesky(A.matmul(torch.t(A)))
        inv = torch.cholesky_inverse(L)
        return (torch.t(A).matmul(inv)).matmul(b)

    def compute_projection_matrix(self, A, H):
        '''
        This computes the projection matrix for the CG. (16.33 Nocedal)
        Input:
            A : jacobian matrix of equality constraints
            H : pre conditioner for CG
        '''
        inv_H = torch.inverse(H)
        P = torch.inverse(A.matmul(inv_H).matmul(torch.t(A)))
        P = torch.t(A).matmul((P.matmul(A)).matmul(inv_H))
        P = torch.eye(A.size()[1], dtype=float) - P

        return inv_H.matmul(P)

    def compute_augmented_system_solution(self, H, A, r):
        '''
        This function computes g+ and v+ using 16.34 Nocedal
        Input:
            H : preconditioner
            A : equailty constraint matrix
            r : residual
        '''

        K1 = torch.cat((H, torch.t(A)), 1)
        Z = torch.zeros(A.size()[0], K1.size()[1] - A.size()[1], dtype = float)
        K2 = torch.cat((A, Z), 1)
        K = torch.cat((K1, K2))
        z = torch.zeros(K.size()[0] - r.size()[0], 1, dtype=float)
        b = torch.cat((r, z))
        
        LU = torch.lu(K)
        x = torch.lu_solve(b, *LU)

        return x[0:H.size()[0]] 

    def compute_current_cost(self, x, G, c):
        '''
        This function evaluates the current value of the quadtratic cost
        Input:
            x : current x at iterate k
            G : positive definite cost
            c : linear cost component (c^{T}x)
        '''
        q = 0.5*torch.t(x).matmul(G).matmul(x)

        return q + torch.t(x).matmul(c)

    def optimize(self, G, c, A, b, maxit, H = None, use_aug_met = False):
        '''
        This function optimizes the equality constrained QP
        Input:
            G : positive definite cost matrix
            c : linear cost component (c^{T}x)
            A : jacobain of equality constraints
            b : values of the equality constraints (Ax = b)
            maxit : maximum numer of iterations
            H : preconditioner matrix for the CG algorithm
                Best options are Identity matrix or diagonal matrix
                whose diagonal elements are the diagonal elemnts of the 
                G matrix (abs(diag(G)))
            use_aug_met : Use augmented method to compute g+.
                Note: aug met may lead to rounding errors
        '''

        x = self.compute_feasible_start(A, b)
        if H == None:
            H = torch.eye(A.size()[1], dtype=float)
        P = self.compute_projection_matrix(A, H)
        r = G.matmul(x) + c
        if use_aug_met:
            g = self.compute_augmented_system_solution(H, A, r)
        else:
            g = P.matmul(r)
        d = -g
            
        for k in range(maxit):
            if torch.norm(d) < self.tol:
                self.f_all.append(self.compute_current_cost(x, G, c))
                break

            self.f_all.append(self.compute_current_cost(x, G, c))
            alpha = (torch.t(r).matmul(g))/(torch.t(d).matmul(G).matmul(d))
            x = x + alpha*d
            r_plus = r + alpha*G.matmul(d)
            if use_aug_met:
                g_plus = self.compute_augmented_system_solution(H, A, r_plus)
            else:
                g_plus = P.matmul(r_plus)
            beta = (torch.t(r_plus).matmul(g_plus))/(torch.t(r).matmul(g))
            d = -g_plus + beta*d
            g = g_plus
            r = r_plus
            
            
        
        return x, self.compute_current_cost(x, G, c)


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
