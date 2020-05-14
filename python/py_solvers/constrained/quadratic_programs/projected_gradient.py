# This file contains implementation of algorithms that solve Quadratic
# programs (Projected Gradient, Active Set Method, Interior point)
# Author : Avadesh Meduri
# Date : 21/04/2020

import matplotlib.pyplot as plt

import torch
from py_autodiff.torch_autodiff import TorchAutoDiff
from py_solvers.constrained.quadratic_programs.conjugate_gradient import ProjectedConjugateGradient


class GradientProjection:

    def __init__(self, tol = 0.00001):
        '''
        This is an implementation of the Gradient projection method based
        on cauchy point. This algorithm is suited for bound constrained
        QPs. (Used in Augmented lagrangian method)
        Input:
            tol : tolerance for convergence
        '''
        self.tol = tol
        self.f_all = []
        self.pcg = ProjectedConjugateGradient()

    def compute_break_points(self, x_k, g, l, u):
        '''
        This function computes the step length intervals at which a particular
        component reaches the bound
        input:
            x_k : current iterate of x
            g : gradient
            l : lower bound vector on x
            u : upper bound vector on x
        '''

        t = torch.zeros(g.size()[0])
        for i in range(int(g.size()[0])):
            if g[i] < 0 and u[i] < float("inf"):
                t[i] = (x_k[i] - u[i])/g[i]
            elif g[i] > 0 and l[i] > -float("inf"):
                t[i] = (x_k[i] - l[i])/g[i]
            else:
                t[i] = float("inf")

        return t

    def project_x(self, x_k, l, u):
        '''
        Projects x
        Input:
            x_k : current x value
            l : lower bound
            u : upper bound
        '''
        for i in range(int(x_k.size()[0])):
            if x_k[i] > u[i]:
                x_k[i] = u[i]
            elif x_k[i] < l[i]:
                x_k[i] = l[i]
        return x_k

    def compute_projected_x(self, x_k, g, t_tilda, t):
        '''
        Computes the projected x based on the bounds. 16.71 Nocedal
        Also computes the projected descent direction 16.72
        Input:
            x_k : current value of x
            g : gradient
            t_tilde : the component wise value of step length
            t : current base step length
        '''
        x = torch.zeros(x_k.size()[0], dtype = float)
        p = torch.zeros(x_k.size()[0], dtype = float)
        for i in range(int(x_k.size()[0])):
            if t < t_tilda[i]:
               x[i] = x_k[i] - t*g[i]
               p[i] = -g[i]
            else:
                x[i] = x_k[i] - t_tilda[i]*g[i]
                p[i] = 0

        return x, p

    def compute_cauchy_point(self, x_k, G, c, l, u):
        '''
        This function computes the cauchy point for the current iterate
        input:
            x_k : current iterate of x
            G : positive definite cost matrix
            c : linear cost component (c^{T}x)
            l : lower bound vector on x
            u : upper bound vector on x
        '''

        g = G.matmul(x_k) + c
        t_tilda = self.compute_break_points(x_k, g, l, u)
        step_range = torch.cat((torch.zeros(1), torch.sort(t_tilda).values))
        x_c = None
        for i in range(step_range.size()[0] - 1):
            if abs(step_range[i+1] - step_range[i]) < self.tol:
                continue
            else:
                x_proj, p_proj = self.compute_projected_x(x_k, g, t_tilda, step_range[i])                
                fd = torch.t(c).matmul(p_proj) + torch.t(x_proj).matmul(G).matmul(p_proj)
                fdd = torch.t(p_proj).matmul(G).matmul(p_proj)
                if fd > 0:
                    x_c = x_proj
                    break
                elif -fd/fdd < step_range[i+1] - step_range[i]:
                    x_c = x_proj - (fd/fdd)*p_proj
                    break
        if x_c == None:
            x_c, _ = self.compute_projected_x(x_k, g, t_tilda, step_range[-1])
        return x_c
            

    def generate_active_set(self, x_c, l, u):
        '''
        forms the equality jacobian based active set using cauchy points
        It also returns the b vector from the eaulity constraint
        Input:
            x_c : cuachy poiny
            l : lower bound
            u : upper bound
        '''
        A = None
        b = None
        for i in range(x_c.size()[0]):
            if abs(x_c[i] - l[i]) < abs(self.tol):
                if A == None:
                    A = torch.zeros(1, x_c.size()[0], dtype = float)
                    A[0][i] = 1.0
                else:
                    A = torch.cat((A, torch.zeros(1, x_c.size()[0], dtype = float)))
                    A[-1][i] = 1.0
                if b == None:
                    b = torch.zeros(1,1, dtype = float) 
                    b[0][0] = l[i]
                else:
                    b = torch.cat((b, torch.zeros(1,1, dtype = float)))
                    b[-1][0] = l[i]
            elif abs(x_c[i] - u[i]) < abs(self.tol):
                if A == None:
                    A = torch.zeros(1, x_c.size()[0], dtype = float)
                    A[0][i] = 1.0
                else:
                    A = torch.cat((A, torch.zeros(1, x_c.size()[0], dtype = float)))
                    A[-1][i] = 1.0
                if b == None:
                    b = torch.zeros(1,1, dtype = float) 
                    b[0][0] = u[i]
                else:
                    b = torch.cat((b, torch.zeros(1,1, dtype = float)))
                    b[-1][0] = u[i]
        return A, b

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

    def optimize(self, G, c, l, u, maxit):
        '''
        This function optimizes the QP
        Input:
            G : positive definite cost matrix
            c : linear cost component (c^{T}x)
            l : lower bound
            u : upper bound
            maxit : maximum iterationts
        '''
        assert torch.gt(u - l, -self.tol).all()
        x = (l + u)/2.0 # feasible starting point
        for k in range(maxit):
            self.f_all.append(self.pcg.compute_current_cost(x, G, c))
            x_c = self.compute_cauchy_point(x, G, c, l, u)
            A_k, b_k = self.generate_active_set(x_c, l, u)
            if A_k != None and b_k != None:
                x_new, _ = self.pcg.optimize(G, c, A_k, b_k, maxit)
            else: 
                # This handles unconstrained case where cauchy point does
                # not lie on boundary (Using Newton step to minimize) 
                # Using LU factorization as G may be indefinite
                G_LU = torch.lu(G)
                x_new = -torch.lu_solve(c, *G_LU)

            x = self.project_x(x_new, l, u)
            if k > 1:
                # Exit criteria (This is temporary)
                # The correct exit criteria is to check KKT conditionsp
                if self.f_all[-1] - self.f_all[-2] < self.tol:
                    break
    
        return x, self.pcg.compute_current_cost(x, G, c)

                
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
