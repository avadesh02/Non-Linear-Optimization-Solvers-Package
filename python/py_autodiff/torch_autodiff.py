# This class uses torch to compute jacobians and hessians
# Author: Avadesh Meduri
# Date : 14/04/2020

import torch

class TorchAutoDiff:

    def __init__(self):
        self.name = "TorchAutoDiff"

    def compute_gradient(self, fun, x):
        '''
        Computes gradient at x
        Input:
            fun : function to be diffrentiated
            x : point at which gradient is to be computed
        '''
        assert torch.is_tensor(x)
        # defining as torch tensor to ensure gradients can be computed every time
        x_torch = torch.clone(x).detach().requires_grad_(True)
        y = fun(x_torch)
        y.backward()
        grad = x_torch.grad
            
        return grad
