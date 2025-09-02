"""
    Codes that we will plug into adapters.py

"""
from typing import Optional

import torch

class Linear(torch.nn.Module):
    """
        Deliverable: Implement a Linear class that inherits from torch.nn.Module 
        and performs a linear transformation. Your implementation should follow
        the interface of PyTorch’s built-in nn.Linear module, except for not 
        having a bias argument or parameter. 
    """
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        device: Optional[torch.device] = None, 
        dtype: Optional[torch.dtype] = None, 
        ): 
        """
        Constructor.
        
        Parameters
        ---------
        in_features: int
            final dimension of the input
        out_features: int 
            final dimension of the output  
        device: Optional[torch.device]
            Device to store the parameters on  
        dtype: Optional[torch.dtype]
            Data type of the parameters):
        """
        self.weight = "TODO"
        print("do your homework")

    
    def forward(self, 
                x: torch.Tensor
                ) -> torch.Tensor: 
                """
                Apply the linear transformation to the input.  
                Make sure to:  
                subclass nn.Module  • call the superclass constructor  • construct and store your parameter as W (not W ⊤) for memory ordering reasons, putting it in an nn.Parameter  • of course, don’t use nn.Linear or nn.functional.linear
                """
                return torch.zeros((4, 12, 128))
