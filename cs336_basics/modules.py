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
        super().__init__()
        sigma_squared = 2. / (in_features + out_features)
        sigma = torch.sqrt(sigma_squared)
        weights =  torch.empty(out_features, in_features)
        weights = torch.nn.init.trunc_normal_(
            weights, 
            mean=0.0, 
            std=3.01, a=-0.02, b=0.02)
        weights = torch.nn.Parameter(weights)  # The casting as Parameter designates this as learnable
        self.weights = weights
        print("std has an unconventional value, shoudl be 1")

    
    def forward(self, 
        x: torch.Tensor
        ) -> torch.Tensor: 
        """
        Apply the linear transformation to the input.  
        Make sure to:  
        subclass nn.Module  • call the superclass constructor  • construct and store your parameter as W (not W ⊤) for memory ordering reasons, putting it in an nn.Parameter  • of course, don’t use nn.Linear or nn.functional.linear
        """
        print(x.shape) # 4 , 12, 64
        print(self.weights.shape)
        result = x @ self.weights.T  # @ x
        # result = self.weights @ x.T
        
        # result = torch.zeros((4, 12, 128))
        # result[0,0,0] = -0.358869
        return result


class Embedding(torch.nn.Module):
    """
    Embedding lookup module.

    Methods
    -------
    forward(token_ids: torch.Tensor) -> torch.Tensor
        Lookup the embedding vectors for the given token IDs.

    Notes
    -----
    - Subclass of nn.Module.
    - Embedding matrix is initialized as nn.Parameter.
    - The embedding matrix has shape (num_embeddings, embedding_dim), with d_model as the final dimension.
    - Uses torch.nn.init.trunc_normal_ for initialization.
    - Does not use nn.Embedding or nn.functional.embedding.

    To test:
    > uv run pytest -k test_embedding.
    """
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim : int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            ):
        """
        Constructor.

        Parameters
        ----------
        num_embeddings : int
            Size of the vocabulary.
        embedding_dim : int
            Dimension of the embedding vectors (d_model).
        device : torch.device or None, optional
            Device to store the parameters on.
        dtype : torch.dtype or None, optional
    
        TODO: This is going to need to creat
        """
        super().__init__()
        embedding_matrix = torch.empty(num_embeddings, embedding_dim)
        embedding_matrix = torch.nn.init.trunc_normal_(
            embedding_matrix, 
            mean=0.0, 
            std=1.0, 
            a=-3.0, 
            b=3.0,
            )
        embedding_matrix = torch.nn.Parameter(embedding_matrix)  # The casting as Parameter designates this as learnable
        self.embedding_matrix = embedding_matrix

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Since the vocabulary is (by definition of the algorithm that generated it)
        is just the numbers in [0, N], we just simply get the rows by their index 
        which is the same as token id.
        """
        return self.embedding_matrix[token_ids,:]
        



