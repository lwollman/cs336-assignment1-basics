"""
    Codes that we will plug into adapters.py

   

"""
from typing import Optional

import einx
import math
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
        sigma = math.sqrt(sigma_squared)
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
        

class RMSLayerNormalization(torch.nn.Module):
    """
    Problem (rmsnorm): Root Mean Square Layer Normalization (1 point)
    Deliverable: Implement RMSNorm as a torch.nn.Module. 
    
    Note: Remember to upcast your input to torch.float32 before performing the normalization (and
    later downcast to the original dtype), as described above.
    To test your implementation, implement the test adapter at [adapters.run_rmsnorm]. Then, run uv
    run pytest -k test_rmsnorm.
    """

    def __init__(
            self, 
            d_model: int, 
            eps: float = 1e-5, 
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            ):  # -> 
        """
        Parameters
        ----------
        d_model : int
            Hidden dimension of the model.
            i.e. the dimension of the vector space into which the tokens are embedded.
        eps : float
            Epsilon value for numerical stability
        device : torch.device or None, optional
            Device to store the parameters on.
        dtype : torch.dtype or None, optional

        """
        super().__init__()
        self._eps = eps
        self._d_model = d_model
        g = torch.empty(d_model)
        g = torch.ones(d_model)
        # g = torch.nn.init.trunc_normal_(
        #     g, 
        #     mean=0.0,
        # ) 
        g = torch.nn.Parameter(g)  # learnable
        self.g = g

    @property
    def eps(self) -> float:
        return self._eps
    
    def forward_crude(self, a: torch.Tensor) -> torch.Tensor:
        """
        This example uses very crude logic.
        Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.

        Normalize the input tensor by the scalar RMS(a) -- see Equation (4) in the notes.

        """

        a_squared = a ** 2. # x.pow. 
        rms_a = a_squared.mean(axis=-1) + self.eps  #
        norm_rms = a / rms_a
        
        return self.mc_rms

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.

        Normalize the input tensor by the scalar RMS(a) -- see Equation (4) in the notes.
        """
        # Upcast to float32
        in_dtype = a.dtype
        a_float32 = a.to(torch.float32)

        # Use einx
        a_squared_einx = einx.dot("..., ... -> ...", a_float32, a_float32)
        a_squared = a_float32 ** 2. # x.pow. 

        # use a space to separate dimensions in the einx input string.
        # here we are saying use the last dimension.
        # To use second last would be einx.mean("... [d_model] [qq]")
        ms_a_einx = einx.mean(
            description="... [d_model]", 
            tensor=a_squared_einx, 
            keepdims=True
            ) + self.eps
        rms_a_einx = ms_a_einx.pow(0.5)        

        ms_a = a_squared.mean(axis=-1) + self.eps  #
        rms_a = ms_a.pow(0.5)

        #norm_rms = a_float32 / rms_a. # has wrong dims because lost one in mean
        norm_rms_einx = a_float32 / rms_a_einx  # dims OK here becaue of kkeepdims
        # return g * norm_rms_einx
        weighted = self.g * norm_rms_einx

        return weighted.to(in_dtype)

class SwiGLUFFN(torch.nn.Module):
    """
    
    Deliverable: 
    Implement the SwiGLU feed-forward network, composed of a SiLU activation
    function and a GLU.

    **Note**: In this particular case, you should feel free to use torch.sigmoid in your implementation
    for numerical stability.

    You should set dff to approximately 8/3 × dmodel in your implementation, while ensuring that
    the dimensionality of the inner feed-forward layer is a multiple of 64 to make good use of your
    hardware. 
    To test your implementation against our provided tests, you will need to implement
    the test adapter at [adapters.run_swiglu]. 
    
    Then, run uv run pytest -k test_swiglu to test your implementation.
    """
    def __init__(
            self, 
            d_model: int, 
            d_ff: Optional[int] = None, 
            # dimensional_expansion_factor: Optional[float] = 8/3,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            ):  # -> 
        """
        Parameters
        ----------
        d_model : int
            Hidden dimension of the model.
            i.e. the dimension of the vector space into which the tokens are embedded.
        d_ff: int
            This is a scaled version of hte d_model, approximately equal to
            8./3 * d_model.  Note that other implemenations may benefit form using 
            dimensional_expansion_factor
        device : torch.device or None, optional
            Device to store the parameters on.
        dtype : torch.dtype or None, optional

        """
        super().__init__(*args, **kwargs)
        

    def forward(self, stuff):
        pass