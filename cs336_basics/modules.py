"""
    Codes that we will plug into adapters.py

"""

from jaxtyping import Float
from loguru import logger
from torch import Tensor
from typing import Optional

import einx
import math
import numpy as np
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
            std=3.01, # This is usually 1.
            a=-0.02, 
            b=0.02
            )
        weights = torch.nn.Parameter(weights)  # The casting as Parameter designates this as learnable
        self.weights = weights  # cause state_dict to track weights
        # print("std has an unconventional value, shoudl be 1")

    
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

    Defines a matrix of size num_embeddings (vocabulary size) by embedding_dim (d_model).
    Thus, this is really just a lookup table of the embedded vectors for each element of the vocabulary.
    Conventionally, E is an element of R_{n_vocab, d_model} and each row corresponds to an embedded token.

    It is interesting to note that this is initialized randomly, on a truncated normal distrubution, 
    and this means that the embedded vector length will be random as well, but will tend to have
    length [TODO: check this -- sqrt(d_model)] as it is a random walk of d_model steps, where the step 
    size is normally 1.

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
        super().__init__()
        # if np.mod(d_ff, 64) != 0: 
        #     pass  # round it up!, no need to worry about this yet

        # Leverage your Linear class from the very first exercise to declare
        # arrays of learnable parameters.
        self.W1 = Linear(
            in_features=d_model,
            out_features=d_ff
            )
        # transposed W2 -- see EQuation 7 in the text.
        self.W2 = Linear(
            in_features=d_ff,
            out_features=d_model
            )  
        self.W3 = Linear(
            in_features=d_model,
            out_features=d_ff
            )


    def forward(self, x):
        """
        implement equation 7
        """

        w1x = self.W1.forward(x) # Note this is the same as below: 
        # w1x = self.W1(x)
        w3x = self.W3.forward(x) # Note this is the same as below: 
        
        silu = w1x * torch.sigmoid(w1x)

        w2_input = silu * w3x
        result = self.W2(w2_input)
        return result


class RotaryPositionalEmbedding(torch.nn.Module):
    """
    From page 24/50 of the Assigment1 pdf

    Deliverable: Implement a class RotaryPositionalEmbedding that applies RoPE to the input tensor.

    This basically amounts to creating the R rotation matrix and applying it... but constructing
    the specific instance of the R matrix that you will use to apply RoPE in the forward methods 
    requires wrangling the indices i, k.
    i is the position of a token within a given sequence ... (this may seem a bit weird
    at first, that the rotation is not tied to the token's position or order in the vocabulary, 
    but rather in the sequence, and that the same token willl be rotated differently based on its position
    within a sequence, but this is actually fundamental to how rope (and positional encoding 
    transformations in general) works.

    There is one R matrix for each relative position i.
    
    , but RoPE forward method will normally combine
    submatrices from many of the R_i of equation 9 (or R_i^k of Eqn 8) since it will probably be 
    normal to feed sequences of tokens
    
    References:
    The RoPE paper is worth looking at.
    For some shortcuts, 
    ~/software/ramayer/google-colab-examples/HelloWorld_Transformer_with_RoPE.ipynb
    goes through a use case where RoPE is actually used for a toy problem about predicting
    the future value of a sinusoidal function.
    This tutorial uses fairly dense code.

    A more "following the pdf" version of the RoPE stuffs can be found in Ron's RoPE notebook here:
    https://github.com/rmayer-sst/stanford-cs336-assignment1-basics/blob/ron/cs336_basics/ron_rope.py


    To test your implementation, complete [adapters.run_rope] and make sure it passes 
    uv run pytest -k test_rope.

    """
    def __init__(
            self, 
            theta: float, 
            d_k: int, 
            max_seq_len: int, 
            device=None
            ):
        """
        Construct the RoPE module and create buffers if needed.

        The init makes the values that are needed to populate the Rotation matrix. 

        Parameters
        ----------
        theta: float 
            Θ value for the RoPE (has value 10000 in the pdf ... should grow with sequence length)
        d_k: int 
            dimension of query and key vectors.
            This is the dimension of a vector that gets multiplied by the $R^{i}$ matrix.
            This is not the embedding dimension (unless using single-headed attention).
            We do not need to worry about attention head splitting stuffs here.
            But if we wanted to mention them ... the embedding dimension d is split approximately evenly 
            among the heads ... this module is concerned with operations on one of those chunks.
            
        max_seq_len: int 
            Maximum sequence length that will be inputted.
            a.k.a. the "context length" -- numbers of tokens).

        device: torch.device | None = None 
            Device to store the buffer on

        TODO: spend more time with einx, 
        it can replicate rows and columns as well with shorthand
        exx = einx.rearrange(" a -> 1 1 a 1 2",x)

        For a fixed position i, there will be k rotation submatrices.

        """
        super().__init__()

        k = torch.arange(d_k//2, device=device)  # subspace dimensions; shape (d_k//2,)
        i = torch.arange(max_seq_len, device=device)   # positions; shape (max_seq_len,)

        # reshape to make rectangles
        k = einx.rearrange("a -> 1 a",k)  # shape(1,d_k//2)
        i = einx.rearrange("a -> a 1",i)  # shape(max_seq_len,1)

        assert isinstance(k, torch.Tensor) # just to make VS Code not complain
        assert isinstance(i, torch.Tensor) # just to make VS Code not complain

        # theta_{i,k} = i / (THETA^{2k/d_k})
        exponents = 2 * k / d_k  # step from zero to 1 in k steps
        theta_i_k = i / (theta**exponents)
        cos_theta_i_k = torch.cos(theta_i_k)
        sin_theta_i_k = torch.sin(theta_i_k)

        # Put these into a "buffer" as they are not learned things ... 
        # We used the "parameter" designation for things that are learned .. 
        # whereas, these, that are 
        # the buffers are designations for quantities we want to keep around (for efficiency)
        # and are static -- not learned,
        # ...     
        # persistent=False says we dont care to store this in the praemters file
        # data dertived features (such as empirical average signal strength) may wish to be saved.
        # Once registered we can access them via self.name (self.cosine_table for example)
        #
        # to see your buffers:
        # print(list(self.named_buffers()))
        #
        # 
        self.register_buffer(
            name="cosine_table",
            tensor=cos_theta_i_k,
            persistent=False
            )
        self.register_buffer(
            name="sine_table",
            tensor=sin_theta_i_k,
            persistent=False
            )
        return 

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of 
        the same shape. Note that you should tolerate x with an arbitrary number of 
        batch dimensions. You should assume that the token positions are a tensor of 
        shape (..., seq_len) specifying the token positions of x along the sequence 
        dimension.

        You should use the token positions to slice your (possibly precomputed) cos and sin tensors along
        the sequence dimension.

        When this is finally working it will take the following form:
        - A tensor of shape (..., seq_len, d_k, d_k) will be formed called R
        - A crafty application of einx will multiply every vector
         in the input x, by the d_k x d_k array.
        Parameters
        ----------
        x: torch.Tensor
            The embedded tokens that will be encoded with rotary operation
            shape is (..., seq_len, d_k).
            d_k should be even so that the elements can be paired over 2D rotation matrices.
        token_positions: torch.Tensor
            A map of the token positions (within the sequence)
            (..., seq_len)
            These are going to be used to "seek into the cosine and sine tables"
            ...
            it seems like we can either compute the R matrices for these token positions, 
            or we can sort the tokens and then use a sequential R ... but if these
            are not just 1, 2, 3 ... seq_len .. then i guess

        """
        
        # d_k = x.shape[-1]  # Defines the k-index (embedding attention subspace)
        # d_k_over_2 = d_k // 2  # num submatrices making up R_i
        # seq_len = x.shape[-2]  # This corresponds to the i-index
        # if np.mod(d_k,2) != 0:
        #     logger.error(f"d_k should be even ... instead its {d_k}")
        
        # extract the cos and sin values for the token positions
        #  ... 
        s = self.sine_table[token_positions]  # shape (..., seq_len, d_k//2)
        c = self.cosine_table[token_positions]  # shape (..., seq_len, d_k//2)
        
        # reshape x to get the pairs for rotation
        pairs = einx.rearrange("... s (d c) -> ... s d c", x, c=2)
        rotated = torch.empty_like(pairs)
        rotated[..., 0] = c * pairs[..., 0] - s * pairs[..., 1]
        rotated[..., 1] = s * pairs[..., 0] + c * pairs[..., 1]

        # reshape back to original shape (interleaved even/odd)
        rotated_vectors = einx.rearrange("... s d c -> ... s (d c)", rotated)
        return rotated_vectors
    

def softmax(
        in_features: Float[Tensor, " ..."], 
        dim: int
        ) -> torch.Tensor:
        """
        Apply the softmax operation to the input tensor along the last dimension.

        """
        x = in_features
        print("softmax input shape:", x.shape)
        maxx, _ = torch.max(x, dim=dim, keepdim=True)
        print("max values shape:", maxx.shape)
        shifted_x = x - maxx  # now won't blow up from large exponents

        exp_x = torch.exp(shifted_x)
        norm_by = torch.sum(dim=dim, keepdim=True, input=exp_x)

        result = exp_x / norm_by

        # # Subtract max for numerical stability
        # x_max = torch.max(x, dim=-1, keepdim=True).values
        # x_exp = torch.exp(x - x_max)
        # sum_exp = torch.sum(x_exp, dim=-1, keepdim=True)
        # softmax_result = x_exp / sum_exp
        return result

def scaled_dot_product_attention(
    queries: Float[Tensor, " ... n d_k"],
    keys: Float[Tensor, " ... m d_k"],
    values: Float[Tensor, " ... m d_v"],
    mask: Optional[Float[Tensor, " ... queries key"]] = None,
    ) -> Float[Tensor, " ... queries d_v"]:
    """
    Deliverable: Implement the scaled dot-product attention (SDPA) mechanism as a function.

    ** from the assignment: ... but this looks like an error, the seq_len for q, k are not equal in general.**
    Your implementation should handle keys and queries of shape (batch_size, ..., seq_len, d_k).

    Note: You should not use nn.MultiheadAttention or nn.functional.multi_head_attention_forward 
    in your implementation.

    To test your implementation, implement the test adapter at [adapters.run_sdpa]. Then, run 
    uv run pytest -k test_sdpa.
    uv run pytest -k test_scaled_dot_product_attention

    TODO: Review the shapes of these inputs .. the assignment seems to call for the second
    last dimension of q,k,v being "seq_len", but the auto AI doc, had the second last mode
    being same for for Q,K, but allow different for V.  The reality seems to be k,v agree in the second
    last mode, anbd q can be different (n,m)
    these m,n relate to sequence length concept though, where as d_k is the token-subspace dimension.
    It is along this d_k that the inner products are taken.

    Learning Notes:
    - The attention scores are computed as the dot product of queries and keys, scaled by the
      square root of the key dimension (d_k).
      The dimension d_k is the only one that it makes sense to take an inner product over.

    Parameters:
        queries: Float[Tensor, " ... queries d_k"]: Query tensor. 
        keys: Float[Tensor, " ... queries d_k"]: Key tensor.
        values: Float[Tensor, " ... key d_v"]: Value tensor.
        mask: Optional[Float[Tensor, " ... queries key"]]: Optional mask tensor.

    Returns:
        Float[Tensor, " ... queries d_v"]: Output of SDPA.
    """
    # use torch nicenes to multiply QK with ...
    logger.info("Shape check:")
    logger.info(f"queries : {queries.shape}")
    logger.info(f"keys : {keys.shape}")
    logger.info(f"values : {values.shape}")
    
    if mask is None:
        logger.error("No mask provided")
        # Consider making on the fly?        
    logger.info(f"mask : {mask.shape}")
        

    dk = keys.shape[-1]
    dv = values.shape[-1]
    # seq_len = dk = keys.shape[-2]
    # assert values.shape[-2] == seq_len
    
    logger.info(f"d_k: {dk}")
    logger.info(f"d_v: {dv}")
    qk = einx.dot("... q dk, ... k dk -> ... q k", queries, keys)
    qk = qk / math.sqrt(dk)  # normalize by sqrt d_k

    # float_mask = mask.to(dtype=torch.float32)

    # apply the mask if given:
    min_qk_elt = torch.min(qk) + float("-inf") #
        
    # do masking
    masked_qk = qk.clone()
    masked_qk[~mask] = min_qk_elt
    softmaxed_masked_qk = softmax(masked_qk, dim=-1)

    # old cruft here:    
    # masked_qk.masked_fill_(~mask, min_qk_elt)  # masked_fill_ modifies in place
    # masked_qk = einx.dot("..., ... -> ...", qk, float_mask) + (1.0 - float_mask) # * (-1e9)

    # finally multiply by V
    A = einx.dot("... n m, ... m dv -> ... n dv", softmaxed_masked_qk, values)
    
    return A

class MultiheadedSelfAttention(torch.nn.Module):
    """
    Deliverable: Implement multi-headed self-attention as a torch.nn.Module.

    Note: You should not use nn.MultiheadAttention or nn.functional.multi_head_attention_forward in your implementation.

    To test your implementation, implement the test adapter at [adapters.run_mhsa]. Then, run 
    
    uv run pytest -k test_mhsa.
    uv run pytest -k test_multihead_self_attention
    uv run pytest -s -v tests/test_model.py::test_multihead_self_attention

    Development notes
    """

    def __init__(
            self,
            d_model: int,
            num_heads: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            ):          
        """
        Constructor.

        Parameters
        ----------
        d_model : int
            Hidden dimension of the model.
            i.e. the dimension of the vector space into which the tokens are embedded.
        num_heads : int
            Number of attention heads.
        q_proj_weight : Float[Tensor, " d_model d_k * num_heads"]
            Query projection weights.
        k_proj_weight : Float[Tensor, " d_model d_k * num_heads"]
            Key projection weights.
        v_proj_weight : Float[Tensor, " d_model d_v * num_heads"]
            Value projection weights.
        o_proj_weight : Float[Tensor, " d_model d_model"]
            Output projection weights.
    """
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        # per Pdf: page 26 (Vaswani et al. 2017)
        d_k = d_model // num_heads
        d_v = d_k  # usually d_v == d_k, cross attention may differ

        # To implement multi-headed attention,
        # we need to split the inputs into heads
        # then loop over the heads, applying out attention mechanism
        # then concatenate the results (this is equations 12-13 in the notes)
        #. d_k can be thought of as the "query size"
        self.Q = Linear(
            in_features=d_model,
            out_features=d_k * num_heads,
            device=device,
            dtype=dtype
            )
        self.K = Linear(
            in_features=d_model,
            out_features=d_k * num_heads,
            device=device,
            dtype=dtype
            )
        self.V = Linear(
            in_features=d_model,
            out_features=d_v * num_heads,  # this is d_model in most cases
            device=device,
            dtype=dtype
            )
        self.O = Linear(
            in_features=d_v * num_heads,
            out_features=d_model,
            device=device,
            dtype=dtype
            )
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Development notes:
                # To implement multi-headed attention,
        # we need to split the inputs into heads
        # then loop over the heads, applying out attention mechanism
        # then concatenate the results (this is equations 12-13 in the notes)
        - We will use einx to help with the reshaping and combining of the heads.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (batch_size, seq_len, d_model).
        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, seq_len, d_model).
            Unexpected key(s) in state_dict: "q_proj.weights", "k_proj.weights", "v_proj.weights", "output_proj.weights".
        """
        d_k = self.d_model // self.num_heads
        d_v = self.d_model // self.num_heads

        Q = self.Q(x)  # shape (batch_size, seq_len, d_model)
        K = self.K(x)  # shape (batch_size, seq_len, d_model)
        V = self.V(x)  # shape (batch_size, seq_len, d_model
        logger.info(f"Shape check: {__class__.__name__}")
        logger.info(f"queries : {Q.shape}")
        logger.info(f"keys : {K.shape}")
        logger.info(f"values : {V.shape}")
        logger.info(f"d_k: {d_k}")
        logger.info(f"d_v: {d_v}")

        # reshape Q, K, V to (batch_size, num_heads, seq_len, d_k)
        Q_reshaped = einx.rearrange("b s (h dk) -> b h s dk", Q, h=self.num_heads)  # ummm b h s dk or b s h dk?
        K_reshaped = einx.rearrange("b s (h dk) -> b h s dk", K, h=self.num_heads)
        V_reshaped = einx.rearrange("b s (h dv) -> b h s dv", V, h=self.num_heads)  
        
        # TODO: Add causal masking .. this is going to be a simple triangular matrix, 
        # but the twist is a high dim torch tensor. ... however, we basically want the 
        # attention mechanism to consider ... ? what? You passed it some stuff ... 
        # and you want it to be "considered" by the attention ... yet somehow we 
        # are thinking about treating it as the first token only, then the first two, then the 
        # first 3 ...something seems amiss here .. 
        # actually, for training this triangular mask is not concerning,
        # and for inference, we won't need this -- so no worries:).

        # rope is next (TODO)        
        # make the mask here ...
        mask = torch.tril(torch.ones(Q_reshaped.shape[-2], K_reshaped.shape[-2], dtype=torch.bool, device=Q_reshaped.device))
        mask = mask.to(dtype=torch.bool)
        logger.info(f"generated causal mask of shape: {mask.shape}")
        mask = einx.rearrange(" q k ->1 1 q k", mask)  # broadcast to match batch and head dims
        #logger.info(f"generated causal mask of shape. after rearrange: {mask.shape}")
        mask = mask.repeat_interleave(self.num_heads, dim=0)  # repeat for each head
        mask = mask.repeat_interleave(self.num_heads, dim=1)  # repeat for each head
        logger.info(f"generated causal mask of shape. after repeat: {mask.shape}") 

        A_for_all_heads_needs_reshaping = scaled_dot_product_attention(
            Q_reshaped, K_reshaped, V_reshaped, mask=mask)

	# Contcatenation over the heads
        A_reshaped = einx.rearrange(
            "b h s dv -> b s (h dv)", 
            A_for_all_heads_needs_reshaping,
            )  # concatenate heads
        output = self.O(A_reshaped)  # shape (batch_size, seq_len, d_model)
    
    # ) -> Float[Tensor, " ... queries d_v"]: 
        # next step is apply our attenation mechanism per head

        # then we concatenate the results (undo the reshaping)

        # the masking stuff ... and rope ... are TBD
        return output
