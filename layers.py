"""
    ./layers.py

        Contains definition for layers of the encoders (multi-head attention layer and a fully connected layer).

    Author: William MAILLET
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MultiHeadAttention(nn.Module):
    """
        A simple multi-head attention module that 
        computes the attention 

        Code adapted from:
            uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html

        Parameters:
        -----------
        input_dim: int 
            The input dimension
        embed_dim: int 
            The embedding dimension (must be 0 % n_attn_heads)
        n_attn_heads: int
            The number of heads of the multi-attention 
    """

    def __init__(self, input_dim: int, embed_dim: int, n_attn_heads: int):
        super().__init__()

        # Verifying that the embedding dimension is right 
        assert embed_dim % n_attn_heads == 0, "Wrong embedding dimension : not divisible by n_attn_heads"

        self.embed_dim = embed_dim
        self.input_dim = input_dim
        self.n_attn_heads = n_attn_heads
        self.head_dim = embed_dim // n_attn_heads

        # Stacking all weights matrices together to be more efficient
        self.in_proj = nn.Linear(input_dim, 3*embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, input_dim, bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        """
            Resests the parameters of the multi-attention module
            and initialize the weights with Glorot initialization
        """
        nn.init.xavier_uniform_(self.in_proj.weight)
        self.in_proj.bias.data.fill_(0.)
        nn.init.xavier_uniform_(self.out_proj.weight)
        self.out_proj.bias.data.fill_(0.)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """
            Scaled dot product attention computation, using queries, 
            key and values (and optional mask).
            
            Parameters:
            -----------
            q: torch.Tensor, shape [batch_size, n_attn_heads, seq_len, head_dim]
                The queries
            k: torch.Tensor, shape [batch_size, n_attn_heads, seq_len, head_dim]
                The keys
            v: torch.Tensor, shape [batch_size, n_attn_heads, seq_len, head_dim]
                The values
            mask: torch.Tensor, shape [batch_size, seq_len]
                The mask to apply to the attention

            Returns:
            --------
            values: torch.Tensor, shape [batch_size, n_attn_heads, seq_len, head_dim]
                The output of the attention 
            attention: torch.Tensor, shape [batch_size, n_attn_heads, seq_len, seq_len]
                The attention matrix
        """
        d_k = q.size()[-1]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        values = torch.matmul(attention, v)
        return values, attention 


    def forward(self, x, mask=None, return_attention=False):
        """
            Forward pass of the module, computes the multi-head
            attention

            Parameters:
            -----------
            x: torch.Tensor, shape [batch_size, seq_len, input_dim]
                The input to the module
            mask: torch.Tensor, shape [batch_size, seq_len]
                The mask to apply to the attention
            return_attention: bool
                Whether to return the attention or not

            Returns:
            --------
            torch.Tensor, shape [batch_size, seq_len, input_dim]
                The output of the module
            torch.Tensor, shape [batch_size, n_attn_heads, seq_len, seq_len]
        """
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.in_proj(x)

        # Separate Q, K, V from linear output
        qkv = rearrange(qkv, 'b n (h d qkv) -> (qkv) b h n d', h=self.n_attn_heads, qkv=3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Determine value outputs
        values, attention = self.scaled_dot_product_attention(q, k, v, mask=mask)
        values = rearrange(values, 'b n s h -> b s (n h)', n=self.n_attn_heads, h=self.head_dim)
        o = self.out_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


class MultiLayerPerceptronBlock(nn.Module):
    """
        The fully connected block of the ViT encoder
        network where the data is passed before the output.

        Parameters:
        -----------
        config: dict
            The configuration of the model
    """
    def __init__(self, config: dict):
        super().__init__()
        self.conf = config

        self.dropout = nn.Dropout(self.conf['dropout'])
        self.act = nn.GELU()
        self.linear_1 = nn.Linear(self.conf['emb_size'], self.conf['emb_size'] * self.conf['fc_exp'])
        self.linear_2 = nn.Linear(self.conf['emb_size']*self.conf['fc_exp'], self.conf['emb_size'])
        
    def forward(self, x):
        """
            The forward pass of the fully connected block
            where the output of the multi-head attention

            Parameters:
            -----------
            x: torch.Tensor
                The input tensor
            
            Returns:
            --------
            torch.Tensor
                The output of the fully connected block
        """
        x = self.linear_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x
