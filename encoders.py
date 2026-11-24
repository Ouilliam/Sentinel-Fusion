"""
    ./encoders.py

        Contains definition for multiple types of encoders and their "encoder-blocks" (each block is an encoder layer).

    Author: William MAILLET
"""
import numpy as np
import torch
import torch.nn as nn

from layers import MultiHeadAttention, MultiLayerPerceptronBlock


class VanillaEncoderBlock(nn.Module):
    """
        An Encoder block, which is a sequence of layers
        using multi-head attention and feed forward layers

        Code adapted from:
            uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html

        Parameters:
        -----------
        config: dict
            The global network configuration dictionary
            Used parameters:
                -> model_dim: int
                    The dimension of the model
                -> n_attn_heads: int
                    The number of heads for the multi-head attention
                -> ff_dim: int
                    The dimension of the feed forward layers
                -> dropout: float
                    The dropout rate
    """

    def __init__(self, config, e_type: str):
        super().__init__()
        self.config = config[f'{e_type}_config']
        self.gconf  = config['global_config']
        self.module_name = "VanillaEncoderBlock"
        model_dim, n_attn_heads, ff_dim, dropout = self.config['model_dim'], self.config['n_attn_heads'], self.config['ff_dim'], config['global_config']['dropout']
        
        # Attention layer
        self.attn = MultiHeadAttention(model_dim, model_dim, n_attn_heads)

        # Two layers MLP
        self.linear_net = nn.Sequential(
            nn.Linear(model_dim, ff_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(ff_dim, model_dim)
        )

        # Normalization & dropout layers
        self.norm_1 = nn.LayerNorm(model_dim)
        self.norm_2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
            Forward pass of the module, computes the multi-head 
            attention, layer normalization, dropout and uses the 
            two layers MLP to compute the output

            Parameters:
            -----------
            x: torch.Tensor, shape [batch_size, seq_len, input_dim]
                The input to the module
            mask: torch.Tensor, shape [batch_size, seq_len]
                The mask to apply to the attention
            
            Returns:
            --------
            torch.Tensor, shape [batch_size, seq_len, input_dim]    
                The output of the encoder block
        """
        attn_out = self.attn(x, mask=mask)
        x = self.norm_1(x + self.dropout(attn_out))

        # Feed forward
        x = self.linear_net(x)
        x = self.norm_2(x + self.dropout(x))
        
        return x

    def model_size(self):
        """
            Function that returns the size of the model in MB.

            Returns:
            --------
            size: float
                The size of the model in MB
        """
        param_size, buff_size = 0, 0
        for p in self.parameters():
            param_size += p.nelement() * p.element_size()
        for p in self.buffers():
            buff_size += p.nelement() * p.element_size()

        return round((param_size + buff_size) / (1024**2), 3)
    
    def summary(self):
        """
            Function that returns a summary of the model.

            The summary details the number of parameters, their name,
            as well as the size of the model in MB.
        """
        print(f'\n{self.module_name} summary:')
        print("-------------------------------------------------------")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"{name: <35} | {param.numel():<9,} params.")
        print("-------------------------------------------------------")
        print(f"Total number of parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        print("Size of the model: {:.3f} MB".format(self.model_size()))
        print("-------------------------------------------------------")


class VanillaEncoder(nn.Module):
    """
        A simple PyTorch Encoder using the multi-head attention 
        for our weather data.

        Parameters:
        -----------
        config: dict
            The global network configuration dictionary
            Used parameters:
                -> n_layers: int
                    The number of layers in the encoder
    """
    def __init__(self, config, e_type: str=  'wam'):
        super().__init__()
        self.module_name = "VanillaEncoder"
        self.layers = nn.ModuleList([
            VanillaEncoderBlock(config, e_type=e_type) for _ in range(config[f"{e_type}_config"]['n_layers'])
        ])

    def forward(self, x, mask=None):
        """
            Forward pass of the module, applies the encoder blocks 
            to the input

            Parameters:
            -----------
            x: torch.Tensor, shape [batch_size, seq_len, input_dim]
                The input to the module
            mask: torch.Tensor, shape [batch_size, seq_len]
                The mask to apply to the attention
            
            Returns:
            --------
            torch.Tensor, shape [batch_size, seq_len, input_dim]    
                The output of the encoder block
        """
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x
    
    def get_attn(self, x: torch.Tensor, mask: torch.Tensor =None):
        """
            Returns the attention map of an input sequence

            Parameters:
            -----------
            x: torch.Tensor, shape [batch_size, seq_len, input_dim]
                The input to the module
            mask: torch.Tensor, shape [batch_size, seq_len]
                The mask to apply to the attention
            
            Returns:
            --------
            list of torch.Tensor, shape [batch_size, seq_len, input_dim]
                The output of the encoder block attention
        """
        attention_maps = []
        for layer in self.layers:
            _, attn_map = layer.attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map.detach().cpu().numpy())
            x = layer(x, mask=mask)
        return np.array(attention_maps).squeeze(axis=1)

    def model_size(self):
        """
            Function that returns the size of the model in MB.

            Returns:
            --------
            size: float
                The size of the model in MB
        """
        param_size, buff_size = 0, 0
        for p in self.parameters():
            param_size += p.nelement() * p.element_size()
        for p in self.buffers():
            buff_size += p.nelement() * p.element_size()

        return round((param_size + buff_size) / (1024**2), 3)
    
    def summary(self):
        """
            Function that returns a summary of the model.

            The summary details the number of parameters, their name,
            as well as the size of the model in MB.
        """
        print(f'\n{self.module_name} summary:')
        print("-------------------------------------------------------")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"{name: <35} | {param.numel():<9,} params.")
        print("-------------------------------------------------------")
        print(f"Total number of parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        print("Size of the model: {:.3f} MB".format(self.model_size()))
        print("-------------------------------------------------------")


class ViTEncoderBlock(nn.Sequential):
    """
        The transformer encoder block
        that uses the residual connections

        Parameters:
        -----------
        config: dictionary
            The configuration of the encoder
    """

    def __init__(self, config):
        super().__init__()
        self.conf = config
        self.module_name = 'ViTEncoderBlock'

        # Layer normalization layers
        self.ln_1 = nn.LayerNorm(self.conf['emb_size'])
        self.ln_2 = nn.LayerNorm(self.conf['emb_size'])
        # Attention layer
        self.self_attention = MultiHeadAttention(self.conf['emb_size'], self.conf['emb_size'], self.conf['n_attn_heads'])
        # Multilayer perceptron
        self.mlp = MultiLayerPerceptronBlock(self.conf)
        # Dropout
        self.dropout = nn.Dropout(self.conf['dropout'])

    def forward(self, x, return_attention=False):
        """
            The forward pass of the encoder block.
            First, the multi-head attention is applied to the normalized input and then it is 
            passed to a fully connected network.

            Parameters:
            -----------
            x: torch.Tensor
                The input to the encoder block
            
            Returns:
            --------
            torch.Tensor
                The output of the encoder block
        """

        # First part : Multi-head attention
        z = self.ln_1(x)
        if return_attention:
            z, attention = self.self_attention(z, return_attention=True)
        else:
            z = self.self_attention(z)
        z = self.dropout(z) + x

        # Second part : Fully connected block
        out = self.ln_2(z)
        out = self.mlp(out)
        out = self.dropout(out) + z 

        if return_attention:
            return out, attention
        else:
            return out

    def model_size(self):
        """
            Function that returns the size of the model in MB.

            Returns:
            --------
            size: float
                The size of the model in MB
        """
        param_size, buff_size = 0, 0
        for p in self.parameters():
            param_size += p.nelement() * p.element_size()
        for p in self.buffers():
            buff_size += p.nelement() * p.element_size()

        return round((param_size + buff_size) / (1024**2), 3)
    
    def summary(self):
        """
            Function that returns a summary of the model.

            The summary details the number of parameters, their name,
            as well as the size of the model in MB.
        """
        print(f'\n{self.module_name} summary:')
        print("-------------------------------------------------------")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"{name: <35} | {param.numel():<9,} params.")
        print("-------------------------------------------------------")
        print(f"Total number of parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        print("Size of the model: {:.3f} MB".format(self.model_size()))
        print("-------------------------------------------------------")

class ViTEncoder(nn.Module):
    """
        The complete transformer encoder
        using transformer encoder blocks

        Parameters:
        -----------
        config: dict
            The configuration dictionary
    """

    def __init__(self, config):
        super().__init__()
        self.conf = config
        self.module_name = 'ViTEncoder'
        
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1,(self.conf['img_size'] // self.conf['patch_size']) ** 2 + 1, self.conf['emb_size']), requires_grad=True)

        # Encoder blocks
        self.layers = nn.ModuleList()
        for i in range(self.conf['n_layers']):
            self.layers.add_module("encoder_layer_{}".format(i), ViTEncoderBlock(self.conf))

    def forward(self, x, return_attention=False):
        """
            The forward pass of the encoder
            where the input is passed through the encoder blocks

            Parameters:
            -----------
            x: torch.Tensor
                The input to the encoder
            
            Returns:
            --------
            torch.Tensor
                The output of the encoder
        """

        # Positional embedding
        pos = self.pos_embedding.repeat(x.shape[0], 1, 1)
        x = x + pos

        # Layers 
        for i, layer in enumerate(self.layers):
            if return_attention:
                x, attention = layer(x, return_attention=return_attention)
            else:
                x = layer(x)

        if return_attention:
            return x, attention
        else:
            return x   


    def model_size(self):
        """
            Function that returns the size of the model in MB.

            Returns:
            --------
            size: float
                The size of the model in MB
        """
        param_size, buff_size = 0, 0
        for p in self.parameters():
            param_size += p.nelement() * p.element_size()
        for p in self.buffers():
            buff_size += p.nelement() * p.element_size()

        return round((param_size + buff_size) / (1024**2), 3)
    
    def summary(self):
        """
            Function that returns a summary of the model.

            The summary details the number of parameters, their name,
            as well as the size of the model in MB.
        """
        print(f'\n{self.module_name} summary:')
        print("-------------------------------------------------------")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"{name: <35} | {param.numel():<9,} params.")
        print("-------------------------------------------------------")
        print(f"Total number of parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        print("Size of the model: {:.3f} MB".format(self.model_size()))
        print("-------------------------------------------------------")