"""
    ./fusion_model.py

        Contains definition for the FusionBottom module which is the 
        bottleneck of our network.

    Author: William MAILLET 
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders import VanillaEncoder

class FusionBottleneck(nn.Module):
    """
        The fusion bottlebeck structure that will take as input the two (14, 14) maps 
        given as input by the ViT & Encoder networks 

        Parameters:
        -----------
        config: dict 
            The config of the network
    """
    def __init__(self, config):
        super().__init__()
        self.module_name = "FusionBottleneck"
        self.gconf = config['global_config']
        self.fconf = config['fusion_config']
        self.device = self.gconf['device']

        # Input net 
        self.input_net = nn.Sequential(
            nn.Linear(self.fconf['input_dim'], self.fconf['model_dim']),
        )
        
        # The encoder
        self.encoder = VanillaEncoder(config, 'fusion')

        # The output net 
        self.in_shape = (self.fconf['input_dim']*2, self.fconf['input_dim'])
        self.out_net = nn.Sequential(
            nn.Dropout(0.1),
            nn.Flatten(),
            nn.Linear(self.fconf['model_dim']*self.in_shape[0], self.fconf['output_dim']),
        )
    
    def norm_cat(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
            Normalizes vectors and then concatenates them

            Parameters:
            -----------
            x: torch.Tensor
                The first vector
            y: torch.Tensor
                The second vector

            Returns:
            --------
            torch.Tensor
                The concatenated vector
        """
        nx = (x - x.min()) / (x.max() - x.min())
        ny = (y - y.min()) / (y.max() - y.min())
        return torch.cat((nx, ny), dim=1).float()

    def forward(self, x_weather: torch.Tensor, x_vit: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
            The forward pass of the fusion bottleneck

            Parameters:
            -----------
            x_weather: torch.Tensor
                The output of the weather encoder network 
            x_vit: torch.Tensor
                The output of the ViT encoder network 
            mask: torch.Tensor
                The mask to be applied to the encoder if needed

            Returns:
            --------
            torch.Tensor
                The final tensor of the fusion bottleneck which
                dimension is yet to be defined 
        """
        # Concatenating x_weather and x_vit (normalizing them before)
        x = self.norm_cat(x_weather, x_vit)
        x = self.input_net(x)
        x = self.encoder(x, mask)
        x = self.out_net(x)
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
        for layer in self.named_children():
            print(f"{layer[0]:<20} | {sum(p.numel() for p in layer[1].parameters() if p.requires_grad):,} params. ({round(layer[1].model_size(), 3)} MB)")
        print("-------------------------------------------------------")
        print(f"Total number of parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        print("Size of the model: {:.3f} MB".format(self.model_size()))
        print("-------------------------------------------------------")