"""
    ./weather_attention.py

        Contains definition for the WeatherAttentionModel.
    
    Author: William MAILLET 
"""
import torch 
import torch.nn as nn 
from encoders import VanillaEncoder 

class WeatherAttentionModel(nn.Module):
    """
        The weather network that will encode our weather 
        as a feature attention map

        Parameters:
        -----------
        config: dict
            The configuration dictionary of the model
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.module_name = "WeatherAttentionModel"
        self.wconf = config['wam_config']
        self.gconf = config['global_config']
        self.device = self.gconf['device']

        self.input_net = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.wconf['input_dim'], self.wconf['model_dim']),
        )

        self.encoder = VanillaEncoder(config, 'wam')

        # Create output net that output (batch_size, 14, 14)
        self.out_net = nn.ModuleList()
        self.out_net.add_module('dropout', nn.Dropout(0.1, inplace=True))
        self.out_net.add_module('flatten', nn.Flatten())
        self.out_net.add_module('linear', nn.Linear(self.wconf['model_dim']*7, self.wconf['output_dim']))

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
            Forward pass of the model, that computes
            the dimension expansion, positional encoding and 
            encodes the weather 

            Parameters:
            -----------
            x: torch.Tensor 
                The input 
            mask: torch.Tensor
                A mask that can be applied (used by the encoder)

            Returns:
            --------
            torch.Tensor
                The encoded weather
        """
        x = self.input_net(x)
        x = self.encoder(x, mask)
        for lay in self.out_net:
            x = lay(x)
        x = x.view(x.shape[0], *self.wconf['reshape'])
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
        for name, p in self.named_parameters():
            print(f'{name:<25} | {p.numel():,} params.')
        print("-------------------------------------------------------")
        print(f"Total number of parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        print("Size of the model: {:.3f} MB".format(self.model_size()))
        print("-------------------------------------------------------")