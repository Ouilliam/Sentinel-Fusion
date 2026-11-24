"""
    ./sentinel_model.py 

        Contains definition for the SentinelFusion model.
    
    Author: William MAILLET
"""
import nopdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from weather_model import WeatherAttentionModel
from vit_model import ViT 
from fusion_model import FusionBottleneck


class SentinelFusion(nn.Module):
    """
        The Multimodal Sentinel Fusion network. 
        This network takes images and weather as an input, and then :
            - Generates an attention map of the images thanks to the pretrained ViT model
            - Generates attention features of the weather thanks to an encoder
            - Concatenates the attention features and the weather features
            - Passes the concatenated features through a fusion bottleneck
            - Outputs [1, 0] or [0, 1]

        Parameters:
        ----------- 
        config: dict 
            The configuration dictionary of the network
    """
    
    def __init__(self, config:dict):
        super().__init__()
        self.config = config 
        self.device = self.config['global_config']['device']

        # ViT pretrained model 
        self.vit_model = ViT(config)
        self.vit_model.load_pretrained_weights()

        # Weather Attention Model
        self.wa_model = WeatherAttentionModel(self.config)

        # Fusion Encoder
        self.fusion_bottleneck = FusionBottleneck(self.config)

        # Summary purposes
        self.module_name = "SentinelFusion"

    def forward(self, images_vector: torch.Tensor, weather_vector: torch.Tensor):
        """
            Forward pass of the model.
            
            First, the ViT model gives us a 14x14 attention map. The WeatherAttentionModel
            gives us a 14x14 output that we then pass to the fusion bottleneck with 
            the attention map as input.

            Parameters:
            -----------
            weather_vector: torch.Tensor
                The weather vector. Intended shape: [batch, 1, features]
            images_vector: torch.Tensor
                The images vector. Intended shape: [batch, 1, n_channels, height, width]

            Returns:
            --------
            torch.Tensor
                The output of the fusion network. Intended shape: [batch, 1, n_channels, height, width]
        """
        # Getting everything on GPU
        weather_vector, images_vector = weather_vector.to(self.device), images_vector.to(self.device)

        # Forward pass of the ViT model
        vit_feat = self.vit_model(images_vector)

        # Forward pass of the weather attention model
        weather_feat = self.wa_model(weather_vector)

        # Forward pass of the fusion bottleneck
        fusion_output = self.fusion_bottleneck(weather_feat, vit_feat)
        
        return fusion_output

    def get_bottleneck_attn(self, weather_vector: torch.Tensor, images_vector: torch.Tensor):
        """
            A method that returns us the attention map of the fusion bottleneck
            from the weather and image inputs

            Parameters:
            -----------
            weather_vector: torch.Tensor
                The weather vector. Intended shape: [batch, 1, features]
            images_vector: torch.Tensor
                The images vector. Intended shape: [batch, 1, n_channels, height, width]
            
            Returns:
            --------
            torch.Tensor
                The attention map of the fusion bottleneck. Intended shape: [batch, 1, 14, 14]
        """
        # Verifying that the batch size is the same
        assert weather_vector.shape[0] == images_vector.shape[0]
        attn_dim = self.config['fusion_config']['model_dim'] * 2 
        enc_layers = self.config['fusion_config']['n_layers']
        n_attn_heads = self.config['fusion_config']['n_attn_heads']

        # Storing attention maps 
        attention_maps = np.zeros((weather_vector.shape[0], enc_layers, n_attn_heads, attn_dim, attn_dim))

        for b in range(images_vector.shape[0]):
            # Get features
            weather_feat = self.wa_model(weather_vector[b].unsqueeze(0))
            vit_attn_maps = self.get_vit_attn(images_vector[b].unsqueeze(0))

            # Concatenate the weather features and the attention maps
            cat = self.fusion_bottleneck.norm_cat(weather_feat, vit_attn_maps)
            attention_maps[b, :, :] = self.fusion_bottleneck.encoder.get_attn(self.fusion_bottleneck.input_net(cat))

        return attention_maps

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