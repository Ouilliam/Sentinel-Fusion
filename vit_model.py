"""
    ./vit_model.py

        Contains the torch implementation of the ViT model, modified so that he 
        can output the attention map instead of the usual features.

    Author: William MAILLET 
"""
import torch
import torch.hub
import torch.nn as nn
import torchvision
from einops import repeat
from einops.layers.torch import Rearrange
from einops import rearrange

from encoders import ViTEncoder

class ViT(nn.Module):
    """
        The ViT model modified so that he can output the 
        attention map instead of the usual features.

        Parameters:
        -----------
        config: dict 
            The configuration dictionary
    """

    def __init__(self, config):
        super().__init__()
        self.conf = config['vit_config']
        self.gconf = config['global_config']
        self.module_name = 'ViT (custom)'

        # Patch embedding part
        self.class_token = nn.Parameter(torch.randn(1, 1, self.conf['emb_size']), requires_grad=True)

        self.conv_proj = nn.Conv2d(
                in_channels=self.conf['n_channels'],
                out_channels=self.conf['emb_size'],
                kernel_size=self.conf['patch_size'],
                stride=self.conf['patch_size']
            )        

        self.encoder = ViTEncoder(self.conf)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.conf['emb_size']*197, 196),
            Rearrange('b (h w) -> b h w', h=14, w=14),
        )

    def load_pretrained_weights(self):
        """
            Methods that loads pretrained weights for the ViT model from
            the torchvision pretrained model.
        """
        # Load the pretrained weights and the model weights
        pretrained = torchvision.models.vit_b_16(pretrained=True).state_dict()
        model_dict = self.state_dict()
    
        # Trying (very bad code) to load the pretrained weights to the right place 
        for k, v in pretrained.items():
            if not ("heads" in k or "encoder.ln" in k):
                if 'conv_proj' in k and self.conf['n_channels'] != 3:
                    continue
                if 'in_proj_weight' in k:
                    nk = k.replace('in_proj_weight', 'in_proj.weight')
                    model_dict[nk] = v
                elif 'in_proj_bias' in k:
                    nk = k.replace('in_proj_bias', 'in_proj.bias')
                    model_dict[nk] = v
                else:
                    model_dict[k] = pretrained[k]

        # Loading and activating gradient
        self.load_state_dict(model_dict)
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        """
            The forward pass of our custom ViT
            The image is embedded as patches and then passed into the transformer made 
            of 12 encoders. The image is then classified.

            Parameters:
            -----------
            x: torch.Tensor
                The input tensor
            
            Returns:
            --------
            out: torch.Tensor
                The output tensor
        """
        batch_size = x.shape[0]
        
        # Project the image to patches
        x = self.conv_proj(x)
        x = rearrange(x, 'b e (h) (w) -> b (h w) e')
        # Create class token
        class_token = repeat(self.class_token, '() n e -> b n e ', b=batch_size)
        # Append class token
        x = torch.cat((x, class_token), dim=1)

        # Encoder forward pass 
        x = self.encoder(x)

        # Classifier 
        out = self.classifier(x)

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
            print(f'{name:<25} {param.shape} {param.numel()} params.')
        print("-------------------------------------------------------")
        print(f"Total number of parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")
        print("Size of the model: {:.3f} MB".format(self.model_size()))
        print("-------------------------------------------------------")

    def get_attn(self, x):
        """
            Returns the attention of the model on the image.

            Parameters:
            -----------
            img: torch.Tensor
                The input image (shape (batch, 3, 224, 224))
            
            Returns:
            --------
            attn: torch.Tensor
                The attention map (shape (1, 1, 224, 224))
            
        """
        batch_size = x.shape[0]
        # Project the image to patches
        x = self.conv_proj(x)
        x = rearrange(x, 'b e (h) (w) -> b (h w) e')
        # Create class token
        class_token = repeat(self.class_token, '() n e -> b n e ', b=batch_size)
        # Append class token
        x = torch.cat((x, class_token), dim=1)

        # Encoder forward pass 
        x, attn_list = self.encoder(x, return_attention=True)

        return attn_list