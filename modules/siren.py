#!/usr/bin/env python

import pdb
import math
from math import ceil

import numpy as np

import torch
from torch import nn

from .utils import build_montage, normalize
    
class SineLayer(nn.Module):
    '''
        See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for
        discussion of omega.
    
        If is_first=True, omega is a frequency factor which simply multiplies
        the activations before the nonlinearity. Different signals may require
        different omega in the first layer - this is a hyperparameter.
    
        If is_first=False, then the weights will be divided by omega so as to
        keep the magnitude of activations constant, but boost gradients to the
        weight matrix (see supplement Sec. 1.5)
    '''
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega=30, scale=10.0, init_weights=True):
        super().__init__()
        self.omega = omega
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega, 
                                             np.sqrt(6 / self.in_features) / self.omega)
        
    def forward(self, input):
        return torch.sin(self.omega * self.linear(input))
    
class INR(nn.Module):
    def __init__(self, in_features, hidden_features, 
                 hidden_layers, 
                 out_features,skip,omega,  **kwargs):
        super().__init__()
        self.nonlin = SineLayer
        self.skip =skip
        self.hidden_layers = hidden_layers
            
        self.net = nn.ModuleList()
        self.net.append(self.nonlin(in_features, hidden_features, 
                                  is_first=True, omega=omega))       

        for i in range(hidden_layers):
            if self.skip and i == ceil(hidden_layers / 2):
                self.net.append(self.nonlin(hidden_features+in_features, hidden_features, 
                                        is_first=False, omega=omega))
            else:
                self.net.append(
                    self.nonlin(hidden_features, hidden_features, is_first=False, omega=omega)
                )
        dtype = torch.float
        final_linear = nn.Linear(hidden_features,
                                     out_features,
                                     dtype=dtype)
        
        with torch.no_grad():
            const = np.sqrt(6/hidden_features)/max(omega, 1e-12)
            final_linear.weight.uniform_(-const, const)
                    
        self.net.append(final_linear)
        
    def forward(self, x):
        x_in = x
        for i, layer in enumerate(self.net):
            if self.skip and i == ceil(self.hidden_layers/2)+1:
                x = torch.cat([x, x_in], dim=-1)
            x = layer(x)
                    
        return x
  