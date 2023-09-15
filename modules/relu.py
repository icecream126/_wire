#!/usr/bin/env python

import pdb
import math
from math import pi, ceil

import numpy as np

import torch
from torch import nn

from .utils import build_montage, normalize, PosEncoding
    
class ReLULayer(nn.Module):
    '''
        Drop in replacement for SineLayer but with ReLU non linearity
    '''
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, scale=10.0):
        '''
            is_first, and omega_0 are not used.
        '''
        super().__init__()
        self.in_features = in_features
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, input):
        return nn.functional.relu(self.linear(input))
    
class INR(nn.Module):
    def __init__(self, posenc_freq, in_features,
                 hidden_features, hidden_layers, 
                 out_features, skip, **kwargs):
        super().__init__()
        self.skip = skip
        self.hidden_layers = hidden_layers
        self.posenc = PosEncoding(in_features=in_features, num_frequencies=posenc_freq)
        self.posenc_dim = 2 * posenc_freq * in_features + in_features

        self.nonlin = ReLULayer

        self.net = nn.ModuleList()
        self.net.append(self.nonlin(self.posenc_dim, hidden_features))

        for i in range(hidden_layers):
            if self.skip and i == ceil(hidden_layers / 2):
                self.net.append(self.nonlin(hidden_features + self.posenc_dim, hidden_features))
            else:
                self.net.append(self.nonlin(hidden_features, hidden_features))

        final_linear = nn.Linear(hidden_features, out_features)

        self.net.append(final_linear)
    
    def forward(self, x):
        x = self.posenc(x)
        x_in = x
        for i, layer in enumerate(self.net):
            if self.skip and i == ceil(self.hidden_layers / 2) + 1:
                x = torch.cat([x, x_in], dim=-1)
            x = layer(x)
        return x