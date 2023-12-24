import torch
from torch import nn
from math import pi, ceil

from .relu import ReLULayer


class SphericalGaborLayer(nn.Module):
    def __init__(
        self,
        out_features,
        wavelet_dim,
        omega_0,
        sigma_0,
        freq_enc_type,
        **kwargs,
    ):
        super().__init__()
        
        self.freq_enc_type=freq_enc_type

        self.wavelet_dim = wavelet_dim
        self.out_features = out_features
        
        self.omega = nn.Parameter(torch.empty(1, wavelet_dim))
        self.sigma = nn.Parameter(torch.empty(1, wavelet_dim))
        nn.init.normal_(self.omega)
        nn.init.normal_(self.sigma)
        self.omega_0 = omega_0
        self.sigma_0 = sigma_0        

        self.dilate = nn.Parameter(torch.empty(1, wavelet_dim))
        nn.init.normal_(self.dilate)

        self.u = nn.Parameter(torch.empty(wavelet_dim))
        self.v = nn.Parameter(torch.empty(wavelet_dim))
        self.w = nn.Parameter(torch.empty(wavelet_dim))
        nn.init.uniform_(self.u)
        nn.init.uniform_(self.v)
        nn.init.uniform_(self.w)
        
        self.out_linear = nn.Linear(wavelet_dim, out_features)


    def forward(self, input):
        zeros = torch.zeros(self.wavelet_dim, device=self.u.device)
        ones = torch.ones(self.wavelet_dim, device=self.u.device)
        
        alpha = 2 * pi * self.u
        beta = torch.arccos(torch.clamp(2 * self.v - 1, -1 + 1e-6, 1 - 1e-6))
        gamma = 2 * pi * self.w

        cos_alpha = torch.cos(alpha)
        cos_beta = torch.cos(beta)
        cos_gamma = torch.cos(gamma)
        sin_alpha = torch.sin(alpha)
        sin_beta = torch.sin(beta)
        sin_gamma = torch.sin(gamma)

        Rz_alpha = torch.stack(
            [
                torch.stack([cos_alpha, -sin_alpha, zeros], 1),
                torch.stack([sin_alpha, cos_alpha, zeros], 1),
                torch.stack([zeros, zeros, ones], 1),
            ],
            1,
        )

        Rx_beta = torch.stack(
            [
                torch.stack([ones, zeros, zeros], 1),
                torch.stack([zeros, cos_beta, -sin_beta], 1),
                torch.stack([zeros, sin_beta, cos_beta], 1),
            ],
            1,
        )

        Rz_gamma = torch.stack(
            [
                torch.stack([cos_gamma, -sin_gamma, zeros], 1),
                torch.stack([sin_gamma, cos_gamma, zeros], 1),
                torch.stack([zeros, zeros, ones], 1),
            ],
            1,
        )

        R = torch.bmm(torch.bmm(Rz_gamma, Rx_beta), Rz_alpha)

        points = input[..., 0:3]
        points = torch.matmul(R, points.unsqueeze(-2).unsqueeze(-1))
        points = points.squeeze(-1)

        x, z = points[..., 0], points[..., 2]

        dilate = torch.exp(self.dilate) 

        freq_arg = 2 * dilate * x / (1e-6 + 1 + z)
        gauss_arg = 4 * dilate * dilate * (1 - z) / (1e-6 + 1 + z)


        if self.freq_enc_type=='cos':
            freq_term = torch.cos(self.omega_0 * self.omega * freq_arg)
        else:
            freq_term = torch.sin(self.omega_0 * self.omega * freq_arg)
        
        gauss_term = torch.exp(-self.sigma * self.sigma * self.sigma_0 * gauss_arg)

        out = freq_term * gauss_term
        # return out
        return nn.functional.relu(self.out_linear(out))


class INR(nn.Module):
    def __init__(
        self,
        in_features,
        out_features, 
        wavelet_dim,
        hidden_features,
        hidden_layers,
        skip,
        omega_0,
        sigma_0,
        freq_enc_type,
        **kwargs,
    ):
        super().__init__()

        self.skip = skip
        self.hidden_layers = hidden_layers

        self.first_nonlin = SphericalGaborLayer

        self.net = nn.ModuleList()
        self.net.append(self.first_nonlin(hidden_features, wavelet_dim, omega_0, sigma_0, freq_enc_type))

        self.nonlin = ReLULayer

        for i in range(hidden_layers):
            # if i==0:
                # self.net.append(self.nonlin(wavelet_dim, hidden_features))
            if skip and i == ceil(hidden_layers / 2):
                self.net.append(self.nonlin(hidden_features + in_features, hidden_features))
            else:
                self.net.append(self.nonlin(hidden_features, hidden_features))

        final_linear = nn.Linear(hidden_features, out_features)

        self.net.append(final_linear)

    def forward(self, x):
        x_in = x
        for i, layer in enumerate(self.net):
            if self.skip and i == ceil(self.hidden_layers / 2) + 1:
                x = torch.cat([x, x_in], dim=-1)
            x = layer(x)
        return x
    
    def get_optimizer_parameters(self, weight_decay):
        # Parameters of the last two layers
        decay_parameters = list(self.net[-1].parameters()) + list(self.net[-2].parameters())

        # Parameters of the remaining layers
        no_decay_parameters = [
            p for n, p in self.named_parameters() if not any(
                nd in n for nd in ["net." + str(len(self.net) - 1), "net." + str(len(self.net) - 2)]
            )
        ]

        return [
            {'params': no_decay_parameters, 'weight_decay': 0.0},
            {'params': decay_parameters, 'weight_decay': weight_decay}
        ]