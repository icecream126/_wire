import torch
from torch import nn
from math import pi, ceil

from .relu import ReLULayer


class SphericalGaborLayer(nn.Module):
    def __init__(
        self,
        out_features,
        omega,
        sigma,
        **kwargs,
    ):
        super().__init__()

        self.omega = omega
        self.sigma = sigma
        self.out_features = out_features

        self.dilate = nn.Parameter(torch.empty(1, out_features))
        nn.init.normal_(self.dilate)

        self.u = nn.Parameter(torch.empty(out_features))
        self.v = nn.Parameter(torch.empty(out_features))
        self.w = nn.Parameter(torch.empty(out_features))
        nn.init.uniform_(self.u)
        nn.init.uniform_(self.v)
        nn.init.uniform_(self.w)


    def forward(self, input):
        zeros = torch.zeros(self.out_features, device=self.u.device)
        ones = torch.ones(self.out_features, device=self.u.device)

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

        # points.shape : [ 1,2097125,3 ] / [512, 3]
        # inputs.shape : [ 1,2097125,3] / [512, 3]
        # R.shape : [256, 3, 3] / [256, 3, 3]
        points = input[..., 0:3]
        points = torch.matmul(R, points.unsqueeze(-2).unsqueeze(-1))
        points = points.squeeze(-1)

        x, z = points[..., 0], points[..., 2]

        dilate = torch.exp(self.dilate)

        freq_arg = 2 * dilate * x / (1e-6 + 1 + z)
        gauss_arg = 4 * dilate * dilate * (1 - z) / (1e-6 + 1 + z)

        freq_term = torch.cos(self.omega * freq_arg)
        gauss_term = torch.exp(-self.sigma * self.sigma * gauss_arg)
        return freq_term * gauss_term


class INR(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        hidden_features,
        hidden_layers,
        skip,
        omega,
        sigma,
        **kwargs,
    ):
        super().__init__()

        self.skip = skip
        self.hidden_layers = hidden_layers

        self.first_nonlin = SphericalGaborLayer

        self.net = nn.ModuleList()
        self.net.append(self.first_nonlin(hidden_features, omega, sigma))

        self.nonlin = ReLULayer

        for i in range(hidden_layers):
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