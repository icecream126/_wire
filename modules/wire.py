import torch
from torch import nn
from math import ceil


class GaborLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        omega,
        sigma,
        **kwargs,
    ):
        super().__init__()

        self.omega = omega
        self.sigma = sigma

        self.freqs = nn.Linear(in_features, out_features)
        self.scale = nn.Linear(in_features, out_features)

    def forward(self, input):
        omega = self.omega * self.freqs(input)
        scale = self.scale(input) * self.sigma

        freq_term = torch.cos(omega)
        gauss_term = torch.exp(-(scale**2))
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

        self.nonlin = GaborLayer

        self.net = nn.ModuleList()
        self.net.append(self.nonlin(in_features, hidden_features, omega=omega, sigma=sigma))

        for i in range(hidden_layers):
            if skip and i == ceil(hidden_layers / 2):
                self.net.append(
                    self.nonlin(
                        hidden_features + in_features, hidden_features, omega=omega, sigma=sigma
                    )
                )
            else:
                self.net.append(
                    self.nonlin(hidden_features, hidden_features, omega=omega, sigma=sigma)
                )

        final_linear = nn.Linear(hidden_features, out_features)

        self.net.append(final_linear)

    def forward(self, x):
        x_in = x
        for i, layer in enumerate(self.net):
            if self.skip and i == ceil(self.hidden_layers / 2) + 1:
                x = torch.cat([x, x_in], dim=-1)
            x = layer(x)
        return x