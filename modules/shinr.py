import torch
from torch import nn
from math import ceil

from .relu import ReLULayer

from .utils import components_from_spherical_harmonics


class SphericalHarmonicsLayer(nn.Module):
    def __init__(
        self,
        levels,
        omega,
        **kwargs,
    ):
        super().__init__()

        self.levels = levels
        self.hidden_features = levels**2
        self.omega = omega


    def forward(self, input):
        out = components_from_spherical_harmonics(self.levels, input[..., :3])
        return out


class INR(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        hidden_features,
        hidden_layers,
        levels,
        skip,
        omega,
        **kwargs,
    ):
        super().__init__()

        self.skip = skip
        self.hidden_layers = hidden_layers

        self.first_nonlin = SphericalHarmonicsLayer

        self.net = nn.ModuleList()
        self.net.append(self.first_nonlin(levels, omega))

        self.nonlin = ReLULayer

        for i in range(hidden_layers):
            if i == 0:
                self.net.append(self.nonlin(levels**2, hidden_features))
            elif skip and i == ceil(hidden_layers / 2):
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