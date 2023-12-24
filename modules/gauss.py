import torch
from torch import nn
from math import ceil

from .utils import GaussEncoding


class GaussLayer(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        **kwargs,
    ):
        super().__init__()

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, input):
        return nn.functional.relu(self.linear(input))


class INR(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        hidden_features,
        hidden_layers,
        skip,
        gauss_scale,
        mapping_size=256,
        **kwargs,
    ):
        super().__init__()

        self.skip = skip
        self.hidden_layers = hidden_layers
        self.posenc = GaussEncoding(in_features=in_features, mapping_size = mapping_size, scale=gauss_scale)
        # import pdb
        # pdb.set_trace()
        self.posenc_dim = 2 * mapping_size

        self.nonlin = GaussLayer

        self.net = nn.ModuleList()
        self.net.append(self.nonlin(self.posenc_dim, hidden_features))

        for i in range(hidden_layers):
            if skip and i == ceil(hidden_layers / 2):
                self.net.append(self.nonlin(hidden_features + self.posenc_dim, hidden_features))
            else:
                self.net.append(self.nonlin(hidden_features, hidden_features))

        final_linear = nn.Linear(hidden_features, out_features)

        self.net.append(final_linear)

    def forward(self, x):
        # import pdb
        # pdb.set_trace()
        x = self.posenc(x)
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