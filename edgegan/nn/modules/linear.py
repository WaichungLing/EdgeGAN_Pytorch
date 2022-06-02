import torch
import torch.nn as nn

from edgegan.nn.modules import spectral_normed_weight


class FullyConnected(nn.Module):
    def __init__(self, sn, num_inputs, num_outputs):
        super(FullyConnected, self).__init__()
        self.sn = sn
        self.mlp = torch.nn.Linear(num_inputs, num_outputs)

    def forward(self, input):
        if self.sn:
            self.mlp.weight = nn.Parameter(spectral_normed_weight(self.mlp.weight, num_iters=1))

        linear_out = self.mlp(input)

        return linear_out
