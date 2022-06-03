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
            if self.mlp.weight.is_cuda:
                pre_weight = self.mlp.weight.cpu()
                sn_weight = spectral_normed_weight(pre_weight, num_iters = 1)
                sn_weight = sn_weight.cuda()
                self.mlp.weight = nn.Parameter(sn_weight)
            else:
                self.mlp.weight = nn.Parameter(spectral_normed_weight(self.mlp.weight, num_iters=1))

        linear_out = self.mlp(input)

        return linear_out
