import torch
import torch.nn.functional as F
import torch.nn as nn


def activation_fn(input, name='lrelu'):
    assert name in ['relu', 'lrelu', 'tanh', 'sigmoid', None]
    if name == 'relu':
        return F.relu(input)
    elif name == 'lrelu':
        return torch.maximum(input, 0.2*input)
    elif name == 'tanh':
        return F.tanh(input)
    elif name == 'sigmoid':
        return F.sigmoid(input)
    else:
        return input


def miu_relu(x, miu=0.7):
    return (x + torch.sqrt((1 - miu) ** 2 + x ** 2)) / 2.


def lrelu(x, leak=0.2):
    return torch.maximum(leak * x, x)


class Prelu(nn.Module):
    def __init__(self, leak):
        super(Prelu, self).__init__()
        self.leak = nn.Parameter(torch.Tensor([leak]))

    def forward(self, x):
        return torch.maximum(self.leak * x, x)
