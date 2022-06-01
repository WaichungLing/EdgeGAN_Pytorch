import torch


def upsample(x):
    m = torch.nn.Upsample(scale_factor=4, mode='nearest')
    return m(x)


def upsample2(x):
    output = x.permute(0, 3, 1, 2)
    m = torch.nn.Upsample(scale_factor=4, mode='nearest')
    return m(output)
