import torch


def upsample(x, data_format):
    m = torch.nn.Upsample(scale_factor=4, mode='nearest')
    return m(x)


def upsample2(x, data_format):
    output = x.permute(0, 3, 1, 2)
    return torch.nn.Upsample(scale_factor=4, mode='nearest')
