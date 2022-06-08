import torch
import torch.nn as nn
import torch.nn.functional as F

from edgegan.nn.modules.activation import Prelu

def instance_norm(x, eps=1e-5):
    mean = torch.mean(x, dim=[2,3], keepdim=True)
    sigma = torch.var(x, dim=[2,3], keepdim=True)
    return (x - mean) / (torch.sqrt(sigma) + eps)

def norm(input, norm='batch',
         epsilon=1e-5, momentum=0.9):
    assert norm in ['instance', 'batch', None]
    if norm == 'instance':
        eps = 1e-5
        mean = torch.mean(input, dim=[2,3], keepdims=True)
        sigma = torch.var(input, dim=[2,3], keepdims=True)
        normalized = (input - mean) / (torch.sqrt(sigma) + eps)
        out = normalized
    elif norm == 'batch':
        out = nn.BatchNorm2d(input, momentum=momentum, eps=epsilon)
    else:
        out = input

    return out


def _l2normalize(v, eps=1e-12):
    return v / (torch.sum(v ** 2) ** 0.5 + eps)


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def spectral_normed_weight(W, u=None, num_iters=1, with_sigma=False):
        W_shape = W.shape
        W_reshaped = torch.reshape(W, (-1, W_shape[-1]))
        if u is None:
            u = torch.empty((1, W_shape[-1]))
            u = truncated_normal_(u)

        def power_iteration(u_i, v_i):
            v_ip1 = _l2normalize(torch.matmul(u_i, W_reshaped.T))
            u_ip1 = _l2normalize(torch.matmul(v_ip1, W_reshaped))
            return u_ip1, v_ip1

        u_final = u
        v_final = torch.zeros((1, W_reshaped.shape[0]))
        while num_iters > 0:
            u_final, v_final = power_iteration(u_final, v_final)
            num_iters -= 1

        sigma = torch.matmul(torch.matmul(v_final, W_reshaped),
                             u_final.T)[0, 0]
        W_bar = W_reshaped / sigma
        W_bar = torch.reshape(W_bar, W_shape)

        if with_sigma:
            return W_bar, sigma
        else:
            return W_bar


class ADN(nn.Module):
    def __init__(self):
        super(ADN, self).__init__()
        self.activation = Prelu(0.2)

    def forward(self, x):
        x = self.activation(x)
        x = torch.nn.functional.instance_norm(x, momentum=0.9, eps=1e-05)
        return x