import torch
import torch.nn as nn


def activation_fn(input, name='lrelu'):
    assert name in ['relu', 'lrelu', 'tanh', 'sigmoid', None]
    if name == 'relu':
        return nn.ReLU(input)
    elif name == 'lrelu':
        return torch.maximum(input, 0.2*input)
    elif name == 'tanh':
        return nn.tanh(input)
    elif name == 'sigmoid':
        return nn.sigmoid(input)
    else:
        return input


def miu_relu(x, miu=0.7, name="miu_relu"):
    return (x + torch.sqrt((1 - miu) ** 2 + x ** 2)) / 2.


# def prelu(x, name="prelu"):
#     leak = tf.get_variable("param", shape=None, initializer=0.2, regularizer=None,
#                             trainable=True, caching_device=None)
#     return torch.maximum(leak * x, x)


def lrelu(x, leak=0.2, name="lrelu"):
    return torch.maximum(leak * x, x)