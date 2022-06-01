import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings

def norm(input, is_train, norm='batch',
         epsilon=1e-5, momentum=0.9):
    assert norm in ['instance', 'batch', None]
    if norm == 'instance':
        eps = 1e-5
        mean = torch.mean(input, dim=[1,2], keepdims=True)
        sigma = torch.var(input, dim=[1,2], keepdims=True)
        normalized = (input - mean) / (torch.sqrt(sigma) + eps)
        out = normalized
    elif norm == 'batch':
        out = nn.BatchNorm2d(input, momentum=momentum, eps=epsilon)
    else:
        out = input

    return out

def _l2normalize(v, eps=1e-12):
    return v / (torch.sum(v ** 2) ** 0.5 + eps)

def spectral_normed_weight(W, u=None, num_iters=1, update_collection=None, with_sigma=False):
    W_shape = W.shape.as_list()
    W_reshaped = W.reshape(-1, W_shape[-1])

    if u is None:
        u = nn.init.trunc_normal_(torch.tensor(1, W_shape[-1]))

    def power_iteration(i, u_i, v_i):
        v_ip1 = _l2normalize(torch.matmul(u_i, torch.transpose(W_reshaped)))
        u_ip1 = _l2normalize(torch.matmul(v_ip1, W_reshaped))
        return i + 1, u_ip1, v_ip1

    i = 0
    u_i = u
    v_i = torch.zeros((1, W_reshaped.shape.as_list()[0]),dtype=torch.float32)
    while i < num_iters:
        i,u_i,v_i = power_iteration(i,u_i,v_i)
    u_final, v_final = u_i, v_i

    if update_collection is None:
        warnings.warn(
            'Setting update_collection to None will make u being updated every W execution. This maybe undesirable'
            '. Please consider using a update collection instead.')
        sigma = tf.matmul(tf.matmul(v_final, W_reshaped),
                          tf.transpose(u_final))[0, 0]
        W_bar = W_reshaped / sigma
        with tf.control_dependencies([u.assign(u_final)]):
            W_bar = tf.reshape(W_bar, W_shape)
    else:
        sigma = tf.matmul(tf.matmul(v_final, W_reshaped),
                          tf.transpose(u_final))[0, 0]
        W_bar = W_reshaped / sigma
        W_bar = tf.reshape(W_bar, W_shape)
        if update_collection != NO_OPS:
            tf.add_to_collection(update_collection, u.assign(u_final))
    if with_sigma:
        return W_bar, sigma
    else:
        return W_bar