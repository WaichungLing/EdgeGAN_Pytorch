import numpy as np
import torch
import torch.nn.functional as F
import config


def get_acgan_loss_focal(real_image_logits_out, real_image_label,
                         disc_image_logits_out, condition,
                         num_classes, ld1=1, ld2=0.5, ld_focal=2.):
    real_image_label = torch.argmax(real_image_label, dim=1)
    condition = torch.argmax(condition, dim=1)
    loss_ac_d = torch.mean((1 - torch.sum(F.softmax(real_image_logits_out, dim=1) * torch.squeeze(
        F.one_hot(real_image_label, num_classes)), 1)) ** ld_focal *
                           F.cross_entropy(real_image_logits_out, real_image_label))
    loss_ac_d = ld1 * loss_ac_d
    loss_ac_g = torch.mean(
        F.cross_entropy(disc_image_logits_out, condition))
    loss_ac_g = ld2 * loss_ac_g
    return loss_ac_g, loss_ac_d


def get_class_loss(logits_out, label, num_classes, ld_focal=2.0):
    loss = torch.mean((1 - torch.sum(F.softmax(logits_out) * torch.squeeze(
        F.one_hot(label, num_classes)), 1)) ** ld_focal *
                      F.cross_entropy(logits_out, label))
    return loss


def gradient_penalty(output, on):
    gradients = torch.autograd.grad(output, [on, ], grad_outputs=torch.ones_like(output))[0]
    first_dim = gradients.shape[0]
    grad_l2 = torch.sqrt(torch.sum(torch.square(gradients).view(first_dim, -1), dim=1))
    return torch.mean((grad_l2 - 1) ** 2)


def discriminator_ganloss(output, target):
    return torch.mean(output - target)


def generator_ganloss(output):
    return torch.mean(output * -1)


def l1loss(output, target, weight):
    return weight * torch.mean(torch.abs(output - target))


def random_blend(a, b, batchsize):
    alpha = torch.rand(size=(batchsize, 1, 1, 1)).to(config.device)
    return b + alpha * (a - b)


def penalty(synthesized, real, nn_func, batchsize, weight=10.0):
    batchsize = synthesized.shape[0]
    interpolated = random_blend(synthesized, real, batchsize)
    smd, logit = nn_func(interpolated)
    inte_logit = torch.cat((smd, logit), dim=1)
    return weight * gradient_penalty(inte_logit, interpolated)
