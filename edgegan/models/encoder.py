import torch

import edgegan.nn.functional as F
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, name, in_dim, is_train, norm='batch',
                 image_size=128, latent_dim=8):             # Only use ResNet
        super(Encoder, self).__init__()
        self.name = name
        self._is_train = is_train
        self._norm = norm
        self._image_size = image_size
        self._latent_dim = latent_dim
        self._reuse = False
        self.num_filters = [128, 256, 512, 512]

        self.layers = nn.Sequential(
            nn.Conv2d(in_dim, 64, 4, 2),
            nn.ReLU(inplace=True),


        )
        self.conv1 = nn.Conv2d(in_dim, 64, 4, 2)
        self.relu = nn.ReLU()
        self.resnet = nn.R
