import torch
import torch.nn as nn
import torch.nn.functional as F

from .discriminator import Discriminator
from .encoder import Encoder
from .generator import Generator

class EdgeGAN(nn.Module):
    def __init__(self): 
        super(EdgeGAN, self).__init__()
        self.edge_generator = Generator()
        self.image_generator = Generator()
        self.joint_discriminator = Discriminator()
        self.edge_discriminator = Discriminator()
        self.image_discriminator = Discriminator()

    def forward(self, x, z):

        return x