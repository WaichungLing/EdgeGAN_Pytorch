import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from edgegan.nn.modules.normalization import *

class Generator(nn.Module):
    def __init__(self, output_height=64, output_width=64, in_dim=105, input_channels=64, output_channels=3):
        super(Generator,self).__init__()
        self.s_h, self.s_w = output_height, output_width
        self.input_channels, self.output_channels = input_channels, output_channels
        self.s_h2, self.s_w2 = self._conv_out_size_same(self.s_h, 2), self._conv_out_size_same(self.s_w, 2)
        self.s_h4, self.s_w4 = self._conv_out_size_same(self.s_h2, 2), self._conv_out_size_same(self.s_w2, 2)
        self.s_h8, self.s_w8 = self._conv_out_size_same(self.s_h4, 2), self._conv_out_size_same(self.s_w4, 2)
        self.s_h16, self.s_w16 = self._conv_out_size_same(self.s_h8, 2), self._conv_out_size_same(self.s_w8, 2)

        self.fc = nn.Linear(in_dim, input_channels*8*self.s_h16*self.s_w16) #TODO
        self.deconv1 = nn.ConvTranspose2d(input_channels*8, input_channels*4, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(input_channels*4, input_channels*2, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(input_channels*2, input_channels, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1)

        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)
        nn.init.normal_(self.deconv1.weight, std=0.02)
        nn.init.zeros_(self.deconv1.bias)
        nn.init.normal_(self.deconv2.weight, std=0.02)
        nn.init.zeros_(self.deconv2.bias)
        nn.init.normal_(self.deconv3.weight, std=0.02)
        nn.init.zeros_(self.deconv3.bias)
        nn.init.normal_(self.deconv4.weight, std=0.02)
        nn.init.zeros_(self.deconv4.bias)

    def _conv_out_size_same(self, size, stride):
        return int(math.ceil(float(size) / float(stride)))

    def forward(self, x):
        x = self.fc(x)
        x = x.reshape(-1, self.input_channels*8, self.s_h16, self.s_w16)
        x = instance_norm(x)
        x = F.relu(x)
        x = self.deconv1(x)
        x = instance_norm(x)
        x = F.relu(x)
        x = self.deconv2(x)
        x = instance_norm(x)
        x = F.relu(x)
        x = self.deconv3(x)
        x = instance_norm(x)
        x = F.relu(x)
        x = self.deconv4(x)
        #x = instance_norm(x)
        x = torch.tanh(x)
        return x