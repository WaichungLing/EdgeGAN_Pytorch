import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from nn.modules.normalization import *

class Discriminator(nn.Module):
    def __init__(self, input_height=64, input_width=64, input_channels=3, num_filters=64):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_filters, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(num_filters * 4, num_filters * 8, kernel_size=4, stride=2, padding=1)
        self.fc = nn.Linear(num_filters * 8 * int(input_height / 2 ** 4) * int(input_width / 2 ** 4), 1)

        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)
        nn.init.normal_(self.conv1.weight, std=0.02)
        nn.init.zeros_(self.conv1.bias)
        nn.init.normal_(self.conv2.weight, std=0.02)
        nn.init.zeros_(self.conv2.bias)
        nn.init.normal_(self.conv3.weight, std=0.02)
        nn.init.zeros_(self.conv3.bias)
        nn.init.normal_(self.conv4.weight, std=0.02)
        nn.init.zeros_(self.conv4.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = instance_norm(x)
        x = F.leaky_relu(x)
        x = self.conv3(x)
        x = instance_norm(x)
        x = F.leaky_relu(x)
        x = self.conv4(x)
        x = instance_norm(x)
        x = F.leaky_relu(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return torch.sigmoid(x), x
