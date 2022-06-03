import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class Discriminator(nn.Module):
    def __init__(self, input_height=64, input_width=64, input_channels=3, num_filters=64):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_filters, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(num_filters * 4, num_filters * 8, kernel_size=4, stride=2, padding=1)
        self.fc = nn.Linear(num_filters * 8 * int(input_height / 2 ** 4) * int(input_width / 2 ** 4), 1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = F.instance_norm(x, momentum=0.9, eps=1e-05)
        x = F.leaky_relu(x)
        x = self.conv3(x)
        x = F.instance_norm(x, momentum=0.9, eps=1e-05)
        x = F.leaky_relu(x)
        x = self.conv4(x)
        x = F.instance_norm(x, momentum=0.9, eps=1e-05)
        x = F.leaky_relu(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return torch.sigmoid(x), x
