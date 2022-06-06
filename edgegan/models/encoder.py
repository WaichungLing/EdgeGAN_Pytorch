import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class Encoder(nn.Module):
    def __init__(self, input_channels=3, latent_dim=100): 
        super(Encoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            Residual(input_channels=64, num_filters=128),
            nn.AvgPool2d(kernel_size=2, stride=2),
            Residual(input_channels=128, num_filters=256),
            nn.AvgPool2d(kernel_size=2, stride=2),
            Residual(input_channels=256, num_filters=512),
            nn.AvgPool2d(kernel_size=2, stride=2),
            Residual(input_channels=512, num_filters=512),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=8, stride=8, padding=3)
        )
        self.mu = nn.Linear(512, latent_dim)
        self.log_sigma = nn.Linear(512, latent_dim)
        self.random_normal = torch.normal(mean=0.0, std=1.0, size=(1,latent_dim)).to(config.device)

    def forward(self, x):
        x = self.main(x)
        x = torch.flatten(x, 1)
        m = self.mu(x)
        s = self.log_sigma(x)
        z = m + self.random_normal * torch.exp(s)
        return z


class Residual(nn.Module):
    def __init__(self, input_channels=3, num_filters=64):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_filters, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1, padding_mode='reflect')
        self.shortcut = nn.Conv2d(input_channels, num_filters, kernel_size=1, stride=1)

    def forward(self, x):
        shortcut = self.shortcut(x) 
        x = self.conv1(x)
        x = F.instance_norm(x, momentum=0.9, eps=1e-05)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.instance_norm(x, momentum=0.9, eps=1e-05)
        x = F.relu(x + shortcut)
        return x
