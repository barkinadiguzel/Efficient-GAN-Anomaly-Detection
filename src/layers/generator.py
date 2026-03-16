import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=200, output_channels=1):
        super(Generator, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 7*7*128),
            nn.BatchNorm1d(7*7*128),
            nn.ReLU(True)
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  
        )

    def forward(self, z):
        x = self.fc1(z)
        x = self.fc2(x)
        x = x.view(x.size(0), 128, 7, 7)
        x_hat = self.deconv(x)
        return x_hat
