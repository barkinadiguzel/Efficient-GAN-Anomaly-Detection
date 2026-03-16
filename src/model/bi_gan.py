import torch
import torch.nn as nn
import torch.optim as optim

from layers.encoder import Encoder
from layers.generator import Generator
from layers.discriminator import Discriminator

class BiGAN(nn.Module):
    def __init__(self, input_channels=1, latent_dim=200, device='cuda'):
        super(BiGAN, self).__init__()
        self.device = device
        self.encoder = Encoder(input_channels, latent_dim).to(device)
        self.generator = Generator(latent_dim, input_channels).to(device)
        self.discriminator = Discriminator(input_channels, latent_dim).to(device)

        self.opt_E = optim.Adam(self.encoder.parameters(), lr=1e-5, betas=(0.5,0.5))
        self.opt_G = optim.Adam(self.generator.parameters(), lr=1e-5, betas=(0.5,0.5))
        self.opt_D = optim.Adam(self.discriminator.parameters(), lr=1e-5, betas=(0.5,0.5))

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.generator(z)
        logits = self.discriminator(x, z)
        return x_hat, logits
