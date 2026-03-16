import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_channels=1, latent_dim=200):
        super(Discriminator, self).__init__()
        self.dx = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1), 
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.dz = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.joint = nn.Sequential(
            nn.Linear(64*7*7 + 512, 1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x, z):
        batch_size = x.size(0)
        dx_feat = self.dx(x)
        dx_feat = dx_feat.view(batch_size, -1)
        dz_feat = self.dz(z)
        combined = torch.cat([dx_feat, dz_feat], dim=1)
        logits = self.joint(combined)
        return logits
