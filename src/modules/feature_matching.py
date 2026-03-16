import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, discriminator):
        super(FeatureExtractor, self).__init__()
        self.features = discriminator.dx

    def forward(self, x):
        return self.features(x)
