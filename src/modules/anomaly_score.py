import torch
import torch.nn.functional as F

def compute_anomaly_score(x, encoder, generator, discriminator, alpha=0.9, loss_type="feature_matching", feature_extractor=None):
    z = encoder(x)
    x_hat = generator(z)

    # Reconstruction loss
    L_G = F.l1_loss(x_hat, x, reduction='none').view(x.size(0), -1).mean(dim=1)

    # Discriminator loss
    if loss_type == "feature_matching":
        if feature_extractor is None:
            raise ValueError("feature_extractor must be provided")
        f_real = feature_extractor(x)
        f_fake = feature_extractor(x_hat)
        L_D = F.l1_loss(f_real, f_fake, reduction='none').view(x.size(0), -1).mean(dim=1)
    else:
        raise ValueError("Only feature_matching implemented for CNN version")

    A_x = alpha * L_G + (1 - alpha) * L_D
    return A_x
