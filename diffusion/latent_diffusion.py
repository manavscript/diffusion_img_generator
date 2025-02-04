import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Encoder, Decoder

class VAE(nn.Module):
    def __init__(self, img_channels=3, latent_dim=256):
        super().__init__()
        self.encoder = Encoder([img_channels, 64, 128, 256])
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_var = nn.Linear(256 * 4 * 4, latent_dim)
        self.decoder = Decoder([latent_dim, 256, 128, 64])

    def forward(self, x):
        x, _ = self.encoder(x)
        x = x.view(x.shape[0], -1)  # Flatten
        mu, log_var = self.fc_mu(x), self.fc_var(x)

        # Reparameterization trick
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std

        z = z.view(z.shape[0], z.shape[1], 1, 1)  # Reshape for decoder
        x_recon = self.decoder(z, [])
        return x_recon, mu, log_var

    def encode(self, x):
        x, _ = self.encoder(x)
        x = x.view(x.shape[0], -1)
        mu, log_var = self.fc_mu(x), self.fc_var(x)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std  # Sampled latent vector

    def decode(self, z):
        z = z.view(z.shape[0], z.shape[1], 1, 1)
        return self.decoder(z, [])
