import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalVAE(nn.Module):
    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int, latent_dim: int, dropout: float = 0.1):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.latent_dim = latent_dim

        # Encoder takes [x, y]
        self.enc = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder takes [z, y]
        self.dec = nn.Sequential(
            nn.Linear(latent_dim + y_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, x_dim),
        )

    def encode(self, x, y):
        h = self.enc(torch.cat([x, y], dim=1))
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        return self.dec(torch.cat([z, y], dim=1))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, y)
        return x_hat, mu, logvar

def vae_loss(x, x_hat, mu, logvar, beta_kl: float = 1.0):
    # Reconstruction: MSE works well for standardized expression
    recon = F.mse_loss(x_hat, x, reduction="mean")
    # KL divergence
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta_kl * kl, recon.detach(), kl.detach()
