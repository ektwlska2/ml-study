"""Week 7 · 03 solutions. Look only after a real attempt."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def gmm_m_step(X, gamma):                              # booklet 9.4: the M-step
    n, d = X.shape
    Nk = gamma.sum(axis=0)                             # N_k = sum_i gamma_ik
    pi = Nk / n
    mu = gamma.T @ X / Nk[:, None]
    Sigma = np.array([(gamma[:, k, None] * (X - mu[k])).T @ (X - mu[k]) / Nk[k] + 1e-6 * np.eye(d)
                      for k in range(len(Nk))])
    return pi, mu, Sigma


def reparameterize(mu, logvar, eps):
    return mu + torch.exp(0.5 * logvar) * eps          # sigma = exp(log sigma^2 / 2)


def kl_per_sample(mu, logvar):                         # KL( N(mu, sigma^2) || N(0, I) ), summed over the code
    return -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1)


def vae_loss(x, x_hat, mu, logvar):                    # -ELBO, averaged over the batch
    recon = F.binary_cross_entropy(x_hat, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
    return (recon + kl) / len(x)


class VAE(nn.Module):
    def __init__(self, d, h, k):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d, h), nn.ReLU())
        self.to_mu = nn.Linear(h, k)
        self.to_logvar = nn.Linear(h, k)
        self.decoder = nn.Sequential(nn.Linear(k, h), nn.ReLU(), nn.Linear(h, d), nn.Sigmoid())

    def forward(self, x):
        hdn = self.encoder(x)
        mu, logvar = self.to_mu(hdn), self.to_logvar(hdn)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)    # the reparameterization trick
        return self.decoder(z), mu, logvar
