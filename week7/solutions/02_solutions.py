"""Week 7 · 02 solutions. Look only after a real attempt."""
import numpy as np
import torch
import torch.nn as nn


def pca_reconstruct(X, k):                             # booklet 6.6
    x_bar = X.mean(axis=0)
    Xc = X - x_bar
    lam, W = np.linalg.eigh(Xc.T @ Xc / (len(X) - 1))
    W = W[:, np.argsort(lam)[::-1]][:, :k]             # top-k eigenvectors as columns, (d, k)
    return Xc @ W @ W.T + x_bar                        # scores Z = Xc W, then back: Z W^T + mean


class Autoencoder(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, k))
        self.decoder = nn.Sequential(nn.Linear(k, 128), nn.ReLU(), nn.Linear(128, d), nn.Sigmoid())

    def forward(self, x):
        return self.decoder(self.encoder(x))            # x -> z -> x_hat


def recon_errors(model, X):
    with torch.no_grad():
        return ((model(X) - X) ** 2).mean(dim=1)        # one number per image: mean over its pixels


def denoising_step(model, opt, xb, std):
    noisy = (xb + std * torch.randn_like(xb)).clamp(0, 1)   # corrupt the input ...
    loss = nn.functional.mse_loss(model(noisy), xb)         # ... but compare with the CLEAN image
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


def deconv_out_size(n, k, p=0, s=1, output_padding=0):
    return (n - 1) * s - 2 * p + k + output_padding
