"""Week 6 · 07 Part C solutions."""
import numpy as np


def conv2d_single(x, K):
    a, b = K.shape
    out = np.zeros((x.shape[0] - a + 1, x.shape[1] - b + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(x[i:i + a, j:j + b] * K)
    return out


def batchnorm_train(X, gamma, beta, eps=1e-5):
    return gamma * (X - X.mean(axis=0)) / np.sqrt(X.var(axis=0) + eps) + beta


def early_stop(val_losses, patience):
    best = 0
    for epoch in range(len(val_losses)):
        if val_losses[epoch] < val_losses[best]:
            best = epoch
        elif epoch - best >= patience:
            return epoch, best
    return len(val_losses) - 1, best


def kmeans_step(X, mu):                                    # booklet 7: one assignment + update
    dist = np.linalg.norm(X[:, None, :] - mu[None, :, :], axis=2)   # (n, k)
    c = dist.argmin(axis=1)
    return c, np.array([X[c == j].mean(axis=0) for j in range(len(mu))])


def pca_scores(X, k):                                      # booklet 6
    Xc = X - X.mean(axis=0)
    lam, W = np.linalg.eigh(Xc.T @ Xc / (len(X) - 1))
    W = W[:, np.argsort(lam)[::-1]]                        # largest eigenvalue first
    return Xc @ W[:, :k]
