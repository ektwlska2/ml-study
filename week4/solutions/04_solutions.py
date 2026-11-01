"""Week 4 · 04 solutions. Look only after a real attempt."""
import numpy as np


def linkage_distance(A, B, how):
    D = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))   # every point of A vs every point of B
    return {"single": D.min(), "complete": D.max(), "average": D.mean()}[how]


def dbscan_core(X, eps, min_pts):
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))   # (n, n)
    return (D <= eps).sum(axis=1) >= min_pts                           # each point counts itself (distance 0)


def gaussian_pdf(x, mu, var):
    return np.exp(-(x - mu) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)


def e_step(x, mu, var, pi):
    N = np.stack([pi[k] * gaussian_pdf(x, mu[k], var[k]) for k in range(len(mu))], axis=1)   # (n, K)
    return N / N.sum(axis=1, keepdims=True)


def m_step(x, R):
    Nk = R.sum(axis=0)                                           # (K,) "how many points" each component has
    mu = (R * x[:, None]).sum(axis=0) / Nk
    var = (R * (x[:, None] - mu) ** 2).sum(axis=0) / Nk
    return mu, var, Nk / len(x)


def log_likelihood(x, mu, var, pi):
    p = sum(pi[k] * gaussian_pdf(x, mu[k], var[k]) for k in range(len(mu)))
    return np.log(p).sum()
