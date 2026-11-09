"""Week 5 · 01 solutions. Look only after a real attempt."""
import numpy as np


def softmax_rows(Z):
    E = np.exp(Z - Z.max(axis=1, keepdims=True))      # (n, C); subtracting the row max avoids overflow
    return E / E.sum(axis=1, keepdims=True)


def gaussian_pdf(x, mu, var):
    return np.exp(-(x - mu) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)


def e_step(x, mu, var, pi):
    m = pi[None, :] * gaussian_pdf(x[:, None], mu[None, :], var[None, :])   # (n, K): entry (i, k) = pi_k N(x_i | mu_k, var_k)
    return m / m.sum(axis=1, keepdims=True)                                   # each row divided by its own sum


def weighted_means(x, W):
    return (W * x[:, None]).sum(axis=0) / W.sum(axis=0)      # (K,)


def m_step(x, R):
    Nk = R.sum(axis=0)                                        # (K,)
    mu = (R * x[:, None]).sum(axis=0) / Nk                    # (K,)
    var = (R * (x[:, None] - mu[None, :]) ** 2).sum(axis=0) / Nk
    pi = Nk / len(x)
    return mu, var, pi


def standardize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)


def pairwise_sq_dists(A, B):
    return ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)  # (m, p, d) -> sum over d -> (m, p)


def row_dots(A, B):
    return (A * B).sum(axis=1)


def cosine_similarity(A, B):
    na = np.linalg.norm(A, axis=1, keepdims=True)              # (m, 1)
    nb = np.linalg.norm(B, axis=1, keepdims=True)              # (p, 1)
    return (A @ B.T) / (na * nb.T)                             # (m, p) / ((m, 1) * (1, p))


def dense_forward(X, W, b):
    return X @ W + b


def dense_backward(X, W, G):
    return G @ W.T, X.T @ G, G.sum(axis=0)


def one_hot(y, C):
    return np.eye(C)[y]


def cross_entropy(P, y):
    return -np.log(P[np.arange(len(y)), y]).mean()
