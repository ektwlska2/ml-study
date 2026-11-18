"""Week 6 · 04 solutions. Look only after a real attempt."""
import numpy as np
from scipy.stats import multivariate_normal


def gmm_e_step(X, pi, mu, Sigma):                     # booklet 9.4, the E-step lines
    K = len(pi)
    dens = np.column_stack([pi[k] * multivariate_normal(mu[k], Sigma[k]).pdf(X) for k in range(K)])   # (n, K)
    return dens / dens.sum(axis=1, keepdims=True)     # responsibilities: each row sums to 1


def conv_out_size(n, k, p=0, s=1):
    return (n + 2 * p - k) // s + 1


def conv2d_single(x, K):
    a, b = K.shape
    out = np.zeros((x.shape[0] - a + 1, x.shape[1] - b + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(x[i:i + a, j:j + b] * K)   # the window under the kernel, times the kernel, added up
    return out


def conv_params(c_in, c_out, k):
    return c_out * c_in * k * k + c_out               # one k x k x c_in kernel per output channel, plus one bias each


def maxpool2d(x, size=2):
    h, w = x.shape
    return x.reshape(h // size, size, w // size, size).max(axis=(1, 3))   # cut into size x size windows, max of each
