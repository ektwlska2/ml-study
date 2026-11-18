"""Week 6 · 03 solutions. Look only after a real attempt."""
import numpy as np
import torch
import torch.nn as nn


def adam(w, g, st, lr, b1=0.9, b2=0.999, eps=1e-8):        # booklet 14
    st["t"] = st.get("t", 0) + 1
    st["m"] = b1 * st.get("m", 0) + (1 - b1) * g           # direction
    st["s"] = b2 * st.get("s", 0) + (1 - b2) * g ** 2      # scale
    m_hat, s_hat = st["m"] / (1 - b1 ** st["t"]), st["s"] / (1 - b2 ** st["t"])   # bias correction
    return w - lr * m_hat / (np.sqrt(s_hat) + eps)


def batchnorm_train(X, gamma, beta, eps=1e-5):
    mu = X.mean(axis=0)                                    # one mean per feature (column)
    var = X.var(axis=0)                                    # the batch's variance (divided by n)
    return gamma * (X - mu) / np.sqrt(var + eps) + beta


def layernorm(X, gamma, beta, eps=1e-5):
    mu = X.mean(axis=1, keepdims=True)                     # one mean per sample (row)
    var = X.var(axis=1, keepdims=True)
    return gamma * (X - mu) / np.sqrt(var + eps) + beta


def clip_by_global_norm(grads, max_norm):
    total = np.sqrt(sum((g ** 2).sum() for g in grads))    # the length of ALL gradients together
    scale = min(1.0, max_norm / (total + 1e-6))
    return [g * scale for g in grads]


class ResidualBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)

    def forward(self, x):
        return x + self.fc2(torch.relu(self.fc1(x)))       # the skip: add the input back
