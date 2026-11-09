"""Week 5 · 02 solutions. Look only after a real attempt."""
import numpy as np


def logistic_grad(X, y, w, b):
    p = 1 / (1 + np.exp(-(X @ w + b)))       # (n,)
    return X.T @ (p - y) / len(y), (p - y).mean()


def collapse_linear(W1, b1, W2, b2):
    return W1 @ W2, b1 @ W2 + b2


def mlp_forward(X, Ws, bs):
    H = X
    for l in range(len(Ws)):
        Z = H @ Ws[l] + bs[l]                 # (n, size of layer l+1)
        H = np.maximum(Z, 0) if l < len(Ws) - 1 else Z    # ReLU between layers, nothing after the last
    return H


def softmax_ce_grad(Z, y):
    E = np.exp(Z - Z.max(axis=1, keepdims=True))
    P = E / E.sum(axis=1, keepdims=True)
    return (P - np.eye(Z.shape[1])[y]) / len(y)


def count_params(sizes):
    return sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))


class MyMLP:
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(size=(a, b)) * np.sqrt(2 / a) for a, b in zip(sizes[:-1], sizes[1:])]   # He initialization
        self.b = [np.zeros(b) for b in sizes[1:]]

    def forward(self, X):
        H = X
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            Z = H @ W + b
            H = np.maximum(Z, 0) if l < len(self.W) - 1 else Z
        return H                                   # logits

    def predict_proba(self, X):
        Z = self.forward(X)
        E = np.exp(Z - Z.max(axis=1, keepdims=True))
        return E / E.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.forward(X).argmax(axis=1)
