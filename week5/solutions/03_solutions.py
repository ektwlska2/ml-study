"""Week 5 · 03 solutions. Look only after a real attempt."""
import numpy as np


def dense_backward(X, W, G):
    return G @ W.T, X.T @ G, G.sum(axis=0)


def backprop_221(x, y, W1, b1, W2, b2):
    h = 1 / (1 + np.exp(-(x @ W1 + b1)))        # (h,)
    p = 1 / (1 + np.exp(-(h @ W2 + b2)))        # a number
    d2 = p - y                                   # dL/dz2 for sigmoid + binary cross-entropy
    dh = d2 * W2                                 # (h,)
    dz1 = dh * h * (1 - h)                       # through the sigmoid
    return np.outer(x, dz1), dz1, d2 * h, d2     # dW1 (d, h), db1 (h,), dW2 (h,), db2


def sigmoid_backward(dH, Z):
    s = 1 / (1 + np.exp(-Z))
    return dH * s * (1 - s)


def relu_backward(dH, Z):
    return dH * (Z > 0)


def numerical_grad(f, theta, eps=1e-5):
    g = np.zeros_like(theta, dtype=float)
    for idx in np.ndindex(theta.shape):          # every position (i, j, ...) of theta
        old = theta[idx]
        theta[idx] = old + eps; up = f(theta)
        theta[idx] = old - eps; down = f(theta)
        theta[idx] = old                         # put it back: never leave the caller's array changed
        g[idx] = (up - down) / (2 * eps)
    return g


def mlp_gradients(X, y, Ws, bs):
    Hs, Zs = [X], []
    for l in range(len(Ws)):
        Z = Hs[-1] @ Ws[l] + bs[l]
        Zs.append(Z)
        Hs.append(np.maximum(Z, 0) if l < len(Ws) - 1 else Z)
    E = np.exp(Hs[-1] - Hs[-1].max(axis=1, keepdims=True))
    P = E / E.sum(axis=1, keepdims=True)
    G = (P - np.eye(P.shape[1])[y]) / len(y)
    dWs, dbs = [None] * len(Ws), [None] * len(Ws)
    for l in range(len(Ws) - 1, -1, -1):
        dWs[l] = Hs[l].T @ G
        dbs[l] = G.sum(axis=0)
        if l > 0:
            G = (G @ Ws[l].T) * (Zs[l - 1] > 0)
    return dWs, dbs


class MLPScratch:
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(size=(a, b)) * np.sqrt(2 / a) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]

    def forward(self, X):
        H = X
        for l in range(len(self.W)):
            Z = H @ self.W[l] + self.b[l]
            H = np.maximum(Z, 0) if l < len(self.W) - 1 else Z
        return H

    def predict_proba(self, X):
        Z = self.forward(X)
        E = np.exp(Z - Z.max(axis=1, keepdims=True))
        return E / E.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.forward(X).argmax(axis=1)

    def fit(self, X, y, epochs=100, lr=0.1):
        self.loss_history_ = []
        for _ in range(epochs):
            P = self.predict_proba(X)
            self.loss_history_.append(-np.log(P[np.arange(len(y)), y]).mean())
            dWs, dbs = mlp_gradients(X, y, self.W, self.b)
            self.W = [W - lr * g for W, g in zip(self.W, dWs)]
            self.b = [b - lr * g for b, g in zip(self.b, dbs)]
        return self
