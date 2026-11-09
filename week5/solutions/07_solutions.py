"""Week 5 · 07 solutions. Look only after a real attempt."""
import numpy as np
import torch
import torch.nn as nn


def kmeans_fit(X, C, iters):
    C = C.copy()                                               # don't change the caller's centroids
    for _ in range(iters):
        labels = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)   # (n,)
        C = np.array([X[labels == j].mean(axis=0) for j in range(len(C))])         # (k, d)
    return C


def softmax_cross_entropy(Z, y):
    E = np.exp(Z - Z.max(axis=1, keepdims=True))
    P = E / E.sum(axis=1, keepdims=True)
    n = len(y)
    loss = -np.log(P[np.arange(n), y]).mean()
    dZ = (P - np.eye(Z.shape[1])[y]) / n
    return loss, dZ


def mlp_train_step(X, y, W1, b1, W2, b2, lr):
    Z1 = X @ W1 + b1
    H = np.maximum(Z1, 0)
    _, G = softmax_cross_entropy(H @ W2 + b2, y)
    dW2, db2 = H.T @ G, G.sum(axis=0)
    G1 = (G @ W2.T) * (Z1 > 0)
    dW1, db1 = X.T @ G1, G1.sum(axis=0)
    return W1 - lr * dW1, b1 - lr * db1, W2 - lr * dW2, b2 - lr * db2


def pairwise_sq_dists(A, B):
    return ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)


def torch_fit(model, X, y, epochs, lr):
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    losses = []
    for _ in range(epochs):
        loss = loss_fn(model(X), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses
