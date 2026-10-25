"""Week 3 · 05 solutions. Look only after a real attempt."""
import numpy as np


def signed_distance(X, w, b):
    return (X @ w + b) / np.linalg.norm(w)


def hinge_loss(X, y, w, b, lam):
    return np.mean(np.maximum(0, 1 - y * (X @ w + b))) + lam / 2 * w @ w


def hinge_grad(X, y, w, b, lam):
    active = y * (X @ w + b) < 1                     # (n,) inside the margin or wrong side
    n = len(y)
    gw = lam * w - X[active].T @ y[active] / n       # (d,)
    gb = -y[active].sum() / n                        # ()
    return gw, gb


def svm_train(X, y, lam, lr, steps):
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        gw, gb = hinge_grad(X, y, w, b, lam)
        w, b = w - lr * gw, b - lr * gb
    return w, b


def poly_kernel(A, B, degree, c):
    return (A @ B.T + c) ** degree


def rbf_kernel(A, B, gamma):
    sq = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)     # (m, n) squared distances
    return np.exp(-gamma * sq)


def phi2(X):
    x1, x2 = X[:, 0], X[:, 1]
    r = np.sqrt(2)
    return np.column_stack([np.ones(len(X)), r * x1, r * x2, x1 ** 2, r * x1 * x2, x2 ** 2])
