"""Week 6 · 00 solutions. Look only after a real attempt."""
import numpy as np


def logistic_gd(X, y, lr, steps):                 # booklet 2.5
    n, d = X.shape
    sigmoid = lambda z: 1 / (1 + np.exp(-z))
    w, b = np.zeros(d), 0.0
    for _ in range(steps):
        p = sigmoid(X @ w + b)                    # p_i = sigma(w . x_i + b)
        w = w - lr * X.T @ (p - y) / n            # dL/dw = (1/n) sum_i (p_i - y_i) x_i
        b = b - lr * np.mean(p - y)               # dL/db = (1/n) sum_i (p_i - y_i)
    return w, b
