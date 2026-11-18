"""Week 6 · 01 solutions. Look only after a real attempt."""
import numpy as np
from scipy.special import softmax


def linreg_gd(X1, y, lr, steps):                  # booklet 1.8
    n = len(y)
    w = np.zeros(X1.shape[1])
    for _ in range(steps):
        grad = 2 / n * X1.T @ (X1 @ w - y)        # gradient of the MSE
        w = w - lr * grad                         # step against it
    return w


def backward(Hs, Zs, Y, Ws):                      # booklet 13.7, step 2 and 3 of the algorithm card
    G = (softmax(Hs[-1], axis=1) - Y) / len(Y)    # G_L = (P - Y) / n
    grads = []
    for l in reversed(range(len(Ws))):            # l = L, ..., 1
        grads.insert(0, (Hs[l].T @ G, G.sum(axis=0)))   # dW_l = H_{l-1}^T G_l, db_l = column sums
        if l > 0:
            G = (G @ Ws[l].T) * (Zs[l - 1] > 0)   # pass back, then the ReLU mask
    return grads
