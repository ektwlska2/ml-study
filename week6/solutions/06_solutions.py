"""Week 6 · 06 solutions. Look only after a real attempt."""
import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.tree import DecisionTreeRegressor


def gboost(X, y, n_trees=300, lr=0.1):                # booklet 5: gradient boosting, squared loss
    F = np.full(len(y), y.mean())                     # F_0 = y bar
    trees = []
    for _ in range(n_trees):
        r = y - F                                     # residuals = -dL/dF
        tree = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X, r)
        F = F + lr * tree.predict(X)                  # F_m = F_{m-1} + lr * h_m
        trees.append(tree)
    return y.mean(), trees


def gb_predict(F0, trees, X, lr=0.1):
    return F0 + lr * sum(tree.predict(X) for tree in trees)


def gini(y):
    p = np.bincount(y) / len(y)
    return 1 - np.sum(p ** 2)


def best_split(X, y):                                 # booklet 3, with the Gini impurity
    best = (None, None, 0.0)
    for j in range(X.shape[1]):
        v = np.unique(X[:, j])
        for t in (v[:-1] + v[1:]) / 2:                # midpoints between sorted values
            L = X[:, j] <= t
            delta = gini(y) - (L.mean() * gini(y[L]) + (~L).mean() * gini(y[~L]))
            if delta > best[2]:
                best = (j, t, delta)
    return best


def conv2d(x, K, stride=1, padding=0):
    x = np.pad(x, padding)                            # zeros around the border
    k = K.shape[0]
    h = (x.shape[0] - k) // stride + 1                # the padding is already in x.shape
    w = (x.shape[1] - k) // stride + 1
    out = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            r, c = i * stride, j * stride             # the window's top-left corner
            out[i, j] = np.sum(x[r:r + k, c:c + k] * K)
    return out


def batchnorm_eval(X, running_mean, running_var, gamma, beta, eps=1e-5):
    return gamma * (X - running_mean) / np.sqrt(running_var + eps) + beta


def fit_early_stopping(model, X_tr, y_tr, X_va, y_va, epochs, patience, lr):
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(epochs):
        model.train()
        loss = loss_fn(model(X_tr), y_tr)
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(X_va), y_va).item()
        if val < best:
            best, best_state, best_epoch = val, copy.deepcopy(model.state_dict()), epoch
        elif epoch - best_epoch >= patience:
            break
    model.load_state_dict(best_state)
    return best_epoch
