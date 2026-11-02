"""Week 4 · 07 Part C solutions."""
import numpy as np


def gb_predict_one(f0, stumps, lr, x):
    F = f0
    for t, left, right in stumps:
        F += lr * (left if x <= t else right)
    return F


def adaboost_predict_one(model, x):
    F = sum(alpha * (s if x > t else -s) for t, s, alpha in model)
    return 1 if F >= 0 else -1


def kmeans_predict(X_new, C):
    return ((X_new[:, None, :] - C[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)


def logistic_predict(X, w, b, t):
    return (1 / (1 + np.exp(-(X @ w + b))) >= t).astype(int)


def pca_scores(X_new, mean, components):
    return (X_new - mean) @ components.T
