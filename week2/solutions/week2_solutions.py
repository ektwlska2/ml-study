"""Week 2 solutions. Look only after a real attempt."""
import numpy as np


def add_bias(X):
    return np.column_stack([np.ones(len(X)), X])


def ols_fit(X, y):
    A = add_bias(X)
    return np.linalg.solve(A.T @ A, A.T @ y)


def r2(y, y_hat):
    return 1 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2)


def ridge_fit(X, y, lam):
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)


def gd_linear(X, y, lr, steps):
    w = np.zeros(X.shape[1])
    n = len(y)
    for _ in range(steps):
        w = w - lr * (2 / n) * X.T @ (X @ w - y)
    return w


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def bce(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def logistic_gd(X, y, lr, steps):
    w = np.zeros(X.shape[1])
    n = len(y)
    for _ in range(steps):
        w = w - lr * (1 / n) * X.T @ (sigmoid(X @ w) - y)
    return w


def confusion_counts(y, y_hat):
    tp = ((y == 1) & (y_hat == 1)).sum()
    fp = ((y == 0) & (y_hat == 1)).sum()
    fn = ((y == 1) & (y_hat == 0)).sum()
    tn = ((y == 0) & (y_hat == 0)).sum()
    return tp, fp, fn, tn


def precision_recall_f1(y, y_hat):
    tp, fp, fn, tn = confusion_counts(y, y_hat)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def knn_predict(X_train, y_train, X_test, k):
    d = np.linalg.norm(X_test[:, None, :] - X_train[None, :, :], axis=2)   # (m, n) distances
    nearest = np.argsort(d, axis=1)[:, :k]                                 # (m, k) indices
    return np.array([np.bincount(y_train[row]).argmax() for row in nearest])


def scale_train_test(X_train, X_test):
    mu, sd = X_train.mean(axis=0), X_train.std(axis=0)
    return (X_train - mu) / sd, (X_test - mu) / sd
