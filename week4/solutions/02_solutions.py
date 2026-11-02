"""Week 4 · 02 solutions. Look only after a real attempt."""
import numpy as np


def center(X):
    return X - X.mean(axis=0)


def covariance(Xc):
    return Xc.T @ Xc / (len(Xc) - 1)


def pca_eig(S):
    values, vectors = np.linalg.eigh(S)          # eigh: for symmetric matrices, sorted SMALLEST first
    order = np.argsort(values)[::-1]             # largest first
    return values[order], vectors[:, order]


def project(Xc, W):
    return Xc @ W


def reconstruct(Z, W, mean):
    return Z @ W.T + mean


def explained_ratio(values):
    return values / values.sum()


def n_components_for(values, target):
    cumulative = np.cumsum(values / values.sum())
    return int(np.argmax(cumulative >= target)) + 1


def pca_svd(X, k):
    Xc = X - X.mean(axis=0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:k]                          # (k, d): one direction per row, like scikit-learn
    return Xc @ components.T, components


class MyPCA:
    def __init__(self, k):
        self.k = k

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        values, vectors = np.linalg.eigh(Xc.T @ Xc / (len(X) - 1))
        order = np.argsort(values)[::-1][:self.k]
        self.components_ = vectors[:, order].T           # (k, d)
        self.explained_variance_ = values[order]
        return self

    def transform(self, X):
        return (X - self.mean_) @ self.components_.T     # (m, k)

    def inverse_transform(self, Z):
        return Z @ self.components_ + self.mean_         # (m, d)
