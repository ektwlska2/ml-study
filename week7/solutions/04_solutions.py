"""Week 7 · 04 solutions. Look only after a real attempt."""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def core_points(X, eps, min_samples):                  # booklet 8.5, its first two lines
    nbrs = NearestNeighbors(radius=eps).fit(X).radius_neighbors(X, return_distance=False)
    return np.array([len(nb) >= min_samples for nb in nbrs])     # counts the point itself


def mahalanobis_scores(X_train, X):
    mu = X_train.mean(axis=0)
    S_inv = np.linalg.inv(np.cov(X_train.T))           # np.cov wants variables as rows: hence .T
    D = X - mu                                         # (m, d)
    return np.sqrt(np.sum(D @ S_inv * D, axis=1))      # row i: sqrt(d_i^T S^-1 d_i)


def threshold_at(scores_normal, q):
    return np.quantile(scores_normal, q)               # flag the top (1 - q) of normal data


def precision_recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    flagged, positives = np.sum(y_pred == 1), np.sum(y_true == 1)
    return (tp / flagged if flagged else 0.0), (tp / positives if positives else 0.0)


def average_precision(y, scores):
    order = np.argsort(-scores)                        # highest score first
    hits = y[order] == 1
    precision_at_k = np.cumsum(hits) / np.arange(1, len(y) + 1)
    return precision_at_k[hits].sum() / hits.sum()     # average of the precision at each true anomaly
