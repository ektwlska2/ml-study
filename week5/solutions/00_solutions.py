"""Week 5 · 00 solutions. Look only after a real attempt."""
import numpy as np


def pca_fit(X, k):
    mean = X.mean(axis=0)
    Xc = X - mean
    S = Xc.T @ Xc / (len(X) - 1)
    vals, vecs = np.linalg.eigh(S)          # ascending order
    order = np.argsort(vals)[::-1][:k]      # the k largest
    return mean, vecs[:, order].T           # (k, d): one direction per row


def reweight(w, y, h, alpha):
    new = w * np.exp(-alpha * y * h)        # a NEW array; w itself is untouched
    return new / new.sum()


def linkage_distance(A, B, how):
    D = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))
    return {"single": D.min(), "complete": D.max(), "average": D.mean()}[how]


def closest_pair(clusters, X, how):
    best = None
    for a in range(len(clusters)):
        for b in range(a + 1, len(clusters)):
            d = linkage_distance(X[clusters[a]], X[clusters[b]], how)
            if best is None or d < best[2]:
                best = (a, b, d)
    return best


def agglomerative(X, k, how):
    clusters = [[i] for i in range(len(X))]
    while len(clusters) > k:
        a, b, _ = closest_pair(clusters, X, how)
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]                     # b > a, so deleting b doesn't move a
    labels = np.empty(len(X), dtype=int)
    for c, members in enumerate(clusters):
        labels[members] = c
    return labels
