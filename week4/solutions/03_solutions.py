"""Week 4 · 03 solutions. Look only after a real attempt."""
import numpy as np


def assign(X, C):
    D2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)      # (n, k) squared distances
    return np.argmin(D2, axis=1)                                 # (n,)  choose among the k centroids


def update(X, labels, k):
    return np.array([X[labels == c].mean(axis=0) for c in range(k)])


def inertia(X, C, labels):
    return ((X - C[labels]) ** 2).sum()                          # C[labels]: each point's own centroid, (n, d)


def kmeans(X, C, iters):
    C = C.copy()
    labels = assign(X, C)
    for _ in range(iters):
        C = update(X, labels, len(C))
        new = assign(X, C)
        if np.array_equal(new, labels):
            break
        labels = new
    return C, labels


def kmeanspp_probs(X, C):
    D2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2).min(axis=1)   # squared distance to the NEAREST centroid
    return D2 / D2.sum()


class MyKMeans:
    def __init__(self, init):
        self.init = init

    def fit(self, X, iters=100):
        self.cluster_centers_, self.labels_ = kmeans(X, self.init, iters)
        self.inertia_ = inertia(X, self.cluster_centers_, self.labels_)
        return self

    def predict(self, X):
        return assign(X, self.cluster_centers_)


def silhouette_point(X, labels, i):
    d = np.sqrt(((X - X[i]) ** 2).sum(axis=1))                   # distances from point i to every point
    own = labels == labels[i]
    a = d[own].sum() / (own.sum() - 1)                           # mean over the OTHER members (d[i] = 0 adds nothing)
    b = min(d[labels == c].mean() for c in np.unique(labels) if c != labels[i])
    return (b - a) / max(a, b)
