"""Week 4 · 04 solutions. Look only after a real attempt."""
import numpy as np


def linkage_distance(A, B, how):
    D = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))   # every point of A vs every point of B
    return {"single": D.min(), "complete": D.max(), "average": D.mean()}[how]


def dbscan_core(X, eps, min_pts):
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))   # (n, n)
    return (D <= eps).sum(axis=1) >= min_pts                           # each point counts itself (distance 0)


def gaussian_pdf(x, mu, var):
    return np.exp(-(x - mu) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)


def e_step(x, mu, var, pi):
    N = np.stack([pi[k] * gaussian_pdf(x, mu[k], var[k]) for k in range(len(mu))], axis=1)   # (n, K)
    return N / N.sum(axis=1, keepdims=True)


def m_step(x, R):
    Nk = R.sum(axis=0)                                           # (K,) "how many points" each component has
    mu = (R * x[:, None]).sum(axis=0) / Nk
    var = (R * (x[:, None] - mu) ** 2).sum(axis=0) / Nk
    return mu, var, Nk / len(x)


def log_likelihood(x, mu, var, pi):
    p = sum(pi[k] * gaussian_pdf(x, mu[k], var[k]) for k in range(len(mu)))
    return np.log(p).sum()


def dbscan_predict(X_train, labels, core, x_new, eps):
    d = np.sqrt(((X_train - x_new) ** 2).sum(axis=1))
    ok = core & (d <= eps)                          # only CORE points within eps can take in a new point
    if not ok.any():
        return -1
    idx = np.where(ok)[0]
    return labels[idx[np.argmin(d[idx])]]


def dbscan(X, eps, min_pts):
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    neighbours = D <= eps                           # (n, n)
    core = neighbours.sum(axis=1) >= min_pts
    labels = np.full(len(X), -1)
    cluster = 0
    for i in range(len(X)):
        if not core[i] or labels[i] != -1:
            continue
        labels[i] = cluster
        todo = [i]
        while todo:
            j = todo.pop()
            for nb in np.where(neighbours[j])[0]:
                if labels[nb] == -1:
                    labels[nb] = cluster
                    if core[nb]:                    # only core points spread the cluster further
                        todo.append(nb)
        cluster += 1
    return labels


def gmm_fit_1d(x, mu, var, pi, iters):
    for _ in range(iters):
        R = e_step(x, mu, var, pi)
        mu, var, pi = m_step(x, R)
    return mu, var, pi
