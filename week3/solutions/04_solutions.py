"""Week 3 · 04 solutions. Look only after a real attempt."""
import numpy as np


def gini(y):
    if len(y) == 0:
        return 0.0
    p = np.bincount(y) / len(y)
    return 1 - np.sum(p ** 2)


def entropy(y):
    if len(y) == 0:
        return 0.0
    p = np.bincount(y) / len(y)
    p = p[p > 0]                      # 0 · log 0 counts as 0
    return -np.sum(p * np.log2(p))


def split_gain(y, mask, impurity):
    n = len(y)
    return impurity(y) - mask.sum() / n * impurity(y[mask]) - (~mask).sum() / n * impurity(y[~mask])


def best_threshold(x, y):
    values = np.unique(x)
    best_t, best_g = None, 0.0
    for t in (values[:-1] + values[1:]) / 2:
        g = split_gain(y, x <= t, gini)
        if best_t is None or g > best_g + 1e-12:
            best_t, best_g = t, g
    return best_t, best_g


def best_split(X, y):
    best = (None, None, 0.0)
    for j in range(X.shape[1]):
        t, g = best_threshold(X[:, j], y)
        if t is not None and (best[0] is None or g > best[2] + 1e-12):
            best = (j, t, g)
    return best


def build_tree(X, y, max_depth):
    majority = int(np.bincount(y).argmax())
    if max_depth == 0 or len(np.unique(y)) == 1:
        return {"leaf": majority}
    j, t, g = best_split(X, y)
    if j is None or g <= 1e-12:
        return {"leaf": majority}
    mask = X[:, j] <= t
    return {"feature": j, "threshold": t,
            "left": build_tree(X[mask], y[mask], max_depth - 1),
            "right": build_tree(X[~mask], y[~mask], max_depth - 1)}


def predict_tree(tree, X):
    out = []
    for x in X:
        node = tree
        while "leaf" not in node:
            node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
        out.append(node["leaf"])
    return np.array(out)
