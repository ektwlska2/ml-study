"""Week 3 · 06 solutions. Look only after a real attempt."""
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def bootstrap_indices(n, rng):
    return rng.integers(0, n, size=n)


def majority_vote(P):
    return np.array([np.bincount(col).argmax() for col in P.T])


def bagging_predict(X_train, y_train, X_test, B, seed):
    rng = np.random.default_rng(seed)
    preds = []
    for _ in range(B):
        idx = bootstrap_indices(len(X_train), rng)
        tree = DecisionTreeClassifier(random_state=0).fit(X_train[idx], y_train[idx])
        preds.append(tree.predict(X_test))
    return majority_vote(np.array(preds))


def best_stump(x, y, w):
    values = np.unique(x)
    best = (None, 1, 2.0)
    for t in (values[:-1] + values[1:]) / 2:
        for s in (1, -1):
            err = w[np.where(x > t, s, -s) != y].sum()
            if err < best[2] - 1e-12:
                best = (t, s, err)
    return best


def adaboost_fit(x, y, rounds):
    w = np.full(len(x), 1 / len(x))
    model = []
    for _ in range(rounds):
        t, s, err = best_stump(x, y, w)
        err = min(max(err, 1e-10), 1 - 1e-10)          # avoid log(0) when a stump is perfect
        alpha = 0.5 * np.log((1 - err) / err)
        h = np.where(x > t, s, -s)
        w = w * np.exp(-alpha * y * h)
        w = w / w.sum()
        model.append((t, s, alpha))
    return model


def adaboost_predict(model, x):
    F = sum(alpha * np.where(x > t, s, -s) for t, s, alpha in model)
    return np.where(F >= 0, 1, -1)


def gb_fit(x, y, rounds, lr):
    f0 = y.mean()
    F = np.full(len(y), f0)
    trees = []
    for _ in range(rounds):
        tree = DecisionTreeRegressor(max_depth=1, random_state=0).fit(x[:, None], y - F)
        F = F + lr * tree.predict(x[:, None])
        trees.append(tree)
    return f0, trees


def gb_predict(f0, trees, lr, x):
    return f0 + lr * sum(t.predict(x[:, None]) for t in trees)
