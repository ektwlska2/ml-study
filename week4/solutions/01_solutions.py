"""Week 4 · 01 solutions. Look only after a real attempt."""
import numpy as np


# ---------------- AdaBoost
def stump_predict(x, t, s):
    return np.where(x > t, s, -s)


def weighted_error(y, h, w):
    return w[h != y].sum()


def best_stump(x, y, w):
    values = np.unique(x)
    best = (None, 1, 2.0)
    for t in (values[:-1] + values[1:]) / 2:
        for s in (1, -1):
            err = weighted_error(y, stump_predict(x, t, s), w)
            if err < best[2] - 1e-12:
                best = (t, s, err)
    return best


def stump_alpha(err):
    return 0.5 * np.log((1 - err) / err)


def update_weights(w, y, h, alpha):
    w = w * np.exp(-alpha * y * h)
    return w / w.sum()


def adaboost_fit(x, y, rounds):
    w = np.full(len(x), 1 / len(x))
    model = []
    for _ in range(rounds):
        t, s, err = best_stump(x, y, w)
        err = min(max(err, 1e-10), 1 - 1e-10)
        alpha = stump_alpha(err)
        w = update_weights(w, y, stump_predict(x, t, s), alpha)
        model.append((t, s, alpha))
    return model


def adaboost_predict(model, x):
    F = sum(alpha * stump_predict(x, t, s) for t, s, alpha in model)
    return np.where(F >= 0, 1, -1)


# ---------------- gradient boosting
def residuals(y, F):
    return y - F


def fit_reg_stump(x, r):
    values = np.unique(x)
    best = None
    for t in (values[:-1] + values[1:]) / 2:
        left, right = r[x <= t], r[x > t]
        sse = ((left - left.mean()) ** 2).sum() + ((right - right.mean()) ** 2).sum()
        if best is None or sse < best[0] - 1e-12:
            best = (sse, t, left.mean(), right.mean())
    return best[1], best[2], best[3]


def reg_stump_predict(x, t, left_value, right_value):
    return np.where(x <= t, left_value, right_value)


def gb_fit(x, y, rounds, lr):
    f0 = y.mean()
    F = np.full(len(y), f0)
    stumps = []
    for _ in range(rounds):
        t, a, b = fit_reg_stump(x, residuals(y, F))
        F = F + lr * reg_stump_predict(x, t, a, b)
        stumps.append((t, a, b))
    return f0, stumps


def gb_predict(f0, stumps, lr, x):
    return f0 + lr * sum(reg_stump_predict(x, t, a, b) for t, a, b in stumps)
