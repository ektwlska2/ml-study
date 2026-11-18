"""Week 6 · 02 solutions. Look only after a real attempt."""
import numpy as np


def ridge_closed_form(X1, y, lam):                    # booklet 1.8
    P = np.eye(X1.shape[1])
    P[0, 0] = 0                                       # don't penalize the intercept
    return np.linalg.solve(X1.T @ X1 + lam * P, X1.T @ y)


def sgd_weight_decay_step(w, g, lr, lam):
    return w - lr * (g + lam * w)                     # = (1 - lr*lam) * w - lr * g


def inverted_dropout(h, p, rng):
    mask = rng.random(h.shape) >= p                   # keep each unit with probability 1 - p
    return h * mask / (1 - p), mask                   # scale the survivors so the mean stays h


def early_stop(val_losses, patience):
    best = 0
    for epoch in range(len(val_losses)):
        if val_losses[epoch] < val_losses[best]:
            best = epoch                              # a new best: remember it
        elif epoch - best >= patience:
            return epoch, best                        # too long without a new best: stop here
    return len(val_losses) - 1, best
