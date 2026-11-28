"""Week 7 · 05 Part II solutions. Look only after marking yourself."""
import numpy as np
import torch


def kmeans(X, k, n_iter=100, seed=0):                  # booklet 7.4
    rng = np.random.default_rng(seed)
    mu = X[rng.choice(len(X), size=k, replace=False)]  # initialize: k random data points
    for _ in range(n_iter):
        dist = np.linalg.norm(X[:, None, :] - mu[None, :, :], axis=2)   # (n, k)
        c = dist.argmin(axis=1)                        # assignment step
        new_mu = np.array([X[c == j].mean(axis=0) for j in range(k)])   # update step
        if np.allclose(new_mu, mu):
            break
        mu = new_mu
    J = np.sum((X - mu[c]) ** 2)
    return mu, c, J


def mlp_forward(X, W1, b1, W2, b2):                    # booklet 12.3: sigmoid hidden layer, logits out
    H = torch.sigmoid(X @ W1 + b1)
    return H @ W2 + b2


def ae_flags(model, X_train, X, q):                    # 04 §E in one function
    with torch.no_grad():
        s_train = ((model(X_train) - X_train) ** 2).mean(dim=1).numpy()
        s = ((model(X) - X) ** 2).mean(dim=1).numpy()
    return s > np.quantile(s_train, q)


def freeze_first(model, n):                            # 01 §3, remark (1)
    for p in model[:n].parameters():
        p.requires_grad = False
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def auc_pairs(y, s):                                   # 04 §5: ROC-AUC = fraction of (anomaly, normal) pairs ranked right
    pos, neg = s[y == 1], s[y == 0]
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return wins / (len(pos) * len(neg))
