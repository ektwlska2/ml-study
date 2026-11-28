"""Week 7 · 06 Part C solutions."""
import copy
import numpy as np
import torch
import torch.nn as nn


def freeze_and_replace_head(model, n_classes):        # 01
    new = copy.deepcopy(model)
    for p in new.parameters():
        p.requires_grad = False
    new[-1] = nn.Linear(new[-1].in_features, n_classes)
    return new


def recon_errors(model, X):                            # 02
    with torch.no_grad():
        return ((model(X) - X) ** 2).mean(dim=1)


def kl_per_sample(mu, logvar):                         # 03
    return -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim=1)


def mahalanobis_scores(X_train, X):                    # 04
    mu = X_train.mean(axis=0)
    S_inv = np.linalg.inv(np.cov(X_train.T))
    D = X - mu
    return np.sqrt(np.sum(D @ S_inv * D, axis=1))


def average_precision(y, scores):                      # 04
    order = np.argsort(-scores)
    hits = y[order] == 1
    precision_at_k = np.cumsum(hits) / np.arange(1, len(y) + 1)
    return precision_at_k[hits].sum() / hits.sum()
