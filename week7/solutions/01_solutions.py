"""Week 7 · 01 solutions. Look only after a real attempt."""
import copy
import numpy as np
import torch
import torch.nn as nn


def layernorm(X, gamma, beta, eps=1e-5):               # booklet 17.3: each ROW over its features
    mu, var = X.mean(axis=1, keepdims=True), X.var(axis=1, keepdims=True)
    return gamma * (X - mu) / np.sqrt(var + eps) + beta


def freeze_and_replace_head(model, n_classes):        # feature extraction, step 1
    new = copy.deepcopy(model)                        # never change the source model itself
    for p in new.parameters():
        p.requires_grad = False                       # freeze everything ...
    new[-1] = nn.Linear(new[-1].in_features, n_classes)   # ... then a NEW head (trainable by default)
    return new


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def two_lr_optimizer(model, lr_head, lr_body):        # fine-tuning with two learning rates
    return torch.optim.Adam([{"params": model[-1].parameters(), "lr": lr_head},
                             {"params": model[:-1].parameters(), "lr": lr_body}])
