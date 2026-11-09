"""Week 5 · 06 solutions. Look only after a real attempt."""
import numpy as np
import torch
import torch.nn as nn


def gaussian_pdf(x, mu, var):
    return np.exp(-(x - mu) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)


def e_step(x, mu, var, pi):
    m = pi[None, :] * gaussian_pdf(x[:, None], mu[None, :], var[None, :])
    return m / m.sum(axis=1, keepdims=True)


def to_tensors(X, y):
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.int64)


def linear_from_numpy(W, b):
    lin = nn.Linear(W.shape[0], W.shape[1])                  # (in, out)
    with torch.no_grad():                                    # setting numbers is not part of training
        lin.weight.copy_(torch.tensor(W.T, dtype=torch.float32))   # PyTorch stores (out, in)
        lin.bias.copy_(torch.tensor(b, dtype=torch.float32))
    return lin


def make_mlp(sizes):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)                            # *layers: pass the list's items as separate arguments


def train_epoch(model, loader, loss_fn, opt):
    model.train()
    total = 0.0
    for xb, yb in loader:
        loss = loss_fn(model(xb), yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(xb)                       # the batch mean, times the batch size
    return total / len(loader.dataset)


def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            correct += (model(xb).argmax(dim=1) == yb).sum().item()
    return correct / len(loader.dataset)


class DigitsMLP(nn.Module):
    def __init__(self, d_in=64, hidden=32, n_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden)
        self.fc2 = nn.Linear(hidden, n_classes)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))
