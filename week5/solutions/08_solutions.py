"""Week 5 · 08 Part C solutions."""
import numpy as np
import torch


def mlp_predict(X, Ws, bs):
    H = X
    for l in range(len(Ws)):
        H = H @ Ws[l] + bs[l]
        H = np.maximum(H, 0) if l < len(Ws) - 1 else H
    return H.argmax(axis=1)


def momentum_run(grad, w0, lr, beta, steps):
    w = np.array(w0, dtype=float)
    v = np.zeros_like(w)
    for _ in range(steps):
        v = beta * v + grad(w)
        w = w - lr * v
    return w


def adam_step(w, m, s, g, t, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
    m = b1 * m + (1 - b1) * g
    s = b2 * s + (1 - b2) * g ** 2
    return w - lr * (m / (1 - b1 ** t)) / (np.sqrt(s / (1 - b2 ** t)) + eps), m, s


def torch_predict(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).argmax(dim=1).numpy()


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
