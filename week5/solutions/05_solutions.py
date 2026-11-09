"""Week 5 · 05 solutions. Look only after a real attempt."""
import numpy as np


def numerical_grad(f, theta, eps=1e-5):
    g = np.zeros_like(theta, dtype=float)
    for idx in np.ndindex(theta.shape):
        old = theta[idx]
        theta[idx] = old + eps; up = f(theta)
        theta[idx] = old - eps; down = f(theta)
        theta[idx] = old
        g[idx] = (up - down) / (2 * eps)
    return g


def gd_steps(grad, w0, lr, steps):
    path = [np.array(w0, dtype=float)]            # a copy: never change the caller's w0
    for _ in range(steps):
        path.append(path[-1] - lr * grad(path[-1]))
    return np.array(path)


def minibatches(n, b, rng):
    perm = rng.permutation(n)                     # the indices 0..n-1 in a random order
    return [perm[start:start + b] for start in range(0, n, b)]


def momentum_step(w, v, g, lr, beta):
    v = beta * v + g                              # new arrays; the inputs are untouched
    return w - lr * v, v


def rmsprop_step(w, s, g, lr, rho=0.99, eps=1e-8):
    s = rho * s + (1 - rho) * g ** 2
    return w - lr * g / (np.sqrt(s) + eps), s


def adam_step(w, m, s, g, t, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
    m = b1 * m + (1 - b1) * g
    s = b2 * s + (1 - b2) * g ** 2
    m_hat = m / (1 - b1 ** t)
    s_hat = s / (1 - b2 ** t)
    return w - lr * m_hat / (np.sqrt(s_hat) + eps), m, s


class MyAdam:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = params                      # the model's own arrays (not copies)
        self.lr, (self.b1, self.b2), self.eps = lr, betas, eps
        self.m = [np.zeros_like(p) for p in params]
        self.s = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for p, g, m, s in zip(self.params, grads, self.m, self.s):
            m *= self.b1; m += (1 - self.b1) * g             # in place: these are our own state arrays
            s *= self.b2; s += (1 - self.b2) * g ** 2
            m_hat = m / (1 - self.b1 ** self.t)
            s_hat = s / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(s_hat) + self.eps)  # in place ON PURPOSE: the model sees the change
