"""Week 1 solutions. Look only after a real attempt."""
import numpy as np


class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

    def perimeter(self):
        return 2 * (self.width + self.height)

    def __repr__(self):
        return f"Rectangle({self.width}, {self.height})"


class Stack:
    def __init__(self):
        self.items = []          # a new list for EACH stack

    def push(self, x):
        self.items.append(x)

    def pop(self):
        return self.items.pop() if self.items else None

    def peek(self):
        return self.items[-1] if self.items else None

    def is_empty(self):
        return len(self.items) == 0

    def __len__(self):
        return len(self.items)


class Square(Rectangle):
    def __init__(self, side):
        super().__init__(side, side)


class RunningMean:
    def __init__(self):
        self.n = 0
        self.total = 0

    def add(self, x):
        self.n += 1
        self.total += x

    def mean(self):
        return self.total / self.n if self.n else None


def make_grid(n):
    return np.arange(n * n).reshape(n, n)


def column_means(X):
    return X.mean(axis=0)


def standardize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)


def count_above(a, t):
    return int((a > t).sum())


def row_argmax(scores):
    return scores.argmax(axis=1)


def mse(y, y_hat):
    return np.mean((y - y_hat) ** 2)


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def linear_predict(X, w, b):
    return X @ w + b
