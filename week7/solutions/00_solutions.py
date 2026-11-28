"""Week 7 · 00 solutions. Look only after a real attempt."""
import numpy as np


def maxpool2d(x, size=2):                              # booklet 18.5
    h, w = x.shape
    return x.reshape(h // size, size, w // size, size).max(axis=(1, 3))   # (I, a, J, b): max over a and b
