"""Week 4 · 00 solutions. Look only after a real attempt."""
import numpy as np


def count_leaves(tree):
    if "leaf" in tree:
        return 1
    return count_leaves(tree["left"]) + count_leaves(tree["right"])


def tree_depth(tree):
    if "leaf" in tree:
        return 0
    return 1 + max(tree_depth(tree["left"]), tree_depth(tree["right"]))


def normalize_rows(Z):
    return Z / Z.sum(axis=1, keepdims=True)


def majority_label(labels):
    names, codes = np.unique(labels, return_inverse=True)
    return names[np.bincount(codes).argmax()]
