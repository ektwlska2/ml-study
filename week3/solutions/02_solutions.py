"""Week 3 · 02 solutions. Look only after a real attempt."""
import numpy as np


def distances(X_train, q):
    return np.sqrt(((X_train - q) ** 2).sum(axis=1))


def nearest(dist, k):
    return np.argsort(dist)[:k]


def vote(labels):
    return int(np.bincount(labels).argmax())


def knn_predict_one(X_train, y_train, q, k):
    return vote(y_train[nearest(distances(X_train, q), k)])


def knn_predict(X_train, y_train, X_test, k):
    return np.array([knn_predict_one(X_train, y_train, q, k) for q in X_test])


def distance_matrix(A, B):
    return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))


def knn_predict_fast(X_train, y_train, X_test, k):
    idx = np.argsort(distance_matrix(X_test, X_train), axis=1)[:, :k]
    return np.array([vote(row) for row in y_train[idx]])


def knn_regress(X_train, y_train, X_test, k):
    idx = np.argsort(distance_matrix(X_test, X_train), axis=1)[:, :k]
    return y_train[idx].mean(axis=1)


def feature_means(X):
    return X.mean(axis=0)


def standardize(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)


def row_norms(X):
    return np.sqrt((X ** 2).sum(axis=1))


def predict_linear(X, w, b):
    return X @ w + b


def mse(y, y_hat):
    return np.mean((y - y_hat) ** 2)


def mse_gradient(X, y, w):
    return 2 / len(y) * X.T @ (X @ w - y)


def one_hot(y, k):
    return np.eye(k)[y]


def softmax_rows(Z):
    E = np.exp(Z - Z.max(axis=1, keepdims=True))
    return E / E.sum(axis=1, keepdims=True)


def class_means(X, y, k):
    return np.array([X[y == c].mean(axis=0) for c in range(k)])


def accuracy(y, y_hat):
    return np.mean(y == y_hat)
