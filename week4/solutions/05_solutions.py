"""Week 4 · 05 solutions. Look only after a real attempt."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def add_features(d):
    d = d.copy()
    d["night_hot"] = ((d["shift"] == "night") & (d["temp"] > 74)).astype(float)
    d["vib_spike"] = (d["vibration"] > 2).astype(float)
    return d


def make_preprocessor(num, cat):
    return ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])


def threshold_for_recall(y, scores, target):
    prec, rec, thr = precision_recall_curve(y, scores)       # thr is increasing; rec[:-1] matches thr
    return thr[rec[:-1] >= target].max()


def precision_recall_at(y, scores, t):
    pred = scores >= t
    tp = (pred & (y == 1)).sum()
    return tp / pred.sum(), tp / (y == 1).sum()


def score_by_hand(model, part):
    ct, logreg = model[0], model[-1]
    num_cols, cat_cols = ct.transformers_[0][2], ct.transformers_[1][2]
    imputer, scaler = ct.named_transformers_["num"]
    xs = part[num_cols].to_numpy(dtype=float)[0]
    xs = np.where(np.isnan(xs), imputer.statistics_, xs)                 # training medians
    xs = (xs - scaler.mean_) / scaler.scale_                              # training means and stds
    oh = np.concatenate([(cats == part[c].iloc[0]).astype(float)          # one block per categorical column
                         for c, cats in zip(cat_cols, ct.named_transformers_["cat"].categories_)])
    z = np.r_[xs, oh] @ logreg.coef_[0] + logreg.intercept_[0]
    return 1 / (1 + np.exp(-z))


def logistic_gd(A, y, lr, steps):
    w, b = np.zeros(A.shape[1]), 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(A @ w + b)))
        w = w - lr * A.T @ (p - y) / len(y)
        b = b - lr * np.mean(p - y)
    return w, b
