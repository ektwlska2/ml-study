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
