"""Exercise checkers for Week 3. Each notebook's setup cell runs `checks.load(globals(), "02")`, which defines `check(n)`.

The tests build random data every run and compare your function with a reference. Reading this file spoils the exercises."""

SUITES = {
"02": r'''
_rng = np.random.default_rng()

def _ref_vote(l):
    return int(np.bincount(l).argmax())

def _ref_knn(Xtr, ytr, Xte, k):
    D = np.sqrt(((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(axis=2))
    return np.array([_ref_vote(ytr[np.argsort(r)[:k]]) for r in D])

def _sizes():
    return int(_rng.integers(8, 40)), int(_rng.integers(2, 5))

def _t1():
    n, d = _sizes(); X = _rng.normal(size=(n, d)); q = _rng.normal(size=d)
    out = distances(X, q)
    return out.shape == (n,) and np.allclose(out, np.linalg.norm(X - q, axis=1))

def _t2():
    dist = _rng.permutation(int(_rng.integers(8, 30))).astype(float); k = int(_rng.integers(1, 6))
    out = nearest(dist, k)
    return np.shape(out) == (k,) and list(out) == list(np.argsort(dist)[:k])

def _t3():
    ok = vote(np.array([1, 0, 1])) == 1 and vote(np.array([0, 0, 2, 2])) == 0
    for _ in range(20):
        l = _rng.integers(0, 4, size=int(_rng.integers(1, 8)))
        ok = ok and vote(l) == _ref_vote(l)
    return bool(ok)

def _t4():
    n, d = _sizes(); X = _rng.normal(size=(n, d)); y = _rng.integers(0, 3, n); q = _rng.normal(size=d); k = int(_rng.choice([1, 3, 5]))
    return knn_predict_one(X, y, q, k) == _ref_knn(X, y, q[None, :], k)[0]

def _t5():
    n, d = _sizes(); X = _rng.normal(size=(n, d)); y = _rng.integers(0, 3, n); m = int(_rng.integers(3, 12)); T = _rng.normal(size=(m, d)); k = int(_rng.choice([1, 3, 5]))
    out = knn_predict(X, y, T, k)
    return np.shape(out) == (m,) and np.array_equal(np.asarray(out), _ref_knn(X, y, T, k))

def _t6():
    m, n, d = int(_rng.integers(2, 9)), int(_rng.integers(2, 9)), int(_rng.integers(1, 5))
    A = _rng.normal(size=(m, d)); B = _rng.normal(size=(n, d))
    out = distance_matrix(A, B)
    ref = np.array([[np.linalg.norm(a - b) for b in B] for a in A])
    return out.shape == (m, n) and np.allclose(out, ref)

def _t7():
    n, d = _sizes(); X = _rng.normal(size=(n, d)); y = _rng.integers(0, 3, n); m = int(_rng.integers(3, 12)); T = _rng.normal(size=(m, d)); k = int(_rng.choice([1, 3, 5]))
    out = knn_predict_fast(X, y, T, k)
    return np.shape(out) == (m,) and np.array_equal(np.asarray(out), _ref_knn(X, y, T, k))

def _t8():
    n, d = _sizes(); X = _rng.normal(size=(n, d)); y = _rng.normal(size=n); m = int(_rng.integers(3, 12)); T = _rng.normal(size=(m, d)); k = int(_rng.choice([1, 3, 5]))
    D = np.sqrt(((T[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    ref = np.array([y[np.argsort(r)[:k]].mean() for r in D])
    out = knn_regress(X, y, T, k)
    return np.shape(out) == (m,) and np.allclose(out, ref)

def _Xd():
    return _rng.normal(3, 2, size=(int(_rng.integers(10, 40)), int(_rng.integers(2, 6))))

def _t9():
    X = _Xd(); out = feature_means(X)
    return np.shape(out) == (X.shape[1],) and np.allclose(out, X.mean(axis=0))

def _t10():
    X = _Xd(); Z = standardize(X)
    return Z.shape == X.shape and np.allclose(Z.mean(axis=0), 0) and np.allclose(Z.std(axis=0), 1)

def _t11():
    X = _Xd(); out = row_norms(X)
    return np.shape(out) == (X.shape[0],) and np.allclose(out, np.linalg.norm(X, axis=1))

def _t12():
    X = _Xd(); w = _rng.normal(size=X.shape[1]); b = float(_rng.normal())
    out = predict_linear(X, w, b)
    return np.shape(out) == (X.shape[0],) and np.allclose(out, X @ w + b)

def _t13():
    y = _rng.normal(size=30); yh = _rng.normal(size=30); out = mse(y, yh)
    return np.ndim(out) == 0 and np.isclose(out, np.mean((y - yh) ** 2))

def _t14():
    X = _Xd(); w = _rng.normal(size=X.shape[1]); y = _rng.normal(size=X.shape[0])
    out = mse_gradient(X, y, w)
    return np.shape(out) == (X.shape[1],) and np.allclose(out, 2 / len(y) * X.T @ (X @ w - y))

def _t15():
    k = int(_rng.integers(2, 6)); y = _rng.integers(0, k, size=int(_rng.integers(5, 20)))
    out = one_hot(y, k)
    return np.shape(out) == (len(y), k) and np.array_equal(out, np.eye(k)[y])

def _t16():
    Z = _rng.normal(size=(int(_rng.integers(2, 9)), int(_rng.integers(2, 6)))) * 5
    out = softmax_rows(Z)
    ref = np.exp(Z) / np.exp(Z).sum(axis=1, keepdims=True)
    big = softmax_rows(np.array([[1000.0, 1001.0]]))
    return out.shape == Z.shape and np.allclose(out, ref) and np.all(np.isfinite(big)) and np.allclose(big, [[0.26894142, 0.73105858]])

def _t17():
    k = int(_rng.integers(2, 5)); X = _Xd(); y = np.r_[np.arange(k), _rng.integers(0, k, size=X.shape[0] - k)]
    out = class_means(X, y, k)
    ref = np.array([X[y == c].mean(axis=0) for c in range(k)])
    return np.shape(out) == (k, X.shape[1]) and np.allclose(out, ref)

def _t18():
    y = _rng.integers(0, 2, 40); yh = _rng.integers(0, 2, 40); out = accuracy(y, yh)
    return np.ndim(out) == 0 and np.isclose(out, (y == yh).mean())

TESTS = {i: globals()[f"_t{i}"] for i in range(1, 19)}
''',
"03": r'''
import pandas as pd
_rng = np.random.default_rng()

def _parts(n=None):
    n = n or int(_rng.integers(25, 60))
    temp = _rng.normal(73, 3, n).round(1)
    temp[_rng.choice(n, size=max(1, n // 10), replace=False)] = np.nan
    lines = _rng.choice(list("ABCD"), n); lines[:4] = list("ABCD")          # every line appears
    days = pd.Timestamp("2026-10-05") + pd.to_timedelta(_rng.integers(0, 14, n), unit="D")
    return pd.DataFrame({"part": [f"P{i:03d}" for i in range(n)], "line": lines,
                         "shift": _rng.choice(["day", "night"], n), "temp": temp,
                         "pressure": _rng.integers(90, 110, n), "defect": _rng.integers(0, 2, n),
                         "date": days.strftime("%Y-%m-%d")})

def _same(a, b):
    """Two Series/DataFrames with the same labels and (almost) the same numbers."""
    a, b = pd.DataFrame(a), pd.DataFrame(b)
    if a.shape != b.shape or list(a.index) != list(b.index):
        return False
    return all(np.allclose(a.iloc[:, j].to_numpy(float), b.iloc[:, j].to_numpy(float), equal_nan=True) for j in range(a.shape[1]))

def _t1():
    df = _parts(); out = summary(df)
    return tuple(int(v) for v in out) == (df.shape[0], df.shape[1], int(df.isna().sum().sum()))

def _t2():
    df = _parts(); t = float(_rng.uniform(70, 76)); out = hot_night(df, t)
    ref = df[(df["shift"] == "night") & (df["temp"] > t)]
    return isinstance(out, pd.DataFrame) and out.equals(ref)

def _t3():
    df = _parts(); ls = list(_rng.choice(list("ABCD"), 2, replace=False)); out = parts_on_lines(df, ls)
    return out == df.loc[df["line"].isin(ls), "part"].tolist()

def _t4():
    df = _parts(); k = int(_rng.integers(2, 6)); col = str(_rng.choice(["temp", "pressure"]))
    out = top_k(df, col, k)
    ref = df.sort_values(col, ascending=False).head(k).reset_index(drop=True)
    return list(out.index) == list(range(k)) and np.allclose(out[col].to_numpy(float), ref[col].to_numpy(float))

def _t5():
    df = _parts(); before = df.copy(); out = add_columns(df)
    ok = df.equals(before) and "temp_f" in out and "hot" in out
    return ok and np.allclose(out["temp_f"], df["temp"] * 9 / 5 + 32, equal_nan=True) and out["hot"].tolist() == (df["temp"] > 75).tolist()

def _t6():
    df = _parts(); before = df.copy(); out = fill_temp_by_line(df)
    ref = df["temp"].fillna(df.groupby("line")["temp"].transform("mean"))
    return df.equals(before) and np.allclose(out["temp"], ref) and out["temp"].isna().sum() == 0

def _t7():
    df = _parts(); col = str(_rng.choice(["line", "shift"])); out = defect_rate_by(df, col)
    return _same(out, df.groupby(col)["defect"].mean())

def _t8():
    df = _parts(); out = line_stats(df)
    ref = df.groupby("line").agg(n=("part", "size"), mean_temp=("temp", "mean"), max_pressure=("pressure", "max"))
    return list(out.columns) == ["n", "mean_temp", "max_pressure"] and _same(out, ref)

def _t9():
    df = _parts(); sup = pd.DataFrame({"line": ["A", "B", "E"], "supervisor": ["Kim", "Lee", "Choi"]})
    out = attach_supervisor(df, sup)
    ref = df.merge(sup, on="line", how="left")
    return len(out) == len(df) and out["supervisor"].fillna("?").tolist() == ref["supervisor"].fillna("?").tolist()

def _t10():
    df = _parts(); out = defects_table(df)
    ref = df.pivot_table(index="line", columns="shift", values="defect", aggfunc="sum", fill_value=0)
    return _same(out, ref) and list(out.columns) == list(ref.columns)

def _t11():
    df = _parts(); out = weekday_counts(df)
    return _same(out, pd.to_datetime(df["date"]).dt.day_name().value_counts().sort_index()) and list(out.index) == sorted(out.index)

def _t12():
    df = _parts(); X, y = to_xy(df)
    ref = pd.get_dummies(df[["temp", "pressure", "line", "shift", "defect"]].dropna(), columns=["line", "shift"], dtype=float)
    Xr = ref.drop(columns="defect").to_numpy(); yr = ref["defect"].to_numpy()
    return isinstance(X, np.ndarray) and X.shape == Xr.shape and np.allclose(X, Xr) and np.array_equal(y, yr)

TESTS = {i: globals()[f"_t{i}"] for i in range(1, 13)}
''',
"04": r'''
_rng = np.random.default_rng()

def _gini(y):
    if len(y) == 0:
        return 0.0
    p = np.bincount(y) / len(y)
    return 1 - np.sum(p ** 2)

def _entropy(y):
    if len(y) == 0:
        return 0.0
    p = np.bincount(y) / len(y); p = p[p > 0]
    return -np.sum(p * np.log2(p))

def _gain(y, mask, imp=_gini):
    n = len(y)
    return imp(y) - (mask.sum() / n) * imp(y[mask]) - ((~mask).sum() / n) * imp(y[~mask])

def _cands(x):
    u = np.unique(x)
    return (u[:-1] + u[1:]) / 2

def _labels(n, k=2):
    y = _rng.integers(0, k, n); y[:k] = np.arange(k)
    return y

def _t1():
    ok = np.isclose(gini(np.array([0, 0, 1, 1])), 0.5) and np.isclose(gini(np.array([1, 1, 1])), 0.0)
    for _ in range(10):
        y = _labels(int(_rng.integers(5, 30)), int(_rng.integers(2, 5)))
        ok = ok and np.isclose(gini(y), _gini(y))
    return bool(ok)

def _t2():
    ok = np.isclose(entropy(np.array([0, 1])), 1.0) and np.isclose(entropy(np.array([2, 2, 2])), 0.0)
    for _ in range(10):
        y = _labels(int(_rng.integers(5, 30)), int(_rng.integers(2, 5)))
        ok = ok and np.isclose(entropy(y), _entropy(y))
    return bool(ok)

def _t3():
    ok = True
    for _ in range(10):
        y = _labels(int(_rng.integers(6, 30))); mask = _rng.random(len(y)) < 0.5
        ok = ok and np.isclose(split_gain(y, mask, _gini), _gain(y, mask, _gini)) and np.isclose(split_gain(y, mask, _entropy), _gain(y, mask, _entropy))
    return bool(ok)

def _t4():
    ok = True
    for _ in range(10):
        n = int(_rng.integers(6, 30)); x = _rng.integers(0, 12, n).astype(float); y = _labels(n)
        t, g = best_threshold(x, y)
        best = max(_gain(y, x <= c) for c in _cands(x))
        ok = ok and t in set(_cands(x)) and np.isclose(g, best) and np.isclose(_gain(y, x <= t), best)
    return bool(ok) and best_threshold(np.ones(5), np.array([0, 1, 0, 1, 1]))[0] is None

def _t5():
    ok = True
    for _ in range(10):
        n = int(_rng.integers(8, 30)); d = int(_rng.integers(2, 4)); X = _rng.integers(0, 10, (n, d)).astype(float); y = _labels(n)
        j, t, g = best_split(X, y)
        best = max((_gain(y, X[:, k] <= c) for k in range(d) for c in _cands(X[:, k])), default=0.0)
        ok = ok and 0 <= j < d and np.isclose(g, best) and np.isclose(_gain(y, X[:, j] <= t), best)
    return bool(ok)

def _depth(tree):
    return 0 if "leaf" in tree else 1 + max(_depth(tree["left"]), _depth(tree["right"]))

def _t6():
    temp = np.array([68, 70, 71, 72, 73, 74, 75, 77, 78, 80.]); night = np.array([0, 0, 1, 0, 1, 0, 1, 1, 0, 1.])
    y = np.array([0, 0, 0, 0, 1, 0, 1, 1, 0, 1])
    tree = build_tree(np.column_stack([temp, night]), y, max_depth=2)
    ok = tree.get("feature") == 1 and "leaf" in tree["left"] and tree["left"]["leaf"] == 0
    ok = ok and tree["right"].get("feature") == 0 and np.isclose(tree["right"]["threshold"], 72.0)
    X = _rng.normal(size=(60, 2)); yy = (X[:, 0] + X[:, 1] > 0).astype(int)
    full = build_tree(X, yy, max_depth=20)
    pred = np.array([_walk(full, x) for x in X])
    return bool(ok) and np.array_equal(pred, yy) and _depth(build_tree(X, yy, max_depth=3)) <= 3

def _walk(tree, x):
    while "leaf" not in tree:
        tree = tree["left"] if x[tree["feature"]] <= tree["threshold"] else tree["right"]
    return tree["leaf"]

def _t7():
    tree = {"feature": 1, "threshold": 0.5, "left": {"leaf": 0},
            "right": {"feature": 0, "threshold": 72.0, "left": {"leaf": 0}, "right": {"leaf": 1}}}
    X = np.column_stack([_rng.uniform(65, 80, 30), _rng.integers(0, 2, 30)])
    out = predict_tree(tree, X)
    return np.shape(out) == (30,) and np.array_equal(np.asarray(out), np.array([_walk(tree, x) for x in X]))

TESTS = {i: globals()[f"_t{i}"] for i in range(1, 8)}
''',
"05": r'''
_rng = np.random.default_rng()

def _need(x):
    if x is None:
        raise TypeError("'NoneType' returned")
    return x

def _sep(n=None, d=None):
    n = n or int(_rng.integers(30, 70)); d = d or int(_rng.integers(2, 4))
    X = _rng.normal(size=(n, d)); w = _rng.normal(size=d)
    y = np.where(X @ w + 0.2 > 0, 1, -1)
    return X, y

def _t1():
    X, _ = _sep(); w = _rng.normal(size=X.shape[1]); b = float(_rng.normal())
    out = _need(signed_distance(X, w, b))
    return np.shape(out) == (len(X),) and np.allclose(out, (X @ w + b) / np.linalg.norm(w))

def _hl(X, y, w, b, lam):
    return np.mean(np.maximum(0, 1 - y * (X @ w + b))) + lam / 2 * w @ w

def _t2():
    X, y = _sep(); w = _rng.normal(size=X.shape[1]); b = float(_rng.normal()); lam = float(_rng.uniform(0, 1))
    out = _need(hinge_loss(X, y, w, b, lam))
    return np.ndim(out) == 0 and np.isclose(out, _hl(X, y, w, b, lam))

def _hg(X, y, w, b, lam):
    a = y * (X @ w + b) < 1
    return lam * w - (y[a, None] * X[a]).sum(axis=0) / len(y), -y[a].sum() / len(y)

def _t3():
    X, y = _sep(); w = _rng.normal(size=X.shape[1]); b = float(_rng.normal()); lam = float(_rng.uniform(0, 1))
    gw, gb = hinge_grad(X, y, w, b, lam)
    rw, rb = _hg(X, y, w, b, lam)
    return np.shape(gw) == w.shape and np.allclose(gw, rw) and np.isclose(gb, rb)

def _t4():
    X, y = _sep(); lam = 0.01; lr = float(_rng.choice([0.1, 0.3])); steps = int(_rng.integers(100, 400))
    w, b = svm_train(X, y, lam, lr, steps)
    rw, rb = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        gw, gb = _hg(X, y, rw, rb, lam)
        rw, rb = rw - lr * gw, rb - lr * gb
    acc = (np.sign(X @ w + b) == y).mean()
    return np.allclose(w, rw) and np.isclose(b, rb) and acc >= 0.9

def _AB():
    d = int(_rng.integers(1, 4))
    return _rng.normal(size=(int(_rng.integers(2, 8)), d)), _rng.normal(size=(int(_rng.integers(2, 8)), d))

def _t5():
    A, B = _AB(); deg = int(_rng.integers(1, 4)); c = float(_rng.uniform(0, 2))
    out = _need(poly_kernel(A, B, deg, c))
    return np.shape(out) == (len(A), len(B)) and np.allclose(out, (A @ B.T + c) ** deg)

def _t6():
    A, B = _AB(); g = float(_rng.uniform(0.1, 2))
    out = _need(rbf_kernel(A, B, g))
    ref = np.array([[np.exp(-g * np.sum((a - b) ** 2)) for b in B] for a in A])
    return np.shape(out) == (len(A), len(B)) and np.allclose(out, ref)

def _t7():
    A = _rng.normal(size=(5, 2)); B = _rng.normal(size=(4, 2))
    PA, PB = _need(phi2(A)), _need(phi2(B))
    return np.shape(PA) == (5, 6) and np.allclose(PA @ PB.T, (A @ B.T + 1) ** 2)

TESTS = {i: globals()[f"_t{i}"] for i in range(1, 8)}
''',
"06": r'''
_rng = np.random.default_rng()

def _need(x):
    if x is None:
        raise TypeError("'NoneType' returned")
    return x
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

def _t1():
    n = int(_rng.integers(5, 50)); s = int(_rng.integers(0, 10 ** 6))
    out = _need(bootstrap_indices(n, np.random.default_rng(s)))
    return np.shape(out) == (n,) and np.array_equal(out, np.random.default_rng(s).integers(0, n, size=n))

def _t2():
    B, m = int(_rng.integers(2, 9)), int(_rng.integers(3, 15)); P = _rng.integers(0, 3, (B, m))
    out = _need(majority_vote(P))
    ref = np.array([np.bincount(P[:, j]).argmax() for j in range(m)])
    return np.shape(out) == (m,) and np.array_equal(out, ref)

def _bag_ref(Xtr, ytr, Xte, B, seed):
    rng = np.random.default_rng(seed); n = len(Xtr); P = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        P.append(DecisionTreeClassifier(random_state=0).fit(Xtr[idx], ytr[idx]).predict(Xte))
    P = np.array(P)
    return np.array([np.bincount(P[:, j]).argmax() for j in range(P.shape[1])])

def _t3():
    X = _rng.normal(size=(60, 2)); y = (X[:, 0] * X[:, 1] > 0).astype(int); T = _rng.normal(size=(15, 2))
    B = int(_rng.integers(3, 12)); s = int(_rng.integers(0, 1000))
    out = _need(bagging_predict(X, y, T, B, s))
    return np.shape(out) == (15,) and np.array_equal(out, _bag_ref(X, y, T, B, s))

def _stump_ref(x, y, w):
    xs = np.unique(x); best = (None, 1, 2.0)
    for t in (xs[:-1] + xs[1:]) / 2:
        for s in (1, -1):
            e = w[np.where(x > t, s, -s) != y].sum()
            if e < best[2] - 1e-12:
                best = (t, s, e)
    return best

def _ada_ref(x, y, rounds):
    w = np.full(len(x), 1 / len(x)); model = []
    for _ in range(rounds):
        t, s, e = _stump_ref(x, y, w)
        e = min(max(e, 1e-10), 1 - 1e-10)
        a = 0.5 * np.log((1 - e) / e); h = np.where(x > t, s, -s)
        w = w * np.exp(-a * y * h); w = w / w.sum(); model.append((t, s, a))
    return model

def _t4():
    x = np.arange(1, 9.0); y = np.array([1, 1, -1, -1, 1, 1, -1, -1.0])
    out = adaboost_fit(x, y, 3)
    ok = len(out) == 3 and np.allclose([m[0] for m in out], [2.5, 6.5, 4.5]) and np.allclose([m[2] for m in out], [0.5493, 0.8047, 0.6931], atol=1e-4)
    x2 = np.sort(_rng.uniform(0, 10, 12)); y2 = _rng.choice([-1.0, 1.0], 12); r = int(_rng.integers(2, 6))
    out2 = adaboost_fit(x2, y2, r); ref = _ada_ref(x2, y2, r)
    return bool(ok) and len(out2) == r and np.allclose(np.array(out2, float), np.array(ref, float))

def _t5():
    model = [(2.5, -1, 0.5493), (6.5, -1, 0.8047), (4.5, 1, 0.6931)]
    x = _rng.uniform(0, 9, 20)
    F = sum(a * np.where(x > t, s, -s) for t, s, a in model)
    out = _need(adaboost_predict(model, x))
    return np.shape(out) == (20,) and np.array_equal(out, np.where(F >= 0, 1, -1))

def _gb_ref(x, y, rounds, lr):
    f0 = y.mean(); F = np.full(len(y), f0); trees = []
    for _ in range(rounds):
        t = DecisionTreeRegressor(max_depth=1, random_state=0).fit(x[:, None], y - F)
        F = F + lr * t.predict(x[:, None]); trees.append(t)
    return f0, trees

def _t6():
    x = np.sort(_rng.uniform(0, 6, int(_rng.integers(15, 40)))); y = np.sin(x) + 0.2 * _rng.normal(size=len(x))
    r = int(_rng.integers(2, 20)); lr = float(_rng.choice([0.1, 0.5, 1.0])); T = _rng.uniform(0, 6, 10)
    f0, trees = gb_fit(x, y, r, lr)
    rf0, rtrees = _gb_ref(x, y, r, lr)
    pred = _need(gb_predict(f0, trees, lr, T))
    ref = rf0 + lr * sum(t.predict(T[:, None]) for t in rtrees)
    return np.isclose(f0, rf0) and len(trees) == r and np.shape(pred) == (10,) and np.allclose(pred, ref)

TESTS = {i: globals()[f"_t{i}"] for i in range(1, 7)}
''',
}

CHECK_HELPER = r'''
def check(n):
    """Run the hidden tests for exercise n and say whether they pass."""
    try:
        ok = bool(TESTS[n]())
        print(f"Exercise {n}: " + ("PASS" if ok else "not right yet. A test case gives the wrong answer (re-read the description and try the edge cases)"))
    except NameError as e:
        print(f"Exercise {n}: run the cell with your code first ({e})")
    except TypeError as e:
        if "NoneType" in str(e):
            print(f"Exercise {n}: something returned None. Not written yet, or a missing `return`?")
        elif "takes no arguments" in str(e):
            print(f"Exercise {n}: your class has no __init__ yet, or its __init__ doesn't take these inputs ({e})")
        else:
            print(f"Exercise {n}: your code raised TypeError: {e}")
    except AttributeError as e:
        if "'NoneType'" in str(e):
            print(f"Exercise {n}: something returned None. Not written yet, or a missing `return`?")
        else:
            print(f"Exercise {n}: a method or attribute is missing: {e}. Check the spelling, `self.`, and super().__init__()")
    except Exception as e:
        print(f"Exercise {n}: your code raised {type(e).__name__}: {e}")
'''


def load(g, notebook):
    exec("import numpy as np", g)
    exec(SUITES[notebook], g)
    exec(CHECK_HELPER, g)
