"""Exercise checkers for Week 4. Each notebook's setup cell runs `checks.load(globals(), "02")`, which defines `check(n)`.

The tests build random data every run and compare your function with a reference. Reading this file spoils the exercises."""

SUITES = {
"00": r'''
_rng = np.random.default_rng()

def _need(x):
    if x is None:
        raise TypeError("'NoneType' returned")
    return x

def _rand_tree(depth):
    if depth == 0 or _rng.random() < 0.25:
        return {"leaf": int(_rng.integers(0, 2))}
    return {"feature": int(_rng.integers(0, 3)), "threshold": float(_rng.normal()),
            "left": _rand_tree(depth - 1), "right": _rand_tree(depth - 1)}

def _leaves(t):
    return 1 if "leaf" in t else _leaves(t["left"]) + _leaves(t["right"])

def _depth(t):
    return 0 if "leaf" in t else 1 + max(_depth(t["left"]), _depth(t["right"]))

def _t1():
    ok = _need(count_leaves({"leaf": 1})) == 1
    for _ in range(15):
        t = _rand_tree(int(_rng.integers(1, 6))); ok = ok and count_leaves(t) == _leaves(t)
    return bool(ok)

def _t2():
    ok = _need(tree_depth({"leaf": 0})) == 0
    for _ in range(15):
        t = _rand_tree(int(_rng.integers(1, 6))); ok = ok and tree_depth(t) == _depth(t)
    return bool(ok)

def _t3():
    Z = _rng.uniform(0.1, 5, size=(int(_rng.integers(2, 8)), int(_rng.integers(2, 6))))
    out = _need(normalize_rows(Z))
    return out.shape == Z.shape and np.allclose(out.sum(axis=1), 1) and np.allclose(out, Z / Z.sum(axis=1, keepdims=True))

def _t4():
    names = np.array(["cat", "dog", "owl", "bee"])
    ok = _need(majority_label(np.array(["dog", "cat", "dog"]))) == "dog"
    for _ in range(15):
        lab = names[_rng.integers(0, 4, size=int(_rng.integers(1, 9)))]
        u, inv = np.unique(lab, return_inverse=True)
        ok = ok and majority_label(lab) == u[np.bincount(inv).argmax()]
    return bool(ok)

TESTS = {1: _t3, 2: _t4, 3: _t1, 4: _t2}
''',
"01": r'''
_rng = np.random.default_rng()
X8 = np.arange(1, 9.0); Y8 = np.array([1, 1, -1, -1, 1, 1, -1, -1.0])

def _need(x):
    if x is None:
        raise TypeError("'NoneType' returned")
    return x

def _sp(x, t, s):
    return np.where(x > t, s, -s)

def _bs(x, y, w):
    v = np.unique(x); best = (None, 1, 2.0)
    for t in (v[:-1] + v[1:]) / 2:
        for s in (1, -1):
            e = w[_sp(x, t, s) != y].sum()
            if e < best[2] - 1e-12:
                best = (t, s, e)
    return best

def _ada(x, y, rounds):
    w = np.full(len(x), 1 / len(x)); model = []
    for _ in range(rounds):
        t, s, e = _bs(x, y, w); e = min(max(e, 1e-10), 1 - 1e-10); a = 0.5 * np.log((1 - e) / e)
        h = _sp(x, t, s); w = w * np.exp(-a * y * h); w = w / w.sum(); model.append((t, s, a))
    return model

def _rs(x, r):
    v = np.unique(x); best = None
    for t in (v[:-1] + v[1:]) / 2:
        L, R = r[x <= t], r[x > t]; sse = ((L - L.mean()) ** 2).sum() + ((R - R.mean()) ** 2).sum()
        if best is None or sse < best[0] - 1e-12:
            best = (sse, t, L.mean(), R.mean())
    return best[1:]

def _gb(x, y, rounds, lr):
    f0 = y.mean(); F = np.full(len(y), f0); st = []
    for _ in range(rounds):
        t, a, b = _rs(x, y - F); F = F + lr * np.where(x <= t, a, b); st.append((t, a, b))
    return f0, st

def _data1d(n=None):
    n = n or int(_rng.integers(6, 15))
    x = np.sort(_rng.choice(np.arange(1, 40), n, replace=False)).astype(float)
    y = _rng.choice([-1.0, 1.0], n); y[0], y[-1] = 1.0, -1.0
    return x, y

def _t1():
    out = _need(stump_predict(X8, 2.5, -1))
    ok = np.array_equal(out, [1, 1, -1, -1, -1, -1, -1, -1])
    x, _ = _data1d(); t = float(_rng.choice(x)) + 0.5; s = int(_rng.choice([1, -1]))
    return bool(ok) and np.array_equal(stump_predict(x, t, s), _sp(x, t, s))

def _t2():
    ok = np.isclose(_need(weighted_error(Y8, _sp(X8, 2.5, -1), np.full(8, 1 / 8))), 0.25)
    x, y = _data1d(); w = _rng.random(len(x)); w /= w.sum(); h = _rng.choice([-1.0, 1.0], len(x))
    return bool(ok) and np.isclose(weighted_error(y, h, w), w[h != y].sum())

def _t3():
    t, s, e = _need(best_stump(X8, Y8, np.full(8, 1 / 8)))
    ok = np.isclose(t, 2.5) and s == -1 and np.isclose(e, 0.25)
    for _ in range(10):
        x, y = _data1d(); w = _rng.random(len(x)); w /= w.sum()
        out = best_stump(x, y, w); ref = _bs(x, y, w)
        ok = ok and np.isclose(out[0], ref[0]) and out[1] == ref[1] and np.isclose(out[2], ref[2])
    return bool(ok)

def _t4():
    ok = np.isclose(_need(stump_alpha(0.25)), 0.5 * np.log(3)) and np.isclose(stump_alpha(0.5), 0.0)
    e = float(_rng.uniform(0.05, 0.95))
    return bool(ok) and np.isclose(stump_alpha(e), 0.5 * np.log((1 - e) / e))

def _t5():
    h = _sp(X8, 2.5, -1); a = 0.5 * np.log(3)
    out = _need(update_weights(np.full(8, 1 / 8), Y8, h, a))
    ok = np.allclose(out, [1 / 12] * 4 + [0.25, 0.25] + [1 / 12] * 2)
    x, y = _data1d(); w = _rng.random(len(x)); w /= w.sum(); hh = _rng.choice([-1.0, 1.0], len(x)); aa = float(_rng.uniform(0.1, 1))
    ref = w * np.exp(-aa * y * hh); ref /= ref.sum()
    return bool(ok) and np.allclose(update_weights(w, y, hh, aa), ref) and np.isclose(update_weights(w, y, hh, aa).sum(), 1)

def _t6():
    out = _need(adaboost_fit(X8, Y8, 3))
    ok = len(out) == 3 and np.allclose(np.array(out, float), np.array(_ada(X8, Y8, 3), float), atol=1e-6)
    for _ in range(5):
        x, y = _data1d(); r = int(_rng.integers(1, 6))
        ok = ok and np.allclose(np.array(adaboost_fit(x, y, r), float), np.array(_ada(x, y, r), float))
    return bool(ok)

def _t7():
    model = _ada(X8, Y8, 3)
    ok = np.array_equal(_need(adaboost_predict(model, X8)), Y8)
    x = _rng.uniform(0, 9, 20); F = sum(a * _sp(x, t, s) for t, s, a in model)
    return bool(ok) and np.array_equal(adaboost_predict(model, x), np.where(F >= 0, 1, -1))

XG = np.array([1, 2, 3, 4.0]); YG = np.array([1, 3, 6, 10.0])

def _t8():
    ok = np.allclose(_need(residuals(YG, np.full(4, 5.0))), [-4, -2, 1, 5])
    y = _rng.normal(size=9); F = _rng.normal(size=9)
    return bool(ok) and np.allclose(residuals(y, F), y - F)

def _t9():
    t, a, b = _need(fit_reg_stump(XG, np.array([-4, -2, 1, 5.0])))
    ok = np.isclose(t, 2.5) and np.isclose(a, -3) and np.isclose(b, 3)
    for _ in range(10):
        x = np.sort(_rng.choice(np.arange(1, 30), 8, replace=False)).astype(float); r = _rng.normal(size=8)
        out = fit_reg_stump(x, r); ref = _rs(x, r)
        ok = ok and np.allclose(out, ref)
    return bool(ok)

def _t10():
    x = _rng.uniform(0, 10, 12); t = float(_rng.uniform(2, 8)); a, b = _rng.normal(size=2)
    out = _need(reg_stump_predict(x, t, a, b))
    return np.allclose(out, np.where(x <= t, a, b)) and np.allclose(reg_stump_predict(XG, 2.5, -3.0, 3.0), [-3, -3, 3, 3])

def _t11():
    f0, st = _need(gb_fit(XG, YG, 2, 0.5))
    ok = np.isclose(f0, 5.0) and np.allclose(np.array(st, float), [[2.5, -3, 3], [3.5, -7 / 6, 3.5]])
    x = np.sort(_rng.choice(np.arange(1, 30), 10, replace=False)).astype(float); y = np.sin(x / 4) + 0.1 * _rng.normal(size=10)
    r = int(_rng.integers(1, 8)); lr = float(_rng.choice([0.1, 0.5, 1.0]))
    a0, a1 = gb_fit(x, y, r, lr); b0, b1 = _gb(x, y, r, lr)
    return bool(ok) and np.isclose(a0, b0) and np.allclose(np.array(a1, float), np.array(b1, float))

def _t12():
    f0, st = _gb(XG, YG, 2, 0.5)
    ok = np.allclose(_need(gb_predict(f0, st, 0.5, XG)), [2.9167, 2.9167, 5.9167, 8.25], atol=1e-3)
    x = np.sort(_rng.choice(np.arange(1, 30), 10, replace=False)).astype(float); y = _rng.normal(size=10)
    f0, st = _gb(x, y, 4, 0.3); T = _rng.uniform(0, 31, 7)
    ref = f0 + 0.3 * sum(np.where(T <= t, a, b) for t, a, b in st)
    return bool(ok) and np.allclose(gb_predict(f0, st, 0.3, T), ref)

TESTS = {i: globals()[f"_t{i}"] for i in range(1, 13)}
''',
"02": r'''
_rng = np.random.default_rng()
XP = np.array([[3, 3], [4, 2], [5, 4], [6, 6], [7, 5]], float)

def _need(x):
    if x is None:
        raise TypeError("'NoneType' returned")
    return x

def _X():
    n, d = int(_rng.integers(8, 30)), int(_rng.integers(2, 6))
    A = _rng.normal(size=(d, d)); return _rng.normal(size=(n, d)) @ A + _rng.normal(3, 2, size=d)

def _same_dirs(A, B):
    """Columns equal up to sign."""
    return A.shape == B.shape and all(np.isclose(abs(A[:, j] @ B[:, j]), 1, atol=1e-6) for j in range(A.shape[1]))

def _t1():
    ok = np.allclose(_need(center(XP)), XP - [5, 4])
    X = _X(); out = center(X)
    return bool(ok) and out.shape == X.shape and np.allclose(out.mean(axis=0), 0) and np.allclose(out, X - X.mean(axis=0))

def _t2():
    ok = np.allclose(_need(covariance(XP - [5, 4])), [[2.5, 2], [2, 2.5]])
    X = _X(); Xc = X - X.mean(0)
    return bool(ok) and np.allclose(covariance(Xc), np.cov(X.T))

def _t3():
    vals, vecs = _need(pca_eig(np.array([[2.5, 2], [2, 2.5]])))
    ok = np.allclose(vals, [4.5, 0.5]) and _same_dirs(vecs, np.array([[1, -1], [1, 1]]) / np.sqrt(2))
    X = _X(); S = np.cov(X.T); v, V = pca_eig(S)
    w, W = np.linalg.eigh(S); o = np.argsort(w)[::-1]
    return bool(ok) and np.allclose(v, w[o]) and _same_dirs(V, W[:, o]) and np.all(np.diff(v) <= 1e-12)

def _t4():
    W = np.array([[1], [1]]) / np.sqrt(2)
    ok = np.allclose(_need(project(XP - [5, 4], W)).ravel(), [-2.1213, -2.1213, 0, 2.1213, 2.1213], atol=1e-4)
    X = _X(); Xc = X - X.mean(0); k = int(_rng.integers(1, X.shape[1] + 1)); W2 = np.linalg.qr(_rng.normal(size=(X.shape[1], k)))[0]
    return bool(ok) and np.allclose(project(Xc, W2), Xc @ W2) and project(Xc, W2).shape == (len(X), k)

def _t5():
    W = np.array([[1], [1]]) / np.sqrt(2); Z = (XP - [5, 4]) @ W
    ok = np.allclose(_need(reconstruct(Z, W, np.array([5.0, 4.0]))), [[3.5, 2.5], [3.5, 2.5], [5, 4], [6.5, 5.5], [6.5, 5.5]])
    X = _X(); m = X.mean(0); W2 = np.linalg.qr(_rng.normal(size=(X.shape[1], X.shape[1])))[0]
    return bool(ok) and np.allclose(reconstruct((X - m) @ W2, W2, m), X)

def _t6():
    ok = np.allclose(_need(explained_ratio(np.array([4.5, 0.5]))), [0.9, 0.1])
    v = np.sort(_rng.uniform(0, 5, 6))[::-1]
    return bool(ok) and np.allclose(explained_ratio(v), v / v.sum())

def _t7():
    ok = _need(n_components_for(np.array([5.0, 3.0, 2.0]), 0.8)) == 2 and n_components_for(np.array([5.0, 3.0, 2.0]), 0.81) == 3
    v = np.sort(_rng.uniform(0, 1, 8))[::-1]; t = float(_rng.uniform(0.5, 0.99))
    ref = int(np.argmax(np.cumsum(v / v.sum()) >= t - 1e-12)) + 1
    return bool(ok) and n_components_for(v, t) == ref and n_components_for(np.array([4.5, 0.5]), 0.9) == 1

def _t8():
    Z, C = _need(pca_svd(XP, 1))
    ok = Z.shape == (5, 1) and C.shape == (1, 2) and np.allclose(np.abs(Z.ravel()), [2.1213, 2.1213, 0, 2.1213, 2.1213], atol=1e-4)
    X = _X(); k = int(_rng.integers(1, X.shape[1] + 1)); Z, C = pca_svd(X, k)
    from sklearn.decomposition import PCA
    p = PCA(k).fit(X); R = p.transform(X)
    return bool(ok) and Z.shape == R.shape and _same_dirs(C.T, p.components_.T) and all(np.isclose(abs(Z[:, j] @ R[:, j]), R[:, j] @ R[:, j]) for j in range(k))

def _t9():
    m = _need(MyPCA(1).fit(XP))
    ok = np.allclose(m.mean_, [5, 4]) and np.shape(m.components_) == (1, 2) and np.isclose(abs(m.components_[0] @ np.array([1, 1]) / np.sqrt(2)), 1)
    Z = _need(m.transform(np.array([[7.0, 7.0]])))
    ok = ok and np.allclose(np.abs(Z), [[3.5355]], atol=1e-3) and np.allclose(m.inverse_transform(Z), [[7.5, 6.5]])
    from sklearn.decomposition import PCA
    X = _X(); k = int(_rng.integers(1, X.shape[1] + 1)); mine = MyPCA(k).fit(X); ref = PCA(k).fit(X)
    T = _rng.normal(size=(5, X.shape[1])); A, B = mine.transform(T), ref.transform(T)
    return (bool(ok) and np.allclose(mine.explained_variance_, ref.explained_variance_) and A.shape == B.shape
            and all(np.allclose(np.abs(A[:, j]), np.abs(B[:, j])) for j in range(k)) and np.allclose(mine.inverse_transform(A), ref.inverse_transform(B)))

TESTS = {i: globals()[f"_t{i}"] for i in range(1, 10)}
''',
"03": r'''
_rng = np.random.default_rng()
XK = np.array([[1, 1], [1, 2], [2, 1], [4, 3], [5, 4], [5, 3]], float)

def _need(x):
    if x is None:
        raise TypeError("'NoneType' returned")
    return x

def _blobs():
    k = int(_rng.integers(2, 5)); d = int(_rng.integers(2, 4)); n = int(_rng.integers(20, 50))
    centers = _rng.normal(0, 6, (k, d)); lab = np.r_[np.arange(k), _rng.integers(0, k, n - k)]
    return centers[lab] + _rng.normal(size=(n, d)), k

def _assign(X, C):
    return np.argmin(((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2), axis=1)

def _t1():
    ok = np.array_equal(_need(assign(XK, np.array([[1, 1], [2, 1.0]]))), [0, 0, 1, 1, 1, 1])
    X, k = _blobs(); C = X[_rng.choice(len(X), k, replace=False)] + 0.01
    return bool(ok) and np.array_equal(assign(X, C), _assign(X, C))

def _t2():
    ok = np.allclose(_need(update(XK, np.array([0, 0, 1, 1, 1, 1]), 2)), [[1, 1.5], [4, 2.75]])
    X, k = _blobs(); lab = np.r_[np.arange(k), _rng.integers(0, k, len(X) - k)]
    return bool(ok) and np.allclose(update(X, lab, k), np.array([X[lab == c].mean(0) for c in range(k)]))

def _t3():
    ok = np.isclose(_need(inertia(XK, np.array([[1, 1.5], [4, 2.75]]), np.array([0, 0, 1, 1, 1, 1]))), 11.25)
    X, k = _blobs(); C = _rng.normal(size=(k, X.shape[1])); lab = _assign(X, C)
    return bool(ok) and np.isclose(inertia(X, C, lab), ((X - C[lab]) ** 2).sum())

def _t4():
    C, lab = _need(kmeans(XK, np.array([[1, 1], [2, 1.0]]), 10))
    ok = np.allclose(C, [[4 / 3, 4 / 3], [14 / 3, 10 / 3]]) and np.array_equal(lab, [0, 0, 0, 1, 1, 1])
    X, k = _blobs(); C0 = X[_rng.choice(len(X), k, replace=False)].copy(); keep = C0.copy()
    Cr = C0.copy(); lab = _assign(X, Cr)
    for _ in range(300):
        Cr = np.array([X[lab == j].mean(0) if np.any(lab == j) else np.full(X.shape[1], np.nan) for j in range(k)])
        new = _assign(X, np.nan_to_num(Cr, nan=1e9))
        if np.array_equal(new, lab):
            break
        lab = new
    if np.isnan(Cr).any():
        return bool(ok)
    C2, lab2 = kmeans(X, C0, 300)
    return bool(ok) and np.allclose(C2, Cr) and np.array_equal(lab2, lab) and np.allclose(C0, keep)

def _t5():
    ok = np.allclose(_need(kmeanspp_probs(XK, np.array([[1, 1.0]]))), np.array([0, 1, 1, 13, 25, 20]) / 60)
    X, k = _blobs(); C = X[:2]
    D2 = ((X[:, None, :] - C[None]) ** 2).sum(2).min(1)
    return bool(ok) and np.allclose(kmeanspp_probs(X, C), D2 / D2.sum())

def _t6():
    lab = np.array([0, 0, 0, 1, 1, 1])
    ok = np.isclose(_need(silhouette_point(XK, lab, 2)), 0.6609, atol=1e-3)
    from sklearn.metrics import silhouette_samples
    X, k = _blobs(); lab = _assign(X, X[_rng.choice(len(X), k, replace=False)] + 0.01)
    if len(np.unique(lab)) < 2:
        return bool(ok)
    ref = silhouette_samples(X, lab); i = int(_rng.integers(0, len(X)))
    if np.sum(lab == lab[i]) == 1:
        return bool(ok)
    return bool(ok) and np.isclose(silhouette_point(X, lab, i), ref[i])

def _ref_km(X, C, iters=100):
    C = C.copy(); lab = _assign(X, C)
    for _ in range(iters):
        C = np.array([X[lab == j].mean(0) for j in range(len(C))]); new = _assign(X, C)
        if np.array_equal(new, lab):
            break
        lab = new
    return C, lab

def _t7():
    km = _need(MyKMeans(np.array([[1, 1], [2, 1.0]])).fit(XK))
    ok = np.allclose(km.cluster_centers_, [[4 / 3, 4 / 3], [14 / 3, 10 / 3]]) and np.array_equal(km.labels_, [0, 0, 0, 1, 1, 1]) and np.isclose(km.inertia_, 8 / 3)
    ok = ok and np.array_equal(_need(km.predict(np.array([[3.5, 2.0], [0.0, 0.0]]))), [1, 0])
    for _ in range(5):
        X, k = _blobs(); C0 = X[_rng.choice(len(X), k, replace=False)].copy()
        with np.errstate(all="ignore"):
            Cr, lr_ = _ref_km(X, C0)
        if np.isnan(Cr).any() or any(np.sum(lr_ == j) == 0 for j in range(k)):
            continue
        mine = MyKMeans(C0.copy()).fit(X); T = _rng.normal(0, 6, (8, X.shape[1]))
        ok = ok and np.allclose(mine.cluster_centers_, Cr) and np.array_equal(mine.labels_, lr_) and np.array_equal(mine.predict(T), _assign(T, Cr))
        ok = ok and np.isclose(mine.inertia_, ((X - Cr[lr_]) ** 2).sum())
    return bool(ok)

TESTS = {i: globals()[f"_t{i}"] for i in range(1, 8)}
''',
"04": r'''
_rng = np.random.default_rng()

def _need(x):
    if x is None:
        raise TypeError("'NoneType' returned")
    return x

def _pd(A, B):
    return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))

def _t1():
    A = np.array([[1.0], [2.0]]); B = np.array([[4.0]])
    ok = np.isclose(_need(linkage_distance(A, B, "single")), 2) and np.isclose(linkage_distance(A, B, "complete"), 3) and np.isclose(linkage_distance(A, B, "average"), 2.5)
    d = int(_rng.integers(1, 4)); A = _rng.normal(size=(int(_rng.integers(1, 6)), d)); B = _rng.normal(size=(int(_rng.integers(1, 6)), d)); D = _pd(A, B)
    return bool(ok) and np.isclose(linkage_distance(A, B, "single"), D.min()) and np.isclose(linkage_distance(A, B, "complete"), D.max()) and np.isclose(linkage_distance(A, B, "average"), D.mean())

def _t2():
    X = np.array([1, 1.5, 2, 2.5, 5, 8, 8.5, 9, 12.0])[:, None]
    ok = np.array_equal(_need(dbscan_core(X, 0.6, 3)), [False, True, True, False, False, False, True, False, False])
    Y = _rng.normal(size=(int(_rng.integers(10, 40)), 2)); eps = float(_rng.uniform(0.2, 1.0)); m = int(_rng.integers(2, 6))
    ref = (_pd(Y, Y) <= eps).sum(axis=1) >= m
    return bool(ok) and np.array_equal(dbscan_core(Y, eps, m), ref)

def _npdf(x, m, v):
    return np.exp(-(x - m) ** 2 / (2 * v)) / np.sqrt(2 * np.pi * v)

def _t3():
    ok = np.isclose(_need(gaussian_pdf(1.0, 2.0, 4.0)), 0.17603, atol=1e-5)
    x = _rng.normal(size=7); m, v = float(_rng.normal()), float(_rng.uniform(0.3, 3))
    return bool(ok) and np.allclose(gaussian_pdf(x, m, v), _npdf(x, m, v))

XG1 = np.array([1, 2, 3, 6, 7, 8.0])

def _resp(x, mu, var, pi):
    N = np.stack([pi[k] * _npdf(x, mu[k], var[k]) for k in range(len(mu))], axis=1)
    return N / N.sum(axis=1, keepdims=True)

def _t4():
    R = _need(e_step(XG1, np.array([2.0, 5.0]), np.array([4.0, 4.0]), np.array([0.5, 0.5])))
    ok = R.shape == (6, 2) and np.allclose(R[:, 0], [0.867, 0.755, 0.593, 0.133, 0.068, 0.033], atol=1e-3)
    K = int(_rng.integers(2, 4)); x = _rng.normal(0, 3, 15); mu = _rng.normal(0, 3, K); var = _rng.uniform(0.5, 3, K); pi = _rng.dirichlet(np.ones(K))
    return bool(ok) and np.allclose(e_step(x, mu, var, pi), _resp(x, mu, var, pi))

def _t5():
    R = _resp(XG1, np.array([2.0, 5.0]), np.array([4.0, 4.0]), np.array([0.5, 0.5]))
    mu, var, pi = _need(m_step(XG1, R))
    ok = np.allclose(mu, [2.324, 6.0], atol=2e-3) and np.allclose(var, [2.536, 4.424], atol=2e-3) and np.allclose(pi, [0.408, 0.592], atol=2e-3)
    x = _rng.normal(0, 3, 12); R = _rng.dirichlet(np.ones(3), size=12); Nk = R.sum(0)
    m2 = (R * x[:, None]).sum(0) / Nk; v2 = (R * (x[:, None] - m2) ** 2).sum(0) / Nk
    a, b, c = m_step(x, R)
    return bool(ok) and np.allclose(a, m2) and np.allclose(b, v2) and np.allclose(c, Nk / 12)

def _t6():
    ok = np.isclose(_need(log_likelihood(XG1, np.array([2.0, 5.0]), np.array([4.0, 4.0]), np.array([0.5, 0.5]))), -14.638, atol=1e-3)
    x = _rng.normal(size=10); mu = _rng.normal(size=2); var = _rng.uniform(0.5, 2, 2); pi = np.array([0.3, 0.7])
    ref = np.log(sum(pi[k] * _npdf(x, mu[k], var[k]) for k in range(2))).sum()
    return bool(ok) and np.isclose(log_likelihood(x, mu, var, pi), ref)

def _t7():
    X = np.array([1, 1.5, 2, 2.5, 5, 8, 8.5, 9, 12.0])[:, None]; lab = np.array([0, 0, 0, 0, -1, 1, 1, 1, -1])
    core = np.array([False, True, True, False, False, False, True, False, False])
    outs = [_need(dbscan_predict(X, lab, core, np.array([v]), 0.6)) for v in (1.9, 2.8, 8.9, 6.5)]
    ok = [int(o) for o in outs] == [0, -1, 1, -1]
    from sklearn.cluster import DBSCAN
    Y = np.r_[_rng.normal(0, 0.3, (20, 2)), _rng.normal(4, 0.3, (20, 2))]; eps = 0.5; m = DBSCAN(eps=eps, min_samples=4).fit(Y)
    cm = np.zeros(len(Y), bool); cm[m.core_sample_indices_] = True
    for _ in range(10):
        q = _rng.uniform(-1, 5, 2); d = np.sqrt(((Y - q) ** 2).sum(1)); cand = cm & (d <= eps)
        ref = m.labels_[cand][np.argmin(d[cand])] if cand.any() else -1
        ok = ok and int(dbscan_predict(Y, m.labels_, cm, q, eps)) == ref
    return bool(ok)

def _t8():
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import adjusted_rand_score
    X = np.array([1, 1.5, 2, 2.5, 5, 8, 8.5, 9, 12.0])[:, None]
    ok = np.array_equal(_need(dbscan(X, 0.6, 3)), [0, 0, 0, 0, -1, 1, 1, 1, -1])
    for _ in range(5):
        k = int(_rng.integers(2, 4)); C = np.array([[0, 0], [10, 0], [0, 10]])[:k] + _rng.uniform(-1, 1, (k, 2))     # well apart: no shared border points
        Y = np.r_[np.concatenate([c + _rng.normal(0, 0.4, (int(_rng.integers(10, 25)), 2)) for c in C]), _rng.uniform(-3, 23, (5, 2))]
        out = np.asarray(dbscan(Y, 0.6, 4)); ref = DBSCAN(eps=0.6, min_samples=4).fit(Y).labels_
        ok = ok and out.shape == ref.shape and np.array_equal(out < 0, ref < 0) and adjusted_rand_score(ref, out) == 1.0
    return bool(ok)

def _t9():
    x = XG1; m0, v0, p0 = np.array([2.0, 5.0]), np.array([4.0, 4.0]), np.array([0.5, 0.5])
    mu, var, pi = _need(gmm_fit_1d(x, m0.copy(), v0.copy(), p0.copy(), 1))
    ok = np.allclose(mu, [2.324, 6.0], atol=2e-3)
    y = np.r_[_rng.normal(0, 1, 15), _rng.normal(5, 1.5, 15)]; it = int(_rng.integers(1, 30))
    a, b, c = np.array([-1.0, 3.0]), np.array([1.0, 1.0]), np.array([0.5, 0.5])
    for _ in range(it):
        R = _resp(y, a, b, c); Nk = R.sum(0); a = (R * y[:, None]).sum(0) / Nk; b = (R * (y[:, None] - a) ** 2).sum(0) / Nk; c = Nk / len(y)
    r1, r2, r3 = gmm_fit_1d(y, np.array([-1.0, 3.0]), np.array([1.0, 1.0]), np.array([0.5, 0.5]), it)
    return bool(ok) and np.allclose(r1, a) and np.allclose(r2, b) and np.allclose(r3, c)

TESTS = {i: globals()[f"_t{i}"] for i in range(1, 10)}
''',
"05": r'''
import pandas as pd
_rng = np.random.default_rng()

def _need(x):
    if x is None:
        raise TypeError("'NoneType' returned")
    return x

def _frame(n=None):
    n = n or int(_rng.integers(30, 80))
    d = pd.DataFrame({"shift": _rng.choice(["day", "night"], n), "temp": _rng.normal(73, 3, n), "vibration": _rng.lognormal(0, 0.5, n),
                      "line": _rng.choice(list("ABC"), n), "pressure": _rng.normal(100, 4, n)})
    d.loc[_rng.random(n) < 0.1, "temp"] = np.nan
    return d

def _t1():
    d = _frame(); before = d.copy(); out = _need(add_features(d))
    ok = d.equals(before) and {"night_hot", "vib_spike"} <= set(out.columns)
    ref_nh = ((d["shift"] == "night") & (d["temp"] > 74)).astype(float); ref_vs = (d["vibration"] > 2).astype(float)
    return bool(ok) and np.allclose(out["night_hot"], ref_nh) and np.allclose(out["vib_spike"], ref_vs) and out["night_hot"].dtype == float

def _t2():
    d = _frame(); num, cat = ["temp", "vibration", "pressure"], ["shift", "line"]
    pre = _need(make_preprocessor(num, cat))
    Z = pre.fit_transform(d)
    Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
    n_cat = d["shift"].nunique() + d["line"].nunique()
    ok = Z.shape == (len(d), 3 + n_cat) and not np.isnan(Z).any()
    ok = ok and np.allclose(Z[:, :3].mean(axis=0), 0, atol=1e-8) and np.allclose(Z[:, :3].std(axis=0), 1, atol=1e-8)
    new = d.head(3).copy(); new["line"] = "Z"                       # an unseen category must not crash
    return bool(ok) and pre.transform(new).shape[0] == 3

def _t3():
    y = np.array([1, 0, 1, 1, 0]); s = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    ok = np.isclose(_need(threshold_for_recall(y, s, 0.6)), 0.7) and np.isclose(threshold_for_recall(y, s, 1.0), 0.6)
    y = _rng.integers(0, 2, 60); y[:3] = 1; s = _rng.random(60); target = float(_rng.uniform(0.3, 0.95))
    t = threshold_for_recall(y, s, target)
    rec = lambda t: ((s >= t) & (y == 1)).sum() / (y == 1).sum()
    higher = s[s > t]
    best_ok = all(rec(c) < target for c in higher)
    return bool(ok) and rec(t) >= target and best_ok

def _t4():
    y = np.array([1, 0, 1, 1, 0]); s = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    p, r = _need(precision_recall_at(y, s, 0.7))
    ok = np.isclose(p, 2 / 3) and np.isclose(r, 2 / 3)
    y = _rng.integers(0, 2, 50); y[0] = 1; s = _rng.random(50); s[0] = 1.0; t = float(_rng.uniform(0.2, 0.8)); pred = s >= t
    tp = (pred & (y == 1)).sum()
    p2, r2 = precision_recall_at(y, s, t)
    return bool(ok) and np.isclose(p2, tp / pred.sum()) and np.isclose(r2, tp / (y == 1).sum())

def _t5():
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    d = _frame(80); yy = (_rng.random(80) < 0.4).astype(int); yy[:2] = [0, 1]
    numc, catc = ["temp", "vibration", "pressure"], ["shift", "line"]
    pre = ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numc), ("cat", OneHotEncoder(handle_unknown="ignore"), catc)])
    model = make_pipeline(pre, LogisticRegression(max_iter=2000)).fit(d, yy)
    ok = True
    for _ in range(6):
        part = _frame(1)
        if _rng.random() < 0.3:
            part.loc[0, "temp"] = np.nan
        if _rng.random() < 0.2:
            part.loc[0, "line"] = "Z"
        out = _need(score_by_hand(model, part))
        ok = ok and np.isclose(float(out), model.predict_proba(part)[0, 1])
    return bool(ok)

def _t6():
    A = np.array([[2.0, 1.0], [0.0, 2.0]]); y = np.array([1, 0])
    w, b = _need(logistic_gd(A, y, 0.5, 1))
    ok = np.allclose(w, [0.25, -0.125]) and np.isclose(b, 0)
    A = _rng.normal(size=(40, 3)); y = (A @ [1.0, -2.0, 0.5] > 0).astype(int); lr = float(_rng.choice([0.1, 0.5])); st = int(_rng.integers(10, 200))
    rw, rb = np.zeros(3), 0.0
    for _ in range(st):
        ph = 1 / (1 + np.exp(-(A @ rw + rb))); rw = rw - lr * A.T @ (ph - y) / 40; rb = rb - lr * np.mean(ph - y)
    w, b = logistic_gd(A, y, lr, st)
    return bool(ok) and np.allclose(w, rw) and np.isclose(b, rb)

TESTS = {1: _t2, 2: _t1, 3: _t3, 4: _t4, 5: _t5, 6: _t6}
''',
"07": r'''
_rng = np.random.default_rng()

def _need(x):
    if x is None:
        raise TypeError("'NoneType' returned")
    return x

def _t1():
    st = [(2.5, -3.0, 3.0), (3.5, -1.1667, 3.5)]
    ok = np.isclose(_need(gb_predict_one(5.0, st, 0.5, 3.2)), 5.91665) and np.isclose(gb_predict_one(5.0, st, 0.5, 0.5), 2.91665)
    st = [(float(_rng.uniform(0, 10)), float(_rng.normal()), float(_rng.normal())) for _ in range(5)]; f0 = float(_rng.normal()); lr = 0.3; x = float(_rng.uniform(0, 10))
    return bool(ok) and np.isclose(gb_predict_one(f0, st, lr, x), f0 + lr * sum(l if x <= t else r for t, l, r in st))

def _t2():
    model = [(2.5, -1, 0.549), (6.5, -1, 0.805), (4.5, 1, 0.693)]
    ok = _need(adaboost_predict_one(model, 5.5)) == 1 and adaboost_predict_one(model, 7.0) == -1
    m2 = [(float(_rng.uniform(0, 10)), int(_rng.choice([1, -1])), float(_rng.uniform(0.1, 1))) for _ in range(4)]
    for x in _rng.uniform(0, 10, 6):
        F = sum(a * (s if x > t else -s) for t, s, a in m2)
        ok = ok and adaboost_predict_one(m2, float(x)) == (1 if F >= 0 else -1)
    return bool(ok)

def _t3():
    C = np.array([[4 / 3, 4 / 3], [14 / 3, 10 / 3]])
    ok = np.array_equal(_need(kmeans_predict(np.array([[3.5, 2.0], [0.0, 0.0]]), C)), [1, 0])
    k, d = int(_rng.integers(2, 5)), int(_rng.integers(1, 4)); C = _rng.normal(size=(k, d)); X = _rng.normal(size=(10, d))
    return bool(ok) and np.array_equal(kmeans_predict(X, C), ((X[:, None] - C[None]) ** 2).sum(2).argmin(1))

def _t4():
    ok = np.array_equal(_need(logistic_predict(np.array([[2.0, 2.0]]), np.array([0.672, 0.663]), -2.61, 0.5)), [1])
    ok = ok and np.array_equal(logistic_predict(np.array([[2.0, 2.0]]), np.array([0.672, 0.663]), -2.61, 0.8), [0])
    X = _rng.normal(size=(12, 3)); w = _rng.normal(size=3); b = float(_rng.normal()); t = float(_rng.uniform(0.2, 0.8))
    return bool(ok) and np.array_equal(logistic_predict(X, w, b, t), (1 / (1 + np.exp(-(X @ w + b))) >= t).astype(int))

def _t5():
    out = _need(pca_scores(np.array([[7.0, 7.0]]), np.array([5.0, 4.0]), np.array([[0.7071, 0.7071]])))
    ok = np.allclose(out, [[3.5355]], atol=1e-3)
    d, k = 4, 2; m = _rng.normal(size=d); Cm = np.linalg.qr(_rng.normal(size=(d, k)))[0].T; X = _rng.normal(size=(6, d))
    return bool(ok) and np.allclose(pca_scores(X, m, Cm), (X - m) @ Cm.T)

TESTS = {i: globals()[f"_t{i}"] for i in range(1, 6)}
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
