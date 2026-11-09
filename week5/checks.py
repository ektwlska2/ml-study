"""Exercise checkers for Week 5. Each notebook's setup cell runs `checks.load(globals(), "02")`, which defines `check(n)`.

The tests build random data every run and compare your function with a reference. Reading this file spoils the exercises."""

SUITES = {
"00": r'''
_rng = np.random.default_rng()

def _need(x):
    if x is None:
        raise TypeError("'NoneType' returned")
    return x

def _link(A, B, how):
    D = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))
    return {"single": D.min(), "complete": D.max(), "average": D.mean()}[how]

def _same_grouping(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and np.array_equal(a[:, None] == a[None, :], b[:, None] == b[None, :])

def _t1():
    n, d = int(_rng.integers(8, 20)), int(_rng.integers(2, 5)); k = int(_rng.integers(1, d + 1))
    X = _rng.normal(size=(n, d)) * _rng.uniform(0.5, 3, size=d)
    mean, comps = _need(pca_fit(X, k))
    S = np.cov(X - X.mean(0), rowvar=False); vals, vecs = np.linalg.eigh(S); ref = vecs[:, ::-1][:, :k].T
    comps = np.asarray(comps)
    return np.allclose(mean, X.mean(0)) and comps.shape == (k, d) and np.allclose(np.abs((comps * ref).sum(1)), 1, atol=1e-6)

def _t2():
    n = int(_rng.integers(4, 12)); w = _rng.uniform(0.1, 1, n); w /= w.sum()
    y = _rng.choice([-1, 1], n); h = _rng.choice([-1, 1], n); a = float(_rng.uniform(0.1, 1.5)); w0 = w.copy()
    out = _need(reweight(w, y, h, a)); ref = w0 * np.exp(-a * y * h); ref /= ref.sum()
    return np.allclose(out, ref) and np.array_equal(w, w0)

def _rand_clusters(n):
    perm = list(_rng.permutation(n)); cuts = sorted(_rng.choice(np.arange(1, n), size=int(_rng.integers(2, min(5, n - 1) + 1)) - 1, replace=False))
    parts, last = [], 0
    for c in list(cuts) + [n]:
        parts.append([int(i) for i in perm[last:c]]); last = c
    return parts

def _t3():
    ok = True
    for how in ["single", "complete", "average"]:
        n = int(_rng.integers(6, 12)); X = _rng.normal(size=(n, 2)); cl = _rand_clusters(n)
        best = None
        for a in range(len(cl)):
            for b in range(a + 1, len(cl)):
                dd = _link(X[cl[a]], X[cl[b]], how)
                if best is None or dd < best[2]:
                    best = (a, b, dd)
        out = _need(closest_pair(cl, X, how))
        ok = ok and tuple(out[:2]) == best[:2] and np.isclose(out[2], best[2])
    return bool(ok)

def _t4():
    from sklearn.cluster import AgglomerativeClustering
    ok = _same_grouping(_need(agglomerative(np.array([[1.], [2.], [4.], [8.], [13.]]), 2, "single")), [0, 0, 0, 0, 1])
    for how in ["single", "complete", "average"]:
        n = int(_rng.integers(6, 14)); X = _rng.normal(size=(n, 2)); k = int(_rng.integers(2, 4))
        ref = AgglomerativeClustering(n_clusters=k, linkage=how).fit(X).labels_
        ok = ok and _same_grouping(agglomerative(X, k, how), ref)
    return bool(ok)

TESTS = {i: globals()[f"_t{i}"] for i in range(1, 5)}
''',
"01": r'''
_rng = np.random.default_rng()

def _need(x):
    if x is None:
        raise TypeError("'NoneType' returned")
    return x

def _g(x, mu, var):
    return np.exp(-(x - mu) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)

def _t1():
    Z = _rng.normal(size=(int(_rng.integers(2, 6)), int(_rng.integers(2, 6)))) * 5
    out = _need(softmax_rows(Z)); E = np.exp(Z - Z.max(1, keepdims=True))
    big = _need(softmax_rows(np.array([[1000.0, 1001.0]])))
    return out.shape == Z.shape and np.allclose(out, E / E.sum(1, keepdims=True)) and np.allclose(big, [[0.26894142, 0.73105858]])

def _t2():
    n, K = int(_rng.integers(4, 12)), int(_rng.integers(2, 5))
    x = _rng.normal(size=n) * 3; mu = _rng.normal(size=K) * 3; var = _rng.uniform(0.5, 4, K); pi = _rng.dirichlet(np.ones(K))
    out = _need(e_step(x, mu, var, pi)); m = pi[None] * _g(x[:, None], mu[None], var[None])
    return out.shape == (n, K) and np.allclose(out, m / m.sum(1, keepdims=True))

def _t3():
    n, K = int(_rng.integers(3, 10)), int(_rng.integers(1, 5)); x = _rng.normal(size=n); W = _rng.uniform(0.1, 1, (n, K))
    out = _need(weighted_means(x, W))
    return np.shape(out) == (K,) and np.allclose(out, (W * x[:, None]).sum(0) / W.sum(0))

def _t4():
    n, K = int(_rng.integers(4, 12)), int(_rng.integers(2, 5)); x = _rng.normal(size=n) * 3; R = _rng.dirichlet(np.ones(K), size=n)
    out = _need(m_step(x, R)); Nk = R.sum(0); mu = (R * x[:, None]).sum(0) / Nk
    var = (R * (x[:, None] - mu[None]) ** 2).sum(0) / Nk
    return len(out) == 3 and all(np.shape(o) == (K,) for o in out) and np.allclose(out[0], mu) and np.allclose(out[1], var) and np.allclose(out[2], Nk / n)

def _t5():
    X = _rng.normal(size=(int(_rng.integers(3, 9)), int(_rng.integers(1, 5)))) * _rng.uniform(1, 9) + _rng.normal() * 4
    out = _need(standardize(X))
    return out.shape == X.shape and np.allclose(out, (X - X.mean(0)) / X.std(0))

def _t6():
    d = int(_rng.integers(1, 5)); A = _rng.normal(size=(int(_rng.integers(1, 7)), d)); B = _rng.normal(size=(int(_rng.integers(1, 7)), d))
    out = _need(pairwise_sq_dists(A, B))
    return out.shape == (len(A), len(B)) and np.allclose(out, ((A[:, None] - B[None]) ** 2).sum(2))

def _t7():
    n, d = int(_rng.integers(1, 8)), int(_rng.integers(1, 6)); A = _rng.normal(size=(n, d)); B = _rng.normal(size=(n, d))
    out = _need(row_dots(A, B))
    return np.shape(out) == (n,) and np.allclose(out, (A * B).sum(1))

def _t8():
    d = int(_rng.integers(2, 5)); A = _rng.normal(size=(int(_rng.integers(1, 6)), d)); B = _rng.normal(size=(int(_rng.integers(1, 6)), d))
    out = _need(cosine_similarity(A, B))
    ref = (A @ B.T) / np.linalg.norm(A, axis=1)[:, None] / np.linalg.norm(B, axis=1)[None]
    return out.shape == ref.shape and np.allclose(out, ref)

def _t9():
    n, d, h = (int(v) for v in _rng.integers(1, 6, 3)); X = _rng.normal(size=(n, d)); W = _rng.normal(size=(d, h)); b = _rng.normal(size=h)
    out = _need(dense_forward(X, W, b))
    return out.shape == (n, h) and np.allclose(out, X @ W + b)

def _t10():
    n, d, h = (int(v) for v in _rng.integers(1, 6, 3)); X = _rng.normal(size=(n, d)); W = _rng.normal(size=(d, h)); G = _rng.normal(size=(n, h))
    out = _need(dense_backward(X, W, G))
    dX, dW, db = out
    return (np.shape(dX) == (n, d) and np.shape(dW) == (d, h) and np.shape(db) == (h,)
            and np.allclose(dX, G @ W.T) and np.allclose(dW, X.T @ G) and np.allclose(db, G.sum(0)))

def _t11():
    C = int(_rng.integers(2, 7)); y = _rng.integers(0, C, size=int(_rng.integers(1, 9)))
    out = _need(one_hot(y, C))
    return np.shape(out) == (len(y), C) and np.array_equal(out, np.eye(C)[y])

def _t12():
    n, C = int(_rng.integers(1, 9)), int(_rng.integers(2, 6)); P = _rng.dirichlet(np.ones(C), size=n); y = _rng.integers(0, C, size=n)
    out = _need(cross_entropy(P, y))
    return np.isclose(out, -np.log(P[np.arange(n), y]).mean())

TESTS = {i: globals()[f"_t{i}"] for i in range(1, 13)}
''',
"02": r'''
_rng = np.random.default_rng()

def _need(x):
    if x is None:
        raise TypeError("'NoneType' returned")
    return x

def _sm(Z):
    E = np.exp(Z - Z.max(1, keepdims=True)); return E / E.sum(1, keepdims=True)

def _fwd(X, Ws, bs):
    H = X
    for l, (W, b) in enumerate(zip(Ws, bs)):
        Z = H @ W + b
        H = np.maximum(Z, 0) if l < len(Ws) - 1 else Z
    return H

def _t1():
    n, d = int(_rng.integers(3, 10)), int(_rng.integers(1, 5)); X = _rng.normal(size=(n, d)); y = _rng.integers(0, 2, n).astype(float)
    w = _rng.normal(size=d); b = float(_rng.normal())
    out = _need(logistic_grad(X, y, w, b)); p = 1 / (1 + np.exp(-(X @ w + b)))
    return np.allclose(out[0], X.T @ (p - y) / n) and np.shape(out[0]) == (d,) and np.isclose(out[1], (p - y).mean())

def _t2():
    d, h, m = (int(v) for v in _rng.integers(1, 5, 3))
    W1, b1, W2, b2 = _rng.normal(size=(d, h)), _rng.normal(size=h), _rng.normal(size=(h, m)), _rng.normal(size=m)
    W, b = _need(collapse_linear(W1, b1, W2, b2)); X = _rng.normal(size=(5, d))
    return np.shape(W) == (d, m) and np.shape(b) == (m,) and np.allclose(X @ W + b, (X @ W1 + b1) @ W2 + b2)

def _t3():
    sizes = [int(v) for v in _rng.integers(1, 6, int(_rng.integers(2, 5)))]
    Ws = [_rng.normal(size=(sizes[l], sizes[l + 1])) for l in range(len(sizes) - 1)]; bs = [_rng.normal(size=sizes[l + 1]) for l in range(len(sizes) - 1)]
    X = _rng.normal(size=(int(_rng.integers(1, 6)), sizes[0]))
    out = _need(mlp_forward(X, Ws, bs))
    return np.shape(out) == (len(X), sizes[-1]) and np.allclose(out, _fwd(X, Ws, bs))

def _t4():
    n, C = int(_rng.integers(1, 7)), int(_rng.integers(2, 6)); Z = _rng.normal(size=(n, C)) * 3; y = _rng.integers(0, C, n)
    out = _need(softmax_ce_grad(Z, y))
    return np.shape(out) == (n, C) and np.allclose(out, (_sm(Z) - np.eye(C)[y]) / n)

def _t5():
    sizes = [int(v) for v in _rng.integers(1, 300, int(_rng.integers(2, 6)))]
    ok = _need(count_params([784, 128, 64, 10])) == 109386 and count_params([2, 2, 1]) == 9
    return bool(ok) and count_params(sizes) == sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))

def _t6():
    sizes = [int(_rng.integers(1, 5))] + [int(v) for v in _rng.integers(1, 6, int(_rng.integers(0, 3)))] + [int(_rng.integers(2, 5))]
    m = _need(MyMLP(sizes, seed=int(_rng.integers(0, 1000))))
    L = len(sizes) - 1
    ok = len(m.W) == L and len(m.b) == L
    ok = ok and all(np.shape(m.W[l]) == (sizes[l], sizes[l + 1]) and np.shape(m.b[l]) == (sizes[l + 1],) and np.allclose(m.b[l], 0) for l in range(L))
    ok = ok and all(np.abs(np.asarray(w)).sum() > 0 for w in m.W)
    m.W = [_rng.normal(size=(sizes[l], sizes[l + 1])) for l in range(L)]; m.b = [_rng.normal(size=sizes[l + 1]) for l in range(L)]
    X = _rng.normal(size=(6, sizes[0])); ref = _fwd(X, m.W, m.b)
    return bool(ok) and np.allclose(_need(m.forward(X)), ref) and np.allclose(m.predict_proba(X), _sm(ref)) and np.array_equal(m.predict(X), ref.argmax(1))

TESTS = {i: globals()[f"_t{i}"] for i in range(1, 7)}
''',
"03": r'''
_rng = np.random.default_rng()

def _need(x):
    if x is None:
        raise TypeError("'NoneType' returned")
    return x

def _sg(z):
    return 1 / (1 + np.exp(-z))

def _sm(Z):
    E = np.exp(Z - Z.max(1, keepdims=True)); return E / E.sum(1, keepdims=True)

def _ref_grads(X, y, Ws, bs):
    Hs, Zs = [X], []
    for l, (W, b) in enumerate(zip(Ws, bs)):
        Z = Hs[-1] @ W + b; Zs.append(Z); Hs.append(np.maximum(Z, 0) if l < len(Ws) - 1 else Z)
    P = _sm(Hs[-1]); G = (P - np.eye(P.shape[1])[y]) / len(y)
    dWs, dbs = [None] * len(Ws), [None] * len(Ws)
    for l in range(len(Ws) - 1, -1, -1):
        dWs[l] = Hs[l].T @ G; dbs[l] = G.sum(0)
        if l > 0:
            G = (G @ Ws[l].T) * (Zs[l - 1] > 0)
    return dWs, dbs, -np.log(P[np.arange(len(y)), y]).mean()

def _rand_net():
    sizes = [int(_rng.integers(1, 5))] + [int(v) for v in _rng.integers(1, 6, int(_rng.integers(0, 3)))] + [int(_rng.integers(2, 5))]
    Ws = [_rng.normal(size=(a, b)) for a, b in zip(sizes[:-1], sizes[1:])]; bs = [_rng.normal(size=b) * 0.1 for b in sizes[1:]]
    return sizes, Ws, bs

def _t1():
    n, d, h = (int(v) for v in _rng.integers(1, 6, 3)); X = _rng.normal(size=(n, d)); W = _rng.normal(size=(d, h)); G = _rng.normal(size=(n, h))
    dX, dW, db = _need(dense_backward(X, W, G))
    return np.allclose(dX, G @ W.T) and np.allclose(dW, X.T @ G) and np.allclose(db, G.sum(0)) and np.shape(db) == (h,)

def _t2():
    ok = True
    for y in (0.0, 1.0):
        d, h = int(_rng.integers(1, 4)), int(_rng.integers(1, 4)); x = _rng.normal(size=d)
        W1, b1, W2, b2 = _rng.normal(size=(d, h)), _rng.normal(size=h), _rng.normal(size=h), float(_rng.normal())
        out = _need(backprop_221(x, y, W1, b1, W2, b2))
        hh = _sg(x @ W1 + b1); p = _sg(hh @ W2 + b2); d2 = p - y; dz1 = d2 * W2 * hh * (1 - hh)
        ref = (np.outer(x, dz1), dz1, d2 * hh, d2)
        ok = ok and len(out) == 4 and all(np.shape(a) == np.shape(r) and np.allclose(a, r) for a, r in zip(out, ref))
    return bool(ok)

def _t3():
    Z = _rng.normal(size=(int(_rng.integers(1, 5)), int(_rng.integers(1, 5)))) * 3; dH = _rng.normal(size=Z.shape)
    out = _need(sigmoid_backward(dH, Z)); s = _sg(Z)
    return np.shape(out) == Z.shape and np.allclose(out, dH * s * (1 - s))

def _t4():
    Z = _rng.normal(size=(int(_rng.integers(1, 5)), int(_rng.integers(1, 5)))); dH = _rng.normal(size=Z.shape)
    out = _need(relu_backward(dH, Z))
    return np.shape(out) == Z.shape and np.allclose(out, dH * (Z > 0))

def _t5():
    A = _rng.normal(size=(int(_rng.integers(1, 4)), int(_rng.integers(1, 4)))); keep = A.copy()
    f = lambda T: float((np.sin(T) * A).sum() + (T ** 2).sum())
    theta = _rng.normal(size=A.shape); theta0 = theta.copy()
    g = _need(numerical_grad(f, theta))
    v = _rng.normal(size=int(_rng.integers(1, 6))); gv = numerical_grad(lambda t: float((t ** 3).sum()), v)
    return (np.shape(g) == theta.shape and np.allclose(g, np.cos(theta0) * keep + 2 * theta0, atol=1e-5) and np.array_equal(theta, theta0)
            and np.allclose(gv, 3 * v ** 2, atol=1e-4))

def _t6():
    sizes, Ws, bs = _rand_net(); n = int(_rng.integers(1, 7)); X = _rng.normal(size=(n, sizes[0])); y = _rng.integers(0, sizes[-1], n)
    out = _need(mlp_gradients(X, y, Ws, bs)); rW, rb, _ = _ref_grads(X, y, Ws, bs)
    return (len(out) == 2 and len(out[0]) == len(Ws) and all(np.allclose(a, r) for a, r in zip(out[0], rW))
            and all(np.allclose(a, r) for a, r in zip(out[1], rb)))

def _t7():
    sizes, Ws, bs = _rand_net(); n = int(_rng.integers(2, 8)); X = _rng.normal(size=(n, sizes[0])); y = _rng.integers(0, sizes[-1], n)
    m = _need(MLPScratch(sizes, seed=1))
    ok = [np.shape(w) for w in m.W] == [np.shape(w) for w in Ws]
    m.W = [w.copy() for w in Ws]; m.b = [b.copy() for b in bs]
    epochs, lr = int(_rng.integers(1, 5)), float(_rng.uniform(0.05, 0.5))
    res = _need(m.fit(X, y, epochs=epochs, lr=lr))
    W, b, losses = [w.copy() for w in Ws], [v.copy() for v in bs], []
    for _ in range(epochs):
        gW, gb, L = _ref_grads(X, y, W, b); losses.append(L)
        W = [w - lr * g for w, g in zip(W, gW)]; b = [v - lr * g for v, g in zip(b, gb)]
    ok = ok and res is m and all(np.allclose(a, r) for a, r in zip(m.W, W)) and all(np.allclose(a, r) for a, r in zip(m.b, b))
    ok = ok and np.allclose(m.loss_history_, losses)
    P = m.predict_proba(X)
    return bool(ok) and np.allclose(P.sum(1), 1) and np.array_equal(m.predict(X), P.argmax(1))

TESTS = {i: globals()[f"_t{i}"] for i in range(1, 8)}
''',
"04": r'''
_rng = np.random.default_rng()

def _need(x):
    if x is None:
        raise TypeError("'NoneType' returned")
    return x

def _ref_backward(out):
    topo, seen = [], set()
    def build(v):
        if id(v) not in seen:
            seen.add(id(v))
            for c in v._prev:
                build(c)
            topo.append(v)
    build(out)
    for v in topo:
        v.grad = 0.0
    out.grad = 1.0
    for v in reversed(topo):
        v._backward()

def _t1():
    n, K = int(_rng.integers(4, 12)), int(_rng.integers(2, 5)); x = _rng.normal(size=n) * 3; R = _rng.dirichlet(np.ones(K), size=n)
    out = _need(m_step(x, R)); Nk = R.sum(0); mu = (R * x[:, None]).sum(0) / Nk
    var = (R * (x[:, None] - mu[None]) ** 2).sum(0) / Nk
    return np.allclose(out[0], mu) and np.allclose(out[1], var) and np.allclose(out[2], Nk / n)

def _t2():
    p, q, r = (float(v) for v in _rng.normal(size=3) * 3)
    a, b, c = Value(p), Value(q), Value(r)
    e = _need(a * b)
    d = e + c
    ok = isinstance(e, Value) and np.isclose(e.data, p * q) and np.isclose(d.data, p * q + r) and set(map(id, e._prev)) == {id(a), id(b)}
    _ref_backward(d)
    ok = ok and np.isclose(a.grad, q) and np.isclose(b.grad, p) and np.isclose(c.grad, 1.0)
    f = _need(a * 2.0); _ref_backward(f)
    return bool(ok) and np.isclose(f.data, 2 * p) and np.isclose(a.grad, 2.0)

def _t3():
    p, q = (float(v) for v in _rng.normal(size=2) * 2)
    a, b = Value(p), Value(q)
    f = a * b + a * a + b
    f.backward()
    ok = np.isclose(f.grad, 1.0) and np.isclose(a.grad, q + 2 * p) and np.isclose(b.grad, p + 1)
    x = Value(float(_rng.normal())); s = x
    for _ in range(30):
        s = s + x
    s.backward()
    return bool(ok) and np.isclose(x.grad, 31.0)

def _t4():
    ok = True
    for _ in range(3):
        u = float(_rng.normal()) * 2
        a = Value(u); o = _need(a.tanh()); _ref_backward(o)
        ok = ok and np.isclose(o.data, np.tanh(u)) and np.isclose(a.grad, 1 - np.tanh(u) ** 2)
        a = Value(u); o = _need(a.exp()); _ref_backward(o)
        ok = ok and np.isclose(o.data, np.exp(u)) and np.isclose(a.grad, np.exp(u))
        a = Value(u); o = _need(a.relu()); _ref_backward(o)
        ok = ok and np.isclose(o.data, max(0.0, u)) and np.isclose(a.grad, 1.0 if u > 0 else 0.0)
        pos = abs(u) + 0.5
        for k in (2, 3, -1, 0.5):
            a = Value(pos); o = _need(a ** k); _ref_backward(o)
            ok = ok and np.isclose(o.data, pos ** k) and np.isclose(a.grad, k * pos ** (k - 1))
    return bool(ok)

def _t5():
    p = float(_rng.normal()) * 2; q = float(_rng.uniform(0.5, 3)) * float(_rng.choice([-1, 1]))
    ok = True
    for make, val, ga, gb in [(lambda a, b: a - b, p - q, 1.0, -1.0), (lambda a, b: a / b, p / q, 1 / q, -p / q ** 2)]:
        a, b = Value(p), Value(q); o = _need(make(a, b)); _ref_backward(o)
        ok = ok and np.isclose(o.data, val) and np.isclose(a.grad, ga) and np.isclose(b.grad, gb)
    for make, val, g in [(lambda a: -a, -p, -1.0), (lambda a: 2 + a, 2 + p, 1.0), (lambda a: 3 * a, 3 * p, 3.0), (lambda a: a - 1.5, p - 1.5, 1.0),
                         (lambda a: a / 4, p / 4, 0.25)]:
        a = Value(p); o = _need(make(a)); _ref_backward(o)
        ok = ok and isinstance(o, Value) and np.isclose(o.data, val) and np.isclose(a.grad, g)
    return bool(ok)

def _t6():
    random.seed(int(_rng.integers(0, 10000)))
    nin = int(_rng.integers(1, 4)); nouts = [int(_rng.integers(2, 5)), int(_rng.integers(2, 4)), 1]
    m = _need(MLP(nin, nouts)); sizes = [nin] + nouts
    params = _need(m.parameters())
    ok = len(params) == sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:])) and all(isinstance(p, Value) for p in params)
    x = [float(v) for v in _rng.normal(size=nin)]
    out = m(x)
    ok = ok and isinstance(out, Value)
    h = np.array(x)
    for li, layer in enumerate(m.layers):
        W = np.array([[wi.data for wi in n.w] for n in layer.neurons]); b = np.array([n.b.data for n in layer.neurons])
        z = W @ h + b
        h = np.tanh(z) if li < len(m.layers) - 1 else z
    ok = ok and np.isclose(out.data, h[0])
    _ref_backward(out)
    return bool(ok) and any(abs(p.grad) > 0 for p in m.layers[0].parameters())

TESTS = {i: globals()[f"_t{i}"] for i in range(1, 7)}
''',
"05": r'''
_rng = np.random.default_rng()

def _need(x):
    if x is None:
        raise TypeError("'NoneType' returned")
    return x

def _t1():
    A = _rng.normal(size=(int(_rng.integers(1, 4)), int(_rng.integers(1, 4)))); keep = A.copy()
    f = lambda T: float((np.sin(T) * A).sum() + (T ** 2).sum())
    theta = _rng.normal(size=A.shape); theta0 = theta.copy()
    g = _need(numerical_grad(f, theta))
    return np.shape(g) == theta.shape and np.allclose(g, np.cos(theta0) * keep + 2 * theta0, atol=1e-5) and np.array_equal(theta, theta0)

def _t2():
    d = int(_rng.integers(1, 4)); A = np.diag(_rng.uniform(0.5, 3, d)); bvec = _rng.normal(size=d)
    grad = lambda w: A @ w - bvec
    w0 = _rng.normal(size=d); keep = w0.copy(); lr = float(_rng.uniform(0.05, 0.3)); steps = int(_rng.integers(1, 8))
    out = np.asarray(_need(gd_steps(grad, w0, lr, steps)))
    ref = [keep.copy()]
    for _ in range(steps):
        ref.append(ref[-1] - lr * grad(ref[-1]))
    s = gd_steps(lambda w: 2 * w, 1.0, 0.1, 3)
    return out.shape == (steps + 1, d) and np.allclose(out, ref) and np.array_equal(w0, keep) and np.allclose(s, [1.0, 0.8, 0.64, 0.512])

def _t3():
    n, b = int(_rng.integers(1, 30)), int(_rng.integers(1, 9)); seed = int(_rng.integers(0, 10000))
    out = _need(minibatches(n, b, np.random.default_rng(seed)))
    perm = np.random.default_rng(seed).permutation(n); ref = [perm[i:i + b] for i in range(0, n, b)]
    return len(out) == len(ref) and all(np.array_equal(o, r) for o, r in zip(out, ref))

def _t4():
    d = int(_rng.integers(1, 5)); w, v, g = (_rng.normal(size=d) for _ in range(3)); lr, beta = float(_rng.uniform(0.01, 0.5)), float(_rng.uniform(0.5, 0.99))
    w0, v0 = w.copy(), v.copy()
    out = _need(momentum_step(w, v, g, lr, beta)); vn = beta * v0 + g
    return len(out) == 2 and np.allclose(out[1], vn) and np.allclose(out[0], w0 - lr * vn) and np.array_equal(w, w0) and np.array_equal(v, v0)

def _t5():
    d = int(_rng.integers(1, 5)); w, g = _rng.normal(size=d), _rng.normal(size=d); s = _rng.uniform(0, 2, d)
    lr, rho = float(_rng.uniform(0.001, 0.1)), float(_rng.uniform(0.8, 0.999)); w0, s0 = w.copy(), s.copy()
    out = _need(rmsprop_step(w, s, g, lr, rho)); sn = rho * s0 + (1 - rho) * g ** 2
    return np.allclose(out[1], sn) and np.allclose(out[0], w0 - lr * g / (np.sqrt(sn) + 1e-8)) and np.array_equal(w, w0)

def _t6():
    d = int(_rng.integers(1, 5)); w, g, m = (_rng.normal(size=d) for _ in range(3)); s = _rng.uniform(0, 2, d); t = int(_rng.integers(1, 20))
    lr = float(_rng.uniform(0.001, 0.1)); w0 = w.copy()
    out = _need(adam_step(w, m, s, g, t, lr))
    mn = 0.9 * m + 0.1 * g; sn = 0.999 * s + 0.001 * g ** 2
    wn = w0 - lr * (mn / (1 - 0.9 ** t)) / (np.sqrt(sn / (1 - 0.999 ** t)) + 1e-8)
    one = adam_step(np.array([1.0]), np.zeros(1), np.zeros(1), np.array([2.0]), 1, 0.1)
    return np.allclose(out[0], wn) and np.allclose(out[1], mn) and np.allclose(out[2], sn) and np.array_equal(w, w0) and np.allclose(one[0], [0.9])

def _t7():
    shapes = [tuple(int(v) for v in _rng.integers(1, 4, int(_rng.integers(1, 3)))) for _ in range(int(_rng.integers(1, 4)))]
    params = [_rng.normal(size=sh) for sh in shapes]; ids = [id(p) for p in params]; ref = [p.copy() for p in params]
    lr = float(_rng.uniform(0.001, 0.2)); b1, b2 = 0.9, 0.99
    opt = _need(MyAdam(params, lr=lr, betas=(b1, b2)))
    m = [np.zeros(sh) for sh in shapes]; s = [np.zeros(sh) for sh in shapes]
    for t in range(1, int(_rng.integers(2, 6)) + 1):
        grads = [_rng.normal(size=sh) for sh in shapes]
        opt.step(grads)
        for i, g in enumerate(grads):
            m[i] = b1 * m[i] + (1 - b1) * g; s[i] = b2 * s[i] + (1 - b2) * g ** 2
            ref[i] = ref[i] - lr * (m[i] / (1 - b1 ** t)) / (np.sqrt(s[i] / (1 - b2 ** t)) + 1e-8)
    return all(id(p) == i for p, i in zip(params, ids)) and all(np.allclose(p, r) for p, r in zip(params, ref)) and all(np.allclose(p, r) for p, r in zip(opt.params, ref))

TESTS = {i: globals()[f"_t{i}"] for i in range(1, 8)}
''',
"06": r'''
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_rng = np.random.default_rng()

def _need(x):
    if x is None:
        raise TypeError("'NoneType' returned")
    return x

def _g(x, mu, var):
    return np.exp(-(x - mu) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)

def _t1():
    n, K = int(_rng.integers(4, 12)), int(_rng.integers(2, 5))
    x = _rng.normal(size=n) * 3; mu = _rng.normal(size=K) * 3; var = _rng.uniform(0.5, 4, K); pi = _rng.dirichlet(np.ones(K))
    out = _need(e_step(x, mu, var, pi)); m = pi[None] * _g(x[:, None], mu[None], var[None])
    return np.shape(out) == (n, K) and np.allclose(out, m / m.sum(1, keepdims=True))

def _t2():
    n, d = int(_rng.integers(1, 8)), int(_rng.integers(1, 5)); X = _rng.normal(size=(n, d)); y = _rng.integers(0, 5, n)
    Xt, yt = _need(to_tensors(X, y))
    return (isinstance(Xt, torch.Tensor) and Xt.dtype == torch.float32 and yt.dtype == torch.int64 and tuple(Xt.shape) == (n, d)
            and np.allclose(Xt.numpy(), X, atol=1e-6) and np.array_equal(yt.numpy(), y))

def _t3():
    d, h = int(_rng.integers(1, 6)), int(_rng.integers(1, 6)); W = _rng.normal(size=(d, h)); b = _rng.normal(size=h)
    lin = _need(linear_from_numpy(W, b)); X = _rng.normal(size=(5, d))
    out = lin(torch.tensor(X, dtype=torch.float32)).detach().numpy()
    return isinstance(lin, nn.Linear) and tuple(lin.weight.shape) == (h, d) and lin.weight.dtype == torch.float32 and np.allclose(out, X @ W + b, atol=1e-5)

def _t4():
    sizes = [int(v) for v in _rng.integers(1, 7, int(_rng.integers(2, 5)))]
    m = _need(make_mlp(sizes)); lins = [c for c in m.modules() if isinstance(c, nn.Linear)]
    ok = isinstance(m, nn.Sequential) and [tuple(l.weight.shape) for l in lins] == [(b, a) for a, b in zip(sizes[:-1], sizes[1:])]
    ok = ok and sum(isinstance(c, nn.ReLU) for c in m.modules()) == len(sizes) - 2 and isinstance(list(m.children())[-1], nn.Linear)
    X = torch.randn(4, sizes[0]); H = X
    for i, l in enumerate(lins):
        H = l(H)
        H = torch.relu(H) if i < len(lins) - 1 else H
    return bool(ok) and torch.allclose(m(X), H)

def _t5():
    torch.manual_seed(int(_rng.integers(0, 1000)))
    model = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 2)); ref = copy.deepcopy(model)
    X = torch.randn(10, 3); y = torch.randint(0, 2, (10,))
    loader = DataLoader(TensorDataset(X, y), batch_size=4, shuffle=False); loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.1); opt_r = torch.optim.SGD(ref.parameters(), lr=0.1)
    model.eval()
    out = _need(train_epoch(model, loader, loss_fn, opt))
    total = 0.0
    for xb, yb in loader:
        loss = loss_fn(ref(xb), yb); opt_r.zero_grad(); loss.backward(); opt_r.step(); total += loss.item() * len(xb)
    same = all(torch.allclose(a, b) for a, b in zip(model.parameters(), ref.parameters()))
    return same and np.isclose(float(out), total / 10) and model.training

def _t6():
    model = nn.Sequential(nn.Linear(3, 4)); X = torch.randn(13, 3); y = torch.randint(0, 4, (13,))
    loader = DataLoader(TensorDataset(X, y), batch_size=5); model.train()
    acc = _need(evaluate(model, loader))
    with torch.no_grad():
        ref = (model(X).argmax(1) == y).float().mean().item()
    return np.isclose(float(acc), ref) and not model.training

def _t7():
    d, h, c = (int(v) for v in _rng.integers(1, 9, 3))
    m = _need(DigitsMLP(d, h, c)); params = list(m.parameters())
    ok = isinstance(m, nn.Module) and [tuple(p.shape) for p in params] == [(h, d), (h,), (c, h), (c,)]
    X = torch.randn(5, d)
    ref = torch.relu(X @ params[0].T + params[1]) @ params[2].T + params[3]
    return bool(ok) and torch.allclose(m(X), ref, atol=1e-6) and sum(p.numel() for p in DigitsMLP().parameters()) == 2410

TESTS = {i: globals()[f"_t{i}"] for i in range(1, 8)}
''',
"07": r'''
import copy
import torch
import torch.nn as nn

_rng = np.random.default_rng()

def _need(x):
    if x is None:
        raise TypeError("'NoneType' returned")
    return x

def _sm(Z):
    E = np.exp(Z - Z.max(1, keepdims=True)); return E / E.sum(1, keepdims=True)

def _t1():
    k, d = int(_rng.integers(2, 5)), int(_rng.integers(1, 4)); centers = _rng.normal(size=(k, d)) * 20
    X = np.concatenate([c + _rng.normal(size=(int(_rng.integers(2, 6)), d)) for c in centers]); starts = np.cumsum([0] + [0] * 0)
    C0 = np.array([X[np.argmin(((X - c) ** 2).sum(1))] for c in centers]); iters = int(_rng.integers(1, 4))
    C = C0.copy()
    for _ in range(iters):
        lab = ((X[:, None] - C[None]) ** 2).sum(2).argmin(1); C = np.array([X[lab == j].mean(0) for j in range(k)])
    keep = C0.copy()
    out = _need(kmeans_fit(X, C0, iters))
    return np.shape(out) == (k, d) and np.allclose(out, C) and np.array_equal(C0, keep)

def _t2():
    n, C = int(_rng.integers(1, 7)), int(_rng.integers(2, 6)); Z = _rng.normal(size=(n, C)) * 3; y = _rng.integers(0, C, n)
    loss, dZ = _need(softmax_cross_entropy(Z, y)); P = _sm(Z)
    return np.isclose(loss, -np.log(P[np.arange(n), y]).mean()) and np.allclose(dZ, (P - np.eye(C)[y]) / n)

def _t3():
    n, d, h, C = int(_rng.integers(1, 6)), int(_rng.integers(1, 4)), int(_rng.integers(1, 5)), int(_rng.integers(2, 4))
    X = _rng.normal(size=(n, d)); y = _rng.integers(0, C, n); W1, b1, W2, b2 = _rng.normal(size=(d, h)), _rng.normal(size=h), _rng.normal(size=(h, C)), _rng.normal(size=C)
    keep = [a.copy() for a in (W1, b1, W2, b2)]; lr = float(_rng.uniform(0.05, 0.5))
    out = _need(mlp_train_step(X, y, W1, b1, W2, b2, lr))
    Z1 = X @ W1 + b1; H = np.maximum(Z1, 0); P = _sm(H @ W2 + b2); G = (P - np.eye(C)[y]) / n
    dW2, db2 = H.T @ G, G.sum(0); G1 = (G @ W2.T) * (Z1 > 0); dW1, db1 = X.T @ G1, G1.sum(0)
    ref = (W1 - lr * dW1, b1 - lr * db1, W2 - lr * dW2, b2 - lr * db2)
    return len(out) == 4 and all(np.allclose(a, r) for a, r in zip(out, ref)) and all(np.array_equal(a, k) for a, k in zip((W1, b1, W2, b2), keep))

def _t4():
    d = int(_rng.integers(1, 5)); A = _rng.normal(size=(int(_rng.integers(1, 7)), d)); B = _rng.normal(size=(int(_rng.integers(1, 7)), d))
    out = _need(pairwise_sq_dists(A, B))
    return np.shape(out) == (len(A), len(B)) and np.allclose(out, ((A[:, None] - B[None]) ** 2).sum(2))

def _t5():
    torch.manual_seed(int(_rng.integers(0, 1000)))
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 3)); ref = copy.deepcopy(model)
    X = torch.randn(8, 3); y = torch.randint(0, 3, (8,)); epochs, lr = int(_rng.integers(1, 5)), float(_rng.uniform(0.05, 0.5))
    losses = _need(torch_fit(model, X, y, epochs, lr))
    opt = torch.optim.SGD(ref.parameters(), lr=lr); lf = nn.CrossEntropyLoss(); rl = []
    for _ in range(epochs):
        loss = lf(ref(X), y); opt.zero_grad(); loss.backward(); opt.step(); rl.append(loss.item())
    return len(losses) == epochs and np.allclose(losses, rl) and all(torch.allclose(a, b) for a, b in zip(model.parameters(), ref.parameters()))

TESTS = {i: globals()[f"_t{i}"] for i in range(1, 6)}
''',
"08": r'''
import torch
import torch.nn as nn

_rng = np.random.default_rng()

def _need(x):
    if x is None:
        raise TypeError("'NoneType' returned")
    return x

def _t1():
    sizes = [int(v) for v in _rng.integers(1, 6, int(_rng.integers(2, 5)))] + [int(_rng.integers(2, 5))]
    Ws = [_rng.normal(size=(a, b)) for a, b in zip(sizes[:-1], sizes[1:])]; bs = [_rng.normal(size=b) for b in sizes[1:]]
    X = _rng.normal(size=(7, sizes[0])); H = X
    for l in range(len(Ws)):
        H = H @ Ws[l] + bs[l]
        H = np.maximum(H, 0) if l < len(Ws) - 1 else H
    out = _need(mlp_predict(X, Ws, bs))
    return np.shape(out) == (7,) and np.array_equal(out, H.argmax(1))

def _t2():
    d = int(_rng.integers(1, 4)); A = np.diag(_rng.uniform(0.2, 2, d)); grad = lambda w: A @ w
    w0 = _rng.normal(size=d); keep = w0.copy(); lr, beta, steps = float(_rng.uniform(0.01, 0.3)), float(_rng.uniform(0.5, 0.95)), int(_rng.integers(1, 10))
    w, v = keep.copy(), np.zeros(d)
    for _ in range(steps):
        v = beta * v + grad(w); w = w - lr * v
    out = _need(momentum_run(grad, w0, lr, beta, steps))
    return np.allclose(out, w) and np.array_equal(w0, keep) and np.allclose(momentum_run(lambda x: 2 * x, np.array([1.0]), 0.1, 0.9, 3), [0.062])

def _t3():
    d = int(_rng.integers(1, 5)); w, g, m = (_rng.normal(size=d) for _ in range(3)); s = _rng.uniform(0, 2, d); t = int(_rng.integers(1, 20)); lr = float(_rng.uniform(0.001, 0.1))
    out = _need(adam_step(w, m, s, g, t, lr)); mn = 0.9 * m + 0.1 * g; sn = 0.999 * s + 0.001 * g ** 2
    wn = w - lr * (mn / (1 - 0.9 ** t)) / (np.sqrt(sn / (1 - 0.999 ** t)) + 1e-8)
    return np.allclose(out[0], wn) and np.allclose(out[1], mn) and np.allclose(out[2], sn)

def _t4():
    torch.manual_seed(int(_rng.integers(0, 1000)))
    model = nn.Sequential(nn.Linear(4, 6), nn.Dropout(0.5), nn.ReLU(), nn.Linear(6, 3)); model.train()
    X = _rng.normal(size=(9, 4))
    out = _need(torch_predict(model, X))
    with torch.no_grad():
        model.eval(); ref = model(torch.tensor(X, dtype=torch.float32)).argmax(1).numpy()
    return isinstance(out, np.ndarray) and np.array_equal(out, ref) and not model.training

def _t5():
    a, b, c = (int(v) for v in _rng.integers(1, 9, 3))
    model = nn.Sequential(nn.Linear(a, b), nn.ReLU(), nn.Linear(b, c))
    ok = _need(count_trainable(model)) == a * b + b + b * c + c
    for p in model[0].parameters():
        p.requires_grad = False                     # "freeze" the first layer
    return bool(ok) and count_trainable(model) == b * c + c

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
