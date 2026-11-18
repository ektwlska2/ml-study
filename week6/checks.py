"""Exercise checkers for Week 6. Each notebook's setup cell runs `checks.load(globals(), "02")`, which defines `check(n)`.

The tests build random data every run and compare your function with a reference. Reading this file spoils the exercises."""

SUITES = {
"00": r'''
_rng = np.random.default_rng()

def _need(x):
    if x is None:
        raise TypeError("'NoneType' returned")
    return x

def _t1():
    n, d = int(_rng.integers(20, 60)), int(_rng.integers(1, 5)); X = _rng.normal(size=(n, d)); y = (_rng.random(n) < 0.5).astype(float)
    lr, steps = float(_rng.uniform(0.1, 0.8)), int(_rng.integers(5, 60))
    w, b = np.zeros(d), 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        w = w - lr * X.T @ (p - y) / n
        b = b - lr * np.mean(p - y)
    out = _need(logistic_gd(X, y, lr, steps))
    return len(out) == 2 and np.allclose(out[0], w) and np.isclose(out[1], b)

TESTS = {1: _t1}
''',
"01": r'''
_rng = np.random.default_rng()

def _need(x):
    if x is None:
        raise TypeError("'NoneType' returned")
    return x

def _t1():
    n, d = int(_rng.integers(10, 40)), int(_rng.integers(1, 4)); X1 = np.column_stack([np.ones(n), _rng.normal(size=(n, d))]); y = _rng.normal(size=n)
    lr, steps = float(_rng.uniform(0.01, 0.2)), int(_rng.integers(1, 50))
    w = np.zeros(d + 1)
    for _ in range(steps):
        w = w - lr * 2 / n * X1.T @ (X1 @ w - y)
    out = _need(linreg_gd(X1, y, lr, steps))
    return np.shape(out) == (d + 1,) and np.allclose(out, w)

def _t2():
    from scipy.special import softmax
    sizes = [int(_rng.integers(2, 6))] + [int(v) for v in _rng.integers(2, 6, int(_rng.integers(1, 3)))] + [int(_rng.integers(2, 5))]
    Ws = [_rng.normal(size=(a, b)) for a, b in zip(sizes[:-1], sizes[1:])]; bs = [_rng.normal(size=b) * 0.1 for b in sizes[1:]]
    n = int(_rng.integers(2, 8)); X = _rng.normal(size=(n, sizes[0])); Y = np.eye(sizes[-1])[_rng.integers(0, sizes[-1], n)]
    Hs, Zs = [X], []
    for l, (W, b) in enumerate(zip(Ws, bs)):
        Zs.append(Hs[-1] @ W + b); Hs.append(np.maximum(Zs[-1], 0) if l < len(Ws) - 1 else Zs[-1])
    G = (softmax(Hs[-1], axis=1) - Y) / n; ref = []
    for l in reversed(range(len(Ws))):
        ref.insert(0, (Hs[l].T @ G, G.sum(axis=0)))
        if l > 0:
            G = (G @ Ws[l].T) * (Zs[l - 1] > 0)
    out = _need(backward(Hs, Zs, Y, Ws))
    return len(out) == len(ref) and all(np.allclose(a[0], r[0]) and np.allclose(a[1], r[1]) for a, r in zip(out, ref))

TESTS = {1: _t1, 2: _t2}
''',
"02": r'''
import torch
_rng = np.random.default_rng()

def _need(x):
    if x is None:
        raise TypeError("'NoneType' returned")
    return x

def _t1():
    n, d = int(_rng.integers(8, 30)), int(_rng.integers(1, 5)); X1 = np.column_stack([np.ones(n), _rng.normal(size=(n, d))]); y = _rng.normal(size=n)
    lam = float(_rng.uniform(0.1, 20)); P = np.eye(d + 1); P[0, 0] = 0
    out = _need(ridge_closed_form(X1, y, lam))
    return np.shape(out) == (d + 1,) and np.allclose(out, np.linalg.solve(X1.T @ X1 + lam * P, X1.T @ y))

def _t2():
    d = int(_rng.integers(1, 6)); w0 = _rng.normal(size=d); g = _rng.normal(size=d); lr, lam = float(_rng.uniform(0.01, 0.5)), float(_rng.uniform(0, 1))
    wt = torch.tensor(w0, requires_grad=True); opt = torch.optim.SGD([wt], lr=lr, weight_decay=lam)
    (wt * torch.tensor(g)).sum().backward(); opt.step()
    out = _need(sgd_weight_decay_step(w0.copy(), g, lr, lam))
    return np.allclose(out, wt.detach().numpy())

def _t3():
    shape = tuple(int(v) for v in _rng.integers(1, 6, int(_rng.integers(1, 3)))); h = _rng.normal(size=shape); p = float(_rng.uniform(0.1, 0.8)); seed = int(_rng.integers(0, 1000))
    out = _need(inverted_dropout(h, p, np.random.default_rng(seed)))
    mask = np.random.default_rng(seed).random(h.shape) >= p
    return len(out) == 2 and np.allclose(out[0], h * mask / (1 - p)) and np.array_equal(np.asarray(out[1], bool), mask)

def _t4():
    ok = tuple(_need(early_stop([0.9, 0.7, 0.6, 0.65, 0.62, 0.7, 0.8], 3))) == (5, 2)
    ok = ok and tuple(early_stop([0.9, 0.8, 0.7], 3)) == (2, 2)
    v = list(_rng.uniform(0, 1, int(_rng.integers(3, 20)))); pat = int(_rng.integers(1, 5))
    best, stop = 0, len(v) - 1
    for e in range(len(v)):
        if v[e] < v[best]:
            best = e
        elif e - best >= pat:
            stop = e; break
    return bool(ok) and tuple(early_stop(v, pat)) == (stop, best)

TESTS = {i: globals()[f"_t{i}"] for i in range(1, 5)}
''',
"03": r'''
import torch
import torch.nn as nn
_rng = np.random.default_rng()

def _need(x):
    if x is None:
        raise TypeError("'NoneType' returned")
    return x

def _t1():
    d = int(_rng.integers(1, 5)); lr = float(_rng.uniform(0.01, 0.3))
    w, st, wr, m, s = _rng.normal(size=d), {}, None, np.zeros(d), np.zeros(d)
    wr = w.copy()
    for t in range(1, int(_rng.integers(2, 6)) + 1):
        g = _rng.normal(size=d)
        w = _need(adam(w, g, st, lr))
        m = 0.9 * m + 0.1 * g; s = 0.999 * s + 0.001 * g ** 2
        wr = wr - lr * (m / (1 - 0.9 ** t)) / (np.sqrt(s / (1 - 0.999 ** t)) + 1e-8)
    return np.allclose(w, wr) and st.get("t") == t

def _t2():
    n, d = int(_rng.integers(2, 8)), int(_rng.integers(1, 5)); X = _rng.normal(size=(n, d)) * 5 + 3
    gamma, beta = _rng.normal(size=d), _rng.normal(size=d)
    out = _need(batchnorm_train(X, gamma, beta))
    bn = nn.BatchNorm1d(d).double()
    with torch.no_grad():
        bn.weight.copy_(torch.tensor(gamma)); bn.bias.copy_(torch.tensor(beta))
    return np.allclose(out, bn(torch.tensor(X)).detach().numpy(), atol=1e-6)

def _t3():
    n, d = int(_rng.integers(1, 6)), int(_rng.integers(2, 6)); X = _rng.normal(size=(n, d)) * 4 - 1
    gamma, beta = _rng.normal(size=d), _rng.normal(size=d)
    out = _need(layernorm(X, gamma, beta))
    ln = nn.LayerNorm(d).double()
    with torch.no_grad():
        ln.weight.copy_(torch.tensor(gamma)); ln.bias.copy_(torch.tensor(beta))
    return np.allclose(out, ln(torch.tensor(X)).detach().numpy(), atol=1e-6)

def _t4():
    grads = [_rng.normal(size=tuple(int(v) for v in _rng.integers(1, 4, 2))) * float(_rng.uniform(0.1, 5)) for _ in range(int(_rng.integers(1, 4)))]
    c = float(_rng.uniform(0.2, 3)); keep = [g.copy() for g in grads]
    out = _need(clip_by_global_norm(grads, c))
    params = [torch.zeros(g.shape, dtype=torch.float64, requires_grad=True) for g in grads]
    for p_, g in zip(params, keep):
        p_.grad = torch.tensor(g)
    torch.nn.utils.clip_grad_norm_(params, max_norm=c)
    return len(out) == len(grads) and all(np.allclose(o, p_.grad.numpy(), atol=1e-6) for o, p_ in zip(out, params))

def _t5():
    d = int(_rng.integers(1, 7)); blk = _need(ResidualBlock(d)); ps = list(blk.parameters())
    ok = isinstance(blk, nn.Module) and [tuple(p_.shape) for p_ in ps] == [(d, d), (d,), (d, d), (d,)]
    x = torch.randn(5, d)
    ref = x + (torch.relu(x @ ps[0].T + ps[1]) @ ps[2].T + ps[3])
    return bool(ok) and torch.allclose(blk(x), ref, atol=1e-6)

TESTS = {i: globals()[f"_t{i}"] for i in range(1, 6)}
''',
"04": r'''
import torch
import torch.nn as nn
import torch.nn.functional as F
_rng = np.random.default_rng()

def _need(x):
    if x is None:
        raise TypeError("'NoneType' returned")
    return x

def _t1():
    from scipy.stats import multivariate_normal
    n, d, K = int(_rng.integers(4, 12)), int(_rng.integers(1, 3)), int(_rng.integers(2, 4))
    X = _rng.normal(size=(n, d)) * 2; pi = _rng.dirichlet(np.ones(K)); mu = _rng.normal(size=(K, d)) * 2
    Sigma = np.array([np.eye(d) * float(_rng.uniform(0.5, 2)) for _ in range(K)])
    dens = np.column_stack([pi[k] * multivariate_normal(mu[k], Sigma[k]).pdf(X) for k in range(K)])
    out = _need(gmm_e_step(X, pi, mu, Sigma))
    return np.shape(out) == (n, K) and np.allclose(out, dens / dens.sum(axis=1, keepdims=True))

def _t2():
    ok = _need(conv_out_size(28, 3)) == 26 and conv_out_size(28, 3, 1) == 28 and conv_out_size(28, 3, 1, 2) == 14 and conv_out_size(7, 3, 0, 2) == 3
    for _ in range(10):
        n, k = int(_rng.integers(5, 40)), int(_rng.integers(1, 6)); p, s = int(_rng.integers(0, 3)), int(_rng.integers(1, 4))
        if n + 2 * p < k:
            continue
        ok = ok and conv_out_size(n, k, p, s) == F.conv2d(torch.zeros(1, 1, n, n), torch.zeros(1, 1, k, k), padding=p, stride=s).shape[-1]
    return bool(ok)

def _t3():
    h, w = int(_rng.integers(3, 9)), int(_rng.integers(3, 9)); a, b = int(_rng.integers(1, min(h, 4) + 1)), int(_rng.integers(1, min(w, 4) + 1))
    x = _rng.normal(size=(h, w)); K = _rng.normal(size=(a, b))
    out = _need(conv2d_single(x, K))
    ref = F.conv2d(torch.tensor(x)[None, None], torch.tensor(K)[None, None])[0, 0].numpy()
    return np.shape(out) == ref.shape and np.allclose(out, ref)

def _t4():
    ok = _need(conv_params(3, 16, 3)) == 448
    ci, co, k = (int(v) for v in _rng.integers(1, 20, 3))
    return bool(ok) and conv_params(ci, co, k) == sum(p.numel() for p in nn.Conv2d(ci, co, k).parameters())

def _t5():
    s = int(_rng.integers(2, 4)); h, w = s * int(_rng.integers(1, 5)), s * int(_rng.integers(1, 5)); x = _rng.normal(size=(h, w))
    out = _need(maxpool2d(x, s))
    ref = F.max_pool2d(torch.tensor(x)[None, None], s)[0, 0].numpy()
    return np.shape(out) == ref.shape and np.allclose(out, ref)

TESTS = {i: globals()[f"_t{i}"] for i in range(1, 6)}
''',
"05": r'''
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
_rng = np.random.default_rng()

def _need(x):
    if x is None:
        raise TypeError("'NoneType' returned")
    return x

def _t1():
    torch.manual_seed(int(_rng.integers(0, 1000)))
    model = nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 2)); ref = copy.deepcopy(model)
    X = torch.randn(10, 3); y = torch.randint(0, 2, (10,)); loader = DataLoader(TensorDataset(X, y), batch_size=4)
    loss_fn = nn.CrossEntropyLoss(); opt = torch.optim.SGD(model.parameters(), lr=0.1); opt_r = torch.optim.SGD(ref.parameters(), lr=0.1)
    model.eval()
    out = _need(train_epoch(model, loader, loss_fn, opt))
    tot = 0.0
    for xb, yb in loader:
        loss = loss_fn(ref(xb), yb); opt_r.zero_grad(); loss.backward(); opt_r.step(); tot += loss.item() * len(xb)
    return all(torch.allclose(a, b) for a, b in zip(model.parameters(), ref.parameters())) and np.isclose(float(out), tot / 10) and model.training

def _t2():
    m = _need(SmallCNN())
    shapes = [tuple(p.shape) for p in m.parameters()]
    out = m(torch.zeros(3, 1, 28, 28))
    return isinstance(m, nn.Module) and shapes == [(8, 1, 3, 3), (8,), (16, 8, 3, 3), (16,), (10, 784), (10,)] and tuple(out.shape) == (3, 10)

def _t3():
    c = int(_rng.integers(1, 4)); n = int(_rng.integers(1, 4)); h = 2 * int(_rng.integers(3, 8))
    model = nn.Sequential(nn.Conv2d(c, 4, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(4 * (h // 2) ** 2, 3))
    x = torch.randn(n, c, h, h); ref, y = [], x
    for layer in model:
        y = layer(y); ref.append(tuple(y.shape))
    out = _need(shapes_after_each_layer(model, x))
    return [tuple(s) for s in out] == ref

def _t4():
    ok = tuple(_need(cnn_output_shape((1, 28, 28), [("conv", 8, 3, 1, 1), ("pool", 2), ("conv", 16, 3, 1, 1), ("pool", 2)]))) == (16, 7, 7)
    for _ in range(5):
        shape = (int(_rng.integers(1, 4)), int(_rng.integers(16, 40)), int(_rng.integers(16, 40))); layers, mods = [], []
        c = shape[0]
        for _ in range(int(_rng.integers(1, 4))):
            if _rng.random() < 0.6:
                co, k, p, s = int(_rng.integers(1, 9)), int(_rng.choice([1, 3, 5])), int(_rng.integers(0, 3)), int(_rng.integers(1, 3))
                layers.append(("conv", co, k, p, s)); mods.append(nn.Conv2d(c, co, k, padding=p, stride=s)); c = co
            else:
                layers.append(("pool", 2)); mods.append(nn.MaxPool2d(2))
        try:
            y = nn.Sequential(*mods)(torch.zeros(1, *shape))
        except RuntimeError:                       # the random layers shrank the image below the kernel: skip
            continue
        ok = ok and tuple(cnn_output_shape(shape, layers)) == tuple(y.shape[1:])
    return bool(ok)

TESTS = {i: globals()[f"_t{i}"] for i in range(1, 5)}
''',
"06": r'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.tree import DecisionTreeRegressor
_rng = np.random.default_rng()

def _need(x):
    if x is None:
        raise TypeError("'NoneType' returned")
    return x

def _t1():
    n, d = int(_rng.integers(40, 80)), int(_rng.integers(1, 4)); X = _rng.normal(size=(n, d)); y = np.sin(X[:, 0]) * 3 + _rng.normal(size=n) * 0.3
    nt, lr = int(_rng.integers(3, 15)), float(_rng.uniform(0.05, 0.5))
    F_ = np.full(n, y.mean()); trees = []
    for _ in range(nt):
        t = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X, y - F_); F_ = F_ + lr * t.predict(X); trees.append(t)
    Xn = _rng.normal(size=(10, d)); ref = y.mean() + lr * sum(t.predict(Xn) for t in trees)
    F0, ts = _need(gboost(X, y, nt, lr))
    return np.isclose(F0, y.mean()) and len(ts) == nt and np.allclose(gb_predict(F0, ts, Xn, lr), ref, atol=1e-6)

def _gini(y):
    p = np.bincount(y) / len(y); return 1 - np.sum(p ** 2)

def _t2():
    n, d = int(_rng.integers(10, 40)), int(_rng.integers(1, 4)); X = np.round(_rng.normal(size=(n, d)), 2); y = _rng.integers(0, 3, n)
    best = (None, None, 0.0)
    for j in range(d):
        v = np.unique(X[:, j])
        for t in (v[:-1] + v[1:]) / 2:
            L = X[:, j] <= t
            delta = _gini(y) - (L.mean() * _gini(y[L]) + (~L).mean() * _gini(y[~L]))
            if delta > best[2]:
                best = (j, t, delta)
    out = _need(best_split(X, y))
    return out[0] == best[0] and (best[1] is None or np.isclose(out[1], best[1])) and np.isclose(out[2], best[2])

def _t3():
    h, w = int(_rng.integers(4, 10)), int(_rng.integers(4, 10)); k = int(_rng.integers(1, 4)); s, p = int(_rng.integers(1, 3)), int(_rng.integers(0, 2))
    x = _rng.normal(size=(h, w)); K = _rng.normal(size=(k, k))
    out = _need(conv2d(x, K, s, p))
    ref = F.conv2d(torch.tensor(x)[None, None], torch.tensor(K)[None, None], stride=s, padding=p)[0, 0].numpy()
    return np.shape(out) == ref.shape and np.allclose(out, ref)

def _t4():
    n, d = int(_rng.integers(1, 6)), int(_rng.integers(1, 5)); X = _rng.normal(size=(n, d)) * 3
    rm, rv = _rng.normal(size=d), _rng.uniform(0.5, 3, d); g, b = _rng.normal(size=d), _rng.normal(size=d)
    bn = nn.BatchNorm1d(d).double()
    with torch.no_grad():
        bn.running_mean.copy_(torch.tensor(rm)); bn.running_var.copy_(torch.tensor(rv)); bn.weight.copy_(torch.tensor(g)); bn.bias.copy_(torch.tensor(b))
    bn.eval()
    out = _need(batchnorm_eval(X, rm, rv, g, b))
    return np.allclose(out, bn(torch.tensor(X)).detach().numpy(), atol=1e-6)

def _t5():
    torch.manual_seed(int(_rng.integers(0, 1000)))
    X_tr, X_va = torch.randn(30, 4), torch.randn(20, 4); w = torch.randn(4, 3)
    y_tr, y_va = (X_tr @ w).argmax(1), (X_va @ w).argmax(1)
    model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 3)); ref = copy.deepcopy(model)
    epochs, patience, lr = int(_rng.integers(5, 40)), int(_rng.integers(1, 5)), 0.05
    best_epoch = _need(fit_early_stopping(model, X_tr, y_tr, X_va, y_va, epochs, patience, lr))
    opt = torch.optim.SGD(ref.parameters(), lr=lr); lf = nn.CrossEntropyLoss(); best, best_state, be = float("inf"), None, 0
    for epoch in range(epochs):
        ref.train(); loss = lf(ref(X_tr), y_tr); opt.zero_grad(); loss.backward(); opt.step()
        ref.eval()
        with torch.no_grad():
            v = lf(ref(X_va), y_va).item()
        if v < best:
            best, best_state, be = v, copy.deepcopy(ref.state_dict()), epoch
        elif epoch - be >= patience:
            break
    ref.load_state_dict(best_state)
    return best_epoch == be and all(torch.allclose(a, b) for a, b in zip(model.state_dict().values(), ref.state_dict().values()))

TESTS = {i: globals()[f"_t{i}"] for i in range(1, 6)}
''',
"07": r'''
import torch
import torch.nn as nn
import torch.nn.functional as F
_rng = np.random.default_rng()

def _need(x):
    if x is None:
        raise TypeError("'NoneType' returned")
    return x

def _t1():
    h, w = int(_rng.integers(3, 9)), int(_rng.integers(3, 9)); a = int(_rng.integers(1, min(h, w, 4) + 1))
    x = _rng.normal(size=(h, w)); K = _rng.normal(size=(a, a))
    out = _need(conv2d_single(x, K))
    ref = F.conv2d(torch.tensor(x)[None, None], torch.tensor(K)[None, None])[0, 0].numpy()
    return np.shape(out) == ref.shape and np.allclose(out, ref)

def _t2():
    n, d = int(_rng.integers(2, 8)), int(_rng.integers(1, 5)); X = _rng.normal(size=(n, d)) * 4 + 2; g, b = _rng.normal(size=d), _rng.normal(size=d)
    out = _need(batchnorm_train(X, g, b))
    return np.allclose(out, g * (X - X.mean(0)) / np.sqrt(X.var(0) + 1e-5) + b)

def _t3():
    ok = tuple(_need(early_stop([1.0, 0.8, 0.75, 0.77, 0.74, 0.76, 0.78, 0.79], 2))) == (6, 4)
    v = list(_rng.uniform(0, 1, int(_rng.integers(3, 20)))); pat = int(_rng.integers(1, 5)); best, stop = 0, len(v) - 1
    for e in range(len(v)):
        if v[e] < v[best]:
            best = e
        elif e - best >= pat:
            stop = e; break
    return bool(ok) and tuple(early_stop(v, pat)) == (stop, best)

def _t4():
    k, d = int(_rng.integers(2, 4)), int(_rng.integers(1, 3)); centers = np.arange(k)[:, None] * 20.0 + _rng.normal(size=(k, d))   # well apart: no centre is left empty
    X = np.concatenate([c + _rng.normal(size=(int(_rng.integers(2, 6)), d)) for c in centers]); mu = centers + _rng.normal(size=(k, d))
    c_ref = np.linalg.norm(X[:, None] - mu[None], axis=2).argmin(1); mu_ref = np.array([X[c_ref == j].mean(0) for j in range(k)])
    c, new_mu = _need(kmeans_step(X, mu))
    return np.array_equal(c, c_ref) and np.allclose(new_mu, mu_ref)

def _t5():
    n, d = int(_rng.integers(6, 20)), int(_rng.integers(2, 5)); k = int(_rng.integers(1, d + 1)); X = _rng.normal(size=(n, d)) * _rng.uniform(0.5, 3, d)
    Z = np.asarray(_need(pca_scores(X, k)))
    Xc = X - X.mean(0); lam, W = np.linalg.eigh(Xc.T @ Xc / (n - 1)); W = W[:, np.argsort(lam)[::-1]]
    ref = Xc @ W[:, :k]
    return Z.shape == (n, k) and np.allclose(np.abs(Z), np.abs(ref), atol=1e-6)

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
