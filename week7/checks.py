"""Exercise checkers for Week 7. Each notebook's setup cell runs `checks.load(globals(), "02")`, which defines `check(n)`.

The tests build random data every run and compare your function with a reference. Reading this file spoils the exercises."""

SUITES = {
"00": r'''
import torch
import torch.nn.functional as F
_rng = np.random.default_rng()

def _need(x):
    if x is None:
        raise TypeError("'NoneType' returned")
    return x

def _t1():
    ok = True
    for size in (2, 3):
        h, w = size * int(_rng.integers(1, 5)), size * int(_rng.integers(1, 5))
        x = _rng.normal(size=(h, w))
        out = np.asarray(_need(maxpool2d(x, size)))
        ref = F.max_pool2d(torch.tensor(x)[None, None], kernel_size=size)[0, 0].numpy()
        ok = ok and out.shape == ref.shape and np.allclose(out, ref)
    return ok

TESTS = {1: _t1}
''',
"01": r'''
import copy
import torch
import torch.nn as nn
_rng = np.random.default_rng()

def _need(x):
    if x is None:
        raise TypeError("'NoneType' returned")
    return x

def _t1():
    n, d = int(_rng.integers(2, 7)), int(_rng.integers(2, 6)); X = _rng.normal(size=(n, d)) * 3 + 1
    g, b = _rng.normal(size=d), _rng.normal(size=d)
    out = _need(layernorm(X, g, b))
    mu, var = X.mean(axis=1, keepdims=True), X.var(axis=1, keepdims=True)
    return np.allclose(out, g * (X - mu) / np.sqrt(var + 1e-5) + b)

def _model():
    torch.manual_seed(int(_rng.integers(0, 1000)))
    h = int(_rng.integers(3, 9))
    return nn.Sequential(nn.Linear(4, h), nn.ReLU(), nn.Linear(h, 5), nn.ReLU(), nn.Linear(5, 3))

def _t2():
    m = _model(); before = copy.deepcopy(m.state_dict()); k = int(_rng.integers(2, 7))
    new = _need(freeze_and_replace_head(m, k))
    same_src = all(torch.equal(before[n], m.state_dict()[n]) for n in before) and all(p.requires_grad for p in m.parameters())
    head = new[-1]
    frozen_ok = all(not p.requires_grad for p in new[:-1].parameters()) and all(p.requires_grad for p in head.parameters())
    shapes_ok = isinstance(head, nn.Linear) and head.in_features == 5 and head.out_features == k
    copied = torch.equal(new[0].weight, m[0].weight) and new[0].weight is not m[0].weight
    return same_src and frozen_ok and shapes_ok and copied and tuple(new(torch.zeros(2, 4)).shape) == (2, k)

def _t3():
    m = _model(); ok = _need(count_trainable(m)) == sum(p.numel() for p in m.parameters())
    for p in m[0].parameters():
        p.requires_grad = False
    ok = ok and count_trainable(m) == sum(p.numel() for p in m[2:].parameters())
    for p in m.parameters():
        p.requires_grad = False
    return ok and count_trainable(m) == 0

def _t4():
    m = _model(); lh, lb = float(_rng.uniform(1e-3, 1e-2)), float(_rng.uniform(1e-6, 1e-4))
    opt = _need(two_lr_optimizer(m, lh, lb))
    if not isinstance(opt, torch.optim.Adam) or len(opt.param_groups) != 2:
        return False
    g = {round(pg["lr"], 12): {id(p) for p in pg["params"]} for pg in opt.param_groups}
    head = {id(p) for p in m[-1].parameters()}; body = {id(p) for p in m[:-1].parameters()}
    return g.get(round(lh, 12)) == head and g.get(round(lb, 12)) == body

TESTS = {i: globals()[f"_t{i}"] for i in range(1, 5)}
''',
"02": r'''
import copy
import torch
import torch.nn as nn
_rng = np.random.default_rng()

def _need(x):
    if x is None:
        raise TypeError("'NoneType' returned")
    return x

def _t1():
    n, d = int(_rng.integers(8, 30)), int(_rng.integers(2, 6)); k = int(_rng.integers(1, d + 1))
    X = _rng.normal(size=(n, d)) * _rng.uniform(0.5, 3, d) + _rng.normal(size=d)
    out = np.asarray(_need(pca_reconstruct(X, k)))
    from sklearn.decomposition import PCA
    p = PCA(k).fit(X)
    return out.shape == X.shape and np.allclose(out, p.inverse_transform(p.transform(X)), atol=1e-8)

def _t2():
    d, k = int(_rng.integers(5, 20)), int(_rng.integers(1, 5))
    m = _need(Autoencoder(d, k))
    shapes = [tuple(p.shape) for p in m.parameters()]
    want = [(128, d), (128,), (k, 128), (k,), (128, k), (128,), (d, 128), (d,)]
    x = torch.rand(3, d); out = m(x)
    z = m.encoder(x)
    return isinstance(m, nn.Module) and shapes == want and tuple(out.shape) == (3, d) and tuple(z.shape) == (3, k) and bool((out > 0).all() and (out < 1).all())

def _t3():
    torch.manual_seed(int(_rng.integers(0, 1000))); d = int(_rng.integers(3, 10))
    m = nn.Sequential(nn.Linear(d, 2), nn.Linear(2, d)); X = torch.randn(int(_rng.integers(2, 9)), d)
    out = _need(recon_errors(m, X))
    out = out.detach().numpy() if torch.is_tensor(out) else np.asarray(out)
    with torch.no_grad():
        ref = ((m(X) - X) ** 2).mean(dim=1).numpy()
    return out.shape == ref.shape and np.allclose(out, ref, atol=1e-6)

def _t4():
    s = int(_rng.integers(0, 1000)); torch.manual_seed(s); d = int(_rng.integers(3, 10)); std = float(_rng.uniform(0.1, 0.6))
    m = nn.Sequential(nn.Linear(d, 2), nn.ReLU(), nn.Linear(2, d), nn.Sigmoid()); r = copy.deepcopy(m)
    opt = torch.optim.SGD(m.parameters(), lr=0.1); opt_r = torch.optim.SGD(r.parameters(), lr=0.1)
    xb = torch.rand(5, d)
    torch.manual_seed(s + 1); out = _need(denoising_step(m, opt, xb, std))
    torch.manual_seed(s + 1); noisy = (xb + std * torch.randn_like(xb)).clamp(0, 1)
    loss = nn.functional.mse_loss(r(noisy), xb); opt_r.zero_grad(); loss.backward(); opt_r.step()
    return np.isclose(float(out), loss.item()) and all(torch.allclose(a, b) for a, b in zip(m.parameters(), r.parameters()))

def _t5():
    ok = True
    for _ in range(10):
        n, k, s = int(_rng.integers(2, 15)), int(_rng.integers(1, 6)), int(_rng.integers(1, 4)); p = int(_rng.integers(0, k // 2 + 1)); op = int(_rng.integers(0, s))
        if (n - 1) * s - 2 * p + k + op < 1:
            continue
        y = nn.ConvTranspose2d(1, 1, k, stride=s, padding=p, output_padding=op)(torch.zeros(1, 1, n, n))
        ok = ok and _need(deconv_out_size(n, k, p, s, op)) == y.shape[-1]
    return ok

TESTS = {i: globals()[f"_t{i}"] for i in range(1, 6)}
''',
"03": r'''
import torch
import torch.nn as nn
import torch.nn.functional as F
_rng = np.random.default_rng()

def _need(x):
    if x is None:
        raise TypeError("'NoneType' returned")
    return x

def _t1():
    n, d, K = int(_rng.integers(6, 20)), int(_rng.integers(1, 4)), int(_rng.integers(2, 4))
    X = _rng.normal(size=(n, d)); G = _rng.random((n, K)); G = G / G.sum(axis=1, keepdims=True)
    pi, mu, Sigma = _need(gmm_m_step(X, G))
    Nk = G.sum(axis=0); mu_r = G.T @ X / Nk[:, None]
    S_r = np.array([(G[:, k, None] * (X - mu_r[k])).T @ (X - mu_r[k]) / Nk[k] + 1e-6 * np.eye(d) for k in range(K)])
    return np.allclose(pi, Nk / n) and np.allclose(mu, mu_r) and np.allclose(Sigma, S_r)

def _t2():
    s = tuple(int(v) for v in _rng.integers(1, 5, 2))
    mu, lv, eps = torch.randn(s), torch.randn(s), torch.randn(s)
    out = _need(reparameterize(mu, lv, eps))
    return torch.allclose(out, mu + torch.exp(0.5 * lv) * eps, atol=1e-6)

def _t3():
    n, k = int(_rng.integers(1, 6)), int(_rng.integers(1, 5)); mu, lv = torch.randn(n, k), torch.randn(n, k)
    out = _need(kl_per_sample(mu, lv))
    ref = 0.5 * (lv.exp() + mu ** 2 - 1 - lv).sum(dim=1)
    return tuple(out.shape) == (n,) and torch.allclose(out, ref, atol=1e-5)

def _t4():
    n, d, k = int(_rng.integers(1, 6)), int(_rng.integers(3, 10)), int(_rng.integers(1, 4))
    x, xh = torch.rand(n, d), torch.rand(n, d) * 0.98 + 0.01; mu, lv = torch.randn(n, k), torch.randn(n, k)
    out = _need(vae_loss(x, xh, mu, lv))
    ref = (F.binary_cross_entropy(xh, x, reduction="sum") - 0.5 * torch.sum(1 + lv - mu ** 2 - lv.exp())) / n
    return torch.is_tensor(out) and out.dim() == 0 and torch.allclose(out, ref, atol=1e-4)

def _t5():
    d, h, k = int(_rng.integers(5, 20)), int(_rng.integers(4, 30)), int(_rng.integers(1, 4))
    m = _need(VAE(d, h, k))
    want = [(h, d), (h,), (k, h), (k,), (k, h), (k,), (h, k), (h,), (d, h), (d,)]
    if [tuple(p.shape) for p in m.parameters()] != want:
        return False
    x = torch.rand(4, d); torch.manual_seed(0); xh, mu, lv = m(x)
    hdn = m.encoder(x); mu_r, lv_r = m.to_mu(hdn), m.to_logvar(hdn)
    torch.manual_seed(0); z = mu_r + torch.exp(0.5 * lv_r) * torch.randn_like(mu_r)
    return torch.allclose(mu, mu_r) and torch.allclose(lv, lv_r) and torch.allclose(xh, m.decoder(z), atol=1e-6) and bool((xh > 0).all() and (xh < 1).all())

TESTS = {i: globals()[f"_t{i}"] for i in range(1, 6)}
''',
"04": r'''
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import average_precision_score
from scipy.spatial.distance import mahalanobis as _maha
_rng = np.random.default_rng()

def _need(x):
    if x is None:
        raise TypeError("'NoneType' returned")
    return x

def _t1():
    n, d = int(_rng.integers(10, 60)), int(_rng.integers(1, 4)); X = np.round(_rng.normal(size=(n, d)), 2)
    eps, ms = float(_rng.uniform(0.3, 1.5)), int(_rng.integers(2, 8))
    nb = NearestNeighbors(radius=eps).fit(X).radius_neighbors(X, return_distance=False)
    ref = np.array([len(v) >= ms for v in nb])
    out = np.asarray(_need(core_points(X, eps, ms)))
    return out.dtype == bool and np.array_equal(out, ref)

def _t2():
    n, d = int(_rng.integers(20, 60)), int(_rng.integers(2, 4)); A = _rng.normal(size=(d, d))
    Xtr = _rng.normal(size=(n, d)) @ A + _rng.normal(size=d); X = _rng.normal(size=(int(_rng.integers(1, 8)), d)) * 2
    out = np.asarray(_need(mahalanobis_scores(Xtr, X)))
    mu, VI = Xtr.mean(axis=0), np.linalg.inv(np.cov(Xtr.T))
    ref = np.array([_maha(x, mu, VI) for x in X])
    return out.shape == ref.shape and np.allclose(out, ref)

def _t3():
    s = _rng.exponential(size=int(_rng.integers(20, 200))); q = float(_rng.uniform(0.8, 0.999))
    return np.isclose(float(_need(threshold_at(s, q))), np.quantile(s, q))

def _t4():
    n = int(_rng.integers(5, 40)); y = _rng.integers(0, 2, n); p = _rng.integers(0, 2, n)
    y[0], p[0] = 1, 1
    pr, rc = _need(precision_recall(y, p))
    tp = np.sum((y == 1) & (p == 1))
    ok = np.isclose(pr, tp / p.sum()) and np.isclose(rc, tp / y.sum())
    pr0, rc0 = precision_recall(np.array([1, 0, 1]), np.array([0, 0, 0]))
    return bool(ok) and pr0 == 0 and rc0 == 0

def _t5():
    n = int(_rng.integers(5, 60)); y = _rng.integers(0, 2, n); y[0], y[1] = 1, 0; s = _rng.random(n)
    return np.isclose(float(_need(average_precision(y, s))), average_precision_score(y, s))

TESTS = {i: globals()[f"_t{i}"] for i in range(1, 6)}
''',
"05": r'''
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
_rng = np.random.default_rng()

def _need(x):
    if x is None:
        raise TypeError("'NoneType' returned")
    return x

def _t1():
    k = int(_rng.integers(2, 4)); centers = np.arange(k)[:, None] * 12.0 + _rng.normal(size=(k, 2))
    X = np.concatenate([c + _rng.normal(size=(int(_rng.integers(5, 15)), 2)) for c in centers]); seed = int(_rng.integers(0, 100))
    rng = np.random.default_rng(seed)
    mu = X[rng.choice(len(X), size=k, replace=False)]
    for _ in range(100):
        c = np.linalg.norm(X[:, None, :] - mu[None, :, :], axis=2).argmin(axis=1)
        if any((c == j).sum() == 0 for j in range(k)):
            return True                                # an empty cluster: skip this draw
        new = np.array([X[c == j].mean(axis=0) for j in range(k)])
        if np.allclose(new, mu):
            break
        mu = new
    J = np.sum((X - mu[c]) ** 2)
    m2, c2, J2 = _need(kmeans(X, k, seed=seed))
    return np.allclose(m2, mu) and np.array_equal(c2, c) and np.isclose(J2, J)

def _t2():
    n, d, h, C = int(_rng.integers(1, 6)), int(_rng.integers(2, 7)), int(_rng.integers(2, 9)), int(_rng.integers(2, 5))
    X = torch.randn(n, d); W1, b1, W2, b2 = torch.randn(d, h), torch.randn(h), torch.randn(h, C), torch.randn(C)
    out = _need(mlp_forward(X, W1, b1, W2, b2))
    return torch.allclose(out, torch.sigmoid(X @ W1 + b1) @ W2 + b2, atol=1e-5)

def _t3():
    torch.manual_seed(int(_rng.integers(0, 1000))); d = int(_rng.integers(3, 9))
    m = nn.Sequential(nn.Linear(d, 2), nn.Linear(2, d)); Xtr = torch.randn(int(_rng.integers(20, 60)), d); X = torch.randn(int(_rng.integers(3, 12)), d) * 2
    q = float(_rng.uniform(0.8, 0.99))
    with torch.no_grad():
        st = ((m(Xtr) - Xtr) ** 2).mean(dim=1).numpy(); s = ((m(X) - X) ** 2).mean(dim=1).numpy()
    out = np.asarray(_need(ae_flags(m, Xtr, X, q)))
    return out.dtype == bool and np.array_equal(out, s > np.quantile(st, q))

def _t4():
    L = int(_rng.integers(3, 6)); layers = []
    for i in range(L):
        layers += [nn.Linear(4, 4), nn.ReLU()]
    m = nn.Sequential(*layers); n = int(_rng.integers(0, len(m) + 1))
    out = _need(freeze_first(m, n))
    exp_frozen = all(not p.requires_grad for p in m[:n].parameters()) and all(p.requires_grad for p in m[n:].parameters())
    return exp_frozen and out == sum(p.numel() for p in m[n:].parameters())

def _t5():
    n = int(_rng.integers(4, 40)); y = _rng.integers(0, 2, n); y[0], y[1] = 0, 1; s = np.round(_rng.random(n), 1)
    return np.isclose(float(_need(auc_pairs(y, s))), roc_auc_score(y, s))

TESTS = {i: globals()[f"_t{i}"] for i in range(1, 6)}
''',
"06": r'''
import copy
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from scipy.spatial.distance import mahalanobis as _maha
_rng = np.random.default_rng()

def _need(x):
    if x is None:
        raise TypeError("'NoneType' returned")
    return x

def _t1():
    torch.manual_seed(int(_rng.integers(0, 1000))); h = int(_rng.integers(3, 8)); k = int(_rng.integers(2, 6))
    m = nn.Sequential(nn.Linear(4, h), nn.ReLU(), nn.Linear(h, 3)); before = copy.deepcopy(m.state_dict())
    new = _need(freeze_and_replace_head(m, k))
    same = all(torch.equal(before[n], m.state_dict()[n]) for n in before) and all(p.requires_grad for p in m.parameters())
    return same and isinstance(new[-1], nn.Linear) and new[-1].in_features == h and new[-1].out_features == k \
        and all(not p.requires_grad for p in new[:-1].parameters()) and all(p.requires_grad for p in new[-1].parameters())

def _t2():
    torch.manual_seed(int(_rng.integers(0, 1000))); d = int(_rng.integers(3, 9))
    m = nn.Sequential(nn.Linear(d, 2), nn.Linear(2, d)); X = torch.randn(int(_rng.integers(2, 9)), d)
    out = _need(recon_errors(m, X)); out = out.detach().numpy() if torch.is_tensor(out) else np.asarray(out)
    with torch.no_grad():
        ref = ((m(X) - X) ** 2).mean(dim=1).numpy()
    return out.shape == ref.shape and np.allclose(out, ref, atol=1e-6)

def _t3():
    n, k = int(_rng.integers(1, 6)), int(_rng.integers(1, 5)); mu, lv = torch.randn(n, k), torch.randn(n, k)
    out = _need(kl_per_sample(mu, lv))
    return tuple(out.shape) == (n,) and torch.allclose(out, 0.5 * (lv.exp() + mu ** 2 - 1 - lv).sum(dim=1), atol=1e-5)

def _t4():
    n, d = int(_rng.integers(20, 60)), int(_rng.integers(2, 4)); A = _rng.normal(size=(d, d))
    Xtr = _rng.normal(size=(n, d)) @ A; X = _rng.normal(size=(int(_rng.integers(1, 8)), d))
    out = np.asarray(_need(mahalanobis_scores(Xtr, X)))
    mu, VI = Xtr.mean(axis=0), np.linalg.inv(np.cov(Xtr.T))
    return np.allclose(out, [_maha(x, mu, VI) for x in X])

def _t5():
    n = int(_rng.integers(5, 60)); y = _rng.integers(0, 2, n); y[0], y[1] = 1, 0; s = _rng.random(n)
    return np.isclose(float(_need(average_precision(y, s))), average_precision_score(y, s))

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
