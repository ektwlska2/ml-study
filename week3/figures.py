"""Picture helpers for Week 3. Each function draws one figure from the lessons.

You don't need to read this file to follow the lessons: the pictures are the point.
If you're curious how a plot is made, every function is plain matplotlib.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

C0, C1 = "#2f6fb3", "#e07b2a"           # class 0 = blue, class 1 = orange
SOFT = ListedColormap(["#d7e6f6", "#fbe3cf"])
GREY = "#555555"
plt.rcParams.update({"figure.dpi": 96, "axes.grid": True, "grid.alpha": 0.25, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})

# The 9 points used all through notebook 01 (P1–P9). P5 and P9 sit on the "wrong" side on purpose.
X_LOGI = np.array([[3, 2.5], [2, 3.5], [3.5, 3.5], [4, 1.5], [1, 1.5], [1, 0.5], [0.5, 2], [2, 1], [3, 3]], float)
Y_LOGI = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0])


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def _scatter(ax, X, y, names=True, size=None):
    for c, col, lab in [(0, C0, "y = 0"), (1, C1, "y = 1")]:
        m = y == c
        ax.scatter(X[m, 0], X[m, 1], s=70 if size is None else size[m], c=col, edgecolor="k", linewidth=0.6, label=lab, zorder=3)
    if names:
        for i, (a, b) in enumerate(X):
            ax.annotate(f"P{i + 1}", (a, b), textcoords="offset points", xytext=(6, 5), fontsize=8, color=GREY)


def _line(ax, w, b, xlim=(-0.5, 4.8), **kw):
    """Draw the line w·x + b = 0."""
    xs = np.linspace(*xlim, 200)
    if abs(w[1]) > 1e-9:
        ax.plot(xs, -(w[0] * xs + b) / w[1], **kw)
    else:
        ax.axvline(-b / w[0], **kw)


def _frame(ax, xlim=(-0.5, 4.8), ylim=(-0.5, 4.5)):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")


def _halves(ax, w, b, xlim=(-0.5, 4.8), ylim=(-0.5, 4.5)):
    g1, g2 = np.meshgrid(np.linspace(*xlim, 300), np.linspace(*ylim, 300))
    z = w[0] * g1 + w[1] * g2 + b
    ax.contourf(g1, g2, (z >= 0).astype(int), levels=[-0.5, 0.5, 1.5], cmap=SOFT, alpha=0.9, zorder=0)


def _arrow_w(ax, w, b, color="k", label="$w$"):
    """Draw w starting from the point of the boundary closest to the centre of the plot."""
    w = np.asarray(w, float)
    c = np.array([2.0, 2.0])
    foot = c - (w @ c + b) / (w @ w) * w            # project the centre onto the line
    d = w / np.linalg.norm(w)
    ax.annotate("", xy=foot + d, xytext=foot, arrowprops=dict(arrowstyle="-|>", color=color, lw=2))
    ax.annotate(label, foot + 1.08 * d, fontsize=11, color=color, fontweight="bold")


# ------------------------------------------------------------------ notebook 01: logistic regression in pictures
def fig_setup(w=(0.5, 1.0), b=-1.0):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    _halves(ax, w, b)
    _line(ax, w, b, color="k", lw=2, label="boundary $w^\\top x + b = 0$")
    _scatter(ax, X_LOGI, Y_LOGI)
    _arrow_w(ax, w, b)
    ax.text(3.2, 0.1, "predict 1\n($z > 0$)", color=C1, fontsize=10, fontweight="bold")
    ax.text(-0.35, -0.35, "predict 0 ($z < 0$)", color=C0, fontsize=10, fontweight="bold")
    _frame(ax); ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"The model $w = ({w[0]:g}, {w[1]:g})$, $b = {b:g}$ is a line")
    plt.show()


def fig_score_contours(w=(0.5, 1.0), b=-1.0):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    g1, g2 = np.meshgrid(np.linspace(-0.5, 4.8, 300), np.linspace(-0.5, 4.5, 300))
    z = w[0] * g1 + w[1] * g2 + b
    cs = ax.contour(g1, g2, z, levels=[-2, -1, 0, 1, 2, 3, 4, 5], colors=GREY, linewidths=[0.8] * 2 + [2.4] + [0.8] * 5, negative_linestyles="solid")
    ax.clabel(cs, fmt=lambda v: f"z = {v:g}", fontsize=8)
    _scatter(ax, X_LOGI, Y_LOGI)
    _arrow_w(ax, w, b, label="$w$ (uphill)")
    _frame(ax); ax.set_title("The score $z = w^\\top x + b$: a height over the plane")
    plt.show()


def fig_sigmoid_and_map(w=(0.5, 1.0), b=-1.0, mark=(5, 7, 2)):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    zs = np.linspace(-6, 6, 300)
    a1.plot(zs, sigmoid(zs), color="k", lw=2)
    a1.axhline(0.5, color=GREY, lw=0.8, ls="--"); a1.axvline(0, color=GREY, lw=0.8, ls="--")
    for i in mark:
        z = np.dot(w, X_LOGI[i]) + b
        col = C1 if Y_LOGI[i] else C0
        a1.plot([z, z], [0, sigmoid(z)], color=col, lw=1); a1.plot(z, sigmoid(z), "o", color=col, mec="k")
        a1.annotate(f"P{i + 1}: z = {z:.2f}\n$\\hat p$ = {sigmoid(z):.3f}", (z, sigmoid(z)), xytext=(8, -26), textcoords="offset points", fontsize=8)
    a1.set_xlabel("score $z$"); a1.set_ylabel("$\\hat p = \\sigma(z)$"); a1.set_title("The sigmoid squashes a height into (0, 1)")
    g1, g2 = np.meshgrid(np.linspace(-0.5, 4.8, 300), np.linspace(-0.5, 4.5, 300))
    p = sigmoid(w[0] * g1 + w[1] * g2 + b)
    im = a2.contourf(g1, g2, p, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.4)
    cs = a2.contour(g1, g2, p, levels=[0.1, 0.5, 0.9], colors="k", linewidths=[0.8, 2.2, 0.8])
    a2.clabel(cs, fmt=lambda v: f"{v:g}", fontsize=8)
    _scatter(a2, X_LOGI, Y_LOGI)
    _frame(a2); a2.set_title("$\\hat p$ at every point of the plane")
    fig.colorbar(im, ax=a2, fraction=0.046, label="$\\hat p$")
    plt.tight_layout(); plt.show()


def fig_three_knobs(w=(0.67, 0.66), b=-2.6):
    w = np.asarray(w, float)
    rot = np.array([[np.cos(0.5), -np.sin(0.5)], [np.sin(0.5), np.cos(0.5)]]) @ w
    panels = [("1. Change the direction of $w$:\nthe line ROTATES", [(w, b, "before"), (rot, b, "after")]),
              ("2. Change $b$:\nthe line SHIFTS (stays parallel)", [(w, b, "before"), (w, b + 1.0, "after: b + 1")]),
              ("3. Multiply $w$ and $b$ by 3:\nsame line, SHARPER", [(w, b, "before"), (3 * w, 3 * b, "after: ×3")])]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    g1, g2 = np.meshgrid(np.linspace(-0.5, 4.8, 250), np.linspace(-0.5, 4.5, 250))
    for ax, (title, models) in zip(axes, panels):
        for (ww, bb, lab), ls in zip(models, ["--", "-"]):
            p = sigmoid(ww[0] * g1 + ww[1] * g2 + bb)
            if ax is axes[2]:
                cs = ax.contour(g1, g2, p, levels=[0.1, 0.9], colors=[C0, C1], linestyles=ls, linewidths=1)
            _line(ax, ww, bb, color="k", ls=ls, lw=2, label=lab)
        _scatter(ax, X_LOGI, Y_LOGI, names=False)
        _frame(ax); ax.set_title(title, fontsize=10); ax.legend(loc="upper left", fontsize=8)
    axes[2].text(-0.3, -0.3, "thin lines: where $\\hat p$ = 0.1 and 0.9", fontsize=8)
    plt.tight_layout(); plt.show()


def fig_losses():
    p = np.linspace(0.005, 0.995, 300)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(p, -np.log(p), color=C1, lw=2, label="true label y = 1:  loss $= -\\log \\hat p$")
    ax.plot(p, -np.log(1 - p), color=C0, lw=2, label="true label y = 0:  loss $= -\\log(1 - \\hat p)$")
    for pp, lab in [(0.9, "right and sure: 0.11"), (0.2, "wrong and sure: 1.61")]:
        ax.plot(pp, -np.log(pp), "o", color=C1, mec="k")
        ax.annotate(lab, (pp, -np.log(pp)), xytext=(6, 8), textcoords="offset points", fontsize=8)
    ax.set_ylim(0, 5); ax.set_xlabel("$\\hat p$ (the model's probability of class 1)"); ax.set_ylabel("loss")
    ax.legend(fontsize=8, loc="upper center"); ax.set_title("The loss is the surprise of the true label")
    plt.tight_layout(); plt.show()


def fig_pulls(w=(0.5, 1.0), b=-1.0):
    p = sigmoid(X_LOGI @ np.asarray(w) + b)
    err = p - Y_LOGI
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    _halves(ax, w, b)
    _line(ax, w, b, color="k", lw=2)
    _scatter(ax, X_LOGI, Y_LOGI, names=False, size=40 + 900 * np.abs(err))
    for i, (a, c) in enumerate(X_LOGI):
        ax.annotate(f"P{i + 1}: $\\hat p - y$ = {err[i]:+.2f}", (a, c), xytext=(9, -12), textcoords="offset points", fontsize=8)
    _frame(ax); ax.set_title("Each point pulls with strength $|\\hat p - y|$ (dot size)")
    plt.show()


def fig_rotation(w=(1.0, -0.5), x=(1.0, 2.0), y=1, eta=1.0):
    """Vector picture with b = 0: the step adds a multiple of x to w, so w turns toward x (y = 1) or away from it (y = 0)."""
    w = np.asarray(w, float); x = np.asarray(x, float)
    p = sigmoid(w @ x)
    step = -eta * (p - y) * x
    w2 = w + step
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    lim = 3
    for ww, ls, lab in [(w, "--", "old boundary"), (w2, "-", "new boundary")]:
        _line(ax, ww, 0.0, xlim=(-lim, lim), color="k", ls=ls, lw=1.6, label=lab)
    col = C1 if y else C0
    ax.annotate("", xy=x, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=2))
    ax.plot(*x, "o", color=col, mec="k", ms=9)
    ax.annotate(f"x = ({x[0]:g}, {x[1]:g}), y = {y}", x, xytext=(8, 4), textcoords="offset points", color=col)
    ax.annotate("", xy=w, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="grey", lw=2))
    ax.annotate("old $w$", w, xytext=(6, -12), textcoords="offset points", color="grey")
    ax.annotate("", xy=w2, xytext=w, arrowprops=dict(arrowstyle="-|>", color="green", lw=1.6, ls=":"))
    ax.text(0.98, 0.03, f"green: the step $-\\eta(\\hat p - y)\\,x$ = ({step[0]:+.2f}, {step[1]:+.2f})", transform=ax.transAxes, ha="right", color="green", fontsize=8,
            bbox=dict(fc="white", ec="none", alpha=0.85))
    ax.annotate("", xy=w2, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="k", lw=2))
    ax.annotate("new $w$", w2, xytext=(6, 4), textcoords="offset points")
    ax.axhline(0, color=GREY, lw=0.6); ax.axvline(0, color=GREY, lw=0.6)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal"); ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"$\\hat p$ was {p:.2f}, the label is {y}: $w$ turns {'toward' if y else 'away from'} $x$")
    plt.show()


def logistic_path(w=(0.5, 1.0), b=-1.0, eta=0.5, steps=3000):
    """Full-batch gradient descent on the 9 points. Returns the list of (w, b, loss) after 0, 1, 2, ... steps."""
    w = np.asarray(w, float); out = []
    for _ in range(steps + 1):
        p = sigmoid(X_LOGI @ w + b)
        loss = -np.mean(Y_LOGI * np.log(p) + (1 - Y_LOGI) * np.log(1 - p))
        out.append((w.copy(), b, loss))
        w = w - eta * X_LOGI.T @ (p - Y_LOGI) / len(Y_LOGI)
        b = b - eta * np.mean(p - Y_LOGI)
    return out


def fig_one_step(w=(0.5, 1.0), b=-1.0, eta=0.5):
    (w0, b0, l0), (w1, b1, l1) = logistic_path(w, b, eta, steps=1)
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    _line(ax, w0, b0, color="k", ls="--", lw=1.6, label=f"before: loss {l0:.3f}")
    _line(ax, w1, b1, color="k", lw=2.2, label=f"after 1 step: loss {l1:.3f}")
    _halves(ax, w1, b1)
    _scatter(ax, X_LOGI, Y_LOGI)
    _frame(ax); ax.legend(loc="upper left", fontsize=8); ax.set_title("One gradient step on all 9 points")
    plt.show()


def fig_many_steps(w=(0.5, 1.0), b=-1.0, eta=0.5, show=(0, 1, 5, 20, 100, 3000)):
    path = logistic_path(w, b, eta, steps=max(show))
    fig = plt.figure(figsize=(13, 6.6))
    for k, t in enumerate(show):
        ax = fig.add_subplot(2, 4, k + 1 + (k >= 3))
        wt, bt, lt = path[t]
        _halves(ax, wt, bt); _line(ax, wt, bt, color="k", lw=2)
        _scatter(ax, X_LOGI, Y_LOGI, names=False, size=np.full(9, 30))
        _frame(ax); ax.set_xlabel(""); ax.set_ylabel("")
        ax.set_title(f"step {t}: loss {lt:.3f}", fontsize=9)
    ax = fig.add_subplot(1, 4, 4)
    ax.plot([l for _, _, l in path], color="k"); ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("step"); ax.set_ylabel("average loss"); ax.set_title("The loss goes down, then flattens", fontsize=9)
    plt.tight_layout(); plt.show()


def fig_thresholds(w=(0.672, 0.663), b=-2.61, ts=(0.2, 0.5, 0.8)):
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    styles = [":", "-", "--"]
    for t, ls in zip(ts, styles):
        shift = np.log(t / (1 - t))                     # predict 1 when z >= logit(t)
        pred = (X_LOGI @ np.asarray(w) + b >= shift).astype(int)
        tp = int(((pred == 1) & (Y_LOGI == 1)).sum()); fp = int(((pred == 1) & (Y_LOGI == 0)).sum())
        _line(ax, w, b - shift, color="k", ls=ls, lw=1.8, label=f"t = {t}: line z = {shift:+.2f}  (TP {tp}, FP {fp})")
    _scatter(ax, X_LOGI, Y_LOGI)
    _frame(ax); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=8)
    ax.set_title("A threshold is a line parallel to the boundary")
    plt.show()


def fig_pr_roc(y, scores):
    y = np.asarray(y); s = np.asarray(scores)
    order = np.argsort(-s); ys = y[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    prec = tp / np.arange(1, len(y) + 1); rec = tp / ys.sum(); fpr = fp / (1 - ys).sum()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.3))
    a1.step(np.r_[0, rec], np.r_[1, prec], where="pre", color="k", lw=1.6)
    a1.fill_between(np.r_[0, rec], np.r_[1, prec], step="pre", alpha=0.15, color=C1)
    for r, p, t in zip(rec, prec, s[order]):
        a1.plot(r, p, "o", color=C1, mec="k", ms=5)
        a1.annotate(f"{t:g}", (r, p), xytext=(4, 4), textcoords="offset points", fontsize=7)
    a1.axhline(ys.mean(), color=GREY, ls="--", lw=1); a1.text(0.02, ys.mean() + 0.02, f"random guessing: precision = {ys.mean():g}", fontsize=8)
    a1.set_xlim(0, 1.05); a1.set_ylim(0, 1.08); a1.set_xlabel("recall"); a1.set_ylabel("precision")
    a1.set_title("Precision–recall curve (labels = threshold)", fontsize=10)
    a2.step(np.r_[0, fpr], np.r_[0, rec], where="post", color="k", lw=1.6)
    a2.fill_between(np.r_[0, fpr], np.r_[0, rec], step="post", alpha=0.15, color=C0)
    a2.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1)
    a2.set_xlabel("false positive rate"); a2.set_ylabel("recall (true positive rate)")
    a2.set_title("ROC curve, for comparison", fontsize=10)
    plt.tight_layout(); plt.show()


# ------------------------------------------------------------------ notebook 02: kNN
KNN_TRAIN = np.array([[1, 2], [2, 1], [2, 3], [5, 4], [4, 5], [6, 6]], float)
KNN_LABELS = np.array([0, 0, 0, 1, 1, 1])


def fig_knn_query(q=(3.5, 3.0), k=3, X=KNN_TRAIN, y=KNN_LABELS):
    q = np.asarray(q, float)
    d = np.linalg.norm(X - q, axis=1); order = np.argsort(d)
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    for rank, i in enumerate(order):
        near = rank < k
        ax.plot([q[0], X[i, 0]], [q[1], X[i, 1]], color="k" if near else "#bbbbbb", lw=1.4 if near else 0.8, ls="-" if near else ":")
        mid = (q + X[i]) / 2
        ax.annotate(f"{d[i]:.2f}", mid, fontsize=8, color="k" if near else "#999999")
    ax.add_patch(plt.Circle(q, d[order[k - 1]], fill=False, ls="--", color=GREY))
    _scatter(ax, X, y)
    ax.plot(*q, "*", ms=16, color="k"); ax.annotate("query q", q, xytext=(8, -14), textcoords="offset points")
    ax.set_xlim(0, 7); ax.set_ylim(0, 7); ax.set_aspect("equal"); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title(f"k = {k}: the {k} closest points vote"); ax.legend(loc="upper left", fontsize=8)
    plt.show()


def _regions(ax, model, X, y, pad=0.6, n=250):
    x1 = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, n)
    x2 = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, n)
    g1, g2 = np.meshgrid(x1, x2)
    zz = model.predict(np.c_[g1.ravel(), g2.ravel()]).reshape(g1.shape)
    ax.contourf(g1, g2, zz, levels=[-0.5, 0.5, 1.5], cmap=SOFT, alpha=0.95)
    ax.scatter(X[:, 0], X[:, 1], c=np.where(y == 1, C1, C0), s=14, edgecolor="k", linewidth=0.3)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)


def moons(n=200, noise=0.3, seed=0):
    from sklearn.datasets import make_moons
    return make_moons(n_samples=n, noise=noise, random_state=seed)


def fig_knn_regions(ks=(1, 5, 25)):
    from sklearn.neighbors import KNeighborsClassifier
    X, y = moons()
    fig, axes = plt.subplots(1, len(ks), figsize=(4 * len(ks), 3.6))
    for ax, k in zip(axes, ks):
        m = KNeighborsClassifier(n_neighbors=k).fit(X, y)
        _regions(ax, m, X, y)
        ax.set_title(f"k = {k}: training accuracy {m.score(X, y):.2f}", fontsize=10)
    plt.tight_layout(); plt.show()


# ------------------------------------------------------------------ notebook 04: decision trees
def fig_impurity_curves():
    p = np.linspace(0.001, 0.999, 300)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ent = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    ax.plot(p, ent, color="k", lw=2, label="entropy (bits)")
    ax.plot(p, 2 * p * (1 - p), color=C1, lw=2, label="Gini $= 2p(1-p)$")
    ax.plot(p, np.minimum(p, 1 - p), color=C0, lw=1.5, ls="--", label="misclassification $= \\min(p, 1-p)$")
    ax.set_xlabel("fraction $p$ of class 1 in the node"); ax.set_ylabel("impurity")
    ax.legend(fontsize=8); ax.set_title("All three are 0 for a pure node and largest at 50/50")
    plt.tight_layout(); plt.show()


def fig_split_scan(x, y, name="feature"):
    """Points on one axis, every candidate threshold, and the weighted Gini of the two children for each."""
    x = np.asarray(x, float); y = np.asarray(y)
    xs = np.unique(x); cands = (xs[:-1] + xs[1:]) / 2

    def gini(v):
        if len(v) == 0:
            return 0.0
        p = v.mean()
        return 2 * p * (1 - p)
    scores = [(np.sum(x <= t) * gini(y[x <= t]) + np.sum(x > t) * gini(y[x > t])) / len(y) for t in cands]
    best = cands[int(np.argmin(scores))]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 4.6), sharex=True, gridspec_kw={"height_ratios": [1, 2]})
    a1.scatter(x, np.zeros_like(x), c=np.where(y == 1, C1, C0), s=80, edgecolor="k", zorder=3)
    for t in cands:
        a1.axvline(t, color="#bbbbbb", lw=0.8, ls=":")
    a1.axvline(best, color="k", lw=2); a1.set_yticks([]); a1.set_title("Every midpoint is a candidate threshold", fontsize=10)
    a2.plot(cands, scores, "o-", color="k")
    for t, s in zip(cands, scores):
        a2.annotate(f"{s:.3f}", (t, s), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
    a2.axhline(gini(y), color=GREY, ls="--", lw=1); a2.text(cands[0], gini(y) + 0.008, f"parent Gini {gini(y):.3f}", fontsize=8)
    a2.set_xlabel(name); a2.set_ylabel("weighted Gini of the children"); a2.set_ylim(0, gini(y) + 0.06)
    plt.tight_layout(); plt.show()


def fig_tree_regions(depths=(1, 2, 4, None)):
    from sklearn.tree import DecisionTreeClassifier
    X, y = moons()
    fig, axes = plt.subplots(1, len(depths), figsize=(3.6 * len(depths), 3.4))
    for ax, d in zip(axes, depths):
        m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X, y)
        _regions(ax, m, X, y)
        ax.set_title(f"max_depth = {d}: {m.get_n_leaves()} leaves", fontsize=10)
    plt.tight_layout(); plt.show()


def fig_depth_curve():
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    X, y = moons(n=400, noise=0.35, seed=1)
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.5, random_state=0)
    ds = list(range(1, 16)); tr, te = [], []
    for d in ds:
        m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xa, ya)
        tr.append(m.score(Xa, ya)); te.append(m.score(Xb, yb))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(ds, tr, "o-", color=C1, label="training accuracy"); ax.plot(ds, te, "o-", color=C0, label="test accuracy")
    ax.set_xlabel("max_depth"); ax.set_ylabel("accuracy"); ax.legend(fontsize=8)
    ax.set_title("Deeper trees memorize the training set"); plt.tight_layout(); plt.show()


def fig_regression_tree(depths=(1, 2, 5)):
    from sklearn.tree import DecisionTreeRegressor
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(0, 6, 60)); y = np.sin(x) + 0.25 * rng.normal(size=60)
    xs = np.linspace(0, 6, 500)
    fig, axes = plt.subplots(1, len(depths), figsize=(4 * len(depths), 3.2), sharey=True)
    for ax, d in zip(axes, depths):
        m = DecisionTreeRegressor(max_depth=d).fit(x[:, None], y)
        ax.scatter(x, y, s=12, color=GREY); ax.plot(xs, m.predict(xs[:, None]), color=C1, lw=2)
        ax.set_title(f"max_depth = {d}: {m.get_n_leaves()} flat steps", fontsize=10)
    plt.tight_layout(); plt.show()


# ------------------------------------------------------------------ notebook 05: SVM
X_SVM = np.array([[1, 1], [2, 1], [1, 2], [3, 3], [4, 3], [3, 4]], float)
Y_SVM = np.array([-1, -1, -1, 1, 1, 1])


def _svm_frame(ax, lim=(-0.2, 5.2)):
    ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal"); ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")


def _svm_points(ax, X, y):
    ax.scatter(X[y < 0, 0], X[y < 0, 1], c=C0, s=70, edgecolor="k", label="y = −1", zorder=3)
    ax.scatter(X[y > 0, 0], X[y > 0, 1], c=C1, s=70, edgecolor="k", label="y = +1", zorder=3)


def fig_many_separators():
    fig, ax = plt.subplots(figsize=(5, 4.6))
    _svm_points(ax, X_SVM, Y_SVM)
    for (w, b), ls, lab in [(((1, 0.15), -2.6), ":", "A: grazes the blue points"),
                            (((0.2, 1), -2.7), "--", "B: grazes both classes"),
                            (((1, 1), -5), "-", "C: as far from both as possible")]:
        _line(ax, w, b, xlim=(-0.2, 5.2), color="k", ls=ls, lw=1.8, label=lab)
    _svm_frame(ax); ax.legend(loc="upper left", fontsize=8); ax.set_title("All three separate the data. Which is best?")
    plt.show()


def fig_margin(X=X_SVM, y=Y_SVM, C=1e6, ax=None, title=None):
    from sklearn.svm import SVC
    m = SVC(kernel="linear", C=C).fit(X, y)
    w, b = m.coef_[0], m.intercept_[0]
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(5.2, 4.8))
    for off, ls, lw in [(0, "-", 2), (1, "--", 1), (-1, "--", 1)]:
        _line(ax, w, b - off, xlim=(-0.5, 5.5), color="k", ls=ls, lw=lw)
    _svm_points(ax, X, y)
    sv = m.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=260, facecolors="none", edgecolors="k", linewidths=1.5, label="support vectors")
    ax.text(0.05, 0.03, f"$w$ = ({w[0]:.2f}, {w[1]:.2f}), $b$ = {b:.2f}\nmargin width $2/\\|w\\|$ = {2 / np.linalg.norm(w):.3f}",
            transform=ax.transAxes, fontsize=8, bbox=dict(fc="white", ec=GREY, alpha=0.9))
    _svm_frame(ax, (-0.5, 5.5)); ax.legend(loc="upper left", fontsize=8)
    ax.set_title(title or "The widest street: solid = boundary, dashed = margins")
    if own:
        plt.show()
    return m


def svm_overlap_data(seed=3):
    rng = np.random.default_rng(seed)
    a = rng.normal([1.5, 1.5], 0.8, size=(25, 2)); c = rng.normal([3.5, 3.5], 0.8, size=(25, 2))
    return np.vstack([a, c]), np.r_[-np.ones(25), np.ones(25)]


def fig_soft_margin(Cs=(0.01, 1, 100)):
    X, y = svm_overlap_data()
    fig, axes = plt.subplots(1, len(Cs), figsize=(4.4 * len(Cs), 4.2))
    for ax, C in zip(axes, Cs):
        m = fig_margin(X, y, C=C, ax=ax, title=f"C = {C:g}: {len(m_sv(X, y, C))} support vectors")
        ax.get_legend().remove()
    plt.tight_layout(); plt.show()


def m_sv(X, y, C):
    from sklearn.svm import SVC
    return SVC(kernel="linear", C=C).fit(X, y).support_


def fig_margin_losses():
    m = np.linspace(-2.5, 3, 400)
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    ax.plot(m, (m <= 0).astype(float), color=GREY, lw=1.5, label="0–1 loss (wrong or not)")
    ax.plot(m, np.maximum(0, 1 - m), color="k", lw=2.2, label="hinge: $\\max(0,\\ 1 - yz)$  (SVM)")
    ax.plot(m, np.log2(1 + np.exp(-m)), color=C1, lw=2, label="logistic: $\\log_2(1 + e^{-yz})$")
    ax.axvline(0, color=GREY, lw=0.6); ax.axvline(1, color=GREY, lw=0.6, ls=":")
    ax.text(1.05, 2.4, "hinge is exactly 0\nfrom $yz = 1$ on", fontsize=8)
    ax.set_xlabel("$yz$: positive = right side, bigger = more confident"); ax.set_ylabel("loss")
    ax.set_ylim(0, 3.3); ax.legend(fontsize=8); ax.set_title("Three losses as functions of the signed score $yz$")
    plt.tight_layout(); plt.show()


def fig_lift_1d():
    x = np.array([-3, -2.5, -2, -0.5, 0, 0.5, 1, 2.2, 3])
    y = np.where(np.abs(x) < 1.5, 1, -1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, yy in [(a1, np.zeros_like(x)), (a2, x ** 2)]:
        ax.scatter(x[y < 0], yy[y < 0], c=C0, s=70, edgecolor="k"); ax.scatter(x[y > 0], yy[y > 0], c=C1, s=70, edgecolor="k")
    a1.set_yticks([]); a1.set_xlabel("$x$"); a1.set_title("1-D: no single threshold separates them")
    a2.axhline(2.0, color="k", lw=2); a2.set_xlabel("$x$"); a2.set_ylabel("$x^2$")
    a2.set_title("Add the feature $x^2$: now a straight line does")
    plt.tight_layout(); plt.show()


def fig_kernels():
    from sklearn.datasets import make_circles
    from sklearn.svm import SVC
    X, y = make_circles(n_samples=200, noise=0.12, factor=0.4, random_state=0)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (name, m) in zip(axes, [("linear", SVC(kernel="linear")), ("polynomial, degree 2", SVC(kernel="poly", degree=2, coef0=1)),
                                    ("RBF", SVC(kernel="rbf", gamma=1))]):
        m.fit(X, y); _regions(ax, m, X, y)
        ax.set_title(f"{name}: accuracy {m.score(X, y):.2f}", fontsize=10)
    plt.tight_layout(); plt.show()


def fig_rbf_gamma(gammas=(0.1, 1, 30)):
    from sklearn.svm import SVC
    X, y = moons()
    fig, axes = plt.subplots(1, len(gammas), figsize=(4 * len(gammas), 3.6))
    for ax, g in zip(axes, gammas):
        m = SVC(kernel="rbf", gamma=g, C=1).fit(X, y); _regions(ax, m, X, y)
        ax.set_title(f"gamma = {g:g}: training acc. {m.score(X, y):.2f}", fontsize=10)
    plt.tight_layout(); plt.show()


# ------------------------------------------------------------------ notebook 06: ensembles
def fig_variance_of_average(rhos=(0.0, 0.3, 0.7)):
    B = np.arange(1, 101)
    fig, ax = plt.subplots(figsize=(6, 3.7))
    for r in rhos:
        ax.plot(B, r + (1 - r) / B, lw=2, label=f"$\\rho$ = {r:g}: levels off at {r:g}")
    ax.set_xscale("log"); ax.set_xlabel("number of models $B$ averaged"); ax.set_ylabel("variance of the average ($\\sigma^2 = 1$)")
    ax.legend(fontsize=8); ax.set_title("Averaging only cancels what isn't shared"); plt.tight_layout(); plt.show()


def fig_bagging():
    from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier
    X, y = moons(n=300, noise=0.35, seed=2)
    models = [("one deep tree", DecisionTreeClassifier(random_state=0)),
              ("bagging: 200 deep trees", BaggingClassifier(DecisionTreeClassifier(), n_estimators=200, random_state=0)),
              ("random forest: 200 trees", RandomForestClassifier(n_estimators=200, random_state=0))]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    for ax, (name, m) in zip(axes, models):
        m.fit(X, y); _regions(ax, m, X, y); ax.set_title(name, fontsize=10)
    plt.tight_layout(); plt.show()


def fig_adaboost_rounds(x, y, rounds=3):
    """1-D AdaBoost with stumps, drawn round by round. Dot size = sample weight before the round."""
    x = np.asarray(x, float); y = np.asarray(y, float); n = len(x)
    wts = np.full(n, 1 / n)
    fig, axes = plt.subplots(rounds, 1, figsize=(7.5, 1.7 * rounds), sharex=True)
    for r, ax in enumerate(np.atleast_1d(axes)):
        t, s, err = best_stump(x, y, wts)
        alpha = 0.5 * np.log((1 - err) / err)
        h = np.where(x > t, s, -s)
        ax.scatter(x, np.zeros(n), s=2500 * wts, c=np.where(y > 0, C1, C0), edgecolor="k", zorder=3)
        for xi, wi, ok in zip(x, wts, h == y):
            ax.annotate(f"{wi:.3f}" + ("" if ok else " ✗"), (xi, 0), xytext=(0, -26), textcoords="offset points", ha="center", fontsize=7)
        ax.axvline(t, color="k", lw=2)
        ax.set_yticks([]); ax.set_ylim(-1, 1)
        ax.set_title(f"round {r + 1}: stump at x = {t:g} (predicts {'+1' if s > 0 else '−1'} on the right), weighted error ε = {err:.3f}, α = {alpha:.3f}", fontsize=9)
        wts = wts * np.exp(-alpha * y * h); wts = wts / wts.sum()
    plt.tight_layout(); plt.show()


def best_stump(x, y, wts):
    """The threshold t and sign s that minimize the weighted error of: predict s if x > t, else -s."""
    xs = np.unique(x); best = (None, 1, 2.0)
    for t in (xs[:-1] + xs[1:]) / 2:
        for s in (1, -1):
            err = wts[np.where(x > t, s, -s) != y].sum()
            if err < best[2] - 1e-12:
                best = (t, s, err)
    return best


def fig_gb_rounds(show=(1, 2, 5, 50), lr=0.5):
    from sklearn.tree import DecisionTreeRegressor
    rng = np.random.default_rng(1)
    x = np.sort(rng.uniform(0, 6, 50)); y = np.sin(x) + 0.2 * rng.normal(size=50)
    xs = np.linspace(0, 6, 400)
    F = np.full_like(y, y.mean()); Fs = np.full_like(xs, y.mean()); snaps = {}
    for m in range(1, max(show) + 1):
        r = y - F
        t = DecisionTreeRegressor(max_depth=1).fit(x[:, None], r)
        F = F + lr * t.predict(x[:, None]); Fs = Fs + lr * t.predict(xs[:, None])
        snaps[m] = Fs.copy()
    fig, axes = plt.subplots(1, len(show), figsize=(3.6 * len(show), 3.1), sharey=True)
    for ax, m in zip(axes, show):
        ax.scatter(x, y, s=10, color=GREY); ax.plot(xs, snaps[m], color=C1, lw=2)
        ax.set_title(f"after {m} round{'s' if m > 1 else ''}", fontsize=10)
    plt.tight_layout(); plt.show()


def fig_compare_models():
    from sklearn.datasets import make_circles
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    rng = np.random.default_rng(4)
    Xl = np.vstack([rng.normal(-1, 0.9, size=(100, 2)), rng.normal(1, 0.9, size=(100, 2))]); yl = np.r_[np.zeros(100), np.ones(100)].astype(int)
    datasets = [("moons", moons()), ("circles", make_circles(n_samples=200, noise=0.12, factor=0.45, random_state=0)), ("linear", (Xl, yl))]
    models = [("logistic", LogisticRegression()), ("kNN, k=5", KNeighborsClassifier(5)), ("tree, depth 4", DecisionTreeClassifier(max_depth=4, random_state=0)),
              ("SVM, RBF", SVC(gamma=1)), ("random forest", RandomForestClassifier(n_estimators=100, random_state=0)),
              ("gradient boosting", GradientBoostingClassifier(random_state=0))]
    fig, axes = plt.subplots(len(datasets), len(models), figsize=(2.6 * len(models), 2.5 * len(datasets)))
    for i, (dname, (X, y)) in enumerate(datasets):
        for j, (mname, m) in enumerate(models):
            pipe = make_pipeline(StandardScaler(), m).fit(X, y)
            _regions(axes[i, j], pipe, X, y)
            if i == 0:
                axes[i, j].set_title(mname, fontsize=9)
            if j == 0:
                axes[i, j].set_ylabel(dname)
    plt.tight_layout(); plt.show()


# ------------------------------------------------------------------ table pictures (pandas, NumPy mechanics)
KEY = {"A": "#cfe2f7", "B": "#fde2c8", "C": "#e4e4e4", "D": "#e6d8f2"}
HL, NAN_BG, HDR = "#fff1a8", "#f8d0d0", "#e9edf2"


def _table(ax, x0, y0, header, rows, cw=0.95, ch=0.42, colors=None, fs=9, index=None, title=None, tfs=10, bold_hdr=True):
    """Draw a table whose top-left corner is (x0, y0); y grows downward. colors: {(r, c): facecolor}. Returns (width, height)."""
    colors = colors or {}
    cws = list(cw) if isinstance(cw, (list, tuple)) else [cw] * len(header)
    xs = [x0]
    for w in cws:
        xs.append(xs[-1] + w)
    if title:
        ax.text(x0, y0 + 0.18, title, fontsize=tfs, fontweight="bold", va="bottom", family="monospace")
    for c, h in enumerate(header):
        ax.add_patch(plt.Rectangle((xs[c], y0 - ch), cws[c], ch, fc=HDR, ec="#9aa3ad", lw=0.8))
        ax.text(xs[c] + cws[c] / 2, y0 - ch / 2, h, ha="center", va="center", fontsize=fs, fontweight="bold" if bold_hdr else "normal")
    for r, row in enumerate(rows):
        yy = y0 - (r + 2) * ch
        for c, v in enumerate(row):
            v = str(v)
            fc = colors.get((r, c), NAN_BG if v == "NaN" else "white")
            ax.add_patch(plt.Rectangle((xs[c], yy), cws[c], ch, fc=fc, ec="#9aa3ad", lw=0.8))
            ax.text(xs[c] + cws[c] / 2, yy + ch / 2, v, ha="center", va="center", fontsize=fs, color="#b00020" if v == "NaN" else "k")
        if index is not None:
            ax.text(x0 - 0.12, yy + ch / 2, str(index[r]), ha="right", va="center", fontsize=fs - 1, color=GREY)
    return xs[-1] - x0, (len(rows) + 1) * ch


def _canvas(w, h, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, w); ax.set_ylim(-h, 0.6); ax.axis("off")
    return fig, ax


def _arrow(ax, a, b, text=None, color="k", rad=0.0, fs=9, lw=1.4):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.12, text, ha="center", va="bottom", fontsize=fs, color=color, fontweight="bold")


PARTS6 = [["P01", "A", "day", "71.2"], ["P02", "A", "night", "74.8"], ["P03", "B", "day", "69.5"],
          ["P04", "C", "night", "NaN"], ["P05", "B", "night", "77.1"], ["P06", "A", "day", "70.4"]]


def fig_select():
    """Five ways to pick from the first 6 rows of `parts`; the yellow cells are what you get back."""
    panels = [
        ('parts["temp"]', {(r, 3) for r in range(6)}, "ONE name → a Series (a single column)\nshape (6,)"),
        ('parts[["part", "temp"]]', {(r, c) for r in range(6) for c in (0, 3)}, "a LIST of names → a DataFrame\nshape (6, 2)"),
        ('parts.loc[0:2, ["part", "temp"]]', {(r, c) for r in range(3) for c in (0, 3)}, "loc = by LABEL. 0:2 means labels 0, 1, 2\n(the end label IS included) → shape (3, 2)"),
        ("parts.iloc[0:2, 1:3]", {(r, c) for r in range(2) for c in (1, 2)}, "iloc = by POSITION, like NumPy. rows 0, 1 and\ncolumns 1, 2 (stops BEFORE 2 and 3) → shape (2, 2)"),
        ('parts[(parts["shift"] == "night") & (parts["temp"] > 74)]', {(r, c) for r in (1, 4) for c in range(4)},
         "a True/False mask keeps WHOLE rows where both\nconditions are True: P02 and P05 (P04's NaN > 74 is False)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.6))
    for ax, (code, cells, note) in zip(axes.ravel(), panels):
        ax.set_xlim(-0.5, 4.6); ax.set_ylim(-3.9, 0.9); ax.axis("off")
        _table(ax, 0.3, 0, ["part", "line", "shift", "temp"], PARTS6, cw=1.0, colors={k: HL for k in cells}, index=range(6))
        for c in range(4):
            ax.text(0.3 + c + 0.5, 0.08, f"col {c}", ha="center", va="bottom", fontsize=7, color=GREY)
        ax.text(-0.25, 0.08, "label", ha="center", va="bottom", fontsize=7, color=GREY)
        ax.set_title(code, fontsize=9.5, family="monospace", fontweight="bold", loc="left")
        ax.text(0.3, -3.1, note, fontsize=9, va="top")
    ax = axes.ravel()[5]; ax.axis("off")
    ax.text(0.02, 0.85, "How to read these pictures", fontsize=11, fontweight="bold", transform=ax.transAxes)
    ax.text(0.02, 0.78, "• Each panel is the same table: the first 6 rows of parts.\n• Grey numbers on the left are the row LABELS (the index).\n"
            "• Grey 'col 0 … col 3' are column POSITIONS.\n• Yellow cells = what the line of code returns.\n\n"
            "loc vs iloc: here labels and positions happen\nto be equal. After sort_values they are not:\nloc follows the labels, iloc the positions.",
            fontsize=9.5, va="top", transform=ax.transAxes)
    plt.tight_layout(); plt.show()


PARTS12 = [["P01", "A", "71.2"], ["P02", "A", "74.8"], ["P03", "B", "69.5"], ["P04", "C", "NaN"], ["P05", "B", "77.1"], ["P06", "A", "70.4"],
           ["P07", "C", "72.9"], ["P08", "B", "79.3"], ["P09", "C", "68.8"], ["P10", "A", "75.6"], ["P11", "B", "70.1"], ["P12", "C", "73.4"]]


def _key_colors(rows, col):
    return {(r, c): KEY[row[col]] for r, row in enumerate(rows) for c in range(len(row))}


def fig_groupby():
    """groupby("line")["temp"].mean() in three stages: split the rows, average each group, put the answers together."""
    fig, ax = _canvas(15.5, 8.6, (15.5, 8.6))
    _table(ax, 0.4, 0, ["part", "line", "temp"], PARTS12, cw=0.95, colors=_key_colors(PARTS12, 1), title="parts (12 rows)")
    groups = {g: [r for r in PARTS12 if r[1] == g] for g in "ABC"}
    calc = {"A": "(71.2 + 74.8 + 70.4 + 75.6) / 4 = 73.0", "B": "(69.5 + 77.1 + 79.3 + 70.1) / 4 = 74.0",
            "C": "(72.9 + 68.8 + 73.4) / 3 = 71.7\nNaN is skipped: 3 values, not 4"}
    y = 0
    for g in "ABC":
        _table(ax, 5.0, y, ["part", "line", "temp"], groups[g], cw=0.95, colors=_key_colors(groups[g], 1), title=f"group {g}")
        ax.text(8.1, y - 1.25, f"mean of temp:\n{calc[g]}", fontsize=9, va="center")
        y -= 2.9
    _table(ax, 12.6, -2.6, ["line", "temp"], [["A", "73.0"], ["B", "74.0"], ["C", "71.7"]], cw=1.0,
           colors={(r, c): KEY["ABC"[r]] for r in range(3) for c in range(2)}, title="result: one row per group")
    _arrow(ax, (3.4, -2.6), (4.8, -1.0), "1. SPLIT", rad=-0.15)
    _arrow(ax, (3.4, -2.6), (4.8, -3.9), rad=0.0)
    _arrow(ax, (3.4, -2.6), (4.8, -6.8), rad=0.15)
    ax.text(8.1, 0.45, "2. APPLY .mean() to each group", fontsize=10, fontweight="bold")
    for yy in (-1.25, -4.15, -7.05):
        _arrow(ax, (11.4, yy), (12.5, -3.4), rad=0.0, lw=1.0)
    ax.text(12.6, -0.95, "3. COMBINE", fontsize=10, fontweight="bold")
    plt.show()


def fig_transform():
    """.mean() gives one row per group; .transform("mean") writes each group's mean back onto every row."""
    fig, ax = _canvas(12.5, 6.0, (12.5, 6.2))
    means = {"A": "73.0", "B": "74.0", "C": "71.7"}
    rows = [r + [means[r[1]]] for r in PARTS12]
    cols = _key_colors(rows, 1)
    _table(ax, 0.4, 0, ["part", "line", "temp", "transform(\"mean\")"], rows, cw=[0.95, 0.95, 0.95, 1.9], colors=cols,
           title='parts.groupby("line")["temp"].transform("mean")')
    _table(ax, 7.6, -1.6, ["line", "mean()"], [["A", "73.0"], ["B", "74.0"], ["C", "71.7"]], cw=1.0,
           colors={(r, c): KEY["ABC"[r]] for r in range(3) for c in range(2)}, title='...["temp"].mean()')
    ax.text(7.6, -3.6, "mean():  3 rows, one per group.\n\ntransform(\"mean\"):  12 rows, the same as parts.\nEvery row gets ITS OWN group's mean,\n"
            "so it can sit next to the original column.\n\nThat's what fillna needs: P04 (line C)\nis missing, so it gets line C's 71.7.", fontsize=9.5, va="top")
    plt.show()


MERGE_LEFT = [["P01", "A"], ["P02", "A"], ["P03", "B"], ["P04", "C"]]
MERGE_RIGHT = [["A", "Kim"], ["B", "Lee"], ["D", "Park"]]


def fig_merge():
    """merge(on="line"): every row of the left table looks up its line in the right table."""
    fig, ax = _canvas(15.5, 7.6, (15.5, 7.9))
    lc = {(r, c): KEY[row[1]] for r, row in enumerate(MERGE_LEFT) for c in range(2)}
    rc = {(r, c): KEY[row[0]] for r, row in enumerate(MERGE_RIGHT) for c in range(2)}
    _table(ax, 1.0, 0, ["part", "line"], MERGE_LEFT, cw=1.0, colors=lc, title="left table: small")
    _table(ax, 7.0, 0, ["line", "supervisor"], MERGE_RIGHT, cw=[1.0, 1.4], colors=rc, title="right table: lines")
    ch = 0.42
    yrow = lambda r: -(r + 1.5) * ch
    for r, (_, key) in enumerate(MERGE_LEFT):
        match = [i for i, row in enumerate(MERGE_RIGHT) if row[0] == key]
        if match:
            _arrow(ax, (3.05, yrow(r)), (6.95, yrow(match[0])), color="#2e7d32", lw=1.2)
        else:
            ax.text(3.2, yrow(r), "✗ no line C on the right", color="#b00020", fontsize=9, va="center")
    ax.text(9.6, yrow(2), "← no part has line D:\n    nobody looks it up", color="#6a3d9a", fontsize=9, va="center")
    ax.text(1.0, -2.75, 'Each row of the LEFT table takes its "line" and looks it up in the RIGHT table (green arrows).\n'
            "A match glues the two rows together. The question is what to do with rows that find no partner.", fontsize=10, va="top")
    results = [
        ('how="inner"', [["P01", "A", "Kim"], ["P02", "A", "Kim"], ["P03", "B", "Lee"]],
         "keep only rows that found a partner:\nP04 (line C) is dropped,\nPark (line D) never appears."),
        ('how="left"', [["P01", "A", "Kim"], ["P02", "A", "Kim"], ["P03", "B", "Lee"], ["P04", "C", "NaN"]],
         "keep EVERY left row:\nP04 stays, with supervisor NaN.\nPark still doesn't appear."),
        ('how="outer"', [["P01", "A", "Kim"], ["P02", "A", "Kim"], ["P03", "B", "Lee"], ["P04", "C", "NaN"], ["NaN", "D", "Park"]],
         "keep every row of BOTH tables:\nNaN wherever a side is missing."),
    ]
    for k, (name, rows, note) in enumerate(results):
        x = 0.4 + 5.2 * k
        cols = {(r, c): KEY[row[1]] for r, row in enumerate(rows) for c in range(3) if row[c] != "NaN"}
        _table(ax, x, -4.4, ["part", "line", "supervisor"], rows, cw=[1.0, 0.9, 1.4], colors=cols, title=f"small.merge(lines, on=\"line\", {name})", tfs=8.5)
        ax.text(x, -4.4 - (len(rows) + 1) * ch - 0.25, note, fontsize=9, va="top")
    plt.show()


def fig_pivot():
    """pivot_table: the (line, shift) pairs of a long list become the rows and columns of a grid."""
    fig, ax = _canvas(12.0, 4.4, (12, 4.6))
    long_rows = [["A", "day", "0"], ["A", "night", "2"], ["B", "day", "0"], ["B", "night", "2"], ["C", "day", "0"], ["C", "night", "0"]]
    sc = {"day": "#d7e6f6", "night": "#fbe3cf"}
    _table(ax, 0.4, 0, ["line", "shift", "defects"], long_rows, cw=1.0, colors={(r, c): sc[row[1]] for r, row in enumerate(long_rows) for c in range(3)},
           title="long: one row per (line, shift) pair")
    grid = [["A", "0", "2"], ["B", "0", "2"], ["C", "0", "0"]]
    gc = {(r, 1): sc["day"] for r in range(3)} | {(r, 2): sc["night"] for r in range(3)}
    _table(ax, 6.4, -0.6, ["line", "day", "night"], grid, cw=1.0, colors=gc, title="grid: lines down, shifts across")
    _arrow(ax, (3.7, -1.6), (6.2, -1.6), "pivot_table", rad=0.0)
    ax.text(6.4, -2.8, 'index="line"  → the rows\ncolumns="shift" → the columns\nvalues="defect", aggfunc="sum" → the number in each cell\n'
            "e.g. the long row (A, night, 2) becomes the cell row A, column night", fontsize=9, va="top")
    plt.show()


# ------------------------------------------------------------------ notebook 00: frozen gradient
def fig_frozen_gradient():
    x = np.array([1.0, 2.0, 3.0]); y = 2 * x; eta = 0.05
    L = lambda w: np.mean((w * x - y) ** 2)
    g = lambda w: 2 / 3 * np.sum(x * (w * x - y))
    good, bad = [0.0], [0.0]
    for _ in range(6):
        good.append(good[-1] - eta * g(good[-1])); bad.append(bad[-1] - eta * g(0.0))
    ws = np.linspace(-0.5, 6.2, 300)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, path, title, col in [(axes[0], good, "gradient INSIDE the loop: the slope is re-measured at every step", "#2e7d32"),
                                 (axes[1], bad, "gradient computed ONCE: the slope at w = 0 is reused forever", "#b00020")]:
        ax.plot(ws, [L(w) for w in ws], color=GREY, lw=1.5)
        ax.axvline(2, color=GREY, ls=":", lw=1); ax.text(2.05, 70, "best w = 2", fontsize=8, color=GREY)
        for k in range(len(path) - 1):
            ax.annotate("", xy=(path[k + 1], L(path[k + 1])), xytext=(path[k], L(path[k])), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.3))
        ax.plot(path, [L(w) for w in path], "o", color=col, ms=5)
        for k, w in enumerate(path):
            ax.annotate(str(k), (w, L(w)), xytext=(0, 7), textcoords="offset points", fontsize=7, ha="center")
        ax.set_title(title, fontsize=9.5); ax.set_xlabel("$w$"); ax.set_ylim(-2, 80)
    axes[0].set_ylabel("loss $L(w)$")
    plt.tight_layout(); plt.show()


# ------------------------------------------------------------------ notebook 02: kNN mechanics
def fig_broadcast_distances():
    X = KNN_TRAIN; q = np.array([3.5, 3.0]); D = X - q; S = D ** 2; s = S.sum(axis=1); d = np.sqrt(s)
    fmt = lambda a: [[f"{v:g}" for v in row] for row in a]
    fig, ax = _canvas(18.6, 4.0, (18.6, 4.3))
    row0 = lambda ncol: {(0, c): HL for c in range(ncol)}
    blocks = [(0.2, "X_train (6, 2)", ["x1", "x2"], fmt(X), row0(2)),
              (2.9, "q copied down (6, 2)", ["x1", "x2"], fmt(np.tile(q, (6, 1))), {**{(r, c): "#f1f1f1" for r in range(1, 6) for c in range(2)}, **row0(2)}),
              (5.6, "difference (6, 2)", ["x1", "x2"], fmt(D), row0(2)),
              (8.3, "squared (6, 2)", ["x1", "x2"], fmt(S), row0(2)),
              (11.0, ".sum(axis=1) (6,)", ["sum"], [[f"{v:g}"] for v in s], row0(1)),
              (13.1, "np.sqrt (6,)", ["dist"], [[f"{v:.3f}"] for v in d], row0(1))]
    for x0, title, hdr, rows, cols in blocks:
        _table(ax, x0, 0, hdr, rows, cw=0.95 if len(hdr) == 2 else 1.1, colors=cols, title=title, tfs=9)
    for xa, sym in [(2.3, "−"), (5.0, "="), (7.75, "**2"), (10.45, "→"), (12.55, "→")]:
        ax.text(xa + 0.15, -1.5, sym, fontsize=15, ha="center", va="center", fontweight="bold")
    ax.text(15.0, -0.4, "Yellow = row 0, followed\nthrough every step:\n\n(1, 2) − (3.5, 3) = (−2.5, −1)\n"
            "squares: 6.25 and 1\nsum: 7.25\nsqrt: 2.693", fontsize=9, va="top")
    ax.text(0.2, -3.35, "q has shape (2,) and X_train has (6, 2). Lined up from the right, 2 matches 2, so NumPy reuses q for every row\n"
            "(grey: copies that are never actually made in memory). sum(axis=1) adds ACROSS each row, so the column axis disappears.",
            fontsize=9, va="top")
    plt.show()


def fig_argsort(dist=(2.693, 2.5, 1.5, 1.803, 2.062, 3.905), k=3):
    dist = np.array(dist); order = np.argsort(dist)
    fig, ax = _canvas(12.5, 3.1, (12.5, 3.3))
    cw, ch = 1.25, 0.5
    for p, v in enumerate(dist):
        ax.add_patch(plt.Rectangle((1.6 + p * cw, -ch), cw, ch, fc="white", ec="#9aa3ad"))
        ax.text(1.6 + p * cw + cw / 2, -ch / 2, f"{v:g}", ha="center", va="center", fontsize=10)
        ax.text(1.6 + p * cw + cw / 2, 0.1, f"position {p}", ha="center", va="bottom", fontsize=8, color=GREY)
    ax.text(0.1, -ch / 2, "dist", fontsize=10, fontweight="bold", va="center", family="monospace")
    y2 = -2.4
    for rank, p in enumerate(order):
        fc = HL if rank < k else "white"
        ax.add_patch(plt.Rectangle((1.6 + rank * cw, y2), cw, ch, fc=fc, ec="#9aa3ad"))
        ax.text(1.6 + rank * cw + cw / 2, y2 + ch / 2, str(p), ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(1.6 + rank * cw + cw / 2, y2 - 0.08, f"({dist[p]:g})", ha="center", va="top", fontsize=8, color=GREY)
        _arrow(ax, (1.6 + p * cw + cw / 2, -ch - 0.05), (1.6 + rank * cw + cw / 2, y2 + ch + 0.05), color="#7a8794", lw=0.9)
    ax.text(0.1, y2 + ch / 2, "argsort", fontsize=10, fontweight="bold", va="center", family="monospace")
    ax.text(1.6 + 6 * cw + 0.3, y2 + ch / 2, f"← the POSITIONS, smallest distance first.\n   Yellow = [:{k}] → the {k} nearest: rows {order[:k].tolist()}",
            fontsize=9, va="center")
    plt.show()


def fig_fancy_index(y=(0, 0, 0, 1, 1, 1), idx=(2, 3, 4)):
    fig, ax = _canvas(11.0, 3.4, (11, 3.6))
    cw, ch = 1.0, 0.5
    for p, v in enumerate(y):
        ax.add_patch(plt.Rectangle((2.0 + p * cw, -ch), cw, ch, fc=HL if p in idx else "white", ec="#9aa3ad"))
        ax.text(2.0 + p * cw + cw / 2, -ch / 2, str(v), ha="center", va="center", fontsize=11)
        ax.text(2.0 + p * cw + cw / 2, 0.1, f"pos {p}", ha="center", va="bottom", fontsize=8, color=GREY)
    ax.text(0.1, -ch / 2, "y_train", fontsize=10, fontweight="bold", va="center", family="monospace")
    y2 = -2.4
    for j, p in enumerate(idx):
        ax.add_patch(plt.Rectangle((3.0 + j * cw, y2), cw, ch, fc=HL, ec="#9aa3ad"))
        ax.text(3.0 + j * cw + cw / 2, y2 + ch / 2, str(y[p]), ha="center", va="center", fontsize=11, fontweight="bold")
        ax.text(3.0 + j * cw + cw / 2, y2 - 0.08, f"y_train[{p}]", ha="center", va="top", fontsize=8, color=GREY)
        _arrow(ax, (2.0 + p * cw + cw / 2, -ch - 0.05), (3.0 + j * cw + cw / 2, y2 + ch + 0.05), color="#7a8794", lw=0.9)
    ax.text(0.1, y2 + ch / 2, f"y_train[{list(idx)}]", fontsize=9, fontweight="bold", va="center", family="monospace")
    ax.text(6.4, y2 + ch / 2, "← one label per neighbor, in the order of idx\n    shape (3,)", fontsize=9, va="center")
    plt.show()


def fig_bincount(labels=(0, 1, 1)):
    labels = np.array(labels); counts = np.bincount(labels)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [1, 1.2]})
    a1.axis("off"); a1.set_xlim(0, 4); a1.set_ylim(-1.5, 1)
    for j, v in enumerate(labels):
        a1.add_patch(plt.Rectangle((0.5 + j, -0.25), 1, 0.5, fc=C1 if v == 1 else C0, ec="k", alpha=0.75))
        a1.text(1.0 + j, 0, str(v), ha="center", va="center", fontsize=12, color="white", fontweight="bold")
    a1.set_title("labels of the 3 nearest", fontsize=10, family="monospace")
    a2.bar(range(len(counts)), counts, color=[C0, C1][:len(counts)], edgecolor="k")
    for c, n in enumerate(counts):
        a2.text(c, n + 0.05, f"count of {c}s = {n}", ha="center", va="bottom", fontsize=9)
    a2.set_xticks(range(len(counts))); a2.set_xticklabels([f"position {c}\n(label {c})" for c in range(len(counts))])
    a2.set_ylim(0, counts.max() + 0.8); a2.set_yticks([])
    a2.set_title(f"np.bincount → {counts.tolist()}   .argmax() → {counts.argmax()} (the tallest bar)", fontsize=10, family="monospace")
    plt.tight_layout(); plt.show()


# ------------------------------------------------------------------ notebook 04: the small tree, drawn by scikit-learn
def fig_small_tree():
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    X = np.column_stack([[68, 70, 71, 72, 73, 74, 75, 77, 78, 80], [0, 0, 1, 0, 1, 0, 1, 1, 0, 1]])
    y = np.array([0, 0, 0, 0, 1, 0, 1, 1, 0, 1])
    m = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_tree(m, feature_names=["temp", "night"], class_names=["OK", "defect"], filled=True, impurity=True, rounded=True, fontsize=10, ax=ax)
    ax.set_title("Read from the top. Each box: the question, its Gini, how many parts reach it,\n[how many OK, how many defective], and the majority class. Left (True) = yes, right (False) = no.\n\"night <= 0.5\" is True for day-shift parts (night = 0).", fontsize=10)
    plt.show()


# ------------------------------------------------------------------ notebook 06: bootstrap samples
def fig_bootstrap(n=10, B=5, seed=3):
    rng = np.random.default_rng(seed)
    counts = np.array([np.bincount(rng.integers(0, n, size=n), minlength=n) for _ in range(B)])
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.imshow(np.where(counts == 0, np.nan, counts), cmap="Oranges", vmin=0, vmax=4, aspect="auto")
    for b in range(B):
        for j in range(n):
            c = counts[b, j]
            ax.text(j, b, "out" if c == 0 else f"×{c}", ha="center", va="center", fontsize=9, color=GREY if c == 0 else "k")
        ax.text(n - 0.3, b, f"   {np.mean(counts[b] > 0):.0%} of the rows used", va="center", fontsize=9)
    ax.set_xticks(range(n)); ax.set_xticklabels([f"row {j}" for j in range(n)], fontsize=8)
    ax.set_yticks(range(B)); ax.set_yticklabels([f"sample {b + 1}" for b in range(B)])
    ax.set_xticks(np.arange(-0.5, n), minor=True); ax.set_yticks(np.arange(-0.5, B), minor=True)
    ax.grid(which="minor", color="white", lw=2); ax.grid(which="major", visible=False); ax.tick_params(which="minor", length=0)
    ax.set_xlim(-0.5, n + 2.2)
    ax.set_title("5 bootstrap samples of 10 rows: ×2 = drawn twice, out = never drawn (out-of-bag for that tree)", fontsize=10)
    plt.tight_layout(); plt.show()
