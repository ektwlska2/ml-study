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
