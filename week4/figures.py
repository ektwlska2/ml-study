"""Picture helpers for Week 4. Each function draws one figure from the lessons.

You don't need to read this file to follow the lessons. Shared tools (colours, table pictures, flow maps) are in fig_common.py.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from fig_common import *  # noqa: F401,F403
from fig_common import _arrow, _canvas, _regions, _table  # noqa: F401


# ------------------------------------------------------------------ 00 · the week, NumPy pictures, SVM pictures
def fig_week_map():
    flow({"sup": (12, 80, "Weeks 2–3", "supervised: X and labels y", "grey"),
          "pca": (40, 80, "02 PCA", "fewer features", "blue"),
          "km": (40, 52, "03 k-means", "groups, k given", "orange"),
          "more": (40, 24, "04 hierarchical,", "DBSCAN, GMM", "orange"),
          "proj": (66, 66, "05 mini-project", "everything, end to end", "green"),
          "mock": (66, 32, "06 mock exam", "90 min, on paper", "purple"),
          "dl": (88, 49, "Week 5", "neural networks", "grey")},
         [("sup", "pca", "drop the labels"), ("sup", "km", ""), ("km", "more", "k-means' limits"), ("pca", "proj", ""), ("more", "proj", ""),
          ("proj", "mock", ""), ("mock", "dl", "")],
         size=(12, 5.2), title="Week 4: learning without labels (unsupervised), then the ML checkpoint",
         notes=[(2, 12, "01 rebuilds AdaBoost and gradient boosting in small pieces first (your request from Week 3).")])


def fig_eye_index(y=(2, 0, 1), k=3):
    E = np.eye(k).astype(int)
    fig, ax = _canvas(13.5, 3.6, (13.5, 3.7))
    rows = [[str(v) for v in r] for r in E]
    rc = {(r, c): CLUSTER[r] if False else ["#d7e6f6", "#fde2c8", "#dcefe3"][r] for r in range(k) for c in range(k)}
    _table(ax, 0.9, 0, [f"col {c}" for c in range(k)], rows, cw=0.9, colors=rc, index=[f"row {r}" for r in range(k)], title=f"np.eye({k})")
    ax.text(4.0, 0.35, f"y = {list(y)}  (one label per sample)", fontsize=10.5, family="monospace", fontweight="bold")
    out = [[str(v) for v in E[c]] for c in y]
    oc = {(i, c): ["#d7e6f6", "#fde2c8", "#dcefe3"][y[i]] for i in range(len(y)) for c in range(k)}
    _table(ax, 8.6, 0, [f"col {c}" for c in range(k)], out, cw=0.9, colors=oc, index=[f"sample {i}" for i in range(len(y))],
           title=f"np.eye({k})[y]   shape ({len(y)}, {k})")
    for i, c in enumerate(y):
        _arrow(ax, (3.6, -(c + 1.5) * 0.42), (7.0, -(i + 1.5) * 0.42), color="#7a8794", lw=1.0)
    ax.text(0.2, -2.55, "y is a list of row numbers. Indexing with it picks those rows of the identity matrix, in that order:\n"
            "sample 0 has label 2 → row 2 = [0 0 1];  sample 1 has label 0 → row 0 = [1 0 0];  sample 2 has label 1 → row 1 = [0 1 0].",
            fontsize=9.5, va="top")
    plt.show()


def fig_keepdims():
    Z = np.array([[1, 2, 3], [2, 4, 6]])
    fig, ax = _canvas(16.5, 3.9, (16.5, 4.0))
    _table(ax, 0.3, 0, ["c0", "c1", "c2"], [[str(v) for v in r] for r in Z], cw=0.8, title="Z  (2, 3)")
    _table(ax, 3.5, 0, ["max"], [["3"], ["6"]], cw=1.0, colors={(0, 0): HL, (1, 0): HL}, title="max(axis=1, keepdims=True)  (2, 1)", tfs=8.5)
    ax.text(3.5, -1.85, "still a COLUMN:\none value per row", fontsize=8.5, va="top")
    _table(ax, 8.1, 0, ["c0", "c1", "c2"], [["3", "3", "3"], ["6", "6", "6"]], cw=0.8, colors={(r, c): "#f3f3f3" for r in range(2) for c in range(3)},
           title="stretched to (2, 3)", tfs=9)
    _table(ax, 11.9, 0, ["c0", "c1", "c2"], [["-2", "-1", "0"], ["-4", "-2", "0"]], cw=0.8, title="Z - max  (2, 3)")
    ax.text(2.85, -0.7, "→", fontsize=16); ax.text(7.25, -0.7, "→", fontsize=16); ax.text(11.0, -0.7, "→", fontsize=16)
    ax.text(0.3, -2.5, "Without keepdims, Z.max(axis=1) is [3, 6] with shape (2,). Lined up from the right against (2, 3), that's 2 vs 3: ERROR.\n"
            "With keepdims=True the result keeps a length-1 column axis, (2, 1). Lined up: 1 vs 3 → the 1 stretches, 2 vs 2 ✓.\n"
            "So each row has its OWN maximum subtracted.", fontsize=9.5, va="top")
    plt.show()


def fig_svm_distance():
    w = np.array([1.0, 1.0]); b = -4.5
    x0 = np.array([3.0, 1.5]); x = np.array([1.0, 1.0]); u = w / np.linalg.norm(w)
    p = u @ (x - x0)                                   # signed length of the projection = -1.768
    foot = x - p * u                                   # closest point of the line to x
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    xs = np.linspace(-0.5, 5, 50); ax.plot(xs, 4.5 - xs, color="k", lw=2, label="the line $w^\\top x + b = 0$")
    ax.annotate("", xy=x0 + 1.4 * u, xytext=x0, arrowprops=dict(arrowstyle="-|>", color=C2, lw=2.2))
    ax.text(*(x0 + 1.45 * u + [0.05, 0.0]), "direction of $w$\n(unit length)", color=C2, fontsize=9)
    ax.annotate("", xy=x, xytext=x0, arrowprops=dict(arrowstyle="-|>", color=C1, lw=2))
    ax.text(*((x + x0) / 2 + [-0.15, -0.45]), "$x - x_0$", color=C1, fontsize=11)
    ax.plot([x0[0], x0[0] + p * u[0]], [x0[1], x0[1] + p * u[1]], color=C0, lw=4, alpha=0.7, label=f"its shadow on the $w$ direction: length {p:.3f}")
    ax.plot([x0[0] + p * u[0], x[0]], [x0[1] + p * u[1], x[1]], color=GREY, ls=":", lw=1.2)
    ax.plot([x[0], foot[0]], [x[1], foot[1]], color=C0, lw=1.5, ls="--", label=f"the distance itself: {abs(p):.3f}")
    for pt, name, off in [(x0, "$x_0$ (any point on the line)", (0.12, 0.05)), (x, "$x = (1, 1)$", (-0.2, -0.35)), (foot, "closest point", (0.12, 0.0))]:
        ax.plot(*pt, "o", color="k", ms=6); ax.text(pt[0] + off[0], pt[1] + off[1], name, fontsize=9)
    ax.set_xlim(-0.5, 5); ax.set_ylim(-0.5, 5); ax.set_aspect("equal"); ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Distance to the line = length of $(x - x_0)$'s shadow on $w$", fontsize=10)
    plt.show()


def fig_dual_sum():
    """w as a weighted sum of the support vectors: w = sum_i alpha_i y_i x_i."""
    sv = np.array([[2, 1], [1, 2], [3, 3]], float); coef = np.array([-2 / 9, -2 / 9, 4 / 9])
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    pos = np.zeros(2)
    cols = [C0, C0, C1]
    for v, c, col in zip(sv, coef, cols):
        step = c * v
        ax.annotate("", xy=pos + step, xytext=pos, arrowprops=dict(arrowstyle="-|>", color=col, lw=2))
        mid = pos + step / 2; off = np.array([-step[1], step[0]]) / (np.linalg.norm(step) + 1e-9) * 0.12
        ax.text(*(mid + off), f"{c:+.3f}·({v[0]:g}, {v[1]:g})", fontsize=9, color=col, ha="center", bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.3))
        pos = pos + step
    ax.annotate("", xy=pos, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="k", lw=2.5, ls="--"))
    ax.text(pos[0] + 0.03, pos[1] + 0.05, f"w = ({pos[0]:.3f}, {pos[1]:.3f})", fontsize=10, fontweight="bold")
    ax.axhline(0, color=GREY, lw=0.6); ax.axvline(0, color=GREY, lw=0.6)
    ax.set_xlim(-1.0, 1.6); ax.set_ylim(-0.8, 1.6); ax.set_aspect("equal")
    ax.set_title("$w$ is built from the support vectors only:\n$w = \\sum_i \\alpha_i y_i x_i$, walked tip to tail", fontsize=10)
    plt.show()


def fig_dual_bumps(gamma=1.0):
    from sklearn.svm import SVC
    x = np.array([-3, -2.5, -2, -0.5, 0, 0.5, 1, 2.2, 3]); y = np.where(np.abs(x) < 1.5, 1, -1)
    m = SVC(kernel="rbf", gamma=gamma, C=10).fit(x[:, None], y)
    xs = np.linspace(-4, 4, 400)
    fig, ax = plt.subplots(figsize=(10, 4.4))
    total = np.full_like(xs, m.intercept_[0])
    for a, s in zip(m.dual_coef_[0], m.support_vectors_[:, 0]):
        bump = a * np.exp(-gamma * (xs - s) ** 2); total += bump
        ax.plot(xs, bump, color=C1 if a > 0 else C0, lw=1, alpha=0.8)
    ax.plot(xs, total, color="k", lw=2.5, label="$f(x) = \\sum_i \\alpha_i y_i\\, k(x_i, x) + b$")
    ax.axhline(0, color=GREY, lw=0.8)
    ax.fill_between(xs, -3, 3, where=total > 0, color=C1, alpha=0.08)
    ax.scatter(x[y > 0], np.zeros((y > 0).sum()), c=C1, s=70, edgecolor="k", zorder=3, label="class +1")
    ax.scatter(x[y < 0], np.zeros((y < 0).sum()), c=C0, s=70, edgecolor="k", zorder=3, label="class −1")
    ax.set_ylim(-2.2, 2.2); ax.legend(fontsize=8, loc="lower left")
    ax.set_title("Each support vector adds one bump $\\alpha_i y_i k(x_i, x)$ (thin lines: orange up, blue down).\n"
                 "Their sum (thick) is positive exactly where the shaded region predicts +1.", fontsize=10)
    plt.tight_layout(); plt.show()


# ------------------------------------------------------------------ 01 · boosting, one piece at a time
X8 = np.arange(1, 9.0)
Y8 = np.array([1, 1, -1, -1, 1, 1, -1, -1.0])


def fig_boost_map():
    flow({"w": (12, 84, "weights w", "start equal: 1/n", "orange"),
          "st": (37, 84, "best stump", "§2–4: predict, error, search", "orange"),
          "al": (62, 84, "its vote α", "§5", "orange"),
          "rw": (49.5, 62, "reweight", "§6: mistakes get heavier", "orange"),
          "F": (88, 84, "AdaBoost model", "§7: sign of Σ α h", "purple"),
          "f0": (12, 38, "F₀ = mean of y", "§8", "blue"),
          "r": (37, 38, "residuals", "§9: y − F", "blue"),
          "h": (62, 38, "stump on residuals", "§10", "blue"),
          "up": (49.5, 14, "F ← F + η h", "§11", "blue"),
          "G": (88, 38, "boosted model", "§12: F after M rounds", "purple")},
         [("w", "st", ""), ("st", "al", ""), ("al", "rw", ""), ("rw", "st", "next round"), ("al", "F", ""),
          ("f0", "r", ""), ("r", "h", ""), ("h", "up", ""), ("up", "r", "next round"), ("h", "G", "")],
         size=(12, 5.6), title="Two ways to add up stumps: AdaBoost (top, §1–7) and gradient boosting (bottom, §8–12)")


def fig_ada_errors(w=None):
    """Every candidate stump of round 1 and its weighted error."""
    w = np.full(8, 1 / 8) if w is None else np.asarray(w)
    ts = (X8[:-1] + X8[1:]) / 2
    errs = {s: [w[np.where(X8 > t, s, -s) != Y8].sum() for t in ts] for s in (1, -1)}
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(ts, errs[1], "o-", color=C1, label="s = +1: predict +1 to the right of t")
    ax.plot(ts, errs[-1], "o-", color=C0, label="s = −1: predict −1 to the right of t")
    for s, col in [(1, C1), (-1, C0)]:
        for t, e in zip(ts, errs[s]):
            ax.annotate(f"{e:.3f}", (t, e), xytext=(0, 6 if s == 1 else -12), textcoords="offset points", ha="center", fontsize=7, color=col)
    ax.axhline(0.5, color=GREY, ls=":", lw=1); ax.set_xlabel("threshold t"); ax.set_ylabel("weighted error ε")
    ax.legend(fontsize=8); ax.set_title("Round 1: all 14 candidate stumps. The lowest is 0.25, first reached at t = 2.5 with s = −1", fontsize=10)
    plt.tight_layout(); plt.show()


def fig_alpha_curve():
    e = np.linspace(0.02, 0.98, 300)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(e, 0.5 * np.log((1 - e) / e), color="k", lw=2)
    for ee in (0.167, 0.25, 0.5):
        a = 0.5 * np.log((1 - ee) / ee); ax.plot(ee, a, "o", color=C1); ax.annotate(f"ε = {ee:g} → α = {a:.3f}", (ee, a), xytext=(8, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0, color=GREY, lw=0.8); ax.axvline(0.5, color=GREY, lw=0.8, ls=":")
    ax.set_xlabel("weighted error ε"); ax.set_ylabel("vote α"); ax.set_title("A stump's vote: big when it's accurate, 0 for a coin flip", fontsize=10)
    plt.tight_layout(); plt.show()


def fig_reweight():
    w = np.full(8, 1 / 8); t, s = 2.5, -1
    h = np.where(X8 > t, s, -s); a = 0.5 * np.log(3)
    w2 = w * np.exp(-a * Y8 * h); w2n = w2 / w2.sum()
    fig, axes = plt.subplots(1, 3, figsize=(13, 2.9), sharey=True)
    for ax, ww, title in [(axes[0], w, "before: every weight 1/8 = 0.125"), (axes[1], w2, "× e^(−α) if right, × e^(+α) if wrong\n(sum is now 0.866)"),
                          (axes[2], w2n, "÷ 0.866 so they sum to 1:\nright → 1/12, wrong → 1/4")]:
        cols = [C1 if v > 0 else C0 for v in Y8]
        bars = ax.bar(X8, ww, color=cols, edgecolor=["#b00020" if hh != yy else "k" for hh, yy in zip(h, Y8)], linewidth=[2.5 if hh != yy else 0.6 for hh, yy in zip(h, Y8)])
        for xx, v in zip(X8, ww):
            ax.text(xx, v + 0.005, f"{v:.3f}", ha="center", fontsize=7)
        ax.set_title(title, fontsize=9); ax.set_xticks(X8)
    axes[0].set_ylabel("weight"); fig.suptitle("Reweighting after round 1 (red outline = the 2 mistakes: x = 5 and 6)", fontsize=10)
    plt.tight_layout(); plt.show()


def fig_adaboost_rounds(x=X8, y=Y8, rounds=3):
    x = np.asarray(x, float); y = np.asarray(y, float); n = len(x); wts = np.full(n, 1 / n)
    fig, axes = plt.subplots(rounds, 1, figsize=(8, 1.7 * rounds), sharex=True)
    for r, ax in enumerate(np.atleast_1d(axes)):
        xs = np.unique(x); best = (None, 1, 2.0)
        for t in (xs[:-1] + xs[1:]) / 2:
            for s in (1, -1):
                err = wts[np.where(x > t, s, -s) != y].sum()
                if err < best[2] - 1e-12:
                    best = (t, s, err)
        t, s, err = best; alpha = 0.5 * np.log((1 - err) / err); h = np.where(x > t, s, -s)
        ax.scatter(x, np.zeros(n), s=2500 * wts, c=np.where(y > 0, C1, C0), edgecolor="k", zorder=3)
        for xi, wi, ok in zip(x, wts, h == y):
            ax.annotate(f"{wi:.3f}" + ("" if ok else " ✗"), (xi, 0), xytext=(0, -26), textcoords="offset points", ha="center", fontsize=7)
        ax.axvline(t, color="k", lw=2); ax.set_yticks([]); ax.set_ylim(-1, 1)
        ax.set_title(f"round {r + 1}: stump at t = {t:g}, s = {s:+d}; ε = {err:.3f}, α = {alpha:.3f}", fontsize=9)
        wts = wts * np.exp(-alpha * y * h); wts = wts / wts.sum()
    plt.tight_layout(); plt.show()


XG = np.array([1, 2, 3, 4.0]); YG = np.array([1, 3, 6, 10.0])


def fig_gb_steps(lr=0.5, rounds=2):
    """Gradient boosting on 4 points: the residuals each stump is fitted to, and the prediction after each round."""
    F = np.full(4, YG.mean())
    fig, axes = plt.subplots(2, rounds + 1, figsize=(4.2 * (rounds + 1), 6.2))
    for r in range(rounds + 1):
        ax = axes[0, r]
        ax.plot(XG, YG, "o", color="k", ms=7, label="y"); ax.step(XG, F, where="mid", color=C1, lw=2, label="prediction F")
        for xx, yy, ff in zip(XG, YG, F):
            ax.plot([xx, xx], [ff, yy], color="#b00020", lw=1.2, ls=":")
        ax.set_title(f"F_{r}: MSE = {np.mean((YG - F) ** 2):.2f}", fontsize=10); ax.set_ylim(-1, 12); ax.legend(fontsize=7, loc="upper left")
        res = YG - F
        ax2 = axes[1, r]
        if r < rounds:
            best = None
            for t in (XG[:-1] + XG[1:]) / 2:
                L, R = res[XG <= t], res[XG > t]
                sse = ((L - L.mean()) ** 2).sum() + ((R - R.mean()) ** 2).sum()
                if best is None or sse < best[0] - 1e-12:
                    best = (sse, t, L.mean(), R.mean())
            sse, t, lv, rv = best; h = np.where(XG <= t, lv, rv)
            ax2.bar(XG, res, color="#f3c6c6", edgecolor="#b00020", label="residual r = y − F")
            ax2.step(XG, h, where="mid", color=C0, lw=2.5, label=f"stump: split {t:g}, {lv:.2f} | {rv:.2f}")
            ax2.axhline(0, color=GREY, lw=0.8); ax2.set_ylim(-5, 6); ax2.legend(fontsize=7, loc="upper left")
            ax2.set_title(f"stump {r + 1} fitted to the residuals", fontsize=10)
            F = F + lr * h
        else:
            ax2.axis("off"); ax2.text(0.02, 0.7, f"after {rounds} rounds with η = {lr}:\nF = {np.round(F, 3).tolist()}\nresiduals = {np.round(YG - F, 3).tolist()}", fontsize=9, transform=ax2.transAxes)
    plt.tight_layout(); plt.show()


# ------------------------------------------------------------------ 02 · PCA
XP = np.array([[3, 3], [4, 2], [5, 4], [6, 6], [7, 5]], float)


def fig_pca_map():
    flow({"x": (11, 72, "data X", "n × d", "grey"), "c": (37, 72, "center", "§2: subtract the mean", "blue"),
          "s": (63, 72, "covariance S", "§4: d × d", "blue"), "e": (89, 72, "eigenvectors", "§5: directions of spread", "blue"),
          "p": (89, 38, "project", "§6: scores, n × k", "green"), "ev": (63, 38, "how many?", "§7: explained variance", "green"),
          "r": (37, 38, "reconstruct", "§8: back to d features", "green"), "svd": (11, 38, "same via SVD", "§9", "purple")},
         [("x", "c", ""), ("c", "s", ""), ("s", "e", ""), ("e", "p", "keep top k"), ("p", "ev", ""), ("p", "r", ""), ("c", "svd", "")],
         size=(12, 4.8), title="PCA in one picture: find the directions of most spread, then describe each point by them",
         notes=[(2, 14, "§1 asks the question (which direction?), §3 measures spread along any direction, §5 shows the best one is an eigenvector.\n"
                        "§10–12: scaling, real data, and what to use instead when PCA isn't enough.")])


def _proj_panel(ax, deg, Xc, show_var=True):
    u = np.array([np.cos(np.radians(deg)), np.sin(np.radians(deg))])
    s = Xc @ u
    ax.plot(*(np.outer([-3.5, 3.5], u).T), color=C2, lw=1.8)
    for p, sc in zip(Xc, s):
        q = sc * u; ax.plot([p[0], q[0]], [p[1], q[1]], color=GREY, ls=":", lw=1)
        ax.plot(*q, "s", color=C2, ms=6)
    ax.scatter(Xc[:, 0], Xc[:, 1], c=C0, s=55, edgecolor="k", zorder=3)
    ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.2, 3.2); ax.set_aspect("equal")
    if show_var:
        ax.set_title(f"direction {deg}°\nvariance of the shadows = {np.var(s, ddof=1):.2f}", fontsize=10)


def fig_pca_directions():
    Xc = XP - XP.mean(0)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, deg in zip(axes, [0, 45, 135]):
        _proj_panel(ax, deg, Xc)
    fig.suptitle("Project the (centred) points onto a line: green squares are their shadows. Which line keeps them most spread out?", fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.93)); plt.show()


def fig_center():
    mu = XP.mean(0)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for ax, P, title in [(axes[0], XP, f"raw data: mean = ({mu[0]:g}, {mu[1]:g})"), (axes[1], XP - mu, "centred: subtract the mean, now the mean is (0, 0)")]:
        ax.scatter(P[:, 0], P[:, 1], c=C0, s=60, edgecolor="k", zorder=3)
        m = P.mean(0); ax.plot(*m, "X", color=C1, ms=12, label="mean")
        for i, p in enumerate(P):
            ax.annotate(f"({p[0]:g}, {p[1]:g})", p, xytext=(6, -10), textcoords="offset points", fontsize=8)
        ax.axhline(0, color=GREY, lw=0.6); ax.axvline(0, color=GREY, lw=0.6); ax.set_aspect("equal"); ax.legend(fontsize=8); ax.set_title(title, fontsize=10)
    axes[0].set_xlim(-0.5, 8); axes[0].set_ylim(-0.5, 7); axes[1].set_xlim(-3.5, 3.5); axes[1].set_ylim(-3, 3)
    plt.tight_layout(); plt.show()


def fig_variance_by_angle():
    Xc = XP - XP.mean(0); S = Xc.T @ Xc / (len(XP) - 1)
    deg = np.linspace(0, 180, 361); U = np.stack([np.cos(np.radians(deg)), np.sin(np.radians(deg))], 1)
    v = np.einsum("ij,jk,ik->i", U, S, U)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(deg, v, color="k", lw=2)
    for d in (0, 45, 90, 135):
        u = np.array([np.cos(np.radians(d)), np.sin(np.radians(d))]); vv = u @ S @ u
        ax.plot(d, vv, "o", color=C1); ax.annotate(f"{d}°: {vv:.2f}", (d, vv), xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("direction of the line (degrees)"); ax.set_ylabel("variance of the shadows $u^\\top S u$")
    ax.set_title("Spread along every possible direction: largest at 45° (4.5), smallest at 135° (0.5)", fontsize=10)
    plt.tight_layout(); plt.show()


def fig_pca_axes():
    Xc = XP - XP.mean(0); vals, vecs = np.linalg.eigh(Xc.T @ Xc / 4); order = np.argsort(vals)[::-1]; vals, vecs = vals[order], vecs[:, order]
    vecs[:, 0] *= np.sign(vecs[0, 0]); vecs[:, 1] *= np.sign(vecs[1, 1])     # same signs as scikit-learn: PC1 = (0.707, 0.707), PC2 = (-0.707, 0.707)
    fig, ax = plt.subplots(figsize=(6, 5.6))
    ax.scatter(Xc[:, 0], Xc[:, 1], c=C0, s=60, edgecolor="k", zorder=3)
    for k, col in [(0, C1), (1, C2)]:
        v = vecs[:, k] * np.sqrt(vals[k])
        ax.annotate("", xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5))
        ax.plot(*(np.outer([-3.5, 3.5], vecs[:, k]).T), color=col, lw=0.8, ls="--", label=f"PC{k + 1} = ({vecs[0, k]:.3f}, {vecs[1, k]:.3f}), variance λ = {vals[k]:.1f}")
    s1, s2 = Xc @ vecs[:, 0], Xc @ vecs[:, 1]
    for p, a, b in zip(Xc, s1, s2):
        if abs(a) + abs(b) > 1e-9:
            ax.annotate(f"({a:.2f}, {b:.2f})", p, xytext=(-8, 8), textcoords="offset points", fontsize=8, color=GREY, ha="right")
    ax.annotate("(0, 0)", (0, 0), xytext=(-10, -14), textcoords="offset points", fontsize=8, color=GREY, ha="right")
    ax.legend(fontsize=8, loc="lower right"); ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5); ax.set_aspect("equal")
    ax.set_title("The new axes: PC1 along the spread, PC2 across it.\nGrey: each point's coordinates (scores) in the new axes", fontsize=10)
    plt.show()


def _blob(n=60, seed=1):
    rng = np.random.default_rng(seed)
    t = rng.normal(0, 2, n); X = np.c_[t, 0.6 * t + rng.normal(0, 0.9, n)]
    return X - X.mean(0)


def fig_regression_vs_pca():
    X = _blob()
    b = np.polyfit(X[:, 0], X[:, 1], 1)
    vals, vecs = np.linalg.eigh(np.cov(X.T)); u = vecs[:, np.argmax(vals)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    xs = np.linspace(-6, 6, 10)
    axes[0].plot(xs, np.polyval(b, xs), color=C1, lw=2)
    for p in X:
        axes[0].plot([p[0], p[0]], [p[1], np.polyval(b, p[0])], color="#b00020", lw=0.8)
    axes[0].set_title("Least squares: predict $x_2$ from $x_1$.\nMinimizes the VERTICAL gaps", fontsize=10)
    axes[1].plot(*(np.outer([-7, 7], u).T), color=C2, lw=2)
    for p in X:
        q = (p @ u) * u; axes[1].plot([p[0], q[0]], [p[1], q[1]], color="#b00020", lw=0.8)
    axes[1].set_title("PCA: no feature is special.\nMinimizes the PERPENDICULAR gaps", fontsize=10)
    for ax in axes:
        ax.scatter(X[:, 0], X[:, 1], c=C0, s=18, edgecolor="k", linewidth=0.3, zorder=3); ax.set_aspect("equal"); ax.set_xlim(-6, 6); ax.set_ylim(-5, 5)
    plt.tight_layout(); plt.show()


def fig_reconstruction():
    mu = XP.mean(0); Xc = XP - mu; u = np.array([1, 1]) / np.sqrt(2); s = Xc @ u; R = mu + np.outer(s, u)
    fig, ax = plt.subplots(figsize=(6, 5.4))
    ax.plot(*(np.outer([-4, 4], u) + mu).T, color=C2, lw=1.5, label="PC1 line through the mean")
    for p, r in zip(XP, R):
        ax.plot([p[0], r[0]], [p[1], r[1]], color="#b00020", lw=1.2, ls="--")
    ax.scatter(XP[:, 0], XP[:, 1], c=C0, s=60, edgecolor="k", zorder=3, label="original")
    ax.scatter(R[:, 0], R[:, 1], c=C2, s=60, marker="s", edgecolor="k", zorder=3, label="reconstructed from 1 number each")
    for p, r in zip(XP, R):
        ax.annotate(f"({r[0]:g}, {r[1]:g})", r, xytext=(6, -12), textcoords="offset points", fontsize=8, color=C2)
    ax.set_aspect("equal"); ax.set_xlim(1.5, 8.5); ax.set_ylim(0.5, 7.5); ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Keep 1 of 2 numbers per point: the red gaps are what's lost\n(total squared gap 2.0 = the dropped variance 0.5 × (n − 1))", fontsize=10)
    plt.show()


def fig_scaling_pca():
    rng = np.random.default_rng(0)
    age = rng.normal(40, 10, 80); income = 1000 * (20 + 0.8 * (age - 40) + rng.normal(0, 8, 80))
    X = np.c_[age, income]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, Z, title in [(axes[0], X - X.mean(0), "raw units: income is in won (thousands), age in years"),
                         (axes[1], (X - X.mean(0)) / X.std(0), "standardized: both have std 1")]:
        vals, vecs = np.linalg.eigh(np.cov(Z.T)); u = vecs[:, np.argmax(vals)]
        ax.scatter(Z[:, 0], Z[:, 1], s=12, c=C0)
        u = u * (1 if u[1] >= 0 else -1)
        tip = u / np.max(np.abs(u)) * np.array([np.abs(Z[:, 0]).max(), np.abs(Z[:, 1]).max()]) * 0.85
        ax.annotate("", xy=tip, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=C1, lw=3))
        ax.set_title(f"{title}\nPC1 = ({abs(u[0]):.3f}, {abs(u[1]):.3f}) (age, income)", fontsize=9)
        ax.set_xlabel("age (centred)"); ax.set_ylabel("income (centred)")
    axes[0].set_aspect("auto"); axes[1].set_aspect("equal")
    plt.tight_layout(); plt.show()


def fig_scree():
    from sklearn.datasets import load_breast_cancer
    from sklearn.decomposition import PCA
    X = load_breast_cancer().data; Z = (X - X.mean(0)) / X.std(0)
    r = PCA().fit(Z).explained_variance_ratio_
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(range(1, 31), r, color=C0, label="each component")
    ax.plot(range(1, 31), np.cumsum(r), "o-", color=C1, ms=4, label="cumulative")
    k90 = int(np.argmax(np.cumsum(r) >= 0.9)) + 1
    ax.axhline(0.9, color=GREY, ls=":", lw=1); ax.axvline(k90, color=GREY, ls=":", lw=1)
    ax.text(k90 + 0.3, 0.5, f"{k90} components\nkeep 90%", fontsize=9)
    ax.set_xlabel("component"); ax.set_ylabel("share of the total variance"); ax.legend(fontsize=8)
    ax.set_title(f"Breast cancer data, 30 standardized features: PC1 alone keeps {r[0]:.0%}, PC1 + PC2 keep {r[:2].sum():.0%}", fontsize=10)
    plt.tight_layout(); plt.show()


def fig_pca_real():
    from sklearn.datasets import load_breast_cancer, load_digits
    from sklearn.decomposition import PCA
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    d = load_breast_cancer(); Z = (d.data - d.data.mean(0)) / d.data.std(0); S = PCA(2).fit_transform(Z)
    for c, col, name in [(0, C1, "malignant"), (1, C0, "benign")]:
        axes[0].scatter(S[d.target == c, 0], S[d.target == c, 1], s=8, c=col, label=name, alpha=0.7)
    axes[0].legend(fontsize=8); axes[0].set_title("Breast cancer: 30 features → 2 numbers per tumour.\nThe labels were NOT used, yet the classes separate", fontsize=9)
    g = load_digits(); S2 = PCA(2).fit_transform(g.data)
    sc = axes[1].scatter(S2[:, 0], S2[:, 1], c=g.target, cmap="tab10", s=6)
    for k in range(10):
        m = S2[g.target == k].mean(0); axes[1].text(*m, str(k), fontsize=13, fontweight="bold", ha="center", va="center",
                                                    bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.5))
    axes[1].set_title("Handwritten digits: 64 pixels → 2 numbers per image.\n(each number shows where that digit's images sit on average)", fontsize=9)
    for ax in axes:
        ax.set_xlabel("PC1 score"); ax.set_ylabel("PC2 score")
    plt.tight_layout(); plt.show()


def fig_pca_vs_tsne():
    from sklearn.datasets import load_digits
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    g = load_digits(); X = g.data[:900]; y = g.target[:900]
    A = PCA(2).fit_transform(X); B = TSNE(2, random_state=0, init="pca", perplexity=30).fit_transform(X)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, S, title in [(axes[0], A, "PCA: a straight-line projection"), (axes[1], B, "t-SNE: keeps neighbours together (non-linear)")]:
        ax.scatter(S[:, 0], S[:, 1], c=y, cmap="tab10", s=6)
        for k in range(10):
            m = np.median(S[y == k], 0); ax.text(*m, str(k), fontsize=13, fontweight="bold", ha="center", va="center", bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.5))
        ax.set_title(title, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.show()


def fig_pca_vs_lda():
    rng = np.random.default_rng(3)
    base = rng.normal(0, 1, (60, 2)) * [3.0, 0.45]
    X = np.r_[base[:30] + [0, -0.9], base[30:] + [0, 0.9]]; y = np.r_[np.zeros(30), np.ones(30)]
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    u_pca = np.linalg.eigh(np.cov((X - X.mean(0)).T))[1][:, -1]
    u_lda = LinearDiscriminantAnalysis().fit(X, y).scalings_[:, 0]; u_lda = u_lda / np.linalg.norm(u_lda)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.0), gridspec_kw={"width_ratios": [1.4, 1, 1]})
    ax = axes[0]
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c=C0, s=15); ax.scatter(X[y == 1, 0], X[y == 1, 1], c=C1, s=15)
    for u, col, name in [(u_pca, C2, "PCA direction"), (u_lda, C3, "LDA direction")]:
        ax.plot(*(np.outer([-8, 8], u).T), color=col, lw=2, label=name)
    ax.set_aspect("equal"); ax.set_xlim(-8, 8); ax.set_ylim(-4, 4); ax.legend(fontsize=8); ax.set_title("two classes, spread out sideways", fontsize=10)
    for ax, u, name in [(axes[1], u_pca, "shadows on the PCA direction:\nthe classes overlap"), (axes[2], u_lda, "shadows on the LDA direction:\nthe classes separate")]:
        s = X @ u
        ax.hist(s[y == 0], bins=15, color=C0, alpha=0.7); ax.hist(s[y == 1], bins=15, color=C1, alpha=0.7)
        ax.set_title(name, fontsize=10); ax.set_yticks([])
    plt.tight_layout(); plt.show()


# ------------------------------------------------------------------ 03 · k-means
XK = np.array([[1, 1], [1, 2], [2, 1], [4, 3], [5, 4], [5, 3]], float)
NAMES = list("ABCDEF")


def fig_kmeans_map():
    flow({"k": (11, 76, "choose k", "§1", "grey"), "i": (37, 76, "initial centroids", "§2 (k-means++: §6)", "orange"),
          "a": (63, 76, "assign", "§3: nearest centroid", "orange"), "u": (63, 44, "update", "§4: centroid = mean", "orange"),
          "c": (89, 60, "converged?", "§5: nothing moves", "orange"), "ev": (89, 18, "how good? which k?", "§7: inertia, silhouette", "green"),
          "lim": (37, 18, "where it fails", "§8 → notebook 04", "purple")},
         [("k", "i", ""), ("i", "a", ""), ("a", "u", ""), ("u", "a", "repeat"), ("u", "c", ""), ("c", "ev", "yes"), ("ev", "lim", "")],
         size=(12, 4.8), title="k-means: alternate two easy steps until nothing changes")


def _kmeans_trace(C0_=np.array([[1, 1], [2, 1]], float)):
    C = C0_.copy(); snaps = []
    for _ in range(3):
        D = np.sqrt(((XK[:, None] - C[None]) ** 2).sum(2)); lab = D.argmin(1)
        snaps.append(("assign", C.copy(), lab.copy()))
        C = np.array([XK[lab == k].mean(0) for k in range(len(C))])
        snaps.append(("update", C.copy(), lab.copy()))
    return snaps


def _voronoi(ax, C, lim=(0, 6.5)):
    g1, g2 = np.meshgrid(np.linspace(*lim, 200), np.linspace(*lim, 200))
    G = np.c_[g1.ravel(), g2.ravel()]; lab = np.sqrt(((G[:, None] - C[None]) ** 2).sum(2)).argmin(1)
    ax.contourf(g1, g2, lab.reshape(g1.shape), levels=[-0.5, 0.5, 1.5], colors=["#d7e6f6", "#fbe3cf"], alpha=0.8)


def fig_kmeans_iterations():
    snaps = _kmeans_trace()
    show = [("start", np.array([[1, 1], [2, 1]], float), None)] + [(f"{s} {i // 2 + 1}", C, lab) for i, (s, C, lab) in enumerate(snaps[:4])]
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.9))
    prevC = None
    for ax, (title, C, lab) in zip(axes, show):
        _voronoi(ax, C)
        cols = [CLUSTER[l] for l in lab] if lab is not None else ["#9aa3ad"] * 6
        ax.scatter(XK[:, 0], XK[:, 1], c=cols, s=70, edgecolor="k", zorder=3)
        for p, nm in zip(XK, NAMES):
            ax.text(p[0] + 0.12, p[1] + 0.12, nm, fontsize=9)
        ax.scatter(C[:, 0], C[:, 1], marker="X", s=180, c=[C0, C1], edgecolor="k", zorder=4)
        if title.startswith("update") and prevC is not None:
            for a, b in zip(prevC, C):
                if np.linalg.norm(a - b) > 1e-9:
                    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color="k", lw=1.5))
        if lab is not None:
            J = ((XK - C[lab]) ** 2).sum()
            title += f"\nJ = {J:.2f}"
        ax.set_title(title, fontsize=10); ax.set_xlim(0, 6.5); ax.set_ylim(0, 6.5); ax.set_aspect("equal")
        prevC = C
    plt.tight_layout(); plt.show()


def fig_kmeans_local():
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=300, centers=[[0, 0], [5, 0], [0, 5], [5, 5]], cluster_std=0.9, random_state=2)
    inits = [np.array([[0, 0], [5, 0], [0, 5], [5, 5]], float) + 0.3, np.array([[-1, 2], [0, 2.5], [1, 3], [5, 2.5]], float),
             np.array([[2.5, 0], [2.5, 1], [2.5, 4], [2.5, 5]], float)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, C in zip(axes, inits):
        C = C.copy()
        for _ in range(50):
            lab = ((X[:, None] - C[None]) ** 2).sum(2).argmin(1)
            C = np.array([X[lab == k].mean(0) if np.any(lab == k) else C[k] for k in range(4)])
        J = ((X - C[lab]) ** 2).sum()
        ax.scatter(X[:, 0], X[:, 1], c=[CLUSTER[l] for l in lab], s=8)
        ax.scatter(C[:, 0], C[:, 1], marker="X", s=200, c="k")
        ax.set_title(f"inertia J = {J:.0f}", fontsize=10); ax.set_aspect("equal")
    fig.suptitle("Same data, same k = 4, three different starting centroids: k-means can get stuck in a worse answer", fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.93)); plt.show()


def fig_kmeanspp():
    C = np.array([[1, 1]], float)
    D2 = ((XK - C[0]) ** 2).sum(1); p = D2 / D2.sum()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.0))
    a1.scatter(XK[:, 0], XK[:, 1], s=60 + 800 * p, c=C1, edgecolor="k", zorder=3)
    offs = {"A": (-0.1, -0.45), "B": (0.15, 0.15), "C": (0.15, -0.35), "D": (-1.6, 0.25), "E": (0.2, 0.2), "F": (0.25, -0.35)}
    for pt, nm, d2 in zip(XK, NAMES, D2):
        a1.text(pt[0] + offs[nm][0], pt[1] + offs[nm][1], f"{nm}: D² = {d2:g}", fontsize=8)
    a1.scatter(*C[0], marker="X", s=200, c="k", zorder=4); a1.set_title("first centroid = A (picked at random).\nDot size ∝ chance to be picked next", fontsize=9)
    a1.set_xlim(0, 6.5); a1.set_ylim(0, 5.5); a1.set_aspect("equal")
    a2.bar(NAMES, p, color=C1, edgecolor="k")
    for i, v in enumerate(p):
        a2.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    a2.set_title(f"P(next centroid) = D² / ΣD² = D² / {D2.sum():g}", fontsize=9)
    plt.tight_layout(); plt.show()


def fig_silhouette_point():
    lab = np.array([0, 0, 0, 1, 1, 1]); i = 2
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    for j in range(6):
        if j == i:
            continue
        col = C0 if lab[j] == lab[i] else C1
        ax.plot([XK[i, 0], XK[j, 0]], [XK[i, 1], XK[j, 1]], color=col, lw=1.4, ls="-" if lab[j] == lab[i] else "--")
        d = np.linalg.norm(XK[i] - XK[j]); m = XK[i] + 0.62 * (XK[j] - XK[i])
        ax.text(m[0] + 0.05, m[1] - 0.22, f"{d:.3f}", fontsize=8, color=col, bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.3))
    ax.scatter(XK[:, 0], XK[:, 1], c=[CLUSTER[l] for l in lab], s=80, edgecolor="k", zorder=3)
    for p, nm in zip(XK, NAMES):
        ax.text(p[0] - 0.3, p[1] + 0.12, nm, fontsize=10, fontweight="bold")
    a = np.mean([np.linalg.norm(XK[i] - XK[j]) for j in (0, 1)]); b = np.mean([np.linalg.norm(XK[i] - XK[j]) for j in (3, 4, 5)])
    ax.set_title(f"Silhouette of C: a = mean solid = {a:.3f}, b = mean dashed = {b:.3f}\ns = (b − a) / max(a, b) = {(b - a) / max(a, b):.3f}", fontsize=10)
    ax.set_xlim(0, 6); ax.set_ylim(0, 5); ax.set_aspect("equal")
    plt.show()


def fig_elbow_silhouette():
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs
    from sklearn.metrics import silhouette_score
    X, _ = make_blobs(n_samples=400, centers=4, cluster_std=1.0, random_state=7)
    ks = range(2, 9); J, S = [], []
    for k in ks:
        m = KMeans(k, n_init=10, random_state=0).fit(X); J.append(m.inertia_); S.append(silhouette_score(X, m.labels_))
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9))
    axes[0].scatter(X[:, 0], X[:, 1], s=6, c=GREY); axes[0].set_title("the data (4 groups)", fontsize=10)
    axes[1].plot(ks, J, "o-", color="k"); axes[1].set_xlabel("k"); axes[1].set_ylabel("inertia J"); axes[1].set_title("elbow: J always falls; look for the bend", fontsize=10)
    axes[2].plot(ks, S, "o-", color=C1); axes[2].set_xlabel("k"); axes[2].set_ylabel("mean silhouette"); axes[2].set_title("silhouette: higher is better; peaks at the right k", fontsize=10)
    for ax in axes[1:]:
        ax.axvline(4, color=GREY, ls=":")
    plt.tight_layout(); plt.show()


def fig_kmeans_failures():
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs
    X1, _ = moons(300, 0.06, 0)
    X2 = np.r_[np.random.default_rng(1).normal([0, 0], 2.2, (300, 2)), np.random.default_rng(2).normal([5.5, 0], 0.4, (40, 2))]
    rng = np.random.default_rng(3); X3 = np.c_[np.r_[rng.normal(-2, 0.5, 100), rng.normal(2, 0.5, 100)], rng.normal(0, 40, 200)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0))
    for ax, X, title in [(axes[0], X1, "curved shapes: cut straight through"), (axes[1], X2, "a big spread cluster next to a tight one:\nthe border is pulled to the middle"),
                         (axes[2], X3, "unscaled: the feature with big numbers decides\n(the real split is in the y-axis, not x)")]:
        lab = KMeans(2, n_init=10, random_state=0).fit_predict(X)
        ax.scatter(X[:, 0], X[:, 1], c=[CLUSTER[l] for l in lab], s=8); ax.set_title(title, fontsize=9)
    axes[2].set_title("unscaled: the real groups differ in x (left/right),\nbut y has bigger numbers, so k-means cuts top/bottom", fontsize=9)
    plt.tight_layout(); plt.show()


# ------------------------------------------------------------------ 04 · hierarchical, DBSCAN, GMM
X1D = np.array([1, 2, 4, 8, 13.0])


def fig_cluster_map():
    flow({"km": (12, 60, "k-means (03)", "needs k; round blobs", "grey"),
          "h": (40, 82, "hierarchical", "§1: no k up front, a tree", "blue"),
          "d": (40, 55, "DBSCAN", "§2: any shape, finds noise", "green"),
          "g": (40, 28, "Gaussian mixture", "§3: soft, elliptical (EM)", "orange"),
          "cmp": (72, 55, "which one when?", "§4: side by side", "purple"),
          "an": (72, 22, "anomaly detection", "§5: the lab's topic", "purple")},
         [("km", "h", "don't know k"), ("km", "d", "odd shapes, noise"), ("km", "g", "overlap, ovals"),
          ("h", "cmp", ""), ("d", "cmp", ""), ("g", "cmp", ""), ("d", "an", "noise points"), ("g", "an", "low density")],
         size=(12, 5.0), title="Three answers to k-means' three weaknesses")


def fig_dendrograms():
    from scipy.cluster.hierarchy import dendrogram, linkage
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    ax = axes[0]
    ax.scatter(X1D, np.zeros(5), s=90, c="k", zorder=3)
    for v in X1D:
        ax.text(v, 0.12, f"{v:g}", ha="center", fontsize=11)
    ax.set_yticks([]); ax.set_ylim(-0.5, 0.6); ax.set_title("five points on a line", fontsize=10); ax.set_xlim(0, 14)
    for ax, m in zip(axes[1:], ["single", "complete"]):
        Z = linkage(X1D[:, None], m)
        dendrogram(Z, labels=[f"{v:g}" for v in X1D], ax=ax, color_threshold=0, above_threshold_color="k")
        for row in Z:
            ax.axhline(row[2], color=GREY, lw=0.5, ls=":")
        ax.set_title(f"{m} linkage: merge heights {[round(float(h), 1) for h in Z[:, 2]]}", fontsize=10); ax.set_ylabel("distance when merged")
    plt.tight_layout(); plt.show()


X_DB = np.array([1, 1.5, 2, 2.5, 5, 8, 8.5, 9, 12.0])


def fig_dbscan_1d(eps=0.6, min_pts=3):
    n = len(X_DB); nb = [np.where(np.abs(X_DB - x) <= eps)[0] for x in X_DB]
    core = np.array([len(v) >= min_pts for v in nb])
    border = np.array([(not core[i]) and core[nb[i]].any() for i in range(n)])
    fig, ax = plt.subplots(figsize=(12, 2.8))
    for i, x in enumerate(X_DB):
        kind = "core" if core[i] else "border" if border[i] else "noise"
        col = {"core": C1, "border": "#f6c89b", "noise": "#bbbbbb"}[kind]
        ax.add_patch(plt.Rectangle((x - eps, -0.15), 2 * eps, 0.3, fc=col, alpha=0.18 if core[i] else 0, ec="none"))
        ax.scatter([x], [0], s=150, c=col, edgecolor="k", zorder=3)
        up = i % 2 == 0
        ax.text(x, 0.22 if up else -0.28, f"x = {x:g}: {kind}\n{len(nb[i])} within eps", ha="center", va="bottom" if up else "top", fontsize=7.5)
    ax.set_ylim(-0.75, 0.75); ax.set_yticks([]); ax.set_xlim(0, 13)
    ax.set_title(f"eps = {eps}, minPts = {min_pts} (counting the point itself). Shaded: the eps-window of each core point", fontsize=10)
    plt.tight_layout(); plt.show()


def fig_dbscan_vs_kmeans():
    from sklearn.cluster import DBSCAN, KMeans
    X, _ = moons(300, 0.06, 0)
    X = np.r_[X, np.random.default_rng(5).uniform([-1.2, -0.8], [2.2, 1.3], (20, 2))]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    lab = KMeans(2, n_init=10, random_state=0).fit_predict(X)
    axes[0].scatter(X[:, 0], X[:, 1], c=[CLUSTER[l] for l in lab], s=10); axes[0].set_title("k-means, k = 2", fontsize=10)
    lab = DBSCAN(eps=0.2, min_samples=5).fit_predict(X)
    axes[1].scatter(X[:, 0], X[:, 1], c=["#222222" if l < 0 else CLUSTER[l] for l in lab], s=[30 if l < 0 else 10 for l in lab], marker="o")
    axes[1].set_title(f"DBSCAN, eps = 0.2, minPts = 5: {lab.max() + 1} clusters, {np.sum(lab < 0)} noise points (black)", fontsize=10)
    plt.tight_layout(); plt.show()


def fig_kdistance():
    from sklearn.neighbors import NearestNeighbors
    X, _ = moons(300, 0.06, 0)
    X = np.r_[X, np.random.default_rng(5).uniform([-1.2, -0.8], [2.2, 1.3], (20, 2))]
    d = np.sort(NearestNeighbors(n_neighbors=5).fit(X).kneighbors(X)[0][:, -1])
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(d, color="k"); ax.axhline(0.2, color=C1, ls="--"); ax.text(10, 0.22, "eps = 0.2: just below the bend", color=C1, fontsize=9)
    ax.set_xlabel("points, sorted"); ax.set_ylabel("distance to 4th nearest neighbour")
    ax.set_title("Choosing eps: sort every point's distance to its (minPts − 1)-th neighbour and look for the bend", fontsize=10)
    plt.tight_layout(); plt.show()


XG1 = np.array([1, 2, 3, 6, 7, 8.0])


def _npdf(x, m, v):
    return np.exp(-(x - m) ** 2 / (2 * v)) / np.sqrt(2 * np.pi * v)


def fig_gmm_1d(mu=(2.0, 5.0), var=(4.0, 4.0), pi=(0.5, 0.5), title=None):
    xs = np.linspace(-4, 13, 400)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True)
    comps = [pi[k] * _npdf(xs, mu[k], var[k]) for k in range(2)]
    a1.plot(xs, comps[0], color=C0, label=f"π₁·N(μ₁ = {mu[0]:g}, σ₁² = {var[0]:g})"); a1.plot(xs, comps[1], color=C1, label=f"π₂·N(μ₂ = {mu[1]:g}, σ₂² = {var[1]:g})")
    a1.plot(xs, comps[0] + comps[1], color="k", lw=2, label="mixture density p(x)")
    a1.scatter(XG1, np.zeros(6), c="k", zorder=3); a1.legend(fontsize=8); a1.set_ylabel("density")
    a1.set_title(title or "Two Gaussian bumps, and the six points", fontsize=10)
    r = comps[1] / (comps[0] + comps[1])
    a2.plot(xs, r, color=C1, lw=2, label="r₂(x): chance the point came from bump 2"); a2.plot(xs, 1 - r, color=C0, lw=2, label="r₁(x) = 1 − r₂(x)")
    for x in XG1:
        rr = pi[1] * _npdf(x, mu[1], var[1]) / sum(pi[k] * _npdf(x, mu[k], var[k]) for k in range(2))
        a2.plot([x, x], [0, rr], color=GREY, ls=":"); a2.plot(x, rr, "o", color=C1); a2.text(x, rr + 0.04, f"{rr:.3f}", ha="center", fontsize=8)
    a2.set_ylim(0, 1.15); a2.legend(fontsize=8, loc="center left"); a2.set_xlabel("x"); a2.set_ylabel("responsibility")
    plt.tight_layout(); plt.show()


def _ellipse(ax, m, S, col):
    vals, vecs = np.linalg.eigh(S); ang = np.degrees(np.arctan2(vecs[1, -1], vecs[0, -1]))
    for k in (1, 2):
        ax.add_patch(Ellipse(m, 2 * k * np.sqrt(vals[-1]), 2 * k * np.sqrt(vals[0]), angle=ang, fc="none", ec=col, lw=1.5, alpha=0.9 / k))


def gmm_data(seed=4):
    rng = np.random.default_rng(seed)
    a = rng.multivariate_normal([0, 0], [[2.0, 1.4], [1.4, 1.5]], 150); b = rng.multivariate_normal([4, -1], [[0.6, -0.3], [-0.3, 0.8]], 100)
    c = rng.multivariate_normal([2, 4], [[0.5, 0], [0, 0.5]], 80)
    return np.r_[a, b, c]


def em_snapshots(X, K=3, iters=(0, 1, 5, 30), seed=0):
    rng = np.random.default_rng(seed); n, d = X.shape
    mu = X[rng.choice(n, K, replace=False)]; S = np.array([np.cov(X.T)] * K); pi = np.full(K, 1 / K); out = {}
    for it in range(max(iters) + 1):
        dens = np.stack([pi[k] * _mvn(X, mu[k], S[k]) for k in range(K)], 1); R = dens / dens.sum(1, keepdims=True)
        if it in iters:
            out[it] = (mu.copy(), S.copy(), R.copy(), np.log(dens.sum(1)).sum())
        Nk = R.sum(0); mu = (R.T @ X) / Nk[:, None]
        S = np.array([((R[:, k, None] * (X - mu[k])).T @ (X - mu[k])) / Nk[k] + 1e-6 * np.eye(d) for k in range(K)]); pi = Nk / n
    return out


def _mvn(X, m, S):
    d = X.shape[1]; inv = np.linalg.inv(S); diff = X - m
    return np.exp(-0.5 * np.einsum("ij,jk,ik->i", diff, inv, diff)) / np.sqrt((2 * np.pi) ** d * np.linalg.det(S))


def fig_em_iterations():
    X = gmm_data(); snaps = em_snapshots(X)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.3))
    for ax, (it, (mu, S, R, ll)) in zip(axes, snaps.items()):
        cols = R @ np.array([[0.18, 0.44, 0.70], [0.88, 0.48, 0.16], [0.18, 0.55, 0.34]])
        ax.scatter(X[:, 0], X[:, 1], c=np.clip(cols, 0, 1), s=8)
        for k, col in enumerate([C0, C1, C2]):
            _ellipse(ax, mu[k], S[k], col); ax.plot(*mu[k], "X", color=col, ms=12, mec="k")
        ax.set_title(f"after {it} EM step{'s' if it != 1 else ''}: log-likelihood {ll:.0f}", fontsize=10); ax.set_aspect("equal"); ax.set_xlim(-5, 7); ax.set_ylim(-5, 7)
    fig.suptitle("EM on 330 points with 3 components. Point colour = its responsibilities mixed (in-between colours = unsure). Ellipses = 1 and 2 std", fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.93)); plt.show()


def fig_compare_clustering():
    from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
    from sklearn.datasets import make_blobs
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    sets = [("moons", moons(300, 0.06, 0)[0], 2, 0.25), ("blobs, unequal spread", make_blobs(400, centers=[[0, 0], [4, 4], [8, 0]], cluster_std=[1.6, 0.5, 0.9], random_state=1)[0], 3, 0.25),
            ("stretched blobs", gmm_data(), 3, 0.2)]
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    for i, (name, X, k, eps) in enumerate(sets):
        Z = StandardScaler().fit_transform(X)
        models = [("k-means", KMeans(k, n_init=10, random_state=0)), ("hierarchical (Ward)", AgglomerativeClustering(k)),
                  ("DBSCAN", DBSCAN(eps=eps, min_samples=6)), ("Gaussian mixture", GaussianMixture(k, random_state=0))]
        for j, (mname, m) in enumerate(models):
            lab = m.fit_predict(Z); ax = axes[i, j]
            ax.scatter(Z[:, 0], Z[:, 1], c=["#222222" if l < 0 else CLUSTER[l % 6] for l in lab], s=6)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(mname, fontsize=10)
            if j == 0:
                ax.set_ylabel(name, fontsize=10)
    plt.tight_layout(); plt.show()


def fig_anomaly_gmm():
    from sklearn.mixture import GaussianMixture
    X = gmm_data(); rng = np.random.default_rng(9); out = rng.uniform([-6, -5], [8, 8], (12, 2))
    m = GaussianMixture(3, covariance_type="full", random_state=0).fit(X)
    A = np.r_[X, out]; s = m.score_samples(A); thr = np.percentile(m.score_samples(X), 1)
    g1, g2 = np.meshgrid(np.linspace(-7, 9, 200), np.linspace(-6, 9, 200)); G = m.score_samples(np.c_[g1.ravel(), g2.ravel()]).reshape(g1.shape)
    fig, ax = plt.subplots(figsize=(7, 5.6))
    ax.contour(g1, g2, G, levels=[thr - 6, thr - 3, thr, thr + 2], colors=GREY, linewidths=0.8)
    flag = s < thr
    ax.scatter(A[~flag, 0], A[~flag, 1], s=8, c=C0, label="normal: high density")
    ax.scatter(A[flag, 0], A[flag, 1], s=40, c="#b00020", marker="x", label=f"flagged: log-density below the 1st percentile ({flag.sum()} points)")
    ax.legend(fontsize=8, loc="upper right"); ax.set_aspect("equal")
    ax.set_title("Anomaly detection with a Gaussian mixture: fit 'normal' data, flag what the model finds unlikely", fontsize=10)
    plt.show()


# ------------------------------------------------------------------ 05 · project
def fig_project_map():
    flow({"d": (11, 76, "load + look", "§1–2: pandas", "grey"), "s": (37, 76, "split", "§3: stratified", "blue"),
          "p": (63, 76, "preprocess", "§4: impute, scale, one-hot", "blue"), "b": (89, 76, "baseline", "§5: the number to beat", "blue"),
          "m": (89, 47, "compare models", "§6: CV, PR-AUC", "green"), "t": (89, 18, "tune", "§7: GridSearchCV", "green"),
          "th": (63, 18, "threshold", "§8: recall target", "orange"), "e": (37, 18, "test once", "§9 + why (§10)", "orange"),
          "w": (11, 18, "write-up", "§11", "purple")},
         [("d", "s", ""), ("s", "p", ""), ("p", "b", ""), ("b", "m", ""), ("m", "t", ""), ("t", "th", ""), ("th", "e", ""), ("e", "w", "")],
         size=(12, 4.8), title="The mini-project: the order every ML project follows (and the order the exam's \"design a pipeline\" questions expect)")


# ------------------------------------------------------------------ "start from the end": finished models meeting a new point
def fig_pca_new_point(x_new=(7.0, 7.0)):
    mu = XP.mean(0); u1 = np.array([1, 1]) / np.sqrt(2); u2 = np.array([-1, 1]) / np.sqrt(2)
    x = np.asarray(x_new, float); c = x - mu; s1, s2 = c @ u1, c @ u2; rec = mu + s1 * u1
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    ax.scatter(XP[:, 0], XP[:, 1], c="#9aa3ad", s=45, edgecolor="k", zorder=3, label="training points (only used to FIT)")
    ax.plot(*mu, "X", color=C1, ms=13, mec="k", zorder=4, label="stored mean (5, 4)")
    ax.plot(*(np.outer([-3.5, 4.5], u1) + mu).T, color=C2, lw=1.6, label="stored PC1 direction (0.707, 0.707)")
    ax.annotate("", xy=mu + 1.4 * u2, xytext=mu, arrowprops=dict(arrowstyle="-|>", color=C3, lw=1.8))
    ax.text(*(mu + 1.5 * u2 + [-0.6, 0.05]), "stored PC2", color=C3, fontsize=9)
    ax.plot(*x, "*", color="k", ms=16, zorder=5, label=f"new point ({x[0]:g}, {x[1]:g})")
    ax.annotate("", xy=x, xytext=mu, arrowprops=dict(arrowstyle="-|>", color=C0, lw=1.5, ls="--"))
    ax.text(*((x + mu) / 2 + [-1.55, 0.2]), f"x − mean = ({c[0]:g}, {c[1]:g})", color=C0, fontsize=9)
    ax.plot([x[0], rec[0]], [x[1], rec[1]], color="#b00020", lw=1.2, ls=":")
    ax.plot(*rec, "s", color=C2, ms=9, mec="k", zorder=5, label=f"kept with 1 number: ({rec[0]:g}, {rec[1]:g})")
    ax.text(rec[0] + 0.15, rec[1] - 0.45, f"PC1 score = {s1:.3f}", color=C2, fontsize=9)
    ax.set_xlim(1.5, 9); ax.set_ylim(1, 8.5); ax.set_aspect("equal"); ax.legend(fontsize=7.5, loc="upper left")
    ax.set_title(f"The finished PCA = a mean + directions. A new point: subtract the mean,\nthen take its shadow on each direction (scores {s1:.3f}, {s2:.3f})", fontsize=10)
    plt.show()


def fig_kmeans_new_point(x_new=(3.5, 2.0)):
    C = np.array([[4 / 3, 4 / 3], [14 / 3, 10 / 3]]); x = np.asarray(x_new, float); d = np.sqrt(((C - x) ** 2).sum(1))
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    _voronoi(ax, C)
    ax.scatter(XK[:, 0], XK[:, 1], c="#9aa3ad", s=45, edgecolor="k", zorder=3)
    for j, (col, name) in enumerate([(C0, "centroid 0"), (C1, "centroid 1")]):
        ax.plot(*C[j], "X", ms=15, color=col, mec="k", zorder=4)
        ax.text(C[j, 0] + 0.12, C[j, 1] + 0.18, f"{name}\n({C[j, 0]:.3f}, {C[j, 1]:.3f})", fontsize=8, color=col)
        ax.plot([x[0], C[j, 0]], [x[1], C[j, 1]], color=col, lw=1.5, ls="--")
        m = (x + C[j]) / 2; ax.text(m[0], m[1] - 0.3, f"{d[j]:.3f}", fontsize=9, color=col, fontweight="bold")
    ax.plot(*x, "*", ms=17, color="k", zorder=5); ax.text(x[0] + 0.12, x[1] - 0.3, f"new point ({x[0]:g}, {x[1]:g})", fontsize=9)
    ax.set_xlim(0, 6.5); ax.set_ylim(0, 6.5); ax.set_aspect("equal")
    ax.set_title(f"The finished k-means = 2 centroids. A new point goes to the nearer one:\n{d[1]:.3f} < {d[0]:.3f}, so cluster {int(np.argmin(d))}", fontsize=10)
    plt.show()


def fig_new_points_clusterings():
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.4))
    eps = 0.6; nb = [np.where(np.abs(X_DB - v) <= eps)[0] for v in X_DB]; core = np.array([len(v) >= 3 for v in nb])
    lab = np.array([0, 0, 0, 0, -1, 1, 1, 1, -1])
    for v, c, l in zip(X_DB, core, lab):
        col = "#bbbbbb" if l < 0 else CLUSTER[l]
        a1.scatter([v], [0], s=140 if c else 70, c=col, edgecolor="k", zorder=3, marker="o")
        if c:
            a1.add_patch(plt.Rectangle((v - eps, -0.12), 2 * eps, 0.24, fc=col, alpha=0.15, ec="none"))
    for v, txt, off in [(1.9, "1.9 → cluster 0\n(core 2 is 0.1 away)", (-40, -48)), (2.8, "2.8 → noise: 2.5 is close, but it's only\na BORDER point; core 2 is 0.8 away", (55, 22)),
                        (8.9, "8.9 → cluster 1\n(core 8.5 is 0.4 away)", (0, 22)), (6.5, "6.5 → noise\n(no core point within 0.6)", (0, 22))]:
        a1.plot(v, 0, "*", ms=16, color="k", zorder=4)
        a1.annotate(txt, (v, 0), xytext=off, textcoords="offset points", ha="center", fontsize=8, arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6))
    a1.set_ylim(-0.75, 0.8); a1.set_yticks([]); a1.set_xlim(0, 13)
    a1.set_title("DBSCAN, finished: big dots = core points (shaded = their 0.6-windows). A new point joins a cluster only through a CORE point", fontsize=10)
    mu, var, pi = np.array([2.324, 6.0]), np.array([2.536, 4.424]), np.array([0.408, 0.592])
    xs = np.linspace(-3, 13, 400); comps = [pi[k] * _npdf(xs, mu[k], var[k]) for k in range(2)]
    a2.plot(xs, comps[0], color=C0); a2.plot(xs, comps[1], color=C1); a2.plot(xs, comps[0] + comps[1], color="k", lw=2)
    for v in (4.5, 10.0):
        n = pi * _npdf(v, mu, var); r = n / n.sum()
        a2.plot([v, v], [0, n.sum()], color=GREY, ls=":"); a2.plot(v, n.sum(), "*", ms=14, color="k")
        a2.annotate(f"x = {v:g}: r = ({r[0]:.3f}, {r[1]:.3f})\np(x) = {n.sum():.4f}", (v, n.sum()), xytext=(14, -6 if v < 6 else 12), textcoords="offset points", fontsize=8,
                    bbox=dict(fc="white", ec="none", alpha=0.85))
    a2.set_title("Gaussian mixture, finished (after one EM step): a new point gets a probability for each bump, and a density p(x)", fontsize=10)
    a2.set_xlabel("x"); a2.set_ylabel("density"); a2.set_ylim(0, 0.17)
    plt.tight_layout(); plt.show()


def _project_pipeline():
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    df = pd.read_csv("factory_parts.csv")
    num = ["operator_years", "temp", "pressure", "humidity", "vibration", "cycle_time"]; cat = ["line", "shift", "material"]
    Xtr, _, ytr, _ = train_test_split(df[num + cat], df["defect"].to_numpy(), test_size=0.25, random_state=0, stratify=df["defect"])
    def add(d):
        d = d.copy(); d["night_hot"] = ((d["shift"] == "night") & (d["temp"] > 74)).astype(float); d["vib_spike"] = (d["vibration"] > 2).astype(float); return d
    pre = ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num + ["night_hot", "vib_spike"]),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    return make_pipeline(pre, LogisticRegression(max_iter=2000)).fit(add(Xtr), ytr), add


def fig_project_end():
    import pandas as pd
    model, add = _project_pipeline()
    part = add(pd.DataFrame([{"operator_years": 2, "temp": 76.5, "pressure": 101.0, "humidity": np.nan, "vibration": 1.1, "cycle_time": 29.5,
                              "line": "D", "shift": "night", "material": "M3"}]))
    ct, lr = model[0], model[-1]; row = np.asarray(ct.transform(part))[0]; z = row @ lr.coef_[0] + lr.intercept_[0]; p = 1 / (1 + np.exp(-z))
    med = ct.named_transformers_["num"][0].statistics_[3]
    flow({"raw": (11, 74, "a new part", "night · 76.5°C · M3 · line D", "grey"),
          "imp": (37, 74, "1. fill gaps", f"humidity ? → median {med:.1f}", "blue"),
          "sc": (63, 74, "2. scale numbers", "temp → (76.5 − 72.51)/3.0 = 1.33", "blue"),
          "oh": (89, 74, "3. one-hot", "line D → [0, 0, 0, 1]", "blue"),
          "z": (89, 36, "4. score", f"z = w·x + b = {z:.2f}", "orange"),
          "p": (63, 36, "5. probability", f"p̂ = σ({z:.2f}) = {p:.3f}", "orange"),
          "t": (37, 36, "6. threshold", f"{p:.3f} ≥ 0.083", "purple"),
          "out": (11, 36, "ALARM", "send for inspection", "purple")},
         [("raw", "imp", ""), ("imp", "sc", ""), ("sc", "oh", ""), ("oh", "z", ""), ("z", "p", ""), ("p", "t", ""), ("t", "out", "")],
         size=(13.5, 4.6), title="Where this project ends: the finished pipeline scoring ONE new part (every number is stored from training)",
         notes=[(2, 14, "Stored after fitting: the medians (imputer), means and stds (scaler), the category lists (one-hot), 17 weights + a bias (logistic regression), and the threshold.")])
