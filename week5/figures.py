"""Picture helpers for Week 5. Each function draws one figure from the lessons.

You don't need to read this file to follow the lessons. Shared tools (colours, table pictures, flow maps) are in fig_common.py.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from fig_common import *  # noqa: F401,F403
from fig_common import _arrow, _canvas, _regions, _table  # noqa: F401

PALE = ["#d7e6f6", "#fde2c8", "#dcefe3", "#ebe0f3", "#f6d5d5", "#e4e4e4"]


def _f(v, d=3):
    """A number as short text: 0.5 -> '0.5', 2.0 -> '2', 0.1234 -> '0.123'."""
    s = f"{v:.{d}f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


# ------------------------------------------------------------------ 00 · the week, your code, hierarchical clustering
def fig_week5_map():
    flow({"rev": (12, 80, "00 review", "your Week 4, the plan", "grey"),
          "np": (12, 48, "01 formula → NumPy", "write vectorized code", "blue"),
          "nn": (40, 80, "02 neural networks", "the forward pass", "orange"),
          "bp": (40, 48, "03 backprop", "gradients by hand", "orange"),
          "mg": (40, 16, "04 micrograd", "an autograd engine", "orange"),
          "opt": (68, 48, "05 optimizers", "SGD, momentum, Adam", "green"),
          "pt": (68, 16, "06 PyTorch", "all of it, for real", "purple"),
          "exam": (91, 32, "07 + 08", "paper practice, quiz", "grey")},
         [("rev", "np", ""), ("np", "nn", "shapes"), ("nn", "bp", "loss → gradients"), ("bp", "mg", "automate it"),
          ("bp", "opt", "use the gradients"), ("mg", "pt", "same idea"), ("opt", "pt", ""), ("pt", "exam", "")],
         size=(12.5, 5.6), title="Week 5: deep learning starts. Every DL idea this week is a NumPy line you can write.",
         notes=[(2, 4, "01 is the Python practice you asked for: it turns formulas with indices into NumPy, starting from your own loops.")])


def fig_shared_array():
    fig, ax = _canvas(16.5, 3.9, (16.5, 4.0))
    hdr = ["0", "1", "2", "3"]
    ax.text(0.2, 0.32, "1. before the call", fontsize=10.5, fontweight="bold")
    _table(ax, 1.6, -0.3, hdr, [["0.25", "0.25", "0.25", "0.25"]], cw=0.62, colors={(0, c): PALE[0] for c in range(4)})
    ax.text(0.2, -0.95, "w_start", fontsize=10, family="monospace", fontweight="bold", va="center")
    _arrow(ax, (1.05, -0.95), (1.55, -0.95), color=C0)
    ax.text(5.9, 0.32, "2. inside: w is a second NAME for the same array", fontsize=10.5, fontweight="bold")
    _table(ax, 7.4, -0.3, hdr, [["0.144", "0.433", "0.144", "0.144"]], cw=0.62, colors={(0, c): HL for c in range(4)})
    ax.text(5.9, -0.75, "w_start", fontsize=10, family="monospace", fontweight="bold", va="center")
    ax.text(5.9, -1.2, "w", fontsize=10, family="monospace", fontweight="bold", va="center")
    _arrow(ax, (6.85, -0.75), (7.35, -0.9), color=C0); _arrow(ax, (6.2, -1.2), (7.35, -1.0), color=C1)
    ax.text(5.9, -1.7, "w[i] = w[i] * np.exp(...)  writes INTO that array (yellow).", fontsize=9, va="top", family="monospace")
    ax.text(11.9, 0.32, "3. w = w / w.sum() makes a NEW array", fontsize=10.5, fontweight="bold")
    _table(ax, 13.0, -0.3, hdr, [["0.167", "0.5", "0.167", "0.167"]], cw=0.62, colors={(0, c): PALE[2] for c in range(4)})
    ax.text(11.9, -0.95, "w", fontsize=10, family="monospace", fontweight="bold", va="center")
    _arrow(ax, (12.2, -0.95), (12.95, -0.95), color=C1)
    ax.text(11.9, -1.7, "Only the local name w moves to it.\nw_start still points at the yellow array.", fontsize=9, va="top")
    ax.text(0.2, -2.55, "After the call: w_start = [0.144 0.433 0.144 0.144]. It was changed, and it doesn't even sum to 1.\n"
            "The fix: never write into an array you were given. Start with  w = w.copy()  or compute a new array:  w = w * np.exp(-alpha * y * h).",
            fontsize=9.5, va="top")
    plt.show()


X_AGG = np.array([1, 2, 4, 8, 13.0])


def _agglo_steps(X, how):
    """Each step: (clusters as lists of indices, the merged pair (a, b) or None, merge distance or None)."""
    clusters = [[i] for i in range(len(X))]
    out = [([c[:] for c in clusters], None, None)]
    while len(clusters) > 1:
        best = None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                D = np.abs(X[clusters[a]][:, None] - X[clusters[b]][None, :])
                d = {"single": D.min(), "complete": D.max(), "average": D.mean()}[how]
                if best is None or d < best[0] - 1e-12:
                    best = (d, a, b)
        d, a, b = best
        pair = (clusters[a][:], clusters[b][:])
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]
        out.append(([c[:] for c in clusters], pair, d))
    return out


def fig_agglomerative(how="single"):
    from scipy.cluster.hierarchy import dendrogram, linkage
    X = X_AGG
    steps = _agglo_steps(X, how)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.9, 1]})
    for s, (clusters, pair, d) in enumerate(steps):
        y = -s
        ax.plot([0, 14], [y, y], color="#d0d4d9", lw=0.8, zorder=1)
        for c in clusters:
            vals = X[c]
            new = pair is not None and sorted(c) == sorted(pair[0] + pair[1])
            col = C1 if new else C0
            ax.add_patch(FancyBboxPatch((vals.min() - 0.32, y - 0.17), vals.max() - vals.min() + 0.64, 0.34,
                                        boxstyle="round,pad=0.02,rounding_size=0.15", fc=HL if new else PALE[0], ec=col, lw=1.4, zorder=2))
            ax.scatter(vals, np.full(len(vals), y), s=40, c="k", zorder=3)
        label = f"step {s}: {len(clusters)} clusters"
        ax.text(-0.6, y, label, ha="right", va="center", fontsize=9.5)
        if pair is not None:
            A = "{" + ", ".join(_f(v) for v in X[pair[0]]) + "}"
            B = "{" + ", ".join(_f(v) for v in X[pair[1]]) + "}"
            ax.text(14.4, y, f"merge {A} + {B}\n{how} distance = {_f(d)}", va="center", fontsize=8.8)
    for v in X:
        ax.text(v, 0.32, _f(v), ha="center", fontsize=10, fontweight="bold")
    ax.set_xlim(-4.6, 20.5); ax.set_ylim(-len(steps) + 0.5, 0.7); ax.axis("off")
    ax.set_title(f"{how} linkage: at every step, merge the two clusters with the smallest linkage_distance (yellow = just merged)", fontsize=10)
    Z = linkage(X[:, None], how)
    dendrogram(Z, labels=[_f(v) for v in X], ax=ax2, color_threshold=0, above_threshold_color="k")
    for row in Z:
        ax2.axhline(row[2], color=GREY, lw=0.5, ls=":")
    ax2.set_title("the same merges as a dendrogram:\nmerge heights " + ", ".join(_f(h) for h in Z[:, 2]), fontsize=10)
    ax2.set_ylabel("linkage distance when merged")
    plt.tight_layout(); plt.show()


# ------------------------------------------------------------------ 01 · formula → NumPy
X_EM = np.array([1, 2, 3, 6, 7, 8.0])
MU_EM, VAR_EM, PI_EM = np.array([2.0, 5.0]), np.array([4.0, 4.0]), np.array([0.5, 0.5])


def _gauss(x, mu, var):
    return np.exp(-(x - mu) ** 2 / (2 * var)) / np.sqrt(2 * np.pi * var)


def fig_method_map():
    flow({"f": (12, 72, "1. formula", "with every index", "grey"),
          "a": (40, 72, "2. anatomy", "index → axis → shape", "blue"),
          "n": (68, 72, "3. line up", "None where an index is missing", "blue"),
          "c": (68, 30, "4. combine", "+ − × ÷, any function", "orange"),
          "s": (40, 30, "5. sum", "the index that disappears", "orange"),
          "k": (12, 30, "6. check", "shapes + one entry by hand", "green")},
         [("f", "a", ""), ("a", "n", ""), ("n", "c", "broadcasting = the loop"), ("c", "s", ""), ("s", "k", "")],
         size=(12, 4.6), title="The method: from a formula with indices to one line of NumPy",
         notes=[(2, 8, "Stuck? Write the loop first (always allowed), then translate it with the table in §5.")])


def fig_estep_broadcast():
    x, mu = X_EM, MU_EM
    fig, ax = _canvas(16.2, 4.3, (16.2, 4.6))
    idx = [f"i={i}" for i in range(6)]
    hl = (2, 1)
    _table(ax, 0.75, 0, ["x_i"], [[_f(v)] for v in x], cw=0.75, index=idx, title="x[:, None]\n(6, 1)", tfs=9.5,
           colors={(hl[0], 0): HL})
    ax.text(1.75, -1.5, "→", fontsize=16)
    _table(ax, 2.3, 0, ["k=0", "k=1"], [[_f(v), _f(v)] for v in x], cw=0.72, title="copied\nacross", tfs=9,
           colors={(r, c): "#f3f3f3" for r in range(6) for c in range(2)} | {hl: HL})
    _table(ax, 4.6, 0, ["k=0", "k=1"], [[_f(mu[0]), _f(mu[1])]], cw=0.72, title="mu[None, :]\n(1, 2)", tfs=9.5,
           colors={(0, 1): HL})
    ax.text(6.25, -0.65, "→", fontsize=16)
    _table(ax, 6.8, 0, ["k=0", "k=1"], [[_f(mu[0]), _f(mu[1])] for _ in x], cw=0.72, title="copied\ndown", tfs=9,
           colors={(r, c): "#f3f3f3" for r in range(6) for c in range(2)} | {hl: HL})
    ax.text(8.55, -1.5, "=", fontsize=16)
    D = x[:, None] - mu[None, :]
    _table(ax, 9.25, 0, ["k=0 (μ=2)", "k=1 (μ=5)"], [[_f(v) for v in r] for r in D], cw=1.15,
           title="x[:, None] - mu[None, :]\n(6, 2)", tfs=9.5, colors={hl: HL})
    ax.text(12.0, -0.25, "Entry (i, k) = x_i − μ_k.\n\nYellow, (2, 1):\nx_2 − μ_1 = 3 − 5 = −2.\n\n"
            "Rows come from x\n(one row per point i),\ncolumns from μ\n(one column per bump k).\n\n"
            "Your double loop filled\nthese 12 cells one by one;\nbroadcasting fills them\nall at once.", fontsize=9.3, va="top")
    plt.show()


def fig_estep_rest():
    x, mu, var, pi = X_EM, MU_EM, VAR_EM, PI_EM
    G = _gauss(x[:, None], mu[None, :], var[None, :]); m = pi[None, :] * G; s = m.sum(axis=1, keepdims=True); R = m / s
    fig, ax = _canvas(16.5, 3.9, (16.5, 4.0))
    hlr = {(2, 0): HL, (2, 1): HL}
    _table(ax, 0.7, 0, ["k=0", "k=1"], [[f"{v:.3f}" for v in r] for r in G], cw=0.8, index=[f"i={i}" for i in range(6)],
           title="G = gaussian_pdf(...)\n(6, 2)", tfs=9, colors=hlr)
    ax.text(2.55, -1.5, "×π→", fontsize=12)
    _table(ax, 3.45, 0, ["k=0", "k=1"], [[f"{v:.3f}" for v in r] for r in m], cw=0.8, title="m = pi[None, :] * G\n(6, 2)", tfs=9, colors=hlr)
    ax.text(5.3, -1.5, "→", fontsize=16)
    _table(ax, 5.9, 0, ["row sum"], [[f"{v:.3f}"] for v in s.ravel()], cw=1.1, title="m.sum(axis=1,\n keepdims=True)\n(6, 1)", tfs=9,
           colors={(2, 0): HL})
    ax.text(7.2, -1.5, "→", fontsize=16)
    _table(ax, 7.8, 0, ["k=0", "k=1"], [[f"{v:.3f}" for v in r] for r in R], cw=0.8, title="R = m / sums\n(6, 2)", tfs=9, colors=hlr)
    ax.text(9.75, -0.25, "Follow row 2 (x = 3), with 4 decimals:\n"
            f"G:  N(3 | 2, 4) = {G[2, 0]:.4f},  N(3 | 5, 4) = {G[2, 1]:.4f}\n"
            f"m:  × π = 0.5 each  →  {m[2, 0]:.4f}, {m[2, 1]:.4f}\n"
            f"row sum:  {m[2, 0]:.4f} + {m[2, 1]:.4f} = {s[2, 0]:.4f}\n"
            f"R:  {m[2, 0]:.4f} / {s[2, 0]:.4f} = {R[2, 0]:.3f},  {m[2, 1]:.4f} / {s[2, 0]:.4f} = {R[2, 1]:.3f}\n\n"
            "Every row of R sums to 1: it splits point i\nbetween the two bumps.\n\n"
            "keepdims=True keeps the sums as a COLUMN (6, 1),\nso m / sums divides each row by its own sum.", fontsize=9.3, va="top")
    plt.show()


def fig_mstep_mean():
    x, mu, var, pi = X_EM, MU_EM, VAR_EM, PI_EM
    G = _gauss(x[:, None], mu[None, :], var[None, :]); m = pi[None, :] * G; R = m / m.sum(axis=1, keepdims=True)
    Nk = R.sum(axis=0); RX = R * x[:, None]; mu_new = RX.sum(axis=0) / Nk
    fig, ax = _canvas(16.5, 4.4, (16.5, 4.5))
    col0 = {(r, 0): PALE[0] for r in range(6)}
    _table(ax, 0.7, 0, ["k=0", "k=1"], [[_f(v) for v in r] for r in R], cw=0.78, index=[f"x={_f(v)}" for v in x], title="R\n(6, 2)", tfs=9.5,
           colors=col0)
    ax.text(2.4, -1.5, "×", fontsize=16)
    _table(ax, 2.85, 0, ["k=0", "k=1"], [[_f(v), _f(v)] for v in x], cw=0.72, title="x[:, None]\n(6, 1), copied", tfs=9,
           colors={(r, c): "#f3f3f3" for r in range(6) for c in range(2)})
    ax.text(4.45, -1.5, "=", fontsize=16)
    _table(ax, 4.9, 0, ["k=0", "k=1"], [[_f(v) for v in r] for r in RX], cw=0.78, title="R * x[:, None]\n(6, 2)", tfs=9.5, colors=col0)
    _table(ax, 4.9, -3.15, ["k=0", "k=1"], [[_f(v) for v in RX.sum(axis=0)]], cw=0.78, colors={(0, 0): HL, (0, 1): HL})
    ax.text(6.55, -3.55, "← .sum(axis=0): the 6 rows (index i) collapse; shape (2,)", fontsize=9, va="center")
    _arrow(ax, (5.69, -2.98), (5.69, -3.12), color=GREY, lw=1.0)
    ax.text(7.2, -0.2, "Formula:  μ_k = (1 / N_k) · Σ_i r_ik · x_i\n\n"
            "Inside the sum the indices are (i, k)\n→ build an (n, K) table: entry (i, k) = r_ik · x_i.\n\n"
            "The sum runs over i, the ROW index\n→ i disappears → .sum(axis=0) → shape (K,).\n\n"
            f"N_k = R.sum(axis=0) = [{_f(Nk[0])}, {_f(Nk[1])}]   (the 'number of points' in each bump)\n"
            f"μ = [{_f(RX.sum(axis=0)[0])} / {_f(Nk[0])},  {_f(RX.sum(axis=0)[1])} / {_f(Nk[1])}] = [{_f(mu_new[0])}, {_f(mu_new[1])}]",
            fontsize=9.6, va="top")
    plt.show()


def fig_matmul_cell():
    X = np.array([[1, 2], [3, 4], [5, 6]]); W = np.array([[1, 0, -1], [2, 1, 0]]); Z = X @ W
    fig, ax = _canvas(15.5, 2.9, (15.5, 3.0))
    _table(ax, 0.9, 0, ["k=0", "k=1"], [[str(v) for v in r] for r in X], cw=0.75, index=[f"i={i}" for i in range(3)], title="X  (3, 2)",
           colors={(1, 0): HL, (1, 1): HL})
    ax.text(2.6, -0.9, "@", fontsize=15)
    _table(ax, 3.4, 0, ["j=0", "j=1", "j=2"], [[str(v) for v in r] for r in W], cw=0.75, index=["k=0", "k=1"], title="W  (2, 3)",
           colors={(0, 0): HL, (1, 0): HL})
    ax.text(5.85, -0.9, "=", fontsize=15)
    _table(ax, 6.6, 0, ["j=0", "j=1", "j=2"], [[str(v) for v in r] for r in Z], cw=0.75, title="X @ W  (3, 3)", colors={(1, 0): HL})
    ax.text(9.3, -0.1, "(X @ W)[i, j] = Σ_k X[i, k] · W[k, j]\n\n"
            "Yellow: i = 1, j = 0  →  3·1 + 4·2 = 11\n"
            "(row 1 of X, walked along; column 0 of W, walked down)\n\n"
            "k is the SHARED index: it is summed and disappears.\n"
            "(3, 2) @ (2, 3) → the inner 2's must match → (3, 3).", fontsize=9.6, va="top")
    plt.show()


def _grid(ax, x, y, rows, cols, cell, fc, ec, label, sub=None):
    """A rows × cols block of cells, top-left at (x, y); y grows downward."""
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(plt.Rectangle((x + c * cell, y - (r + 1) * cell), cell, cell, fc=fc, ec=ec, lw=0.7))
    ax.text(x + cols * cell / 2, y + 0.12, label, ha="center", va="bottom", fontsize=10, fontweight="bold", family="monospace")
    if sub:
        ax.text(x + cols * cell / 2, y - rows * cell - 0.12, sub, ha="center", va="top", fontsize=8.5, color="#3d4652")


def fig_dense_shapes():
    fig, ax = _canvas(15, 4.2, (15, 4.3))
    c = 0.5
    _grid(ax, 0.3, -0.2, 4, 3, c, PALE[0], C0, "X  (n, d)", "row = one sample\ncolumn = one input feature")
    ax.text(2.15, -1.2, "@", fontsize=16)
    _grid(ax, 2.8, -0.2, 3, 2, c, PALE[1], C1, "W  (d, h)", "row j = input feature j\ncolumn k = unit k\nW[j, k]: weight from j to k")
    ax.text(4.35, -1.2, "+", fontsize=16)
    _grid(ax, 5.0, -0.2, 1, 2, c, PALE[2], C2, "b  (h,)", "one bias per unit\n(copied to every row)")
    ax.text(6.55, -1.2, "=", fontsize=16)
    _grid(ax, 7.2, -0.2, 4, 2, c, PALE[3], C3, "Z  (n, h)", "row = the same sample\ncolumn = one unit's value")
    ax.text(9.0, -0.25, "Z[i, k] = Σ_j X[i, j] · W[j, k] + b[k]\n\n"
            "• d (inputs) is shared by X and W → summed → gone\n"
            "• n (samples) comes from X, h (units) from W\n"
            "• b (h,) lines up with the LAST axis of Z (n, h):\n   the same bias is added to every sample\n\n"
            "Here n = 4, d = 3, h = 2:\n(4, 3) @ (3, 2) + (2,) → (4, 2)", fontsize=9.6, va="top")
    plt.show()


def fig_dense_backward_shapes():
    fig, ax = _canvas(16, 3.5, (16, 3.6))
    c = 0.42
    ax.text(0.1, 0.3, "dW = X.T @ dZ", fontsize=10.5, fontweight="bold", family="monospace")
    _grid(ax, 0.2, -0.45, 3, 4, c, PALE[0], C0, "X.T (d, n)")
    ax.text(2.0, -0.95, "@", fontsize=14)
    _grid(ax, 2.5, -0.45, 4, 2, c, PALE[3], C3, "dZ (n, h)")
    ax.text(3.5, -0.95, "=", fontsize=14)
    _grid(ax, 3.95, -0.45, 3, 2, c, PALE[1], C1, "dW (d, h)", "same shape as W")
    ax.text(5.9, 0.3, "db = dZ.sum(axis=0)", fontsize=10.5, fontweight="bold", family="monospace")
    _grid(ax, 6.1, -0.45, 4, 2, c, PALE[3], C3, "dZ (n, h)")
    ax.text(7.1, -0.95, "→", fontsize=14)
    _grid(ax, 7.6, -0.45, 1, 2, c, PALE[2], C2, "db (h,)", "same shape as b")
    ax.text(9.4, 0.3, "dX = dZ @ W.T", fontsize=10.5, fontweight="bold", family="monospace")
    _grid(ax, 9.5, -0.45, 4, 2, c, PALE[3], C3, "dZ (n, h)")
    ax.text(10.5, -0.95, "@", fontsize=14)
    _grid(ax, 11.0, -0.45, 2, 3, c, PALE[1], C1, "W.T (h, d)")
    ax.text(12.45, -0.95, "=", fontsize=14)
    _grid(ax, 12.9, -0.45, 4, 3, c, PALE[0], C0, "dX (n, d)", "same shape as X")
    ax.text(0.1, -2.75, "Each gradient has the shape of the thing it is the gradient OF. With only X (n, d), W (d, h) and dZ (n, h) on the table, "
            "there is exactly one way\nto multiply them into each target shape, so the shapes tell you the formula. The sum over n in dW and db: "
            "every sample adds its own contribution.", fontsize=9.3, va="top")
    plt.show()


def fig_pick_entries():
    Z = np.array([[1, 2, 3], [2, 1, 0], [0.5, 2.5, 0], [1, 1, 1.0]]); y = np.array([2, 0, 1, 2])
    P = np.exp(Z - Z.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    picked = P[np.arange(4), y]
    fig, ax = _canvas(15, 2.9, (15, 3.0))
    _table(ax, 1.0, 0, ["c=0", "c=1", "c=2"], [[_f(v) for v in r] for r in P], cw=0.8, index=[f"i={i}" for i in range(4)],
           title="P  (4, 3): softmax of each row", colors={(i, int(y[i])): HL for i in range(4)})
    _table(ax, 4.0, 0, ["y_i"], [[str(v)] for v in y], cw=0.6, title="y  (4,)")
    _table(ax, 5.4, 0, ["p_(i, y_i)"], [[_f(v)] for v in picked], cw=1.15, title="P[np.arange(4), y]", tfs=9.5, colors={(i, 0): HL for i in range(4)})
    _table(ax, 7.3, 0, ["−log"], [[_f(-np.log(v))] for v in picked], cw=0.9, title="loss_i")
    ax.text(8.8, -0.1, "np.arange(4) = [0, 1, 2, 3] gives the rows, y gives the columns:\n"
            "they are paired up → (0, 2), (1, 0), (2, 1), (3, 2).\n\n"
            "One entry per row: the probability of the TRUE class.\n\n"
            f"Cross-entropy = mean of −log = {_f(-np.log(picked).mean())}", fontsize=9.6, va="top")
    plt.show()


# ------------------------------------------------------------------ 02 · neural networks
XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1.0]])
XOR_Y = np.array([0, 1, 1, 0])
XOR_W1, XOR_B1 = np.array([[6.0, -6.0], [6.0, -6.0]]), np.array([-3.0, 9.0])
XOR_W2, XOR_B2 = np.array([[8.0], [8.0]]), np.array([-12.0])


def _sig(z):
    return 1 / (1 + np.exp(-z))


def _xor_net(X):
    H = _sig(X @ XOR_W1 + XOR_B1)
    return H, _sig(H @ XOR_W2 + XOR_B2).ravel()


def fig_nn_map():
    flow({"neuron": (12, 82, "1 one neuron", "= logistic regression", "blue"),
          "xor": (40, 82, "2 XOR", "one line can't do it", "orange"),
          "hidden": (68, 82, "3 a hidden layer", "bends the space", "orange"),
          "act": (68, 50, "4 activations", "why they must be nonlinear", "orange"),
          "layers": (40, 50, "5 layers = matrices", "shapes of a whole batch", "blue"),
          "out": (12, 50, "6–7 output + loss", "softmax, CE, gradient p − y", "green"),
          "count": (12, 18, "8 counting", "parameters", "grey"),
          "ua": (40, 18, "9 universal", "approximation", "grey"),
          "cls": (68, 18, "10 your MLP class", "vs PyTorch, same numbers", "purple")},
         [("neuron", "xor", "fails on"), ("xor", "hidden", "fix"), ("hidden", "act", ""), ("act", "layers", ""),
          ("layers", "out", ""), ("out", "count", ""), ("count", "ua", ""), ("ua", "cls", "")],
         size=(12, 6.0), title="02: a neural network is logistic regressions stacked, with a nonlinearity between them")


def _node(ax, x, y, text, fc, r=0.55, fs=9):
    ax.add_patch(plt.Circle((x, y), r, fc=fc, ec="#33404d", lw=1.3, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=4)


def fig_xor_network(x1=0.9, x2=0.2):
    x = np.array([x1, x2]); z1 = x @ XOR_W1 + XOR_B1; h = _sig(z1); z2 = float(h @ XOR_W2.ravel() + XOR_B2[0]); p = _sig(z2)
    fig, ax = plt.subplots(figsize=(12, 4.6))
    ax.set_xlim(-0.6, 13.5); ax.set_ylim(-0.4, 4.4); ax.set_aspect("equal"); ax.axis("off")
    pos = {"x1": (1, 3.2), "x2": (1, 0.8), "h1": (5, 3.2), "h2": (5, 0.8), "p": (9, 2.0)}
    edges = [("x1", "h1", 6, 0.22), ("x2", "h1", 6, 0.28), ("x1", "h2", -6, 0.28), ("x2", "h2", -6, 0.22), ("h1", "p", 8, 0.3), ("h2", "p", 8, 0.3)]
    for a, b, w, t in edges:
        (xa, ya), (xb, yb) = pos[a], pos[b]
        ax.annotate("", xy=(xb - 0.6 * (xb - xa) / np.hypot(xb - xa, yb - ya), yb - 0.6 * (yb - ya) / np.hypot(xb - xa, yb - ya)),
                    xytext=(xa, ya), arrowprops=dict(arrowstyle="-|>", color=C0 if w > 0 else C1, lw=1.6))
        ax.text(xa + t * (xb - xa), ya + t * (yb - ya) + 0.18, f"{w:+g}", fontsize=10, fontweight="bold", color=C0 if w > 0 else C1,
                ha="center", bbox=dict(fc="white", ec="none", pad=0.5))
    _node(ax, *pos["x1"], f"x1\n{x1:g}", PALE[5]); _node(ax, *pos["x2"], f"x2\n{x2:g}", PALE[5])
    _node(ax, *pos["h1"], f"h1\n{h[0]:.3f}", PALE[0]); _node(ax, *pos["h2"], f"h2\n{h[1]:.3f}", PALE[0])
    _node(ax, *pos["p"], f"p\n{p:.3f}", PALE[1])
    ax.text(5, 4.05, f"bias −3 · 'OR-ish'\nz = 6·{x1:g} + 6·{x2:g} − 3 = {z1[0]:.1f}", ha="center", fontsize=8.8)
    ax.text(5, -0.15, f"bias +9 · 'NAND-ish'\nz = −6·{x1:g} − 6·{x2:g} + 9 = {z1[1]:.1f}", ha="center", va="top", fontsize=8.8)
    ax.text(9, 2.85, f"bias −12 · 'AND'\nz = 8·{h[0]:.3f} + 8·{h[1]:.3f} − 12 = {z2:.3f}", ha="center", fontsize=8.8)
    ax.text(10.1, 1.0, f"p = σ({z2:.3f}) = {p:.3f}\n→ class 1\n\nblue edge: positive weight\norange edge: negative weight\nnode values: after σ", fontsize=9, va="top")
    ax.set_title(f"The finished XOR network on a new input x = ({x1:g}, {x2:g}). Every number on the picture is computed in §E's code.", fontsize=10)
    plt.show()


def fig_xor_problem():
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    ax.scatter(XOR_X[:, 0], XOR_X[:, 1], s=260, c=[C1 if v else C0 for v in XOR_Y], edgecolor="k", zorder=3)
    for (a, b), v in zip(XOR_X, XOR_Y):
        ax.text(a + 0.09, b + 0.09, f"({a:g}, {b:g}) → {v}", fontsize=9)
    xs = np.linspace(-0.4, 1.4, 10)
    ax.plot(xs, 0.5 - 0 * xs, ls="--", color=GREY); ax.plot(xs, 1.4 - xs, ls=":", color=GREY); ax.plot(0.5 + 0 * xs, xs, ls="-.", color=GREY)
    ax.set_xlim(-0.4, 1.6); ax.set_ylim(-0.4, 1.4); ax.set_xlabel("x1"); ax.set_ylabel("x2")
    ax.set_title("XOR: orange = 1, blue = 0. Every line\nleaves an orange and a blue on the same side", fontsize=10)
    plt.tight_layout(); plt.show()


def fig_xor_hidden_space():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    g = np.linspace(-0.4, 1.4, 200); G1, G2 = np.meshgrid(g, g)
    _, P = _xor_net(np.c_[G1.ravel(), G2.ravel()])
    a1.contourf(G1, G2, P.reshape(G1.shape), levels=[0, 0.5, 1], colors=["#d7e6f6", "#fbe3cf"], alpha=0.95)
    a1.plot(g, 0.5 - g, color=C0, lw=1.5, label="h1 = 0.5: 6x1 + 6x2 − 3 = 0")
    a1.plot(g, 1.5 - g, color=C3, lw=1.5, label="h2 = 0.5: −6x1 − 6x2 + 9 = 0")
    a1.scatter(XOR_X[:, 0], XOR_X[:, 1], s=200, c=[C1 if v else C0 for v in XOR_Y], edgecolor="k", zorder=3)
    a1.scatter([0.9], [0.2], marker="*", s=380, c=C1, edgecolor="k", zorder=4)
    a1.set_xlim(-0.4, 1.4); a1.set_ylim(-0.4, 1.4); a1.set_xlabel("x1"); a1.set_ylabel("x2"); a1.legend(fontsize=8, loc="upper right")
    a1.set_title("input space: the network's answer (orange = class 1)\nis the band BETWEEN the two hidden lines", fontsize=10)
    H, _ = _xor_net(XOR_X)
    hg = np.linspace(-0.1, 1.1, 200); HG1, HG2 = np.meshgrid(hg, hg)
    PZ = _sig(8 * HG1 + 8 * HG2 - 12)
    a2.contourf(HG1, HG2, PZ, levels=[0, 0.5, 1], colors=["#d7e6f6", "#fbe3cf"], alpha=0.95)
    a2.plot(hg, 1.5 - hg, color="k", lw=1.5, label="output line: 8h1 + 8h2 − 12 = 0")
    a2.scatter(H[:, 0], H[:, 1], s=200, c=[C1 if v else C0 for v in XOR_Y], edgecolor="k", zorder=3)
    hn, _ = _xor_net(np.array([[0.9, 0.2]]))
    a2.scatter(hn[:, 0], hn[:, 1], marker="*", s=380, c=C1, edgecolor="k", zorder=4)
    a2.text(H[0, 0] + 0.04, H[0, 1] - 0.09, "(0,0)", fontsize=9); a2.text(H[1, 0] - 0.2, H[1, 1] - 0.12, "(0,1) and (1,0)\nland here", fontsize=9)
    a2.text(H[3, 0] - 0.12, H[3, 1] + 0.06, "(1,1)", fontsize=9)
    a2.set_xlim(-0.1, 1.1); a2.set_ylim(-0.1, 1.1); a2.set_xlabel("h1"); a2.set_ylabel("h2"); a2.legend(fontsize=8, loc="lower left")
    a2.set_title("hidden space: the same 4 points after the hidden layer.\nNow ONE line separates them", fontsize=10)
    plt.tight_layout(); plt.show()


def fig_activations():
    z = np.linspace(-5, 5, 400)
    s = _sig(z); t = np.tanh(z); r = np.maximum(z, 0)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, f, d, name, note in [(axes[0], s, s * (1 - s), "sigmoid σ(z)", "σ'(z) = σ(z)(1 − σ(z)) ≤ 0.25"),
                                 (axes[1], t, 1 - t ** 2, "tanh(z)", "tanh'(z) = 1 − tanh²(z) ≤ 1"),
                                 (axes[2], r, (z > 0).astype(float), "ReLU: max(0, z)", "ReLU'(z) = 1 if z > 0, else 0")]:
        ax.plot(z, f, color=C0, lw=2, label="function"); ax.plot(z, d, color=C1, lw=2, ls="--", label="derivative")
        ax.axhline(0, color=GREY, lw=0.6); ax.axvline(0, color=GREY, lw=0.6)
        ax.set_title(f"{name}\n{note}", fontsize=10); ax.legend(fontsize=8, loc="upper left")
    axes[2].set_ylim(-0.5, 5)
    plt.tight_layout(); plt.show()


def fig_linear_vs_relu():
    from sklearn.neural_network import MLPClassifier
    import warnings
    X, y = moons(200, 0.2, 0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, act in zip(axes, ["identity", "relu"]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = MLPClassifier((16, 16), activation=act, solver="lbfgs", max_iter=3000, random_state=0).fit(X, y)
        _regions(ax, m, X, y)
        name = "no activation (identity)" if act == "identity" else "ReLU activations"
        ax.set_title(f"2 hidden layers of 16 units, {name}\ntraining accuracy {m.score(X, y):.2f}", fontsize=10)
    plt.tight_layout(); plt.show()


def fig_mlp_shapes():
    fig, ax = _canvas(16, 3.4, (16, 3.5))
    c = 0.42
    blocks = [(0.3, 4, 2, "X (4, 2)", "4 samples\n2 features", PALE[5], GREY), (2.3, 4, 3, "Z1 (4, 3)", "X @ W1 + b1\nW1 (2, 3), b1 (3,)", PALE[0], C0),
              (4.75, 4, 3, "H (4, 3)", "ReLU(Z1)\nsame shape", PALE[0], C0), (7.2, 4, 2, "Z2 (4, 2)", "H @ W2 + b2\nW2 (3, 2), b2 (2,)", PALE[1], C1),
              (9.3, 4, 2, "P (4, 2)", "softmax per row\nrows sum to 1", PALE[2], C2)]
    for i, (x, r, k, lab, sub, fc, ec) in enumerate(blocks):
        _grid(ax, x, -0.4, r, k, c, fc, ec, lab, sub)
        if i < len(blocks) - 1:
            nxt = blocks[i + 1][0]
            ax.annotate("", xy=(nxt - 0.15, -1.25), xytext=(x + k * c + 0.15, -1.25), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.4))
    ax.text(11.6, -0.3, "Row i stays sample i all the way through.\nOnly the number of COLUMNS changes:\n"
            "2 features → 3 hidden units → 2 classes.\n\nEach weight matrix is (columns in, columns out),\n"
            "so its shape is fixed by the layer sizes,\nnot by the batch size 4.", fontsize=9.5, va="top")
    plt.show()


def fig_softmax_grad(z=(2.0, 1.0, 0.1), y=0):
    z = np.array(z); p = np.exp(z - z.max()); p /= p.sum(); g = p - np.eye(len(z))[y]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    cls = [f"class {c}" + (" (true)" if c == y else "") for c in range(len(z))]
    for ax, vals, title, col in [(axes[0], z, "logits z", GREY), (axes[1], p, "p = softmax(z)", C0), (axes[2], g, "∂L/∂z = p − y", C1)]:
        bars = ax.bar(cls, vals, color=col)
        for b_, v in zip(bars, vals):
            ax.text(b_.get_x() + b_.get_width() / 2, v + (0.04 if v >= 0 else -0.04), f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
        ax.axhline(0, color="k", lw=0.6); ax.set_title(title, fontsize=10)
    axes[2].set_ylim(-0.55, 0.45)
    axes[2].text(0.98, 0.04, "gradient descent moves z AGAINST\nthe gradient: the true class's logit\ngoes up, the others go down,\nin proportion to their probability",
                 transform=axes[2].transAxes, fontsize=8, ha="right", va="bottom")
    plt.tight_layout(); plt.show()


def fig_universal():
    xs = np.linspace(0, 2 * np.pi, 400); target = np.sin(xs) + 0.3 * np.sin(3 * xs)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    for ax, K in zip(axes, [2, 5, 14]):
        knots = np.linspace(0, 2 * np.pi, K + 2)[1:-1]
        F = np.c_[np.ones_like(xs), xs, np.maximum(0, xs[:, None] - knots[None, :])]   # one hidden ReLU unit per knot
        coef = np.linalg.lstsq(F, target, rcond=None)[0]
        ax.plot(xs, target, color=GREY, lw=3, alpha=0.5, label="target")
        ax.plot(xs, F @ coef, color=C1, lw=2, label=f"{K} ReLU units")
        for t in knots:
            ax.axvline(t, color=C0, lw=0.6, ls=":")
        ax.set_title(f"{K} hidden ReLU units = {K} bends (dotted)\nmax error {np.abs(F @ coef - target).max():.2f}", fontsize=10)
        ax.legend(fontsize=8, loc="lower left")
    plt.tight_layout(); plt.show()


# ------------------------------------------------------------------ 03 · backprop
BP_X, BP_Y = np.array([1.0, 2.0]), 1.0
BP_W1, BP_B1 = np.array([[0.1, 0.3], [0.2, -0.1]]), np.array([0.0, 0.1])
BP_W2, BP_B2 = np.array([0.4, -0.5]), 0.2


def _bp221():
    x, y, W1, b1, W2, b2 = BP_X, BP_Y, BP_W1, BP_B1, BP_W2, BP_B2
    z1 = x @ W1 + b1; h = _sig(z1); z2 = h @ W2 + b2; p = _sig(z2)
    d2 = p - y; dW2 = d2 * h; dh = d2 * W2; dz1 = dh * h * (1 - h); dW1 = np.outer(x, dz1)
    return dict(z1=z1, h=h, z2=z2, p=p, L=-np.log(p), d2=d2, dW2=dW2, db2=d2, dh=dh, dz1=dz1, dW1=dW1, db1=dz1)


def fig_bp_map():
    flow({"fwd": (12, 82, "forward pass", "02: compute the loss", "grey"),
          "graph": (40, 82, "1–2 graph + chain rule", "local gradients", "blue"),
          "hand": (68, 82, "3 a 2-2-1 net by hand", "every number", "blue"),
          "dense": (68, 50, "4 a dense layer", "matrix form (01 §8)", "orange"),
          "card": (40, 50, "5 the algorithm", "backprop, as a card", "orange"),
          "check": (12, 50, "6 gradient check", "is my code right?", "green"),
          "vanish": (12, 18, "7 vanishing", "why ReLU, why init", "grey"),
          "train": (40, 18, "8 train on moons", "loss goes down", "purple"),
          "cls": (68, 18, "9 your class", "fit / predict", "purple")},
         [("fwd", "graph", ""), ("graph", "hand", ""), ("hand", "dense", "generalize"), ("dense", "card", ""), ("card", "check", ""),
          ("check", "vanish", ""), ("vanish", "train", ""), ("train", "cls", "")],
         size=(12, 6.0), title="03: backpropagation = the chain rule, applied from the loss backwards, one layer at a time")


def _box(ax, x, y, text, fc, w=1.5, h=0.75, fs=10):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec="#33404d", lw=1.2, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=4)


def fig_chain_graph(x=3.0, y=-1.0, z=2.0):
    q = x + y; f = q * z
    fig, ax = plt.subplots(figsize=(11, 4.2)); ax.set_xlim(0, 11); ax.set_ylim(0, 4.6); ax.axis("off")
    P = {"x": (1, 3.6), "y": (1, 2.2), "z": (1, 0.8), "+": (4.5, 2.9), "*": (8, 1.85)}
    for n in ["x", "y", "z"]:
        _box(ax, *P[n], n, PALE[5], w=1.0)
    _box(ax, *P["+"], "+", PALE[0], w=1.0); _box(ax, *P["*"], "×", PALE[1], w=1.0)
    def edge(a, b, fwd, bwd, dy=0.0):
        (xa, ya), (xb, yb) = P[a], P[b]
        ax.annotate("", xy=(xb - 0.55, yb + dy), xytext=(xa + 0.55, ya), arrowprops=dict(arrowstyle="-|>", color="#33404d", lw=1.3))
        mx, my = (xa + xb) / 2, (ya + yb + dy) / 2
        ax.text(mx, my + 0.2, fwd, ha="center", fontsize=10, color="k", fontweight="bold")
        ax.text(mx, my - 0.32, bwd, ha="center", fontsize=10, color="#c0392b", fontweight="bold")
    edge("x", "+", f"{x:g}", f"grad {z:g}", 0.15); edge("y", "+", f"{y:g}", f"grad {z:g}", -0.15)
    edge("+", "*", f"q = {q:g}", f"grad {z:g}", 0.15); edge("z", "*", f"{z:g}", f"grad {q:g}", -0.15)
    ax.annotate("", xy=(10.3, 1.85), xytext=(8.55, 1.85), arrowprops=dict(arrowstyle="-|>", color="#33404d", lw=1.3))
    ax.text(9.45, 2.05, f"f = {f:g}", ha="center", fontsize=10, fontweight="bold"); ax.text(9.45, 1.4, "grad 1", ha="center", fontsize=10, color="#c0392b", fontweight="bold")
    ax.text(0.1, 4.45, "black = forward value;  red = gradient ∂f/∂(that value), computed right to left", fontsize=9.5)
    ax.text(5.3, 4.0, "× node: each input's gradient = upstream × the OTHER input\n+ node: each input's gradient = upstream × 1", fontsize=9, color="#3d4652")
    plt.show()


def fig_gates():
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.3))
    specs = [("add: copies", [("a = 2", "3"), ("b = 5", "3")], "+", "out = 7", "3"),
             ("multiply: swaps", [("a = 2", "3 × 5 = 15"), ("b = 5", "3 × 2 = 6")], "×", "out = 10", "3"),
             ("max: routes", [("a = 2", "0"), ("b = 5", "3")], "max", "out = 5", "3"),
             ("fan-out: adds", [("x = 3 → used twice", "")], "", "", "")]
    for ax, (title, ins, op, outv, up) in zip(axes, specs):
        ax.set_xlim(0, 6); ax.set_ylim(0, 4); ax.axis("off"); ax.set_title(title, fontsize=10.5, fontweight="bold")
        if op:
            _box(ax, 3.4, 2.0, op, PALE[0], w=1.0)
            for (lab, g), yy in zip(ins, [3.0, 1.0]):
                ax.annotate("", xy=(2.85, 2.0 + (yy - 2.0) * 0.25), xytext=(1.2, yy), arrowprops=dict(arrowstyle="-|>", color="#33404d"))
                ax.text(0.1, yy + 0.25, lab, fontsize=9.5); ax.text(0.1, yy - 0.45, f"grad {g}", fontsize=9.5, color="#c0392b")
            ax.annotate("", xy=(5.6, 2.0), xytext=(3.95, 2.0), arrowprops=dict(arrowstyle="-|>", color="#33404d"))
            ax.text(4.75, 2.2, outv, ha="center", fontsize=9.5); ax.text(4.75, 1.5, f"grad {up}", ha="center", fontsize=9.5, color="#c0392b")
        else:
            _box(ax, 1.0, 2.0, "x = 3", PALE[5], w=1.3)
            _box(ax, 4.0, 3.1, "u = x·y", PALE[0], w=1.6); _box(ax, 4.0, 0.9, "v = x", PALE[0], w=1.6)
            ax.annotate("", xy=(3.2, 3.0), xytext=(1.65, 2.2), arrowprops=dict(arrowstyle="-|>", color="#33404d"))
            ax.annotate("", xy=(3.2, 1.0), xytext=(1.65, 1.8), arrowprops=dict(arrowstyle="-|>", color="#33404d"))
            ax.text(0.3, 3.35, "via u: ∂f/∂u · y = 1 · 2 = 2", fontsize=8.5, color="#c0392b"); ax.text(0.3, 0.45, "via v: ∂f/∂v · 1 = 1", fontsize=8.5, color="#c0392b")
            ax.text(0.0, -0.35, "f = u + v = x·y + x, y = 2:  ∂f/∂x = 2 + 1 = 3", fontsize=9)
    plt.tight_layout(); plt.show()


def fig_bp_221_graph():
    r = _bp221()
    fig, ax = plt.subplots(figsize=(13, 5.0)); ax.set_xlim(-0.5, 14.5); ax.set_ylim(-0.6, 5.0); ax.set_aspect("equal"); ax.axis("off")
    pos = {"x1": (1, 3.6), "x2": (1, 1.0), "h1": (5.5, 3.6), "h2": (5.5, 1.0), "p": (10, 2.3)}
    W1, W2 = BP_W1, BP_W2
    edges = [("x1", "h1", W1[0, 0], r["dW1"][0, 0], (3.25, 3.72)), ("x2", "h1", W1[1, 0], r["dW1"][1, 0], (4.45, 2.55)),
             ("x1", "h2", W1[0, 1], r["dW1"][0, 1], (4.45, 1.55)), ("x2", "h2", W1[1, 1], r["dW1"][1, 1], (3.25, 1.12)),
             ("h1", "p", W2[0], r["dW2"][0], (7.75, 3.12)), ("h2", "p", W2[1], r["dW2"][1], (7.75, 1.82))]
    for a, b, w, g, (lx, ly) in edges:
        (xa, ya), (xb, yb) = pos[a], pos[b]
        L = np.hypot(xb - xa, yb - ya)
        ax.annotate("", xy=(xb - 0.62 * (xb - xa) / L, yb - 0.62 * (yb - ya) / L), xytext=(xa, ya), arrowprops=dict(arrowstyle="-|>", color="#7a8794", lw=1.3))
        ax.text(lx, ly, f"w = {w:g}\ngrad {g:.4f}", fontsize=8.3, ha="center", va="bottom",
                bbox=dict(fc="white", ec="none", pad=0.3), color="#1b2430")
    _node(ax, *pos["x1"], "x1\n1", PALE[5], r=0.6); _node(ax, *pos["x2"], "x2\n2", PALE[5], r=0.6)
    for k, n in enumerate(["h1", "h2"]):
        _node(ax, *pos[n], f"{n}\n{r['h'][k]:.4f}", PALE[0], r=0.6)
        yy = pos[n][1] + (0.75 if k == 0 else -0.75)
        ax.text(pos[n][0], yy, f"z = {r['z1'][k]:.1f},  b = {BP_B1[k]:g}\n∂L/∂h = {r['dh'][k]:.4f},  ∂L/∂z = {r['dz1'][k]:.4f}",
                ha="center", va="bottom" if k == 0 else "top", fontsize=8.5, color="#1b2430")
    _node(ax, *pos["p"], f"p\n{r['p']:.4f}", PALE[1], r=0.6)
    ax.text(10, 3.05, f"z = {r['z2']:.4f},  b = {BP_B2:g}\n∂L/∂z = p − y = {r['d2']:.4f}", ha="center", va="bottom", fontsize=8.5)
    ax.text(11.2, 1.7, f"y = 1\nL = −log p = {float(r['L']):.4f}", fontsize=9.5, va="top")
    ax.set_title("Forward values in the nodes; each edge shows its weight and the gradient of the loss with respect to it (red text in §3's table)", fontsize=10)
    plt.show()


def fig_vanishing(L=10, width=32, n=64):
    rng = np.random.default_rng(0)
    out = {}
    for act in ["sigmoid", "relu"]:
        Xb = rng.normal(size=(n, width))
        Ws = [rng.normal(size=(width, width)) * (np.sqrt(1 / width) if act == "sigmoid" else np.sqrt(2 / width)) for _ in range(L)]
        Hs, Zs = [Xb], []
        for W in Ws:
            Z = Hs[-1] @ W; Zs.append(Z); Hs.append(_sig(Z) if act == "sigmoid" else np.maximum(Z, 0))
        G = np.ones_like(Hs[-1]) / n; norms = []
        for l in range(L - 1, -1, -1):
            dZ = G * (Hs[l + 1] * (1 - Hs[l + 1]) if act == "sigmoid" else (Zs[l] > 0))
            norms.append(np.linalg.norm(Hs[l].T @ dZ)); G = dZ @ Ws[l].T
        out[act] = norms[::-1]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    idx = np.arange(1, L + 1)
    ax.bar(idx - 0.2, out["sigmoid"], 0.4, color=C1, label="sigmoid layers")
    ax.bar(idx + 0.2, out["relu"], 0.4, color=C0, label="ReLU layers (He initialization)")
    ax.set_yscale("log"); ax.set_xticks(idx); ax.set_xlabel("layer (1 = nearest the input, 10 = nearest the loss)")
    ax.set_ylabel("size of ∂L/∂W (log scale)"); ax.legend(fontsize=9, loc="lower right")
    ax.set_title("Gradient size per layer in a 10-layer network at the start of training: with sigmoids it shrinks\n"
                 "about 4–5× per layer going back, so the first layers barely learn", fontsize=10)
    plt.tight_layout(); plt.show()


def fig_eps_tradeoff():
    r = _bp221(); exact = r["dW1"][0, 0]
    def loss(w00):
        W1 = BP_W1.copy(); W1[0, 0] = w00
        return float(-np.log(_sig(_sig(BP_X @ W1 + BP_B1) @ BP_W2 + BP_B2)))
    eps = 10.0 ** -np.arange(1, 14, 0.5)
    c = [abs((loss(BP_W1[0, 0] + e) - loss(BP_W1[0, 0] - e)) / (2 * e) - exact) for e in eps]
    f = [abs((loss(BP_W1[0, 0] + e) - loss(BP_W1[0, 0])) / e - exact) for e in eps]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.loglog(eps, f, "o-", color=C1, label="forward difference (f(w+ε) − f(w)) / ε")
    ax.loglog(eps, c, "o-", color=C0, label="centered difference (f(w+ε) − f(w−ε)) / 2ε")
    ax.invert_xaxis(); ax.set_xlabel("ε (getting smaller →)"); ax.set_ylabel("|numerical − exact|")
    ax.axvspan(1e-4, 1e-6, color=HL, alpha=0.5)
    ax.text(3e-5, 1e-3, "sweet spot\nfor centered", fontsize=9, ha="center")
    ax.text(3e-2, 1e-12, "too large ε:\nthe straight-line\napproximation is off", fontsize=8.5); ax.text(1e-10, 1e-12, "too small ε: rounding\nerror ≈ 1e-16 / ε", fontsize=8.5)
    ax.legend(fontsize=8.5, loc="upper center"); ax.set_title("Error of a numerical derivative of the 2-2-1 network's loss, as ε shrinks", fontsize=10)
    plt.tight_layout(); plt.show()


def _train_moons(sizes=(2, 16, 16, 2), lr=0.5, epochs=2000, seed=0):
    X, y = moons(200, 0.2, 0)
    rng = np.random.default_rng(seed)
    Ws = [rng.normal(size=(a, b)) * np.sqrt(2 / a) for a, b in zip(sizes[:-1], sizes[1:])]; bs = [np.zeros(b) for b in sizes[1:]]
    hist = []
    for _ in range(epochs):
        Hs, Zs = [X], []
        for l, (W, b) in enumerate(zip(Ws, bs)):
            Z = Hs[-1] @ W + b; Zs.append(Z); Hs.append(np.maximum(Z, 0) if l < len(Ws) - 1 else Z)
        E = np.exp(Hs[-1] - Hs[-1].max(1, keepdims=True)); P = E / E.sum(1, keepdims=True)
        hist.append(-np.log(P[np.arange(len(y)), y]).mean())
        G = (P - np.eye(2)[y]) / len(y)
        for l in range(len(Ws) - 1, -1, -1):
            dW = Hs[l].T @ G; db = G.sum(0)
            if l > 0:
                G = (G @ Ws[l].T) * (Zs[l - 1] > 0)
            Ws[l] -= lr * dW; bs[l] -= lr * db
    return X, y, Ws, bs, hist


def fig_moons_training():
    X, y, Ws, bs, hist = _train_moons()

    class _M:
        def predict(self, A):
            H = A
            for l, (W, b) in enumerate(zip(Ws, bs)):
                H = H @ W + b
                H = np.maximum(H, 0) if l < len(Ws) - 1 else H
            return H.argmax(1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
    a1.plot(hist, color=C0, lw=2); a1.set_yscale("log"); a1.set_xlabel("step (one full-batch gradient step each)"); a1.set_ylabel("mean cross-entropy (log scale)")
    a1.set_title(f"loss: {hist[0]:.3f} at the start → {hist[-1]:.3f} after {len(hist)} steps", fontsize=10)
    _regions(a2, _M(), X, y)
    a2.set_title(f"the trained 2-16-16-2 network's regions; training accuracy {(_M().predict(X) == y).mean():.2f}", fontsize=10)
    plt.tight_layout(); plt.show()


# ------------------------------------------------------------------ 04 · micrograd
MG = dict(x1=1.0, w1=0.5, x2=-2.0, w2=0.25, b=0.3)


def _mg_values():
    v = dict(MG)
    v["x1w1"] = v["x1"] * v["w1"]; v["x2w2"] = v["x2"] * v["w2"]; v["s"] = v["x1w1"] + v["x2w2"]; v["z"] = v["s"] + v["b"]; v["o"] = np.tanh(v["z"])
    g = {"o": 1.0}
    g["z"] = 1 - v["o"] ** 2; g["s"] = g["z"]; g["b"] = g["z"]; g["x1w1"] = g["s"]; g["x2w2"] = g["s"]
    g["x1"] = v["w1"] * g["x1w1"]; g["w1"] = v["x1"] * g["x1w1"]; g["x2"] = v["w2"] * g["x2w2"]; g["w2"] = v["x2"] * g["x2w2"]
    return v, g


def fig_micro_map():
    flow({"py": (12, 80, "1 Python first", "dunder methods, closures", "grey"),
          "val": (40, 80, "2 Value", "a number that remembers", "blue"),
          "bw": (68, 80, "3 backward()", "topological order", "blue"),
          "ops": (68, 48, "4 build it", "+ × ** exp tanh relu", "orange"),
          "nn": (40, 48, "5 Neuron, Layer, MLP", "made of Values", "orange"),
          "train": (12, 48, "6 train it", "and match PyTorch", "green"),
          "pt": (40, 16, "7 what PyTorch adds", "tensors, GPUs, many ops", "purple")},
         [("py", "val", ""), ("val", "bw", ""), ("bw", "ops", ""), ("ops", "nn", ""), ("nn", "train", ""), ("train", "pt", "")],
         size=(12, 5.4), title="04: an autograd engine in about 60 lines: what loss.backward() does, for single numbers")


def fig_value_graph(mode="grad"):
    v, g = _mg_values()
    fig, ax = plt.subplots(figsize=(13.5, 4.8)); ax.set_xlim(0, 14.2); ax.set_ylim(-0.3, 5.2); ax.axis("off")
    P = {"x1": (0.9, 4.4), "w1": (0.9, 3.3), "x2": (0.9, 2.0), "w2": (0.9, 0.9), "x1w1": (4.4, 3.85), "x2w2": (4.4, 1.45),
         "s": (7.6, 2.65), "b": (7.6, 0.6), "z": (10.4, 1.65), "o": (13.0, 1.65)}
    ops = {"x1w1": ("×", (2.85, 3.85)), "x2w2": ("×", (2.85, 1.45)), "s": ("+", (6.1, 2.65)), "z": ("+", (9.1, 1.65)), "o": ("tanh", (11.75, 1.65))}
    kids = {"x1w1": ["x1", "w1"], "x2w2": ["x2", "w2"], "s": ["x1w1", "x2w2"], "z": ["s", "b"], "o": ["z"]}
    order = ["x1", "w1", "x1w1", "x2", "w2", "x2w2", "s", "b", "z", "o"]
    for node, (op, (ox, oy)) in ops.items():
        for k in kids[node]:
            ax.annotate("", xy=(ox - 0.33, oy), xytext=(P[k][0] + 0.78, P[k][1]), arrowprops=dict(arrowstyle="-|>", color="#7a8794", lw=1.1))
        ax.add_patch(plt.Circle((ox, oy), 0.32, fc="#fff7d6", ec="#33404d", lw=1.1, zorder=3))
        ax.text(ox, oy, op, ha="center", va="center", fontsize=9 if op == "tanh" else 11, zorder=4)
        ax.annotate("", xy=(P[node][0] - 0.8, P[node][1]), xytext=(ox + 0.33, oy), arrowprops=dict(arrowstyle="-|>", color="#7a8794", lw=1.1))
    for name, (x, y) in P.items():
        leaf = name in MG
        ax.add_patch(FancyBboxPatch((x - 0.75, y - 0.36), 1.5, 0.72, boxstyle="round,pad=0.02,rounding_size=0.1",
                                    fc=PALE[5] if leaf else PALE[0], ec="#33404d", lw=1.1, zorder=3))
        if mode == "grad":
            ax.text(x, y + 0.12, f"{name}  {v[name]:.4g}", ha="center", va="center", fontsize=8.6, zorder=4)
            ax.text(x, y - 0.17, f"grad {g[name]:.4f}", ha="center", va="center", fontsize=8.6, color="#c0392b", zorder=4)
        else:
            ax.text(x, y, f"{name}\n#{order.index(name) + 1}", ha="center", va="center", fontsize=9, zorder=4)
    if mode == "grad":
        ax.set_title("o = tanh(x1·w1 + x2·w2 + b). Black: data (forward). Red: grad = ∂o/∂(that value), filled in by backward()", fontsize=10)
    else:
        ax.set_title("Topological order (#1 … #10): every value comes after the values it was made from. backward() walks it in REVERSE: #10 first", fontsize=10)
    plt.show()


# ------------------------------------------------------------------ 05 · optimizers
VALLEY_H = np.array([1.0, 100.0])          # f(w) = 0.5 * (1 * w1^2 + 100 * w2^2): a long, narrow valley
VALLEY_START = np.array([-4.0, 1.0])
VALLEY_SETTINGS = {"gradient descent": dict(lr=0.019), "momentum": dict(lr=0.005, beta=0.9),
                   "RMSProp": dict(lr=0.1, rho=0.99), "Adam": dict(lr=0.3, b1=0.9, b2=0.999)}


def _valley_path(name, steps=1000, tol=1e-3):
    w = VALLEY_START.copy(); v = np.zeros(2); s = np.zeros(2); m = np.zeros(2); path = [w.copy()]; hit = None
    cfg = VALLEY_SETTINGS[name]
    for t in range(1, steps + 1):
        if 0.5 * (VALLEY_H * w ** 2).sum() < tol and hit is None:
            hit = t - 1
            break
        g = VALLEY_H * w
        if name == "gradient descent":
            w = w - cfg["lr"] * g
        elif name == "momentum":
            v = cfg["beta"] * v + g; w = w - cfg["lr"] * v
        elif name == "RMSProp":
            s = cfg["rho"] * s + (1 - cfg["rho"]) * g ** 2; w = w - cfg["lr"] * g / (np.sqrt(s) + 1e-8)
        else:
            m = cfg["b1"] * m + (1 - cfg["b1"]) * g; s = cfg["b2"] * s + (1 - cfg["b2"]) * g ** 2
            w = w - cfg["lr"] * (m / (1 - cfg["b1"] ** t)) / (np.sqrt(s / (1 - cfg["b2"] ** t)) + 1e-8)
        path.append(w.copy())
    return np.array(path), (hit if hit is not None else steps)


def fig_opt_map():
    flow({"gd": (12, 80, "1 gradient descent", "the learning rate", "blue"),
          "sgd": (40, 80, "2 minibatches (SGD)", "cheap, noisy steps", "blue"),
          "mom": (68, 80, "3 momentum", "average the directions", "orange"),
          "rms": (68, 48, "4 RMSProp", "a step size per weight", "orange"),
          "adam": (40, 48, "5 Adam", "3 + 4 + bias correction", "green"),
          "sched": (12, 48, "6 schedules", "big steps, then small", "grey"),
          "torch": (40, 16, "7 torch.optim", "the same numbers", "purple")},
         [("gd", "sgd", "too slow on big data"), ("sgd", "mom", "zigzags"), ("mom", "rms", "one step size for all"), ("rms", "adam", "combine"),
          ("adam", "sched", ""), ("sched", "torch", "")],
         size=(12, 5.4), title="05: what to do with the gradient. Each method fixes one problem of the one before")


def fig_valley_paths():
    g1, g2 = np.meshgrid(np.linspace(-4.6, 1.2, 300), np.linspace(-1.3, 1.3, 300))
    F = 0.5 * (VALLEY_H[0] * g1 ** 2 + VALLEY_H[1] * g2 ** 2)
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.2))
    for ax, (name, col) in zip(axes.ravel(), zip(VALLEY_SETTINGS, [GREY, C1, C3, C0])):
        path, hit = _valley_path(name)
        ax.contour(g1, g2, F, levels=[0.01, 0.1, 0.5, 2, 5, 10, 25, 50], colors="#b9c2cc", linewidths=0.8)
        shown = path[:61]
        ax.plot(shown[:, 0], shown[:, 1], "o-", color=col, ms=2.5, lw=1.2)
        ax.scatter([0], [0], marker="*", s=180, c="k", zorder=3); ax.scatter([VALLEY_START[0]], [VALLEY_START[1]], s=40, c="k", zorder=3)
        cfg = ", ".join(f"{k} = {v}" for k, v in VALLEY_SETTINGS[name].items())
        ax.set_title(f"{name} ({cfg}): {hit} steps to f < 0.001", fontsize=9.5)
        ax.set_xlim(-4.6, 1.2); ax.set_ylim(-1.3, 1.3); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("The first 60 steps on f(w) = ½(w1² + 100·w2²), from (−4, 1) to the minimum ★. Steep across (up–down), shallow along (left–right).", fontsize=10)
    plt.tight_layout(); plt.show()


def fig_lr_effect():
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    ws = np.linspace(-1.5, 1.5, 200)
    for ax, lr, verdict in zip(axes, [0.1, 0.45, 0.9, 1.05], ["too small: slow", "about right", "too large: zigzags in", "beyond 1: diverges"]):
        w, path = 1.0, [1.0]
        for _ in range(6):
            w = w - lr * 2 * w; path.append(w)
        path = np.array(path)
        ax.plot(ws, ws ** 2, color="#b9c2cc", lw=2)
        shown = path[np.abs(path) < 1.6]
        ax.plot(shown, shown ** 2, "o-", color=C1 if lr > 1 else C0, ms=4, lw=1)
        for i, v in enumerate(shown[:4]):
            ax.text(v, v ** 2 + 0.12, str(i), fontsize=8, ha="center")
        ax.set_title(f"η = {lr}: w ← (1 − 2η)·w = {1 - 2 * lr:+.2f}·w\n{verdict}", fontsize=9.5); ax.set_ylim(-0.1, 2.4)
    fig.suptitle("Gradient descent on f(w) = w², from w = 1 (the numbers are the step index t)", fontsize=10)
    plt.tight_layout(); plt.show()


def fig_sgd_noise():
    rng = np.random.default_rng(0)
    n, d = 512, 5
    X = rng.normal(size=(n, d)); w_true = rng.normal(size=d); y = X @ w_true + 0.5 * rng.normal(size=n)
    full = lambda w: ((X @ w - y) ** 2).mean() / 2
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for b, lr, col in [(n, 0.1, GREY), (32, 0.1, C0), (1, 0.01, C1)]:
        w = np.zeros(d); xs, losses = [0.0], [full(w)]; r2 = np.random.default_rng(1); seen = 0
        for epoch in range(5):
            perm = r2.permutation(n)
            for start in range(0, n, b):
                idx = perm[start:start + b]
                w = w - lr * X[idx].T @ (X[idx] @ w - y[idx]) / len(idx)
                seen += len(idx)
                if b != 1 or seen % 8 == 0:
                    xs.append(seen / n); losses.append(full(w))
        label = "full batch (1 step per epoch)" if b == n else f"minibatch b = {b} ({n // b} steps per epoch)"
        ax.plot(xs, losses, color=col, lw=1.6 if b == n else 1.0, label=f"{label}, η = {lr}")
    ax.set_yscale("log"); ax.set_xlabel("epochs (passes over the 512 samples)"); ax.set_ylabel("loss on ALL the data (log)")
    ax.legend(fontsize=8.5); ax.set_title("Linear regression trained three ways: smaller batches take many more (noisier) steps for the same amount of data", fontsize=10)
    plt.tight_layout(); plt.show()


def fig_ewma(beta=0.9):
    rng = np.random.default_rng(3)
    t = np.arange(1, 61)
    g = np.where(t % 2 == 0, 1.0, -1.0) * 2.0 + 0.6 + 0.4 * rng.normal(size=len(t))      # zigzag ±2 plus a steady 0.6
    v = np.zeros(len(t)); m = np.zeros(len(t)); acc_v = 0.0; acc_m = 0.0
    for i, gi in enumerate(g):
        acc_v = beta * acc_v + gi; v[i] = acc_v
        acc_m = beta * acc_m + (1 - beta) * gi; m[i] = acc_m / (1 - beta ** (i + 1))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 3.6))
    a1.plot(t, g, "o-", color="#b9c2cc", ms=3, lw=1, label="gradient g_t: flips sign every step (±2) + a steady 0.6")
    a1.plot(t, v * (1 - beta), color=C1, lw=2, label="(1 − β)·v_t: the momentum average")
    a1.axhline(0.6, color="k", lw=0.8, ls="--"); a1.legend(fontsize=8, loc="lower right"); a1.set_xlabel("step t")
    a1.set_title(f"momentum (β = {beta}) cancels the zigzag and keeps the steady part", fontsize=10)
    wts = (1 - beta) * beta ** np.arange(30)[::-1]
    a2.bar(np.arange(30) - 29, wts, color=C0)
    a2.set_xlabel("how many steps ago (0 = now)"); a2.set_ylabel("weight")
    a2.set_title(f"the weights (1 − β)·β^k on past gradients: about 1/(1 − β) = {1 / (1 - beta):.0f} steps of memory", fontsize=10)
    plt.tight_layout(); plt.show()


def fig_schedules(epochs=100, lr0=0.1):
    e = np.arange(epochs)
    step = lr0 * 0.1 ** (e // 30)
    expo = lr0 * 0.96 ** e
    cos = 0.5 * lr0 * (1 + np.cos(np.pi * e / epochs))
    warm = np.where(e < 10, lr0 * (e + 1) / 10, 0.5 * lr0 * (1 + np.cos(np.pi * (e - 10) / (epochs - 10))))
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for y, lab, col in [(step, "step decay: ×0.1 every 30 epochs (StepLR)", GREY), (expo, "exponential: ×0.96 every epoch (ExponentialLR)", C2),
                        (cos, "cosine (CosineAnnealingLR)", C0), (warm, "10 epochs of linear warm-up, then cosine", C1)]:
        ax.plot(e, y, lw=2, color=col, label=lab)
    ax.set_xlabel("epoch"); ax.set_ylabel("learning rate"); ax.legend(fontsize=8.5)
    ax.set_title("Learning-rate schedules: large steps early for fast progress, small steps late to settle", fontsize=10)
    plt.tight_layout(); plt.show()


# ------------------------------------------------------------------ 06 · PyTorch
def digits_data():
    """sklearn's 8×8 digits, pixels scaled to 0–1, split 75/25 (stratified, random_state=0)."""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    X, y = load_digits(return_X_y=True)
    return train_test_split(X / 16.0, y, test_size=0.25, random_state=0, stratify=y)


def train_digits_model(epochs=30, history=False):
    """The exact recipe of 06 §8: 64 → 32 → 10, Adam(lr=0.01), batches of 64, seed 0. Deterministic on CPU."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    X_tr, X_te, y_tr, y_te = digits_data()
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
    loader = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr)), batch_size=64, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    hist = []
    for _ in range(epochs):
        model.train(); total = 0.0
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            acc = (model(torch.tensor(X_te, dtype=torch.float32)).argmax(1).numpy() == y_te).mean()
        hist.append((total / len(X_tr), acc))
    return (model, hist) if history else model


def fig_torch_map():
    flow({"t": (12, 80, "1 tensors", "NumPy + gradients + GPU", "blue"),
          "ag": (40, 80, "2 autograd", "04's engine, for arrays", "blue"),
          "mod": (68, 80, "3 nn.Linear, nn.Module", "layers and models", "orange"),
          "loss": (68, 48, "4 losses", "CrossEntropyLoss takes logits", "orange"),
          "loop": (40, 48, "5 the training loop", "five lines", "green"),
          "data": (12, 48, "6–7 data, train/eval", "Dataset, DataLoader, no_grad", "green"),
          "digits": (12, 16, "8 digits", "end to end, save, load", "purple"),
          "mnist": (40, 16, "9 MNIST", "28×28, on the GPU", "purple"),
          "read": (68, 16, "10 reading code", "exam drills", "grey")},
         [("t", "ag", ""), ("ag", "mod", ""), ("mod", "loss", ""), ("loss", "loop", ""), ("loop", "data", ""), ("data", "digits", ""),
          ("digits", "mnist", ""), ("mnist", "read", "")],
         size=(12, 6.0), title="06: PyTorch is 01–05 packaged: the same arrays, gradients, layers and optimizers")


def fig_digit_new(i=0):
    import torch
    import torch.nn as nn
    X_tr, X_te, y_tr, y_te = digits_data()
    model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
    try:
        model.load_state_dict(torch.load("digits_mlp.pt"))
    except Exception:
        model = train_digits_model()
    model.eval()
    with torch.no_grad():
        p = torch.softmax(model(torch.tensor(X_te[i:i + 1], dtype=torch.float32)), dim=1).numpy().ravel()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1, 2.3]})
    a1.imshow(X_te[i].reshape(8, 8), cmap="gray_r"); a1.set_xticks([]); a1.set_yticks([]); a1.grid(False)
    a1.set_title(f"a new image: 8×8 pixels = 64 numbers\n(true label {y_te[i]})", fontsize=10)
    bars = a2.bar(range(10), p, color=[C1 if k == p.argmax() else C0 for k in range(10)])
    for b_, v in zip(bars, p):
        if v > 0.005:
            a2.text(b_.get_x() + b_.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    a2.set_xticks(range(10)); a2.set_ylim(0, 1.12); a2.set_xlabel("digit"); a2.set_ylabel("probability")
    a2.set_title(f"the model's softmax output: predicts {p.argmax()} with probability {p.max():.3f}", fontsize=10)
    plt.tight_layout(); plt.show()


def fig_dataloader(n=10, b=4):
    rng = np.random.default_rng(0); perm = rng.permutation(n)
    fig, ax = _canvas(15, 2.6, (15, 2.7))
    _table(ax, 0.3, 0, [str(i) for i in range(n)], [[f"x{i}, y{i}" for i in range(n)]], cw=0.95, fs=8.5, title=f"the Dataset: {n} samples, ds[i] = (x_i, y_i)")
    starts = list(range(0, n, b))
    x0 = 0.3
    for k, s in enumerate(starts):
        idx = perm[s:s + b]
        _table(ax, x0, -1.35, [f"batch {k}"] + [""] * (len(idx) - 1), [[str(i) for i in idx]], cw=0.62, fs=8.5,
               colors={(0, c): PALE[k % len(PALE)] for c in range(len(idx))})
        x0 += 0.62 * len(idx) + 0.5
    ax.text(x0 + 0.2, -1.55, f"DataLoader(ds, batch_size={b}, shuffle=True):\nshuffle the indices once per epoch, cut into batches of {b};\n"
            f"each batch is stacked into tensors xb ({b}, d) and yb ({b},). {n} samples → {len(starts)} batches,\nthe last one smaller ({n - b * (len(starts) - 1)}).",
            fontsize=9, va="top")
    plt.show()


def fig_digits_training():
    model, hist = train_digits_model(history=True)
    loss = [h[0] for h in hist]; acc = [h[1] for h in hist]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 3.6))
    a1.plot(range(1, len(loss) + 1), loss, "o-", color=C0, ms=3); a1.set_yscale("log"); a1.set_xlabel("epoch"); a1.set_ylabel("mean training loss (log)")
    a1.set_title("training loss per epoch", fontsize=10)
    a2.plot(range(1, len(acc) + 1), acc, "o-", color=C1, ms=3); a2.set_xlabel("epoch"); a2.set_ylabel("test accuracy")
    a2.set_title(f"test accuracy: {acc[0]:.3f} after 1 epoch → {acc[-1]:.3f} after {len(acc)}", fontsize=10); a2.set_ylim(0.75, 1.0)
    plt.tight_layout(); plt.show()
