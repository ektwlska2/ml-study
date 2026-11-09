"""Shared drawing tools for the study notebooks: colours, table pictures, arrows, decision regions, and flow maps.

The notebook-specific pictures live in figures.py, which imports everything from here.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch

C0, C1, C2, C3 = "#2f6fb3", "#e07b2a", "#2e8b57", "#8e44ad"     # blue, orange, green, purple
CLUSTER = [C0, C1, C2, C3, "#c0392b", "#7f8c8d"]
SOFT = ListedColormap(["#d7e6f6", "#fbe3cf"])
GREY = "#555555"
plt.rcParams.update({"figure.dpi": 96, "axes.grid": True, "grid.alpha": 0.25, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
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


FLOW_COLORS = {"blue": ("#dbe8f6", "#2f6fb3"), "orange": ("#fde6d2", "#c0611a"), "green": ("#dcefe3", "#2e7d4f"),
               "purple": ("#ebe0f3", "#7b3fa0"), "grey": ("#eceff2", "#5b6675")}


def flow(boxes, arrows, size=(12, 6), title=None, notes=None):
    """A big-picture map. boxes: {name: (x, y, title, subtitle, colour)} with x, y the box centre in 0–100 units.
    arrows: [(from, to, label)]. notes: [(x, y, text)]."""
    fig, ax = plt.subplots(figsize=size)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    bw, bh = 21, 11
    for name, (x, y, t, sub, col) in boxes.items():
        fc, ec = FLOW_COLORS[col]
        ax.add_patch(FancyBboxPatch((x - bw / 2, y - bh / 2), bw, bh, boxstyle="round,pad=0.4,rounding_size=1.5", fc=fc, ec=ec, lw=1.6))
        ax.text(x, y + (2.0 if sub else 0), t, ha="center", va="center", fontsize=10.5, fontweight="bold", color="#1b2430")
        if sub:
            ax.text(x, y - 2.6, sub, ha="center", va="center", fontsize=8.5, color="#3d4652")
    for a, b, label in arrows:
        (x1, y1), (x2, y2) = boxes[a][:2], boxes[b][:2]
        dx, dy = x2 - x1, y2 - y1
        L = float(np.hypot(dx, dy)); ux, uy = dx / L, dy / L
        # how far from the centre the straight line leaves the box (plus a small gap)
        tx = (bw / 2 + 1.0) / abs(ux) if abs(ux) > 1e-9 else np.inf
        ty = (bh / 2 + 1.4) / abs(uy) if abs(uy) > 1e-9 else np.inf
        t = min(tx, ty)
        start = (x1 + ux * t, y1 + uy * t); end = (x2 - ux * t, y2 - uy * t)
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="-|>", color="#5b6675", lw=1.4, shrinkA=0, shrinkB=0))
        if label:
            ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 1.2, label, ha="center", va="bottom", fontsize=8.5, color="#3d4652",
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.5))
    for x, y, text in notes or []:
        ax.text(x, y, text, fontsize=8.5, color="#3d4652", va="top")
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")
    plt.show()
