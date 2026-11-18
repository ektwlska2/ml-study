"""Picture helpers for Week 6. Each function draws one figure from the lessons.

You don't need to read this file to follow the lessons. Shared tools (colours, table pictures, flow maps) are in fig_common.py.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from fig_common import *  # noqa: F401,F403
from fig_common import _arrow, _canvas, _regions, _table  # noqa: F401

PALE = ["#d7e6f6", "#fde2c8", "#dcefe3", "#ebe0f3", "#f6d5d5", "#e4e4e4"]
RED = "#c0392b"
MNIST_ROOT = "../week5/data"          # MNIST was downloaded in Week 5; nothing is downloaded twice


def _f(v, d=3):
    s = f"{v:.{d}f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _grid(ax, x, y, rows, cols, cell, fc, ec, label, sub=None, fs=10):
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(plt.Rectangle((x + c * cell, y - (r + 1) * cell), cell, cell, fc=fc, ec=ec, lw=0.7))
    ax.text(x + cols * cell / 2, y + 0.12, label, ha="center", va="bottom", fontsize=fs, fontweight="bold", family="monospace")
    if sub:
        ax.text(x + cols * cell / 2, y - rows * cell - 0.12, sub, ha="center", va="top", fontsize=8.3, color="#3d4652")


def _box(ax, x, y, text, fc, w=1.5, h=0.75, fs=10, ec="#33404d"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.2, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=4)


def _sig(z):
    return 1 / (1 + np.exp(-z))


# ------------------------------------------------------------------ 00 · the week and the words
def fig_week6_map():
    flow({"rev": (12, 82, "00 review", "the words of deep learning", "grey"),
          "gd": (40, 82, "01 slowly", "gradient descent, backprop", "blue"),
          "reg": (68, 82, "02 regularization", "weight decay, dropout, ...", "orange"),
          "stab": (68, 50, "03 stability", "init, batch norm, residual", "orange"),
          "conv": (40, 50, "04 convolution", "by hand, sizes, pooling", "green"),
          "cnn": (12, 50, "05 CNNs in PyTorch", "shapes, training, ResNet", "green"),
          "exam": (40, 18, "06 + 07", "paper practice, quiz", "purple")},
         [("rev", "gd", ""), ("gd", "reg", "train well"), ("reg", "stab", ""), ("stab", "conv", "images"), ("conv", "cnn", ""), ("cnn", "exam", "")],
         size=(12, 5.4), title="Week 6: training networks that generalize, then networks for images",
         notes=[(55, 12, "Coding exercises come only from your booklet's\n\"In Python\" blocks, or new blocks written the same way.")])


def fig_training_step():
    fig, ax = _canvas(16.5, 4.9, (16.5, 5.0))
    c = 0.42
    _grid(ax, 0.3, -0.5, 4, 3, c, PALE[5], GREY, "X  (4, 3)", "a BATCH: 4 samples,\n3 features each")
    ax.annotate("", xy=(3.2, -1.3), xytext=(1.7, -1.3), arrowprops=dict(arrowstyle="-|>", color=C0, lw=2))
    _box(ax, 4.2, -1.3, "model\n(the weights W)", PALE[0], w=1.8, h=1.0)
    ax.annotate("", xy=(6.6, -1.3), xytext=(5.2, -1.3), arrowprops=dict(arrowstyle="-|>", color=C0, lw=2))
    _grid(ax, 6.7, -0.5, 4, 2, c, PALE[1], C1, "logits (4, 2)", "one score per class,\nfor each sample")
    ax.annotate("", xy=(9.3, -1.3), xytext=(7.7, -1.3), arrowprops=dict(arrowstyle="-|>", color=C0, lw=2))
    _box(ax, 10.2, -1.3, "loss\n(ONE number)", HL, w=1.6, h=1.0)
    ax.text(4.9, 0.45, "FORWARD PASS: input → prediction → loss", color=C0, fontsize=10.5, fontweight="bold", ha="center")
    ax.annotate("", xy=(4.2, -2.75), xytext=(10.2, -2.75), arrowprops=dict(arrowstyle="-|>", color=RED, lw=2, connectionstyle="arc3,rad=-0.2"))
    ax.text(7.2, -3.75, "BACKWARD PASS: loss.backward() works out, for every weight,\nhow much the loss would change if that weight moved a little (its gradient)",
            color=RED, fontsize=9.5, ha="center", va="top")
    _grid(ax, 12.2, -0.5, 3, 2, c, PALE[4], RED, "grad  (3, 2)", "same shape as W")
    ax.annotate("", xy=(12.1, -1.1), xytext=(11.05, -1.3), arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.5))
    ax.text(13.3, -2.55, "UPDATE (one STEP):\nW ← W − η · grad", fontsize=10, fontweight="bold", ha="center", va="top", color=C2)
    plt.show()


def fig_epoch_steps(n=10, b=4):
    fig, ax = _canvas(15, 3.3, (15, 3.4))
    for e in range(2):
        perm = np.random.default_rng(e).permutation(n)
        y0 = -0.2 - e * 1.55
        ax.text(0.1, y0 - 0.45, f"epoch {e + 1}", fontsize=10, fontweight="bold", va="center")
        x0 = 1.6
        for k, s in enumerate(range(0, n, b)):
            idx = perm[s:s + b]
            for j, i in enumerate(idx):
                ax.add_patch(plt.Rectangle((x0 + j * 0.62, y0 - 0.75), 0.6, 0.6, fc=PALE[k], ec=GREY))
                ax.text(x0 + j * 0.62 + 0.3, y0 - 0.45, str(i), ha="center", va="center", fontsize=9)
            ax.text(x0 + len(idx) * 0.31, y0 - 0.95, f"step {e * 3 + k + 1}: batch of {len(idx)}", ha="center", va="top", fontsize=8.5)
            x0 += len(idx) * 0.62 + 0.6
    ax.text(10.6, -0.3, f"{n} samples, batch size {b}:\n• a STEP = one weight update, from one batch\n• an EPOCH = one pass over all {n} samples\n  = ⌈{n}/{b}⌉ = 3 steps (the last batch has {n % b})\n"
            "• each epoch reshuffles, so the batches differ", fontsize=9.5, va="top")
    plt.show()


def fig_upstream_downstream():
    fig, ax = plt.subplots(figsize=(14, 3.6)); ax.set_xlim(0, 14); ax.set_ylim(0, 3.6); ax.axis("off")
    nodes = [(1.0, "x = 2", PALE[5]), (4.6, "z = w·x = 1.5\n(w = 0.75)", PALE[0]), (8.2, "h = σ(z) = 0.818", PALE[0]), (11.8, "L = −log h = 0.201", HL)]
    for x, t, fc in nodes:
        _box(ax, x, 2.4, t, fc, w=2.4, h=0.9, fs=9.5)
    for (x1, _, _), (x2, _, _) in zip(nodes[:-1], nodes[1:]):
        ax.annotate("", xy=(x2 - 1.25, 2.4), xytext=(x1 + 1.25, 2.4), arrowprops=dict(arrowstyle="-|>", color=C0, lw=1.6))
    grads = [(1.0, "∂L/∂x = −0.137"), (4.6, "∂L/∂z = −0.182"), (8.2, "∂L/∂h = −1.222"), (11.8, "∂L/∂L = 1")]
    for x, t in grads:
        ax.text(x, 1.45, t, ha="center", fontsize=9.5, color=RED, fontweight="bold")
    for (x1, _), (x2, _) in zip(grads[1:], grads[:-1]):
        ax.annotate("", xy=(x2 + 1.0, 1.5), xytext=(x1 - 1.0, 1.5), arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.4))
    ax.add_patch(FancyBboxPatch((6.6, 1.15), 3.2, 1.9, boxstyle="round,pad=0.02", fc="none", ec=C3, lw=1.5, ls="--"))
    ax.text(8.2, 0.55, "at the σ box:  upstream −1.222  ×  local σ'(z) = h(1−h) = 0.149\n= downstream −0.182", ha="center", fontsize=8.8, color=C3)
    ax.text(0.1, 0.2, "upstream = arriving from the LOSS side (right)\ndownstream = sent on towards the INPUT (left)", fontsize=9, va="bottom")
    ax.set_title("Forward values (blue arrows, left → right); gradients (red, right → left). Every box: downstream = upstream × its local gradient", fontsize=10)
    plt.show()


# ------------------------------------------------------------------ 01 · gradient descent and backprop, slowly
def fig_gd_1d():
    L = lambda w: (w - 3) ** 2 + 1
    ws = np.linspace(-1, 6, 300)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2))
    a1.plot(ws, L(ws), color=GREY, lw=2)
    w0 = 0.0; slope = 2 * (w0 - 3)
    t = np.linspace(-0.8, 1.6, 10); a1.plot(t, L(w0) + slope * (t - w0), color=C1, lw=1.5, ls="--")
    a1.scatter([w0], [L(w0)], s=60, c=C1, zorder=3)
    a1.annotate("", xy=(0.6, L(0.6)), xytext=(w0, L(w0)), arrowprops=dict(arrowstyle="-|>", color=C0, lw=2))
    a1.text(0.15, L(0) + 0.8, "at w = 0 the slope is −6:\ngoing RIGHT lowers the loss", fontsize=9, color=C1)
    a1.text(1.0, 6.2, "step = −η × slope = −0.1 × (−6) = +0.6\nso w goes 0 → 0.6", fontsize=9, color=C0)
    a1.set_xlabel("w"); a1.set_ylabel("loss L(w) = (w − 3)² + 1"); a1.set_title("one step, read off the slope", fontsize=10)
    a2.plot(ws, L(ws), color=GREY, lw=2)
    w, path = 0.0, [0.0]
    for _ in range(8):
        w = w - 0.1 * 2 * (w - 3); path.append(w)
    path = np.array(path)
    a2.plot(path, L(path), "o-", color=C0, ms=5)
    for i, v in enumerate(path[:5]):
        a2.text(v, L(v) + 0.5, str(i), fontsize=8, ha="center")
    a2.set_xlabel("w"); a2.set_title("8 steps with η = 0.1: big steps where it's steep, small near the bottom", fontsize=10)
    plt.tight_layout(); plt.show()


def fig_gd_2d():
    A = np.array([1.0, 4.0])
    g1, g2 = np.meshgrid(np.linspace(-3.2, 3.2, 200), np.linspace(-2.2, 2.2, 200))
    F = 0.5 * (A[0] * g1 ** 2 + A[1] * g2 ** 2)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.contour(g1, g2, F, levels=[0.1, 0.5, 1, 2, 4, 6, 9], colors="#b9c2cc")
    w = np.array([-3.0, 1.5]); path = [w.copy()]
    for _ in range(12):
        w = w - 0.2 * A * w; path.append(w.copy())
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], "o-", color=C0, ms=4)
    g = A * path[0]
    ax.annotate("", xy=path[0] + 0.25 * g, xytext=path[0], arrowprops=dict(arrowstyle="-|>", color=RED, lw=2))
    ax.text(*(path[0] + 0.25 * g + np.array([0.05, 0.1])), "gradient: points UPHILL\n(steepest increase)", color=RED, fontsize=9)
    ax.annotate("", xy=path[1], xytext=path[0], arrowprops=dict(arrowstyle="-|>", color=C2, lw=2.2))
    ax.text(path[1][0] + 0.1, path[1][1] - 0.35, "the step: −η × gradient (downhill)", color=C2, fontsize=9)
    ax.scatter([0], [0], marker="*", s=200, c="k", zorder=3)
    ax.set_xlabel("w₁"); ax.set_ylabel("w₂")
    ax.set_title("Two weights: the loss is a landscape; each step goes against the gradient", fontsize=10)
    plt.tight_layout(); plt.show()


def fig_backprop_shapes():
    fig, ax = _canvas(17, 6.6, (17, 6.8))
    c = 0.36
    ax.text(0.1, 0.35, "FORWARD (keep every H and Z)", fontsize=11, fontweight="bold", color=C0)
    y = -0.35
    items = [(0.2, 4, 3, "X (4,3)", PALE[5], GREY, "the batch"), (2.0, 3, 5, "W1 (3,5)", PALE[1], C1, "@"),
             (4.4, 4, 5, "Z1 (4,5)", PALE[0], C0, "= X@W1+b1"), (6.8, 4, 5, "H1 (4,5)", PALE[0], C0, "ReLU(Z1)"),
             (9.2, 5, 2, "W2 (5,2)", PALE[1], C1, "@"), (10.6, 4, 2, "Z2 (4,2)", PALE[3], C3, "= H1@W2+b2"), (12.2, 4, 2, "P (4,2)", PALE[2], C2, "softmax")]
    for x, r, k, lab, fc, ec, sub in items:
        _grid(ax, x, y, r, k, c, fc, ec, lab, sub, fs=9)
    ax.text(14.0, -0.5, "4 samples, 3 features → 5 hidden units → 2 classes.\nRow i of every (4, ·) table is still sample i.", fontsize=9, va="top")
    ax.text(0.1, -2.85, "BACKWARD (from the loss to the input; each line uses the one before)", fontsize=11, fontweight="bold", color=RED)
    y = -3.55
    items = [(0.2, 4, 2, "G2 (4,2)", PALE[4], RED, "= (P − Y)/4"), (2.0, 5, 4, "H1ᵀ (5,4)", PALE[0], C0, "dW2 = H1ᵀ@G2"),
             (4.2, 5, 2, "dW2 (5,2)", PALE[4], RED, "shape of W2"), (5.6, 4, 5, "G2@W2ᵀ (4,5)", PALE[4], RED, "pass back"),
             (7.9, 4, 5, "mask (4,5)", "#f3f3f3", GREY, "1 where Z1 > 0"), (10.2, 4, 5, "G1 (4,5)", PALE[4], RED, "= G2@W2ᵀ ⊙ mask"),
             (12.5, 3, 5, "dW1 (3,5)", PALE[4], RED, "= Xᵀ@G1")]
    for x, r, k, lab, fc, ec, sub in items:
        _grid(ax, x, y, r, k, c, fc, ec, lab, sub, fs=9)
    ax.text(14.6, -3.6, "Every gradient has the\nshape of what it belongs to:\ndW2 like W2, dW1 like W1.\n\nThe bias gradients are the\ncolumn sums of G2 and G1.", fontsize=9, va="top")
    plt.show()


def fig_vanishing_chain(L=8):
    fig, ax = plt.subplots(figsize=(11, 3.6))
    layers = np.arange(L, 0, -1)
    sig = 0.25 ** np.arange(L)
    ax.bar(np.arange(L) - 0.2, sig, 0.4, color=C1, label="sigmoid: × at most 0.25 per layer")
    ax.bar(np.arange(L) + 0.2, np.ones(L), 0.4, color=C0, label="ReLU (active units): × 1 per layer")
    ax.set_yscale("log"); ax.set_xticks(np.arange(L)); ax.set_xticklabels([f"layer {l}" for l in layers])
    for i, v in enumerate(sig):
        ax.text(i - 0.2, v * 1.3, f"{v:.0e}" if v < 0.01 else _f(v, 4), ha="center", fontsize=8)
    ax.set_ylabel("largest possible gradient size (log)"); ax.legend(fontsize=9, loc="lower left")
    ax.set_title("Going back from the loss (left) to the first layer (right), the gradient is multiplied once per layer", fontsize=10)
    plt.tight_layout(); plt.show()


# ------------------------------------------------------------------ 02 · regularization
def small_digits(n_train=150, seed=0):
    """A deliberately SMALL training set (150 images) so that a big network overfits; the rest is for validation."""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    X, y = load_digits(return_X_y=True)
    X_tr, X_va, y_tr, y_va = train_test_split(X / 16, y, train_size=n_train, random_state=seed, stratify=y)
    return X_tr, X_va, y_tr, y_va


def train_regularized(weight_decay=0.0, p_drop=0.0, epochs=400, seed=0):
    """Full-batch Adam on small_digits(); returns per-epoch (train loss, val loss, val accuracy). Deterministic on CPU."""
    import torch
    import torch.nn as nn
    X_tr, X_va, y_tr, y_va = small_digits()
    X_tr, X_va = torch.tensor(X_tr, dtype=torch.float32), torch.tensor(X_va, dtype=torch.float32)
    y_tr, y_va = torch.tensor(y_tr), torch.tensor(y_va)
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Dropout(p_drop), nn.Linear(256, 256), nn.ReLU(), nn.Dropout(p_drop), nn.Linear(256, 10))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(); hist = []
    for _ in range(epochs):
        model.train()
        loss = loss_fn(model(X_tr), y_tr)
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            out = model(X_va)
            hist.append((loss.item(), loss_fn(out, y_va).item(), (out.argmax(1) == y_va).float().mean().item()))
    return np.array(hist)


def fig_overfit_curves():
    base = train_regularized()
    reg = train_regularized(weight_decay=1.0, p_drop=0.5)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.0), sharey=True)
    for ax, h, name in zip(axes, [base, reg], ["no regularization", "weight decay 1.0 + dropout 0.5"]):
        ax.plot(h[:, 0], color=C0, lw=2, label="training loss")
        ax.plot(h[:, 1], color=C1, lw=2, label="validation loss")
        best = int(h[:, 1].argmin())
        ax.axvline(best, color=GREY, ls=":", lw=1)
        ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.legend(fontsize=9)
        ax.set_title(f"{name}\nbest val loss {h[best, 1]:.3f} at epoch {best}; val accuracy at the end {h[-1, 2]:.3f}", fontsize=10)
    axes[0].set_ylabel("loss (log)")
    fig.suptitle("A 64 → 256 → 256 → 10 network trained on only 150 digit images; validated on the other 1647", fontsize=10.5)
    plt.tight_layout(); plt.show()


def fig_weight_decay():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 3.8))
    w = np.linspace(-1, 3.5, 300)
    for lam, col in [(0, GREY), (0.5, C0), (2, C1)]:
        total = (w - 2) ** 2 + lam * w ** 2
        a1.plot(w, total, color=col, lw=2, label=f"λ = {lam}: minimum at w = {2 / (1 + lam):.2f}")
        a1.scatter([2 / (1 + lam)], [((2 / (1 + lam)) - 2) ** 2 + lam * (2 / (1 + lam)) ** 2], color=col, zorder=3)
    a1.set_ylim(0, 10); a1.set_xlabel("w"); a1.set_ylabel("(w − 2)² + λ w²"); a1.legend(fontsize=8.5)
    a1.set_title("the penalty pulls the best weight towards 0", fontsize=10)
    t = np.arange(0, 60)
    for lr_lam, col in [(0.01, C0), (0.05, C1)]:
        a2.plot(t, (1 - lr_lam) ** t, color=col, lw=2, label=f"ηλ = {lr_lam}: × {1 - lr_lam} per step")
    a2.set_xlabel("step"); a2.set_ylabel("weight, if the loss gradient were 0"); a2.legend(fontsize=9)
    a2.set_title("\"decay\": every step multiplies each weight by (1 − ηλ)", fontsize=10)
    plt.tight_layout(); plt.show()


def fig_dropout_masks():
    rng = np.random.default_rng(3)
    sizes = [4, 5, 5, 2]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for k, ax in enumerate(axes):
        ax.set_xlim(-0.5, 3.5); ax.set_ylim(-0.6, 5); ax.axis("off")
        keep = [np.ones(sizes[0], bool)] + [rng.random(s) > 0.5 if (k < 3 and 0 < l < 3) else np.ones(s, bool) for l, s in enumerate(sizes) if l > 0]
        pos = [[(l, (5 - s) / 2 + i) for i in range(s)] for l, s in enumerate(sizes)]
        for l in range(3):
            for i, (xa, ya) in enumerate(pos[l]):
                for j, (xb, yb) in enumerate(pos[l + 1]):
                    on = keep[l][i] and keep[l + 1][j]
                    ax.plot([xa, xb], [ya, yb], color="#7a8794" if on else "#e8e8e8", lw=0.8 if on else 0.5, zorder=1)
        for l in range(4):
            for i, (xa, ya) in enumerate(pos[l]):
                ax.add_patch(plt.Circle((xa, ya), 0.18, fc=(PALE[0] if keep[l][i] else "white"), ec=("#33404d" if keep[l][i] else "#cccccc"), zorder=2))
                if not keep[l][i]:
                    ax.text(xa, ya, "×", ha="center", va="center", color=RED, fontsize=11, zorder=3)
        ax.set_title(f"training step {k + 1}: a random half\nof the hidden units is off" if k < 3 else "testing: every unit on\n(no randomness)", fontsize=9.5)
    plt.tight_layout(); plt.show()


def fig_early_stopping(patience=20):
    h = train_regularized()
    val = h[:, 1]; best = int(val.argmin())
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(val, color=C1, lw=2, label="validation loss"); ax.plot(h[:, 0], color=C0, lw=1.5, label="training loss")
    ax.axvline(best, color=C2, lw=1.5); ax.axvspan(best, best + patience, color=HL, alpha=0.6)
    ax.text(best - 2, val[best] * 0.35, f"best epoch {best}:\nsave these weights", color=C2, fontsize=9, ha="right")
    ax.text(best + patience + 4, val[best] * 0.25, f"{patience} epochs with no new best\n(the patience) → stop,\nand go back to epoch {best}'s weights", fontsize=9)
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("loss (log)"); ax.legend(fontsize=9, loc="lower left")
    ax.set_title("Early stopping: keep the weights from the epoch with the lowest validation loss", fontsize=10)
    plt.tight_layout(); plt.show()


def fig_augment():
    from scipy.ndimage import rotate, shift
    from sklearn.datasets import load_digits
    img = load_digits().images[0] / 16
    variants = [("original", img), ("shift right 1", shift(img, (0, 1), order=0)), ("shift down 1", shift(img, (1, 0), order=0)),
                ("rotate +15°", rotate(img, 15, reshape=False, order=1)), ("rotate −15°", rotate(img, -15, reshape=False, order=1)),
                ("+ noise", np.clip(img + np.random.default_rng(0).normal(0, 0.15, img.shape), 0, 1))]
    fig, axes = plt.subplots(1, 6, figsize=(13, 2.6))
    for ax, (t, im) in zip(axes, variants):
        ax.imshow(im, cmap="gray_r", vmin=0, vmax=1); ax.set_title(t, fontsize=9.5); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle("Data augmentation: each training image becomes several, all with the same label \"0\"", fontsize=10)
    plt.tight_layout(); plt.show()


# ------------------------------------------------------------------ 03 · training stability
def fig_init_spread(L=12, width=256):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, width))
    fig, ax = plt.subplots(figsize=(10, 4.0))
    for name, std, col in [("too small: std 0.01", 0.01, GREY), (f"He: std √(2 / n_in) = {np.sqrt(2 / width):.3f}", np.sqrt(2 / width), C0), ("too large: std 0.2", 0.2, C1)]:
        H = X; stds = []
        for _ in range(L):
            H = np.maximum(H @ (rng.normal(size=(width, width)) * std), 0); stds.append(H.std())
        ax.plot(range(1, L + 1), stds, "o-", color=col, label=name)
    ax.set_yscale("log"); ax.set_xlabel("layer"); ax.set_ylabel("spread (std) of the layer's outputs, log")
    ax.set_title(f"A 12-layer ReLU network, {width} units per layer, at the start of training: only He keeps the spread steady", fontsize=10); ax.legend(fontsize=9)
    plt.tight_layout(); plt.show()


BN_X = np.array([[1.0, 50.0, 0.2], [3.0, 70.0, 0.4], [5.0, 60.0, 0.1], [7.0, 40.0, 0.3]])


def fig_bn_vs_ln():
    X = BN_X
    bn = (X - X.mean(0)) / np.sqrt(X.var(0) + 1e-5)
    ln = (X - X.mean(1, keepdims=True)) / np.sqrt(X.var(1, keepdims=True) + 1e-5)
    fig, ax = _canvas(17.5, 3.7, (17.5, 3.8))
    hdr = ["feat 0", "feat 1", "feat 2"]; idx = [f"sample {i}" for i in range(4)]
    _table(ax, 1.3, 0, hdr, [[_f(v, 2) for v in r] for r in X], cw=0.85, index=idx, title="a batch X  (4, 3)", colors={(r, 1): PALE[0] for r in range(4)} | {(2, c): PALE[1] for c in range(3)})
    _table(ax, 6.6, 0, hdr, [[f"{v:.2f}" for v in r] for r in bn], cw=0.85, title="batch norm: each COLUMN → mean 0, std 1", tfs=9.5, colors={(r, 1): PALE[0] for r in range(4)})
    _table(ax, 11.6, 0, hdr, [[f"{v:.2f}" for v in r] for r in ln], cw=0.85, title="layer norm: each ROW → mean 0, std 1", tfs=9.5, colors={(2, c): PALE[1] for c in range(3)})
    ax.text(0.2, -2.4, "Blue: batch norm works down a column. Feature 1 had values 40–70; across the 4 samples its mean is 55 and std 11.2, so 50 → (50 − 55)/11.2 = −0.45.\n"
            "Orange: layer norm works along a row. Sample 2's three values (5, 60, 0.1) are normalized among themselves, whatever the other samples are.", fontsize=9.2, va="top")
    plt.show()


def fig_residual_block():
    fig, ax = plt.subplots(figsize=(12, 3.4)); ax.set_xlim(0, 12); ax.set_ylim(0, 3.4); ax.axis("off")
    _box(ax, 0.9, 1.4, "x", PALE[5], w=1.0)
    _box(ax, 3.4, 1.4, "Linear\n→ ReLU", PALE[0], w=1.8, h=1.0)
    _box(ax, 6.2, 1.4, "Linear", PALE[0], w=1.6, h=0.8)
    ax.add_patch(plt.Circle((8.4, 1.4), 0.35, fc=HL, ec="#33404d", zorder=3)); ax.text(8.4, 1.4, "+", ha="center", va="center", fontsize=15, zorder=4)
    _box(ax, 10.6, 1.4, "out = x + F(x)", PALE[2], w=2.2)
    for a, b in [(1.4, 2.5), (4.3, 5.4), (7.0, 8.05), (8.75, 9.5)]:
        ax.annotate("", xy=(b, 1.4), xytext=(a, 1.4), arrowprops=dict(arrowstyle="-|>", color="#33404d", lw=1.4))
    ax.annotate("", xy=(8.4, 1.75), xytext=(0.9, 1.8), arrowprops=dict(arrowstyle="-|>", color=C1, lw=2, connectionstyle="arc3,rad=-0.35"))
    ax.text(4.7, 3.05, "the skip connection: x is added back unchanged", color=C1, fontsize=10, ha="center", fontweight="bold")
    ax.text(4.8, 0.35, "F(x): the two layers only have to learn a CORRECTION to x", fontsize=9.5, ha="center")
    ax.text(10.6, 0.55, "gradient: ∂out/∂x = 1 + ∂F/∂x\nthe 1 never vanishes", fontsize=9, ha="center", color=RED)
    plt.show()


def fig_clip(max_norm=1.0):
    fig, ax = plt.subplots(figsize=(6.4, 5))
    for g, col, lab in [(np.array([3.0, 4.0]), C1, "g = (3, 4), length 5"), (np.array([-0.3, 0.4]), C0, "g = (−0.3, 0.4), length 0.5")]:
        n = np.linalg.norm(g); gc = g * min(1, max_norm / n)
        ax.annotate("", xy=g, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4, alpha=0.5))
        ax.annotate("", xy=gc, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=2.6))
        ax.text(g[0] + 0.1, g[1], lab + (f"\n→ clipped to ({gc[0]:.1f}, {gc[1]:.1f})" if n > max_norm else "\n→ unchanged"), color=col, fontsize=9)
    t = np.linspace(0, 2 * np.pi, 200); ax.plot(max_norm * np.cos(t), max_norm * np.sin(t), color=GREY, ls="--", lw=1)
    ax.text(0.75, -0.95, f"max_norm = {max_norm}", color=GREY, fontsize=9)
    ax.set_aspect("equal"); ax.set_xlim(-1.5, 5.5); ax.set_ylim(-1.5, 4.8)
    ax.set_title("Gradient clipping: too-long gradients are shrunk\nto length max_norm; the direction stays", fontsize=10)
    plt.tight_layout(); plt.show()


def deep_mlp_run(use_bn, use_res, depth=20, epochs=60, seed=0, lr=0.1):
    """A deep, narrow MLP with sigmoid activations on the full digits data; returns the training loss per epoch."""
    import torch
    import torch.nn as nn
    from sklearn.datasets import load_digits
    X, y = load_digits(return_X_y=True)
    X, y = torch.tensor(X / 16, dtype=torch.float32), torch.tensor(y)
    torch.manual_seed(seed)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(64, 32)
            self.hidden = nn.ModuleList([nn.Linear(32, 32) for _ in range(depth)])
            self.bns = nn.ModuleList([nn.BatchNorm1d(32) if use_bn else nn.Identity() for _ in range(depth)])
            self.out = nn.Linear(32, 10)

        def forward(self, x):
            h = torch.sigmoid(self.inp(x))
            for lin, bn in zip(self.hidden, self.bns):
                f = torch.sigmoid(bn(lin(h)))
                h = h + f if use_res else f
            return self.out(h)
    net = Net(); opt = torch.optim.SGD(net.parameters(), lr=lr); loss_fn = nn.CrossEntropyLoss(); out = []
    for _ in range(epochs):
        for idx in torch.randperm(len(X)).split(128):
            loss = loss_fn(net(X[idx]), y[idx]); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            out.append(loss_fn(net(X), y).item())        # all 1797 images as one batch
    return out


def fig_deep_training():
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for bn, res, col, lab in [(False, False, GREY, "plain"), (True, False, C0, "+ batch norm after every Linear")]:
        ax.plot(deep_mlp_run(bn, res), color=col, lw=2, label=lab)
    ax.axhline(np.log(10), color=GREY, ls=":", lw=1); ax.text(1, np.log(10) * 1.05, "log 10 = 2.30: guessing", fontsize=8.5, color=GREY)
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("training loss (log)"); ax.legend(fontsize=9)
    ax.set_title("A 22-layer sigmoid network on the digits, SGD (η = 0.1): the plain one never gets below guessing", fontsize=10)
    plt.tight_layout(); plt.show()


# ------------------------------------------------------------------ 04 · convolution
CONV_X = np.array([[1, 2, 0, 1], [0, 1, 3, 1], [2, 1, 0, 0], [1, 0, 1, 2]], dtype=float)
CONV_K = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=float)


def _conv(x, k, s=1):
    n, m = x.shape; a, b = k.shape
    out = np.zeros(((n - a) // s + 1, (m - b) // s + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(x[i * s:i * s + a, j * s:j * s + b] * k)
    return out


def fig_conv_slide():
    X, K = CONV_X, CONV_K; Y = _conv(X, K)
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6), gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.9]})
    for p, ax in enumerate(axes[:4]):
        i, j = divmod(p, 2)
        ax.imshow(np.zeros_like(X), cmap="Greys", vmin=0, vmax=1)
        for r in range(4):
            for c in range(4):
                inside = i <= r < i + 3 and j <= c < j + 3
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fc=HL if inside else "white", ec="#9aa3ad"))
                ax.text(c, r - 0.12, _f(X[r, c]), ha="center", va="center", fontsize=11, fontweight="bold" if inside else "normal")
                if inside:
                    ax.text(c, r + 0.27, f"×{_f(K[r - i, c - j])}", ha="center", va="center", fontsize=8, color=C0)
        ax.set_xlim(-0.5, 3.5); ax.set_ylim(3.5, -0.5); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        win = X[i:i + 3, j:j + 3]
        ax.set_title(f"output[{i}, {j}] = sum of the 9 products\n= {_f(np.sum(win * K))}", fontsize=9.5)
    ax = axes[4]
    for r in range(2):
        for c in range(2):
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fc=PALE[2], ec="#9aa3ad")); ax.text(c, r, _f(Y[r, c]), ha="center", va="center", fontsize=12, fontweight="bold")
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(1.5, -0.5); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.set_aspect("equal")
    ax.set_title("the output\n(4 − 3 + 1) × (4 − 3 + 1) = 2 × 2", fontsize=9.5)
    fig.suptitle("A 3×3 kernel (columns 1, 0, −1: 'left minus right') slides over a 4×4 image. Yellow = where it sits; blue = the kernel weight under each pixel", fontsize=10, y=1.06)
    plt.tight_layout(); plt.show()


def fig_padding_stride():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (p, s, title) in zip(axes, [(0, 1, "no padding, stride 1"), (1, 1, "padding 1, stride 1"), (1, 2, "padding 1, stride 2")]):
        n = 5 + 2 * p
        for r in range(n):
            for c in range(n):
                pad = r < p or c < p or r >= n - p or c >= n - p
                ax.add_patch(plt.Rectangle((c, n - 1 - r), 1, 1, fc="#f2f2f2" if pad else PALE[0], ec="#b0b8c0"))
                if pad:
                    ax.text(c + 0.5, n - 0.5 - r, "0", ha="center", va="center", fontsize=8, color=GREY)
        out = (5 + 2 * p - 3) // s + 1
        starts = [k * s for k in range(out)]
        for k, st in enumerate(starts[:2]):
            ax.add_patch(plt.Rectangle((st, n - 3), 3, 3, fc="none", ec=[C1, C3][k], lw=2.5))
        ax.set_xlim(-0.2, 7.2); ax.set_ylim(-0.2, 7.2); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{title}\noutput size ⌊(5 + 2·{p} − 3)/{s}⌋ + 1 = {out}", fontsize=10)
    fig.suptitle("A 5×5 image (blue), a 3×3 kernel. Orange: the first position; purple: the next one (moved by the stride). Grey zeros = padding", fontsize=10)
    plt.tight_layout(); plt.show()


def _slab(ax, x, y, w, h, depth, fc, ec, label=None, sub=None, dx=0.18, dy=0.14):
    for k in range(depth):
        ax.add_patch(plt.Rectangle((x + k * dx, y + k * dy), w, h, fc=fc, ec=ec, lw=1))
    if label:
        ax.text(x + w / 2 + depth * dx / 2, y + h + depth * dy + 0.15, label, ha="center", va="bottom", fontsize=9.5, fontweight="bold", family="monospace")
    if sub:
        ax.text(x + w / 2, y - 0.2, sub, ha="center", va="top", fontsize=8.5, color="#3d4652")


def fig_channels():
    fig, ax = plt.subplots(figsize=(14, 4.2)); ax.set_xlim(0, 14); ax.set_ylim(-1.2, 4); ax.axis("off")
    _slab(ax, 0.3, 0.3, 2.4, 2.4, 3, "#f6d5d5", RED, "input (3, 32, 32)", "3 channels (R, G, B)\nof 32 × 32 pixels")
    ax.text(3.9, 1.5, "*", fontsize=22, ha="center", va="center")
    _slab(ax, 4.6, 1.0, 0.9, 0.9, 3, PALE[0], C0, "one kernel (3, 3, 3)", "depth = in_channels:\n3 × 3 × 3 = 27 weights + 1 bias")
    ax.text(6.6, 1.5, "→", fontsize=22, ha="center", va="center")
    _slab(ax, 7.2, 0.3, 2.4, 2.4, 1, PALE[2], C2, "one map (1, 30, 30)", "27 products summed\nat every position")
    ax.text(10.6, 1.5, "16 kernels\n→", fontsize=12, ha="center", va="center")
    _slab(ax, 11.3, 0.3, 2.0, 2.0, 8, PALE[2], C2, "output (16, 30, 30)", "out_channels = 16 maps", dx=0.12, dy=0.1)
    ax.text(7.0, -1.1, "nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3): weight shape (16, 3, 3, 3), bias (16,): 16·3·3·3 + 16 = 448 parameters",
            ha="center", fontsize=9.5, fontweight="bold")
    plt.show()


POOL_X = np.array([[1, 3, 2, 0], [4, 2, 1, 5], [0, 1, 3, 2], [2, 6, 1, 1]], dtype=float)


def fig_maxpool():
    X = POOL_X; Y = X.reshape(2, 2, 2, 2).max(axis=(1, 3))
    fig, ax = _canvas(11, 2.9, (11, 3.0))
    col = {(r, c): PALE[(r // 2) * 2 + c // 2] for r in range(4) for c in range(4)}
    _table(ax, 0.5, 0, ["", "", "", ""], [[_f(v) for v in r] for r in X], cw=0.7, colors=col, title="4 × 4 input", bold_hdr=False)
    ax.text(3.6, -1.1, "max of each\n2 × 2 window →", fontsize=10, va="center")
    _table(ax, 5.8, 0, ["", ""], [[_f(v) for v in r] for r in Y], cw=0.8, colors={(r, c): PALE[r * 2 + c] for r in range(2) for c in range(2)}, title="2 × 2 output", bold_hdr=False)
    ax.text(7.7, -0.3, "nn.MaxPool2d(2): window 2, stride 2.\nNo weights to learn.\nHalves the height and the width;\nkeeps the strongest response\nin each window.", fontsize=9.3, va="top")
    plt.show()


def fig_receptive_field():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    for ax, (L, size) in zip(axes, [(1, 3), (2, 5), (3, 7)]):
        n = 9
        for r in range(n):
            for c in range(n):
                lo = (n - size) // 2
                inside = lo <= r < lo + size and lo <= c < lo + size
                ax.add_patch(plt.Rectangle((c, r), 1, 1, fc=PALE[1] if inside else "white", ec="#c0c6cc"))
        ax.add_patch(plt.Rectangle((4, 4), 1, 1, fc=C1, ec="k"))
        ax.set_xlim(0, 9); ax.set_ylim(0, 9); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"after {L} layer{'s' if L > 1 else ''} of 3×3 convolutions:\none output pixel (dark) sees {size} × {size} input pixels", fontsize=10)
    fig.suptitle("Receptive field: each extra 3×3 layer widens what one pixel 'sees' by 2 (1 on each side)", fontsize=10)
    plt.tight_layout(); plt.show()


def mnist_image(i=0):
    from torchvision import datasets, transforms
    ds = datasets.MNIST(root=MNIST_ROOT, train=True, download=True, transform=transforms.ToTensor())
    img, label = ds[i]
    return img[0].numpy(), label


def fig_edge_maps():
    import torch
    import torch.nn.functional as F
    img, label = mnist_image(0)
    kernels = {"vertical edges\n[[1,0,−1]] × 3 rows": CONV_K, "horizontal edges\n(the same, turned)": CONV_K.T}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    axes[0].imshow(img, cmap="gray_r"); axes[0].set_title(f"an MNIST image (label {label}), 28 × 28", fontsize=10)
    for ax, (t, k) in zip(axes[1:], kernels.items()):
        out = F.conv2d(torch.tensor(img)[None, None], torch.tensor(k, dtype=torch.float32)[None, None])[0, 0].numpy()
        ax.imshow(out, cmap="RdBu", vmin=-2, vmax=2); ax.set_title(f"after the kernel for {t}\nshape {out.shape}", fontsize=9.5)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle("Red/blue = strong response of opposite sign: the kernel 'lights up' where its pattern is in the image", fontsize=10)
    plt.tight_layout(); plt.show()


# ------------------------------------------------------------------ 05 · CNNs in PyTorch
def make_small_cnn():
    import torch.nn as nn
    return nn.Sequential(nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Flatten(), nn.Linear(16 * 7 * 7, 10))


def mnist_sets():
    from torchvision import datasets, transforms
    tr = datasets.MNIST(root=MNIST_ROOT, train=True, download=True, transform=transforms.ToTensor())
    te = datasets.MNIST(root=MNIST_ROOT, train=False, download=True, transform=transforms.ToTensor())
    return tr, te


def train_mnist_cnn(n_train=60000, epochs=2, seed=0):
    """The recipe of 05 §4 on CPU: all 60,000 training images, 2 epochs, Adam 1e-3, batches of 128. Deterministic."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset
    tr, _ = mnist_sets()
    torch.manual_seed(seed)
    model = make_small_cnn()
    loader = DataLoader(tr if n_train >= len(tr) else Subset(tr, range(n_train)), batch_size=128, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3); loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb); opt.zero_grad(); loss.backward(); opt.step()
    return model


def _load_cnn():
    import torch
    model = make_small_cnn()
    try:
        model.load_state_dict(torch.load("mnist_cnn.pt"))
    except Exception:
        model = train_mnist_cnn()
    return model.eval()


def fig_cnn_shapes():
    fig, ax = plt.subplots(figsize=(16, 4.4)); ax.set_xlim(0, 16); ax.set_ylim(-1.6, 4.2); ax.axis("off")
    blocks = [(0.2, 2.4, 1, "#e4e4e4", GREY, "input\n(1, 28, 28)"), (3.3, 2.4, 4, PALE[0], C0, "conv 3×3, pad 1\n→ ReLU\n(8, 28, 28)"),
              (6.9, 1.2, 4, PALE[1], C1, "max pool 2\n(8, 14, 14)"), (9.3, 1.2, 7, PALE[0], C0, "conv 3×3, pad 1\n→ ReLU\n(16, 14, 14)"),
              (12.0, 0.6, 7, PALE[1], C1, "max pool 2\n(16, 7, 7)")]
    for x, s_, d, fc, ec, t in blocks:
        _slab(ax, x, 0.2, s_, s_, d, fc, ec, dx=0.14, dy=0.11)
        ax.text(x + s_ / 2 + d * 0.07, -0.15, t, ha="center", va="top", fontsize=9)
    ax.add_patch(plt.Rectangle((14.0, 0.2), 0.3, 3.0, fc=PALE[3], ec=C3)); ax.text(14.15, -0.15, "flatten\n784", ha="center", va="top", fontsize=9)
    ax.add_patch(plt.Rectangle((15.2, 1.1), 0.3, 1.2, fc=PALE[2], ec=C2)); ax.text(15.35, 0.85, "Linear\n10 logits", ha="center", va="top", fontsize=9)
    for a_, b_ in [(2.7, 3.2), (6.35, 6.8), (8.75, 9.2), (11.5, 11.9), (13.5, 13.9), (14.4, 15.1)]:
        ax.annotate("", xy=(b_, 1.4), xytext=(a_, 1.4), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.3))
    ax.text(8.0, 4.05, "(channels, height, width) after each layer, for ONE image. A batch adds a first dimension: (N, C, H, W)", ha="center", fontsize=10, fontweight="bold")
    plt.show()


def fig_cnn_filters():
    import torch
    model = _load_cnn()
    img, label = mnist_image(0)
    W = model[0].weight.detach().numpy()[:, 0]
    with torch.no_grad():
        maps = torch.relu(model[0](torch.tensor(img)[None, None]))[0].numpy()
    fig, axes = plt.subplots(2, 8, figsize=(15, 4.2))
    for k in range(8):
        axes[0, k].imshow(W[k], cmap="RdBu", vmin=-np.abs(W).max(), vmax=np.abs(W).max()); axes[0, k].set_title(f"kernel {k}", fontsize=9)
        axes[1, k].imshow(maps[k], cmap="magma"); axes[1, k].set_title(f"map {k}", fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle(f"Top: the 8 LEARNED 3×3 kernels of the first layer (red +, blue −). Bottom: what each produces (after ReLU) for the image of a {label}", fontsize=10)
    plt.tight_layout(); plt.show()


def fig_cnn_prediction(i=0):
    import torch
    model = _load_cnn()
    _, te = mnist_sets()
    img, label = te[i]
    with torch.no_grad():
        p = torch.softmax(model(img[None]), dim=1)[0].numpy()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.3), gridspec_kw={"width_ratios": [1, 2.3]})
    a1.imshow(img[0], cmap="gray_r"); a1.set_xticks([]); a1.set_yticks([]); a1.grid(False); a1.set_title(f"a test image (true label {label})", fontsize=10)
    bars = a2.bar(range(10), p, color=[C1 if k == p.argmax() else C0 for k in range(10)])
    for b_, v in zip(bars, p):
        if v > 0.005:
            a2.text(b_.get_x() + b_.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    a2.set_xticks(range(10)); a2.set_ylim(0, 1.12); a2.set_xlabel("digit"); a2.set_title(f"the CNN's probabilities: {p.argmax()} with {p.max():.3f}", fontsize=10)
    plt.tight_layout(); plt.show()
