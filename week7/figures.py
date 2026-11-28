"""Picture helpers and shared models for Week 7. Each fig_ function draws one figure from the lessons.

You don't need to read this file to follow the lessons. Shared tools (colours, table pictures, flow maps) are in fig_common.py.
The models the lessons load (ae8.pt, ae2.pt, ae32.pt, vae2.pt, ae_zeros.pt) are made by the train_* functions below,
with exactly the code shown in the notebooks, so the numbers match.
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, Polygon
from fig_common import *  # noqa: F401,F403
from fig_common import _arrow, _canvas, _table  # noqa: F401

PALE = ["#d7e6f6", "#fde2c8", "#dcefe3", "#ebe0f3", "#f6d5d5", "#e4e4e4"]
RED = "#c0392b"
INK = "#1b2430"
SUB = "#3d4652"
FROZEN_FC, FROZEN_EC = "#e3e6ea", "#8a939d"
TRAIN_FC, TRAIN_EC = "#fde2c8", "#c0611a"
MNIST_ROOT = "../week5/data"          # MNIST was downloaded in Week 5; nothing is downloaded twice
CNN_PATH = "../week6/mnist_cnn.pt"    # the CNN you trained in Week 6 · 05


def _f(v, d=3):
    s = f"{v:.{d}f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _box(ax, x, y, text, fc, w=1.5, h=0.75, fs=10, ec="#33404d", bold=False, lw=1.2):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=lw, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=4, fontweight="bold" if bold else "normal", color=INK)


def _arr(ax, a, b, color="#5b6675", lw=1.5, text=None, fs=9, rad=0.0, style="-|>", ls="-", tcolor=None, toff=(0, 0.12), zorder=2):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle=style, color=color, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0,
                                                    connectionstyle=f"arc3,rad={rad}"), zorder=zorder)
    if text:
        ax.text((a[0] + b[0]) / 2 + toff[0], (a[1] + b[1]) / 2 + toff[1], text, ha="center", va="bottom", fontsize=fs, color=tcolor or color)


def _slab(ax, x, y, w, h, depth, fc, ec, dx=0.14, dy=0.11, lw=1.0):
    """A stack of `depth` sheets (a feature map with channels), front sheet bottom-left at (x, y)."""
    for k in reversed(range(depth)):
        ax.add_patch(plt.Rectangle((x + k * dx, y + k * dy), w, h, fc=fc, ec=ec, lw=lw, zorder=2 + (depth - k) * 0.01))


def _img(ax, im, title=None, fs=9, cmap="gray_r", vmin=0, vmax=1):
    ax.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True); s.set_color("#c0c6cc")
    if title:
        ax.set_title(title, fontsize=fs)


def _off(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.axis("off")


# ================================================================== data and models
def mnist_tensors():
    """All of MNIST as tensors: X_tr (60000, 784), y_tr (60000,), X_te (10000, 784), y_te (10000,). Pixels in [0, 1]."""
    import torch
    from torchvision import datasets
    tr = datasets.MNIST(root=MNIST_ROOT, train=True, download=True)
    te = datasets.MNIST(root=MNIST_ROOT, train=False, download=True)
    f = lambda d: d.data.float().div(255).reshape(-1, 784)
    return f(tr), tr.targets.clone(), f(te), te.targets.clone()


def make_small_cnn():
    import torch.nn as nn
    return nn.Sequential(nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Flatten(), nn.Linear(16 * 7 * 7, 10))


def load_mnist_cnn():
    """The Week 6 CNN (98% on MNIST), trained on 60,000 labelled MNIST digits."""
    import torch
    model = make_small_cnn()
    model.load_state_dict(torch.load(CNN_PATH))
    return model.eval()


def digits28():
    """scikit-learn's 1797 digits (8 × 8, values 0..16), made to look like MNIST: scaled to [0, 1], enlarged to 20 × 20, padded to 28 × 28.
    Returns X (1797, 1, 28, 28) float32 and y (1797,)."""
    import torch
    import torch.nn.functional as F
    from sklearn.datasets import load_digits
    X, y = load_digits(return_X_y=True)
    X = torch.tensor(X / 16, dtype=torch.float32).reshape(-1, 1, 8, 8)
    X = F.interpolate(X, size=(20, 20), mode="bilinear", align_corners=False)   # 8 × 8 -> 20 × 20
    X = F.pad(X, (4, 4, 4, 4))                                                  # a 4-pixel frame of 0s, like MNIST
    return X, torch.tensor(y)


def few_labels(y, per_class, seed=0):
    """Indices of `per_class` labelled examples of every class (the rest is the test set)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    y = np.asarray(y)
    idx = np.concatenate([[i for i in order if y[i] == c][:per_class] for c in range(10)])
    rest = np.setdiff1d(np.arange(len(y)), idx)
    return idx, rest


def train_classifier(model, X, y, epochs=100, lr=1e-3, params=None, batch_size=50, seed=1):
    """The usual loop, on a small labelled set held in memory."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(seed)
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(params if params is not None else model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


def accuracy(model, X, y):
    import torch
    model.eval()
    with torch.no_grad():
        return (model(X).argmax(dim=1) == y).float().mean().item()


def transfer_runs(per_class, strategy):
    """One run of 01 §4: returns the accuracy on the digits that were NOT used for training."""
    import copy
    import torch
    import torch.nn as nn
    X, y = digits28()
    idx, rest = few_labels(y, per_class)
    Xs, ys, Xr, yr = X[idx], y[idx], X[rest], y[rest]
    if strategy == "pretrained, no training":
        return accuracy(load_mnist_cnn(), Xr, yr)
    if strategy == "from scratch":
        torch.manual_seed(0)
        model = train_classifier(make_small_cnn(), Xs, ys)
    elif strategy == "feature extraction":
        model = copy.deepcopy(load_mnist_cnn())
        for p in model.parameters():
            p.requires_grad = False
        torch.manual_seed(0)
        model[7] = nn.Linear(784, 10)
        model = train_classifier(model, Xs, ys, params=model[7].parameters())
    else:                                              # fine-tuning: everything, small learning rate
        model = copy.deepcopy(load_mnist_cnn())
        model = train_classifier(model, Xs, ys, lr=3e-4)
    return accuracy(model, Xr, yr)


TRANSFER_JSON = "transfer_results.json"
STRATEGIES = ["from scratch", "pretrained, no training", "feature extraction", "fine-tuning"]
PER_CLASS = [1, 3, 5, 10, 30]


def transfer_table():
    if os.path.exists(TRANSFER_JSON):
        with open(TRANSFER_JSON) as fh:
            return json.load(fh)
    res = {s: [round(transfer_runs(n, s), 3) for n in PER_CLASS] for s in STRATEGIES}
    with open(TRANSFER_JSON, "w") as fh:
        json.dump(res, fh)
    return res


def make_ae(k):
    """The autoencoder of 02: 784 -> 128 -> k -> 128 -> 784."""
    import torch.nn as nn

    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, k))
            self.decoder = nn.Sequential(nn.Linear(k, 128), nn.ReLU(), nn.Linear(128, 784), nn.Sigmoid())

        def forward(self, x):
            return self.decoder(self.encoder(x))
    return AE()


def train_ae(k, X=None, epochs=5, seed=0):
    """02 §4's recipe: Adam 1e-3, batches of 128, mean squared error between the reconstruction and the input."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    if X is None:
        X = mnist_tensors()[0]
    torch.manual_seed(seed)
    model = make_ae(k)
    loader = DataLoader(TensorDataset(X), batch_size=128, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        for (xb,) in loader:
            loss = loss_fn(model(xb), xb)              # the target is the input itself
            opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


def load_ae(k, path=None):
    import torch
    path = path or f"ae{k}.pt"
    model = make_ae(k)
    try:
        model.load_state_dict(torch.load(path))
    except Exception:
        model = train_ae(k)
    return model.eval()


def make_vae(k=2):
    import torch
    import torch.nn as nn

    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(784, 256), nn.ReLU())
            self.to_mu = nn.Linear(256, k)
            self.to_logvar = nn.Linear(256, k)
            self.decoder = nn.Sequential(nn.Linear(k, 256), nn.ReLU(), nn.Linear(256, 784), nn.Sigmoid())

        def forward(self, x):
            h = self.encoder(x)
            mu, logvar = self.to_mu(h), self.to_logvar(h)
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            return self.decoder(z), mu, logvar
    return VAE()


def vae_loss_fn(x, x_hat, mu, logvar):
    import torch
    import torch.nn.functional as F
    recon = F.binary_cross_entropy(x_hat, x, reduction="sum") / len(x)
    kl = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp()) / len(x)
    return recon + kl, recon, kl


def train_vae(epochs=10, seed=0, beta=1.0):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    X = mnist_tensors()[0]
    torch.manual_seed(seed)
    model = make_vae(2)
    loader = DataLoader(TensorDataset(X), batch_size=128, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        model.train()
        for (xb,) in loader:
            x_hat, mu, logvar = model(xb)
            _, recon, kl = vae_loss_fn(xb, x_hat, mu, logvar)
            loss = recon + beta * kl
            opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


def load_vae(path="vae2.pt"):
    import torch
    model = make_vae(2)
    try:
        model.load_state_dict(torch.load(path))
    except Exception:
        model = train_vae()
    return model.eval()


def zeros_data():
    """04's factory: every MNIST 0 is a "good part". Returns train zeros, test zeros, and the 100 scratched test zeros."""
    X_tr, y_tr, X_te, y_te = mnist_tensors()
    Z_tr, Z_te = X_tr[y_tr == 0], X_te[y_te == 0]
    good = Z_te[100:]                                   # 880 good parts to test on
    scratched = add_scratches(Z_te[:100])               # 100 parts with a scratch
    others = X_te[y_te != 0][:100]                      # 100 wrong parts (other digits)
    return Z_tr, good, scratched, others


def add_scratches(X, seed=0):
    """A bright 3 × 9 bar ("scratch") at a random place on each image."""
    rng = np.random.default_rng(seed)
    out = X.clone().reshape(-1, 28, 28)
    for i in range(len(out)):
        r, c = rng.integers(6, 20), rng.integers(5, 15)
        out[i, r:r + 3, c:c + 9] = 1.0
    return out.reshape(-1, 784)


def train_ae_zeros(epochs=20):
    return train_ae(8, X=zeros_data()[0], epochs=epochs)


def load_ae_zeros(path="ae_zeros.pt"):
    import torch
    model = make_ae(8)
    try:
        model.load_state_dict(torch.load(path))
    except Exception:
        model = train_ae_zeros()
    return model.eval()


def recon_scores(model, X):
    import torch
    with torch.no_grad():
        return ((model(X) - X) ** 2).mean(dim=1).numpy()


# ================================================================== 00 · the week
def fig_week7_map():
    boxes = {"00": (13, 80, "00 review", "Week 6, and your notes", "grey"),
             "01": (40, 80, "01 transfer learning", "reuse a trained network", "blue"),
             "02": (70, 80, "02 autoencoders", "compress, then rebuild", "green"),
             "03": (70, 45, "03 the VAE", "a latent space you can sample", "purple"),
             "04": (40, 45, "04 anomaly detection", "flag what doesn't fit", "orange"),
             "05": (13, 45, "05 · 06 practice", "paper exam + quiz", "grey"),
             }
    arrows = [("00", "01", ""), ("01", "02", "an encoder is a\nfeature extractor"), ("02", "03", "make the code\na distribution"),
              ("03", "04", ""), ("02", "04", "large reconstruction\nerror = suspicious"), ("04", "05", "")]
    notes = [(3, 20, "The thread of the week: a network's middle layers hold a compact description of the input (its features, or code).\n"
                     "01 reuses that description for a new task; 02 learns one without labels; 03 makes it something you can sample from;\n"
                     "04 notices when an input can't be described well. Domain shift (01 §2) and anomaly detection (04) are your lab's core topics.")]
    flow(boxes, arrows, size=(13, 6.4), title="Week 7: what a network's middle layers know, and three ways to use it", notes=notes)


def fig_maxpool_reshape():
    """Your 04 §5 note, drawn: reshape (4, 4) -> (2, 2, 2, 2), then max over axes 1 and 3."""
    x = np.array([[1, 3, 2, 0], [4, 2, 1, 1], [0, 1, 5, 2], [2, 0, 3, 4]])
    win = [[PALE[0], PALE[1]], [PALE[2], PALE[3]]]
    edge = [[C0, C1], [C2, C3]]
    fig, ax = plt.subplots(figsize=(14, 4.6)); ax.set_xlim(0, 14); ax.set_ylim(-0.6, 4.6); _off(ax)
    # left: the 4 × 4 image coloured by window
    for r in range(4):
        for c in range(4):
            ax.add_patch(plt.Rectangle((0.4 + c * 0.7, 3.4 - r * 0.7), 0.7, 0.7, fc=win[r // 2][c // 2], ec="#9aa3ad"))
            ax.text(0.75 + c * 0.7, 3.75 - r * 0.7, str(x[r, c]), ha="center", va="center", fontsize=12)
    ax.text(1.8, 4.3, "x   shape (4, 4)", ha="center", fontsize=11, fontweight="bold", family="monospace")
    for r in range(4):
        ax.text(0.25, 3.75 - r * 0.7, f"r={r}", ha="right", va="center", fontsize=8, color=SUB)
    ax.text(1.8, 0.45, "pixel x[r, c] lives in window (I, J) = (r // 2, c // 2)\nand at place (a, b) = (r % 2, c % 2) inside it", ha="center", va="top", fontsize=9, color=SUB)
    _arr(ax, (3.5, 2.3), (4.6, 2.3), text="reshape(2, 2, 2, 2)", fs=9.5, tcolor=INK)
    # middle: the four windows as x4[I, :, J, :]
    for I in range(2):
        for J in range(2):
            ox, oy = 5.0 + J * 1.9, 2.9 - I * 1.85
            for a in range(2):
                for b in range(2):
                    ax.add_patch(plt.Rectangle((ox + b * 0.62, oy - a * 0.62), 0.62, 0.62, fc=win[I][J], ec=edge[I][J], lw=1.2))
                    ax.text(ox + b * 0.62 + 0.31, oy - a * 0.62 + 0.31, str(x[2 * I + a, 2 * J + b]), ha="center", va="center", fontsize=11)
            ax.text(ox + 0.62, oy + 0.75, f"x4[{I}, :, {J}, :]", ha="center", fontsize=9, family="monospace", color=edge[I][J], fontweight="bold")
    ax.text(6.9, 4.3, "x4   shape (2, 2, 2, 2) = (I, a, J, b)", ha="center", fontsize=11, fontweight="bold", family="monospace")
    ax.text(6.9, 0.05, "axes 1 and 3 (a, b) move INSIDE a window;\naxes 0 and 2 (I, J) choose the window", ha="center", va="top", fontsize=9, color=SUB)
    _arr(ax, (9.0, 2.3), (10.3, 2.3), text=".max(axis=(1, 3))", fs=9.5, tcolor=INK)
    # right: the result
    out = x.reshape(2, 2, 2, 2).max(axis=(1, 3))
    for I in range(2):
        for J in range(2):
            ax.add_patch(plt.Rectangle((10.8 + J * 0.9, 2.65 - I * 0.9), 0.9, 0.9, fc=win[I][J], ec=edge[I][J], lw=1.6))
            ax.text(11.25 + J * 0.9, 3.1 - I * 0.9, str(out[I, J]), ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(11.7, 4.3, "result  (2, 2)", ha="center", fontsize=11, fontweight="bold", family="monospace")
    ax.text(11.7, 0.6, "one number per window:\nthe largest of its 4", ha="center", va="top", fontsize=9, color=SUB)
    ax.set_title("Max pooling with reshape: give every pixel a window address, then take the max over the within-window axes", fontsize=11, fontweight="bold")
    plt.show()


# ================================================================== 01 · transfer learning
def _cnn_row(ax, y, status, title, note, scale=1.0):
    """The Week 6 CNN as a row of layer blocks. status: list of 'frozen' / 'train' / 'random' for (conv1, conv2, linear)."""
    col = {"frozen": (FROZEN_FC, FROZEN_EC), "train": (TRAIN_FC, TRAIN_EC), "copied": (PALE[0], C0), "random": ("white", "#9aa3ad")}
    ax.add_patch(plt.Rectangle((0.3, y + 0.25), 0.9, 0.9, fc="#f4f4f4", ec="#9aa3ad")); ax.text(0.75, y + 0.7, "image\n28×28", ha="center", va="center", fontsize=7.5)
    blocks = [(1.8, 0.95, 4, "conv1\n80", status[0]), (3.7, 0.75, 6, "conv2\n1,168", status[1])]
    for x, s, d, t, st in blocks:
        fc, ec = col[st]
        _slab(ax, x, y + 0.2, s, s, d, fc, ec, dx=0.09, dy=0.07)
        ax.text(x + s / 2 + d * 0.045, y - 0.05, t, ha="center", va="top", fontsize=8)
        if st == "frozen":
            ax.text(x + s / 2 + d * 0.045, y + 0.2 + s + d * 0.07 + 0.05, "frozen", ha="center", va="bottom", fontsize=7.5, color=FROZEN_EC, fontweight="bold")
    fc, ec = col[status[2]]
    ax.add_patch(plt.Rectangle((5.75, y + 0.15), 0.32, 1.25, fc=fc, ec=ec, lw=1.2)); ax.text(5.91, y - 0.05, "linear\n7,850", ha="center", va="top", fontsize=8)
    if status[2] == "frozen":
        ax.text(5.91, y + 1.45, "frozen", ha="center", va="bottom", fontsize=7.5, color=FROZEN_EC, fontweight="bold")
    for a, b in [(1.25, 1.75), (3.15, 3.65), (4.95, 5.7)]:
        _arr(ax, (a, y + 0.7), (b, y + 0.7), lw=1.1)
    ax.text(6.4, y + 1.05, title, fontsize=10, fontweight="bold", va="center", color=INK)
    ax.text(6.4, y + 0.45, note, fontsize=8.6, va="center", color=SUB)


def fig_transfer_strategies():
    fig, ax = plt.subplots(figsize=(13, 7.8)); ax.set_xlim(0, 13); ax.set_ylim(-0.6, 9.9); _off(ax)
    _cnn_row(ax, 7.9, ["copied", "copied", "copied"], "source model: the Week 6 CNN", "trained on 60,000 labelled MNIST digits (98% on MNIST)")
    ax.text(0.2, 6.75, "copy all 9,098 weights, then pick what may change on the 30 labelled target digits:", fontsize=9.5, color=INK, fontweight="bold")
    _cnn_row(ax, 5.2, ["random", "random", "train"], "(a) from scratch", "random start; all 9,098 weights train on the 30 images\n(no copying: the baseline)")
    _cnn_row(ax, 2.9, ["frozen", "frozen", "train"], "(b) feature extraction", "conv layers frozen (requires_grad = False);\na NEW last layer trains: 7,850 weights")
    _cnn_row(ax, 0.6, ["train", "train", "train"], "(c) fine-tuning", "start from the copied weights; all 9,098 train,\nwith a small learning rate (3e-4 instead of 1e-3)")
    for k, (fc, ec, t) in enumerate([(PALE[0], C0, "copied from the source"), (FROZEN_FC, FROZEN_EC, "copied, frozen: never changes"),
                                      (TRAIN_FC, TRAIN_EC, "trains on the target data"), ("white", "#9aa3ad", "random start")]):
        ax.add_patch(plt.Rectangle((0.3 + k * 3.2, -0.45), 0.35, 0.25, fc=fc, ec=ec)); ax.text(0.75 + k * 3.2, -0.33, t, fontsize=8.5, va="center")
    ax.set_title("Three ways to use 30 labelled images of a new dataset", fontsize=11, fontweight="bold")
    plt.show()


def fig_two_domains():
    import torch
    X_tr, y_tr, X_te, y_te = mnist_tensors()
    D, yd = digits28()
    fig, axes = plt.subplots(2, 10, figsize=(14, 3.6))
    for c in range(10):
        i = int(np.flatnonzero(y_te.numpy() == c)[0]); j = int(np.flatnonzero(yd.numpy() == c)[0])
        _img(axes[0, c], X_te[i].reshape(28, 28).numpy(), f"{c}")
        _img(axes[1, c], D[j, 0].numpy(), f"{c}")
    axes[0, 0].set_ylabel("source:\nMNIST", fontsize=9.5, rotation=0, ha="right", va="center", labelpad=10)
    axes[1, 0].set_ylabel("target:\nsklearn\ndigits", fontsize=9.5, rotation=0, ha="right", va="center", labelpad=10)
    cnn = load_mnist_cnn()
    a_src = accuracy(cnn, X_te.reshape(-1, 1, 28, 28), y_te); a_tgt = accuracy(cnn, D, yd)
    fig.suptitle(f"Same ten classes, different look: thick handwriting (top) vs blurry, enlarged 8×8 scans (bottom).\n"
                 f"The Week 6 CNN gets {a_src:.3f} of the MNIST test set right, but only {a_tgt:.3f} of the sklearn digits.", fontsize=10.5)
    plt.tight_layout(); plt.show()


def fig_conv1_still_works():
    """The first layer's 8 edge detectors applied to a target image: strokes are still found."""
    import torch
    cnn = load_mnist_cnn()
    D, yd = digits28()
    X_tr, y_tr, X_te, y_te = mnist_tensors()
    imgs = [(X_te[3].reshape(1, 1, 28, 28), f"MNIST (a {int(y_te[3])})"), (D[0:1], f"sklearn (a {int(yd[0])})")]
    fig, axes = plt.subplots(2, 6, figsize=(14, 4.6))
    with torch.no_grad():
        for r, (im, name) in enumerate(imgs):
            maps = torch.relu(cnn[0](im))[0].numpy()
            _img(axes[r, 0], im[0, 0].numpy(), name, fs=9.5)
            for k in range(5):
                _img(axes[r, k + 1], maps[k], f"feature map {k}" if r == 0 else None, cmap="magma", vmin=0, vmax=max(maps[k].max(), 1e-6))
    fig.suptitle("Layer 1 (conv1, frozen, from MNIST) on both domains: the same kernels still light up along the strokes of the new images.\n"
                 "Early layers learn general things (edges); it's the LAST layer, which maps these features to digits, that is tuned to MNIST.", fontsize=10.5)
    plt.tight_layout(); plt.show()


def fig_transfer_curves():
    res = transfer_table()
    fig, ax = plt.subplots(figsize=(10, 5.2))
    sty = {"from scratch": ("#9aa3ad", "o", "--"), "pretrained, no training": (C3, "s", ":"), "feature extraction": (C0, "D", "-"), "fine-tuning": (C1, "^", "-")}
    x = np.array(PER_CLASS) * 10
    for s in STRATEGIES:
        col, mk, ls = sty[s]
        ax.plot(x, res[s], marker=mk, ls=ls, color=col, lw=2, ms=7, label=s)
        if s != "pretrained, no training":
            ax.text(x[-1] * 1.08, res[s][-1] + {"fine-tuning": 0.012, "feature extraction": -0.004, "from scratch": -0.012}[s], f"{res[s][-1]:.3f}", fontsize=9, color=col, va="center")
    for i in (0, 2):
        ax.annotate(f"{res['fine-tuning'][i]:.3f} vs {res['from scratch'][i]:.3f}", xy=(x[i], res["fine-tuning"][i]), xytext=(x[i] * 1.35, res["fine-tuning"][i] + 0.06),
                    fontsize=8.5, color=C1, arrowprops=dict(arrowstyle="-", color=C1, lw=0.8))
    ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels([f"{v}\n({v // 10}/class)" for v in x])
    ax.set_xlabel("labelled target images used for training"); ax.set_ylabel("accuracy on the other sklearn digits")
    ax.set_ylim(0.5, 1.02); ax.set_xlim(8, 420); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=9, frameon=False)
    ax.set_title("The fewer labels you have, the more the copied weights are worth", fontsize=11, fontweight="bold")
    plt.tight_layout(); plt.show()


def fig_feature_space():
    """2-D PCA pictures of the sklearn digits: raw pixels vs the frozen CNN's 784 features."""
    import torch
    from sklearn.decomposition import PCA
    cnn = load_mnist_cnn()
    D, yd = digits28()
    with torch.no_grad():
        feats = cnn[:7](D).numpy()                      # everything except the last layer: (1797, 784)
    raw = D.reshape(len(D), -1).numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, (Z, t) in zip(axes, [(raw, "raw pixels (784 numbers per image)"), (feats, "features from the frozen MNIST layers (784 numbers per image)")]):
        P = PCA(2).fit_transform(Z)
        for c in range(10):
            m = yd.numpy() == c
            ax.scatter(P[m, 0], P[m, 1], s=7, color=plt.cm.tab10(c), alpha=0.75)
            ax.text(*np.median(P[m], axis=0), str(c), fontsize=13, fontweight="bold", color=plt.cm.tab10(c), ha="center", va="center",
                    bbox=dict(fc="white", ec=plt.cm.tab10(c), boxstyle="circle,pad=0.2", alpha=0.9))
        ax.set_title(t, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    fig.suptitle("The sklearn digits, squashed to 2-D with PCA (Week 4). Left: classes overlap. Right: after the MNIST layers they sit in clearer groups,\n"
                 "so even a linear last layer (feature extraction) separates them well. That's what 'useful features' means.", fontsize=10.5)
    plt.tight_layout(); plt.show()


def fig_dann():
    fig, ax = plt.subplots(figsize=(14, 6.6)); ax.set_xlim(0, 14); ax.set_ylim(-0.6, 6.6); _off(ax)
    _box(ax, 1.3, 4.8, "source images\n(labelled)", PALE[0], w=2.1, h=0.95, fs=9.5)
    _box(ax, 1.3, 2.4, "target images\n(NO labels)", PALE[1], w=2.1, h=0.95, fs=9.5)
    _box(ax, 4.6, 3.6, "feature extractor G\n(the conv layers)", PALE[2], w=2.5, h=1.3, fs=10, bold=True)
    _box(ax, 9.6, 5.2, "label classifier C\n→ which digit?", PALE[0], w=2.6, h=1.0, fs=9.5)
    _box(ax, 7.3, 1.6, "GRL", "#fff1a8", w=0.9, h=0.62, fs=10, bold=True)
    _box(ax, 10.1, 1.6, "domain classifier D\n→ source or target?", PALE[3], w=3.0, h=1.0, fs=9.5)
    _arr(ax, (2.4, 4.75), (3.33, 3.95)); _arr(ax, (2.4, 2.45), (3.33, 3.25))
    _arr(ax, (5.87, 3.95), (8.28, 5.05), text="features f  (source only)", fs=8.8, tcolor=INK, toff=(-0.3, 0.1))
    _arr(ax, (5.6, 2.95), (6.83, 1.75), text="f  (both)", fs=8.8, tcolor=INK, toff=(-0.55, -0.2))
    _arr(ax, (7.76, 1.6), (8.58, 1.6))
    _box(ax, 12.5, 5.2, "loss_label", PALE[4], w=1.5, h=0.6, fs=9)
    _box(ax, 12.5, 0.2, "loss_domain", PALE[4], w=1.6, h=0.6, fs=9)
    _arr(ax, (10.92, 5.2), (11.73, 5.2)); _arr(ax, (11.2, 1.1), (11.9, 0.5))
    # backward: gradients
    _arr(ax, (8.25, 4.8), (5.9, 4.15), color=RED, lw=1.6, rad=-0.15, ls="--")
    ax.text(9.1, 6.15, "gradient of loss_label: G learns features that tell DIGITS apart", fontsize=9, color=RED, ha="center")
    _arr(ax, (8.6, 1.25), (7.75, 1.25), color=RED, lw=1.6, ls="--")
    ax.text(8.15, 0.9, "+g", fontsize=10, color=RED, ha="center", fontweight="bold")
    _arr(ax, (6.85, 1.25), (5.25, 2.8), color=RED, lw=1.6, ls="--", rad=-0.1)
    ax.text(5.6, 1.7, "−λg", fontsize=10.5, color=RED, ha="center", fontweight="bold")
    ax.text(7.3, -0.25, "Gradient Reversal Layer: forward, f passes unchanged;\nbackward, the gradient is multiplied by −λ",
            fontsize=9, ha="center", va="top", color=INK, bbox=dict(fc="#fffbe0", ec="#e0c200", boxstyle="round,pad=0.3"))
    ax.text(0.2, 0.75, "D learns to tell the domains apart (its own gradient, +g).\nG gets the REVERSED gradient: it learns features that\nmake D fail, i.e. features that look the same in both\ndomains. C trained on source then works on target.",
            fontsize=9, va="top", color=SUB)
    ax.set_title("Domain-adversarial training (DANN, Ganin & Lempitsky 2015): learn features that are good for the task and can't reveal the domain", fontsize=10.5, fontweight="bold")
    plt.show()



# ================================================================== 02 · autoencoders
def fig_ae_hourglass(i=0):
    import torch
    model = load_ae(8)
    X_tr, y_tr, X_te, y_te = mnist_tensors()
    x = X_te[i:i + 1]
    with torch.no_grad():
        z = model.encoder(x)[0].numpy(); xh = model(x)[0].numpy()
    fig = plt.figure(figsize=(14, 5.6))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 14); ax.set_ylim(-0.8, 5.6); _off(ax)
    a1 = fig.add_axes([0.01, 0.30, 0.13, 0.36]); _img(a1, x[0].reshape(28, 28).numpy(), "input x\n(784 pixels)", fs=9.5)
    a2 = fig.add_axes([0.86, 0.30, 0.13, 0.36]); _img(a2, xh.reshape(28, 28), "reconstruction x̂\n(784 numbers)", fs=9.5)
    widths = [(2.6, 784, PALE[0], C0, "784"), (3.9, 128, PALE[0], C0, "128"), (6.3, 8, PALE[2], C2, "8 = k"), (8.7, 128, PALE[1], C1, "128"), (10.0, 784, PALE[1], C1, "784")]
    for x0, n, fc, ec, t in widths:
        h = 0.35 + 3.9 * (n / 784) ** 0.55
        ax.add_patch(plt.Rectangle((x0 - 0.18, 2.4 - h / 2), 0.36, h, fc=fc, ec=ec, lw=1.3))
        ax.text(x0, 2.4 - h / 2 - 0.12, t, ha="center", va="top", fontsize=9)
    for (xa, na), (xb, nb) in zip([(2.6, 784), (3.9, 128), (6.3, 8), (8.7, 128)], [(3.9, 128), (6.3, 8), (8.7, 128), (10.0, 784)]):
        ha_ = (0.35 + 3.9 * (na / 784) ** 0.55) / 2; hb = (0.35 + 3.9 * (nb / 784) ** 0.55) / 2
        ax.add_patch(Polygon([[xa + 0.18, 2.4 + ha_], [xb - 0.18, 2.4 + hb], [xb - 0.18, 2.4 - hb], [xa + 0.18, 2.4 - ha_]], fc="#f2f4f7", ec="#c9ced5", lw=0.8, zorder=1))
    ax.text(3.25, 4.75, "Linear → ReLU", ha="center", fontsize=8.5, color=SUB); ax.text(5.1, 3.7, "Linear", ha="center", fontsize=8.5, color=SUB)
    ax.text(7.5, 3.7, "Linear → ReLU", ha="center", fontsize=8.5, color=SUB); ax.text(9.35, 4.75, "Linear → Sigmoid", ha="center", fontsize=8.5, color=SUB)
    ax.text(4.45, 5.3, "ENCODER  (compress)", ha="center", fontsize=11, fontweight="bold", color=C0)
    ax.text(8.15, 5.3, "DECODER  (rebuild)", ha="center", fontsize=11, fontweight="bold", color=C1)
    ax.text(6.3, -0.15, "the code z  (8 numbers for this image):\n[" + ", ".join(f"{v:.1f}" for v in z) + "]", ha="center", va="top", fontsize=9.5,
            family="monospace", bbox=dict(fc=PALE[2], ec=C2, boxstyle="round,pad=0.3"))
    _arr(ax, (6.3, 0.0), (6.3, 1.55), color=C2, lw=1.2)
    err = float(((xh - x[0].numpy()) ** 2).mean())
    ax.text(11.5, 0.9, f"loss = mean of (x̂ − x)²\n= {err:.4f} for this image", ha="center", va="top", fontsize=9.5, color=INK)
    ax.text(7.0, 4.95, "the bottleneck", ha="center", fontsize=9, color=C2, fontweight="bold")
    plt.show()


def fig_bottleneck_sizes(n=8):
    import torch
    from sklearn.decomposition import PCA
    X_tr, y_tr, X_te, y_te = mnist_tensors()
    x = X_te[:n]
    rows = [("input", x.numpy())]
    for k in (2, 8, 32):
        model = load_ae(k)
        with torch.no_grad():
            rows.append((f"autoencoder, k = {k}", model(x).numpy()))
    pca = PCA(8).fit(X_tr[:10000].numpy())
    rows.insert(3, ("PCA, k = 8", pca.inverse_transform(pca.transform(x.numpy())).clip(0, 1)))
    fig, axes = plt.subplots(len(rows), n, figsize=(13, 8.4))
    for r, (name, ims) in enumerate(rows):
        for c in range(n):
            _img(axes[r, c], ims[c].reshape(28, 28))
        err = float(((ims - x.numpy()) ** 2).mean())
        axes[r, 0].set_ylabel(name + ("" if r == 0 else f"\nerror {err:.4f}"), fontsize=9, rotation=0, ha="right", va="center", labelpad=8)
    fig.suptitle("A smaller bottleneck k forces a coarser summary. At the same k = 8, the (nonlinear) autoencoder rebuilds better than PCA (linear)", fontsize=10.5)
    plt.tight_layout(); plt.show()


def linear_ae_vs_pca(k=2, steps=3000, seed=0):
    """A LINEAR autoencoder on centred 8 × 8 digits, trained by Adam; returns its loss curve, PCA's error, and both codes."""
    import torch
    import torch.nn as nn
    from sklearn.datasets import load_digits
    from sklearn.decomposition import PCA
    X = load_digits().data / 16
    Xc = torch.tensor(X - X.mean(axis=0), dtype=torch.float32)
    torch.manual_seed(seed)
    enc, dec = nn.Linear(64, k, bias=False), nn.Linear(k, 64, bias=False)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-2)
    curve = []
    for t in range(steps):
        loss = ((dec(enc(Xc)) - Xc) ** 2).sum(dim=1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        curve.append(loss.item())
    pca = PCA(k).fit(X)
    Z = pca.transform(X)
    pca_err = float(((pca.inverse_transform(Z) - X) ** 2).sum(axis=1).mean())
    with torch.no_grad():
        codes = enc(Xc).numpy()
    D = dec.weight.detach().numpy()                     # (64, k): the decoder's columns span the AE's plane
    Q, _ = np.linalg.qr(D)
    W = pca.components_.T                               # (64, k): PCA's plane
    cosines = np.linalg.svd(Q.T @ W, compute_uv=False)  # cosines of the angles between the two planes
    return curve, pca_err, codes, Z, cosines


def fig_linear_ae_pca():
    from sklearn.datasets import load_digits
    curve, pca_err, codes, Z, cos = linear_ae_vs_pca()
    y = load_digits().target
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    axes[0].plot(curve, color=C0, lw=1.8, label="linear autoencoder (Adam)")
    axes[0].axhline(pca_err, color=C1, ls="--", lw=1.6, label=f"PCA, k = 2: {pca_err:.3f}")
    axes[0].set_ylim(pca_err * 0.95, pca_err * 1.6); axes[0].set_xlabel("training step"); axes[0].set_ylabel("mean squared error per image")
    axes[0].set_title(f"the AE's error falls to PCA's ({curve[-1]:.3f})", fontsize=10); axes[0].legend(fontsize=8.5)
    for ax, (P, t) in zip(axes[1:], [(Z, "PCA scores (PC1, PC2)"), (codes, "the linear AE's codes (z1, z2)")]):
        ax.scatter(P[:, 0], P[:, 1], c=[plt.cm.tab10(v) for v in y], s=6, alpha=0.7)
        ax.set_title(t, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"Same plane, different axes: the angles between the AE's 2-D subspace and PCA's have cosines {cos[0]:.4f} and {cos[1]:.4f} (1 = identical).\n"
                 "The middle and right pictures are the same cloud, turned and stretched: the AE learned PCA's subspace, but not PCA's particular axes.", fontsize=10.5)
    plt.tight_layout(); plt.show()


def fig_latent_ae2(n=5000):
    import torch
    model = load_ae(2)
    X_tr, y_tr, X_te, y_te = mnist_tensors()
    with torch.no_grad():
        Z = model.encoder(X_te[:n]).numpy()
    yy = y_te[:n].numpy()
    fig = plt.figure(figsize=(13, 6.2))
    ax = fig.add_axes([0.05, 0.08, 0.55, 0.82])
    for c in range(10):
        m = yy == c
        ax.scatter(Z[m, 0], Z[m, 1], s=4, color=plt.cm.tab10(c), alpha=0.6)
        ax.text(*np.median(Z[m], axis=0), str(c), fontsize=12, fontweight="bold", color=plt.cm.tab10(c), ha="center", va="center",
                bbox=dict(fc="white", ec=plt.cm.tab10(c), boxstyle="circle,pad=0.2", alpha=0.9))
    c0, c1 = np.median(Z[yy == 0], axis=0), np.median(Z[yy == 1], axis=0)
    probes = [c0, (c0 + c1) / 2, np.array([-8.0, -20.0])]
    names = ["A: the middle of the 0s", "B: halfway, 0s to 1s", "C: empty space"]
    for k, p in enumerate(probes):
        ax.plot(*p, marker="X", ms=14, color="k", mec="white", zorder=5); ax.text(p[0], p[1], f"  {'ABC'[k]}", fontsize=13, fontweight="bold", va="center", zorder=5)
        with torch.no_grad():
            im = model.decoder(torch.tensor(np.array([p]), dtype=torch.float32))[0].numpy().reshape(28, 28)
        a = fig.add_axes([0.64 + 0.115 * k, 0.48, 0.105, 0.24]); _img(a, im, names[k], fs=9)
    ax.set_xlabel("z1"); ax.set_ylabel("z2")
    ax.set_title("The codes of 5000 test images (k = 2), coloured by the true digit (never shown to the AE)", fontsize=10)
    fig.text(0.65, 0.40, "Decoding three codes (the X marks):\nA, inside a cluster: a clean 0.\nB, between two clusters: a smudge that is neither.\nC, empty space: a blob.\n\nThe AE was only trained to rebuild its inputs,\nso nothing makes the space between and around\nthe clusters meaningful. 03 fixes that.", fontsize=9.5, va="top", color=SUB)
    plt.show()


def fig_denoise(n=8, std=0.5):
    import torch
    X_tr, y_tr, X_te, y_te = mnist_tensors()
    try:
        model = make_ae(32); model.load_state_dict(torch.load("dae32.pt")); model.eval()
    except Exception:
        model = train_dae()
    torch.manual_seed(0)
    x = X_te[:n]; noisy = (x + std * torch.randn_like(x)).clamp(0, 1)
    with torch.no_grad():
        out = model(noisy)
    fig, axes = plt.subplots(3, n, figsize=(13, 5.2))
    for c in range(n):
        _img(axes[0, c], noisy[c].reshape(28, 28).numpy()); _img(axes[1, c], out[c].reshape(28, 28).numpy()); _img(axes[2, c], x[c].reshape(28, 28).numpy())
    for r, t in enumerate(["noisy input\n(what goes in)", "denoising AE's\noutput", "the clean original\n(the target)"]):
        axes[r, 0].set_ylabel(t, fontsize=9.5, rotation=0, ha="right", va="center", labelpad=8)
    fig.suptitle(f"A denoising autoencoder: trained with noisy inputs (noise std {std}) and CLEAN targets, so it must learn what digits look like, not just copy", fontsize=10.5)
    plt.tight_layout(); plt.show()


def train_dae(std=0.5, epochs=5, seed=0):
    """02 §5's recipe: the input gets noise, the target stays clean."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    X = mnist_tensors()[0]
    torch.manual_seed(seed)
    model = make_ae(32)
    loader = DataLoader(TensorDataset(X), batch_size=128, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        model.train()
        for (xb,) in loader:
            noisy = (xb + std * torch.randn_like(xb)).clamp(0, 1)
            loss = nn.functional.mse_loss(model(noisy), xb)      # rebuild the CLEAN image
            opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()


def fig_conv_transpose():
    """ConvTranspose2d, kernel 2, stride 2 on a 2 × 2 input: each input value stamps a scaled copy of the kernel."""
    import torch
    import torch.nn.functional as F
    x = np.array([[1.0, 2.0], [3.0, 0.0]]); K = np.array([[1.0, 0.5], [0.0, 1.0]])
    y = F.conv_transpose2d(torch.tensor(x)[None, None], torch.tensor(K)[None, None], stride=2)[0, 0].numpy()
    x3 = np.array([[1.0, 2.0], [3.0, 0.0]]); K3 = np.ones((3, 3))
    y3 = F.conv_transpose2d(torch.tensor(x3)[None, None], torch.tensor(K3)[None, None], stride=1)[0, 0].numpy()
    colors = [[PALE[0], PALE[1]], [PALE[2], PALE[3]]]; edges = [[C0, C1], [C2, C3]]
    fig, ax = plt.subplots(figsize=(15, 5.4)); ax.set_xlim(0, 15); ax.set_ylim(-1.2, 5.0); _off(ax)

    def grid(ox, oy, M, cell, fcs=None, ecs=None, fs=12, bold=False):
        for r in range(M.shape[0]):
            for c in range(M.shape[1]):
                fc = fcs[r][c] if fcs else "white"; ec = ecs[r][c] if ecs else "#9aa3ad"
                ax.add_patch(plt.Rectangle((ox + c * cell, oy - (r + 1) * cell), cell, cell, fc=fc, ec=ec, lw=1.2))
                ax.text(ox + c * cell + cell / 2, oy - r * cell - cell / 2, _f(M[r, c], 2), ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")
    grid(0.3, 3.6, x, 0.75, colors, edges, bold=True); ax.text(1.05, 3.85, "input (2, 2)", ha="center", fontsize=10, fontweight="bold")
    grid(2.4, 3.6, K, 0.75); ax.text(3.15, 3.85, "kernel (2, 2)", ha="center", fontsize=10, fontweight="bold")
    ax.text(1.8, 1.6, "stride 2: input value (i, j) adds\nvalue × kernel into the 2 × 2 block\nstarting at (2i, 2j) of the output", ha="center", va="top", fontsize=9, color=SUB)
    fcs = [[colors[r // 2][c // 2] for c in range(4)] for r in range(4)]; ecs = [[edges[r // 2][c // 2] for c in range(4)] for r in range(4)]
    _arr(ax, (4.3, 2.85), (5.2, 2.85))
    grid(5.4, 4.0, y, 0.75, fcs, ecs); ax.text(6.9, 4.25, "output (4, 4)", ha="center", fontsize=10, fontweight="bold")
    ax.text(6.9, 0.7, "e.g. the blue block = 1 × kernel,\nthe orange block = 2 × kernel", ha="center", va="top", fontsize=9, color=SUB)
    # stride 1, 3 × 3 kernel of ones: overlaps add up
    ax.plot([9.0, 9.0], [-1.0, 4.6], color="#d0d5db", lw=1)
    grid(9.4, 3.6, x3, 0.6, colors, edges, fs=11, bold=True); ax.text(10.0, 3.85, "input (2, 2)", ha="center", fontsize=10, fontweight="bold")
    ax.text(10.0, 2.1, "kernel 3 × 3 of 1s,\nstride 1", ha="center", va="top", fontsize=9, color=SUB)
    _arr(ax, (10.9, 2.85), (11.7, 2.85))
    grid(11.75, 4.1, y3, 0.6, fs=11); ax.text(12.95, 4.35, "output (4, 4)", ha="center", fontsize=10, fontweight="bold")
    ax.text(12.95, 1.45, "stamps overlap and ADD: the middle\nfour cells get all four: 1 + 2 + 3 + 0 = 6", ha="center", va="top", fontsize=9, color=SUB)
    ax.text(7.5, -0.75, "Output size  =  (n − 1)·s − 2p + k.    Left: (2 − 1)·2 − 0 + 2 = 4.    Right: (2 − 1)·1 − 0 + 3 = 4.", ha="center", fontsize=10.5, color=INK, fontweight="bold")
    ax.set_title("Transposed convolution (the decoder's 'upsampling' layer): every input pixel paints a scaled copy of the kernel; overlapping paint adds up", fontsize=10.5, fontweight="bold")
    plt.show()


# ================================================================== 03 · VAE
def fig_vae_architecture(i=0):
    import torch
    model = load_vae()
    X_tr, y_tr, X_te, y_te = mnist_tensors()
    x = X_te[i:i + 1]
    with torch.no_grad():
        h = model.encoder(x); mu, logvar = model.to_mu(h)[0].numpy(), model.to_logvar(h)[0].numpy()
    sd = np.exp(0.5 * logvar); eps = np.array([0.5, -1.0]); z = mu + sd * eps
    with torch.no_grad():
        xh = model.decoder(torch.tensor(z[None], dtype=torch.float32))[0].numpy()
    fig = plt.figure(figsize=(15, 6.2))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 15); ax.set_ylim(-1.0, 6.2); _off(ax)
    a1 = fig.add_axes([0.005, 0.40, 0.1, 0.25]); _img(a1, x[0].reshape(28, 28).numpy(), "input x", fs=9.5)
    a2 = fig.add_axes([0.895, 0.40, 0.1, 0.25]); _img(a2, xh.reshape(28, 28), "x̂ (decoded z)", fs=9.5)
    _box(ax, 2.6, 3.4, "encoder\n784 → 256\nLinear, ReLU", PALE[0], w=1.9, h=1.4, fs=9)
    _box(ax, 5.0, 4.7, f"μ  =  ({mu[0]:.2f}, {mu[1]:.2f})", PALE[0], w=2.4, h=0.65, fs=9.5)
    _box(ax, 5.0, 2.1, f"log σ²  =  ({logvar[0]:.2f}, {logvar[1]:.2f})\nσ  =  ({sd[0]:.3f}, {sd[1]:.3f})", PALE[0], w=2.6, h=0.95, fs=9)
    _arr(ax, (3.55, 3.7), (3.8, 4.55)); _arr(ax, (3.55, 3.1), (3.7, 2.45))
    _box(ax, 8.4, 3.4, f"z = μ + σ ⊙ ε\n= ({z[0]:.2f}, {z[1]:.2f})", PALE[2], w=2.4, h=1.0, fs=10, bold=True)
    _arr(ax, (6.2, 4.6), (7.25, 3.75)); _arr(ax, (6.3, 2.3), (7.25, 3.05))
    _box(ax, 8.4, 0.9, f"ε ~ N(0, I)\nthis time ε = ({eps[0]}, {eps[1]})", "#fff1a8", w=2.5, h=0.85, fs=9)
    _arr(ax, (8.4, 1.33), (8.4, 2.88), color="#b59a00")
    _box(ax, 11.5, 3.4, "decoder\n2 → 256 → 784\nReLU, Sigmoid", PALE[1], w=2.0, h=1.4, fs=9)
    _arr(ax, (9.6, 3.4), (10.5, 3.4)); _arr(ax, (12.5, 3.4), (13.35, 3.4)); _arr(ax, (1.6, 3.4), (1.65, 3.4))
    ax.text(5.0, 5.75, "the encoder outputs a DISTRIBUTION for z:\na normal with mean μ and standard deviation σ", ha="center", fontsize=9.5, color=C0, fontweight="bold")
    ax.text(10.0, 5.6, "loss  =  reconstruction  (x̂ close to x)\n          +  KL( N(μ, σ²)  ‖  N(0, 1) )  (the code stays near 0, spread about 1)",
            ha="left", fontsize=9.5, color=INK, family="monospace", va="center", bbox=dict(fc="#f6f7f9", ec="#c9ced5", boxstyle="round,pad=0.35"))
    ax.text(8.4, -0.35, "the random part sits OUTSIDE the path from μ, σ to the loss, so gradients flow through μ and σ (§3)", ha="center", fontsize=9, color=SUB)
    plt.show()


def fig_reparam():
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    for ax in axes:
        ax.set_xlim(0, 7); ax.set_ylim(0, 4.4); _off(ax)
    a = axes[0]
    _box(a, 1.1, 3.2, "μ", PALE[0], w=0.9, h=0.6, fs=12); _box(a, 1.1, 1.4, "σ", PALE[0], w=0.9, h=0.6, fs=12)
    _box(a, 3.5, 2.3, "z ~ N(μ, σ²)\n(draw a sample)", "#fff1a8", w=2.2, h=0.95, fs=10)
    _box(a, 6.0, 2.3, "loss", PALE[4], w=1.0, h=0.6, fs=10)
    _arr(a, (1.55, 3.1), (2.4, 2.6)); _arr(a, (1.55, 1.5), (2.4, 2.0)); _arr(a, (4.6, 2.3), (5.5, 2.3))
    _arr(a, (5.5, 1.95), (4.6, 1.95), color=RED, ls="--"); a.text(5.05, 1.55, "∂L/∂z", color=RED, ha="center", fontsize=10)
    a.text(3.5, 1.15, "✕", color=RED, fontsize=26, ha="center", va="center", fontweight="bold")
    a.text(3.5, 0.45, "∂z/∂μ = ?   ∂z/∂σ = ?   'draw a random number' has no derivative:\nthe gradient stops here, so the encoder can't learn", ha="center", fontsize=9, color=RED)
    a.set_title("without the trick", fontsize=11, fontweight="bold")
    b = axes[1]
    _box(b, 0.9, 3.6, "μ", PALE[0], w=0.8, h=0.6, fs=12); _box(b, 0.9, 2.2, "σ", PALE[0], w=0.8, h=0.6, fs=12)
    _box(b, 0.9, 0.8, "ε ~ N(0, 1)", "#fff1a8", w=1.6, h=0.6, fs=10)
    _box(b, 3.3, 2.2, "z = μ + σ·ε", PALE[2], w=2.0, h=0.8, fs=11, bold=True)
    _box(b, 6.0, 2.2, "loss", PALE[4], w=1.0, h=0.6, fs=10)
    _arr(b, (1.3, 3.5), (2.4, 2.55)); _arr(b, (1.3, 2.2), (2.3, 2.2)); _arr(b, (1.6, 1.05), (2.4, 1.85)); _arr(b, (4.3, 2.2), (5.5, 2.2))
    _arr(b, (5.5, 2.6), (4.3, 2.6), color=RED, ls="--"); b.text(4.9, 2.75, "∂L/∂z", color=RED, ha="center", fontsize=10)
    _arr(b, (2.35, 2.75), (1.3, 3.75), color=RED, ls="--"); b.text(1.75, 3.95, "∂z/∂μ = 1", color=RED, fontsize=10)
    _arr(b, (2.3, 2.45), (1.3, 2.45), color=RED, ls="--"); b.text(1.85, 2.55, "∂z/∂σ = ε", color=RED, fontsize=10, ha="center")
    b.text(3.6, 0.45, "the randomness is an INPUT (ε), not a step on the path:\nz is now an ordinary function of μ and σ, so backprop works", ha="center", fontsize=9, color=C2)
    b.set_title("with the reparameterization trick", fontsize=11, fontweight="bold")
    plt.tight_layout(); plt.show()


def kl_normal(mu, var):
    """KL( N(mu, var) || N(0, 1) ) for one dimension."""
    return 0.5 * (var + mu ** 2 - 1 - np.log(var))


def fig_kl_gaussians():
    t = np.linspace(-5, 5, 500)
    pdf = lambda m, v: np.exp(-(t - m) ** 2 / (2 * v)) / np.sqrt(2 * np.pi * v)
    cases = [(0.0, 1.0), (2.0, 1.0), (0.0, 0.09), (0.0, 4.0)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    for ax, (m, v) in zip(axes, cases):
        ax.fill_between(t, pdf(0, 1), color="#cfd5dc", alpha=0.7, label="N(0, 1): the prior")
        ax.plot(t, pdf(m, v), color=C3, lw=2.2, label="q = N(μ, σ²)")
        ax.set_title(f"μ = {_f(m)}, σ² = {_f(v)}\nKL = {kl_normal(m, v):.3f}", fontsize=10.5)
        ax.set_xlim(-5, 5); ax.set_ylim(0, 1.45); ax.set_yticks([])
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("KL(q ‖ N(0, 1)) = ½(σ² + μ² − 1 − log σ²): zero only when q IS N(0, 1); it grows when q moves away (μ) or is too narrow or too wide (σ²)", fontsize=10.5)
    plt.tight_layout(); plt.show()


def fig_vae_latent(n=5000):
    import torch
    vae = load_vae(); ae = load_ae(2)
    X_tr, y_tr, X_te, y_te = mnist_tensors()
    with torch.no_grad():
        Za = ae.encoder(X_te[:n]).numpy(); Zv = vae.to_mu(vae.encoder(X_te[:n])).numpy()
    yy = y_te[:n].numpy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (Z, t) in zip(axes, [(Za, "autoencoder (02): codes z"), (Zv, "VAE: the means μ")]):
        for c in range(10):
            m = yy == c
            ax.scatter(Z[m, 0], Z[m, 1], s=4, color=plt.cm.tab10(c), alpha=0.55)
            ax.text(*np.median(Z[m], axis=0), str(c), fontsize=12, fontweight="bold", color=plt.cm.tab10(c), ha="center", va="center",
                    bbox=dict(fc="white", ec=plt.cm.tab10(c), boxstyle="circle,pad=0.2", alpha=0.9))
        ax.set_title(t + f"   (range z1: {Z[:, 0].min():.0f} to {Z[:, 0].max():.0f})", fontsize=10.5)
        ax.set_xlabel("z1"); ax.set_ylabel("z2")
    for r in (1, 2):
        axes[1].add_patch(plt.Circle((0, 0), r, fill=False, ls="--", ec="#5b6675", lw=1))
    axes[1].text(0, -2.15, "circles: 1 and 2 standard deviations of N(0, I)", ha="center", fontsize=8.5, color=SUB)
    axes[1].set_aspect("equal")
    fig.suptitle("The same 5000 test digits. The AE spreads them anywhere, with empty gaps. The KL term pulls the VAE's codes into one round,\n"
                 "gap-free blob around 0, which is exactly where we'll sample from", fontsize=10.5)
    plt.tight_layout(); plt.show()


def fig_vae_manifold(n=15, r=2.5):
    import torch
    vae = load_vae()
    g = np.linspace(-r, r, n)
    canvas = np.ones((n * 28, n * 28))
    with torch.no_grad():
        for a, z2 in enumerate(g[::-1]):
            for b, z1 in enumerate(g):
                canvas[a * 28:(a + 1) * 28, b * 28:(b + 1) * 28] = vae.decoder(torch.tensor([[z1, z2]], dtype=torch.float32))[0].numpy().reshape(28, 28)
    fig, ax = plt.subplots(figsize=(8.2, 8.2))
    ax.imshow(canvas, cmap="gray_r", extent=(-r - (g[1] - g[0]) / 2, r + (g[1] - g[0]) / 2, -r - (g[1] - g[0]) / 2, r + (g[1] - g[0]) / 2))
    ax.set_xlabel("z1"); ax.set_ylabel("z2"); ax.grid(False)
    ax.set_title(f"Decoding a {n} × {n} grid of codes from −{r} to {r}: every point gives a digit-like image,\nand neighbouring codes give similar images (it morphs smoothly)", fontsize=10.5)
    plt.tight_layout(); plt.show()


def fig_vae_samples(n=10, seed=0):
    import torch
    vae = load_vae(); ae = load_ae(2)
    X_tr, y_tr, X_te, y_te = mnist_tensors()
    with torch.no_grad():
        Za = ae.encoder(X_te[:5000]).numpy()
    g = torch.Generator().manual_seed(seed)
    zv = torch.randn(n, 2, generator=g)
    lo, hi = torch.tensor(Za.min(axis=0)), torch.tensor(Za.max(axis=0))
    za = lo + (hi - lo) * torch.rand(n, 2, generator=g)
    with torch.no_grad():
        xv = vae.decoder(zv).numpy(); xa = ae.decoder(za.float()).numpy()
    fig, axes = plt.subplots(2, n, figsize=(14, 3.6))
    for c in range(n):
        _img(axes[0, c], xa[c].reshape(28, 28)); _img(axes[1, c], xv[c].reshape(28, 28))
    axes[0, 0].set_ylabel("AE: random z\nin its box", fontsize=9.5, rotation=0, ha="right", va="center", labelpad=8)
    axes[1, 0].set_ylabel("VAE: z ~ N(0, I)", fontsize=9.5, rotation=0, ha="right", va="center", labelpad=8)
    fig.suptitle("Generating new images: pick a code, decode it. Random codes give AE garbage half the time; the VAE's prior N(0, I) gives digits", fontsize=10.5)
    plt.tight_layout(); plt.show()


# ================================================================== 04 · anomaly detection
SENSOR_MU = np.array([60.0, 4.0])                    # temperature (°C), vibration (mm/s)
SENSOR_COV = np.array([[16.0, 3.2], [3.2, 1.0]])     # sd 4 and 1, correlation 0.8


def sensor_data(seed=0):
    """500 normal machine readings (temperature, vibration) and 5 faulty ones."""
    rng = np.random.default_rng(seed)
    X = rng.multivariate_normal(SENSOR_MU, SENSOR_COV, size=500)
    A = np.array([[75.0, 8.5], [45.0, 0.6], [70.0, 3.0], [52.0, 5.5], [62.0, 8.0]])
    return X, A


def fig_sensor_data():
    X, A = sensor_data()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [2.2, 1, 1]})
    ax = axes[0]
    ax.scatter(X[:, 0], X[:, 1], s=10, color=C0, alpha=0.5, label="500 normal readings")
    ax.scatter(A[:, 0], A[:, 1], s=90, marker="X", color=RED, ec="white", label="faulty readings", zorder=3)
    for k, (a, b) in enumerate(A):
        ax.text(a + 0.8, b, "PQRST"[k], fontsize=11, fontweight="bold", color=RED, va="center")
    ax.set_xlabel("temperature (°C)"); ax.set_ylabel("vibration (mm/s)"); ax.legend(fontsize=8.5, loc="upper left")
    ax.set_title("A machine's sensors: hot ⇔ shaking more (correlation 0.8)", fontsize=10)
    axes[1].hist(X[:, 0], bins=30, color=C0, alpha=0.6); [axes[1].axvline(a, color=RED, lw=1) for a in A[:, 0]]
    axes[1].set_title("temperature alone", fontsize=10); axes[1].set_yticks([])
    axes[2].hist(X[:, 1], bins=30, color=C0, alpha=0.6); [axes[2].axvline(b, color=RED, lw=1) for b in A[:, 1]]
    axes[2].set_title("vibration alone", fontsize=10); axes[2].set_yticks([])
    for k in (2, 3):
        axes[1].text(A[k, 0], axes[1].get_ylim()[1] * 0.92, "RS"[k - 2], color=RED, fontsize=10, fontweight="bold", ha="center")
        axes[2].text(A[k, 1], axes[2].get_ylim()[1] * 0.92, "RS"[k - 2], color=RED, fontsize=10, fontweight="bold", ha="center")
    fig.suptitle("R (hot but calm) and S (cool but shaking) look normal on each sensor separately; they're only odd TOGETHER", fontsize=10.5)
    plt.tight_layout(); plt.show()


def mahalanobis(X, mu, cov):
    d = X - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", d, np.linalg.inv(cov), d))


def fig_mahalanobis():
    X, A = sensor_data()
    mu, cov = X.mean(axis=0), np.cov(X.T)
    g1, g2 = np.meshgrid(np.linspace(40, 80, 300), np.linspace(-0.5, 9.5, 300))
    G = np.c_[g1.ravel(), g2.ravel()]
    zs = (G - mu) / X.std(axis=0, ddof=1)
    eu = np.sqrt((zs ** 2).sum(axis=1)).reshape(g1.shape)
    ma = mahalanobis(G, mu, cov).reshape(g1.shape)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for ax, (D, t, s_) in zip(axes, [(eu, "distance after standardizing each sensor (circles)", (X - mu) / X.std(axis=0, ddof=1)), (ma, "Mahalanobis distance (ellipses along the correlation)", None)]):
        cs = ax.contour(g1, g2, D, levels=[1, 2, 3], colors=["#7f8c8d", C1, RED], linewidths=[1, 1.3, 1.8])
        ax.clabel(cs, fmt=lambda v: f"{v:.0f}", fontsize=9)
        ax.scatter(X[:, 0], X[:, 1], s=8, color=C0, alpha=0.4)
        ax.scatter(A[:, 0], A[:, 1], s=80, marker="X", color=RED, ec="white", zorder=3)
        for k, (a, b) in enumerate(A):
            v = (np.sqrt((((np.array([a, b]) - mu) / X.std(axis=0, ddof=1)) ** 2).sum()) if s_ is not None else mahalanobis(np.array([[a, b]]), mu, cov)[0])
            ax.text(a + 0.8, b + 0.15, f"{'PQRST'[k]}: {v:.1f}", fontsize=9.5, fontweight="bold", color=RED)
        ax.set_title(t, fontsize=10); ax.set_xlabel("temperature (°C)")
    axes[0].set_ylabel("vibration (mm/s)")
    fig.suptitle("Contours at distance 1, 2, 3. On the left, R and S sit inside the '3' circle (missed); on the right they're far outside the ellipse (caught)", fontsize=10.5)
    plt.tight_layout(); plt.show()


def fig_isolation(seed=3):
    """Random splits until a point is alone: the outlier takes far fewer."""
    X, A = sensor_data()
    pts = np.r_[X[:60], A[:1]]
    target_out, target_in = len(pts) - 1, int(np.argmin(mahalanobis(X[:60], X.mean(axis=0), np.cov(X.T))))
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0), sharey=True)
    for ax, target, name in [(axes[0], target_out, "the faulty reading P"), (axes[1], target_in, "a typical reading")]:
        lo, hi = pts.min(axis=0) - 1, pts.max(axis=0) + 1
        idx = np.arange(len(pts)); box = [lo.copy(), hi.copy()]; k = 0
        ax.scatter(pts[:, 0], pts[:, 1], s=14, color=C0, alpha=0.6)
        while len(idx) > 1 and k < 40:
            f = int(rng.integers(0, 2)); cut = rng.uniform(pts[idx, f].min(), pts[idx, f].max())
            if f == 0:
                ax.plot([cut, cut], [box[0][1], box[1][1]], color=C1, lw=1.1, alpha=0.9)
            else:
                ax.plot([box[0][0], box[1][0]], [cut, cut], color=C1, lw=1.1, alpha=0.9)
            k += 1
            side = pts[target, f] < cut
            idx = idx[(pts[idx, f] < cut) == side]
            if side:
                box[1][f] = cut
            else:
                box[0][f] = cut
        ax.add_patch(plt.Rectangle(box[0], *(box[1] - box[0]), fc=PALE[4], ec=RED, lw=1.5, alpha=0.6))
        ax.scatter(*pts[target], s=110, marker="X" if target == target_out else "o", color=RED, ec="white", zorder=4)
        ax.set_title(f"isolating {name}: {k} random cuts", fontsize=10.5); ax.set_xlabel("temperature (°C)")
    axes[0].set_ylabel("vibration (mm/s)")
    fig.suptitle("Isolation forest: cut the space at random (a random sensor, a random value) until the point is alone in its box.\n"
                 "Unusual points end up alone after few cuts. Score = the average number of cuts over many random trees: fewer = more anomalous", fontsize=10.5)
    plt.tight_layout(); plt.show()


TOY_SCORES = np.array([0.95, 0.90, 0.80, 0.70, 0.62, 0.55, 0.40, 0.30, 0.20, 0.10])
TOY_LABELS = np.array([1, 1, 0, 1, 0, 0, 0, 0, 1, 0])


def roc_pr_points(scores, labels):
    """Walk the threshold down through the sorted scores: one (FPR, TPR, precision, recall) per step."""
    order = np.argsort(-scores)
    y = labels[order]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    P, N = y.sum(), len(y) - y.sum()
    return fp / N, tp / P, tp / (tp + fp), tp / P


def fig_roc_pr_toy():
    fpr, tpr, prec, rec = roc_pr_points(TOY_SCORES, TOY_LABELS)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    ax = axes[0]; _off(ax); ax.set_xlim(0, 6.2); ax.set_ylim(-5.6, 0.6)
    rows = []
    for k in range(10):
        rows.append([f"{TOY_SCORES[k]:.2f}", "yes" if TOY_LABELS[k] else "no", f"{int(round(tpr[k] * 4))}", f"{int(round(fpr[k] * 6))}", f"{tpr[k]:.2f}", f"{fpr[k]:.2f}", f"{prec[k]:.2f}"])
    _table(ax, 0.0, 0.3, ["score", "faulty?", "TP", "FP", "TPR", "FPR", "prec."], rows, cw=[0.85, 0.85, 0.6, 0.6, 0.8, 0.8, 0.8], ch=0.5,
           colors={(k, 1): PALE[4] for k in range(10) if TOY_LABELS[k]}, fs=9)
    ax.text(0.0, -5.45, "threshold just below row k's score: everything above it is flagged.\n4 faulty, 6 good in total.  TPR = TP / 4,  FPR = FP / 6,  precision = TP / (TP + FP)", fontsize=8.6, color=SUB, va="top")
    a = axes[1]
    a.plot(np.r_[0, fpr], np.r_[0, tpr], "-o", color=C0, lw=2, ms=5)
    a.fill_between(np.r_[0, fpr], np.r_[0, tpr], step=None, color=C0, alpha=0.12)
    a.plot([0, 1], [0, 1], ls=":", color="#9aa3ad")
    from sklearn.metrics import roc_auc_score, average_precision_score
    auc = roc_auc_score(TOY_LABELS, TOY_SCORES); ap = average_precision_score(TOY_LABELS, TOY_SCORES)
    a.set_title(f"ROC curve: AUC = {auc:.3f}", fontsize=10.5); a.set_xlabel("false positive rate (FP / 6)"); a.set_ylabel("true positive rate = recall (TP / 4)")
    for k in (1, 3, 8):
        a.annotate(f"row {k}", (fpr[k], tpr[k]), xytext=(fpr[k] + 0.08, tpr[k] - 0.1), fontsize=8.5, arrowprops=dict(arrowstyle="-", lw=0.7))
    b = axes[2]
    b.step(np.r_[0, rec], np.r_[1, prec], where="post", color=C1, lw=2)
    b.plot(rec, prec, "o", color=C1, ms=5)
    b.axhline(0.4, ls=":", color="#9aa3ad"); b.text(0.02, 0.36, "random guessing: 4/10 = 0.4", fontsize=8.5, color=SUB)
    b.set_title(f"precision–recall curve: AP = {ap:.3f}", fontsize=10.5); b.set_xlabel("recall (TP / 4)"); b.set_ylabel("precision"); b.set_ylim(0, 1.05)
    fig.suptitle("Ten parts scored by a detector. Lower the threshold one part at a time; each step is one point on both curves", fontsize=10.5)
    plt.tight_layout(); plt.show()


def ae_zero_scores():
    model = load_ae_zeros()
    Z_tr, good, scratched, others = zeros_data()
    return {"train zeros": recon_scores(model, Z_tr), "good test zeros": recon_scores(model, good),
            "scratched zeros": recon_scores(model, scratched), "other digits": recon_scores(model, others)}


def fig_ae_defect_maps(n=5):
    import torch
    model = load_ae_zeros()
    Z_tr, good, scratched, others = zeros_data()
    cases = [(good[0], "a good part")] + [(scratched[i], "scratched") for i in range(n - 2)] + [(others[0], "another digit")]
    fig, axes = plt.subplots(3, n, figsize=(13, 7.4))
    for c, (x, t) in enumerate(cases):
        with torch.no_grad():
            xh = model(x[None])[0]
        err = ((xh - x) ** 2).numpy().reshape(28, 28)
        _img(axes[0, c], x.reshape(28, 28).numpy(), t, fs=10)
        _img(axes[1, c], xh.reshape(28, 28).numpy())
        _img(axes[2, c], err, None, cmap="magma", vmin=0, vmax=0.6)
        axes[2, c].set_xlabel(f"score = {err.mean():.4f}", fontsize=10)
    for r, t in enumerate(["input x", "reconstruction x̂", "(x̂ − x)² per pixel"]):
        axes[r, 0].set_ylabel(t, fontsize=10, rotation=0, ha="right", va="center", labelpad=10)
    fig.suptitle("An autoencoder trained ONLY on good parts (zeros) rebuilds every input as a clean zero. Whatever it can't rebuild lights up in the error map:\n"
                 "the scratch is found AND located. The anomaly score is the mean of the map", fontsize=10.5)
    plt.tight_layout(); plt.show()


def fig_ae_score_hist():
    s = ae_zero_scores()
    thr = float(np.quantile(s["train zeros"], 0.99))
    fig, ax = plt.subplots(figsize=(12, 4.6))
    bins = np.logspace(np.log10(0.003), np.log10(0.2), 60)
    for (k, v), col in zip([(k, s[k]) for k in ("good test zeros", "scratched zeros", "other digits")], [C0, C1, RED]):
        ax.hist(v, bins=bins, alpha=0.55, color=col, label=f"{k} ({len(v)})", density=True)
    ax.axvline(thr, color="k", lw=1.6, ls="--")
    flagged = {k: float((s[k] > thr).mean()) for k in s}
    ax.text(thr * 1.05, ax.get_ylim()[1] * 0.92, f"threshold = 99th percentile\nof the training scores = {thr:.4f}\n"
            f"flags: {flagged['good test zeros']:.1%} of good,\n{flagged['scratched zeros']:.0%} of scratched, {flagged['other digits']:.0%} of other digits", fontsize=9, va="top")
    ax.set_xscale("log"); ax.set_xlabel("anomaly score = reconstruction error (log scale)"); ax.set_yticks([]); ax.legend(fontsize=9, loc="upper left")
    ax.set_title("Scores of the three groups: wrong digits are easy to separate; scratches overlap with the good parts", fontsize=10.5, fontweight="bold")
    plt.tight_layout(); plt.show()


def fig_ae_roc_pr():
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
    s = ae_zero_scores()
    y = np.r_[np.zeros(len(s["good test zeros"])), np.ones(len(s["scratched zeros"]))]
    sc = np.r_[s["good test zeros"], s["scratched zeros"]]
    fpr, tpr, _ = roc_curve(y, sc); prec, rec, _ = precision_recall_curve(y, sc)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].plot(fpr, tpr, color=C0, lw=2); axes[0].plot([0, 1], [0, 1], ":", color="#9aa3ad")
    axes[0].set_title(f"ROC: AUC = {roc_auc_score(y, sc):.3f}  (looks great)", fontsize=10.5); axes[0].set_xlabel("false positive rate"); axes[0].set_ylabel("recall")
    axes[1].plot(rec, prec, color=C1, lw=2); axes[1].axhline(y.mean(), ls=":", color="#9aa3ad")
    axes[1].text(0.02, y.mean() + 0.02, f"random: {y.mean():.3f}", fontsize=9, color=SUB)
    axes[1].set_title(f"precision–recall: AP = {average_precision_score(y, sc):.3f}  (the honest view)", fontsize=10.5); axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision"); axes[1].set_ylim(0, 1.05)
    fig.suptitle("Scratched vs good zeros (100 vs 880). With 8 good parts per bad one, a small FALSE POSITIVE RATE still means many false alarms;\n"
                 "ROC hides that (it divides by the 880 good ones), precision shows it", fontsize=10.5)
    plt.tight_layout(); plt.show()


# ================================================================== maps (section M of each notebook)
def fig_transfer_map():
    boxes = {"E": (13, 78, "E. the end", "30 labels: 0.61 → 0.87", "grey"),
             "1": (40, 78, "1. why reuse", "early layers are general", "blue"),
             "2": (67, 78, "2. domain shift", "same classes, new look", "orange"),
             "3": (88, 50, "3. three strategies", "scratch / freeze / fine-tune", "green"),
             "4": (62, 30, "4. how many labels", "when copying pays off", "green"),
             "5": (36, 30, "5. why it works", "the features, in 2-D", "blue"),
             "6": (13, 50, "6. no target labels", "domain-adversarial (DANN)", "purple")}
    arrows = [("E", "1", ""), ("1", "2", ""), ("2", "3", "fix it with\na few labels"), ("3", "4", ""), ("4", "5", ""), ("5", "6", "")]
    notes = [(3, 12, "7 is the same recipe with a big pretrained network (torchvision's ResNet), 8 is the exercises.")]
    flow(boxes, arrows, size=(13, 5.6), title="01 · Transfer learning: reuse what a network learned on a big dataset", notes=notes)


def fig_ae_map():
    boxes = {"E": (13, 78, "E. the end", "784 → 8 → 784", "grey"),
             "1": (40, 78, "1. the idea", "copy, through a bottleneck", "green"),
             "2": (67, 78, "2. linear AE = PCA", "a proposition, proved", "blue"),
             "3": (88, 50, "3. training it", "the target is the input", "green"),
             "4": (62, 30, "4. the code space", "what k = 2 looks like", "purple"),
             "5": (36, 30, "5. denoising", "noisy in, clean out", "orange"),
             "6": (13, 50, "6. conv AE", "transposed convolution", "blue")}
    arrows = [("E", "1", ""), ("1", "2", ""), ("2", "3", ""), ("3", "4", ""), ("4", "5", ""), ("5", "6", "")]
    notes = [(3, 12, "7 is the exercises. 03 (the VAE) starts from 4's gaps; 04 (anomaly detection) starts from 3's reconstruction error.")]
    flow(boxes, arrows, size=(13, 5.6), title="02 · Autoencoders: learn a compact code without labels", notes=notes)


def fig_vae_map():
    boxes = {"E": (13, 78, "E. the end", "sample z, decode a digit", "grey"),
             "1": (40, 78, "1. the problem", "AE codes have gaps", "orange"),
             "2": (67, 78, "2. a distribution", "encoder → μ, σ", "purple"),
             "3": (88, 50, "3. reparameterize", "z = μ + σε", "green"),
             "4": (62, 30, "4. the KL term", "closed form, proved", "purple"),
             "5": (36, 30, "5. the ELBO", "why this loss", "blue"),
             "6": (13, 50, "6. train, generate", "the latent space", "green")}
    arrows = [("E", "1", ""), ("1", "2", ""), ("2", "3", ""), ("3", "4", ""), ("4", "5", ""), ("5", "6", "")]
    notes = [(3, 12, "7 is the exercises.")]
    flow(boxes, arrows, size=(13, 5.6), title="03 · The variational autoencoder: a code space you can sample from", notes=notes)


def fig_anomaly_map():
    boxes = {"E": (13, 78, "E. the end", "a scratch found & located", "grey"),
             "1": (40, 78, "1. the setting", "few, unlabelled anomalies", "orange"),
             "2": (67, 78, "2. Gaussian score", "Mahalanobis distance", "blue"),
             "3": (88, 50, "3. isolation forest", "few cuts = odd", "green"),
             "4": (62, 30, "4. threshold", "flag the top 1%", "orange"),
             "5": (36, 30, "5. evaluate", "ROC vs precision–recall", "purple"),
             "6": (13, 50, "6. AE score", "reconstruction error", "green")}
    arrows = [("E", "1", ""), ("1", "2", ""), ("2", "3", ""), ("3", "4", ""), ("4", "5", ""), ("5", "6", "")]
    notes = [(3, 12, "7 compares the methods and lists the deep ones (Deep SVDD, PatchCore); 8 is the exercises.")]
    flow(boxes, arrows, size=(13, 5.6), title="04 · Anomaly detection: learn what normal looks like, flag what doesn't fit", notes=notes)
