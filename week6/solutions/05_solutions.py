"""Week 6 · 05 solutions. Look only after a real attempt."""
import torch
import torch.nn as nn


def train_epoch(model, loader, loss_fn, opt):         # booklet 15: the loop, one epoch
    model.train()
    total = 0.0
    for xb, yb in loader:
        logits = model(xb)                            # 1 guess
        loss = loss_fn(logits, yb)                    # 2 score
        opt.zero_grad()                               # 3 wipe
        loss.backward()                               # 4 blame
        opt.step()                                    # 5 adjust
        total += loss.item() * len(xb)
    return total / len(loader.dataset)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)     # (1, 28, 28) -> (8, 28, 28)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)    # (8, 14, 14) -> (16, 14, 14)
        self.pool = nn.MaxPool2d(2)                                 # halves H and W, no parameters
        self.fc = nn.Linear(16 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))                    # (N, 8, 14, 14)
        x = self.pool(torch.relu(self.conv2(x)))                    # (N, 16, 7, 7)
        return self.fc(x.flatten(start_dim=1))                      # (N, 784) -> (N, 10) logits


def shapes_after_each_layer(model, x):
    shapes = []
    for layer in model:                               # an nn.Sequential can be looped over, layer by layer
        x = layer(x)
        shapes.append(tuple(x.shape))
    return shapes


def cnn_output_shape(in_shape, layers):
    c, h, w = in_shape
    for layer in layers:
        if layer[0] == "conv":
            _, c_out, k, p, s = layer
            c, h, w = c_out, (h + 2 * p - k) // s + 1, (w + 2 * p - k) // s + 1
        else:
            size = layer[1]
            h, w = h // size, w // size
    return c, h, w
