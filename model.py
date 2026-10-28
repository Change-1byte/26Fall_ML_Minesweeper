# simple CNN
import numpy as np
import torch
from torch import nn

from data import COVERED


def density(visible, num_mines):
    """Prior-only baseline: remaining mines / covered cells on every covered cell, NaN on revealed.

    Works on one state [H,W] or a batch [N,H,W]. No flags, so remaining mines = num_mines.
    """
    covered = visible == COVERED
    p = num_mines / covered.sum(axis=(-2, -1), keepdims=True)
    return np.where(covered, p, np.nan)


def encode(visible, num_mines):
    """[N,H,W] visible boards -> [N,11,H,W]: covered mask, one-hot clue 0-8, remaining-mine density.

    Never sees hidden mines. Density is given to both CNNs, as in the proposal, even though it carries
    mine-count information that the LOCAL label excludes.
    """
    covered = visible == COVERED
    x = np.zeros((len(visible), 11, *visible.shape[1:]), np.float32)
    x[:, 0] = covered
    for k in range(9):
        x[:, 1 + k] = visible == k
    x[:, 10] = (num_mines / covered.sum(axis=(1, 2)))[:, None, None]
    return torch.from_numpy(x)


class CNN(nn.Module):
    """Fully convolutional mine-probability model; outputs logits [N,H,W].

    Trunk: four 3x3 convs (stride 1, padding 1) -> each output sees exactly a 9x9 window.
    global_pool=True adds: average over the board -> MLP -> broadcast and concatenate to every cell.
    Both share the same 1x1 head, so the only difference is access to the whole board.
    """

    def __init__(self, width=64, global_pool=False):
        super().__init__()
        layers, c = [], 11
        for _ in range(4):
            layers += [nn.Conv2d(c, width, 3, padding=1), nn.ReLU()]
            c = width
        self.trunk = nn.Sequential(*layers)
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.ReLU()) if global_pool else None
        self.head = nn.Sequential(nn.Conv2d(width * (2 if global_pool else 1), width, 1), nn.ReLU(), nn.Conv2d(width, 1, 1))

    def forward(self, x):
        f = self.trunk(x)
        if self.mlp is not None:
            g = self.mlp(f.mean(dim=(2, 3)))
            f = torch.cat([f, g[:, :, None, None].expand_as(f)], dim=1)
        return self.head(f)[:, 0]
