# simple CNN
import numpy as np

from data import COVERED


def density(visible, num_mines):
    """Prior-only baseline: remaining mines / covered cells on every covered cell, NaN on revealed.

    Works on one state [H,W] or a batch [N,H,W]. No flags, so remaining mines = num_mines.
    """
    covered = visible == COVERED
    p = num_mines / covered.sum(axis=(-2, -1), keepdims=True)
    return np.where(covered, p, np.nan)
