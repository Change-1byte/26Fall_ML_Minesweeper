# fixed board for MVP
import numpy as np

COVERED = -1


def neighbors(r, c, h, w):
    """Yield the up-to-8 in-bounds neighbours of (r, c)."""
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if (dr or dc) and 0 <= r + dr < h and 0 <= c + dc < w:
                yield r + dr, c + dc


class Board:
    """Minesweeper game. `mines` is hidden from the model; only `visible()` is model input."""

    def __init__(self, mines):
        self.mines = np.asarray(mines, dtype=bool)
        self.num_mines = int(self.mines.sum())
        padded = np.pad(self.mines, 1).astype(int)
        h, w = self.mines.shape
        self.clues = sum(padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
                         for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)
        self.revealed = np.zeros_like(self.mines)
        self.lost = False

    @classmethod
    def new_game(cls, h=16, w=16, num_mines=40, first_click=None, seed=None):
        """Place mines uniformly, excluding the first click and its neighbours, then reveal it."""
        rng = np.random.default_rng(seed)
        if first_click is None:
            first_click = (int(rng.integers(h)), int(rng.integers(w)))
        safe = {first_click, *neighbors(*first_click, h, w)}
        allowed = [i for i in range(h * w) if divmod(i, w) not in safe]
        if num_mines > len(allowed):
            raise ValueError(f"{num_mines} mines do not fit on {h}x{w} with a safe opening")
        mines = np.zeros(h * w, dtype=bool)
        mines[rng.choice(allowed, num_mines, replace=False)] = True
        board = cls(mines.reshape(h, w))
        board.reveal(*first_click)
        return board

    def reveal(self, r, c):
        """Reveal a cell (flood-filling zeros). Returns True if it was a mine."""
        if self.mines[r, c]:
            self.lost = True
            return True
        h, w = self.mines.shape
        stack = [(r, c)]
        while stack:
            cell = stack.pop()
            if self.revealed[cell]:
                continue
            self.revealed[cell] = True
            if self.clues[cell] == 0:
                stack.extend(neighbors(*cell, h, w))
        return False

    def visible(self):
        """Clue numbers on revealed cells, COVERED (-1) elsewhere."""
        return np.where(self.revealed, self.clues, COVERED)

    @property
    def won(self):
        return not self.lost and (~self.revealed).sum() == self.num_mines
