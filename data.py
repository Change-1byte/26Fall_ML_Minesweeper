# fixed board for MVP
import numpy as np

COVERED = -1


def neighbors(r, c, h, w):
    """Yield the up-to-8 in-bounds neighbours of (r, c)."""
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if (dr or dc) and 0 <= r + dr < h and 0 <= c + dc < w:
                yield r + dr, c + dc


def neighbor_sum(a):
    """For each cell, how many of its 8 neighbours are True in a."""
    h, w = a.shape
    p = np.pad(a, 1).astype(int)
    return sum(p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w] for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)


def frontier(visible):
    """Covered cells next to at least one revealed clue: where headline metrics are computed."""
    return (visible == COVERED) & (neighbor_sum(visible != COVERED) > 0)


def split_mask(board_ids, split):
    """70/15/15 train/val/test by board, so states of one game never cross splits."""
    b = np.asarray(board_ids) % 20
    return {"train": b < 14, "val": (b >= 14) & (b < 17), "test": b >= 17}[split]


class Board:
    """Minesweeper game. `mines` is hidden from the model; only `visible()` is model input."""

    def __init__(self, mines):
        self.mines = np.asarray(mines, dtype=bool)
        self.num_mines = int(self.mines.sum())
        self.clues = neighbor_sum(self.mines)
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


def generate(num_games, seed=0, h=16, w=16, num_mines=40):
    """Let the exact-solver agent play games; record every state it sees, labelled."""
    from solver import exact_probabilities, label  # solver imports data, so import here

    rows = {k: [] for k in ["board_id", "move_index", "visible", "probs", "mines", "cell_labels", "state_label"]}
    for g in range(num_games):
        rng = np.random.default_rng([seed, g])
        b = Board.new_game(h, w, num_mines, seed=rng)
        move = 0
        while not (b.lost or b.won):
            v = b.visible()
            p = exact_probabilities(v, num_mines)
            cells, state = label(v, p)
            for k, x in zip(rows, [g, move, v, p, b.mines, cells, state]):
                rows[k].append(x)
            best = np.argwhere(p == np.nanmin(p))  # agent sees only the visible board and solver output
            b.reveal(*best[rng.integers(len(best))])
            move += 1
    return {k: np.array(x) for k, x in rows.items()} | {"height": h, "width": w, "num_mines": num_mines, "seed": seed}


def stats(d):
    """Print dataset counts: boards, states, frontier cells, state and frontier-cell label mix."""
    from solver import LABELS

    fr = np.array([frontier(v) for v in d["visible"]])
    print(f"boards {len(set(d['board_id'].tolist()))}  states {len(d['state_label'])}  frontier cells {fr.sum()}")
    print("label,states,state_pct,frontier_cells,cell_pct")
    for i, name in enumerate(LABELS):
        s, c = (d["state_label"] == i).sum(), (d["cell_labels"][fr] == i).sum()
        print(f"{name},{s},{100 * s / len(fr):.1f},{c},{100 * c / fr.sum():.1f}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--num-games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/states.npz")
    args = ap.parse_args()
    d = generate(args.num_games, args.seed)
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(args.out, **d)
    stats(d)
