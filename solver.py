# Small exact solver
from itertools import combinations
from math import comb

import numpy as np

from data import COVERED, neighbors


def _comb(n, k):
    return comb(n, k) if 0 <= k <= n else 0


def _convolve(a, b):
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] += x * y
    return out


def components(visible):
    """Split the frontier into independent groups.

    Returns (groups, unconstrained): each group is (cells, constraints) where
    constraints are (tuple of indices into cells, mines needed).
    """
    h, w = visible.shape
    cons = []
    for r, c in np.argwhere(visible != COVERED).tolist():
        cov = [n for n in neighbors(r, c, h, w) if visible[n] == COVERED]
        if not 0 <= visible[r, c] <= len(cov):
            raise ValueError(f"impossible clue at {(r, c)}")
        if cov:
            cons.append((cov, int(visible[r, c])))

    parent = {cell: cell for cov, _ in cons for cell in cov}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for cov, _ in cons:
        for cell in cov[1:]:
            parent[find(cell)] = find(cov[0])

    grouped = {}
    for cov, need in cons:
        grouped.setdefault(find(cov[0]), []).append((cov, need))
    groups = []
    for gcons in grouped.values():
        cells = list(dict.fromkeys(cell for cov, _ in gcons for cell in cov))
        idx = {cell: i for i, cell in enumerate(cells)}
        groups.append((cells, [(tuple(idx[x] for x in cov), need) for cov, need in gcons]))
    unconstrained = [tuple(x) for x in np.argwhere(visible == COVERED).tolist() if tuple(x) not in parent]
    return groups, unconstrained


def enumerate_group(n, constraints):
    """Count every assignment satisfying the constraints, by number of mines k.

    Returns (counts[k], cell_counts[i][k]) as exact Python ints.
    """
    counts = [0] * (n + 1)
    cell_counts = [[0] * (n + 1) for _ in range(n)]
    of_cell = [[] for _ in range(n)]
    for ci, (cov, _) in enumerate(constraints):
        for i in cov:
            of_cell[i].append(ci)
    need = [nd for _, nd in constraints]          # mines still required
    free = [len(cov) for cov, _ in constraints]   # cells still unassigned
    assign = []

    def go(i):
        if i == n:
            k = sum(assign)
            counts[k] += 1
            for j, v in enumerate(assign):
                if v:
                    cell_counts[j][k] += 1
            return
        for v in (0, 1):
            ok = True
            for ci in of_cell[i]:
                need[ci] -= v
                free[ci] -= 1
                ok &= 0 <= need[ci] <= free[ci]
            if ok:
                assign.append(v)
                go(i + 1)
                assign.pop()
            for ci in of_cell[i]:
                need[ci] += v
                free[ci] += 1

    go(0)
    return counts, cell_counts


def exact_probabilities(visible, num_mines):
    """Exact mine probability of every covered cell given clues and total mine count.

    Revealed cells get NaN. Every layout consistent with the visible board is equally likely.
    """
    visible = np.asarray(visible)
    groups, unconstrained = components(visible)
    solved = [enumerate_group(len(cells), cons) for cells, cons in groups]
    u = len(unconstrained)
    total_poly = [1]
    for counts, _ in solved:
        total_poly = _convolve(total_poly, counts)
    total = sum(p * _comb(u, num_mines - k) for k, p in enumerate(total_poly))
    if total == 0:
        raise ValueError("no mine layout is consistent with this board")

    probs = np.full(visible.shape, np.nan)
    for g, ((cells, _), (_, cell_counts)) in enumerate(zip(groups, solved)):
        others = [1]
        for h, (counts, _) in enumerate(solved):
            if h != g:
                others = _convolve(others, counts)
        # rest[k] = weight of everything outside this group, given the group uses k mines
        rest = [sum(p * _comb(u, num_mines - k - j) for j, p in enumerate(others)) for k in range(len(cells) + 1)]
        for cell, per_k in zip(cells, cell_counts):
            probs[cell] = sum(x * r for x, r in zip(per_k, rest)) / total  # int / int rounds once
    if u:
        p = sum(c * _comb(u - 1, num_mines - k - 1) for k, c in enumerate(total_poly)) / total
        for cell in unconstrained:
            probs[cell] = p
    return probs


def brute_force_probabilities(visible, num_mines):
    """Reference answer: try every mine layout over the covered cells. Tiny boards only."""
    visible = np.asarray(visible)
    h, w = visible.shape
    covered = [tuple(x) for x in np.argwhere(visible == COVERED).tolist()]
    clues = np.argwhere(visible != COVERED).tolist()
    hits = np.zeros(len(covered), dtype=np.int64)
    n = 0
    for layout in combinations(range(len(covered)), num_mines):
        mines = {covered[i] for i in layout}
        if all(sum(x in mines for x in neighbors(r, c, h, w)) == visible[r, c] for r, c in clues):
            n += 1
            hits[list(layout)] += 1
    if n == 0:
        raise ValueError("no mine layout is consistent with this board")
    probs = np.full(visible.shape, np.nan)
    for cell, k in zip(covered, hits):
        probs[cell] = k / n
    return probs
