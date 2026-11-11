# Solver timing and solver-agent win rate (the numbers in the MVP report's solver-check table)
import time

import numpy as np

from data import Board
from solver import components, exact_probabilities


def bench(num_games, seed=0, h=16, w=16, num_mines=40):
    """Exact-solver agent plays num_games games; time every solve. Same agent and seeding as data.generate."""
    times, largest, wins = [], 0, 0
    for g in range(num_games):
        rng = np.random.default_rng([seed, g])
        b = Board.new_game(h, w, num_mines, seed=rng)
        while not (b.lost or b.won):
            v = b.visible()
            t = time.perf_counter()
            p = exact_probabilities(v, num_mines)
            times.append(time.perf_counter() - t)
            largest = max([largest] + [len(cells) for cells, _ in components(v)[0]])
            best = np.argwhere(p == np.nanmin(p))
            b.reveal(*best[rng.integers(len(best))])
        wins += b.won
    t = np.array(times) * 1000
    print(f"{h}x{w}/{num_mines}, {num_games} games, {len(t)} states")
    print(f"solve ms: mean {t.mean():.1f}  p95 {np.percentile(t, 95):.1f}  max {t.max():.1f}")
    print(f"largest constraint group: {largest} cells")
    print(f"solver agent wins: {wins}/{num_games} ({wins / num_games:.1%})")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--num-games", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--height", type=int, default=16)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--mines", type=int, default=40)
    a = ap.parse_args()
    bench(a.num_games, a.seed, a.height, a.width, a.mines)
