# Board figures for the report
import os

import numpy as np

from data import COVERED, frontier
from solver import LABELS

LABEL_EDGE = {1: "#1b4f9c", 2: "#b8860b"}  # GLOBAL, MINE_COUNT: outlined in the probability panel


def pick_state(d, label="GLOBAL", min_covered=70, max_covered=170):
    """Index of a readable state that actually contains cells with the given label."""
    want = LABELS.index(label)
    covered = (d["visible"] == COVERED).sum(axis=(1, 2))
    fr = np.array([frontier(v) for v in d["visible"]])
    n_want = (fr & (d["cell_labels"] == want)).sum(axis=(1, 2))
    ok = (covered >= min_covered) & (covered <= max_covered) & (n_want > 0)
    if not ok.any():
        raise ValueError(f"no state with {label} cells in this size range")
    return int(np.argmax(np.where(ok, n_want, -1)))


def plot_state(visible, probs, cell_labels=None, path="outputs/figures/board_example.png"):
    """Two panels: the visible board the model sees, and the exact probabilities it is trained on."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D

    h, w = visible.shape
    cmap = LinearSegmentedColormap.from_list("risk", ["#1b7a46", "#f4efc4", "#b2182b"])
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.5, 5.9))

    for r in range(h):
        for c in range(w):
            v = int(visible[r, c])
            y = h - 1 - r
            a0.add_patch(plt.Rectangle((c, y), 1, 1, lw=1.1, edgecolor="white",
                                       facecolor="#c9ccd1" if v == COVERED else "#f4f5f7"))
            if v > 0:
                a0.text(c + .5, y + .5, str(v), ha="center", va="center",
                        fontsize=9, fontweight="bold", color="#1f3a5f")
            elif v == COVERED:
                a0.text(c + .5, y + .5, "−1", ha="center", va="center", fontsize=6.2, color="#8a8f98")

            if v != COVERED:
                a1.add_patch(plt.Rectangle((c, y), 1, 1, lw=1.1, edgecolor="white", facecolor="#f4f5f7"))
                continue
            p = float(probs[r, c])
            lab = -1 if cell_labels is None else int(cell_labels[r, c])
            edge, lw = (LABEL_EDGE.get(lab), 2.2) if lab in LABEL_EDGE else ("white", 1.1)
            a1.add_patch(plt.Rectangle((c, y), 1, 1, lw=lw, edgecolor=edge, facecolor=cmap(p), zorder=lw))
            txt = "1" if p > 1 - 1e-9 else "0" if p < 1e-9 else f"{p:.2f}"[1:]
            a1.text(c + .5, y + .5, txt, ha="center", va="center", zorder=3,
                    fontsize=8.5 if len(txt) == 1 else 5.8,
                    fontweight="bold" if len(txt) == 1 else "normal",
                    color="white" if (p > .6 or p < .1) else "#2b2b2b")

    a0.set_title("Visible board — the model's input\ncovered cells are encoded as −1", fontsize=11)
    a1.set_title("Exact solver probabilities — the training target\n0 = provably safe, 1 = provably a mine", fontsize=11)
    present = [] if cell_labels is None else [i for i in LABEL_EDGE if (cell_labels == i).any()]
    if present:
        a1.legend(handles=[Line2D([], [], color=LABEL_EDGE[i], lw=2.2, label=f"{LABELS[i]} cell") for i in present],
                  loc="upper center", bbox_to_anchor=(.5, -.02), ncol=len(present), frameon=False, fontsize=9)
    for a in (a0, a1):
        a.set_xlim(0, w)
        a.set_ylim(0, h)
        a.set_aspect("equal")
        a.axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(path)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="python figures.py [--label GLOBAL]")
    ap.add_argument("--data", default="data/states.npz")
    ap.add_argument("--label", default="GLOBAL", choices=LABELS)
    ap.add_argument("--out", default="outputs/figures/board_example.png")
    args = ap.parse_args()
    z = np.load(args.data)
    d = {k: z[k] for k in ("visible", "probs", "cell_labels")}
    i = pick_state(d, args.label)
    fr = frontier(d["visible"][i])
    print(f"state {i}: covered {(d['visible'][i] == COVERED).sum()}, frontier {fr.sum()}, "
          + ", ".join(f"{n} {(fr & (d['cell_labels'][i] == j)).sum()}" for j, n in enumerate(LABELS)))
    plot_state(d["visible"][i], d["probs"][i], d["cell_labels"][i], args.out)
