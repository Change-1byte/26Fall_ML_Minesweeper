# Training + evaluation + results
import json
import os

import numpy as np

from data import COVERED, frontier, split_mask
from solver import LABELS, SAFE_EPS

TIE_EPS = 1e-9  # exact risks closer than this count as tied


def _err(pred, probs, m):
    d = pred[m] - probs[m]
    return {"n": int(m.sum()), "mae": float(np.abs(d).mean()) if m.any() else None,
            "brier": float((d ** 2).mean()) if m.any() else None}


def probability_metrics(pred, d):
    """MAE / Brier vs solver probabilities on frontier cells, overall and by cell and state label."""
    fr = np.array([frontier(v) for v in d["visible"]])
    out = {"frontier": _err(pred, d["probs"], fr),
           "unconstrained": _err(pred, d["probs"], (d["visible"] == COVERED) & ~fr)}
    for i, name in enumerate(LABELS):
        out[f"cell_{name}"] = _err(pred, d["probs"], fr & (d["cell_labels"] == i))
        out[f"state_{name}"] = _err(pred, d["probs"], fr & (d["state_label"] == i)[:, None, None])
    return out


def calibration(pred, d, bins=10):
    """Per probability bin of the prediction (frontier cells): count, mean prediction, mean solver p, mine rate."""
    fr = np.array([frontier(v) for v in d["visible"]])
    p, s, mine = pred[fr], d["probs"][fr], d["mines"][fr]
    b = np.minimum((p * bins).astype(int), bins - 1)
    return [{"bin": f"{i / bins:.1f}-{(i + 1) / bins:.1f}", "n": int((b == i).sum()),
             "pred": float(p[b == i].mean()), "solver": float(s[b == i].mean()), "mine_rate": float(mine[b == i].mean())}
            for i in range(bins) if (b == i).any()]


def decide(pred, probs, candidates, rng):
    """Model picks argmin predicted risk among candidates (random tie-break).

    Returns (chosen exact risk, best exact risk) per state; NaN where a state has no candidate.
    """
    chosen, best = np.full(len(pred), np.nan), np.full(len(pred), np.nan)
    for i, (p, s, c) in enumerate(zip(pred, probs, candidates)):
        if c.any():
            ties = np.argwhere(c & (p == p[c].min()))
            chosen[i] = s[tuple(ties[rng.integers(len(ties))])]
            best[i] = s[c].min()
    return chosen, best


def _decision_summary(chosen, best, m):
    m = m & ~np.isnan(best)
    if not m.any():
        return {"n": 0}
    excess = chosen[m] - best[m]
    has_safe = best[m] <= SAFE_EPS
    # reasoning-error rate = 1 - safest_move_acc, so it is not reported separately
    return {"n": int(m.sum()), "safest_move_acc": float((excess <= TIE_EPS).mean()),
            "mean_excess_risk": float(excess.mean()),
            "missed_safe_rate": float((chosen[m][has_safe] > SAFE_EPS).mean()) if has_safe.any() else None}


def decision_metrics(pred, d, seed=0):
    """Decision quality per state label, twice: choosing from all covered cells, and with LOCAL cells
    removed from the choice (the hard decision; grouped by the label of the most local safe cell left)."""
    rng = np.random.default_rng(seed)
    covered, cells = d["visible"] == COVERED, d["cell_labels"]
    out = {}
    for name, cand in [("all", covered), ("no_local", covered & (cells != 0))]:
        chosen, best = decide(pred, d["probs"], cand, rng)
        safe_left = cand & (d["probs"] <= SAFE_EPS)
        group = np.where(safe_left, cells, 3).min(axis=(1, 2))  # 3 = UNDETERMINED when nothing safe is left
        out[name] = {"overall": _decision_summary(chosen, best, np.ones(len(group), bool))}
        out[name] |= {lab: _decision_summary(chosen, best, group == i) for i, lab in enumerate(LABELS)}
    return out


def evaluate(pred, d):
    return {"probability": probability_metrics(pred, d), "calibration": calibration(pred, d),
            "decisions": decision_metrics(pred, d)}


def load(path, split):
    z = np.load(path)
    m = split_mask(z["board_id"], split)
    return {k: z[k][m] if z[k].ndim else z[k].item() for k in z.files}


if __name__ == "__main__":
    import argparse

    from model import density

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="density")
    ap.add_argument("--split", default="test")
    ap.add_argument("--data", default="data/states.npz")
    args = ap.parse_args()
    d = load(args.data, args.split)
    pred = density(d["visible"], d["num_mines"])
    res = evaluate(pred, d)
    os.makedirs("outputs/metrics", exist_ok=True)
    path = f"outputs/metrics/{args.model}_{args.split}.json"
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    p, dec = res["probability"], res["decisions"]
    print(path)
    print("frontier MAE", round(p["frontier"]["mae"], 4), "| by cell label:",
          {k[5:]: round(v["mae"], 4) for k, v in p.items() if k.startswith("cell_") and v["n"]})
    for name in dec:
        print(f"decisions[{name}]:", {k: v for k, v in dec[name].items() if v["n"]})
