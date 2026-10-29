# Training + evaluation + results
import json
import os

import numpy as np
import torch
from torch import nn

from data import COVERED, frontier, split_mask
from solver import LABELS, SAFE_EPS

TIE_EPS = 1e-9  # exact risks closer than this count as tied


def _err(pred, probs, m):
    """Calculate prediction errors for cells selected by the mask."""
    d = pred[m] - probs[m]
    n = int(m.sum())
    if n == 0:
        return {"n": 0, "mae": None, "brier": None}
    mae = float(np.abs(d).mean())
    brier = float((d ** 2).mean())
    return {"n": n, "mae": mae, "brier": brier}


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


def predict(model, d, batch=512):
    """Mine probabilities [N,H,W] from a CNN; NaN on revealed cells like the solver output."""
    from model import encode

    model.eval()
    with torch.no_grad():
        p = torch.cat([torch.sigmoid(model(encode(d["visible"][i:i + batch], d["num_mines"])))
                       for i in range(0, len(d["visible"]), batch)]).numpy().astype(float)
    return np.where(d["visible"] == COVERED, p, np.nan)


def train(name, global_pool, data, epochs, batch, lr, weight_decay, width, seed):
    """Train on solver probabilities (soft-label BCE on covered cells); keep the best val frontier MAE."""
    from model import CNN, encode

    torch.manual_seed(seed)
    np.random.seed(seed)
    # Load training and validation states from different boards.
    tr, va = load(data, "train"), load(data, "val")
    # Prepare CNN inputs and use oracle probabilities as training labels
    x, y = encode(tr["visible"], tr["num_mines"]), torch.from_numpy(tr["probs"]).float()
    covered = torch.from_numpy(tr["visible"] == COVERED)
    y = torch.nan_to_num(y)
    model = CNN(width, global_pool)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{name}: {n_params} parameters, {len(x)} train states")
    best = float("inf")
    os.makedirs("outputs/checkpoints", exist_ok=True)
    for ep in range(epochs):        # Train the CNN using only covered cells
        model.train()
        perm, total = torch.randperm(len(x)), 0.0
        for i in range(0, len(x), batch):
            j = perm[i:i + batch]
            loss = nn.functional.binary_cross_entropy_with_logits(model(x[j])[covered[j]], y[j][covered[j]])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(j)
        # Evaluate after each epoch;
        # Save the model with the lowest validation MAE
        mae = probability_metrics(predict(model, va), va)["frontier"]["mae"]    # validation frontier MAE
        print(f"epoch {ep + 1}: train loss {total / len(x):.4f}  val frontier MAE {mae:.4f}")
        if mae < best:
            best = mae
            torch.save({"state_dict": model.state_dict(), "width": width, "global_pool": global_pool,
                        "params": n_params, "val_mae": mae, "epoch": ep + 1}, f"outputs/checkpoints/{name}.pt")


def report(split="test", models=("density", "local", "global")):
    """Experiment 1 comparison: summary CSV + figure from the saved metric JSONs."""
    import csv

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = {m: json.load(open(f"outputs/metrics/{m}_{split}.json")) for m in models}
    rows = []
    for m, r in res.items():
        p, d = r["probability"], r["decisions"]
        row = {"model": m, "frontier_mae": p["frontier"]["mae"], "frontier_brier": p["frontier"]["brier"]}
        row |= {f"{lab.lower()}_cell_mae": p[f"cell_{lab}"]["mae"] for lab in LABELS}
        for mode in d:
            for g, v in d[mode].items():
                if v["n"]:
                    for k in ("safest_move_acc", "mean_excess_risk", "missed_safe_rate"):
                        row[f"{mode}_{g.lower()}_{k}"] = v[k]
        rows.append(row)
    os.makedirs("outputs/figures", exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(f"outputs/metrics/summary_{split}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)

    colors = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3, fixed per model
    groups = ["LOCAL", "GLOBAL", "MINE_COUNT", "UNDETERMINED"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    x, bw = np.arange(len(groups)), 0.8 / len(models)
    for k, (m, r) in enumerate(res.items()):
        ax[0].bar(x + k * bw, [r["probability"][f"cell_{g}"]["mae"] for g in groups], bw - 0.02, color=colors[k], label=m)
        dec = r["decisions"]["no_local"]
        ax[1].bar(x[1:] + k * bw, [dec[g].get("safest_move_acc", np.nan) for g in groups[1:]], bw - 0.02, color=colors[k], label=m)
        cal = r["calibration"]
        ax[2].plot([c["pred"] for c in cal], [c["solver"] for c in cal], "-o", lw=2, ms=6, color=colors[k], label=m)
    ax[0].set(title="Frontier MAE vs solver, by cell label", ylabel="mean |p_model - p_solver|", xticks=x + bw, xticklabels=[g.replace("_", "\n") for g in groups])
    ax[1].set(title="Safest-move accuracy, LOCAL cells removed", ylabel="fraction of states", xticks=x[1:] + bw,
              xticklabels=[f"{g}\n(n={res[models[0]]['decisions']['no_local'][g]['n']})" for g in groups[1:]], ylim=(0, 1))
    ax[2].plot([0, 1], [0, 1], color="#999999", lw=1, ls="--")
    ax[2].set(title="Calibration (frontier cells)", xlabel="mean predicted p (per bin)", ylabel="mean solver p", xlim=(0, 1), ylim=(0, 1))
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", color="#e5e5e5", lw=0.8)
        a.set_axisbelow(True)
        a.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"outputs/figures/experiment1_{split}.png", dpi=150)
    print(f"outputs/metrics/summary_{split}.csv, outputs/figures/experiment1_{split}.png")


if __name__ == "__main__":
    import argparse

    from model import CNN, density

    ap = argparse.ArgumentParser(description="python main.py train local|global | eval density|local|global | report")
    ap.add_argument("cmd", choices=["train", "eval", "report"])
    ap.add_argument("model", nargs="?")
    ap.add_argument("--split", default="test")
    ap.add_argument("--data", default="data/states.npz")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "train":
        train(args.model, args.model == "global", args.data, args.epochs, args.batch_size, args.lr,
              args.weight_decay, args.width, args.seed)
        raise SystemExit
    if args.cmd == "report":
        report(args.split)
        raise SystemExit
    d = load(args.data, args.split)
    if args.model == "density":
        pred = density(d["visible"], d["num_mines"])
    else:
        ck = torch.load(f"outputs/checkpoints/{args.model}.pt")
        net = CNN(ck["width"], ck["global_pool"])
        net.load_state_dict(ck["state_dict"])
        pred = predict(net, d)
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
