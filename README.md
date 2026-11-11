# Minesweeper constraint reasoning

Can a learned model reproduce exact constraint-based mine probabilities beyond local patterns?
The exact solver is the oracle; models are scored against it, not on win rate.

- `data.py`: board generation (first click and its neighbours are mine-free), reveal, visible board (`-1` = covered)
- `solver.py`: `exact_probabilities(visible, num_mines)` gives the exact mine probability per covered cell (NaN on revealed cells), plus a brute-force reference
- `solver.py` also labels cells/states LOCAL / GLOBAL / MINE_COUNT / UNDETERMINED (`label`). LOCAL uses only clues within
  distance 3 of the cell, so each clue's whole neighbourhood lies in the 9x9 window the local CNN sees; LOCAL and GLOBAL ignore the mine count
- `data.py --num-games N`: the exact-solver agent plays N 16x16/40 games and saves every labelled state to `data/states.npz`;
  `split_mask` gives a 70/15/15 split by board
- `model.py`: density baseline (remaining mines / covered cells) and `CNN`: four 3x3 convs (exactly a 9x9 view) + 1x1 head;
  `global_pool=True` adds a whole-board average -> MLP -> broadcast branch. Input: covered mask, one-hot clues 0-8,
  remaining-mine density (given to both, as in the proposal, though LOCAL labels exclude the mine count)
- `main.py`: training on solver probabilities (soft BCE on covered cells, best val frontier MAE kept) and evaluation:
  MAE/Brier on frontier cells by cell and state label, calibration, decision metrics. Decisions are scored twice:
  over all covered cells, and with LOCAL cells removed from the choice ("no_local"), because ~99% of states have a
  local safe move and would otherwise hide every hard decision
- `bench.py --num-games N`: solver time per state, largest constraint group, solver-agent win rate
- `main.py report` also writes `bootstrap_test.csv`: 95% CIs from 1000 resamples of test boards, plus paired
  differences vs the local CNN (rows `model-local`)
- `main.py train <model> --seed S` (S > 0) saves as `<model>_sS`; `report` writes `seeds_test.csv` (mean/sd/min/max over
  the seed-0 run and every `_sS` run)
- `test_solver.py`: board tests, hand-built solver cases, 300 random tiny boards checked against brute force

```bash
uv venv --python 3.12 && source .venv/bin/activate   # or: python -m venv .venv
uv pip install numpy pytest matplotlib torch --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
pytest -q
python data.py --num-games 300   # ~3 min, prints label stats
python main.py train local       # Experiment 1
python main.py train global
python main.py train local_bal && python main.py train dilated && python main.py train dilated_bal   # ablations: label-balanced loss, 31x31 view
for m in density window local global local_bal dilated dilated_bal; do python main.py eval $m; done   # window: exact solver on 9x9 clues (~1 min)
python bench.py --num-games 20   # solver timing + win rate
python main.py report            # outputs/metrics/summary_test.csv, outputs/figures/experiment1_test.png
```

Label mix, 300 games (22,620 states, 1.15M frontier cells):

| label | states | frontier cells |
|---|---|---|
| LOCAL | 99.4% | 77.6% |
| GLOBAL | 0.0% (2) | 3.7% |
| MINE_COUNT | 0.0% (5) | 0.2% |
| UNDETERMINED | 0.6% | 18.5% |

Experiment 1, preliminary (300 games, test boards, 1 seed, 20 epochs, CPU ~3.5 min per model):

| model | params | frontier MAE | LOCAL | GLOBAL | MINE_COUNT | UNDET. | safest move (all) | safest move, no-LOCAL: GLOBAL / MINE_COUNT |
|---|---|---|---|---|---|---|---|---|
| density | 0 | 0.399 | 0.446 | 0.451 | 0.531 | 0.202 | 0.151 | 0.089 / 0.581 |
| local CNN | 121k | 0.060 | 0.018 | 0.283 | 0.363 | 0.182 | 0.994 | 0.414 / 0.527 |
| global CNN | 130k | 0.059 | 0.016 | 0.296 | 0.372 | 0.182 | 0.992 | 0.427 / 0.634 |

Cell-label columns are frontier MAE vs the solver. No-LOCAL decision groups have n = 1911 / 93 states.
