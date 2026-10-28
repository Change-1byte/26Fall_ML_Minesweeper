# Minesweeper constraint reasoning

Can a learned model reproduce exact constraint-based mine probabilities beyond local patterns?
The exact solver is the oracle; models are scored against it, not on win rate.

- `data.py`: board generation (first click and its neighbours are mine-free), reveal, visible board (`-1` = covered)
- `solver.py`: `exact_probabilities(visible, num_mines)` gives the exact mine probability per covered cell (NaN on revealed cells), plus a brute-force reference
- `solver.py` also labels cells/states LOCAL / GLOBAL / MINE_COUNT / UNDETERMINED (`label`). LOCAL uses only clues within
  distance 3 of the cell, so each clue's whole neighbourhood lies in the 9x9 window the local CNN sees; LOCAL and GLOBAL ignore the mine count
- `data.py --num-games N`: the exact-solver agent plays N 16x16/40 games and saves every labelled state to `data/states.npz`;
  `split_mask` gives a 70/15/15 split by board
- `test_solver.py`: board tests, hand-built solver cases, 300 random tiny boards checked against brute force

```bash
uv venv --python 3.12 && source .venv/bin/activate   # or: python -m venv .venv
uv pip install numpy pytest                           # or: pip install numpy pytest
pytest -q
python data.py --num-games 300   # ~3 min, prints label stats
```

Label mix, 300 games (22,620 states, 1.15M frontier cells):

| label | states | frontier cells |
|---|---|---|
| LOCAL | 99.4% | 77.6% |
| GLOBAL | 0.0% (2) | 3.7% |
| MINE_COUNT | 0.0% (5) | 0.2% |
| UNDETERMINED | 0.6% | 18.5% |
