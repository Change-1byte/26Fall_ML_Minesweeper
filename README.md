# Minesweeper constraint reasoning

Can a learned model reproduce exact constraint-based mine probabilities beyond local patterns?
The exact solver is the oracle; models are scored against it, not on win rate.

- `data.py`: board generation (first click and its neighbours are mine-free), reveal, visible board (`-1` = covered)
- `solver.py`: `exact_probabilities(visible, num_mines)` gives the exact mine probability per covered cell (NaN on revealed cells), plus a brute-force reference
- `test_solver.py`: board tests, hand-built solver cases, 300 random tiny boards checked against brute force

```bash
uv venv --python 3.12 && source .venv/bin/activate   # or: python -m venv .venv
uv pip install numpy pytest                           # or: pip install numpy pytest
pytest -q
```
