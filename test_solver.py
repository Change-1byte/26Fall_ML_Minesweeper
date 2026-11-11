import numpy as np
import pytest

from data import COVERED, Board, generate, neighbors, split_mask
from solver import brute_force_probabilities, exact_probabilities, label

X = -1  # covered


def test_clues_and_flood_fill():
    b = Board([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    assert b.clues.tolist() == [[0, 1, 0], [1, 1, 0], [0, 0, 0]]
    b.reveal(2, 2)  # a zero: opens everything except the mine
    assert b.visible().tolist() == [[X, 1, 0], [1, 1, 0], [0, 0, 0]]
    assert b.won


def test_new_game_safe_opening_and_seeded():
    for seed in range(50):
        click = (seed % 16, seed * 7 % 16)
        b = Board.new_game(16, 16, 40, first_click=click, seed=seed)
        assert b.num_mines == 40 and not b.lost
        assert not any(b.mines[n] for n in [click, *neighbors(*click, 16, 16)])
    assert (Board.new_game(seed=7).mines == Board.new_game(seed=7).mines).all()


@pytest.mark.parametrize("visible, mines, expected", [
    # 1-1-1 under three covered cells: middle is a mine, sides safe
    ([[X, X, X], [1, 1, 1]], 1, [[0, 1, 0], [np.nan] * 3]),
    # 50/50
    ([[X, X], [1, 1]], 1, [[.5, .5], [np.nan] * 2]),
    # far cell (0,3) is decided only by the total count
    ([[X, 1, X, X]], 1, [[.5, np.nan, .5, 0]]),
    ([[X, 1, X, X]], 2, [[.5, np.nan, .5, 1]]),
    # two disconnected groups plus one unconstrained cell
    ([[X, 1, X, X, X, 1, X]], 2, [[.5, np.nan, .5, 0, .5, np.nan, .5]]),
])
def test_hand_built(visible, mines, expected):
    np.testing.assert_allclose(exact_probabilities(np.array(visible), mines), expected, atol=1e-12)


def test_inconsistent_board_raises():
    with pytest.raises(ValueError):
        exact_probabilities(np.array([[2, X]]), 1)
    with pytest.raises(ValueError):
        exact_probabilities(np.array([[1, X]]), 0)


def random_tiny_state(rng):
    h, w = rng.integers(4, 6, size=2)
    mines = int(rng.integers(1, min(7, h * w - 9) + 1))
    b = Board.new_game(int(h), int(w), mines, seed=int(rng.integers(1 << 30)))
    for _ in range(rng.integers(0, 4)):
        safe = np.argwhere(~b.revealed & ~b.mines)
        if len(safe):
            b.reveal(*safe[rng.integers(len(safe))])
    return b.visible(), mines


def test_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(300):
        visible, mines = random_tiny_state(rng)
        np.testing.assert_allclose(exact_probabilities(visible, mines),
                                   brute_force_probabilities(visible, mines), atol=1e-12)


LOCAL, GLOBAL, MINE_COUNT, UNDET = range(4)


@pytest.mark.parametrize("visible, mines, state", [
    ([[X, X, X], [1, 1, 1]], 1, LOCAL),
    # chain of clues: the end clues force mines at columns 2 and 9 locally,
    # but every safe cell needs clues more than 3 columns away
    ([[X] * 12, [1] + [2] * 10 + [1]], 8, GLOBAL),
    ([[X, 1, X, X]], 1, MINE_COUNT),
    ([[X, X], [1, 1]], 1, UNDET),
])
def test_state_labels(visible, mines, state):
    v = np.array(visible)
    assert label(v, exact_probabilities(v, mines))[1] == state


def test_global_example_cells():
    v = np.array([[X] * 12, [1] + [2] * 10 + [1]])
    cells, _ = label(v, exact_probabilities(v, 8))
    assert cells[0].tolist() == [GLOBAL] * 2 + [LOCAL] + [GLOBAL] * 6 + [LOCAL] + [GLOBAL] * 2


def test_generated_games():
    d = generate(2, seed=0)
    covered = d["visible"] == X
    assert ((d["cell_labels"] == -1) == ~covered).all()
    decided = (d["probs"] <= 1e-12) | (d["probs"] >= 1 - 1e-12)
    # local/global-decided cells must be decided by the full solver, with the true mine value
    early = covered & (d["cell_labels"] <= GLOBAL)
    assert decided[early].all() and (d["probs"][early].round() == d["mines"][early]).all()
    # every board lands in exactly one split
    ids = np.arange(100)
    assert (split_mask(ids, "train").astype(int) + split_mask(ids, "val") + split_mask(ids, "test") == 1).all()


def test_evaluation_scores_oracle_perfectly():
    from main import evaluate
    from model import density

    d = generate(3, seed=1)
    res = evaluate(d["probs"], d)
    assert res["probability"]["frontier"]["mae"] == 0
    for dec in res["decisions"].values():
        assert dec["overall"]["safest_move_acc"] == 1 and dec["overall"]["missed_safe_rate"] == 0
    base = evaluate(density(d["visible"], 40), d)
    assert base["probability"]["frontier"]["mae"] > 0.05


def test_cnn_shapes_and_receptive_field():
    import torch
    from model import CNN, encode

    torch.manual_seed(0)
    v = Board.new_game(16, 16, 40, seed=0).visible()
    for g in (False, True):
        m = CNN(8, global_pool=g)
        assert m(encode(np.stack([v, v]), 40)).shape == (2, 16, 16)
        assert m(encode(np.full((1, 9, 30), X), 10)).shape == (1, 9, 30)  # any board size
        a = encode(np.full((1, 16, 16), X), 40)
        b = a.clone()
        b[0, :, 0, 15] = 0
        b[0, 1, 0, 15] = 1  # reveal a zero 5 columns away from target (0, 10)
        changed = (m(a)[0, 0, 10] - m(b)[0, 0, 10]).abs() > 0
        assert changed == g  # only the global model sees outside the 9x9 window


def test_bootstrap_paired_and_pooled():
    from main import bootstrap

    stats = {"x": [[1.0, 3.0, 0.0], [2.0, 2.0, 0.0]]}  # board sums / counts; third board has no cells
    rows = {(m, k): r for m, k, *r in bootstrap({"local": stats, "global": stats}, n=200)}
    assert rows[("local", "x")][0] == 1.0  # pooled: 4/4, not the mean of board means
    assert rows[("global-local", "x")] == [0.0, 0.0, 0.0]  # identical models -> paired diff is exactly 0


def test_window_probabilities():
    from solver import exact_probabilities, window_probabilities

    rng = np.random.default_rng(1)
    checked = 0
    for _ in range(20):
        b = Board.new_game(7, 7, 8, seed=rng)  # small board, so many cells have every clue in their window
        v = b.visible()
        w = window_probabilities(v, 8)
        # a cell whose window holds every clue sees the full problem, so must match the exact solver
        p = exact_probabilities(v, 8)
        clues = np.argwhere(v != COVERED)
        for r, c in np.argwhere(v == COVERED).tolist():
            if (np.abs(clues - (r, c)).max(axis=1) <= 3).all():
                assert np.isclose(w[r, c], p[r, c])
                checked += 1
        assert np.isnan(w[v != COVERED]).all()
    assert checked > 20  # not vacuous
