import numpy as np
import pytest

from data import Board, neighbors
from solver import brute_force_probabilities, exact_probabilities

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
