import random

import pytest

from siseo.game import OVER, PAUSED, READY, RUNNING, GameConfig, SnakeGame, next_action


def make(cols=10, rows=8, seed=0):
    return SnakeGame(GameConfig(cols=cols, rows=rows), random.Random(seed))


def test_starts_ready_centered_heading_right():
    g = make()
    assert g.status == READY
    assert list(g.snake) == [(5, 4), (4, 4), (3, 4)]
    assert g.food not in g.snake
    assert g.step() == "idle"


def test_direction_starts_game_and_moves_one_cell():
    g = make()
    assert g.act("up")
    assert g.status == RUNNING
    assert g.step() == "moved"
    assert g.head == (5, 3)
    assert len(g.snake) == 3


def test_reverse_and_repeated_turns_are_ignored():
    g = make()
    g.act("start")
    assert not g.act("left")  # straight back into the body
    assert not g.act("right")  # already heading right
    assert g.act("down")
    assert not g.act("up")  # reverse of the queued turn
    assert g.act("left")  # two quick turns are both kept
    g.step()
    g.step()
    assert g.head == (4, 5) and g.direction == (-1, 0)


def test_current_heading_starts_a_ready_game():
    g = make()
    assert g.act("right")
    assert g.status == RUNNING
    assert not g.act("right")


def test_eating_grows_scores_and_respawns_food():
    g = make()
    g.food = (6, 4)
    g.act("start")
    assert g.step() == "ate"
    assert g.score == 1 and len(g.snake) == 4
    assert g.food is not None and g.food not in g.snake


def test_wall_collision_ends_game():
    g = make(cols=6, rows=4)
    g.food = (0, 0)
    g.act("start")
    results = [g.step() for _ in range(3)]
    assert results[-1] == "died"
    assert g.status == OVER
    assert not g.act("up")


def test_self_collision_but_tail_cell_is_free():
    g = make()
    g.snake.clear()
    g.snake.extend([(5, 4), (5, 5), (4, 5), (4, 4)])  # head next to its own tail
    g.direction = (0, -1)
    g.food = (0, 0)
    g.act("left")
    assert g.step() == "moved"  # the tail leaves (4, 4) this tick
    g.snake.clear()
    g.snake.extend([(5, 4), (5, 5), (4, 5), (4, 4), (4, 3)])
    g.direction = (0, -1)
    g._queue.clear()
    g.act("left")
    assert g.step() == "died"


def test_pause_toggle_and_restart_after_game_over():
    g = make()
    g.act("start")
    assert g.act("pause") and g.status == PAUSED
    assert g.step() == "idle"
    assert g.act("pause") and g.status == RUNNING
    g.status = OVER
    g.score = 7
    assert g.act("start")
    assert g.status == RUNNING and g.score == 0


def test_speed_grows_with_score_and_is_capped():
    g = SnakeGame(GameConfig(base_speed=4, speed_step=1, speedup_every=2, max_speed=6))
    assert g.speed == 4 and g.interval_ms == 250
    g.score = 4
    assert g.speed == 6
    g.score = 100
    assert g.speed == 6


def test_filling_the_board_wins():
    g = SnakeGame(GameConfig(cols=5, rows=3, start_length=3), random.Random(1))
    g.direction = (1, 0)
    g.status = RUNNING
    g.snake.clear()
    # The snake fills every cell except the food, with its head right next to it.
    path = [(3, 2), (2, 2), (1, 2), (0, 2), (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (4, 0), (3, 0), (2, 0), (1, 0), (0, 0)]
    g.snake.extend(path)
    g.food = (4, 2)
    assert g.step() == "won"
    assert g.won and g.status == OVER


def test_board_too_small_is_rejected():
    with pytest.raises(ValueError):
        SnakeGame(GameConfig(cols=4, rows=2))


@pytest.mark.parametrize("seed", range(5))
def test_autopilot_eats_and_survives(seed):
    g = SnakeGame(GameConfig(cols=12, rows=10), random.Random(seed))
    g.act("start")
    for _ in range(400):
        g.act(next_action(g))
        if g.step() in ("died", "won"):
            break
    assert g.score >= 8
