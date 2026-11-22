"""Snake rules on a grid, with no rendering and no clock.

The UI calls :meth:`SnakeGame.step` on a timer and :meth:`SnakeGame.act` for
every key press or recognized word; everything here is deterministic for a
given random seed, which keeps it easy to test.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

Cell = tuple[int, int]

DIRECTIONS: dict[str, Cell] = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}

READY, RUNNING, PAUSED, OVER = "ready", "running", "paused", "over"


@dataclass(frozen=True)
class GameConfig:
    cols: int = 24
    rows: int = 18
    start_length: int = 3
    base_speed: float = 4.0
    """Cells per second at the start. Voice is slower than fingers, so it starts calm."""
    speed_step: float = 0.35
    speedup_every: int = 3
    max_speed: float = 10.0


class SnakeGame:
    def __init__(self, cfg: GameConfig = GameConfig(), rng: random.Random | None = None):
        if cfg.cols < cfg.start_length + 2 or cfg.rows < 3:
            raise ValueError("board too small")
        self.cfg = cfg
        self.rng = rng if rng is not None else random.Random()
        self.restart()

    # --- state ---------------------------------------------------------
    def restart(self) -> None:
        cx, cy = self.cfg.cols // 2, self.cfg.rows // 2
        self.snake: deque[Cell] = deque((cx - i, cy) for i in range(self.cfg.start_length))
        self.direction: Cell = DIRECTIONS["right"]
        self._queue: deque[Cell] = deque(maxlen=2)
        self.score = 0
        self.status = READY
        self.won = False
        self.steps = 0
        self.food: Cell | None = self._spawn_food()

    @property
    def head(self) -> Cell:
        return self.snake[0]

    @property
    def speed(self) -> float:
        """Cells per second for the current score."""
        boost = (self.score // self.cfg.speedup_every) * self.cfg.speed_step
        return min(self.cfg.max_speed, self.cfg.base_speed + boost)

    @property
    def interval_ms(self) -> int:
        return int(round(1000.0 / self.speed))

    def _spawn_food(self) -> Cell | None:
        occupied = set(self.snake)
        free = [(x, y) for y in range(self.cfg.rows) for x in range(self.cfg.cols) if (x, y) not in occupied]
        return self.rng.choice(free) if free else None

    # --- input ---------------------------------------------------------
    @property
    def heading(self) -> Cell:
        """Direction the snake will have after the queued turns."""
        return self._queue[-1] if self._queue else self.direction

    def act(self, action: str | None) -> bool:
        """Apply a player action: a direction, 'pause' or 'start'. Returns True if it did something."""
        if action is None:
            return False
        if action == "pause":
            return self.toggle_pause()
        if action == "start":
            if self.status == OVER:
                self.restart()
            if self.status in (READY, PAUSED):
                self.status = RUNNING
                return True
            return False
        if action not in DIRECTIONS or self.status == OVER:
            return False
        new = DIRECTIONS[action]
        last = self.heading
        if new == last and self.status in (READY, PAUSED):
            self.status = RUNNING  # saying the current heading just starts moving
            return True
        if new == last or (new[0] == -last[0] and new[1] == -last[1]):
            return False
        self._queue.append(new)
        if self.status in (READY, PAUSED):
            self.status = RUNNING
        return True

    def toggle_pause(self) -> bool:
        if self.status == RUNNING:
            self.status = PAUSED
            return True
        if self.status == PAUSED:
            self.status = RUNNING
            return True
        return False

    # --- simulation ----------------------------------------------------
    def step(self) -> str:
        """Advance one cell. Returns 'idle', 'moved', 'ate', 'died' or 'won'."""
        if self.status != RUNNING:
            return "idle"
        if self._queue:
            self.direction = self._queue.popleft()
        hx, hy = self.head
        nx, ny = hx + self.direction[0], hy + self.direction[1]
        self.steps += 1
        if not (0 <= nx < self.cfg.cols and 0 <= ny < self.cfg.rows):
            self.status = OVER
            return "died"
        eating = (nx, ny) == self.food
        body = list(self.snake)
        if not eating:
            body = body[:-1]  # the tail moves away this tick
        if (nx, ny) in body:
            self.status = OVER
            return "died"
        self.snake.appendleft((nx, ny))
        if not eating:
            self.snake.pop()
            return "moved"
        self.score += 1
        self.food = self._spawn_food()
        if self.food is None:
            self.status = OVER
            self.won = True
            return "won"
        return "ate"
