"""Pure Snake game logic (no rendering) and the demo autopilot."""

from .autopilot import next_action
from .snake import DIRECTIONS, OVER, PAUSED, READY, RUNNING, GameConfig, SnakeGame

__all__ = ["DIRECTIONS", "OVER", "PAUSED", "READY", "RUNNING", "GameConfig", "SnakeGame", "next_action"]
