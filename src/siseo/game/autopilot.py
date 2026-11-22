"""A simple pilot used by demo mode: it decides which word to "say" next.

It runs a breadth-first search from the head to the food over free cells and
returns the first move of the shortest path; if the food is unreachable it
picks the safe move that keeps the most room around the head.
"""

from __future__ import annotations

from collections import deque

from .snake import DIRECTIONS, Cell, SnakeGame


def _free_cells(game: SnakeGame) -> set[Cell]:
    body = list(game.snake)[:-1]  # the tail will have moved by the next step
    return {(x, y) for y in range(game.cfg.rows) for x in range(game.cfg.cols)} - set(body)


def _reachable(start: Cell, free: set[Cell]) -> int:
    seen, todo = {start}, deque([start])
    while todo:
        x, y = todo.popleft()
        for dx, dy in DIRECTIONS.values():
            nxt = (x + dx, y + dy)
            if nxt in free and nxt not in seen:
                seen.add(nxt)
                todo.append(nxt)
    return len(seen)


def next_action(game: SnakeGame) -> str:
    """The direction the pilot wants to head in now (may equal the current one)."""
    free = _free_cells(game)
    head = game.head
    reverse = (-game.direction[0], -game.direction[1])
    options = [(name, d) for name, d in DIRECTIONS.items() if d != reverse]

    if game.food is not None:
        parents: dict[Cell, tuple[Cell, str] | None] = {head: None}
        todo = deque([head])
        while todo:
            cell = todo.popleft()
            if cell == game.food:
                break
            moves = options if cell == head else list(DIRECTIONS.items())
            for name, (dx, dy) in moves:
                nxt = (cell[0] + dx, cell[1] + dy)
                if nxt in free and nxt not in parents:
                    parents[nxt] = (cell, name)
                    todo.append(nxt)
        if game.food in parents:
            cell = game.food
            first = None
            while parents[cell] is not None:
                cell, first = parents[cell]  # type: ignore[misc]
            if first is not None:
                target = DIRECTIONS[first]
                nxt = (head[0] + target[0], head[1] + target[1])
                if _reachable(nxt, free) > len(game.snake):
                    return first

    best, best_room = None, -1
    for name, (dx, dy) in options:
        nxt = (head[0] + dx, head[1] + dy)
        if nxt in free:
            room = _reachable(nxt, free)
            if room > best_room:
                best, best_room = name, room
    if best is not None:
        return best
    for name, d in DIRECTIONS.items():
        if d == game.direction:
            return name
    return "right"
