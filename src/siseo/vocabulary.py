"""The words Siseo listens for and what each one does in the game."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    name: str
    """Stored identifier (upper case)."""
    word: str
    """What the player says."""
    action: str
    """Game action: up, down, left, right or pause."""
    glyph: str


VOCABULARY: tuple[Command, ...] = (
    Command("ARRIBA", "arriba", "up", "↑"),
    Command("ABAJO", "abajo", "down", "↓"),
    Command("IZQUIERDA", "izquierda", "left", "←"),
    Command("DERECHA", "derecha", "right", "→"),
    Command("PAUSA", "pausa", "pause", "Ⅱ"),
)

BY_NAME: dict[str, Command] = {c.name: c for c in VOCABULARY}
NAMES: tuple[str, ...] = tuple(c.name for c in VOCABULARY)

RECOMMENDED_TAKES = 5
"""Takes per command shown as a full training bar; three already work well."""
MIN_TAKES_READY = 2


def action_for(command: str | None) -> str | None:
    if command is None:
        return None
    entry = BY_NAME.get(command.upper())
    return entry.action if entry else None
