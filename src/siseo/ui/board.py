"""The Snake board, painted with QPainter from the pure game state."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..game import OVER, PAUSED, READY, SnakeGame
from . import theme


def _mix(a: str, b: str, t: float) -> QColor:
    ca, cb = QColor(a), QColor(b)
    return QColor(
        int(ca.red() + (cb.red() - ca.red()) * t),
        int(ca.green() + (cb.green() - ca.green()) * t),
        int(ca.blue() + (cb.blue() - ca.blue()) * t),
    )


class BoardView(QWidget):
    def __init__(self, game: SnakeGame, parent: QWidget | None = None):
        super().__init__(parent)
        self.game = game
        self.best = 0
        self.top_scores: list[tuple[int, str, str]] = []
        self.ready_hint = "Di una dirección o pulsa una flecha"
        self.flash = 0.0
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(480, 360)

    def sizeHint(self) -> QSize:
        return QSize(720, 540)

    def _geometry(self) -> tuple[float, QRectF]:
        cols, rows = self.game.cfg.cols, self.game.cfg.rows
        cell = min((self.width() - 8) / cols, (self.height() - 8) / rows)
        w, h = cell * cols, cell * rows
        return cell, QRectF((self.width() - w) / 2, 6, w, h)

    def paintEvent(self, event) -> None:  # noqa: ARG002
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        cell, area = self._geometry()
        p.setPen(QPen(theme.color(theme.LINE), 1))
        p.setBrush(theme.color(theme.PANEL))
        p.drawRoundedRect(area.adjusted(-4, -4, 4, 4), 16, 16)

        # Dot grid, like the sampling grid of a spectrogram.
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(theme.color(theme.LINE))
        for y in range(self.game.cfg.rows):
            for x in range(self.game.cfg.cols):
                p.drawEllipse(QPointF(area.left() + (x + 0.5) * cell, area.top() + (y + 0.5) * cell), 1.3, 1.3)

        if self.game.food is not None:
            fx, fy = self.game.food
            centre = QPointF(area.left() + (fx + 0.5) * cell, area.top() + (fy + 0.5) * cell)
            glow = QRadialGradient(centre, cell * 1.3)
            glow.setColorAt(0, theme.color(theme.EMBER, 120))
            glow.setColorAt(1, theme.color(theme.EMBER, 0))
            p.setBrush(glow)
            p.drawEllipse(centre, cell * 1.3, cell * 1.3)
            p.setBrush(theme.color(theme.EMBER))
            p.drawEllipse(centre, cell * 0.32, cell * 0.32)

        body = list(self.game.snake)
        n = len(body)
        for i, (x, y) in enumerate(reversed(body)):
            t = i / max(1, n - 1)  # 0 at the tail, 1 at the head
            inset = cell * (0.16 - 0.08 * t)
            rect = QRectF(area.left() + x * cell + inset, area.top() + y * cell + inset, cell - 2 * inset, cell - 2 * inset)
            p.setBrush(_mix(theme.MOSS, theme.VENOM, 0.25 + 0.75 * t))
            p.drawRoundedRect(rect, cell * 0.3, cell * 0.3)
        if body:
            hx, hy = body[0]
            dx, dy = self.game.direction
            centre = QPointF(area.left() + (hx + 0.5) * cell, area.top() + (hy + 0.5) * cell)
            side = QPointF(-dy * cell * 0.18, dx * cell * 0.18)
            ahead = QPointF(dx * cell * 0.14, dy * cell * 0.14)
            p.setBrush(theme.color(theme.INK))
            for sgn in (1, -1):
                p.drawEllipse(centre + ahead + side * sgn, cell * 0.07, cell * 0.07)

        if self.flash > 0:
            p.setPen(QPen(theme.color(theme.VENOM, int(160 * self.flash)), 3))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(area.adjusted(-4, -4, 4, 4), 16, 16)

        status = self.game.status
        if status == READY:
            self._banner(p, area, "listo", self.ready_hint)
        elif status == PAUSED:
            self._banner(p, area, "pausa", "Di «pausa» otra vez o pulsa P para seguir")
        elif status == OVER:
            self._game_over(p, area)
        p.end()

    def _veil(self, p: QPainter, area: QRectF) -> None:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(theme.color(theme.INK, 185))
        p.drawRoundedRect(area.adjusted(-4, -4, 4, 4), 16, 16)

    def _banner(self, p: QPainter, area: QRectF, title: str, hint: str) -> None:
        self._veil(p, area)
        p.setPen(theme.color(theme.VENOM))
        p.setFont(theme.font("display", 30, QFont.Weight.DemiBold))
        p.drawText(area.adjusted(0, -30, 0, -30), Qt.AlignmentFlag.AlignCenter, title)
        p.setPen(theme.color(theme.MUTED))
        p.setFont(theme.font("body", 11))
        p.drawText(area.adjusted(0, 34, 0, 34), Qt.AlignmentFlag.AlignCenter, hint)

    def _game_over(self, p: QPainter, area: QRectF) -> None:
        self._veil(p, area)
        g = self.game
        cx, cy = area.center().x(), area.center().y()
        p.setPen(theme.color(theme.EMBER if not g.won else theme.VENOM))
        p.setFont(theme.font("display", 13))
        p.drawText(QRectF(area.left(), cy - 150, area.width(), 24), Qt.AlignmentFlag.AlignCenter,
                   "¡tablero completo!" if g.won else "fin de la partida")
        p.setPen(theme.color(theme.TEXT))
        p.setFont(theme.font("mono", 54, QFont.Weight.Bold))
        p.drawText(QRectF(area.left(), cy - 128, area.width(), 80), Qt.AlignmentFlag.AlignCenter, str(g.score))
        p.setFont(theme.font("display", 11))
        record = g.score >= self.best and g.score > 0
        p.setPen(theme.color(theme.VENOM if record else theme.MUTED))
        p.drawText(QRectF(area.left(), cy - 46, area.width(), 22), Qt.AlignmentFlag.AlignCenter,
                   "nuevo récord" if record else f"récord: {self.best}")
        if self.top_scores:
            p.setPen(theme.color(theme.MUTED))
            p.setFont(theme.font("display", 9))
            p.drawText(QRectF(area.left(), cy - 8, area.width(), 18), Qt.AlignmentFlag.AlignCenter, "MEJORES PARTIDAS")
            p.setFont(theme.font("mono", 10.5))
            for i, (score, mode, _) in enumerate(self.top_scores[:5]):
                y = cy + 16 + i * 22
                p.setPen(theme.color(theme.TEXT if i == 0 else theme.MUTED))
                p.drawText(QRectF(cx - 110, y, 60, 20), Qt.AlignmentFlag.AlignRight, f"{i + 1}.")
                p.drawText(QRectF(cx - 40, y, 50, 20), Qt.AlignmentFlag.AlignRight, str(score))
                p.drawText(QRectF(cx + 24, y, 100, 20), Qt.AlignmentFlag.AlignLeft, mode)
        p.setPen(theme.color(theme.TEXT))
        p.setFont(theme.font("body", 10.5))
        p.drawText(QRectF(area.left(), area.bottom() - 44, area.width(), 22), Qt.AlignmentFlag.AlignCenter,
                   "Enter o di una dirección para jugar otra vez")
