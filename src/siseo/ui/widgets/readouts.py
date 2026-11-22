"""Readouts of a recognition: confidence gauge, distance ranking and take dots."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QFont, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from ...recognition import Score
from ...vocabulary import BY_NAME
from .. import theme


class ConfidenceGauge(QWidget):
    """A 270-degree arc showing confidence, with the percentage in the middle."""

    def __init__(self, parent: QWidget | None = None, diameter: int = 150):
        super().__init__(parent)
        self.setFixedSize(diameter, diameter)
        self.value: float | None = None
        self.accepted = True

    def set_value(self, value: float | None, accepted: bool = True) -> None:
        self.value, self.accepted = value, accepted
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ARG002
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        side = min(self.width(), self.height())
        rect = QRectF(9, 9, side - 18, side - 18)
        p.setPen(QPen(theme.color(theme.PANEL_2), 12, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawArc(rect, 225 * 16, -270 * 16)
        if self.value is not None:
            accent = theme.VENOM if self.accepted else theme.EMBER
            p.setPen(QPen(theme.color(accent), 12, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
            p.drawArc(rect, 225 * 16, int(-270 * 16 * max(0.01, min(1.0, self.value))))
        p.setPen(theme.color(theme.TEXT))
        p.setFont(theme.font("mono", side / 7.0, QFont.Weight.DemiBold))
        text = "—" if self.value is None else f"{self.value * 100:.0f}%"
        p.drawText(rect.adjusted(0, -8, 0, -8), Qt.AlignmentFlag.AlignCenter, text)
        p.setPen(theme.color(theme.MUTED))
        p.setFont(theme.font("display", 8))
        p.drawText(rect.adjusted(0, side / 5.0, 0, side / 5.0), Qt.AlignmentFlag.AlignCenter, "CONFIANZA")
        p.end()


class DistanceBars(QWidget):
    """DTW distance to each command, with that command's acceptance threshold."""

    ROW = 30

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.scores: list[Score] = []
        self.winner: str | None = None
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(self.ROW * 5 + 38)

    def sizeHint(self) -> QSize:
        return QSize(320, self.ROW * 5 + 38)

    def set_scores(self, scores: Sequence[Score], winner: str | None) -> None:
        self.scores, self.winner = list(scores), winner
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ARG002
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w = self.width()
        label_w, value_w = 92, 52
        track_w = max(40, w - label_w - value_w - 12)
        if not self.scores:
            p.setPen(theme.color(theme.FAINT))
            p.setFont(theme.font("body", 9.5))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Las distancias DTW aparecerán aquí")
            p.end()
            return
        top = max(max(s.distance for s in self.scores), max(s.threshold for s in self.scores)) * 1.1
        for row, s in enumerate(self.scores):
            y = row * self.ROW + 4
            cmd = BY_NAME.get(s.command)
            name = cmd.word if cmd else s.command.lower()
            is_win = s.command == self.winner
            p.setPen(theme.color(theme.TEXT if is_win else theme.MUTED))
            p.setFont(theme.font("display", 10.5, QFont.Weight.DemiBold if is_win else QFont.Weight.Normal))
            p.drawText(QRectF(0, y, label_w, 20), Qt.AlignmentFlag.AlignVCenter, name)
            track = QRectF(label_w, y + 6, track_w, 8)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(theme.color(theme.PANEL_2))
            p.drawRoundedRect(track, 4, 4)
            inside = s.distance <= s.threshold
            colour = theme.VENOM if (is_win and inside) else (theme.EMBER if is_win else theme.DUSK)
            p.setBrush(theme.color(colour, 255 if is_win else 150))
            p.drawRoundedRect(QRectF(track.left(), track.top(), track_w * s.distance / top, 8), 4, 4)
            tx = track.left() + track_w * s.threshold / top
            p.setPen(QPen(theme.color(theme.TEXT, 200), 2))
            p.drawLine(QPointF(tx, y + 2), QPointF(tx, y + 18))
            p.setPen(theme.color(theme.MUTED))
            p.setFont(theme.font("mono", 9))
            p.drawText(QRectF(w - value_w, y, value_w, 20), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, f"{s.distance:.2f}")
        p.setPen(theme.color(theme.FAINT))
        p.setFont(theme.font("body", 8))
        p.drawText(QRectF(0, self.ROW * len(self.scores) + 2, w, 34),
                   Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap,
                   "barra: distancia al comando · marca: umbral de aceptación")
        p.end()


class TakeDots(QWidget):
    """Filled dots for recorded takes out of the recommended count."""

    def __init__(self, total: int = 5, parent: QWidget | None = None):
        super().__init__(parent)
        self.total = total
        self.count = 0
        self.setFixedSize(total * 13, 12)

    def set_count(self, count: int) -> None:
        self.count = count
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ARG002
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        for i in range(self.total):
            filled = i < self.count
            p.setPen(QPen(theme.color(theme.VENOM if filled else theme.LINE), 1.5))
            p.setBrush(theme.color(theme.VENOM) if filled else Qt.BrushStyle.NoBrush)
            p.drawEllipse(QPointF(6 + i * 13, 6), 4.2, 4.2)
        p.end()
