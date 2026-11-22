"""Small building blocks shared by every page."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QFrame, QLabel, QPushButton, QVBoxLayout, QWidget

from .. import theme


def label(text: str = "", role: str | None = None, wrap: bool = False) -> QLabel:
    lab = QLabel(text)
    if role:
        lab.setProperty("role", role)
    lab.setWordWrap(wrap)
    return lab


def eyebrow(text: str) -> QLabel:
    lab = label(text.upper(), "eyebrow")
    f = theme.font("display", 8.5)
    f.setLetterSpacing(f.SpacingType.AbsoluteSpacing, 2.0)
    lab.setFont(f)
    return lab


def button(text: str, role: str | None = None, checkable: bool = False) -> QPushButton:
    btn = QPushButton(text)
    if role:
        btn.setProperty("role", role)
    btn.setCheckable(checkable)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    return btn


class Panel(QFrame):
    """Rounded surface with an optional eyebrow caption."""

    def __init__(self, caption: str | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("panel")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(18, 14, 18, 16)
        self.body.setSpacing(10)
        if caption:
            self.body.addWidget(eyebrow(caption))


class Wordmark(QWidget):
    """The Siseo logo: a sound wave that coils into an S, then the name."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedHeight(46)
        self.setMinimumWidth(150)

    def paintEvent(self, event) -> None:  # noqa: ARG002
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath(QPointF(6, 30))
        path.cubicTo(QPointF(6, 6), QPointF(30, 8), QPointF(20, 21))
        path.cubicTo(QPointF(11, 33), QPointF(34, 40), QPointF(34, 16))
        p.setPen(QPen(theme.color(theme.VENOM), 4.2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawPath(path)
        p.setBrush(theme.color(theme.INK))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QPointF(32.6, 17.5), 1.2, 1.2)
        p.setPen(theme.color(theme.TEXT))
        p.setFont(theme.font("display", 19, QFont.Weight.DemiBold))
        p.drawText(QRectF(44, 0, self.width() - 44, 42), Qt.AlignmentFlag.AlignVCenter, "siseo")
        p.end()
