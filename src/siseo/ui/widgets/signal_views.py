"""Views of a signal: waveform, band-energy spectrogram, spectral fingerprint, level meter."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QImage, QLinearGradient, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from .. import theme


def _lut() -> np.ndarray:
    """256-entry RGB colormap from the theme's spectrum stops."""
    xs = np.array([s for s, _ in theme.SPECTRUM_STOPS])
    rgb = np.array([QColor(c).getRgb()[:3] for _, c in theme.SPECTRUM_STOPS], dtype=np.float64)
    pos = np.linspace(0, 1, 256)
    return np.stack([np.interp(pos, xs, rgb[:, k]) for k in range(3)], axis=1).astype(np.uint8)


_LUT = None


class _Canvas(QWidget):
    def __init__(self, min_height: int, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumHeight(min_height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)

    def sizeHint(self) -> QSize:
        return QSize(420, self.minimumHeight())

    def _empty(self, p: QPainter, text: str) -> None:
        p.setPen(theme.color(theme.FAINT))
        p.setFont(theme.font("body", 9.5))
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, text)


class WaveformView(_Canvas):
    """Min/max envelope of a clip, with the detected word highlighted."""

    def __init__(self, parent: QWidget | None = None, min_height: int = 110):
        super().__init__(min_height, parent)
        self.signal: np.ndarray | None = None
        self.span: tuple[int, int] | None = None
        self.accent = theme.VENOM
        self.placeholder = "La forma de onda aparecerá aquí"

    def set_signal(self, signal: np.ndarray | None, span: tuple[int, int] | None = None, accent: str | None = None) -> None:
        self.signal = None if signal is None else np.asarray(signal, dtype=np.float64)
        self.span = span
        self.accent = accent or theme.VENOM
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ARG002
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        mid = h / 2
        p.setPen(QPen(theme.color(theme.LINE), 1, Qt.PenStyle.DashLine))
        p.drawLine(QPointF(0, mid), QPointF(w, mid))
        if self.signal is None or len(self.signal) == 0:
            self._empty(p, self.placeholder)
            p.end()
            return
        n = len(self.signal)
        if self.span is not None:
            x0, x1 = self.span[0] / n * w, self.span[1] / n * w
            p.fillRect(QRectF(x0, 0, x1 - x0, h), theme.color(self.accent, 28))
            p.setPen(QPen(theme.color(self.accent, 150), 1))
            p.drawLine(QPointF(x0, 4), QPointF(x0, h - 4))
            p.drawLine(QPointF(x1, 4), QPointF(x1, h - 4))
        cols = max(1, int(w))
        edges = np.linspace(0, n, cols + 1).astype(int)
        peak = max(float(np.max(np.abs(self.signal))), 1e-4)
        scale = (h * 0.44) / peak
        top = QPainterPath()
        bottom = []
        for i in range(cols):
            seg = self.signal[edges[i] : max(edges[i] + 1, edges[i + 1])]
            hi, lo = float(seg.max()), float(seg.min())
            x = float(i)
            if i == 0:
                top.moveTo(x, mid - hi * scale)
            else:
                top.lineTo(x, mid - hi * scale)
            bottom.append(QPointF(x, mid - lo * scale))
        for pt in reversed(bottom):
            top.lineTo(pt)
        top.closeSubpath()
        p.setPen(Qt.PenStyle.NoPen)
        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0.0, theme.color(self.accent, 230))
        grad.setColorAt(0.5, theme.color(self.accent, 150))
        grad.setColorAt(1.0, theme.color(self.accent, 230))
        p.setBrush(grad)
        p.drawPath(top)
        p.end()


class SpectrogramView(_Canvas):
    """Band energies over time: x is time, y is the mel band (low at the bottom)."""

    def __init__(self, parent: QWidget | None = None, min_height: int = 150):
        super().__init__(min_height, parent)
        self.data: np.ndarray | None = None
        self.span_frames: tuple[int, int] | None = None
        self.placeholder = "Espectrograma por bandas FFT"

    def set_data(self, spectrogram: np.ndarray | None, span: tuple[int, int] | None = None, hop: int = 160) -> None:
        self.data = None if spectrogram is None or len(spectrogram) == 0 else np.asarray(spectrogram)
        self.span_frames = None if span is None else (span[0] // hop, span[1] // hop)
        self.update()

    def image(self) -> QImage | None:
        global _LUT
        if self.data is None:
            return None
        if _LUT is None:
            _LUT = _lut()
        d = self.data.T[::-1]  # bands on y, low frequencies at the bottom
        top = d.max()
        norm = np.clip((d - (top - 9.0)) / 9.0, 0.0, 1.0)  # about 40 dB of range
        rgb = np.ascontiguousarray(_LUT[(norm * 255).astype(np.uint8)])
        img = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format.Format_RGB888)
        return img.copy()

    def paintEvent(self, event) -> None:  # noqa: ARG002
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        rect = QRectF(self.rect()).adjusted(0, 0, 0, 0)
        path = QPainterPath()
        path.addRoundedRect(rect, 8, 8)
        p.setClipPath(path)
        p.fillRect(rect, theme.color(theme.INK))
        img = self.image()
        if img is None:
            self._empty(p, self.placeholder)
            p.end()
            return
        p.drawImage(rect, img)
        if self.span_frames is not None and self.data is not None:
            n = len(self.data)
            p.setPen(QPen(theme.color(theme.TEXT, 170), 1, Qt.PenStyle.DashLine))
            for f in self.span_frames:
                x = f / n * rect.width()
                p.drawLine(QPointF(x, 0), QPointF(x, rect.height()))
        p.setPen(theme.color(theme.TEXT, 140))
        p.setFont(theme.font("mono", 7.5))
        p.drawText(QRectF(6, 2, 80, 14), Qt.AlignmentFlag.AlignLeft, "7.6 kHz")
        p.drawText(QRectF(6, rect.height() - 16, 80, 14), Qt.AlignmentFlag.AlignLeft, "80 Hz")
        p.end()


class BandBars(_Canvas):
    """A command's spectral fingerprint: mean log energy of each FFT band."""

    def __init__(self, parent: QWidget | None = None, min_height: int = 60, compact: bool = False):
        super().__init__(min_height, parent)
        self.values: np.ndarray | None = None
        self.ghost: np.ndarray | None = None
        self.compact = compact
        self.accent = theme.VENOM

    def set_values(self, values: np.ndarray | None, ghost: np.ndarray | None = None, accent: str | None = None) -> None:
        self.values = None if values is None else np.asarray(values, dtype=np.float64)
        self.ghost = None if ghost is None else np.asarray(ghost, dtype=np.float64)
        self.accent = accent or theme.VENOM
        self.update()

    @staticmethod
    def _heights(v: np.ndarray, top: float) -> np.ndarray:
        return np.clip((v - (top - 9.0)) / 9.0, 0.03, 1.0)

    def paintEvent(self, event) -> None:  # noqa: ARG002
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        if self.values is None:
            n = 24
            gap = 2 if self.compact else 3
            bw = (w - gap * (n - 1)) / n
            for i in range(n):
                p.fillRect(QRectF(i * (bw + gap), h - 3, bw, 3), theme.color(theme.LINE))
            if not self.compact:
                self._empty(p, "Sin huella todavía")
            p.end()
            return
        vals = self.values
        top = max(vals.max(), self.ghost.max() if self.ghost is not None else -np.inf)
        n = len(vals)
        gap = 2 if self.compact else 3
        bw = (w - gap * (n - 1)) / n
        if self.ghost is not None:
            for i, hgt in enumerate(self._heights(self.ghost, top)):
                p.setPen(QPen(theme.color(theme.TEXT, 120), 1.2))
                y = h - hgt * h
                p.drawLine(QPointF(i * (bw + gap), y), QPointF(i * (bw + gap) + bw, y))
        p.setPen(Qt.PenStyle.NoPen)
        for i, hgt in enumerate(self._heights(vals, top)):
            bar = QRectF(i * (bw + gap), h - hgt * h, bw, hgt * h)
            grad = QLinearGradient(bar.topLeft(), bar.bottomLeft())
            grad.setColorAt(0, theme.color(self.accent))
            grad.setColorAt(1, theme.color(theme.MOSS if self.accent == theme.VENOM else self.accent, 110))
            p.setBrush(grad)
            p.drawRoundedRect(bar, min(2.0, bw / 2), min(2.0, bw / 2))
        p.end()


class LevelMeter(_Canvas):
    """Microphone level in dB with the voice threshold marked."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(26, parent)
        self.setMaximumHeight(30)
        self.level_db = -90.0
        self.threshold_db: float | None = None
        self.active = False

    def set_level(self, level_db: float, threshold_db: float | None, active: bool = False) -> None:
        self.level_db, self.threshold_db, self.active = level_db, threshold_db, active
        self.update()

    @staticmethod
    def _pos(db: float) -> float:
        return float(np.clip((db + 70.0) / 70.0, 0.0, 1.0))

    def paintEvent(self, event) -> None:  # noqa: ARG002
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        track = QRectF(0, h / 2 - 5, w, 10)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(theme.color(theme.PANEL_2))
        p.drawRoundedRect(track, 5, 5)
        fill = QRectF(track.left(), track.top(), track.width() * self._pos(self.level_db), track.height())
        p.setBrush(theme.color(theme.VENOM if self.active else theme.DUSK))
        p.drawRoundedRect(fill, 5, 5)
        if self.threshold_db is not None:
            x = track.width() * self._pos(self.threshold_db)
            p.setPen(QPen(theme.color(theme.EMBER), 2))
            p.drawLine(QPointF(x, 2), QPointF(x, h - 2))
        p.end()
