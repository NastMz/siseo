"""Siseo's visual identity: a warm night palette with a venom-green accent.

Ink-black backgrounds, cream text, chartreuse for everything the snake and a
recognized word touch, ember for food and rejections, and a dusky teal for
the spectral views. Headings use a condensed DIN-like face; numbers use a
monospace face so readouts do not jitter.
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication

INK = "#100E0C"
PANEL = "#1A1714"
PANEL_2 = "#231F1B"
LINE = "#352F28"
TEXT = "#F2EADB"
MUTED = "#9C907E"
FAINT = "#62594C"
VENOM = "#C6F03E"
MOSS = "#55701C"
EMBER = "#FF7448"
DUSK = "#6FB3B8"

SPECTRUM_STOPS = [
    (0.00, "#100E0C"),
    (0.30, "#1D3436"),
    (0.55, "#3F7C80"),
    (0.78, "#9CCB4A"),
    (0.92, "#D8F25A"),
    (1.00, "#FFF6CC"),
]

DISPLAY_FAMILIES = ["Bahnschrift", "DIN Alternate", "Barlow Condensed", "Segoe UI", "Helvetica Neue", "Arial"]
BODY_FAMILIES = ["Segoe UI", "Inter", "Helvetica Neue", "Noto Sans", "Arial"]
MONO_FAMILIES = ["Cascadia Mono", "Consolas", "JetBrains Mono", "Menlo", "DejaVu Sans Mono", "Courier New"]


def color(hex_code: str, alpha: int = 255) -> QColor:
    c = QColor(hex_code)
    c.setAlpha(alpha)
    return c


def _first_available(families: list[str]) -> str:
    available = set(QFontDatabase.families())
    for fam in families:
        if fam in available:
            return fam
    return families[-1]


def font(kind: str = "body", size: float = 10.0, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    families = {"display": DISPLAY_FAMILIES, "mono": MONO_FAMILIES}.get(kind, BODY_FAMILIES)
    f = QFont(_first_available(families))
    f.setPointSizeF(size)
    f.setWeight(weight)
    if kind == "display":
        f.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 101)
    return f


def stylesheet() -> str:
    display = _first_available(DISPLAY_FAMILIES)
    body = _first_available(BODY_FAMILIES)
    mono = _first_available(MONO_FAMILIES)
    return f"""
    QWidget {{ background: {INK}; color: {TEXT}; }}
    QToolTip {{ background: {PANEL_2}; color: {TEXT}; border: 1px solid {LINE}; padding: 4px 6px; }}
    QFrame#panel {{ background: {PANEL}; border: 1px solid {LINE}; border-radius: 14px; }}
    QFrame#panel QLabel, QFrame#panel QWidget#transparent {{ background: transparent; }}
    QFrame#rail {{ background: {PANEL}; border-right: 1px solid {LINE}; }}
    QFrame#rail QLabel {{ background: transparent; }}
    QLabel[role="eyebrow"] {{ color: {MUTED}; font-family: "{display}"; font-size: 8.5pt; padding-top: 4px; }}
    QLabel[role="title"] {{ font-family: "{display}"; font-size: 22pt; font-weight: 600; }}
    QLabel[role="hint"] {{ color: {MUTED}; }}
    QLabel[role="mono"] {{ font-family: "{mono}"; color: {MUTED}; }}
    QLabel[role="badge"] {{ background: {PANEL_2}; color: {VENOM}; border: 1px solid {MOSS};
        border-radius: 10px; padding: 3px 10px; font-family: "{display}"; font-size: 8.5pt; }}
    QPushButton {{ background: {PANEL_2}; color: {TEXT}; border: 1px solid {LINE}; border-radius: 10px;
        padding: 9px 16px; font-family: "{display}"; font-size: 10.5pt; }}
    QPushButton:hover {{ border-color: {MUTED}; }}
    QPushButton:pressed {{ background: {LINE}; }}
    QPushButton:disabled {{ color: {FAINT}; border-color: {PANEL_2}; }}
    QPushButton[role="primary"] {{ background: {VENOM}; color: {INK}; border: 1px solid {VENOM}; font-weight: 600; }}
    QPushButton[role="primary"]:hover {{ background: #D6F865; }}
    QPushButton[role="primary"]:disabled {{ background: {MOSS}; color: {PANEL}; border-color: {MOSS}; }}
    QPushButton[role="danger"] {{ color: {EMBER}; }}
    QPushButton[role="chip"] {{ padding: 6px 12px; border-radius: 14px; font-size: 9.5pt; }}
    QPushButton[role="chip"]:checked {{ background: {VENOM}; color: {INK}; border-color: {VENOM}; }}
    QPushButton[role="nav"] {{ background: transparent; border: none; border-radius: 12px; text-align: left;
        padding: 12px 14px; font-size: 12pt; color: {MUTED}; }}
    QPushButton[role="nav"]:hover {{ color: {TEXT}; background: {PANEL_2}; }}
    QPushButton[role="nav"]:checked {{ color: {INK}; background: {VENOM}; font-weight: 600; }}
    QComboBox {{ background: {PANEL_2}; border: 1px solid {LINE}; border-radius: 10px; padding: 7px 12px;
        font-family: "{display}"; font-size: 10.5pt; }}
    QComboBox::drop-down {{ border: none; width: 22px; }}
    QComboBox QAbstractItemView {{ background: {PANEL_2}; border: 1px solid {LINE};
        selection-background-color: {VENOM}; selection-color: {INK}; }}
    QScrollArea {{ border: none; background: transparent; }}
    """


def apply(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(INK))
    pal.setColor(QPalette.ColorRole.Base, QColor(PANEL))
    pal.setColor(QPalette.ColorRole.Text, QColor(TEXT))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(VENOM))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(INK))
    app.setPalette(pal)
    app.setFont(font("body", 10))
    app.setStyleSheet(stylesheet())
