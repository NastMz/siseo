"""Main window: a navigation rail and the three pages."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QButtonGroup, QFrame, QHBoxLayout, QMainWindow, QStackedWidget, QVBoxLayout, QWidget

from .. import __version__
from .context import AppContext
from .pages.play import PlayPage
from .pages.probe import ProbePage
from .pages.train import TrainPage
from .widgets import Wordmark, button, label

PAGES = ("entrenar", "probar", "jugar")


class MainWindow(QMainWindow):
    def __init__(self, ctx: AppContext, seed: int | None = None):
        super().__init__()
        self.ctx = ctx
        self.setWindowTitle("Siseo · comandos de voz por FFT")
        self.resize(1240, 800)
        self.setMinimumSize(1080, 720)

        central = QWidget()
        outer = QHBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.setCentralWidget(central)

        rail = QFrame()
        rail.setObjectName("rail")
        rail.setFixedWidth(232)
        rl = QVBoxLayout(rail)
        rl.setContentsMargins(18, 24, 18, 22)
        rl.setSpacing(6)
        rl.addWidget(Wordmark())
        tagline = label("Un reconocedor de comandos de voz hecho de FFT, bandas y plantillas.", "hint", wrap=True)
        rl.addWidget(tagline)
        rl.addSpacing(22)

        self.train = TrainPage(ctx)
        self.probe = ProbePage(ctx)
        self.play = PlayPage(ctx, seed=seed)
        self.train.trained.connect(self.ctx.engine.invalidate)
        self.stack = QStackedWidget()
        self.nav = QButtonGroup(self)
        self.nav_buttons = {}
        for i, (key, text, page) in enumerate(
            (("entrenar", "Entrenar", self.train), ("probar", "Probar", self.probe), ("jugar", "Jugar", self.play))
        ):
            self.stack.addWidget(page)
            b = button(f"0{i + 1}   {text}", "nav", checkable=True)
            b.clicked.connect(lambda _=False, k=key: self.show_page(k))
            self.nav.addButton(b)
            self.nav_buttons[key] = b
            rl.addWidget(b)
        rl.addStretch()

        if ctx.demo:
            rl.addWidget(label("MODO DEMO · voz sintética", "badge"), alignment=Qt.AlignmentFlag.AlignLeft)
        elif ctx.microphone:
            rl.addWidget(label("MICRÓFONO LISTO", "badge"), alignment=Qt.AlignmentFlag.AlignLeft)
        else:
            rl.addWidget(label("SIN MICRÓFONO", "badge"), alignment=Qt.AlignmentFlag.AlignLeft)
        rl.addSpacing(8)
        rl.addWidget(label(f"v{__version__} · 16 kHz · DTW", "mono"))

        outer.addWidget(rail)
        outer.addWidget(self.stack, 1)
        self.show_page("entrenar")

    def show_page(self, key: str) -> None:
        index = PAGES.index(key)
        self.nav_buttons[key].setChecked(True)
        if self.stack.currentWidget() is self.play and key != "jugar":
            self.play.deactivate()
        self.stack.setCurrentIndex(index)
        if key == "probar":
            self.probe.refresh()
        elif key == "entrenar":
            self.train.refresh()
        elif key == "jugar":
            self.play.activate()

    def closeEvent(self, event) -> None:
        self.play.deactivate()
        super().closeEvent(event)
