"""01 · Entrenar: record takes of every command and see their spectral fingerprint."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QMouseEvent
from PySide6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ...engine import TrainingError
from ...vocabulary import BY_NAME, MIN_TAKES_READY, RECOMMENDED_TAKES, VOCABULARY
from .. import theme
from ..context import TAKE_SECONDS, AppContext, Recorder
from ..widgets import BandBars, Panel, SpectrogramView, TakeDots, WaveformView, button, eyebrow, label


class CommandCard(QFrame):
    clicked = Signal(str)

    def __init__(self, name: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.name = name
        cmd = BY_NAME[name]
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(74)
        grid = QGridLayout(self)
        grid.setContentsMargins(14, 10, 14, 10)
        grid.setHorizontalSpacing(12)
        glyph = QLabel(cmd.glyph)
        glyph.setFont(theme.font("display", 20, QFont.Weight.DemiBold))
        glyph.setFixedWidth(28)
        self.word = QLabel(cmd.word)
        self.word.setFont(theme.font("display", 13, QFont.Weight.DemiBold))
        self.dots = TakeDots(RECOMMENDED_TAKES)
        self.count = label("0 tomas", "mono")
        self.bars = BandBars(min_height=30, compact=True)
        self.bars.setFixedWidth(96)
        grid.addWidget(glyph, 0, 0, 2, 1)
        grid.addWidget(self.word, 0, 1)
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(self.dots)
        row.addWidget(self.count)
        row.addStretch()
        grid.addLayout(row, 1, 1)
        grid.addWidget(self.bars, 0, 2, 2, 1)
        for w in (glyph, self.word, self.count):
            w.setStyleSheet("background: transparent;")
        self.set_selected(False)

    def set_selected(self, selected: bool) -> None:
        border = theme.VENOM if selected else theme.LINE
        bg = theme.PANEL_2 if selected else theme.PANEL
        self.setStyleSheet(
            f"CommandCard {{ background: {bg}; border: 1px solid {border}; border-radius: 14px; }}"
            f"CommandCard QLabel {{ background: transparent; }}"
        )

    def update_state(self, count: int, fingerprint: np.ndarray | None) -> None:
        self.dots.set_count(min(count, RECOMMENDED_TAKES))
        self.count.setText(f"{count} toma" + ("" if count == 1 else "s"))
        self.bars.set_values(fingerprint)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.name)
        super().mousePressEvent(event)


class TrainPage(QWidget):
    trained = Signal()

    def __init__(self, ctx: AppContext, parent: QWidget | None = None):
        super().__init__(parent)
        self.ctx = ctx
        self.selected = VOCABULARY[0].name
        self.recorder = Recorder(self)
        self.recorder.finished.connect(self._on_clip)
        self.recorder.failed.connect(self._on_failed)

        root = QVBoxLayout(self)
        root.setContentsMargins(36, 28, 36, 28)
        root.setSpacing(18)
        root.addWidget(eyebrow("01 · Entrenar"))
        root.addWidget(label("Enseña tu voz", "title"))
        root.addWidget(
            label(
                "Graba cada palabra unas cuantas veces. Cada toma se guarda como plantilla y "
                "Siseo aprende cuánto varía tu forma de decirla.",
                "hint",
                wrap=True,
            )
        )

        body = QHBoxLayout()
        body.setSpacing(18)
        root.addLayout(body, 1)

        cards_col = QVBoxLayout()
        cards_col.setSpacing(10)
        self.cards: dict[str, CommandCard] = {}
        for cmd in VOCABULARY:
            card = CommandCard(cmd.name)
            card.clicked.connect(self.select)
            self.cards[cmd.name] = card
            cards_col.addWidget(card)
        self.model_status = label("", "hint", wrap=True)
        cards_col.addSpacing(6)
        cards_col.addWidget(self.model_status)
        cards_col.addStretch()
        left = QWidget()
        left.setLayout(cards_col)
        left.setFixedWidth(330)
        body.addWidget(left)

        right = QVBoxLayout()
        right.setSpacing(16)
        body.addLayout(right, 1)

        head = Panel("Comando seleccionado")
        title_row = QHBoxLayout()
        self.big_glyph = QLabel()
        self.big_glyph.setFont(theme.font("display", 34, QFont.Weight.DemiBold))
        self.big_glyph.setStyleSheet(f"color: {theme.VENOM};")
        self.big_word = QLabel()
        self.big_word.setFont(theme.font("display", 34, QFont.Weight.DemiBold))
        title_row.addWidget(self.big_glyph)
        title_row.addSpacing(10)
        title_row.addWidget(self.big_word)
        title_row.addStretch()
        head.body.addLayout(title_row)
        self.status = label("", "hint", wrap=True)
        head.body.addWidget(self.status)
        actions = QHBoxLayout()
        actions.setSpacing(10)
        self.record_btn = button("●  Grabar toma", "primary")
        self.record_btn.clicked.connect(self.record)
        self.synth_btn = button("Toma sintética")
        self.synth_btn.setToolTip("Genera una toma con el sintetizador de formantes (modo demo)")
        self.synth_btn.clicked.connect(self.synthetic_take)
        self.clear_btn = button("Borrar tomas", "danger")
        self.clear_btn.clicked.connect(self.clear_selected)
        actions.addWidget(self.record_btn)
        actions.addWidget(self.synth_btn)
        actions.addStretch()
        actions.addWidget(self.clear_btn)
        head.body.addLayout(actions)
        right.addWidget(head)

        signal_panel = Panel("Última toma · forma de onda y bandas FFT")
        self.wave = WaveformView(min_height=78)
        self.spec = SpectrogramView(min_height=112)
        signal_panel.body.addWidget(self.wave)
        signal_panel.body.addWidget(self.spec)
        right.addWidget(signal_panel, 1)

        fp_panel = Panel("Huella espectral · 24 bandas de 80 Hz a 7.6 kHz")
        self.fingerprint = BandBars(min_height=58)
        fp_panel.body.addWidget(self.fingerprint)
        fp_panel.body.addWidget(
            label("Barras: energía media de cada banda en todas las tomas · línea: la última toma.", "hint")
        )
        right.addWidget(fp_panel)

        self.synth_btn.setVisible(ctx.demo)
        self.record_btn.setVisible(not ctx.demo)
        self.refresh()
        self.select(self.selected)

    # --- state -----------------------------------------------------------
    def refresh(self) -> None:
        counts = self.ctx.store.counts()
        for name, card in self.cards.items():
            card.update_state(counts.get(name, 0), self.ctx.engine.fingerprint(name))
        ready = sum(1 for c in VOCABULARY if counts.get(c.name, 0) >= MIN_TAKES_READY)
        total = len(VOCABULARY)
        if ready == total:
            self.model_status.setText(f"Modelo listo · {sum(counts.values())} plantillas para {total} comandos.")
        else:
            self.model_status.setText(
                f"{ready} de {total} comandos listos. Cada comando necesita al menos {MIN_TAKES_READY} tomas."
            )
        self._update_selected_info()

    def select(self, name: str) -> None:
        self.selected = name
        for n, card in self.cards.items():
            card.set_selected(n == name)
        cmd = BY_NAME[name]
        self.big_glyph.setText(cmd.glyph)
        self.big_word.setText(cmd.word)
        takes = self.ctx.store.takes(name)
        if takes:
            last = takes[-1]
            audio = last.audio if last.audio is not None else np.zeros(1)
            analysis = self.ctx.engine.analyzer.analyze(audio, already_trimmed=True)
            self.wave.set_signal(audio, None)
            self.spec.set_data(analysis.spectrogram)
            self.fingerprint.set_values(self.ctx.engine.fingerprint(name), ghost=last.fingerprint)
        else:
            self.wave.set_signal(None)
            self.spec.set_data(None)
            self.fingerprint.set_values(None)
        self._update_selected_info()

    def _update_selected_info(self) -> None:
        n = self.ctx.store.counts().get(self.selected, 0)
        word = BY_NAME[self.selected].word
        if not self.recorder.busy:
            if self.ctx.microphone:
                hint = f"Pulsa «Grabar toma» y di «{word}» con naturalidad. Tienes {TAKE_SECONDS:.1f} s."
            elif self.ctx.demo:
                hint = "Modo demo: las tomas se generan con un sintetizador de formantes, sin micrófono."
            else:
                hint = "No se encontró micrófono. Conecta uno o inicia Siseo con --demo."
            done = f"{n} tomas" if n != 1 else "1 toma"
            self.status.setText(f"{done} (se recomiendan {RECOMMENDED_TAKES}). {hint}")
        self.record_btn.setEnabled(self.ctx.microphone and not self.recorder.busy)
        self.clear_btn.setEnabled(n > 0)

    # --- actions ---------------------------------------------------------
    def record(self) -> None:
        if not self.ctx.microphone or self.recorder.busy:
            return
        self.status.setText(f"Escuchando… di «{BY_NAME[self.selected].word}»")
        self.record_btn.setEnabled(False)
        self.recorder.start(TAKE_SECONDS, self.ctx.device)

    def synthetic_take(self) -> None:
        self._on_clip(self.ctx.speak(self.selected), source="synthetic")

    def clear_selected(self) -> None:
        self.ctx.engine.clear(self.selected)
        self.refresh()
        self.select(self.selected)
        self.trained.emit()

    def _on_clip(self, clip: np.ndarray, source: str = "mic") -> None:
        try:
            analysis = self.ctx.engine.add_take(self.selected, clip, source=source)
        except TrainingError as exc:
            self.wave.set_signal(clip, None, accent=theme.EMBER)
            self.refresh()
            self.status.setText(f"{exc} Prueba otra vez, un poco más cerca del micrófono.")
            return
        self.refresh()
        self.select(self.selected)
        self.wave.set_signal(analysis.signal, analysis.span)
        self.spec.set_data(analysis.spectrogram, analysis.span, self.ctx.engine.analyzer.feat.hop)
        self.status.setText("Toma guardada. " + self.status.text())
        self.trained.emit()

    def _on_failed(self, message: str) -> None:
        self._update_selected_info()
        self.status.setText(message)
