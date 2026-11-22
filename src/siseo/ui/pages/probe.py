"""02 · Probar: say a word and see what the recognizer heard, and why."""

from __future__ import annotations

import numpy as np
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ...recognition import Analysis, Recognition
from ...vocabulary import BY_NAME, VOCABULARY
from .. import theme
from ..context import TAKE_SECONDS, AppContext, Recorder
from ..widgets import ConfidenceGauge, DistanceBars, Panel, SpectrogramView, WaveformView, button, eyebrow, label

REASONS = {
    "ok": "Reconocida",
    "unknown": "No reconocida · lejos de todas las plantillas",
    "ambiguous": "No reconocida · dos comandos casi empatados",
    "silence": "No se detectó voz",
    "untrained": "Aún no hay plantillas: entrena primero",
}


class ProbePage(QWidget):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None):
        super().__init__(parent)
        self.ctx = ctx
        self.last: tuple[Analysis, Recognition] | None = None
        self.recorder = Recorder(self)
        self.recorder.finished.connect(self.evaluate)
        self.recorder.failed.connect(self._on_failed)

        root = QVBoxLayout(self)
        root.setContentsMargins(36, 28, 36, 28)
        root.setSpacing(18)
        root.addWidget(eyebrow("02 · Probar"))
        root.addWidget(label("¿Qué escuchó?", "title"))
        root.addWidget(
            label(
                "Di una palabra y mira la decisión completa: dónde empezó la voz, su espectro por "
                "bandas y la distancia a cada comando entrenado.",
                "hint",
                wrap=True,
            )
        )

        controls = QHBoxLayout()
        controls.setSpacing(10)
        self.listen_btn = button("●  Escuchar", "primary")
        self.listen_btn.clicked.connect(self.listen)
        controls.addWidget(self.listen_btn)
        self.word_box = QComboBox()
        for cmd in VOCABULARY:
            self.word_box.addItem(f"{cmd.glyph}  {cmd.word}", cmd.name)
        self.word_box.addItem("?  hola · fuera del vocabulario", "HOLA")
        self.word_box.setMinimumWidth(240)
        self.say_btn = button("Pronunciar con voz sintética")
        self.say_btn.clicked.connect(self.say_selected)
        controls.addWidget(self.word_box)
        controls.addWidget(self.say_btn)
        controls.addStretch()
        root.addLayout(controls)
        self.word_box.setVisible(ctx.demo)
        self.say_btn.setVisible(ctx.demo)

        top = QHBoxLayout()
        top.setSpacing(18)
        result = Panel("Resultado")
        row = QHBoxLayout()
        row.setSpacing(22)
        self.gauge = ConfidenceGauge(diameter=150)
        row.addWidget(self.gauge)
        words = QVBoxLayout()
        words.setSpacing(4)
        self.heard = QLabel("—")
        self.heard.setFont(theme.font("display", 40, QFont.Weight.DemiBold))
        self.reason = label("Esperando una palabra", "hint", wrap=True)
        self.detail = label("", "mono", wrap=True)
        words.addStretch()
        words.addWidget(self.heard)
        words.addWidget(self.reason)
        words.addWidget(self.detail)
        words.addStretch()
        row.addLayout(words, 1)
        result.body.addLayout(row)
        top.addWidget(result, 5)

        ranking = Panel("Distancia DTW a cada comando")
        self.bars = DistanceBars()
        ranking.body.addWidget(self.bars)
        ranking.body.addStretch()
        top.addWidget(ranking, 6)
        root.addLayout(top)

        signal_panel = Panel("Señal · la zona marcada es la voz detectada")
        self.wave = WaveformView(min_height=100)
        self.spec = SpectrogramView(min_height=130)
        signal_panel.body.addWidget(self.wave)
        signal_panel.body.addWidget(self.spec)
        root.addWidget(signal_panel, 1)
        self.refresh()

    def refresh(self) -> None:
        busy = self.recorder.busy
        self.listen_btn.setEnabled(self.ctx.microphone and not busy)
        self.listen_btn.setVisible(self.ctx.microphone or not self.ctx.demo)
        if not self.ctx.microphone and not self.ctx.demo:
            self.reason.setText("No se encontró micrófono. Inicia Siseo con --demo para usar la voz sintética.")

    def listen(self) -> None:
        if not self.ctx.microphone:
            return
        self.reason.setText(f"Escuchando durante {TAKE_SECONDS:.1f} s…")
        self.listen_btn.setEnabled(False)
        self.recorder.start(TAKE_SECONDS, self.ctx.device)

    def say_selected(self) -> None:
        self.say(self.word_box.currentData())

    def say(self, word: str) -> Recognition:
        return self.evaluate(self.ctx.speak(word))

    def evaluate(self, clip: np.ndarray) -> Recognition:
        analysis, recognition = self.ctx.engine.recognize(clip)
        self.show_result(analysis, recognition)
        self.refresh()
        return recognition

    def show_result(self, analysis: Analysis, rec: Recognition) -> None:
        self.last = (analysis, rec)
        accent = theme.VENOM if rec.accepted else theme.EMBER
        if rec.accepted and rec.command:
            cmd = BY_NAME.get(rec.command)
            self.heard.setText(f"{cmd.glyph}  {cmd.word}" if cmd else rec.command.lower())
        elif rec.best:
            cmd = BY_NAME.get(rec.best)
            self.heard.setText(f"¿{cmd.word if cmd else rec.best.lower()}?")
        else:
            self.heard.setText("—")
        self.heard.setStyleSheet(f"color: {accent};")
        self.reason.setText(REASONS.get(rec.reason, rec.reason))
        if rec.scores:
            best = rec.scores[0]
            self.detail.setText(f"distancia {best.distance:.2f} · umbral {best.threshold:.2f}")
        else:
            self.detail.setText("")
        self.gauge.set_value(rec.confidence if rec.reason != "silence" else None, rec.accepted)
        self.bars.set_scores(rec.scores, rec.best)
        self.wave.set_signal(analysis.signal, analysis.span, accent)
        self.spec.set_data(analysis.spectrogram, analysis.span, self.ctx.engine.analyzer.feat.hop)

    def _on_failed(self, message: str) -> None:
        self.reason.setText(message)
        self.refresh()


