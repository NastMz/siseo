"""03 · Jugar: the Snake, steered by voice, keyboard or the synthetic demo pilot."""

from __future__ import annotations

import random

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QKeyEvent
from PySide6.QtWidgets import QButtonGroup, QGridLayout, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ...game import DIRECTIONS, OVER, RUNNING, GameConfig, SnakeGame, next_action
from ...recognition import Analysis, Recognition
from ...vocabulary import BY_NAME, VOCABULARY, action_for
from .. import theme
from ..board import BoardView
from ..context import AppContext, MicListener
from ..widgets import LevelMeter, Panel, WaveformView, button, eyebrow, label

KEYS = {
    Qt.Key.Key_Up: "up",
    Qt.Key.Key_W: "up",
    Qt.Key.Key_Down: "down",
    Qt.Key.Key_S: "down",
    Qt.Key.Key_Left: "left",
    Qt.Key.Key_A: "left",
    Qt.Key.Key_Right: "right",
    Qt.Key.Key_D: "right",
}
WORD_FOR_ACTION = {c.action: c.name for c in VOCABULARY}
MODES = ("teclado", "voz", "demo")


class PlayPage(QWidget):
    def __init__(self, ctx: AppContext, parent: QWidget | None = None, seed: int | None = None):
        super().__init__(parent)
        self.ctx = ctx
        self.game = SnakeGame(GameConfig(), random.Random(seed))
        self.mode = "demo" if ctx.demo else ("voz" if ctx.microphone else "teclado")
        self.listener = MicListener(ctx.engine, self)
        self.listener.heard.connect(self.on_heard)
        self.listener.level.connect(self._on_level)
        self.listener.failed.connect(self._on_mic_failed)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.tick)
        self.active = False

        root = QHBoxLayout(self)
        root.setContentsMargins(36, 28, 36, 28)
        root.setSpacing(22)

        left = QVBoxLayout()
        left.setSpacing(10)
        left.addWidget(eyebrow("03 · Jugar"))
        left.addWidget(label("Habla y la serpiente gira", "title"))
        self.board = BoardView(self.game)
        left.addWidget(self.board, 1)
        root.addLayout(left, 1)

        side = QVBoxLayout()
        side.setSpacing(14)
        side_w = QWidget()
        side_w.setLayout(side)
        side_w.setFixedWidth(300)
        root.addWidget(side_w)

        score_panel = Panel("Puntuación")
        row = QHBoxLayout()
        self.score = QLabel("0")
        self.score.setFont(theme.font("mono", 40, QFont.Weight.Bold))
        self.score.setStyleSheet(f"color: {theme.VENOM};")
        row.addWidget(self.score)
        row.addStretch()
        stats = QVBoxLayout()
        self.best = label("récord 0", "mono")
        self.speed = label("", "mono")
        stats.addStretch()
        stats.addWidget(self.best, alignment=Qt.AlignmentFlag.AlignRight)
        stats.addWidget(self.speed, alignment=Qt.AlignmentFlag.AlignRight)
        row.addLayout(stats)
        score_panel.body.addLayout(row)
        side.addWidget(score_panel)

        mode_panel = Panel("Control")
        chips = QHBoxLayout()
        chips.setSpacing(6)
        self.mode_group = QButtonGroup(self)
        self.mode_buttons = {}
        for mode, text in zip(MODES, ("Teclado", "Voz", "Demo")):
            b = button(text, "chip", checkable=True)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.clicked.connect(lambda _=False, m=mode: self.set_mode(m))
            self.mode_group.addButton(b)
            self.mode_buttons[mode] = b
            chips.addWidget(b)
        chips.addStretch()
        mode_panel.body.addLayout(chips)
        self.mode_hint = label("", "hint", wrap=True)
        mode_panel.body.addWidget(self.mode_hint)
        self.meter = LevelMeter()
        mode_panel.body.addWidget(self.meter)
        side.addWidget(mode_panel)

        heard_panel = Panel("Lo último que oyó")
        self.heard_word = QLabel("—")
        self.heard_word.setFont(theme.font("display", 24, QFont.Weight.DemiBold))
        self.heard_info = label("", "mono")
        self.heard_wave = WaveformView(min_height=56)
        self.heard_wave.placeholder = "Sin palabras todavía"
        heard_panel.body.addWidget(self.heard_word)
        heard_panel.body.addWidget(self.heard_info)
        heard_panel.body.addWidget(self.heard_wave)
        side.addWidget(heard_panel)

        keys_panel = Panel("Comandos")
        legend = QGridLayout()
        legend.setHorizontalSpacing(18)
        legend.setVerticalSpacing(4)
        for i, c in enumerate(VOCABULARY):
            legend.addWidget(label(f"{c.glyph}  {c.word}"), i // 2, i % 2)
        keys_panel.body.addLayout(legend)
        keys_panel.body.addWidget(label("Teclado: flechas o WASD · P pausa · Enter empieza", "hint", wrap=True))
        side.addWidget(keys_panel)
        side.addStretch()

        self.set_mode(self.mode)
        self._sync_hud()

    # --- lifecycle -----------------------------------------------------
    def activate(self) -> None:
        self.active = True
        self._refresh_scores()
        if self.mode == "voz":
            self.listener.start(self.ctx.device)
        self.board.setFocus()
        self._schedule()

    def deactivate(self) -> None:
        self.active = False
        self.timer.stop()
        self.listener.stop()
        if self.game.status == RUNNING:
            self.game.toggle_pause()
        self.board.update()

    def set_mode(self, mode: str) -> None:
        if mode == "voz" and not self.ctx.microphone:
            mode = "teclado"
            self.mode_hint.setText("No hay micrófono disponible: usa el teclado o el modo demo.")
        self.mode = mode
        self.mode_buttons[mode].setChecked(True)
        self.mode_buttons["voz"].setEnabled(self.ctx.microphone)
        if mode == "voz":
            self.mode_hint.setText("Escuchando continuamente. Di una dirección cuando quieras girar.")
            self.board.ready_hint = "Di «arriba», «abajo», «izquierda» o «derecha»"
            if self.active:
                self.listener.start(self.ctx.device)
        else:
            self.listener.stop()
            self.meter.set_level(-90.0, None)
            if mode == "demo":
                self.mode_hint.setText("Un piloto automático «habla» con voz sintética; cada giro pasa por el reconocedor.")
                self.board.ready_hint = "El piloto sintético está a punto de hablar"
            else:
                self.mode_hint.setText("Flechas o WASD para girar. La voz sigue siendo opcional.")
                self.board.ready_hint = "Pulsa una flecha para empezar"
        self.board.update()
        self.board.setFocus()
        if self.active:
            self._schedule()

    # --- game loop -----------------------------------------------------
    def _schedule(self) -> None:
        if not self.active:
            return
        if self.game.status == RUNNING or self.mode == "demo":
            self.timer.start(self.game.interval_ms)

    def tick(self) -> None:
        if self.mode == "demo":
            self._pilot()
        result = self.game.step()
        if result == "ate":
            self.board.flash = 1.0
        else:
            self.board.flash = max(0.0, self.board.flash - 0.35)
        if result in ("died", "won"):
            self.ctx.store.add_score(self.game.score, self.mode)
            self._refresh_scores()
        self._sync_hud()
        self.board.update()
        if self.game.status == OVER and self.mode == "demo":
            if self.active:
                self.timer.start(2500)  # let the score screen breathe, then go again
            return
        self._schedule()

    def _pilot(self) -> None:
        """Demo mode: say the word for the move the autopilot wants, through the recognizer."""
        if self.game.status == OVER:
            self.game.act("start")
        want = next_action(self.game)
        if self.game.status == RUNNING and self.game.heading == DIRECTIONS[want]:
            return
        analysis, rec = self.ctx.engine.recognize(self.ctx.speak(WORD_FOR_ACTION[want]))
        self.on_heard(analysis, rec)

    def apply(self, action: str | None) -> bool:
        if action is None:
            return False
        if self.game.status == OVER and action in ("up", "down", "left", "right", "start"):
            self.game.act("start")
            self._sync_hud()
        changed = self.game.act(action)
        self.board.update()
        self._sync_hud()
        if changed and not self.timer.isActive():
            self._schedule()
        return changed

    # --- input ---------------------------------------------------------
    def on_heard(self, analysis: Analysis, rec: Recognition) -> None:
        accent = theme.VENOM if rec.accepted else theme.EMBER
        if rec.accepted and rec.command:
            cmd = BY_NAME[rec.command]
            self.heard_word.setText(f"{cmd.glyph}  {cmd.word}")
            self.heard_info.setText(f"confianza {rec.confidence * 100:.0f}% · d={rec.scores[0].distance:.2f}")
        else:
            guess = BY_NAME[rec.best].word if rec.best in BY_NAME else "…"
            self.heard_word.setText(f"¿{guess}?")
            self.heard_info.setText("descartada: " + {"unknown": "lejos de las plantillas", "ambiguous": "ambigua"}.get(rec.reason, rec.reason))
        self.heard_word.setStyleSheet(f"color: {accent};")
        self.heard_wave.set_signal(analysis.signal, analysis.span, accent)
        if rec.accepted:
            self.apply(action_for(rec.command))

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        if key in KEYS:
            self.apply(KEYS[key])
        elif key == Qt.Key.Key_P:
            self.apply("pause")
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.apply("start")
        else:
            super().keyPressEvent(event)

    # --- HUD -----------------------------------------------------------
    def _refresh_scores(self) -> None:
        self.board.best = self.ctx.store.best_score()
        self.board.top_scores = self.ctx.store.top_scores(5)
        self.best.setText(f"récord {self.board.best}")

    def _sync_hud(self) -> None:
        self.score.setText(str(self.game.score))
        self.speed.setText(f"{self.game.speed:.1f} casillas/s")

    def _on_level(self, level_db: float, threshold_db: float, speaking: bool) -> None:
        self.meter.set_level(level_db, threshold_db, speaking)

    def _on_mic_failed(self, message: str) -> None:
        self.ctx.microphone = False
        self.set_mode("teclado")
        self.mode_hint.setText(message)
