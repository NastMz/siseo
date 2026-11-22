"""pytest-qt smoke tests: the real windows, off screen, in demo mode."""

import numpy as np
import pytest
from PySide6.QtCore import Qt

from siseo.app import build_context, parse_args
from siseo.dsp import SAMPLE_RATE
from siseo.game import OVER, READY, RUNNING
from siseo.ui import theme
from siseo.ui.main_window import MainWindow


@pytest.fixture
def window(qtbot, qapp):
    theme.apply(qapp)
    ctx = build_context(parse_args(["--demo"]))
    win = MainWindow(ctx, seed=3)
    qtbot.addWidget(win)
    win.show()
    yield win
    win.play.deactivate()


def test_window_builds_with_three_pages(window):
    assert window.windowTitle().startswith("Siseo")
    assert window.stack.count() == 3
    assert window.nav_buttons["entrenar"].isChecked()


def test_training_page_adds_and_clears_synthetic_takes(window):
    page = window.train
    page.select("PAUSA")
    before = window.ctx.store.counts()["PAUSA"]
    page.synthetic_take()
    assert window.ctx.store.counts()["PAUSA"] == before + 1
    assert page.wave.signal is not None and page.wave.span is not None
    assert "6 tomas" in page.cards["PAUSA"].count.text()
    page.clear_selected()
    assert "PAUSA" not in window.ctx.store.counts()
    assert "4 de 5 comandos listos" in page.model_status.text()


def test_training_page_reports_silent_takes(window):
    page = window.train
    page._on_clip(np.zeros(SAMPLE_RATE))
    assert "No se detectó voz" in page.status.text()


def test_probe_page_accepts_a_command_and_rejects_an_unknown_word(window):
    window.show_page("probar")
    page = window.probe
    rec = page.say("ARRIBA")
    assert rec.accepted and rec.command == "ARRIBA"
    assert "arriba" in page.heard.text()
    assert page.bars.scores and page.gauge.value is not None
    rec = page.say("HOLA")
    assert not rec.accepted
    assert page.reason.text().startswith("No reconocida")


def test_play_page_keyboard_controls(window, qtbot):
    window.show_page("jugar")
    play = window.play
    play.set_mode("teclado")
    assert play.game.status == READY
    qtbot.keyClick(play, Qt.Key.Key_Up)
    assert play.game.status == RUNNING
    play.tick()
    assert play.game.head[1] == play.game.cfg.rows // 2 - 1
    qtbot.keyClick(play, Qt.Key.Key_P)
    assert play.game.status != RUNNING
    qtbot.keyClick(play, Qt.Key.Key_P)
    assert play.game.status == RUNNING


def test_demo_pilot_plays_through_the_recognizer_and_records_the_score(window):
    window.show_page("jugar")
    play = window.play
    play.timer.stop()
    play.active = False
    for _ in range(300):
        play.tick()
        if play.game.score >= 3:
            break
    assert play.game.score >= 3
    assert play.heard_word.text() != "—"
    play.set_mode("teclado")
    for _ in range(200):
        play.tick()
        if play.game.status == OVER:
            break
    assert play.game.status == OVER
    assert window.ctx.store.best_score() == play.game.score
    play.apply("start")
    assert play.game.status == RUNNING and play.game.score == 0


def test_voice_mode_falls_back_without_microphone(window):
    play = window.play
    play.set_mode("voz")
    assert play.mode == "teclado"
    assert not play.mode_buttons["voz"].isEnabled()


def test_mic_listener_recognizes_a_stream(window):
    from siseo.ui.context import MicListener

    listener = MicListener(window.ctx.engine)
    heard = []
    listener.heard.connect(lambda a, r: heard.append(r))
    stream = np.concatenate([window.ctx.speak("ABAJO"), np.zeros(SAMPLE_RATE // 2)])
    for i in range(0, len(stream), 480):
        listener.feed(stream[i : i + 480])
    assert [r.command for r in heard] == ["ABAJO"]
