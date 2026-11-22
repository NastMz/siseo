"""Capture the README screenshots from the real app in demo mode, off screen.

    QT_QPA_PLATFORM=offscreen QT_QPA_FONTDIR=C:/Windows/Fonts python scripts/capture_screenshots.py

Everything shown is synthetic: the takes and every spoken word come from the
formant synthesizer, so no microphone or real voice is involved.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from siseo.app import build_context, parse_args  # noqa: E402
from siseo.game import OVER, RUNNING  # noqa: E402
from siseo.ui import theme  # noqa: E402
from siseo.ui.main_window import MainWindow  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "screenshots"


def settle(app: QApplication) -> None:
    for _ in range(5):
        app.processEvents()


def grab(window: MainWindow, app: QApplication, name: str) -> None:
    settle(app)
    path = OUT / f"{name}.png"
    window.grab().save(str(path))
    print("saved", path.relative_to(OUT.parent.parent))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    app = QApplication(sys.argv[:1])
    theme.apply(app)
    ctx = build_context(parse_args(["--demo"]))
    window = MainWindow(ctx, seed=4)
    window.resize(1280, 800)
    window.show()

    window.show_page("entrenar")
    window.train.select("IZQUIERDA")
    window.train.synthetic_take()
    grab(window, app, "01-entrenar")

    window.show_page("probar")
    window.probe.word_box.setCurrentIndex(3)
    window.probe.say("DERECHA")
    grab(window, app, "02-probar")
    window.probe.word_box.setCurrentIndex(5)
    window.probe.say("HOLA")
    grab(window, app, "03-probar-rechazo")

    for score, mode in ((14, "voz"), (11, "teclado"), (9, "voz")):
        ctx.store.add_score(score, mode)
    window.show_page("jugar")
    play = window.play
    play.timer.stop()
    play.active = False  # drive the loop by hand instead of the timer
    for _ in range(600):
        play.tick()
        if play.game.score >= 7 and play.game.status == RUNNING and len(play.game.snake) > 6:
            break
    grab(window, app, "04-jugar")

    play.set_mode("teclado")
    for _ in range(200):
        play.tick()
        if play.game.status == OVER:
            break
    play.timer.stop()
    grab(window, app, "05-fin-de-partida")
    window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
