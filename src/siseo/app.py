"""Desktop entry point: ``siseo`` (add ``--demo`` to run without a microphone)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, audio
from .engine import VoiceEngine, seed_demo
from .store import ProfileStore, default_db_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="siseo", description="Siseo: Snake controlado por voz.")
    ap.add_argument("--demo", action="store_true", help="voz sintética y plantillas de demostración, sin micrófono")
    ap.add_argument("--db", type=Path, default=None, help="ruta de la base de datos (por defecto ~/.siseo/siseo.db)")
    ap.add_argument("--device", type=int, default=None, help="índice del micrófono (ver `siseo-cli devices`)")
    ap.add_argument("--page", choices=("entrenar", "probar", "jugar"), default=None, help="página inicial")
    ap.add_argument("--version", action="version", version=f"siseo {__version__}")
    return ap.parse_args(argv)


def build_context(args: argparse.Namespace):
    from .ui.context import AppContext

    if args.demo:
        engine = VoiceEngine(ProfileStore(":memory:"))
        seed_demo(engine)
        microphone = False
    else:
        engine = VoiceEngine(ProfileStore(args.db or default_db_path()))
        microphone = audio.has_microphone()
    return AppContext(engine=engine, demo=args.demo, microphone=microphone, device=args.device)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from PySide6.QtWidgets import QApplication

    from .ui import theme
    from .ui.main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("Siseo")
    theme.apply(app)
    ctx = build_context(args)
    window = MainWindow(ctx)
    if args.page:
        window.show_page(args.page)
    elif ctx.store.counts():
        window.show_page("jugar")
    window.show()
    code = app.exec()
    ctx.store.close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
