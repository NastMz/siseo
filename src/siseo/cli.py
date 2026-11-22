"""Command line companion: ``siseo-cli`` (train from files, evaluate, inspect)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import __version__, audio
from .engine import TrainingError, VoiceEngine, seed_demo
from .store import ProfileStore, default_db_path
from .synth import synthesize
from .vocabulary import BY_NAME, NAMES


def _engine(args: argparse.Namespace) -> VoiceEngine:
    return VoiceEngine(ProfileStore(args.db or default_db_path()))


def _command(name: str) -> str:
    upper = name.upper()
    if upper not in BY_NAME:
        raise SystemExit(f"comando desconocido: {name} (usa uno de: {', '.join(n.lower() for n in NAMES)})")
    return upper


def cmd_devices(args: argparse.Namespace) -> int:  # noqa: ARG001
    devices = audio.input_devices()
    if not devices:
        print("No se encontraron micrófonos.")
        return 1
    for index, name in devices:
        print(f"{index:3d}  {name}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    engine = _engine(args)
    counts = engine.store.counts()
    print(f"base de datos: {engine.store.path}")
    for name in NAMES:
        print(f"  {BY_NAME[name].word:10s} {counts.get(name, 0):3d} tomas")
    print(f"récord: {engine.store.best_score()}")
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    name = _command(args.word)
    engine = _engine(args)
    for i in range(args.takes):
        input(f"[{i + 1}/{args.takes}] Pulsa Enter y di «{BY_NAME[name].word}»… ")
        try:
            clip = audio.record(args.seconds, args.device)
            engine.add_take(name, clip)
            print("  toma guardada")
        except (audio.AudioUnavailable, TrainingError) as exc:
            print(f"  descartada: {exc}")
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    name = _command(args.word)
    engine = _engine(args)
    ok = 0
    for path in args.wav:
        try:
            engine.add_take(name, audio.read_wav(path), source="file")
            ok += 1
        except (OSError, ValueError, EOFError) as exc:  # TrainingError is a ValueError
            print(f"{path}: descartado ({exc})")
    print(f"{ok} de {len(args.wav)} archivos añadidos a «{BY_NAME[name].word}».")
    return 0 if ok else 1


def cmd_recognize(args: argparse.Namespace) -> int:
    engine = _engine(args)
    for path in args.wav:
        _, rec = engine.recognize(audio.read_wav(path))
        verdict = rec.command.lower() if rec.command else f"(rechazada: {rec.reason})"
        dist = f"d={rec.scores[0].distance:.2f}" if rec.scores else ""
        print(f"{Path(path).name}: {verdict}  confianza={rec.confidence:.2f} {dist}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    engine = _engine(args)
    pattern = f"**/{args.glob}" if args.recursive else args.glob
    files = sorted(p for p in Path(args.directory).glob(pattern) if p.is_file())
    if not files:
        print("No se encontraron archivos.")
        return 2
    label = args.label.upper()
    hits = 0
    for path in files:
        _, rec = engine.recognize(audio.read_wav(path))
        hit = rec.command == label
        hits += hit
        print(f"{'OK ' if hit else 'ERR'} {path.name:40s} -> {rec.command or '-'} ({rec.confidence:.2f})")
    print(f"\nPrecisión {label.lower()}: {hits}/{len(files)} = {hits / len(files):.1%}")
    return 0


def cmd_clear(args: argparse.Namespace) -> int:
    name = _command(args.word)
    engine = _engine(args)
    removed = engine.store.clear_command(name)
    print(f"{removed} tomas eliminadas de «{BY_NAME[name].word}».")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Train on synthetic takes and report accuracy on fresh synthetic utterances."""
    engine = VoiceEngine(ProfileStore(":memory:"))
    seed_demo(engine, takes_per_command=args.takes, seed=args.seed)
    rng = np.random.default_rng(args.seed + 1)
    total = hits = 0
    for name in NAMES:
        for _ in range(args.trials):
            _, rec = engine.recognize(synthesize(name, rng))
            hits += rec.command == name
            total += 1
    rejected = sum(not engine.recognize(synthesize("HOLA", rng))[1].accepted for _ in range(args.trials))
    print(f"comandos reconocidos: {hits}/{total} ({hits / total:.1%})")
    print(f"palabra desconocida rechazada: {rejected}/{args.trials}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="siseo-cli", description="Herramientas de línea de comandos de Siseo.")
    ap.add_argument("--db", type=Path, default=None, help="base de datos (por defecto ~/.siseo/siseo.db)")
    ap.add_argument("--version", action="version", version=f"siseo {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("devices", help="lista los micrófonos").set_defaults(func=cmd_devices)
    sub.add_parser("status", help="tomas por comando y récord").set_defaults(func=cmd_status)

    p = sub.add_parser("record", help="graba tomas desde el micrófono")
    p.add_argument("word")
    p.add_argument("--takes", type=int, default=5)
    p.add_argument("--seconds", type=float, default=1.6)
    p.add_argument("--device", type=int, default=None)
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("import", help="añade tomas desde archivos WAV")
    p.add_argument("word")
    p.add_argument("wav", nargs="+", type=Path)
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("recognize", help="reconoce archivos WAV")
    p.add_argument("wav", nargs="+", type=Path)
    p.set_defaults(func=cmd_recognize)

    p = sub.add_parser("eval", help="precisión sobre una carpeta de WAV de un mismo comando")
    p.add_argument("directory", type=Path)
    p.add_argument("label")
    p.add_argument("--glob", default="*.wav")
    p.add_argument("--recursive", action="store_true")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("clear", help="borra las tomas de un comando")
    p.add_argument("word")
    p.set_defaults(func=cmd_clear)

    p = sub.add_parser("demo", help="prueba el reconocedor con voz sintética, sin micrófono")
    p.add_argument("--takes", type=int, default=5)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(func=cmd_demo)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
