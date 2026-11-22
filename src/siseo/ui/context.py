"""Shared application state and the Qt bridges to the microphone."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

from .. import audio
from ..dsp.vad import StreamSegmenter
from ..engine import VoiceEngine
from ..recognition import Analysis, Recognition
from ..store import ProfileStore
from ..synth import synthesize

TAKE_SECONDS = 1.6


@dataclass
class AppContext:
    engine: VoiceEngine
    demo: bool = False
    microphone: bool = False
    device: int | None = None
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(2022))

    @property
    def store(self) -> ProfileStore:
        return self.engine.store

    def speak(self, word: str) -> np.ndarray:
        """A synthetic utterance of ``word`` (demo mode's stand-in for a voice)."""
        return synthesize(word, self.rng)


class Recorder(QObject):
    """Records one take on a background thread and reports back on the GUI thread."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.busy = False

    def start(self, seconds: float = TAKE_SECONDS, device: int | None = None) -> None:
        if self.busy:
            return
        self.busy = True

        def work() -> None:
            try:
                clip = audio.record(seconds, device)
            except audio.AudioUnavailable as exc:
                self.busy = False
                self.failed.emit(f"Micrófono no disponible: {exc}")
                return
            self.busy = False
            self.finished.emit(clip)

        threading.Thread(target=work, daemon=True).start()


class MicListener(QObject):
    """Continuous listening: microphone → voice activity detection → recognizer."""

    heard = Signal(object, object)  # Analysis, Recognition
    level = Signal(float, float, bool)  # level dB, threshold dB, speaking
    failed = Signal(str)

    def __init__(self, engine: VoiceEngine, parent: QObject | None = None):
        super().__init__(parent)
        self.engine = engine
        self.segmenter = StreamSegmenter(engine.analyzer.vad, engine.analyzer.feat)
        self._stream: audio.MicStream | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._poll)

    @property
    def running(self) -> bool:
        return self._stream is not None

    def start(self, device: int | None = None) -> bool:
        if self._stream is not None:
            return True
        stream = audio.MicStream(device)
        try:
            stream.start()
        except audio.AudioUnavailable as exc:
            self.failed.emit(f"Micrófono no disponible: {exc}")
            return False
        self.segmenter.reset()
        self._stream = stream
        self._timer.start()
        return True

    def stop(self) -> None:
        self._timer.stop()
        if self._stream is not None:
            self._stream.stop()
            self._stream = None

    def feed(self, samples: np.ndarray) -> list[tuple[Analysis, Recognition]]:
        """Push samples through the segmenter and recognize every finished word."""
        results = []
        for utterance in self.segmenter.feed(samples):
            analysis, recognition = self.engine.recognize(utterance)
            results.append((analysis, recognition))
            self.heard.emit(analysis, recognition)
        self.level.emit(self.segmenter.level_db, self.segmenter.threshold_db, self.segmenter.in_speech)
        return results

    def _poll(self) -> None:
        if self._stream is None:
            return
        chunks = self._stream.drain()
        if chunks:
            self.feed(np.concatenate(chunks))
