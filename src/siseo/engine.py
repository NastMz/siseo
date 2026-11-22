"""The recognizer as one object: train takes, recognize clips, keep templates fresh."""

from __future__ import annotations

import threading

import numpy as np

from .dsp.features import FeatureConfig
from .dsp.vad import VadConfig
from .recognition import Analysis, Analyzer, MatchConfig, Recognition, TemplateMatcher
from .store import ProfileStore
from .synth import synthesize
from .vocabulary import NAMES


class TrainingError(ValueError):
    """A take could not be used (for example, nothing but silence was recorded)."""


class VoiceEngine:
    def __init__(
        self,
        store: ProfileStore,
        feat: FeatureConfig = FeatureConfig(),
        vad: VadConfig = VadConfig(),
        match: MatchConfig = MatchConfig(),
    ):
        self.store = store
        self.analyzer = Analyzer(feat, vad)
        self.match_cfg = match
        self._matcher: TemplateMatcher | None = None
        self._lock = threading.Lock()

    @property
    def matcher(self) -> TemplateMatcher:
        with self._lock:
            if self._matcher is None:
                templates = self.store.templates(n_bands=self.analyzer.feat.n_bands)
                self._matcher = TemplateMatcher(templates, cfg=self.match_cfg)
            return self._matcher

    def invalidate(self) -> None:
        with self._lock:
            self._matcher = None

    def add_take(self, command: str, clip: np.ndarray, source: str = "mic") -> Analysis:
        analysis = self.analyzer.analyze(clip)
        if not analysis.has_speech or analysis.span is None:
            raise TrainingError("No se detectó voz en la toma.")
        start, end = analysis.span
        self.store.add_take(
            command,
            analysis.features,  # type: ignore[arg-type]
            analysis.fingerprint,  # type: ignore[arg-type]
            audio=analysis.signal[start:end],
            source=source,
        )
        self.invalidate()
        return analysis

    def delete_take(self, take_id: int) -> None:
        self.store.delete_take(take_id)
        self.invalidate()

    def clear(self, command: str) -> None:
        self.store.clear_command(command)
        self.invalidate()

    def recognize(self, clip: np.ndarray, *, already_trimmed: bool = False) -> tuple[Analysis, Recognition]:
        analysis = self.analyzer.analyze(clip, already_trimmed=already_trimmed)
        return analysis, self.matcher.recognize(analysis.features)

    def fingerprint(self, command: str) -> np.ndarray | None:
        takes = self.store.takes(command, n_bands=self.analyzer.feat.n_bands)
        if not takes:
            return None
        return np.mean([t.fingerprint for t in takes], axis=0)


def seed_demo(engine: VoiceEngine, takes_per_command: int = 5, seed: int = 7) -> None:
    """Fill the store with synthetic takes of every command (demo mode)."""
    rng = np.random.default_rng(seed)
    for name in NAMES:
        for _ in range(takes_per_command):
            engine.add_take(name, synthesize(name, rng), source="synthetic")
