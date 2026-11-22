"""From a raw clip to everything the recognizer and the UI need to know about it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..dsp.features import FeatureConfig, band_energies, compress, normalize
from ..dsp.signal import to_float
from ..dsp.vad import VadConfig, detect_speech


@dataclass(frozen=True)
class Analysis:
    """A clip after voice detection and feature extraction."""

    signal: np.ndarray
    """The whole clip, float in [-1, 1]."""
    span: tuple[int, int] | None
    """Sample range of the detected word, or None if only silence/noise was heard."""
    spectrogram: np.ndarray
    """Log band energies of the whole clip, shape (frames, bands), for display."""
    features: np.ndarray | None
    """Normalized band energies of the word only: what gets matched."""
    fingerprint: np.ndarray | None
    """Mean log energy per band over the word: its spectral signature."""

    @property
    def has_speech(self) -> bool:
        return self.features is not None


class Analyzer:
    """Voice detection followed by the FFT band-energy front end."""

    def __init__(self, feat: FeatureConfig = FeatureConfig(), vad: VadConfig = VadConfig()):
        self.feat = feat
        self.vad = vad

    def analyze(self, clip: np.ndarray, *, already_trimmed: bool = False) -> Analysis:
        x = to_float(clip)
        spectrogram = band_energies(x, self.feat)
        span = (0, len(x)) if already_trimmed and len(x) else detect_speech(x, self.vad, self.feat)
        if span is None:
            return Analysis(x, None, spectrogram, None, None)
        word = band_energies(x[span[0] : span[1]], self.feat)
        features = normalize(compress(word, self.feat.dynamic_range_db))
        return Analysis(x, span, spectrogram, features, word.mean(axis=0))
