"""FFT band energies: the heart of the recognizer.

Each 25 ms frame goes through a Hamming window and a real FFT. The power
spectrum is then summed into ``n_bands`` triangular bands spaced on the mel
scale (narrow at low frequencies, wide at high ones, like the ear). The log of
each band energy is one feature; a word becomes a sequence of band-energy
vectors, one per 10 ms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .signal import SAMPLE_RATE, frame_signal, pre_emphasis

_EPS = 1e-10


@dataclass(frozen=True)
class FeatureConfig:
    """Parameters of the band-energy front end."""

    sample_rate: int = SAMPLE_RATE
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    n_fft: int = 512
    n_bands: int = 24
    f_min: float = 80.0
    f_max: float = 7600.0
    pre_emphasis: float = 0.97
    dynamic_range_db: float = 20.0
    """Band energies more than this below the loudest one are clamped to that floor."""

    @property
    def frame_len(self) -> int:
        return int(round(self.sample_rate * self.frame_ms / 1000.0))

    @property
    def hop(self) -> int:
        return int(round(self.sample_rate * self.hop_ms / 1000.0))


def hz_to_mel(f: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(cfg: FeatureConfig) -> np.ndarray:
    """Triangular filters of shape ``(n_bands, n_fft // 2 + 1)``.

    Band edges are evenly spaced in mel between ``f_min`` and ``f_max``. Each
    filter has unit peak, so a band energy is a weighted sum of FFT bins.
    """
    n_bins = cfg.n_fft // 2 + 1
    freqs = np.linspace(0.0, cfg.sample_rate / 2.0, n_bins)
    edges = mel_to_hz(np.linspace(hz_to_mel(cfg.f_min), hz_to_mel(cfg.f_max), cfg.n_bands + 2))
    bank = np.zeros((cfg.n_bands, n_bins))
    for b in range(cfg.n_bands):
        lo, mid, hi = edges[b], edges[b + 1], edges[b + 2]
        rising = (freqs - lo) / (mid - lo)
        falling = (hi - freqs) / (hi - mid)
        bank[b] = np.clip(np.minimum(rising, falling), 0.0, None)
    return bank


def band_centers(cfg: FeatureConfig) -> np.ndarray:
    """Center frequency in Hz of every band (for axis labels)."""
    edges = mel_to_hz(np.linspace(hz_to_mel(cfg.f_min), hz_to_mel(cfg.f_max), cfg.n_bands + 2))
    return edges[1:-1]


def power_spectrum(frames: np.ndarray, n_fft: int) -> np.ndarray:
    """Hamming-windowed power spectrum ``|FFT|^2 / n_fft`` of every frame."""
    if len(frames) == 0:
        return np.zeros((0, n_fft // 2 + 1))
    windowed = frames * np.hamming(frames.shape[1])[None, :]
    spec = np.fft.rfft(windowed, n=n_fft, axis=1)
    return (np.abs(spec) ** 2) / n_fft


def band_energies(x: np.ndarray, cfg: FeatureConfig = FeatureConfig()) -> np.ndarray:
    """Log band energies of shape ``(n_frames, n_bands)`` for a float signal."""
    emphasized = pre_emphasis(np.asarray(x, dtype=np.float64), cfg.pre_emphasis)
    frames = frame_signal(emphasized, cfg.frame_len, cfg.hop)
    power = power_spectrum(frames, cfg.n_fft)
    return np.log(power @ mel_filterbank(cfg).T + _EPS)


def compress(features: np.ndarray, dynamic_range_db: float) -> np.ndarray:
    """Clamp log energies to ``dynamic_range_db`` below the utterance maximum.

    Quiet bands and the short silences inside a word are dominated by room
    noise, which changes from one recording to the next. Flattening everything
    below the floor makes silence look the same in a quiet room and a noisy one.
    """
    if len(features) == 0:
        return features.copy()
    floor = features.max() - dynamic_range_db * np.log(10.0) / 10.0
    return np.maximum(features, floor)


def normalize(features: np.ndarray) -> np.ndarray:
    """Subtract the per-band mean over time (cepstral mean normalization).

    A louder voice or a different microphone multiplies the spectrum by a
    constant per band, which is an additive offset after the log. Removing the
    mean cancels it, so templates recorded at one volume match at another.
    """
    if len(features) == 0:
        return features.copy()
    return features - features.mean(axis=0, keepdims=True)


def extract_features(x: np.ndarray, cfg: FeatureConfig = FeatureConfig()) -> np.ndarray:
    """Compressed, mean-normalized band-energy sequence used for template matching."""
    return normalize(compress(band_energies(x, cfg), cfg.dynamic_range_db))
