"""A tiny formant synthesizer that "speaks" the command words.

It exists so the app can be demonstrated, screenshotted and tested without a
microphone and without anyone's recorded voice. Vowels are a sum of harmonics
of a glottal pitch, shaped by three formant resonances; consonants are bursts
of band-limited noise or short silences. Each call varies pitch, formants
(the "speaker"), tempo and loudness, so no two utterances are identical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dsp.signal import SAMPLE_RATE

VOWELS: dict[str, tuple[float, float, float]] = {
    "a": (780.0, 1250.0, 2550.0),
    "e": (460.0, 1900.0, 2600.0),
    "i": (300.0, 2300.0, 3000.0),
    "o": (500.0, 900.0, 2450.0),
    "u": (330.0, 780.0, 2300.0),
}

# (kind, value, seconds): "v" vowel, "g" glide between two vowels,
# "n" noise burst centered at Hz, "s" silence (stop closure), "r" tap.
Segment = tuple[str, object, float]

WORDS: dict[str, list[Segment]] = {
    "ARRIBA": [("v", "a", 0.11), ("r", None, 0.04), ("v", "i", 0.11), ("s", None, 0.05), ("v", "a", 0.15)],
    "ABAJO": [("v", "a", 0.10), ("s", None, 0.05), ("v", "a", 0.10), ("n", 1800.0, 0.09), ("v", "o", 0.16)],
    "IZQUIERDA": [
        ("v", "i", 0.08),
        ("n", 5500.0, 0.10),
        ("s", None, 0.05),
        ("g", ("i", "e"), 0.13),
        ("r", None, 0.04),
        ("s", None, 0.04),
        ("v", "a", 0.13),
    ],
    "DERECHA": [
        ("s", None, 0.03),
        ("v", "e", 0.10),
        ("r", None, 0.04),
        ("v", "e", 0.10),
        ("n", 3200.0, 0.09),
        ("v", "a", 0.14),
    ],
    "PAUSA": [("s", None, 0.04), ("g", ("a", "u"), 0.17), ("n", 6000.0, 0.11), ("v", "a", 0.14)],
    # Not a command: used to show and test rejection of unknown words.
    "HOLA": [("n", 1200.0, 0.05), ("v", "o", 0.13), ("r", None, 0.03), ("v", "a", 0.16)],
}


@dataclass(frozen=True)
class Voice:
    """A synthetic speaker: pitch and vocal-tract scale."""

    pitch_hz: float = 130.0
    formant_scale: float = 1.0


def random_voice(rng: np.random.Generator) -> Voice:
    return Voice(pitch_hz=float(rng.uniform(100.0, 190.0)), formant_scale=float(rng.uniform(0.94, 1.08)))


def _formant_gain(freqs: np.ndarray, formants: np.ndarray) -> np.ndarray:
    """Spectral envelope at ``freqs`` (harmonics x samples) for ``formants`` (3 x samples)."""
    gain = np.zeros_like(freqs)
    for k, (bandwidth, weight) in enumerate(((90.0, 1.0), (120.0, 0.6), (170.0, 0.3))):
        gain += weight / (1.0 + ((freqs - formants[k][None, :]) / bandwidth) ** 2)
    return gain


def _vowel(f_start: tuple, f_end: tuple, seconds: float, voice: Voice, rng: np.random.Generator) -> np.ndarray:
    n = max(1, int(seconds * SAMPLE_RATE))
    t = np.linspace(0.0, 1.0, n)
    formants = np.array([(a + (b - a) * t) * voice.formant_scale for a, b in zip(f_start, f_end)])
    vibrato = 1.0 + 0.01 * np.sin(2 * np.pi * 5.0 * np.arange(n) / SAMPLE_RATE + rng.uniform(0, 6.28))
    f0 = voice.pitch_hz * vibrato * (1.0 - 0.08 * t)
    n_harm = int(4000.0 // voice.pitch_hz)
    harmonics = np.arange(1, n_harm + 1)[:, None]
    phase = 2 * np.pi * np.cumsum(harmonics * f0[None, :], axis=1) / SAMPLE_RATE
    amp = _formant_gain(harmonics * f0[None, :], formants) / harmonics**0.5
    wave = (amp * np.sin(phase)).sum(axis=0)
    envelope = np.minimum(1.0, np.minimum(np.arange(n), np.arange(n)[::-1]) / (0.012 * SAMPLE_RATE))
    return wave * envelope


def _noise(center_hz: float, seconds: float, voice: Voice, rng: np.random.Generator) -> np.ndarray:
    n = max(1, int(seconds * SAMPLE_RATE))
    spectrum = np.fft.rfft(rng.standard_normal(n))
    freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
    center = center_hz * voice.formant_scale
    spectrum *= np.exp(-0.5 * ((freqs - center) / (0.25 * center)) ** 2)
    burst = np.fft.irfft(spectrum, n)
    burst /= np.max(np.abs(burst)) + 1e-12
    envelope = np.sin(np.linspace(0, np.pi, n)) ** 0.5
    return 0.35 * burst * envelope


def synthesize(
    word: str,
    rng: np.random.Generator | None = None,
    voice: Voice | None = None,
    clip_seconds: float = 1.6,
    noise_db: float = -55.0,
) -> np.ndarray:
    """Return a float clip of ``clip_seconds`` containing ``word`` with silence around it."""
    rng = rng if rng is not None else np.random.default_rng()
    voice = voice if voice is not None else random_voice(rng)
    tempo = float(rng.uniform(0.85, 1.15))
    parts: list[np.ndarray] = []
    for kind, value, seconds in WORDS[word.upper()]:
        dur = seconds * tempo
        if kind == "v":
            f = VOWELS[value]  # type: ignore[index]
            parts.append(_vowel(f, f, dur, voice, rng))
        elif kind == "g":
            a, b = value  # type: ignore[misc]
            parts.append(_vowel(VOWELS[a], VOWELS[b], dur, voice, rng))
        elif kind == "n":
            parts.append(_noise(float(value), dur, voice, rng))  # type: ignore[arg-type]
        elif kind == "r":
            parts.append(0.15 * _vowel(VOWELS["e"], VOWELS["e"], dur, voice, rng))
        else:
            parts.append(np.zeros(int(dur * SAMPLE_RATE)))
    word_wave = np.concatenate(parts)
    word_wave *= float(rng.uniform(0.25, 0.6)) / (np.max(np.abs(word_wave)) + 1e-12)

    total = max(int(clip_seconds * SAMPLE_RATE), len(word_wave) + SAMPLE_RATE // 10)
    clip = rng.standard_normal(total) * 10 ** (noise_db / 20.0)
    slack = total - len(word_wave)
    offset = int(rng.uniform(0.25, 0.6) * slack)
    clip[offset : offset + len(word_wave)] += word_wave
    return clip
