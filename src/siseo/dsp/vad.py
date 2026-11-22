"""Energy-based voice activity detection.

A frame is "voice" when its energy rises a fixed margin (in dB) above the
noise floor. Short gaps are bridged (the closure before a "b" or "d" is
silent for a few tens of milliseconds) and very short bursts are ignored.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .features import FeatureConfig
from .signal import frame_signal

_EPS = 1e-12


@dataclass(frozen=True)
class VadConfig:
    margin_db: float = 12.0
    """How far above the noise floor a frame must be to count as voice."""
    floor_db: float = -50.0
    """Absolute floor: nothing quieter than this is ever voice."""
    flat_speech_db: float = -30.0
    """For clips with no dynamic range: the level above which the whole clip is voice."""
    min_speech_ms: float = 120.0
    max_speech_ms: float = 1600.0
    hangover_ms: float = 160.0
    """Silences shorter than this inside a word do not split it."""
    pad_ms: float = 40.0
    """Context kept before and after the detected word."""
    calibration_ms: float = 400.0
    """Streaming only: initial noise-floor estimation window."""


def frame_energy_db(x: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """Mean-square energy of every frame, in dB relative to full scale."""
    frames = frame_signal(np.asarray(x, dtype=np.float64), frame_len, hop)
    if len(frames) == 0:
        return np.zeros(0)
    return 10.0 * np.log10(np.mean(frames**2, axis=1) + _EPS)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open ``[start, end)`` index ranges where ``mask`` is True."""
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    diff = np.diff(padded)
    return list(zip(np.flatnonzero(diff == 1).tolist(), np.flatnonzero(diff == -1).tolist()))


def detect_speech(
    x: np.ndarray,
    vad: VadConfig = VadConfig(),
    feat: FeatureConfig = FeatureConfig(),
) -> tuple[int, int] | None:
    """Locate the word inside a recorded clip.

    Returns the ``(start, end)`` sample range of the strongest voiced region,
    padded with a little context, or ``None`` when the clip is silence or noise.
    """
    energy = frame_energy_db(x, feat.frame_len, feat.hop)
    if len(energy) == 0:
        return None
    hop_ms = feat.hop_ms
    p10, p90 = np.percentile(energy, [10, 90])
    if p90 - p10 < vad.margin_db / 2:
        # No dynamic range: either steady noise or a clip that is all voice.
        if p10 < vad.flat_speech_db:
            return None
        active = np.ones_like(energy, dtype=bool)
    else:
        threshold = max(p10 + vad.margin_db, vad.floor_db)
        active = energy > threshold

    runs = _runs(active)
    if not runs:
        return None
    gap = int(vad.hangover_ms / hop_ms)
    merged: list[list[int]] = [list(runs[0])]
    for start, end in runs[1:]:
        if start - merged[-1][1] <= gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    best = max(merged, key=lambda r: float(np.sum(10 ** (energy[r[0] : r[1]] / 10))))
    if (best[1] - best[0]) * hop_ms < vad.min_speech_ms:
        return None
    pad = int(vad.pad_ms / hop_ms)
    start = max(0, best[0] - pad) * feat.hop
    end = min(len(x), (best[1] + pad) * feat.hop + feat.frame_len)
    return int(start), int(end)


class StreamSegmenter:
    """Cut a live microphone stream into words, one 10 ms hop at a time.

    Feed raw chunks of any size with :meth:`feed`; it returns the finished
    utterances. The noise floor is measured during the first
    ``calibration_ms`` and then follows the room slowly while nobody speaks.
    """

    def __init__(self, vad: VadConfig = VadConfig(), feat: FeatureConfig = FeatureConfig()):
        self.vad = vad
        self.feat = feat
        self._hangover_hops = max(1, int(vad.hangover_ms / feat.hop_ms))
        self._min_hops = max(1, int(vad.min_speech_ms / feat.hop_ms))
        self._max_hops = max(self._min_hops + 1, int(vad.max_speech_ms / feat.hop_ms))
        self._calib_hops = max(1, int(vad.calibration_ms / feat.hop_ms))
        self.reset()

    def reset(self) -> None:
        """Forget the noise floor and any partial word."""
        self._pending = np.zeros(0)
        self._calibration: list[float] = []
        self.noise_db: float | None = None
        self.level_db: float = self.vad.floor_db
        self._pre_roll: deque[np.ndarray] = deque(
            maxlen=max(1, int(self.vad.pad_ms / self.feat.hop_ms)) + 3
        )
        self._speech: list[np.ndarray] = []
        self._silent_hops = 0
        self._loud_hops = 0

    @property
    def calibrated(self) -> bool:
        return self.noise_db is not None

    @property
    def threshold_db(self) -> float:
        base = self.noise_db if self.noise_db is not None else self.vad.floor_db
        return max(base + self.vad.margin_db, self.vad.floor_db)

    @property
    def in_speech(self) -> bool:
        return bool(self._speech)

    def feed(self, chunk: np.ndarray) -> list[np.ndarray]:
        """Consume samples and return every utterance that ended inside them."""
        hop = self.feat.hop
        data = np.concatenate((self._pending, np.asarray(chunk, dtype=np.float64).ravel()))
        n_hops = len(data) // hop
        self._pending = data[n_hops * hop :]
        out: list[np.ndarray] = []
        for i in range(n_hops):
            utterance = self._step(data[i * hop : (i + 1) * hop])
            if utterance is not None:
                out.append(utterance)
        return out

    def _step(self, block: np.ndarray) -> np.ndarray | None:
        level = 10.0 * np.log10(float(np.mean(block**2)) + _EPS)
        self.level_db = level
        if self.noise_db is None:
            self._calibration.append(level)
            if len(self._calibration) >= self._calib_hops:
                self.noise_db = float(np.median(self._calibration))
            return None

        loud = level > self.threshold_db
        if not self._speech:
            self._pre_roll.append(block)
            if loud:
                self._loud_hops += 1
                if self._loud_hops >= 3:
                    self._speech = list(self._pre_roll)
                    self._pre_roll.clear()
                    self._silent_hops = 0
            else:
                self._loud_hops = 0
                # Follow slow changes of the room while nobody speaks.
                self.noise_db = 0.95 * self.noise_db + 0.05 * level
            return None

        self._speech.append(block)
        self._silent_hops = 0 if loud else self._silent_hops + 1
        too_long = len(self._speech) >= self._max_hops
        if self._silent_hops >= self._hangover_hops or too_long:
            voiced = len(self._speech) - self._silent_hops
            utterance = np.concatenate(self._speech)
            self._speech = []
            self._loud_hops = 0
            self._silent_hops = 0
            if voiced >= self._min_hops:
                return utterance
        return None
