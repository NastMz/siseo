"""Basic signal helpers shared by the whole pipeline."""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16_000
"""Every signal inside Siseo is mono float64 at 16 kHz (speech band up to 8 kHz)."""


def to_float(x: np.ndarray) -> np.ndarray:
    """Return a mono float64 copy of ``x`` scaled to [-1, 1].

    Integer PCM is divided by its full-scale value; multichannel input is averaged.
    """
    arr = np.asarray(x)
    if np.issubdtype(arr.dtype, np.integer):
        out = arr.astype(np.float64) / (float(np.iinfo(arr.dtype).max) + 1.0)
    else:
        out = arr.astype(np.float64, copy=True)
    return out.mean(axis=1) if out.ndim > 1 else out


def pre_emphasis(x: np.ndarray, coef: float = 0.97) -> np.ndarray:
    """First-order high-pass ``y[n] = x[n] - coef * x[n-1]`` that flattens the voice tilt."""
    if len(x) == 0:
        return x.copy()
    return np.concatenate(([x[0]], x[1:] - coef * x[:-1]))


def frame_signal(x: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """Split ``x`` into overlapping frames of ``frame_len`` samples every ``hop`` samples.

    The tail is zero-padded so that every sample belongs to at least one frame.
    Returns an array of shape ``(n_frames, frame_len)``; empty input gives zero frames.
    """
    if frame_len <= 0 or hop <= 0:
        raise ValueError("frame_len and hop must be positive")
    n = len(x)
    if n == 0:
        return np.zeros((0, frame_len))
    n_frames = 1 + max(0, int(np.ceil((n - frame_len) / hop)))
    padded = np.zeros((n_frames - 1) * hop + frame_len)
    padded[:n] = x
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n_frames)[:, None]
    return padded[idx]
