"""Microphone and WAV input/output.

``sounddevice`` (PortAudio) is imported lazily so the recognizer, the tests
and demo mode work on machines without audio hardware.
"""

from __future__ import annotations

import queue
import wave
from pathlib import Path
from typing import Any

import numpy as np

from .dsp.signal import SAMPLE_RATE, to_float


class AudioUnavailable(RuntimeError):
    """No usable input device (or PortAudio missing)."""


def _sd() -> Any:
    try:
        import sounddevice as sd
    except (ImportError, OSError) as exc:  # PortAudio not installed
        raise AudioUnavailable(str(exc)) from exc
    return sd


def input_devices() -> list[tuple[int, str]]:
    try:
        devices = _sd().query_devices()
    except AudioUnavailable:
        return []
    except Exception:  # PortAudio errors when no host API is present
        return []
    return [(i, d["name"]) for i, d in enumerate(devices) if int(d.get("max_input_channels", 0)) > 0]


def has_microphone() -> bool:
    return bool(input_devices())


def record(seconds: float, device: int | None = None) -> np.ndarray:
    """Blocking capture of ``seconds`` of mono audio at 16 kHz."""
    sd = _sd()
    try:
        data = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="float32", device=device)
        sd.wait()
    except Exception as exc:
        raise AudioUnavailable(str(exc)) from exc
    return data[:, 0].astype(np.float64)


class MicStream:
    """Non-blocking capture: chunks arrive in :attr:`chunks` from the audio thread."""

    def __init__(self, device: int | None = None, block_ms: float = 30.0):
        self.device = device
        self.blocksize = int(SAMPLE_RATE * block_ms / 1000.0)
        self.chunks: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        self._stream: Any = None

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ARG002
        try:
            self.chunks.put_nowait(indata[:, 0].astype(np.float64).copy())
        except queue.Full:
            pass

    def start(self) -> None:
        sd = _sd()
        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                device=self.device,
                blocksize=self.blocksize,
                callback=self._callback,
            )
            self._stream.start()
        except Exception as exc:
            self._stream = None
            raise AudioUnavailable(str(exc)) from exc

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

    def drain(self) -> list[np.ndarray]:
        out = []
        while True:
            try:
                out.append(self.chunks.get_nowait())
            except queue.Empty:
                return out


def resample(x: np.ndarray, rate: int, target: int = SAMPLE_RATE) -> np.ndarray:
    """Linear-interpolation resampling (enough for speech features)."""
    if rate == target or len(x) == 0:
        return x
    n = int(round(len(x) * target / rate))
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)


def read_wav(path: Path | str) -> np.ndarray:
    """Read a PCM WAV file as mono float at 16 kHz."""
    with wave.open(str(path), "rb") as w:
        width, channels, rate = w.getsampwidth(), w.getnchannels(), w.getframerate()
        raw = w.readframes(w.getnframes())
    if width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif width == 2:
        data = to_float(np.frombuffer(raw, dtype="<i2"))
    elif width == 4:
        data = to_float(np.frombuffer(raw, dtype="<i4"))
    else:
        raise ValueError(f"unsupported sample width: {width * 8} bits")
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return resample(data, rate)


def write_wav(path: Path | str, x: np.ndarray, rate: int = SAMPLE_RATE) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
