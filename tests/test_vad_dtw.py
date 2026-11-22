import numpy as np
import pytest

from siseo.dsp import SAMPLE_RATE, StreamSegmenter, VadConfig, detect_speech, dtw_distance
from siseo.synth import synthesize


def burst_in_silence(start_s=0.5, length_s=0.4, total_s=1.5, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(total_s * SAMPLE_RATE)) * 1e-3
    t = np.arange(int(length_s * SAMPLE_RATE)) / SAMPLE_RATE
    a = int(start_s * SAMPLE_RATE)
    x[a : a + len(t)] += 0.4 * np.sin(2 * np.pi * 500 * t)
    return x


def test_detect_speech_finds_the_burst():
    span = detect_speech(burst_in_silence())
    assert span is not None
    start, end = span
    assert 0.42 < start / SAMPLE_RATE <= 0.5  # 40 ms of context is kept
    assert abs(end / SAMPLE_RATE - 0.9) < 0.08


def test_detect_speech_rejects_silence_noise_and_clicks():
    rng = np.random.default_rng(1)
    assert detect_speech(np.zeros(SAMPLE_RATE)) is None
    assert detect_speech(rng.standard_normal(SAMPLE_RATE) * 0.005) is None
    click = burst_in_silence(length_s=0.03)
    assert detect_speech(click) is None


def test_detect_speech_bridges_short_gaps_inside_a_word():
    x = burst_in_silence(start_s=0.3, length_s=0.2)
    second = burst_in_silence(start_s=0.56, length_s=0.2)
    x[int(0.56 * SAMPLE_RATE) :] = second[int(0.56 * SAMPLE_RATE) :]  # 60 ms closure between parts
    span = detect_speech(x)
    assert span is not None
    assert span[0] / SAMPLE_RATE < 0.32
    assert span[1] / SAMPLE_RATE > 0.74


def test_detect_speech_on_clip_that_is_all_voice():
    t = np.arange(SAMPLE_RATE // 2) / SAMPLE_RATE
    span = detect_speech(0.4 * np.sin(2 * np.pi * 300 * t))
    assert span is not None


def test_stream_segmenter_emits_one_word_per_utterance():
    rng = np.random.default_rng(3)
    seg = StreamSegmenter(VadConfig())
    stream = np.concatenate(
        [synthesize("ARRIBA", rng, clip_seconds=1.2), synthesize("PAUSA", rng, clip_seconds=1.2)]
    )
    words = []
    for i in range(0, len(stream), 480):  # 30 ms chunks, like the microphone
        words += seg.feed(stream[i : i + 480])
    words += seg.feed(np.zeros(SAMPLE_RATE // 2))
    assert seg.calibrated
    assert len(words) == 2
    assert all(0.2 < len(w) / SAMPLE_RATE < 1.2 for w in words)


def test_stream_segmenter_ignores_steady_noise():
    rng = np.random.default_rng(4)
    seg = StreamSegmenter()
    assert seg.feed(rng.standard_normal(3 * SAMPLE_RATE) * 0.01) == []
    assert not seg.in_speech


def test_dtw_is_zero_for_identical_and_tolerates_time_stretch():
    rng = np.random.default_rng(5)
    a = rng.standard_normal((40, 6))
    stretched = np.repeat(a, 2, axis=0)
    other = rng.standard_normal((40, 6))
    assert dtw_distance(a, a) == pytest.approx(0.0, abs=1e-6)
    assert dtw_distance(a, stretched) < 0.05 * dtw_distance(a, other)


def test_dtw_matches_naive_recurrence():
    rng = np.random.default_rng(6)
    a, b = rng.standard_normal((7, 3)), rng.standard_normal((5, 3))
    c = np.linalg.norm(a[:, None] - b[None], axis=2)
    d = np.full((8, 6), np.inf)
    d[0, 0] = 0
    for i in range(1, 8):
        for j in range(1, 6):
            d[i, j] = c[i - 1, j - 1] + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
    assert dtw_distance(a, b) == pytest.approx(d[7, 5] / 12)


def test_dtw_validates_shapes():
    with pytest.raises(ValueError):
        dtw_distance(np.zeros((3, 2)), np.zeros((3, 4)))
    assert dtw_distance(np.zeros((0, 2)), np.zeros((3, 2))) == float("inf")
