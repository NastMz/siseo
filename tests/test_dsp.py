import numpy as np
import pytest

from siseo.dsp import (
    SAMPLE_RATE,
    FeatureConfig,
    band_energies,
    extract_features,
    frame_signal,
    mel_filterbank,
    pre_emphasis,
    to_float,
)
from siseo.dsp.features import band_centers, power_spectrum

CFG = FeatureConfig()


def tone(freq: float, seconds: float = 0.5, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return amp * np.sin(2 * np.pi * freq * t)


def test_to_float_scales_int16_and_downmixes():
    pcm = np.array([[32767, -32768], [0, 0]], dtype=np.int16)
    out = to_float(pcm)
    assert out.dtype == np.float64
    assert out.shape == (2,)
    assert out[0] == pytest.approx((32767 - 32768) / 2 / 32768)
    assert out[1] == 0.0
    assert to_float(np.array([16384], dtype=np.int16))[0] == pytest.approx(0.5)


def test_pre_emphasis_removes_dc():
    y = pre_emphasis(np.ones(100), 0.97)
    assert y[0] == 1.0
    assert np.allclose(y[1:], 0.03)


def test_frame_signal_shapes_and_padding():
    frames = frame_signal(np.arange(10.0), frame_len=4, hop=3)
    assert frames.shape == (3, 4)
    assert frames[1].tolist() == [3, 4, 5, 6]
    assert frames[2].tolist() == [6, 7, 8, 9]
    assert frame_signal(np.zeros(0), 4, 2).shape == (0, 4)
    with pytest.raises(ValueError):
        frame_signal(np.zeros(5), 0, 1)


def test_power_spectrum_peaks_at_tone_bin():
    frames = frame_signal(tone(1000.0), CFG.frame_len, CFG.hop)
    power = power_spectrum(frames, CFG.n_fft)
    peak_bin = int(np.argmax(power.mean(axis=0)))
    assert peak_bin * SAMPLE_RATE / CFG.n_fft == pytest.approx(1000.0, abs=SAMPLE_RATE / CFG.n_fft)


def test_mel_filterbank_covers_range_with_unit_peaks():
    bank = mel_filterbank(CFG)
    assert bank.shape == (CFG.n_bands, CFG.n_fft // 2 + 1)
    assert np.all(bank >= 0)
    assert np.allclose(bank.max(axis=1), 1.0, atol=0.2)
    centers = band_centers(CFG)
    assert np.all(np.diff(centers) > 0)
    assert np.all(np.diff(np.diff(centers)) > 0)  # mel bands widen with frequency


@pytest.mark.parametrize("freq", [300.0, 1000.0, 3000.0, 6000.0])
def test_tone_energy_lands_in_matching_band(freq):
    energies = band_energies(tone(freq), CFG).mean(axis=0)
    loudest = int(np.argmax(energies))
    centers = band_centers(CFG)
    assert loudest == int(np.argmin(np.abs(centers - freq)))


def test_chirp_moves_energy_up_the_bands_over_time():
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    chirp = 0.5 * np.sin(2 * np.pi * (200 * t + 0.5 * 5800 * t**2))
    peaks = np.argmax(band_energies(chirp, CFG), axis=1)
    quarter = len(peaks) // 4
    assert peaks[:quarter].mean() < peaks[quarter : 2 * quarter].mean() < peaks[-quarter:].mean()


def test_features_are_gain_invariant():
    rng = np.random.default_rng(0)
    signal = tone(440.0) + 0.05 * rng.standard_normal(SAMPLE_RATE // 2)
    a = extract_features(signal, CFG)
    b = extract_features(signal * 0.1, CFG)
    assert np.allclose(a, b, atol=1e-2)
    assert np.allclose(a.mean(axis=0), 0.0, atol=1e-9)
