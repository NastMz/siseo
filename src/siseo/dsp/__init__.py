"""Digital signal processing: framing, FFT band energies, voice activity and DTW."""

from .dtw import dtw_distance
from .features import FeatureConfig, band_energies, extract_features, mel_filterbank
from .signal import SAMPLE_RATE, frame_signal, pre_emphasis, to_float
from .vad import StreamSegmenter, VadConfig, detect_speech

__all__ = [
    "SAMPLE_RATE",
    "FeatureConfig",
    "StreamSegmenter",
    "VadConfig",
    "band_energies",
    "detect_speech",
    "dtw_distance",
    "extract_features",
    "frame_signal",
    "mel_filterbank",
    "pre_emphasis",
    "to_float",
]
