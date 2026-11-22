import numpy as np
import pytest

from siseo.dsp import SAMPLE_RATE
from siseo.engine import TrainingError, VoiceEngine, seed_demo
from siseo.recognition import Analyzer, MatchConfig, TemplateMatcher, learn_threshold
from siseo.store import ProfileStore
from siseo.synth import WORDS, synthesize
from siseo.vocabulary import NAMES, action_for


@pytest.fixture(scope="module")
def engine():
    eng = VoiceEngine(ProfileStore(":memory:"))
    seed_demo(eng, takes_per_command=5, seed=11)
    return eng


def test_synth_words_cover_the_vocabulary():
    assert set(NAMES) <= set(WORDS)
    clip = synthesize("ARRIBA", np.random.default_rng(0))
    assert clip.ndim == 1 and len(clip) >= int(1.5 * SAMPLE_RATE)
    assert np.max(np.abs(clip)) <= 1.0


def test_analyzer_reports_silence_without_features():
    analysis = Analyzer().analyze(np.zeros(SAMPLE_RATE))
    assert not analysis.has_speech
    assert analysis.span is None and analysis.features is None
    assert analysis.spectrogram.shape[1] == 24


def test_recognizes_unseen_utterances_of_each_command(engine):
    rng = np.random.default_rng(1234)
    hits = 0
    for name in NAMES:
        for _ in range(6):
            _, result = engine.recognize(synthesize(name, rng))
            hits += result.command == name
    assert hits / (6 * len(NAMES)) >= 0.95


def test_recognition_survives_volume_and_noise_changes(engine):
    rng = np.random.default_rng(77)
    hits = 0
    for name in NAMES:
        for _ in range(3):
            quiet_noisy = 0.3 * synthesize(name, rng, noise_db=-38.0)  # about 20 dB SNR
            _, result = engine.recognize(quiet_noisy)
            hits += result.command == name
            assert 0.0 <= result.confidence <= 1.0
    assert hits / (3 * len(NAMES)) >= 0.85


def test_unknown_word_is_rejected(engine):
    rng = np.random.default_rng(5)
    for _ in range(4):
        _, result = engine.recognize(synthesize("HOLA", rng))
        assert not result.accepted
        assert result.reason in {"unknown", "ambiguous"}
        assert result.best is not None


def test_silence_is_rejected(engine):
    analysis, result = engine.recognize(np.random.default_rng(0).standard_normal(SAMPLE_RATE) * 0.002)
    assert analysis.span is None
    assert result.reason == "silence"
    assert result.command is None


def test_scores_are_sorted_and_carry_thresholds(engine):
    _, result = engine.recognize(synthesize("PAUSA", np.random.default_rng(9)))
    distances = [s.distance for s in result.scores]
    assert distances == sorted(distances)
    assert {s.command for s in result.scores} == set(NAMES)
    assert all(s.threshold > 0 for s in result.scores)


def test_untrained_matcher_says_so():
    result = TemplateMatcher({}).recognize(np.zeros((10, 24)))
    assert result.reason == "untrained" and not result.accepted


def test_threshold_learned_from_take_spread():
    rng = np.random.default_rng(2)
    a = rng.standard_normal((20, 4))
    tight = [a, a + 0.01, a - 0.01]
    loose = [a, a + 1.0, a - 1.0]
    assert learn_threshold(tight) < learn_threshold(loose)
    assert learn_threshold([a]) == MatchConfig().default_threshold


def test_ambiguous_when_two_commands_are_equally_close():
    a = np.random.default_rng(3).standard_normal((15, 4))
    matcher = TemplateMatcher({"X": [a], "Y": [a.copy()]}, thresholds={"X": 10.0, "Y": 10.0})
    result = matcher.recognize(a + 0.01)
    assert result.reason == "ambiguous"
    assert result.confidence == pytest.approx(0.5, abs=0.01)


def test_engine_refuses_silent_takes_and_updates_after_changes():
    eng = VoiceEngine(ProfileStore(":memory:"))
    with pytest.raises(TrainingError):
        eng.add_take("ARRIBA", np.zeros(SAMPLE_RATE))
    rng = np.random.default_rng(8)
    eng.add_take("ARRIBA", synthesize("ARRIBA", rng))
    assert eng.matcher.commands == ["ARRIBA"]
    eng.add_take("ABAJO", synthesize("ABAJO", rng))
    assert eng.matcher.commands == ["ABAJO", "ARRIBA"]
    eng.clear("ABAJO")
    assert eng.matcher.commands == ["ARRIBA"]
    assert eng.fingerprint("ARRIBA").shape == (24,)
    assert eng.fingerprint("ABAJO") is None


def test_actions_for_commands():
    assert action_for("ARRIBA") == "up"
    assert action_for("pausa") == "pause"
    assert action_for(None) is None
    assert action_for("HOLA") is None
