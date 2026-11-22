import wave

import numpy as np
import pytest

from siseo import audio, cli
from siseo.dsp import SAMPLE_RATE
from siseo.synth import synthesize


def test_wav_round_trip_and_resampling(tmp_path):
    x = 0.5 * np.sin(2 * np.pi * 440 * np.arange(SAMPLE_RATE // 4) / SAMPLE_RATE)
    path = tmp_path / "tone.wav"
    audio.write_wav(path, x)
    assert np.allclose(audio.read_wav(path), x, atol=1e-4)

    stereo_44k = tmp_path / "stereo.wav"
    pcm = (np.stack([x, x], axis=1) * 32767).astype("<i2")
    with wave.open(str(stereo_44k), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(32000)
        w.writeframes(pcm.tobytes())
    back = audio.read_wav(stereo_44k)
    assert len(back) == len(x) // 2


def test_microphone_absence_is_not_an_error(monkeypatch):
    def broken():
        raise audio.AudioUnavailable("no PortAudio")

    monkeypatch.setattr(audio, "_sd", broken)
    assert audio.input_devices() == []
    assert not audio.has_microphone()
    with pytest.raises(audio.AudioUnavailable):
        audio.record(0.1)


def write_words(folder, word, n, seed):
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(n):
        p = folder / f"{word.lower()}_{i}.wav"
        audio.write_wav(p, synthesize(word, rng))
        paths.append(p)
    return paths


def test_cli_import_recognize_eval_and_status(tmp_path, capsys):
    db = tmp_path / "cli.db"
    for k, word in enumerate(("ARRIBA", "ABAJO", "PAUSA")):
        files = write_words(tmp_path / "train" / word, word, 3, seed=k)
        assert cli.main(["--db", str(db), "import", word.lower(), *map(str, files)]) == 0
    test_files = write_words(tmp_path / "test", "ABAJO", 3, seed=40)

    capsys.readouterr()
    assert cli.main(["--db", str(db), "recognize", str(test_files[0])]) == 0
    assert "abajo" in capsys.readouterr().out

    assert cli.main(["--db", str(db), "eval", str(tmp_path / "test"), "abajo"]) == 0
    assert "3/3" in capsys.readouterr().out

    assert cli.main(["--db", str(db), "status"]) == 0
    out = capsys.readouterr().out
    assert "arriba" in out and "3 tomas" in out

    assert cli.main(["--db", str(db), "clear", "pausa"]) == 0
    assert "3 tomas eliminadas" in capsys.readouterr().out


def test_cli_rejects_unknown_words_and_empty_dirs(tmp_path, capsys):
    with pytest.raises(SystemExit):
        cli.main(["--db", str(tmp_path / "x.db"), "clear", "saltar"])
    assert cli.main(["--db", str(tmp_path / "x.db"), "eval", str(tmp_path), "arriba"]) == 2


def test_cli_demo_reports_accuracy(capsys):
    assert cli.main(["demo", "--trials", "3"]) == 0
    out = capsys.readouterr().out
    assert "comandos reconocidos: 15/15" in out
    assert "rechazada: 3/3" in out
