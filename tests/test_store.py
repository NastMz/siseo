import sqlite3

import numpy as np
import pytest

from siseo.store import ProfileStore, default_db_path


def feats(n=12, bands=24, seed=0):
    return np.random.default_rng(seed).standard_normal((n, bands))


def test_takes_round_trip_features_fingerprint_and_audio(tmp_path):
    path = tmp_path / "nested" / "siseo.db"
    store = ProfileStore(path)
    f = feats()
    audio = np.linspace(-0.5, 0.5, 800)
    tid = store.add_take("arriba", f, f.mean(axis=0), audio=audio, source="mic")
    store.close()

    reopened = ProfileStore(path)
    (take,) = reopened.takes("ARRIBA")
    assert take.id == tid and take.command == "ARRIBA" and take.source == "mic"
    assert np.allclose(take.features, f, atol=1e-5)
    assert np.allclose(take.fingerprint, f.mean(axis=0), atol=1e-5)
    assert np.allclose(take.audio, audio, atol=1e-4)
    reopened.close()


def test_counts_templates_and_band_filter():
    store = ProfileStore()
    store.add_take("ARRIBA", feats(seed=1), np.zeros(24))
    store.add_take("ARRIBA", feats(seed=2), np.zeros(24))
    store.add_take("ABAJO", feats(bands=16), np.zeros(16), source="synthetic")
    assert store.counts() == {"ARRIBA": 2, "ABAJO": 1}
    assert set(store.templates(n_bands=24)) == {"ARRIBA"}
    assert len(store.templates()["ARRIBA"]) == 2


def test_delete_and_clear():
    store = ProfileStore()
    a = store.add_take("PAUSA", feats(), np.zeros(24))
    store.add_take("PAUSA", feats(), np.zeros(24))
    assert store.delete_take(a)
    assert not store.delete_take(a)
    assert store.clear_command("pausa") == 1
    assert store.counts() == {}


def test_rejects_empty_features():
    with pytest.raises(ValueError):
        ProfileStore().add_take("ARRIBA", np.zeros((0, 24)), np.zeros(24))


def test_scores_best_and_ranking():
    store = ProfileStore()
    assert store.best_score() == 0
    store.add_score(4, "voz")
    store.add_score(9, "teclado")
    store.add_score(9, "demo")
    assert store.best_score() == 9
    top = store.top_scores(2)
    assert [(s, m) for s, m, _ in top] == [(9, "teclado"), (9, "demo")]


def test_refuses_newer_schema(tmp_path):
    path = tmp_path / "future.db"
    ProfileStore(path).close()
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE meta SET value='99' WHERE key='schema_version'")
    with pytest.raises(RuntimeError):
        ProfileStore(path)


def test_default_path_honours_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SISEO_HOME", str(tmp_path))
    assert default_db_path() == tmp_path / "siseo.db"
