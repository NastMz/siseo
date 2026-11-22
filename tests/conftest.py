import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Never touch the real ~/.siseo during tests."""
    monkeypatch.setenv("SISEO_HOME", str(tmp_path / "siseo-home"))
    return tmp_path / "siseo-home"
