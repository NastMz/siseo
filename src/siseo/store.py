"""SQLite persistence: training takes (templates) and game scores.

Each take stores its band-energy features, its spectral fingerprint and the
trimmed audio (16-bit PCM) so it can be replayed or re-analyzed later.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS takes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  command TEXT NOT NULL,
  source TEXT NOT NULL,
  n_frames INTEGER NOT NULL,
  n_bands INTEGER NOT NULL,
  features BLOB NOT NULL,
  fingerprint BLOB NOT NULL,
  audio BLOB,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS takes_command ON takes(command);
CREATE TABLE IF NOT EXISTS scores (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  score INTEGER NOT NULL,
  mode TEXT NOT NULL,
  created_at TEXT NOT NULL
);
"""


def data_home() -> Path:
    """``$SISEO_HOME`` if set, otherwise ``~/.siseo``."""
    env = os.environ.get("SISEO_HOME")
    return Path(env) if env else Path.home() / ".siseo"


def default_db_path() -> Path:
    return data_home() / "siseo.db"


@dataclass(frozen=True)
class Take:
    id: int
    command: str
    source: str
    features: np.ndarray
    fingerprint: np.ndarray
    audio: np.ndarray | None
    created_at: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ProfileStore:
    """Thread-safe store; use ``":memory:"`` for demo mode and tests."""

    def __init__(self, path: Path | str = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        with self._lock:
            self._conn.executescript(_SCHEMA)
            version = self._get_meta("schema_version")
            if version is None:
                self._conn.execute(
                    "INSERT INTO meta(key, value) VALUES('schema_version', ?)", (str(SCHEMA_VERSION),)
                )
            elif int(version) > SCHEMA_VERSION:
                raise RuntimeError(f"database schema {version} is newer than this Siseo")
            self._conn.commit()

    def _get_meta(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --- takes -----------------------------------------------------------
    def add_take(
        self,
        command: str,
        features: np.ndarray,
        fingerprint: np.ndarray,
        audio: np.ndarray | None = None,
        source: str = "mic",
    ) -> int:
        feats = np.ascontiguousarray(features, dtype=np.float32)
        if feats.ndim != 2 or len(feats) == 0:
            raise ValueError("features must be a non-empty 2-D array")
        pcm = None
        if audio is not None:
            pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO takes(command, source, n_frames, n_bands, features, fingerprint, audio, created_at)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (
                    command.upper(),
                    source,
                    feats.shape[0],
                    feats.shape[1],
                    feats.tobytes(),
                    np.asarray(fingerprint, dtype="<f4").tobytes(),
                    pcm,
                    _now(),
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def delete_take(self, take_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM takes WHERE id=?", (take_id,))
            self._conn.commit()
            return cur.rowcount > 0

    def clear_command(self, command: str) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM takes WHERE command=?", (command.upper(),))
            self._conn.commit()
            return cur.rowcount

    def takes(self, command: str | None = None, n_bands: int | None = None) -> list[Take]:
        sql = "SELECT id, command, source, n_frames, n_bands, features, fingerprint, audio, created_at FROM takes"
        clauses, args = [], []
        if command is not None:
            clauses.append("command=?")
            args.append(command.upper())
        if n_bands is not None:
            clauses.append("n_bands=?")
            args.append(n_bands)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY id"
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        out = []
        for tid, cmd, source, n_frames, bands, feats, fp, audio, created in rows:
            out.append(
                Take(
                    id=int(tid),
                    command=cmd,
                    source=source,
                    features=np.frombuffer(feats, dtype=np.float32).reshape(n_frames, bands).astype(np.float64),
                    fingerprint=np.frombuffer(fp, dtype="<f4").astype(np.float64),
                    audio=None if audio is None else np.frombuffer(audio, dtype="<i2").astype(np.float64) / 32768.0,
                    created_at=created,
                )
            )
        return out

    def counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute("SELECT command, COUNT(*) FROM takes GROUP BY command").fetchall()
        return {cmd: int(n) for cmd, n in rows}

    def templates(self, n_bands: int | None = None) -> dict[str, list[np.ndarray]]:
        out: dict[str, list[np.ndarray]] = {}
        for take in self.takes(n_bands=n_bands):
            out.setdefault(take.command, []).append(take.features)
        return out

    # --- scores ----------------------------------------------------------
    def add_score(self, score: int, mode: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO scores(score, mode, created_at) VALUES (?,?,?)", (int(score), mode, _now())
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def best_score(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT MAX(score) FROM scores").fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def top_scores(self, limit: int = 5) -> list[tuple[int, str, str]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT score, mode, created_at FROM scores ORDER BY score DESC, id ASC LIMIT ?", (limit,)
            ).fetchall()
        return [(int(s), m, c) for s, m, c in rows]
