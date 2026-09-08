"""Transactional SQLite state shared by the independent H200 services."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  actor_ids_json TEXT NOT NULL,
  rollout_steps INTEGER NOT NULL,
  config_sha256 TEXT NOT NULL,
  active_checkpoint_sha256 TEXT NOT NULL,
  current_round INTEGER NOT NULL DEFAULT 0,
  beta REAL,
  status TEXT NOT NULL,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS barriers (
  run_id TEXT NOT NULL,
  round_id INTEGER NOT NULL,
  actor_id TEXT NOT NULL,
  checkpoint_sha256 TEXT NOT NULL,
  chunk_ids_json TEXT NOT NULL,
  created_at REAL NOT NULL,
  PRIMARY KEY (run_id, round_id, actor_id)
);
CREATE TABLE IF NOT EXISTS chunk_reservations (
  chunk_id TEXT PRIMARY KEY,
  sha256 TEXT NOT NULL,
  size_bytes INTEGER NOT NULL,
  metadata_json TEXT NOT NULL,
  status TEXT NOT NULL,
  path TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS checkpoints (
  sha256 TEXT PRIMARY KEY,
  size_bytes INTEGER NOT NULL,
  metadata_json TEXT NOT NULL,
  path TEXT NOT NULL,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS promotions (
  run_id TEXT PRIMARY KEY,
  round_id INTEGER NOT NULL,
  checkpoint_sha256 TEXT NOT NULL,
  updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS learner_jobs (
  job_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  round_id INTEGER NOT NULL,
  input_checkpoint_sha256 TEXT NOT NULL,
  chunk_ids_json TEXT NOT NULL,
  status TEXT NOT NULL,
  request_json TEXT NOT NULL,
  result_json TEXT,
  log_path TEXT NOT NULL,
  pid INTEGER,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
"""


class StateStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "state.sqlite3"
        with self.connection() as db:
            db.executescript(SCHEMA)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
        finally:
            db.close()

    @contextmanager
    def transaction(self):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
            except Exception:
                db.rollback()
                raise
            else:
                db.commit()
