"""Transactional SQLite state shared by the independent H200 services."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from .ledger_owner import initialize_owner, check_owner


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  actor_ids_json TEXT NOT NULL,
  rollout_steps INTEGER NOT NULL,
  config_sha256 TEXT NOT NULL,
  initial_checkpoint_sha256 TEXT NOT NULL,
  active_checkpoint_sha256 TEXT NOT NULL,
  current_round INTEGER NOT NULL DEFAULT 0,
  beta REAL,
  status TEXT NOT NULL,
  validation_interval_iterations INTEGER NOT NULL DEFAULT 1,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS run_observation_contracts (
  run_id TEXT PRIMARY KEY,
  contract_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS round_iterations (
  run_id TEXT NOT NULL,
  round_id INTEGER NOT NULL,
  iteration_id INTEGER NOT NULL,
  input_checkpoint_sha256 TEXT NOT NULL,
  candidate_checkpoint_sha256 TEXT,
  evaluation_id TEXT,
  validation_goal_reached_rate REAL,
  status TEXT NOT NULL,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  PRIMARY KEY (run_id, round_id, iteration_id)
);
CREATE TABLE IF NOT EXISTS iteration_barriers (
  run_id TEXT NOT NULL,
  round_id INTEGER NOT NULL,
  iteration_id INTEGER NOT NULL,
  actor_id TEXT NOT NULL,
  checkpoint_sha256 TEXT NOT NULL,
  chunk_ids_json TEXT NOT NULL,
  created_at REAL NOT NULL,
  PRIMARY KEY (run_id, round_id, iteration_id, actor_id),
  FOREIGN KEY(run_id, round_id, iteration_id)
    REFERENCES round_iterations(run_id, round_id, iteration_id)
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
CREATE TABLE IF NOT EXISTS chunk_provenance (
  chunk_id TEXT PRIMARY KEY,
  chunk_sha256 TEXT NOT NULL,
  episode_ids_json TEXT NOT NULL,
  rollout_window_id TEXT NOT NULL,
  teacher_checkpoint_sha256 TEXT NOT NULL,
  sealed_window_sha256 TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS validation_collections (
  run_id TEXT PRIMARY KEY,
  registration_sha256 TEXT NOT NULL,
  registration_json TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS validation_collection_actors (
  run_id TEXT NOT NULL,
  actor_id TEXT NOT NULL,
  chunk_id TEXT NOT NULL,
  receipt_json TEXT NOT NULL,
  created_at REAL NOT NULL,
  PRIMARY KEY (run_id, actor_id)
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
CREATE TABLE IF NOT EXISTS learner_job_iterations (
  job_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  round_id INTEGER NOT NULL,
  iteration_id INTEGER NOT NULL,
  FOREIGN KEY(job_id) REFERENCES learner_jobs(job_id),
  FOREIGN KEY(run_id, round_id, iteration_id)
    REFERENCES round_iterations(run_id, round_id, iteration_id)
);
CREATE TABLE IF NOT EXISTS replay_bc_attestations (
  path TEXT PRIMARY KEY,
  size_bytes INTEGER NOT NULL,
  mtime_ns INTEGER NOT NULL,
  sha256 TEXT NOT NULL,
  checked_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS success_evaluations (
  evaluation_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  round_id INTEGER NOT NULL,
  purpose TEXT NOT NULL,
  checkpoint_sha256 TEXT NOT NULL,
  scenario_manifest_sha256 TEXT NOT NULL,
  scenario_ids_json TEXT NOT NULL,
  observation_mode TEXT NOT NULL,
  metric TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS success_evaluation_outcomes (
  evaluation_id TEXT NOT NULL,
  scenario_id TEXT NOT NULL,
  idempotency_key TEXT NOT NULL UNIQUE,
  request_json TEXT NOT NULL,
  goal_reached INTEGER NOT NULL,
  reward REAL NOT NULL,
  steps INTEGER NOT NULL,
  termination TEXT NOT NULL,
  safety_violation INTEGER NOT NULL,
  created_at REAL NOT NULL,
  PRIMARY KEY (evaluation_id, scenario_id),
  FOREIGN KEY(evaluation_id) REFERENCES success_evaluations(evaluation_id)
);
CREATE TABLE IF NOT EXISTS success_evaluation_iterations (
  evaluation_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  round_id INTEGER NOT NULL,
  iteration_id INTEGER NOT NULL,
  FOREIGN KEY(evaluation_id) REFERENCES success_evaluations(evaluation_id),
  FOREIGN KEY(run_id, round_id, iteration_id)
    REFERENCES round_iterations(run_id, round_id, iteration_id)
);
CREATE TABLE IF NOT EXISTS mse_round_protocols (
  run_id TEXT NOT NULL,
  round_id INTEGER NOT NULL,
  protocol_sha256 TEXT NOT NULL,
  protocol_json TEXT NOT NULL,
  selection_json TEXT,
  status TEXT NOT NULL,
  stop_reason TEXT,
  final_test_id TEXT,
  training_provenance_json TEXT NOT NULL,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  PRIMARY KEY(run_id, round_id),
  FOREIGN KEY(run_id) REFERENCES runs(run_id)
);
CREATE TABLE IF NOT EXISTS step_prepares (
  idempotency_key TEXT PRIMARY KEY,
  request_sha256 TEXT NOT NULL,
  run_id TEXT NOT NULL,
  round_id INTEGER NOT NULL,
  actor_id TEXT NOT NULL,
  rollout_window_id TEXT NOT NULL,
  rollout_step_index INTEGER NOT NULL,
  episode_id TEXT NOT NULL,
  episode_step_index INTEGER NOT NULL,
  capture_digest TEXT NOT NULL,
  prepare_receipt TEXT NOT NULL,
  metadata_json TEXT NOT NULL,
  response_json TEXT NOT NULL,
  artifact_path TEXT NOT NULL,
  checkpoint_sha256 TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  UNIQUE(run_id, round_id, actor_id, rollout_window_id, rollout_step_index)
);
CREATE TABLE IF NOT EXISTS step_commits (
  idempotency_key TEXT PRIMARY KEY,
  request_sha256 TEXT NOT NULL,
  prepare_idempotency_key TEXT NOT NULL UNIQUE,
  transition_json TEXT NOT NULL,
  sealed_window_json TEXT,
  response_json TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  FOREIGN KEY(prepare_idempotency_key) REFERENCES step_prepares(idempotency_key)
);
"""


class StateStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "state.sqlite3"
        # New ledgers are owner-host bound and use rollback journals. Do not
        # silently convert/relabel an existing live WAL after a failure.
        with initialize_owner(self.root) as owner:
            self.owner = owner
            self._initialize()

    def _initialize(self):
        bootstrap = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        bootstrap.row_factory = sqlite3.Row
        if bootstrap.execute("PRAGMA journal_mode").fetchone()[0].lower() != 'delete':
            bootstrap.close()
            raise RuntimeError('new single-host ledger requires DELETE journal; offline import required')
        bootstrap.execute("PRAGMA synchronous=FULL")
        bootstrap.execute("PRAGMA foreign_keys=ON")
        bootstrap.execute("PRAGMA busy_timeout=30000")
        try:
            db = bootstrap
            db.executescript(SCHEMA)
            # Additive migration for state directories created before validation
            # cadence became part of the immutable run contract.  Existing runs
            # retain the former behavior (validate every iteration).
            run_columns = {
                row[1] for row in db.execute("PRAGMA table_info(runs)")
            }
            if "validation_interval_iterations" not in run_columns:
                db.execute(
                    "ALTER TABLE runs ADD COLUMN validation_interval_iterations "
                    "INTEGER NOT NULL DEFAULT 1"
                )
            if "initial_checkpoint_sha256" not in run_columns:
                db.execute("ALTER TABLE runs ADD COLUMN initial_checkpoint_sha256 TEXT")
                db.execute(
                    "UPDATE runs SET initial_checkpoint_sha256=COALESCE("
                    "(SELECT input_checkpoint_sha256 FROM round_iterations i "
                    "WHERE i.run_id=runs.run_id ORDER BY round_id,iteration_id LIMIT 1),"
                    "active_checkpoint_sha256)"
                )
            mse_columns = {row[1] for row in db.execute("PRAGMA table_info(mse_round_protocols)")}
            if "training_provenance_json" not in mse_columns:
                db.execute("ALTER TABLE mse_round_protocols ADD COLUMN training_provenance_json TEXT NOT NULL DEFAULT '{}'")
        finally:
            bootstrap.close()

    @contextmanager
    def connection(self):
        check_owner(self.root, self.owner)
        db = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA synchronous=FULL")
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=30000")
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
