#!/usr/bin/env python3
"""Independently launchable H200 coordinator, trajectory, registry and learner roles."""

from __future__ import annotations

import argparse
import hmac
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from . import SERVICE_SCHEMA
from .contracts import (
    CANONICAL_BETA,
    ROLLOUT_STEPS,
    ChunkReservation,
    ContractError,
    canonical_json,
    exact_keys,
    file_sha256,
    identifier,
    sha256_digest,
    validate_dagger_chunk,
)
from .store import StateStore
from . import mse_protocol
from pipeline.dagger.replay import validate_replay_manifest
from pipeline.dagger.services import validation_collection


class ConflictError(RuntimeError):
    pass


class NotFoundError(RuntimeError):
    pass


class NotReadyError(RuntimeError):
    pass


def _now() -> float:
    return time.time()


# The same success-rate stopping rule is used for every DAgger macro-round.
# One iteration is exactly one fresh 81 x B rollout followed by one optimizer
# update.  A run's immutable validation_interval_iterations controls which
# candidates receive the expensive fixed-panel closed-loop evaluation.
VALIDATION_MIN_EPOCHS = 10
VALIDATION_MAX_EPOCHS = 40
VALIDATION_PATIENCE = 4
VALIDATION_MIN_DELTA = 0.01


def _current_iteration(db, run_id: str, round_id: int):
    return db.execute(
        "SELECT * FROM round_iterations WHERE run_id=? AND round_id=? "
        "ORDER BY iteration_id DESC LIMIT 1",
        (run_id, round_id),
    ).fetchone()


def _validation_due(iteration_id: int, interval: int) -> bool:
    if not isinstance(iteration_id, int) or iteration_id < 1:
        raise ValueError("iteration_id must be positive")
    if not isinstance(interval, int) or interval < 1:
        raise ValueError("validation interval must be positive")
    return iteration_id % interval == 0


def _validation_plateau(rows) -> dict[str, Any]:
    """Derive the immutable maximize-success stopping decision from history."""
    best_rate = -math.inf
    best_checkpoint = None
    best_iteration = None
    bad = 0
    history = []
    for row in rows:
        rate = float(row["validation_goal_reached_rate"])
        improved = rate >= best_rate + VALIDATION_MIN_DELTA
        if improved:
            best_rate = rate
            best_checkpoint = row["candidate_checkpoint_sha256"]
            best_iteration = row["iteration_id"]
            bad = 0
        else:
            bad += 1
        history.append({
            "iteration_id": row["iteration_id"],
            "checkpoint_sha256": row["candidate_checkpoint_sha256"],
            "evaluation_id": row["evaluation_id"],
            "goal_reached_rate": rate,
            "improved": improved,
            "bad_validations": bad,
        })
    count = len(history)
    plateau = count >= VALIDATION_MIN_EPOCHS and bad >= VALIDATION_PATIENCE
    maximum = count >= VALIDATION_MAX_EPOCHS
    return {
        "history": history,
        "best_goal_reached_rate": None if not history else best_rate,
        "best_checkpoint_sha256": best_checkpoint,
        "best_iteration_id": best_iteration,
        "bad_validations": bad,
        "saturated": plateau or maximum,
        "stop_reason": (
            "success_rate_plateau" if plateau
            else "maximum_validation_epochs" if maximum
            else None
        ),
    }


def _final_budget_reached(db, run_id: str, round_id: int, iteration_id: int) -> bool:
    """Return whether the accepted learner request declared this round's end."""
    row = db.execute(
        "SELECT j.request_json FROM learner_jobs j "
        "JOIN learner_job_iterations i ON i.job_id=j.job_id "
        "WHERE i.run_id=? AND i.round_id=? AND i.iteration_id=? "
        "AND j.status='succeeded'",
        (run_id, round_id, iteration_id),
    ).fetchone()
    if row is None:
        return False
    try:
        value = json.loads(row["request_json"])["parameters"].get(
            "final_validation_iteration"
        )
    except (KeyError, TypeError, json.JSONDecodeError):
        return False
    return type(value) is int and value == iteration_id


def _apply_final_budget(state: dict[str, Any], reached: bool) -> dict[str, Any]:
    if not reached:
        return state
    return {**state, "saturated": True, "stop_reason": "round_iteration_budget"}


def _final_round_candidate(db, run_id: str, round_id: int) -> str | None:
    """Return the exact candidate from the declared final learner iteration."""
    row = db.execute(
        "SELECT * FROM round_iterations WHERE run_id=? AND round_id=? "
        "ORDER BY iteration_id DESC LIMIT 1",
        (run_id, round_id),
    ).fetchone()
    if (
        row is None
        or row["status"] != "complete"
        or not row["candidate_checkpoint_sha256"]
        or not _final_budget_reached(db, run_id, round_id, row["iteration_id"])
    ):
        return None
    return str(row["candidate_checkpoint_sha256"])


class DaggerHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, *, role: str, root: Path, token: str,
                 max_upload_bytes: int, worker: Path | None):
        super().__init__(address, handler)
        self.role = role
        self.root = root.resolve()
        self.store = StateStore(self.root)
        self.token = token
        self.max_upload_bytes = max_upload_bytes
        self.worker = worker.resolve() if worker else None
        self.started_at = _now()
        self.jobs_lock = threading.Lock()
        if role == "learner":
            with self.store.transaction() as db:
                interrupted = list(db.execute(
                    "SELECT DISTINCT run_id FROM learner_jobs WHERE status IN ('queued','running')"
                ))
                db.execute(
                    "UPDATE learner_jobs SET status='interrupted',updated_at=? "
                    "WHERE status IN ('queued','running')", (_now(),)
                )
                for row in interrupted:
                    db.execute(
                        "UPDATE runs SET status='learner_failed',updated_at=? WHERE run_id=?",
                        (_now(), row["run_id"]),
                    )
                    db.execute(
                        "UPDATE round_iterations SET status='learner_failed',updated_at=? "
                        "WHERE run_id=? AND status IN ('ready_for_learning','learning')",
                        (_now(), row["run_id"]),
                    )


class Handler(BaseHTTPRequestHandler):
    server: DaggerHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write(
            json.dumps({"time": _now(), "role": self.server.role,
                        "client": self.client_address[0], "message": fmt % args}) + "\n"
        )

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = (json.dumps(payload, sort_keys=True) + "\n").encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, content_type: str) -> None:
        size = path.stat().st_size
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self.send_header("X-Content-SHA256", path.stem)
        self.end_headers()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                self.wfile.write(block)

    def _error(self, status: int, code: str, message: str) -> None:
        self._json(status, {"ok": False, "error": code, "message": message})

    def _authorized(self) -> bool:
        expected = f"Bearer {self.server.token}"
        actual = self.headers.get("Authorization", "")
        return hmac.compare_digest(actual, expected)

    def _require_auth(self) -> bool:
        if self._authorized():
            return True
        self._error(HTTPStatus.UNAUTHORIZED, "unauthorized", "valid bearer token required")
        return False

    def _read_json(self, limit: int = 2 * 1024 * 1024) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ContractError("Content-Length is required")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ContractError("Content-Length must be an integer") from exc
        if length < 0 or length > limit:
            raise ContractError(f"JSON body exceeds {limit} bytes")
        try:
            value = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractError(f"invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ContractError("JSON body must be an object")
        return value

    def _dispatch(self, method: str) -> None:
        path = urllib.parse.urlsplit(self.path).path
        if method == "GET" and path in ("/live", "/ready", "/version"):
            return self._health(path)
        if not self._require_auth():
            return
        role = self.server.role
        if role == "trajectory":
            return self._trajectory(method, path)
        if role == "registry":
            return self._registry(method, path)
        if role == "coordinator":
            return self._coordinator(method, path)
        if role == "learner":
            return self._learner(method, path)
        if role == "evaluation":
            return self._evaluation(method, path)
        raise NotFoundError(path)

    def do_GET(self) -> None:
        self._safe_dispatch("GET")

    def do_POST(self) -> None:
        self._safe_dispatch("POST")

    def do_PUT(self) -> None:
        self._safe_dispatch("PUT")

    def _safe_dispatch(self, method: str) -> None:
        try:
            self._dispatch(method)
        except ContractError as exc:
            self._error(HTTPStatus.BAD_REQUEST, "contract_error", str(exc))
        except ConflictError as exc:
            self._error(HTTPStatus.CONFLICT, "conflict", str(exc))
        except NotFoundError as exc:
            self._error(HTTPStatus.NOT_FOUND, "not_found", str(exc))
        except NotReadyError as exc:
            self._error(HTTPStatus.PRECONDITION_FAILED, "not_ready", str(exc))
        except Exception as exc:
            self.log_error("%s", traceback.format_exc())
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error", str(exc))

    def _health(self, path: str) -> None:
        ready = self.server.role != "learner" or (
            self.server.worker is not None and self.server.worker.is_file()
        )
        if path == "/ready" and not ready:
            return self._json(HTTPStatus.SERVICE_UNAVAILABLE, {
                "ok": False, "role": self.server.role, "reason": "learner worker missing"
            })
        self._json(HTTPStatus.OK, {
            "ok": True,
            "schema": SERVICE_SCHEMA,
            "role": self.server.role,
            "pid": os.getpid(),
            "uptime_seconds": _now() - self.server.started_at,
        })

    # ------------------------------------------------------------ trajectory
    def _trajectory(self, method: str, path: str) -> None:
        if method == "GET" and path == "/v1/status":
            with self.server.store.connection() as db:
                counts = {row["status"]: row["count"] for row in db.execute(
                    "SELECT status,COUNT(*) AS count FROM chunk_reservations GROUP BY status"
                )}
            return self._json(HTTPStatus.OK, {"ok": True, "chunks": counts})
        if method == "POST" and path == "/v1/chunks/reserve":
            reservation = ChunkReservation.parse(self._read_json())
            now = _now()
            meta = canonical_json(reservation.as_dict())
            with self.server.store.transaction() as db:
                row = db.execute(
                    "SELECT sha256,size_bytes,metadata_json,status FROM chunk_reservations WHERE chunk_id=?",
                    (reservation.chunk_id,),
                ).fetchone()
                if row:
                    if row["sha256"] != reservation.sha256 or row["metadata_json"] != meta:
                        raise ConflictError("chunk_id already reserved with different content")
                    return self._json(HTTPStatus.OK, {
                        "ok": True, "chunk_id": reservation.chunk_id,
                        "status": row["status"], "idempotent": True,
                    })
                db.execute(
                    "INSERT INTO chunk_reservations VALUES (?,?,?,?,?,?,?,?)",
                    (reservation.chunk_id, reservation.sha256, reservation.size_bytes,
                     meta, "reserved", None, now, now),
                )
            return self._json(HTTPStatus.CREATED, {
                "ok": True, "chunk_id": reservation.chunk_id,
                "status": "reserved", "upload_path": f"/v1/chunks/{reservation.chunk_id}",
            })

        prefix = "/v1/chunks/"
        if not path.startswith(prefix):
            raise NotFoundError(path)
        chunk_id = identifier(path[len(prefix):], "chunk_id")
        with self.server.store.connection() as db:
            row = db.execute("SELECT * FROM chunk_reservations WHERE chunk_id=?", (chunk_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"unknown chunk {chunk_id}")
        if method == "GET":
            payload = dict(row)
            payload["metadata"] = json.loads(payload.pop("metadata_json"))
            return self._json(HTTPStatus.OK, {"ok": True, "chunk": payload})
        if method != "PUT":
            raise NotFoundError(path)
        if row["status"] == "committed":
            return self._json(HTTPStatus.OK, {
                "ok": True, "chunk_id": chunk_id, "sha256": row["sha256"],
                "status": "committed", "idempotent": True,
            })
        try:
            length = int(self.headers.get("Content-Length", "-1"))
        except ValueError as exc:
            raise ContractError("Content-Length must be an integer") from exc
        if length != row["size_bytes"]:
            raise ContractError(f"Content-Length {length} differs from reserved size {row['size_bytes']}")
        if length > self.server.max_upload_bytes:
            raise ContractError("chunk exceeds configured upload limit")
        chunks = self.server.root / "chunks"
        chunks.mkdir(parents=True, exist_ok=True)
        partial = chunks / f".{chunk_id}.{os.getpid()}.{threading.get_ident()}.part"
        final = chunks / f"{chunk_id}.npz"
        remaining = length
        with partial.open("xb") as stream:
            while remaining:
                block = self.rfile.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    partial.unlink(missing_ok=True)
                    raise ContractError("request body ended before Content-Length")
                stream.write(block)
                remaining -= len(block)
            stream.flush()
            os.fsync(stream.fileno())
        digest = file_sha256(partial)
        if digest != row["sha256"]:
            partial.unlink(missing_ok=True)
            raise ContractError(f"chunk SHA-256 mismatch: {digest}")
        validation = validate_dagger_chunk(partial)
        metadata = json.loads(row["metadata_json"])
        expected_beta = CANONICAL_BETA[metadata["round_id"]]
        if validation["beta_min"] is not None and (
            abs(validation["beta_min"] - expected_beta) > 1e-6
            or abs(validation["beta_max"] - expected_beta) > 1e-6
        ):
            partial.unlink(missing_ok=True)
            raise ContractError(
                f"chunk beta differs from canonical round {metadata['round_id']} beta {expected_beta}"
            )
        if final.exists():
            if file_sha256(final) != digest:
                partial.unlink(missing_ok=True)
                raise ConflictError("committed chunk path contains different bytes")
            partial.unlink()
        else:
            os.replace(partial, final)
        with self.server.store.transaction() as db:
            db.execute(
                "UPDATE chunk_reservations SET status='committed',path=?,updated_at=? WHERE chunk_id=?",
                (str(final), _now(), chunk_id),
            )
        self._json(HTTPStatus.CREATED, {
            "ok": True, "chunk_id": chunk_id, "sha256": digest,
            "status": "committed", "validation": validation,
        })

    # --------------------------------------------------------------- registry
    def _registry(self, method: str, path: str) -> None:
        if method == "GET" and path.startswith("/v1/checkpoint-records/"):
            digest = sha256_digest(path.removeprefix("/v1/checkpoint-records/"), "checkpoint_sha256")
            with self.server.store.connection() as db:
                row = db.execute("SELECT * FROM checkpoints WHERE sha256=?", (digest,)).fetchone()
            if row is None:
                raise NotFoundError(f"checkpoint {digest} is not registered")
            checkpoint = Path(row["path"])
            if not checkpoint.is_file() or checkpoint.stat().st_size != row["size_bytes"]:
                raise NotReadyError("registered checkpoint bytes are unavailable")
            # Metadata-only cold-path lookup: avoids re-registration with a
            # different provenance record at each round boundary. This is not
            # a substitute for hashing a new checkpoint at admission/loading.
            return self._json(HTTPStatus.OK, {"ok": True, "checkpoint": {
                "sha256": digest, "size_bytes": row["size_bytes"], "metadata": json.loads(row["metadata_json"]),
            }})
        if method == "GET" and path == "/v1/status":
            with self.server.store.connection() as db:
                checkpoint_count = db.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
                promotion_count = db.execute("SELECT COUNT(*) FROM promotions").fetchone()[0]
            return self._json(HTTPStatus.OK, {
                "ok": True, "checkpoints": checkpoint_count, "promotions": promotion_count,
            })
        if method == "POST" and path == "/v1/checkpoints/register-local":
            payload = self._read_json()
            exact_keys(payload, {"path", "sha256", "metadata"})
            digest = sha256_digest(payload["sha256"], "sha256")
            source = Path(payload["path"]).resolve()
            allowed = Path(os.environ.get("DAGGER_CHECKPOINT_SOURCE_ROOT", "/workspace/project/roborender/checkpoints")).resolve()
            worker_outputs = (self.server.root / "jobs").resolve()
            if not source.is_file() or not any(
                source.is_relative_to(root) for root in (allowed, worker_outputs)
            ):
                raise ContractError(
                    f"checkpoint path must be below {allowed} or {worker_outputs}"
                )
            if file_sha256(source) != digest:
                raise ContractError("checkpoint SHA-256 mismatch")
            blob_dir = self.server.root / "checkpoints"
            blob_dir.mkdir(parents=True, exist_ok=True)
            final = blob_dir / f"{digest}.pt"
            if not final.exists():
                temporary = blob_dir / f".{digest}.{os.getpid()}.part"
                try:
                    os.link(source, temporary)
                except OSError:
                    shutil.copy2(source, temporary)
                os.replace(temporary, final)
            now = _now()
            meta = canonical_json(payload["metadata"])
            with self.server.store.transaction() as db:
                row = db.execute("SELECT metadata_json FROM checkpoints WHERE sha256=?", (digest,)).fetchone()
                if row and row["metadata_json"] != meta:
                    raise ConflictError("checkpoint digest already has different metadata")
                db.execute(
                    "INSERT OR IGNORE INTO checkpoints VALUES (?,?,?,?,?)",
                    (digest, final.stat().st_size, meta, str(final), now),
                )
            return self._json(HTTPStatus.CREATED, {"ok": True, "sha256": digest, "size_bytes": final.stat().st_size})

        if method == "POST" and path == "/v1/promotions":
            # A learner receipt proves only that optimization completed. It is
            # deliberately insufficient to activate a policy. The coordinator
            # advances through /v1/iterations/advance only after the candidate's
            # fixed-panel Raven success evaluation is complete.
            payload = self._read_json()
            exact_keys(payload, {"run_id", "round_id", "checkpoint_sha256"})
            raise NotReadyError(
                "direct promotion is disabled; complete success validation and "
                "use /v1/iterations/advance"
            )

        if method == "GET" and path.startswith("/v1/active/"):
            run_id = identifier(path.removeprefix("/v1/active/"), "run_id")
            with self.server.store.connection() as db:
                row = db.execute("SELECT * FROM promotions WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise NotFoundError(f"no promoted checkpoint for {run_id}")
            return self._json(HTTPStatus.OK, {"ok": True, "promotion": dict(row)})
        if method == "GET" and path.startswith("/v1/checkpoints/"):
            digest = sha256_digest(path.removeprefix("/v1/checkpoints/"), "checkpoint_sha256")
            with self.server.store.connection() as db:
                row = db.execute("SELECT path FROM checkpoints WHERE sha256=?", (digest,)).fetchone()
            if row is None:
                raise NotFoundError(f"checkpoint {digest} is not registered")
            checkpoint = Path(row["path"])
            if not checkpoint.is_file() or file_sha256(checkpoint) != digest:
                raise NotReadyError("registered checkpoint bytes are unavailable or corrupt")
            return self._file(checkpoint, "application/octet-stream")
        raise NotFoundError(path)

    # ------------------------------------------------------------- coordinator
    def _coordinator(self, method: str, path: str) -> None:
        if path.startswith("/v1/validation-collections/"):
            try:
                if method == "POST" and path == "/v1/validation-collections/register":
                    result = validation_collection.register(self.server.store, self._read_json())
                elif method == "POST" and path == "/v1/validation-collections/finish-actor":
                    result = validation_collection.finish_actor(self.server.store, self._read_json())
                elif method == "GET":
                    result = validation_collection.context(self.server.store, identifier(path.removeprefix("/v1/validation-collections/"), "run_id"))
                else:
                    raise NotFoundError(path)
            except ValueError as exc:
                raise ContractError(str(exc)) from exc
            return self._json(HTTPStatus.OK, result)
        if path.startswith("/v1/mse-rounds/"):
            try:
                if method == "POST" and path == "/v1/mse-rounds/register":
                    result = mse_protocol.register(self.server.store, self._read_json())
                elif method == "POST" and path == "/v1/mse-rounds/record":
                    result = mse_protocol.record(self.server.store, self._read_json())
                elif method == "POST" and path == "/v1/mse-rounds/finish-test":
                    result = mse_protocol.finish_test(self.server.store, self._read_json())
                elif method == "GET":
                    run_id, round_text = path.removeprefix("/v1/mse-rounds/").split("/")
                    result = mse_protocol.context(self.server.store, identifier(run_id, "run_id"), int(round_text))
                else:
                    raise NotFoundError(path)
            except ValueError as exc:
                raise ContractError(str(exc)) from exc
            return self._json(HTTPStatus.OK, result)
        if method == "POST" and path == "/v1/runs":
            payload = self._read_json()
            exact_keys(
                payload,
                {
                    "run_id", "actor_ids", "rollout_steps", "config_sha256",
                    "initial_checkpoint_sha256",
                },
                {"validation_interval_iterations", "student_observation"},
            )
            observation_json = None
            if "student_observation" in payload:
                from ..rtx.code.src.dagger_rtx.observation_contract import validate_generated_only
                try:
                    validate_generated_only(payload["student_observation"])
                except ValueError as exc:
                    raise ContractError(str(exc)) from exc
                observation_json = canonical_json(payload["student_observation"])
            run_id = identifier(payload["run_id"], "run_id")
            actors = payload["actor_ids"]
            if not isinstance(actors, list) or not actors:
                raise ContractError("actor_ids must be a non-empty list")
            actors = [identifier(value, "actor_id") for value in actors]
            if len(actors) != len(set(actors)):
                raise ContractError("actor_ids must be unique")
            if payload["rollout_steps"] != ROLLOUT_STEPS:
                raise ContractError(f"rollout_steps must equal {ROLLOUT_STEPS}")
            config_sha = sha256_digest(payload["config_sha256"], "config_sha256")
            initial = sha256_digest(payload["initial_checkpoint_sha256"], "initial_checkpoint_sha256")
            validation_interval = payload.get("validation_interval_iterations", 1)
            if (
                not isinstance(validation_interval, int)
                or isinstance(validation_interval, bool)
                or validation_interval < 1
            ):
                raise ContractError("validation_interval_iterations must be a positive integer")
            now = _now()
            actor_json = canonical_json(sorted(actors))
            with self.server.store.transaction() as db:
                try:
                    validation_collection.forbid_learning(db, run_id)
                except ValueError as exc:
                    raise ContractError(str(exc)) from exc
                if db.execute("SELECT 1 FROM checkpoints WHERE sha256=?", (initial,)).fetchone() is None:
                    raise NotReadyError("initial checkpoint is not registered")
                old = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
                if old:
                    old_observation = db.execute(
                        "SELECT contract_json FROM run_observation_contracts WHERE run_id=?", (run_id,)
                    ).fetchone()
                    same = (
                        old["actor_ids_json"] == actor_json
                        and old["config_sha256"] == config_sha
                        and old["initial_checkpoint_sha256"] == initial
                        and old["validation_interval_iterations"] == validation_interval
                        and (old_observation[0] if old_observation is not None else None) == observation_json
                    )
                    if not same:
                        raise ConflictError("run_id already exists with a different contract")
                    return self._json(HTTPStatus.OK, {"ok": True, "run_id": run_id, "idempotent": True})
                db.execute(
                    "INSERT INTO runs "
                    "(run_id,actor_ids_json,rollout_steps,config_sha256,"
                    "initial_checkpoint_sha256,active_checkpoint_sha256,"
                    "current_round,beta,status,"
                    "validation_interval_iterations,created_at,updated_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        run_id, actor_json, ROLLOUT_STEPS, config_sha, initial,
                        initial, 0, None, "created", validation_interval, now, now,
                    ),
                )
                if observation_json is not None:
                    db.execute("INSERT INTO run_observation_contracts VALUES (?,?)", (run_id, observation_json))
            return self._json(HTTPStatus.CREATED, {
                "ok": True, "run_id": run_id, "status": "created",
                "validation_interval_iterations": validation_interval,
            })

        if method == "POST" and path == "/v1/rounds/start":
            payload = self._read_json()
            exact_keys(payload, {"run_id", "round_id", "beta", "checkpoint_sha256"})
            run_id = identifier(payload["run_id"], "run_id")
            round_id = payload["round_id"]
            if not isinstance(round_id, int) or round_id not in CANONICAL_BETA:
                raise ContractError("round_id must be 1..4")
            beta = float(payload["beta"])
            if abs(beta - CANONICAL_BETA[round_id]) > 1e-12:
                raise ContractError(f"round {round_id} beta must be {CANONICAL_BETA[round_id]}")
            digest = sha256_digest(payload["checkpoint_sha256"], "checkpoint_sha256")
            with self.server.store.transaction() as db:
                try:
                    validation_collection.forbid_learning(db, run_id)
                except ValueError as exc:
                    raise ContractError(str(exc)) from exc
                run = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
                if run is None:
                    raise NotFoundError(run_id)
                protocol = mse_protocol.get_protocol(db, run_id, round_id)
                if protocol is not None and protocol["selection_json"] is None:
                    raise NotReadyError("held-out MSE iteration-zero baseline is required before collection")
                if run["current_round"] == round_id and run["status"] == "collecting":
                    if run["active_checkpoint_sha256"] != digest or run["beta"] != beta:
                        raise ConflictError("active round differs")
                    return self._json(HTTPStatus.OK, {"ok": True, "idempotent": True})
                if round_id != run["current_round"] + 1:
                    raise NotReadyError("rounds must start sequentially")
                if run["status"] not in ("created", "round_complete"):
                    raise NotReadyError(f"run is {run['status']}")
                if run["current_round"] == 0:
                    expected_checkpoint = run["active_checkpoint_sha256"]
                else:
                    previous_protocol = mse_protocol.get_protocol(db, run_id, run["current_round"])
                    if previous_protocol is not None:
                        if protocol is None or previous_protocol["status"] != "complete":
                            raise NotReadyError("MSE-selected rounds require final test and explicit next-round MSE protocol")
                        expected_checkpoint = json.loads(previous_protocol["selection_json"])["best_checkpoint_sha256"]
                    else:
                        expected_checkpoint = _final_round_candidate(db, run_id, run["current_round"])
                    if expected_checkpoint is None:
                        raise NotReadyError(
                            "previous round final learner candidate is unavailable"
                        )
                if digest != expected_checkpoint:
                    raise ConflictError(
                        "actor checkpoint differs from previous final learner candidate"
                    )
                now = _now()
                db.execute(
                    "UPDATE runs SET current_round=?,beta=?,active_checkpoint_sha256=?,"
                    "status='collecting',updated_at=? WHERE run_id=?",
                    (round_id, beta, digest, now, run_id),
                )
                db.execute(
                    "INSERT INTO round_iterations VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (run_id, round_id, 1, digest, None, None, None,
                     "collecting", now, now),
                )
            return self._json(HTTPStatus.OK, {
                "ok": True, "run_id": run_id, "round_id": round_id,
                "iteration_id": 1, "beta": beta,
            })

        if method == "POST" and path == "/v1/barriers":
            payload = self._read_json()
            exact_keys(payload, {"run_id", "round_id", "actor_id", "checkpoint_sha256", "chunk_ids"})
            run_id = identifier(payload["run_id"], "run_id")
            actor_id = identifier(payload["actor_id"], "actor_id")
            digest = sha256_digest(payload["checkpoint_sha256"], "checkpoint_sha256")
            round_id = payload["round_id"]
            chunk_ids = payload["chunk_ids"]
            if not isinstance(chunk_ids, list) or not chunk_ids:
                raise ContractError("chunk_ids must be a non-empty list")
            chunk_ids = [identifier(value, "chunk_id") for value in chunk_ids]
            if len(chunk_ids) != len(set(chunk_ids)):
                raise ContractError("chunk_ids must be unique")
            chunk_json = canonical_json(sorted(chunk_ids))
            with self.server.store.transaction() as db:
                try:
                    validation_collection.forbid_learning(db, run_id, chunk_ids)
                except ValueError as exc:
                    raise ContractError(str(exc)) from exc
                run = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
                if run is None or run["current_round"] != round_id:
                    raise NotReadyError("run is not in this round")
                actors = json.loads(run["actor_ids_json"])
                if actor_id not in actors:
                    raise ContractError("actor_id is not admitted to this run")
                prior = db.execute(
                    "SELECT * FROM iteration_barriers WHERE run_id=? AND round_id=? "
                    "AND actor_id=? ORDER BY iteration_id DESC LIMIT 1",
                    (run_id, round_id, actor_id),
                ).fetchone()
                if prior is not None and prior["chunk_ids_json"] == chunk_json:
                    if prior["checkpoint_sha256"] != digest:
                        raise ConflictError("actor barrier retry changed its checkpoint")
                    arrived = db.execute(
                        "SELECT COUNT(*) FROM iteration_barriers WHERE run_id=? "
                        "AND round_id=? AND iteration_id=?",
                        (run_id, round_id, prior["iteration_id"]),
                    ).fetchone()[0]
                    return self._json(HTTPStatus.OK, {
                        "ok": True, "idempotent": True,
                        "iteration_id": prior["iteration_id"],
                        "arrived": arrived, "expected": len(actors),
                        "ready_for_learning": arrived == len(actors),
                    })
                if run["status"] != "collecting":
                    raise NotReadyError("run is not collecting this round")
                if digest != run["active_checkpoint_sha256"]:
                    raise ConflictError("actor used a different checkpoint")
                iteration = _current_iteration(db, run_id, round_id)
                if iteration is None or iteration["status"] != "collecting":
                    raise NotReadyError("DAgger iteration is not collecting")
                if iteration["input_checkpoint_sha256"] != digest:
                    raise ConflictError("actor checkpoint differs from iteration checkpoint")
                for chunk_id in chunk_ids:
                    row = db.execute("SELECT status,metadata_json FROM chunk_reservations WHERE chunk_id=?", (chunk_id,)).fetchone()
                    if row is None or row["status"] != "committed":
                        raise NotReadyError(f"chunk {chunk_id} is not committed")
                    metadata = json.loads(row["metadata_json"])
                    if (metadata["run_id"], metadata["round_id"], metadata["actor_id"]) != (run_id, round_id, actor_id):
                        raise ConflictError(f"chunk {chunk_id} provenance differs")
                    if metadata["checkpoint_sha256"] != digest:
                        raise ConflictError(
                            f"chunk {chunk_id} was generated by a different checkpoint"
                        )
                old = db.execute(
                    "SELECT * FROM iteration_barriers WHERE run_id=? AND round_id=? "
                    "AND iteration_id=? AND actor_id=?",
                    (run_id, round_id, iteration["iteration_id"], actor_id),
                ).fetchone()
                if old and (old["checkpoint_sha256"] != digest or old["chunk_ids_json"] != chunk_json):
                    raise ConflictError("actor barrier already contains different chunks")
                db.execute(
                    "INSERT OR IGNORE INTO iteration_barriers VALUES (?,?,?,?,?,?,?)",
                    (run_id, round_id, iteration["iteration_id"], actor_id,
                     digest, chunk_json, _now()),
                )
                count = db.execute(
                    "SELECT COUNT(*) FROM iteration_barriers WHERE run_id=? "
                    "AND round_id=? AND iteration_id=?",
                    (run_id, round_id, iteration["iteration_id"]),
                ).fetchone()[0]
                if count == len(actors):
                    now = _now()
                    db.execute(
                        "UPDATE round_iterations SET status='ready_for_learning',updated_at=? "
                        "WHERE run_id=? AND round_id=? AND iteration_id=?",
                        (now, run_id, round_id, iteration["iteration_id"]),
                    )
                    db.execute("UPDATE runs SET status='ready_for_learning',updated_at=? WHERE run_id=?", (now, run_id))
            return self._json(HTTPStatus.OK, {
                "ok": True, "iteration_id": iteration["iteration_id"],
                "arrived": count, "expected": len(actors),
                "ready_for_learning": count == len(actors),
            })

        if method == "POST" and path == "/v1/iterations/advance":
            payload = self._read_json()
            exact_keys(payload, {"run_id", "round_id", "evaluation_id"})
            run_id = identifier(payload["run_id"], "run_id")
            evaluation_id = identifier(payload["evaluation_id"], "evaluation_id")
            round_id = payload["round_id"]
            if not isinstance(round_id, int) or round_id not in CANONICAL_BETA:
                raise ContractError("round_id must be 1..4")
            with self.server.store.transaction() as db:
                run = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
                if mse_protocol.get_protocol(db, run_id, round_id) is not None:
                    raise ConflictError("MSE-selected rounds cannot advance using success-rate selection")
                if run is None or run["current_round"] != round_id:
                    raise NotReadyError("run is not in the requested DAgger round")
                link = db.execute(
                    "SELECT * FROM success_evaluation_iterations WHERE evaluation_id=?",
                    (evaluation_id,),
                ).fetchone()
                evaluation = db.execute(
                    "SELECT * FROM success_evaluations WHERE evaluation_id=?",
                    (evaluation_id,),
                ).fetchone()
                if (
                    link is None or evaluation is None
                    or (link["run_id"], link["round_id"]) != (run_id, round_id)
                ):
                    raise ConflictError("evaluation is not bound to this DAgger round")
                iteration = db.execute(
                    "SELECT * FROM round_iterations WHERE run_id=? AND round_id=? "
                    "AND iteration_id=?",
                    (run_id, round_id, link["iteration_id"]),
                ).fetchone()
                if iteration is None:
                    raise NotReadyError("evaluation iteration does not exist")
                if iteration["evaluation_id"] == evaluation_id and iteration["status"] in {
                    "complete", "advanced"
                }:
                    state = _validation_plateau(list(db.execute(
                        "SELECT * FROM round_iterations WHERE run_id=? AND round_id=? "
                        "AND validation_goal_reached_rate IS NOT NULL ORDER BY iteration_id",
                        (run_id, round_id),
                    )))
                    state = _apply_final_budget(
                        state,
                        _final_budget_reached(
                            db, run_id, round_id, iteration["iteration_id"]
                        ),
                    )
                    return self._json(HTTPStatus.OK, {
                        "ok": True, "idempotent": True,
                        "run_id": run_id, "round_id": round_id,
                        "completed_iteration_id": iteration["iteration_id"],
                        "next_iteration_id": (
                            iteration["iteration_id"] + 1
                            if iteration["status"] == "advanced" else None
                        ),
                        **state,
                    })
                current = _current_iteration(db, run_id, round_id)
                if current is None or current["iteration_id"] != iteration["iteration_id"]:
                    raise ConflictError("evaluation is not bound to the current iteration")
                if evaluation["status"] != "complete":
                    raise NotReadyError("fixed-panel success validation is incomplete")
                if evaluation["checkpoint_sha256"] != iteration["candidate_checkpoint_sha256"]:
                    raise ConflictError("validation did not evaluate the iteration candidate")
                completed = db.execute(
                    "SELECT COUNT(*) FROM success_evaluation_outcomes WHERE evaluation_id=?",
                    (evaluation_id,),
                ).fetchone()[0]
                total = len(json.loads(evaluation["scenario_ids_json"]))
                if completed != total:
                    raise NotReadyError("fixed-panel success validation outcomes are incomplete")
                successes = db.execute(
                    "SELECT COALESCE(SUM(goal_reached),0) FROM success_evaluation_outcomes "
                    "WHERE evaluation_id=?", (evaluation_id,),
                ).fetchone()[0]
                rate = float(successes) / completed
                now = _now()
                db.execute(
                    "UPDATE round_iterations SET evaluation_id=?,"
                    "validation_goal_reached_rate=?,status='validated',updated_at=? "
                    "WHERE run_id=? AND round_id=? AND iteration_id=?",
                    (evaluation_id, rate, now, run_id, round_id, iteration["iteration_id"]),
                )
                rows = list(db.execute(
                    "SELECT * FROM round_iterations WHERE run_id=? AND round_id=? "
                    "AND validation_goal_reached_rate IS NOT NULL ORDER BY iteration_id",
                    (run_id, round_id),
                ))
                state = _apply_final_budget(
                    _validation_plateau(rows),
                    _final_budget_reached(
                        db, run_id, round_id, iteration["iteration_id"]
                    ),
                )
                if state["saturated"]:
                    best = state["best_checkpoint_sha256"]
                    db.execute(
                        "UPDATE round_iterations SET status='complete',updated_at=? "
                        "WHERE run_id=? AND round_id=? AND iteration_id=?",
                        (now, run_id, round_id, iteration["iteration_id"]),
                    )
                    db.execute(
                        "INSERT INTO promotions VALUES (?,?,?,?) "
                        "ON CONFLICT(run_id) DO UPDATE SET round_id=excluded.round_id,"
                        "checkpoint_sha256=excluded.checkpoint_sha256,updated_at=excluded.updated_at",
                        (run_id, round_id, best, now),
                    )
                    db.execute(
                        "UPDATE runs SET active_checkpoint_sha256=?,status='round_complete',"
                        "updated_at=? WHERE run_id=?", (best, now, run_id),
                    )
                    next_iteration = None
                else:
                    candidate = iteration["candidate_checkpoint_sha256"]
                    next_iteration = iteration["iteration_id"] + 1
                    db.execute(
                        "UPDATE round_iterations SET status='advanced',updated_at=? "
                        "WHERE run_id=? AND round_id=? AND iteration_id=?",
                        (now, run_id, round_id, iteration["iteration_id"]),
                    )
                    db.execute(
                        "INSERT INTO round_iterations VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (run_id, round_id, next_iteration, candidate, None, None,
                         None, "collecting", now, now),
                    )
                    db.execute(
                        "UPDATE runs SET active_checkpoint_sha256=?,status='collecting',"
                        "updated_at=? WHERE run_id=?", (candidate, now, run_id),
                    )
            return self._json(HTTPStatus.OK, {
                "ok": True, "run_id": run_id, "round_id": round_id,
                "completed_iteration_id": iteration["iteration_id"],
                "next_iteration_id": next_iteration,
                **state,
            })

        if method == "GET" and path.startswith("/v1/status/"):
            run_id = identifier(path.removeprefix("/v1/status/"), "run_id")
            with self.server.store.connection() as db:
                run = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
                if run is None:
                    raise NotFoundError(run_id)
                iterations = [dict(row) for row in db.execute(
                    "SELECT * FROM round_iterations WHERE run_id=? ORDER BY round_id,iteration_id",
                    (run_id,),
                )]
                barriers = [dict(row) for row in db.execute(
                    "SELECT * FROM iteration_barriers WHERE run_id=? "
                    "ORDER BY round_id,iteration_id,actor_id", (run_id,),
                )]
            result = dict(run)
            result["actor_ids"] = json.loads(result.pop("actor_ids_json"))
            for barrier in barriers:
                barrier["chunk_ids"] = json.loads(barrier.pop("chunk_ids_json"))
            return self._json(HTTPStatus.OK, {
                "ok": True, "run": result, "iterations": iterations,
                "barriers": barriers,
            })
        raise NotFoundError(path)

    # -------------------------------------------------------------- evaluation
    def _evaluation(self, method: str, path: str) -> None:
        if method == "POST" and path == "/v2/evaluations":
            payload = self._read_json()
            exact_keys(payload, {
                "evaluation_id", "run_id", "round_id", "purpose",
                "checkpoint_sha256", "scenario_manifest_sha256", "scenario_ids",
                "observation_mode", "metric",
            })
            evaluation_id = identifier(payload["evaluation_id"], "evaluation_id")
            run_id = identifier(payload["run_id"], "run_id")
            round_id = payload["round_id"]
            if isinstance(round_id, bool) or not isinstance(round_id, int) or round_id not in range(5):
                raise ContractError("round_id must be 0..4")
            purpose = payload["purpose"]
            if purpose not in {"validation", "test"}:
                raise ContractError("purpose must be validation or test")
            observation_mode = payload["observation_mode"]
            if observation_mode not in {"raytraced", "generated"}:
                raise ContractError("observation_mode must be raytraced or generated")
            if payload["metric"] != "raven_goal_reached_rate":
                raise ContractError("metric must be raven_goal_reached_rate")
            checkpoint = sha256_digest(payload["checkpoint_sha256"], "checkpoint_sha256")
            manifest = sha256_digest(
                payload["scenario_manifest_sha256"], "scenario_manifest_sha256"
            )
            scenarios = payload["scenario_ids"]
            if not isinstance(scenarios, list) or not scenarios:
                raise ContractError("scenario_ids must be a non-empty list")
            scenarios = [identifier(item, "scenario_id") for item in scenarios]
            if len(scenarios) != len(set(scenarios)):
                raise ContractError("scenario_ids must be unique")
            request_json = canonical_json(payload)
            now = _now()
            with self.server.store.transaction() as db:
                try:
                    mse_protocol.admit_final_test(db, payload)
                except ValueError as exc:
                    raise ContractError(str(exc)) from exc
                old = db.execute(
                    "SELECT * FROM success_evaluations WHERE evaluation_id=?",
                    (evaluation_id,),
                ).fetchone()
                if old:
                    same = (
                        old["run_id"] == run_id
                        and old["round_id"] == round_id
                        and old["purpose"] == purpose
                        and old["checkpoint_sha256"] == checkpoint
                        and old["scenario_manifest_sha256"] == manifest
                        and old["scenario_ids_json"] == canonical_json(scenarios)
                        and old["observation_mode"] == observation_mode
                        and old["metric"] == payload["metric"]
                    )
                    if not same:
                        raise ConflictError("evaluation_id already has different immutable inputs")
                    return self._json(HTTPStatus.OK, {
                        "ok": True, "evaluation_id": evaluation_id,
                        "status": old["status"], "idempotent": True,
                    })
                registered = db.execute(
                    "SELECT path FROM checkpoints WHERE sha256=?", (checkpoint,)
                ).fetchone()
                if registered is None:
                    raise NotReadyError("evaluation checkpoint is not registered")
                checkpoint_path = Path(registered["path"])
                if not checkpoint_path.is_file() or file_sha256(checkpoint_path) != checkpoint:
                    raise NotReadyError("registered evaluation checkpoint bytes are unavailable")
                iteration = None
                if purpose == "validation" and round_id > 0:
                    run = db.execute(
                        "SELECT * FROM runs WHERE run_id=?", (run_id,)
                    ).fetchone()
                    if (
                        run is None or run["current_round"] != round_id
                        or run["status"] != "awaiting_validation"
                    ):
                        raise NotReadyError(
                            "run is not awaiting validation for this DAgger round"
                        )
                    iteration = _current_iteration(db, run_id, round_id)
                    if (
                        iteration is None
                        or iteration["status"] != "awaiting_validation"
                        or iteration["candidate_checkpoint_sha256"] != checkpoint
                    ):
                        raise ConflictError(
                            "validation checkpoint is not the current learner candidate"
                        )
                db.execute(
                    "INSERT INTO success_evaluations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        evaluation_id, run_id, round_id, purpose, checkpoint, manifest,
                        canonical_json(scenarios), observation_mode, payload["metric"],
                        "collecting", now, now,
                    ),
                )
                if iteration is not None:
                    db.execute(
                        "INSERT INTO success_evaluation_iterations VALUES (?,?,?,?)",
                        (evaluation_id, run_id, round_id, iteration["iteration_id"]),
                    )
                    db.execute(
                        "UPDATE round_iterations SET evaluation_id=?,status='validating',"
                        "updated_at=? WHERE run_id=? AND round_id=? AND iteration_id=?",
                        (evaluation_id, now, run_id, round_id, iteration["iteration_id"]),
                    )
                    db.execute(
                        "UPDATE runs SET status='validating',updated_at=? WHERE run_id=?",
                        (now, run_id),
                    )
            return self._json(HTTPStatus.CREATED, {
                "ok": True, "evaluation_id": evaluation_id,
                "status": "collecting", "scenario_count": len(scenarios),
            })

        v2_prefix = "/v2/evaluations/"
        if path.startswith(v2_prefix):
            suffix = path.removeprefix(v2_prefix)
            if method == "POST" and suffix.endswith("/outcomes"):
                evaluation_id = identifier(suffix.removesuffix("/outcomes"), "evaluation_id")
                payload = self._read_json()
                exact_keys(payload, {
                    "idempotency_key", "scenario_id", "episode_id", "goal_reached",
                    "reward", "steps", "termination", "safety_violation",
                    "checkpoint_sha256", "scenario_manifest_sha256",
                })
                idempotency_key = identifier(payload["idempotency_key"], "idempotency_key")
                scenario_id = identifier(payload["scenario_id"], "scenario_id")
                identifier(payload["episode_id"], "episode_id")
                if not isinstance(payload["goal_reached"], bool):
                    raise ContractError("goal_reached must be boolean")
                if not isinstance(payload["safety_violation"], bool):
                    raise ContractError("safety_violation must be boolean")
                reward = payload["reward"]
                if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(reward):
                    raise ContractError("reward must be finite")
                steps = payload["steps"]
                if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
                    raise ContractError("steps must be a positive integer")
                termination = payload["termination"]
                if termination not in {"success", "failure", "timeout", "safety_violation"}:
                    raise ContractError("termination is invalid")
                checkpoint = sha256_digest(payload["checkpoint_sha256"], "checkpoint_sha256")
                manifest = sha256_digest(
                    payload["scenario_manifest_sha256"], "scenario_manifest_sha256"
                )
                request_json = canonical_json(payload)
                with self.server.store.transaction() as db:
                    evaluation = db.execute(
                        "SELECT * FROM success_evaluations WHERE evaluation_id=?",
                        (evaluation_id,),
                    ).fetchone()
                    if evaluation is None:
                        raise NotFoundError(evaluation_id)
                    existing = db.execute(
                        "SELECT request_json FROM success_evaluation_outcomes WHERE idempotency_key=?",
                        (idempotency_key,),
                    ).fetchone()
                    if existing:
                        if existing["request_json"] != request_json:
                            raise ConflictError("outcome idempotency key was reused with different data")
                        return self._json(HTTPStatus.OK, {
                            "ok": True, "evaluation_id": evaluation_id,
                            "scenario_id": scenario_id, "status": evaluation["status"],
                            "idempotent": True,
                        })
                    if evaluation["status"] == "complete":
                        raise ConflictError("evaluation is already complete")
                    if (
                        checkpoint != evaluation["checkpoint_sha256"]
                        or manifest != evaluation["scenario_manifest_sha256"]
                    ):
                        raise ConflictError("outcome checkpoint/scenario manifest identity mismatch")
                    scenarios = json.loads(evaluation["scenario_ids_json"])
                    if scenario_id not in scenarios:
                        raise ConflictError("scenario is not part of the sealed evaluation panel")
                    prior = db.execute(
                        "SELECT idempotency_key FROM success_evaluation_outcomes "
                        "WHERE evaluation_id=? AND scenario_id=?",
                        (evaluation_id, scenario_id),
                    ).fetchone()
                    if prior:
                        raise ConflictError("scenario already has an outcome under another key")
                    db.execute(
                        "INSERT INTO success_evaluation_outcomes VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (
                            evaluation_id, scenario_id, idempotency_key, request_json,
                            int(payload["goal_reached"]), float(reward), steps, termination,
                            int(payload["safety_violation"]), _now(),
                        ),
                    )
                    count = db.execute(
                        "SELECT COUNT(*) FROM success_evaluation_outcomes WHERE evaluation_id=?",
                        (evaluation_id,),
                    ).fetchone()[0]
                    status = "complete" if count == len(scenarios) else "collecting"
                    db.execute(
                        "UPDATE success_evaluations SET status=?,updated_at=? WHERE evaluation_id=?",
                        (status, _now(), evaluation_id),
                    )
                return self._json(HTTPStatus.CREATED, {
                    "ok": True, "evaluation_id": evaluation_id,
                    "scenario_id": scenario_id, "status": status,
                    "completed": count, "total": len(scenarios),
                })
            if method == "GET" and "/" not in suffix:
                evaluation_id = identifier(suffix, "evaluation_id")
                with self.server.store.connection() as db:
                    evaluation = db.execute(
                        "SELECT * FROM success_evaluations WHERE evaluation_id=?",
                        (evaluation_id,),
                    ).fetchone()
                    if evaluation is None:
                        raise NotFoundError(evaluation_id)
                    outcomes = list(db.execute(
                        "SELECT scenario_id,goal_reached,reward,steps,termination,safety_violation "
                        "FROM success_evaluation_outcomes WHERE evaluation_id=? ORDER BY scenario_id",
                        (evaluation_id,),
                    ))
                total = len(json.loads(evaluation["scenario_ids_json"]))
                completed = len(outcomes)
                successes = sum(row["goal_reached"] for row in outcomes)
                rate = successes / completed if completed else None
                interval = None
                if completed:
                    z = 1.959963984540054
                    denominator = 1.0 + z * z / completed
                    center = (rate + z * z / (2 * completed)) / denominator
                    half = z * math.sqrt(
                        rate * (1 - rate) / completed + z * z / (4 * completed * completed)
                    ) / denominator
                    interval = [max(0.0, center - half), min(1.0, center + half)]
                return self._json(HTTPStatus.OK, {
                    "ok": True,
                    "evaluation": {
                        "evaluation_id": evaluation_id,
                        "run_id": evaluation["run_id"],
                        "round_id": evaluation["round_id"],
                        "purpose": evaluation["purpose"],
                        "checkpoint_sha256": evaluation["checkpoint_sha256"],
                        "scenario_manifest_sha256": evaluation["scenario_manifest_sha256"],
                        "observation_mode": evaluation["observation_mode"],
                        "metric": evaluation["metric"],
                        "status": evaluation["status"],
                        "completed": completed,
                        "total": total,
                        "successes": successes,
                        "failures": completed - successes,
                        "goal_reached_rate": rate,
                        "wilson_95_interval": interval,
                        "safety_violations": sum(row["safety_violation"] for row in outcomes),
                        "mean_reward": (
                            sum(row["reward"] for row in outcomes) / completed if completed else None
                        ),
                        "mean_steps": (
                            sum(row["steps"] for row in outcomes) / completed if completed else None
                        ),
                        "outcomes": [dict(row) for row in outcomes],
                    },
                })
            raise NotFoundError(path)

        raise NotFoundError(path)

    # ---------------------------------------------------------------- learner
    def _learner(self, method: str, path: str) -> None:
        if method == "GET" and path == "/v1/status":
            with self.server.store.connection() as db:
                counts = {row["status"]: row["count"] for row in db.execute(
                    "SELECT status,COUNT(*) AS count FROM learner_jobs GROUP BY status"
                )}
            return self._json(HTTPStatus.OK, {"ok": True, "jobs": counts})
        if method == "GET" and path.startswith("/v1/jobs/"):
            job_id = identifier(path.removeprefix("/v1/jobs/"), "job_id")
            with self.server.store.connection() as db:
                row = db.execute("SELECT * FROM learner_jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                raise NotFoundError(job_id)
            result = dict(row)
            result["chunk_ids"] = json.loads(result.pop("chunk_ids_json"))
            result["request"] = json.loads(result.pop("request_json"))
            result["result"] = json.loads(result.pop("result_json")) if result["result_json"] else None
            return self._json(HTTPStatus.OK, {"ok": True, "job": result})
        if method != "POST" or path != "/v1/jobs":
            raise NotFoundError(path)
        if self.server.worker is None or not self.server.worker.is_file():
            raise NotReadyError("learner worker is not configured")
        payload = self._read_json()
        exact_keys(payload, {
            "job_id", "run_id", "round_id", "input_checkpoint_sha256",
            "train_chunk_ids", "parameters",
        })
        job_id = identifier(payload["job_id"], "job_id")
        run_id = identifier(payload["run_id"], "run_id")
        digest = sha256_digest(payload["input_checkpoint_sha256"], "input_checkpoint_sha256")
        round_id = payload["round_id"]
        if not isinstance(round_id, int) or round_id not in CANONICAL_BETA:
            raise ContractError("round_id must be 1..4")
        values = payload["train_chunk_ids"]
        if not isinstance(values, list) or not values:
            raise ContractError("train_chunk_ids must be a non-empty list")
        chunks = sorted(identifier(value, "chunk_id") for value in values)
        if len(chunks) != len(set(chunks)):
            raise ContractError("train_chunk_ids must be unique")
        if not isinstance(payload["parameters"], dict):
            raise ContractError("parameters must be an object")
        from . import learner_order
        try:
            learner_order.enabled(payload)
        except ValueError as exc:
            raise ContractError(str(exc)) from exc
        from .bc32_diagnostic_parent import PARAMETER as diagnostic_parameter, validate_admission
        diagnostic_admission = payload['parameters'].get(diagnostic_parameter)
        if diagnostic_parameter in payload['parameters']:
            try:
                validate_admission(diagnostic_admission, payload)
            except (ValueError, KeyError, TypeError) as exc:
                raise ContractError(str(exc)) from exc
        from . import diagnostic_reentry
        if diagnostic_reentry.PARAMETER in payload['parameters']:
            try:
                chain = diagnostic_reentry.validate_admission(
                    payload['parameters'][diagnostic_reentry.PARAMETER], payload)
                worker = self.server.worker
                if worker.name != 'run_raven_learner_worker_h200.sh':
                    raise ValueError('learner overlay requires the fixed source-bound launcher')
                diagnostic_reentry.validate_runtime_files(chain,
                    worker_file=worker.parent / 'services/raven_learner_worker.py',
                    server_file=Path(__file__).resolve())
            except (ValueError, KeyError, TypeError, OSError) as exc:
                raise ContractError(str(exc)) from exc
        with self.server.store.transaction() as db:
            try:
                validation_collection.forbid_learning(db, run_id, chunks)
            except ValueError as exc:
                raise ContractError(str(exc)) from exc
            old = db.execute("SELECT request_json,status FROM learner_jobs WHERE job_id=?", (job_id,)).fetchone()
            request_json = canonical_json(payload)
            if old:
                if old["request_json"] != request_json:
                    raise ConflictError("job_id already exists with a different request")
                return self._json(HTTPStatus.OK, {"ok": True, "job_id": job_id, "status": old["status"], "idempotent": True})
            run = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if run is None or run["status"] != "ready_for_learning" or run["current_round"] != round_id:
                raise NotReadyError("run has not reached its learner barrier")
            if diagnostic_admission is not None:
                scope = diagnostic_admission['scope']
                if (scope['config_sha256'] != run['config_sha256']
                        or scope['parent_checkpoint_sha256'] != run['initial_checkpoint_sha256']):
                    raise ConflictError('diagnostic scope differs from the durable registered run')
            if run["active_checkpoint_sha256"] != digest:
                raise ConflictError("learner input differs from active checkpoint")
            iteration = _current_iteration(db, run_id, round_id)
            if iteration is None or iteration["status"] != "ready_for_learning":
                raise NotReadyError("current iteration has not reached its learner barrier")
            if iteration["input_checkpoint_sha256"] != digest:
                raise ConflictError("learner input differs from iteration checkpoint")
            if payload["parameters"].get("iteration_id") != iteration["iteration_id"]:
                raise ConflictError("learner iteration_id differs from the current iteration")
            protocol = mse_protocol.get_protocol(db, run_id, round_id)
            if protocol is not None:
                config = json.loads(protocol["protocol_json"])
                if (protocol["status"] != "active" or protocol["selection_json"] is None
                        or iteration["iteration_id"] > config["max_iterations"]
                        or payload["parameters"].get("replay_manifest") is not None):
                    raise ConflictError("MSE learner budget/baseline/replay contract violation")
                from .fixed100_protocol import is_fixed100
                if is_fixed100(config):
                    expected = dict(metric_protocol='fixed100_final_success',
                        microbatch_environments=32,raven_exact_full_batch=True,num_mini_batches=1,
                        final_validation_iteration=100,validation_interval_iterations=100,
                        learning_rate=.0002,max_grad_norm=1.,deep_validation_samples=0)
                    if len(chunks)!=32 or any(payload['parameters'].get(k)!=v for k,v in expected.items()):
                        raise ConflictError('fixed100 requires one full32x81 update and no validation')
                try:
                    provenance = mse_protocol.check_training_admission(db, run_id, round_id, chunks)
                    db.execute("UPDATE mse_round_protocols SET training_provenance_json=? WHERE run_id=? AND round_id=?",
                               (canonical_json(provenance), run_id, round_id))
                except ValueError as exc:
                    raise ContractError(str(exc)) from exc
            admitted = []
            lane_chunks = {}
            for row in db.execute(
                "SELECT actor_id,chunk_ids_json FROM iteration_barriers WHERE run_id=? "
                "AND round_id=? AND iteration_id=?",
                (run_id, round_id, iteration["iteration_id"]),
            ):
                values = json.loads(row['chunk_ids_json'])
                admitted.extend(values)
                lane_chunks[row['actor_id']] = values
            if learner_order.enabled(payload):
                # Check the submitted ENVIRONMENT order, not the chunk hash
                # order used above solely for set membership/idempotency.
                expected_order = [chunk for actor in sorted(lane_chunks)
                                  for chunk in lane_chunks[actor]]
                if (any(len(values) != 1 for values in lane_chunks.values())
                        or payload['train_chunk_ids'] != expected_order):
                    raise ConflictError('Raven learner chunks must follow registered actor/lane order')
            replay_manifest = payload["parameters"].get("replay_manifest")
            if replay_manifest is None:
                if chunks != sorted(admitted):
                    raise ConflictError(
                        "learner train chunks must equal all barrier-admitted "
                        "current-iteration chunks"
                    )
            else:
                try:
                    replay_dagger_ids, _ = validate_replay_manifest(
                        replay_manifest, db=db, run_id=run_id, round_id=round_id,
                        iteration_id=iteration["iteration_id"],
                        admitted_current=set(admitted),
                        bc_root=Path(os.environ.get(
                            "DAGGER_BC_REPLAY_ROOT",
                            str(Path(__file__).resolve().parents[3] / "data/BC/rendered"),
                        )),
                    )
                except ValueError as exc:
                    raise ConflictError(str(exc)) from exc
                if chunks != sorted(replay_dagger_ids):
                    raise ConflictError(
                        "learner DAgger chunks differ from the replay manifest"
                    )
                for chunk_id in replay_dagger_ids:
                    row = db.execute(
                        "SELECT status FROM chunk_reservations WHERE chunk_id=?",
                        (chunk_id,),
                    ).fetchone()
                    if row is None or row["status"] != "committed":
                        raise ConflictError(
                            f"replay chunk is not committed: {chunk_id}"
                        )
            job_dir = self.server.root / "jobs" / job_id
            job_dir.mkdir(parents=True, exist_ok=False)
            job_path = job_dir / "request.json"
            result_path = job_dir / "result.json"
            log_path = job_dir / "worker.log"
            job_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            now = _now()
            db.execute("INSERT INTO learner_jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                       (job_id, run_id, round_id, digest, canonical_json(chunks), "queued",
                        request_json, None, str(log_path), None, now, now))
            db.execute(
                "INSERT INTO learner_job_iterations VALUES (?,?,?,?)",
                (job_id, run_id, round_id, iteration["iteration_id"]),
            )
            db.execute(
                "UPDATE round_iterations SET status='learning',updated_at=? "
                "WHERE run_id=? AND round_id=? AND iteration_id=?",
                (now, run_id, round_id, iteration["iteration_id"]),
            )
            db.execute("UPDATE runs SET status='learning',updated_at=? WHERE run_id=?", (now, run_id))

        thread = threading.Thread(
            target=self._run_worker, args=(job_id, job_path, result_path, log_path), daemon=True
        )
        thread.start()
        self._json(HTTPStatus.ACCEPTED, {"ok": True, "job_id": job_id, "status": "queued"})

    def _run_worker(self, job_id: str, job_path: Path, result_path: Path, log_path: Path) -> None:
        with self.server.jobs_lock:
            with log_path.open("ab", buffering=0) as log:
                process = subprocess.Popen(
                    [str(self.server.worker), "--job", str(job_path), "--result", str(result_path),
                     "--state-root", str(self.server.root)],
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                with self.server.store.transaction() as db:
                    db.execute("UPDATE learner_jobs SET status='running',pid=?,updated_at=? WHERE job_id=?",
                               (process.pid, _now(), job_id))
                code = process.wait()
            result = None
            status = "failed"
            if result_path.is_file():
                try:
                    result = json.loads(result_path.read_text())
                except (OSError, json.JSONDecodeError):
                    result = None
            if code == 0 and isinstance(result, dict):
                try:
                    output_sha = sha256_digest(result.get("output_checkpoint_sha256"), "output_checkpoint_sha256")
                    output_path = Path(result.get("output_checkpoint_path", "")).resolve()
                    if not output_path.is_file() or file_sha256(output_path) != output_sha:
                        raise ContractError("worker output checkpoint receipt is invalid")
                    # The render/student service resolves active policies from
                    # this content-addressed directory.  Publish by hard link
                    # before advancing the coordinator; otherwise the next
                    # rollout can observe an activated digest whose bytes are
                    # still only inside the learner job directory.
                    checkpoint_root = self.server.root / "checkpoints"
                    checkpoint_root.mkdir(parents=True, exist_ok=True)
                    checkpoint_blob = checkpoint_root / f"{output_sha}.pt"
                    if not checkpoint_blob.exists():
                        temporary = checkpoint_root / (
                            f".{output_sha}.{os.getpid()}.{threading.get_ident()}.{time.time_ns()}.part"
                        )
                        os.link(output_path, temporary)
                        os.replace(temporary, checkpoint_blob)
                    elif checkpoint_blob.stat().st_size != output_path.stat().st_size:
                        raise ContractError("content-addressed checkpoint size collision")
                    status = "succeeded"
                except (ContractError, OSError):
                    status = "failed"
            with self.server.store.transaction() as db:
                db.execute("UPDATE learner_jobs SET status=?,result_json=?,updated_at=? WHERE job_id=?",
                           (status, canonical_json(result) if result is not None else None, _now(), job_id))
                job = db.execute(
                    "SELECT j.run_id,j.round_id,j.request_json,i.iteration_id FROM learner_jobs j "
                    "JOIN learner_job_iterations i ON i.job_id=j.job_id WHERE j.job_id=?",
                    (job_id,),
                ).fetchone()
                now = _now()
                if status == "succeeded":
                    output_sha = result["output_checkpoint_sha256"]
                    output_path = self.server.root / "checkpoints" / f"{output_sha}.pt"
                    cadence = db.execute(
                        "SELECT validation_interval_iterations FROM runs WHERE run_id=?",
                        (job["run_id"],),
                    ).fetchone()
                    if cadence is None:
                        raise RuntimeError("learner job run disappeared")
                    learner_parameters = json.loads(job["request_json"])["parameters"]
                    final_validation_iteration = learner_parameters.get(
                        "final_validation_iteration"
                    )
                    validation_interval = learner_parameters.get(
                        "validation_interval_iterations",
                        cadence["validation_interval_iterations"],
                    )
                    if type(validation_interval) is not int or validation_interval < 1:
                        raise ContractError("learner validation interval is invalid")
                    validation_due = _validation_due(
                        job["iteration_id"], validation_interval
                    ) or job["iteration_id"] == final_validation_iteration
                    mse_row = mse_protocol.get_protocol(db, job["run_id"], job["round_id"])
                    if mse_row is not None:
                        mse_config = json.loads(mse_row["protocol_json"])
                        validation_due = (_validation_due(job["iteration_id"], mse_config["interval_iterations"])
                                          or job["iteration_id"] == mse_config["max_iterations"])
                    from .fixed100_protocol import is_fixed100, accept_final_candidate
                    fixed100 = mse_row is not None and is_fixed100(mse_config)
                    try:
                        fixed100_final = accept_final_candidate(db, job, result, now, commit=False) if fixed100 else False
                    except ValueError as error:
                        # Invalid full-batch/AdamW evidence is a durable failure,
                        # not an unobserved worker-thread exception.
                        status = 'failed'
                        fixed100_final = False
                        result = {**result, 'admission_error': str(error)[:512]}
                        db.execute("UPDATE learner_jobs SET status='failed',result_json=?,updated_at=? WHERE job_id=?",
                                   (canonical_json(result), now, job_id))
                    if fixed100:
                        validation_due = fixed100_final
                    checkpoint_metadata = canonical_json({
                        "schema": "roborender.dagger_candidate_registration.v1",
                        "run_id": job["run_id"],
                        "round_id": job["round_id"],
                        "iteration_id": job["iteration_id"],
                        "parent_checkpoint_sha256": db.execute(
                            "SELECT input_checkpoint_sha256 FROM learner_jobs WHERE job_id=?",
                            (job_id,),
                        ).fetchone()[0],
                        "learner_job_id": job_id,
                        "selection": (
                            "final_update_100_pending_test" if fixed100_final else
                            "pending_heldout_mse_validation" if mse_row is not None and validation_due else
                            "pending_fixed_panel_success_validation"
                            if validation_due else "next_rollout_intermediate"
                        ),
                    })
                    existing_checkpoint = db.execute(
                        "SELECT metadata_json FROM checkpoints WHERE sha256=?", (output_sha,)
                    ).fetchone()
                    if (
                        existing_checkpoint is not None
                        and existing_checkpoint["metadata_json"] != checkpoint_metadata
                    ):
                        status = "failed"
                        if fixed100:
                            result = {**result, 'admission_error': 'checkpoint registration metadata collision'}
                        db.execute(
                            "UPDATE learner_jobs SET status='failed',result_json=?,updated_at=? WHERE job_id=?",
                            (canonical_json(result), now, job_id),
                        )
                    elif status == 'succeeded':
                        db.execute(
                            "INSERT OR IGNORE INTO checkpoints VALUES (?,?,?,?,?)",
                            (
                                output_sha, output_path.stat().st_size,
                                checkpoint_metadata, str(output_path), now,
                            ),
                        )
                if status == "succeeded":
                    output_sha = result["output_checkpoint_sha256"]
                    if fixed100_final:
                        # Publish final selection only after every learner and
                        # checkpoint-registry admission passed, atomically with
                        # the final iteration transition below.
                        accept_final_candidate(db, job, result, now)
                    if validation_due:
                        validation_status = ('selected_pending_test' if fixed100_final else
                            "awaiting_mse_validation" if mse_row is not None else "awaiting_validation")
                        db.execute(
                            "UPDATE round_iterations SET candidate_checkpoint_sha256=?,"
                            "status=?,updated_at=? WHERE run_id=? "
                            "AND round_id=? AND iteration_id=?",
                            (output_sha, validation_status, now, job["run_id"], job["round_id"],
                             job["iteration_id"]),
                        )
                        run_status = validation_status
                    else:
                        next_iteration = job["iteration_id"] + 1
                        db.execute(
                            "UPDATE round_iterations SET candidate_checkpoint_sha256=?,"
                            "status='advanced',updated_at=? WHERE run_id=? AND round_id=? "
                            "AND iteration_id=?",
                            (
                                output_sha, now, job["run_id"], job["round_id"],
                                job["iteration_id"],
                            ),
                        )
                        db.execute(
                            "INSERT INTO round_iterations VALUES (?,?,?,?,?,?,?,?,?,?)",
                            (
                                job["run_id"], job["round_id"], next_iteration,
                                output_sha, None, None, None, "collecting", now, now,
                            ),
                        )
                        db.execute(
                            "UPDATE runs SET active_checkpoint_sha256=?,updated_at=? "
                            "WHERE run_id=?",
                            (output_sha, now, job["run_id"]),
                        )
                        run_status = "collecting"
                else:
                    db.execute(
                        "UPDATE round_iterations SET status='learner_failed',updated_at=? "
                        "WHERE run_id=? AND round_id=? AND iteration_id=?",
                        (now, job["run_id"], job["round_id"], job["iteration_id"]),
                    )
                    run_status = "learner_failed"
                db.execute(
                    "UPDATE runs SET status=?,updated_at=? WHERE run_id=?",
                    (run_status, now, job["run_id"]),
                )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--role",
        choices=("coordinator", "trajectory", "registry", "learner", "evaluation"),
        required=True,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--state-root", type=Path, default=root / "runtime/dagger_services")
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--max-upload-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--worker", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = args.token_file.read_text().strip()
    if len(token) < 32:
        raise SystemExit("token file must contain at least 32 non-whitespace characters")
    if args.role == "learner" and args.worker is None:
        raise SystemExit("learner role requires --worker")
    server = DaggerHTTPServer(
        (args.host, args.port), Handler, role=args.role, root=args.state_root,
        token=token, max_upload_bytes=args.max_upload_bytes, worker=args.worker,
    )
    print(json.dumps({"event": "ready", "role": args.role, "host": args.host,
                      "port": server.server_port, "schema": SERVICE_SCHEMA}), flush=True)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
