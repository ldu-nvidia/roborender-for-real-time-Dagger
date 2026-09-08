#!/usr/bin/env python3
"""Independently launchable H200 coordinator, trajectory, registry and learner roles."""

from __future__ import annotations

import argparse
import hmac
import json
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


class ConflictError(RuntimeError):
    pass


class NotFoundError(RuntimeError):
    pass


class NotReadyError(RuntimeError):
    pass


def _now() -> float:
    return time.time()


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
        if (
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
            payload = self._read_json()
            exact_keys(payload, {"run_id", "round_id", "checkpoint_sha256"})
            run_id = identifier(payload["run_id"], "run_id")
            digest = sha256_digest(payload["checkpoint_sha256"], "checkpoint_sha256")
            round_id = payload["round_id"]
            if not isinstance(round_id, int) or round_id not in CANONICAL_BETA:
                raise ContractError("round_id must be 1..4")
            with self.server.store.transaction() as db:
                checkpoint = db.execute("SELECT 1 FROM checkpoints WHERE sha256=?", (digest,)).fetchone()
                run = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
                if checkpoint is None:
                    raise NotReadyError("checkpoint is not registered")
                if run is None or run["current_round"] != round_id or run["status"] != "ready_for_promotion":
                    raise NotReadyError("coordinator has not approved this round for promotion")
                jobs = db.execute(
                    "SELECT result_json FROM learner_jobs "
                    "WHERE run_id=? AND round_id=? AND status='succeeded'",
                    (run_id, round_id),
                )
                produced = {
                    json.loads(row["result_json"])["output_checkpoint_sha256"]
                    for row in jobs
                    if row["result_json"] is not None
                }
                if digest not in produced:
                    raise NotReadyError(
                        "checkpoint was not produced by this round's successful learner job"
                    )
                now = _now()
                db.execute(
                    "INSERT INTO promotions VALUES (?,?,?,?) ON CONFLICT(run_id) DO UPDATE SET round_id=excluded.round_id,checkpoint_sha256=excluded.checkpoint_sha256,updated_at=excluded.updated_at",
                    (run_id, round_id, digest, now),
                )
                db.execute(
                    "UPDATE runs SET active_checkpoint_sha256=?,status='round_complete',updated_at=? WHERE run_id=?",
                    (digest, now, run_id),
                )
            return self._json(HTTPStatus.OK, {"ok": True, "run_id": run_id, "round_id": round_id, "checkpoint_sha256": digest})

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
        if method == "POST" and path == "/v1/runs":
            payload = self._read_json()
            exact_keys(payload, {"run_id", "actor_ids", "rollout_steps", "config_sha256", "initial_checkpoint_sha256"})
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
            now = _now()
            actor_json = canonical_json(sorted(actors))
            with self.server.store.transaction() as db:
                if db.execute("SELECT 1 FROM checkpoints WHERE sha256=?", (initial,)).fetchone() is None:
                    raise NotReadyError("initial checkpoint is not registered")
                old = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
                if old:
                    same = old["actor_ids_json"] == actor_json and old["config_sha256"] == config_sha and old["active_checkpoint_sha256"] == initial
                    if not same:
                        raise ConflictError("run_id already exists with a different contract")
                    return self._json(HTTPStatus.OK, {"ok": True, "run_id": run_id, "idempotent": True})
                db.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?)",
                           (run_id, actor_json, ROLLOUT_STEPS, config_sha, initial, 0, None, "created", now, now))
            return self._json(HTTPStatus.CREATED, {"ok": True, "run_id": run_id, "status": "created"})

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
                run = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
                if run is None:
                    raise NotFoundError(run_id)
                if run["current_round"] == round_id and run["status"] == "collecting":
                    if run["active_checkpoint_sha256"] != digest or run["beta"] != beta:
                        raise ConflictError("active round differs")
                    return self._json(HTTPStatus.OK, {"ok": True, "idempotent": True})
                if round_id != run["current_round"] + 1:
                    raise NotReadyError("rounds must start sequentially")
                if run["status"] not in ("created", "round_complete"):
                    raise NotReadyError(f"run is {run['status']}")
                if digest != run["active_checkpoint_sha256"]:
                    raise ConflictError("actor checkpoint differs from promoted checkpoint")
                db.execute("UPDATE runs SET current_round=?,beta=?,status='collecting',updated_at=? WHERE run_id=?",
                           (round_id, beta, _now(), run_id))
            return self._json(HTTPStatus.OK, {"ok": True, "run_id": run_id, "round_id": round_id, "beta": beta})

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
                run = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
                if run is None or run["status"] != "collecting" or run["current_round"] != round_id:
                    raise NotReadyError("run is not collecting this round")
                actors = json.loads(run["actor_ids_json"])
                if actor_id not in actors:
                    raise ContractError("actor_id is not admitted to this run")
                if digest != run["active_checkpoint_sha256"]:
                    raise ConflictError("actor used a different checkpoint")
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
                old = db.execute("SELECT * FROM barriers WHERE run_id=? AND round_id=? AND actor_id=?",
                                 (run_id, round_id, actor_id)).fetchone()
                if old and (old["checkpoint_sha256"] != digest or old["chunk_ids_json"] != chunk_json):
                    raise ConflictError("actor barrier already contains different chunks")
                db.execute("INSERT OR IGNORE INTO barriers VALUES (?,?,?,?,?,?)",
                           (run_id, round_id, actor_id, digest, chunk_json, _now()))
                count = db.execute("SELECT COUNT(*) FROM barriers WHERE run_id=? AND round_id=?", (run_id, round_id)).fetchone()[0]
                if count == len(actors):
                    db.execute("UPDATE runs SET status='ready_for_learning',updated_at=? WHERE run_id=?", (_now(), run_id))
            return self._json(HTTPStatus.OK, {"ok": True, "arrived": count, "expected": len(actors), "ready_for_learning": count == len(actors)})

        if method == "GET" and path.startswith("/v1/status/"):
            run_id = identifier(path.removeprefix("/v1/status/"), "run_id")
            with self.server.store.connection() as db:
                run = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
                if run is None:
                    raise NotFoundError(run_id)
                barriers = [dict(row) for row in db.execute("SELECT * FROM barriers WHERE run_id=? ORDER BY round_id,actor_id", (run_id,))]
            result = dict(run)
            result["actor_ids"] = json.loads(result.pop("actor_ids_json"))
            for barrier in barriers:
                barrier["chunk_ids"] = json.loads(barrier.pop("chunk_ids_json"))
            return self._json(HTTPStatus.OK, {"ok": True, "run": result, "barriers": barriers})
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
        exact_keys(payload, {"job_id", "run_id", "round_id", "input_checkpoint_sha256", "chunk_ids", "parameters"})
        job_id = identifier(payload["job_id"], "job_id")
        run_id = identifier(payload["run_id"], "run_id")
        digest = sha256_digest(payload["input_checkpoint_sha256"], "input_checkpoint_sha256")
        round_id = payload["round_id"]
        chunks = payload["chunk_ids"]
        if not isinstance(round_id, int) or round_id not in CANONICAL_BETA:
            raise ContractError("round_id must be 1..4")
        if not isinstance(chunks, list) or not chunks:
            raise ContractError("chunk_ids must be a non-empty list")
        chunks = sorted(identifier(value, "chunk_id") for value in chunks)
        if len(chunks) != len(set(chunks)):
            raise ContractError("chunk_ids must be unique")
        if not isinstance(payload["parameters"], dict):
            raise ContractError("parameters must be an object")
        with self.server.store.transaction() as db:
            old = db.execute("SELECT request_json,status FROM learner_jobs WHERE job_id=?", (job_id,)).fetchone()
            request_json = canonical_json(payload)
            if old:
                if old["request_json"] != request_json:
                    raise ConflictError("job_id already exists with a different request")
                return self._json(HTTPStatus.OK, {"ok": True, "job_id": job_id, "status": old["status"], "idempotent": True})
            run = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if run is None or run["status"] != "ready_for_learning" or run["current_round"] != round_id:
                raise NotReadyError("run has not reached its learner barrier")
            if run["active_checkpoint_sha256"] != digest:
                raise ConflictError("learner input differs from active checkpoint")
            admitted = []
            for row in db.execute("SELECT chunk_ids_json FROM barriers WHERE run_id=? AND round_id=?", (run_id, round_id)):
                admitted.extend(json.loads(row[0]))
            if chunks != sorted(admitted):
                raise ConflictError("learner job must consume exactly the barrier-admitted chunks")
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
                    status = "succeeded"
                except (ContractError, OSError):
                    status = "failed"
            with self.server.store.transaction() as db:
                db.execute("UPDATE learner_jobs SET status=?,result_json=?,updated_at=? WHERE job_id=?",
                           (status, canonical_json(result) if result is not None else None, _now(), job_id))
                job = db.execute("SELECT run_id FROM learner_jobs WHERE job_id=?", (job_id,)).fetchone()
                db.execute("UPDATE runs SET status=?,updated_at=? WHERE run_id=?",
                           ("ready_for_promotion" if status == "succeeded" else "learner_failed", _now(), job["run_id"]))


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("coordinator", "trajectory", "registry", "learner"), required=True)
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
