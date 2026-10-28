#!/usr/bin/env python3
"""Authenticated H200 prepare/commit service for live distributed DAgger."""

from __future__ import annotations

import argparse
import hmac
import io
import json
import os
import sys
import tempfile
import threading
import time
import traceback
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

from .client import ServiceClient
from .batch_contract import (
    BATCH_CONTENT_TYPE, BATCH_CONTRACT, anchor_arrays_digest, decode_binary_batch,
    parse_prepare_batch, STREAM_CONTENT_TYPE, STREAM_CONTRACT, decode_stream_batch,
)
from .contracts import (
    ChunkReservation, ContractError, VIEWS, canonical_json, file_sha256, sha256_digest,
)
from .render_student import (
    FakeRenderStudentBackend, LiveRenderStudentBackend, RenderStudentBackend,
    RenderStudentBackendPool, content_addressed_checkpoint,
)
from .phase_telemetry import PhaseTelemetry
from .step_contract import (
    STEP_CONTRACT, StepKey, bytes_sha256, canonical_bytes, encode_array, parse_prepare,
    response_receipt, validate_sealed_window, validate_transition,
)
from .store import StateStore


class ConflictError(RuntimeError):
    pass


class NotReadyError(RuntimeError):
    pass


def _now() -> float:
    return time.time()


def load_prepared_artifact(reference: str) -> dict[str, np.ndarray]:
    """Load either a legacy per-lane artifact or one member of a v2 batch."""

    prefix = ""
    path = reference
    if reference.startswith("{"):
        decoded = json.loads(reference)
        if set(decoded) != {"path", "prefix"}:
            raise ContractError("batched artifact reference keys differ")
        path = decoded["path"]
        prefix = decoded["prefix"]
    with np.load(path, allow_pickle=False) as artifact:
        if prefix:
            result = {
                name[len(prefix):]: artifact[name].copy()
                for name in artifact.files if name.startswith(prefix)
            }
            if not result:
                raise ContractError("batched artifact prefix has no arrays")
            return result
        return {name: artifact[name].copy() for name in artifact.files}


class AppendOnlyTelemetry:
    """Thread-safe private JSONL timing sink; never enters the wire response."""

    def __init__(self, path: Path):
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(
            self.path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_CLOEXEC,
            0o600,
        )
        self._lock = threading.Lock()

    def append(self, record: dict[str, Any]) -> None:
        payload = (canonical_json({
            "schema": "roborender.dagger.h200-telemetry.v1",
            "recorded_at": _now(),
            **record,
        }) + "\n").encode("utf-8")
        with self._lock:
            written = 0
            while written < len(payload):
                written += os.write(self._fd, payload[written:])

    def close(self) -> None:
        with self._lock:
            if self._fd >= 0:
                os.close(self._fd)
                self._fd = -1


class TrajectorySink:
    def __init__(self, base_url: str, token: str):
        self.client = ServiceClient(base_url, token, timeout=180)

    def publish(self, chunk: Path, reservation: ChunkReservation) -> dict[str, Any]:
        result = self.client.post("/v1/chunks/reserve", reservation.as_dict())
        if result["status"] == "committed":
            return result
        return self.client.put_file(result["upload_path"], chunk)


class SharedFilesystemTrajectorySink:
    """Commit an H200-local learner chunk without copying it over loopback HTTP.

    The render/student gateway and trajectory service intentionally share one
    state root on the H200 node.  Streaming a multi-gigabyte iteration back to
    localhost duplicated every byte and then validated it a second time.  This
    sink preserves the same immutable reservation row while publishing a hard
    link to the already sealed file.  No model tensor is reread here.
    """

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.store = StateStore(self.root)
        self.sealed_root = (self.root / "sealed").resolve()
        self.chunk_root = (self.root / "chunks").resolve()
        self.chunk_root.mkdir(parents=True, exist_ok=True)

    def publish(self, chunk: Path, reservation: ChunkReservation) -> dict[str, Any]:
        source = chunk.resolve()
        if not source.is_file() or not source.is_relative_to(self.sealed_root):
            raise ContractError("shared learner chunk must be below the sealed root")
        if source.stat().st_size != reservation.size_bytes:
            raise ContractError("shared learner chunk size differs from reservation")
        metadata = canonical_json(reservation.as_dict())
        final = self.chunk_root / f"{reservation.chunk_id}{source.suffix}"
        now = _now()
        with self.store.transaction() as db:
            row = db.execute(
                "SELECT sha256,size_bytes,metadata_json,status,path "
                "FROM chunk_reservations WHERE chunk_id=?",
                (reservation.chunk_id,),
            ).fetchone()
            if row is not None:
                if (
                    row["sha256"] != reservation.sha256
                    or row["size_bytes"] != reservation.size_bytes
                    or row["metadata_json"] != metadata
                ):
                    raise ConflictError("chunk_id already identifies different content")
                if row["status"] != "committed" or not row["path"]:
                    raise ConflictError("existing shared chunk reservation is incomplete")
                return {
                    "ok": True, "status": "committed", "idempotent": True,
                    "chunk_id": reservation.chunk_id, "sha256": reservation.sha256,
                }
            if final.exists():
                if not os.path.samefile(source, final):
                    raise ConflictError("shared learner chunk path already exists")
            else:
                os.link(source, final)
            db.execute(
                "INSERT INTO chunk_reservations VALUES (?,?,?,?,?,?,?,?)",
                (
                    reservation.chunk_id, reservation.sha256, reservation.size_bytes,
                    metadata, "committed", str(final), now, now,
                ),
            )
        return {
            "ok": True, "status": "committed", "idempotent": False,
            "chunk_id": reservation.chunk_id, "sha256": reservation.sha256,
        }


class LocalTrajectorySink:
    """Test sink with the same immutable byte/hash behavior."""

    def __init__(self, root: Path):
        self.root = root
        self.calls = 0

    def publish(self, chunk: Path, reservation: ChunkReservation) -> dict[str, Any]:
        self.calls += 1
        self.root.mkdir(parents=True, exist_ok=True)
        final = self.root / f"{reservation.chunk_id}.npz"
        if final.exists() and file_sha256(final) != reservation.sha256:
            raise ConflictError("test sink already contains different chunk bytes")
        if not final.exists():
            os.link(chunk, final)
        return {"ok": True, "status": "committed", "sha256": reservation.sha256}


class RenderStudentHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, *, root: Path, token: str,
                 backend: RenderStudentBackend, trajectory_sink: Any,
                 telemetry_path: Path | None = None,
                 phase_telemetry_path: Path | None = None,
                 ephemeral_window_cache: bool = False,
                 max_request_bytes: int = 64 * 1024 * 1024):
        super().__init__(address, handler)
        self.root = root.resolve()
        self.store = StateStore(self.root)
        self.token = token
        self.backend = backend
        self.trajectory_sink = trajectory_sink
        self.max_request_bytes = max_request_bytes
        self.started_at = _now()
        self.gateway_instance_id = os.urandom(32).hex()
        self.telemetry = (
            None if telemetry_path is None else AppendOnlyTelemetry(telemetry_path)
        )
        self.phase_telemetry = (
            None if phase_telemetry_path is None else PhaseTelemetry(phase_telemetry_path)
        )
        self.operation_lock = threading.Lock()
        self.batch_prepare_lock = threading.Lock()
        # A lean-batch commit request is one vector-environment barrier. Serialize at
        # that granularity so it can be persisted in one SQLite transaction;
        # expanding it into per-lane writer threads causes catastrophic WAL
        # lock convoying on the shared project filesystem.
        self.batch_commit_lock = threading.Lock()
        # In lean streaming mode an incomplete native-simulator window is not
        # recoverable after either endpoint restarts: the Isaac/RNN/RNG state
        # cannot be reconstructed from a SQLite row.  Persisting every large
        # observation and every receipt with FULL fsync therefore bought no
        # usable recovery while placing two storage barriers in every control
        # step.  Keep the 81-step working set in RAM and publish only the final
        # learner chunks.  A PID/generation change is a fail-closed run abort.
        self.ephemeral_window_cache = bool(ephemeral_window_cache)
        self.window_cache_lock = threading.RLock()
        self.window_prepares: dict[str, dict[str, Any]] = {}
        self.window_prepares_by_step: dict[tuple[Any, ...], str] = {}
        self.window_commits: dict[str, dict[str, Any]] = {}
        self.window_commit_by_prepare: dict[str, str] = {}
        self.window_checkpoints: dict[tuple[str, int, str], str] = {}
        self.completed_window_keys: set[tuple[str, int, str, str]] = set()
        self.stream_batch_digests: dict[tuple[str, str], str] = {}
        self.prepare_inflight: dict[
            str, tuple[str, str, threading.Event]
        ] = {}
        self.prepare_inflight_ids: dict[str, tuple[str, str]] = {}
        self.commit_inflight: dict[str, threading.Event] = {}
        # Anchors are immutable/content-addressed and reused at every step of
        # an episode.  Keep them in RAM after the first verification instead
        # of reopening and hashing three images for every lane and every step.
        self.anchor_cache: dict[str, dict[str, np.ndarray]] = {}
        self.anchor_cache_lock = threading.Lock()
        self.observation_contracts: dict[str, dict[str, Any] | None] = {}
        (self.root / "prepared").mkdir(parents=True, exist_ok=True)
        (self.root / "sealed").mkdir(parents=True, exist_ok=True)
        (self.root / "anchors").mkdir(parents=True, exist_ok=True)

    def emit_telemetry(self, record: dict[str, Any]) -> None:
        if self.telemetry is None:
            return
        try:
            self.telemetry.append(record)
        except Exception as exc:  # telemetry must never alter DAgger semantics
            print(f"render/student telemetry write failed: {exc}", file=sys.stderr, flush=True)

    def admit_observation_sources(self, rows) -> None:
        """Enforce registered generated-only policy before inference or replay.

        The run registration is immutable. Cache only registered identities,
        never a missing run, so an early rejected request cannot pin a downgrade.
        """
        from ..rtx.code.src.dagger_rtx.observation_contract import (
            require_generated_source, validate_generated_only,
        )
        for key, metadata in rows:
            if key.run_id not in self.observation_contracts:
                with self.store.connection() as db:
                    run = db.execute("SELECT 1 FROM runs WHERE run_id=?", (key.run_id,)).fetchone()
                    row = db.execute("SELECT contract_json FROM run_observation_contracts WHERE run_id=?", (key.run_id,)).fetchone()
                if run is None:
                    # Legacy isolated component tests may not register runs;
                    # checkpoint/order admission still rejects real unknown runs.
                    continue
                contract = None if row is None else json.loads(row[0])
                if contract is not None:
                    validate_generated_only(contract)
                    workers = self.backend.backends if isinstance(self.backend, RenderStudentBackendPool) else (self.backend,)
                    if not workers or any(not isinstance(worker, LiveRenderStudentBackend) for worker in workers):
                        raise NotReadyError("generated_only requires genuine live Image3F/student backends")
                    for worker in workers:
                        ready, reason = worker.ready()
                        if not ready:
                            raise NotReadyError(f"generated_only renderer unavailable: {reason}")
                        try:
                            worker.require_generated_observations(contract)
                        except (ValueError, RuntimeError, OSError) as exc:
                            raise NotReadyError(f'generated_only renderer contract unavailable: {exc}') from exc
                self.observation_contracts[key.run_id] = contract
            try:
                require_generated_source(self.observation_contracts[key.run_id], metadata["observation_source"])
            except ValueError as exc:
                raise ContractError(str(exc)) from exc

    @contextmanager
    def phase_span(self, key: StepKey, phase: str, **details):
        """Report a timed boundary without allowing telemetry to affect service semantics."""

        # V4 emits one aggregate request record through ``emit_telemetry``.
        # Writing a start and end JSONL record for each nested phase on every
        # gateway recreated avoidable hot-loop filesystem traffic.  Detailed
        # spans remain available on the durable v1/v3 diagnostic path.
        if self.phase_telemetry is None or self.ephemeral_window_cache:
            yield details
            return
        span = self.phase_telemetry.span(
            run_id=key.run_id,
            round_id=key.round_id,
            vector_step_index=key.rollout_step_index,
            actor_id=key.actor_id,
            phase=phase,
            details=details,
        )
        try:
            span_details = span.__enter__()
        except Exception as exc:
            print(f"phase telemetry start failed: {exc}", file=sys.stderr, flush=True)
            yield details
            return
        try:
            yield span_details
        except BaseException as workload_error:
            try:
                span.__exit__(type(workload_error), workload_error, workload_error.__traceback__)
            except Exception as telemetry_error:
                print(f"phase telemetry error write failed: {telemetry_error}", file=sys.stderr, flush=True)
            raise
        else:
            try:
                span.__exit__(None, None, None)
            except Exception as exc:
                print(f"phase telemetry end failed: {exc}", file=sys.stderr, flush=True)

    def server_close(self) -> None:
        try:
            if self.telemetry is not None:
                self.telemetry.close()
            if self.phase_telemetry is not None:
                self.phase_telemetry.close()
        finally:
            super().server_close()

    def active_checkpoint(self, key: StepKey) -> str:
        with self.store.connection() as db:
            run = db.execute("SELECT * FROM runs WHERE run_id=?", (key.run_id,)).fetchone()
            validation = db.execute("SELECT registration_json FROM validation_collections WHERE run_id=?", (key.run_id,)).fetchone()
            if validation is not None:
                registration = json.loads(validation[0])
                if (key.actor_id not in registration["actor_ids"] or key.rollout_window_id != "window-000000"
                        or db.execute("SELECT 1 FROM validation_collection_actors WHERE run_id=? AND actor_id=?",
                                      (key.run_id, key.actor_id)).fetchone() is not None):
                    raise NotReadyError("validation collection permits one sealed window per admitted actor")
        if run is None:
            # Component tests may intentionally run without the coordinator.
            digest = getattr(self.backend, "test_checkpoint_sha256", None)
            if digest:
                return digest
            raise NotReadyError(f"coordinator has no run {key.run_id}")
        if run["current_round"] != key.round_id or run["status"] != "collecting":
            raise NotReadyError("requested round is not active at the coordinator barrier")
        return sha256_digest(run["active_checkpoint_sha256"], "active checkpoint")

    def window_checkpoint(self, key: StepKey) -> str:
        """Resolve a checkpoint once for an immutable rollout window."""

        self._admit_validation_keys([key])
        cache_key = (key.run_id, key.round_id, key.rollout_window_id)
        with self.window_cache_lock:
            cached = self.window_checkpoints.get(cache_key)
            if cached is not None:
                return cached
        checkpoint = self.active_checkpoint(key)
        with self.window_cache_lock:
            return self.window_checkpoints.setdefault(cache_key, checkpoint)

    def batch_window_checkpoint(self, keys: list[StepKey]) -> str:
        """Use one coordinator read for every lane in a vector barrier."""

        if not keys:
            raise ContractError("cannot resolve a checkpoint for an empty batch")
        self._admit_validation_keys(keys)
        cache_keys = [(key.run_id, key.round_id, key.rollout_window_id) for key in keys]
        with self.window_cache_lock:
            cached = {self.window_checkpoints[item] for item in cache_keys
                      if item in self.window_checkpoints}
        if len(cached) > 1:
            raise ConflictError("vector lanes are pinned to different checkpoints")
        checkpoint = next(iter(cached), None)
        if checkpoint is None:
            checkpoint = self.active_checkpoint(keys[0])
        with self.window_cache_lock:
            for item in cache_keys:
                existing = self.window_checkpoints.setdefault(item, checkpoint)
                if existing != checkpoint:
                    raise ConflictError("checkpoint changed inside a vector window")
        return checkpoint

    def _admit_validation_keys(self, keys: list[StepKey]) -> None:
        """Pin all held-out actors once, even on the shared window-cache path."""
        for run_id in {key.run_id for key in keys if key.run_id.startswith("heldout-")}:
            cache = getattr(self, "_validation_registration_cache", None)
            if cache is None:
                cache = self._validation_registration_cache = {}
            registration = cache.get(run_id)
            if registration is None:
                with self.store.connection() as db:
                    row = db.execute("SELECT registration_json FROM validation_collections WHERE run_id=?", (run_id,)).fetchone()
                if row is None:
                    raise NotReadyError("held-out namespace requires a registered validation owner")
                registration = json.loads(row[0])
                cache[run_id] = registration
            for key in keys:
                if key.run_id == run_id and (key.actor_id not in registration["actor_ids"]
                        or key.round_id != registration["round_id"] or key.rollout_window_id != "window-000000"):
                    raise NotReadyError("held-out window cache cannot authorize an undeclared actor/round/window")

    def purge_completed_actor_windows(self, key: StepKey) -> None:
        """Release the previous finalized window when its actor starts the next one."""

        if key.rollout_step_index != 0:
            return
        current = (key.run_id, key.round_id, key.actor_id, key.rollout_window_id)
        with self.window_cache_lock:
            stale = {
                item for item in self.completed_window_keys
                if item[:3] == current[:3] and item != current
            }
            for run_id, round_id, actor_id, window_id in stale:
                step_keys = [
                    item for item in self.window_prepares_by_step
                    if item[:4] == (run_id, round_id, actor_id, window_id)
                ]
                for step_key in step_keys:
                    prepare_identity = self.window_prepares_by_step.pop(step_key)
                    commit_identity = self.window_commit_by_prepare.pop(
                        prepare_identity, None
                    )
                    if commit_identity is not None:
                        self.window_commits.pop(commit_identity, None)
                    self.window_prepares.pop(prepare_identity, None)
                self.window_checkpoints.pop((run_id, round_id, window_id), None)
                self.completed_window_keys.discard(
                    (run_id, round_id, actor_id, window_id)
                )

    def bind_stream_batch(self, route: str, batch_id: str, request_digest: str) -> None:
        """Make each v4 batch identity immutable across ambiguous retries."""

        identity = (route, batch_id)
        digest = sha256_digest(request_digest, "v4 request digest")
        with self.window_cache_lock:
            previous = self.stream_batch_digests.setdefault(identity, digest)
            if previous != digest:
                raise ConflictError("v4 batch_id was reused with changed request bytes")


class Handler(BaseHTTPRequestHandler):
    server: RenderStudentHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        return

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = canonical_bytes(payload) + b"\n"
        if len(body) > 1024 * 1024:
            raise RuntimeError("response exceeds the 1 MiB contract")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, code: str, message: str) -> None:
        self._json(status, {"ok": False, "error": code, "message": message})

    def _authorized(self) -> bool:
        return hmac.compare_digest(
            self.headers.get("Authorization", ""), f"Bearer {self.server.token}"
        )

    def _read_raw(self) -> tuple[str, bytes]:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ContractError("Content-Length is required")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ContractError("Content-Length must be an integer") from exc
        if length < 0 or length > self.server.max_request_bytes:
            raise ContractError("request exceeds the 64 MiB contract")
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise ContractError("request ended before Content-Length")
        return content_type, raw

    def _read(self) -> tuple[dict[str, Any], bytes]:
        content_type, raw = self._read_raw()
        if content_type != "application/json":
            raise ContractError("Content-Type must be application/json")
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractError(f"invalid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ContractError("JSON body must be an object")
        # Reject alternate lexical representations: the wire format itself is canonical.
        if raw != canonical_bytes(payload):
            raise ContractError("request JSON is not canonical")
        return payload, raw

    def do_GET(self) -> None:
        try:
            path = urllib.parse.urlsplit(self.path).path
            if path not in ("/live", "/ready", "/version"):
                return self._error(HTTPStatus.NOT_FOUND, "not_found", path)
            if path == "/ready":
                ready = getattr(self.server.backend, "ready", lambda: (True, "ready"))()
                if not ready[0]:
                    return self._json(HTTPStatus.SERVICE_UNAVAILABLE, {
                        "ok": False, "role": "render_student", "schema": STEP_CONTRACT,
                        "reason": ready[1],
                    })
            supported = [STEP_CONTRACT, BATCH_CONTRACT]
            if self.server.ephemeral_window_cache:
                supported.append(STREAM_CONTRACT)
            return self._json(HTTPStatus.OK, {
                "ok": True, "role": "render_student", "schema": STEP_CONTRACT,
                "supported_contract_versions": supported,
                "gateway_instance_id": self.server.gateway_instance_id,
                "features": {
                    "minimal_compressed_prepare_batch": True,
                    "content_addressed_anchor_cache": True,
                    "source_specific_observation_payload": True,
                    "learner_only_persistence": True,
                    "shared_filesystem_chunk_publish": True,
                    "batched_commit": True,
                    "ephemeral_window_cache": self.server.ephemeral_window_cache,
                    "pipelined_commit_prepare": self.server.ephemeral_window_cache,
                    "aggregate_hot_path_telemetry": self.server.ephemeral_window_cache,
                    "hot_path_fsyncs": 0 if self.server.ephemeral_window_cache else 2,
                },
                "pid": os.getpid(), "uptime_seconds": _now() - self.server.started_at,
            })
        except Exception as exc:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error", str(exc))

    def do_POST(self) -> None:
        try:
            if not self._authorized():
                return self._error(HTTPStatus.UNAUTHORIZED, "unauthorized", "valid bearer token required")
            path = urllib.parse.urlsplit(self.path).path
            if path in (
                "/v3/anchors/register-batch", "/v3/steps/prepare-batch",
                "/v4/steps/prepare-batch", "/v4/steps/exchange-batch",
            ):
                content_type, raw = self._read_raw()
                if path.startswith("/v4/"):
                    if not self.server.ephemeral_window_cache:
                        raise NotReadyError("v4 requires --ephemeral-window-cache")
                    if content_type != STREAM_CONTENT_TYPE:
                        raise ContractError(f"Content-Type must be {STREAM_CONTENT_TYPE}")
                    header, arrays = decode_stream_batch(raw)
                    self._require_gateway_instance(header)
                    if path == "/v4/steps/prepare-batch":
                        response = self._prepare_stream_start(header, arrays, raw)
                    else:
                        response = self._exchange_stream_batch(header, arrays, raw)
                else:
                    if content_type != BATCH_CONTENT_TYPE:
                        raise ContractError(f"Content-Type must be {BATCH_CONTENT_TYPE}")
                    header, arrays = decode_binary_batch(raw)
                    if path == "/v3/anchors/register-batch":
                        response = self._register_anchor_batch(header, arrays)
                    else:
                        response = self._prepare_batch(header, arrays, raw)
            else:
                payload, raw = self._read()
                if path == "/v1/steps/prepare":
                    response = self._prepare(payload, raw)
                elif path == "/v1/steps/commit":
                    response = self._commit(payload, raw)
                elif path == "/v3/steps/commit-batch":
                    response = self._commit_batch(payload)
                elif path == "/v4/steps/finalize-batch":
                    if not self.server.ephemeral_window_cache:
                        raise NotReadyError("v4 requires --ephemeral-window-cache")
                    response = self._finalize_stream_batch(payload, raw)
                else:
                    return self._error(HTTPStatus.NOT_FOUND, "not_found", path)
            self._json(HTTPStatus.OK, response)
        except ContractError as exc:
            self._error(HTTPStatus.BAD_REQUEST, "contract_error", str(exc))
        except ConflictError as exc:
            self._error(HTTPStatus.CONFLICT, "conflict", str(exc))
        except NotReadyError as exc:
            self._error(HTTPStatus.PRECONDITION_FAILED, "not_ready", str(exc))
        except Exception as exc:
            traceback.print_exc()
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "internal_error", str(exc))

    def _require_gateway_instance(self, payload: dict[str, Any]) -> None:
        supplied = sha256_digest(
            payload.get("gateway_instance_id"), "gateway_instance_id"
        )
        if supplied != self.server.gateway_instance_id:
            raise ConflictError("gateway instance changed; abandon the incomplete window")

    def _prepare_stream_start(
        self, header: dict[str, Any], arrays: list[dict[str, np.ndarray]], raw: bytes
    ) -> dict[str, Any]:
        parsed = parse_prepare_batch(
            header, arrays, expected_contract=STREAM_CONTRACT,
            extra_header_keys={"gateway_instance_id"},
        )
        self.server.bind_stream_batch(
            "prepare", header["batch_id"], bytes_sha256(canonical_bytes(header))
        )
        if any(item[0].rollout_step_index != 0 for item in parsed):
            raise ContractError("v4 prepare-batch is only valid for rollout step zero")
        result = self._prepare_batch(
            header, arrays, raw, expected_contract=STREAM_CONTRACT,
            extra_header_keys={"gateway_instance_id"},
        )
        return {**result, "gateway_instance_id": self.server.gateway_instance_id}

    def _exchange_stream_batch(
        self, header: dict[str, Any], arrays: list[dict[str, np.ndarray]], raw: bytes
    ) -> dict[str, Any]:
        from .contracts import exact_keys

        exact_keys(header, {
            "contract_version", "kind", "batch_id", "gateway_instance_id",
            "previous_commits", "lanes", "blob_codec", "raw_nbytes", "raw_sha256",
        })
        if header["contract_version"] != STREAM_CONTRACT or header["kind"] != "exchange_batch":
            raise ContractError("stream exchange contract or kind mismatch")
        self.server.bind_stream_batch(
            "exchange", header["batch_id"], bytes_sha256(canonical_bytes(header))
        )
        current = parse_prepare_batch(
            header, arrays, expected_contract=STREAM_CONTRACT,
            extra_header_keys={"gateway_instance_id", "previous_commits"},
        )
        previous = header["previous_commits"]
        if not isinstance(previous, list) or len(previous) != len(current):
            raise ContractError("exchange must pair every current lane with one prior commit")
        for current_item, previous_lane in zip(current, previous):
            if not isinstance(previous_lane, dict):
                raise ContractError("each previous commit must be an object")
            current_key = current_item[0]
            previous_key = StepKey.parse(previous_lane.get("step_key"))
            if (
                current_key.run_id != previous_key.run_id
                or current_key.round_id != previous_key.round_id
                or current_key.actor_id != previous_key.actor_id
                or current_key.rollout_window_id != previous_key.rollout_window_id
                or current_key.rollout_step_index != previous_key.rollout_step_index + 1
                or current_key.rollout_step_index < 1
                or current_key.rollout_step_index > 80
            ):
                raise ContractError(
                    "exchange commit/current prepare lanes are not consecutive and aligned"
                )

        committed = self._commit_batch({
            "contract_version": STREAM_CONTRACT,
            "kind": "commit_batch",
            "batch_id": f"{header['batch_id']}-previous",
            "lanes": previous,
        }, expected_contract=STREAM_CONTRACT)
        prepared = self._prepare_batch(
            header, arrays, raw, expected_contract=STREAM_CONTRACT,
            extra_header_keys={"gateway_instance_id", "previous_commits"},
        )
        return {
            "contract_version": STREAM_CONTRACT,
            "kind": "exchange_batch_response",
            "batch_id": header["batch_id"],
            "gateway_instance_id": self.server.gateway_instance_id,
            "previous_commit_acks": committed["lanes"],
            "lanes": prepared["lanes"],
        }

    def _finalize_stream_batch(
        self, payload: dict[str, Any], raw: bytes
    ) -> dict[str, Any]:
        from .contracts import exact_keys

        exact_keys(payload, {
            "contract_version", "kind", "batch_id", "gateway_instance_id", "lanes",
        })
        if payload["contract_version"] != STREAM_CONTRACT or payload["kind"] != "finalize_batch":
            raise ContractError("stream finalize contract or kind mismatch")
        self._require_gateway_instance(payload)
        self.server.bind_stream_batch(
            "finalize", payload["batch_id"], bytes_sha256(raw)
        )
        if any(
            StepKey.parse(lane.get("step_key")).rollout_step_index != 80
            for lane in payload["lanes"]
        ):
            raise ContractError("v4 finalize-batch is only valid for rollout step 80")
        result = self._commit_batch({
            "contract_version": STREAM_CONTRACT,
            "kind": "finalize_batch",
            "batch_id": payload["batch_id"],
            "lanes": payload["lanes"],
        }, expected_contract=STREAM_CONTRACT, request_kind="finalize_batch",
            response_kind="finalize_batch_response")
        return {**result, "gateway_instance_id": self.server.gateway_instance_id}

    def _register_anchor_batch(
        self, header: dict[str, Any], arrays_by_anchor: list[dict[str, np.ndarray]]
    ) -> dict[str, Any]:
        from .contracts import exact_keys, identifier

        exact_keys(header, {
            "contract_version", "kind", "batch_id", "anchors",
            "blob_codec", "raw_nbytes", "raw_sha256",
        })
        if header["contract_version"] != BATCH_CONTRACT or header["kind"] != "anchor_batch":
            raise ContractError("anchor batch contract or kind mismatch")
        batch_id = identifier(header["batch_id"], "batch_id")
        if len(header["anchors"]) != len(arrays_by_anchor):
            raise ContractError("anchor batch item count mismatch")
        results = []
        for item, arrays in zip(header["anchors"], arrays_by_anchor):
            exact_keys(item, {"anchor_sha256", "arrays"})
            expected = sha256_digest(item["anchor_sha256"], "anchor_sha256")
            if anchor_arrays_digest(arrays) != expected:
                raise ContractError("registered anchor bytes do not match anchor_sha256")
            final = self.server.root / "anchors" / f"{expected}.npz"
            status = "cached"
            if not final.exists():
                # Eight gateway processes share this content-addressed store.
                # Publish with an atomic hard link so concurrent registration
                # of the same anchor can never overwrite another writer.
                temporary = final.with_name(
                    f".{final.name}.{os.getpid()}.{threading.get_ident()}.part"
                )
                try:
                    with temporary.open("xb") as stream:
                        np.savez(stream, **arrays)
                        stream.flush()
                        os.fsync(stream.fileno())
                    try:
                        os.link(temporary, final)
                        status = "registered"
                    except FileExistsError:
                        status = "cached"
                finally:
                    temporary.unlink(missing_ok=True)
            if status == "cached":
                with np.load(final, allow_pickle=False) as cached:
                    cached_arrays = {name: cached[name].copy() for name in cached.files}
                if anchor_arrays_digest(cached_arrays) != expected:
                    raise ConflictError("cached anchor bytes differ from their content address")
            results.append({
                "anchor_sha256": expected, "status": status, "bytes": final.stat().st_size,
            })
        return {
            "contract_version": BATCH_CONTRACT,
            "kind": "anchor_batch_response",
            "batch_id": batch_id,
            "anchors": results,
        }

    def _cached_anchor(self, digest: str) -> dict[str, np.ndarray]:
        with self.server.anchor_cache_lock:
            cached = self.server.anchor_cache.get(digest)
            if cached is not None:
                return cached
        path = self.server.root / "anchors" / f"{digest}.npz"
        if not path.is_file():
            raise NotReadyError(f"anchor {digest} is not registered")
        with np.load(path, allow_pickle=False) as data:
            arrays = {name: data[name].copy() for name in data.files}
        if anchor_arrays_digest(arrays) != digest:
            raise ConflictError("cached anchor failed its content digest")
        with self.server.anchor_cache_lock:
            return self.server.anchor_cache.setdefault(digest, arrays)

    @staticmethod
    def _artifact_arrays(
        arrays: dict[str, np.ndarray], result: Any, selected: np.ndarray,
        student_26: np.ndarray, use_teacher: bool, beta: float,
    ) -> dict[str, np.ndarray]:
        # The learner consumes exactly three rendered RGB views, actor state,
        # and the teacher target.  All other controls/actions/outcomes are
        # already durably represented by the prepare/commit SQLite ledger.
        # Copying depth, masks, critic/Homie state and behavior actions into
        # every prepared artifact added ~2.7 GB per B=28 window with no effect
        # on the gradient.
        return {
            "actor_obs": arrays["actor_obs"],
            "teacher_action": arrays["teacher_action"],
            **{f"rgb_gen__{view}": result.policy_rgb[view] for view in VIEWS},
        }

    def _prepare_batch(
        self, header: dict[str, Any], arrays_by_lane: list[dict[str, np.ndarray]], raw: bytes,
        *, expected_contract: str = BATCH_CONTRACT,
        extra_header_keys: set[str] | None = None,
    ) -> dict[str, Any]:
        """Prepare a vector barrier and durably persist all new lanes once."""

        batch_started = time.perf_counter()
        parsed = parse_prepare_batch(
            header, arrays_by_lane, expected_contract=expected_contract,
            extra_header_keys=extra_header_keys,
        )
        self.server.admit_observation_sources([(item[0], item[2]) for item in parsed])
        batch_id = header["batch_id"]
        with self.server.batch_prepare_lock:
            existing: dict[str, dict[str, Any]] = {}
            pending = []
            checkpoint_sha = self.server.batch_window_checkpoint(
                [item[0] for item in parsed]
            )
            if self.server.ephemeral_window_cache:
                with self.server.window_cache_lock:
                    for key, beta, metadata, arrays, lane in parsed:
                        self.server.purge_completed_actor_windows(key)
                        logical = {
                            name: value for name, value in lane.items()
                            if name != "arrays"
                        }
                        request_sha = bytes_sha256(canonical_bytes(logical))
                        identity = lane["idempotency_key"]
                        row = self.server.window_prepares.get(identity)
                        if row is not None:
                            if row["request_sha256"] != request_sha:
                                raise ConflictError(
                                    "idempotency key was reused with a changed batched prepare"
                                )
                            existing[identity] = row["response"]
                            continue
                        self._check_order_memory(key, metadata)
                        pending.append(
                            (key, beta, metadata, arrays, lane, request_sha, checkpoint_sha)
                        )
            else:
                with self.server.operation_lock:
                    # This phase is strictly read-only: it checks idempotency,
                    # predecessor ordering, and the coordinator checkpoint. A
                    # BEGIN IMMEDIATE here would unnecessarily acquire
                    # SQLite's sole writer slot before rendering.
                    with self.server.store.connection() as db:
                        for key, beta, metadata, arrays, lane in parsed:
                            logical = {
                                name: value for name, value in lane.items()
                                if name != "arrays"
                            }
                            request_sha = bytes_sha256(canonical_bytes(logical))
                            identity = lane["idempotency_key"]
                            row = db.execute(
                                "SELECT request_sha256,response_json FROM step_prepares "
                                "WHERE idempotency_key=?", (identity,),
                            ).fetchone()
                            if row is not None:
                                if row["request_sha256"] != request_sha:
                                    raise ConflictError(
                                        "idempotency key was reused with a changed batched prepare"
                                    )
                                existing[identity] = json.loads(row["response_json"])
                                continue
                            self._check_order(db, key, metadata)
                            pending.append((
                                key, beta, metadata, arrays, lane,
                                request_sha, checkpoint_sha,
                            ))

            def render_lane(item):
                key, beta, metadata, arrays, lane, request_sha, checkpoint_sha = item
                inference_arrays = dict(arrays)
                anchor_sha = metadata["episode_assignment"]["anchor_sha256"]
                if metadata["observation_source"] == "generated_controls":
                    inference_arrays.update(self._cached_anchor(anchor_sha))
                started = time.perf_counter()
                result = self.server.backend.infer(metadata, inference_arrays, checkpoint_sha)
                return item, result, (time.perf_counter() - started) * 1000.0

            def select_lane(rendered):
                item, result, render_ms = rendered
                key, beta, metadata, arrays, lane, request_sha, checkpoint_sha = item
                started = time.perf_counter()
                student_26 = np.concatenate(
                    (result.student_action_11, arrays["teacher_action"][11:]),
                    dtype=np.float32,
                )
                selection_id = bytes_sha256(canonical_bytes({
                    "contract_version": expected_contract,
                    "idempotency_key": lane["idempotency_key"],
                    "step_key": key.as_dict(),
                    "capture_digest": lane["capture_digest"],
                    "wire_digest": lane["wire_digest"],
                    "checkpoint_sha256": checkpoint_sha,
                }))
                draw = int(selection_id[:16], 16) / float(1 << 64)
                if 'raven_selection' in lane:
                    draw = lane['raven_selection']['draw']
                    selection_id = bytes_sha256(canonical_bytes({
                        'request_selection_id': selection_id,
                        'raven_selection': lane['raven_selection'],
                    }))
                use_teacher = draw < beta
                selected = (
                    arrays["teacher_action"].copy() if use_teacher else student_26.copy()
                )
                body = {
                    "contract_version": expected_contract,
                    "idempotency_key": lane["idempotency_key"],
                    "step_key": key.as_dict(),
                    "prepare_receipt": lane["prepare_receipt"],
                    "capture_digest": lane["capture_digest"],
                    "wire_digest": lane["wire_digest"],
                    "beta": beta,
                    "selection_id": selection_id,
                    "selection_draw": draw,
                    "selected_source": "teacher" if use_teacher else "student",
                    "student_action_11": encode_array(result.student_action_11),
                    "selected_action": encode_array(
                        np.ascontiguousarray(selected, dtype=np.float32)
                    ),
                    "observation_digest": result.observation_digest,
                }
                response = {**body, "selection_receipt": response_receipt(body)}
                return {
                    "key": key, "metadata": metadata, "lane": lane,
                    "request_sha": request_sha, "checkpoint_sha": checkpoint_sha,
                    "response": response,
                    "render_ms": render_ms,
                    "selection_ms": (time.perf_counter() - started) * 1000.0,
                    "backend_telemetry": result.telemetry,
                    "artifact_arrays": self._artifact_arrays(
                        arrays, result, selected, student_26, use_teacher, beta
                    ),
                }

            computed = []
            if pending:
                workers = min(len(pending), max(1, getattr(self.server.backend, "size", 8)))
                key0 = pending[0][0]
                with self.server.phase_span(
                    key0, "h200_image3f_render_and_student_inference",
                    batch_id=batch_id, lane_count=len(pending), worker_count=workers,
                    checkpoint_sha256=pending[0][6],
                ) as details:
                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        rendered = list(executor.map(render_lane, pending))
                    details.update({
                        "generated_controls": sum(
                            item[0][2]["observation_source"] == "generated_controls"
                            for item in rendered
                        ),
                        "raytraced": sum(
                            item[0][2]["observation_source"] == "raytraced"
                            for item in rendered
                        ),
                        "slowest_lane_ms": max(item[2] for item in rendered),
                    })
                with self.server.phase_span(
                    key0, "h200_beta_action_selection",
                    batch_id=batch_id, lane_count=len(pending), beta=pending[0][1],
                    checkpoint_sha256=pending[0][6],
                ) as details:
                    computed = [select_lane(item) for item in rendered]
                    details.update({
                        "teacher": sum(
                            item["response"]["selected_source"] == "teacher"
                            for item in computed
                        ),
                        "student": sum(
                            item["response"]["selected_source"] == "student"
                            for item in computed
                        ),
                        "total_selection_ms": sum(item["selection_ms"] for item in computed),
                    })

                if self.server.ephemeral_window_cache:
                    with self.server.phase_span(
                        computed[0]["key"], "h200_learner_buffer_cache",
                        batch_id=batch_id, lane_count=len(computed),
                        checkpoint_sha256=computed[0]["checkpoint_sha"],
                    ) as details:
                        with self.server.window_cache_lock:
                            for item in computed:
                                key = item["key"]
                                lane = item["lane"]
                                identity = lane["idempotency_key"]
                                step_identity = (
                                    key.run_id, key.round_id, key.actor_id,
                                    key.rollout_window_id, key.rollout_step_index,
                                )
                                prior_identity = self.server.window_prepares_by_step.get(
                                    step_identity
                                )
                                if prior_identity is not None and prior_identity != identity:
                                    raise ConflictError(
                                        "rollout step was already prepared under another key"
                                    )
                                self.server.window_prepares[identity] = {
                                    "request_sha256": item["request_sha"],
                                    "key": key,
                                    "metadata": item["metadata"],
                                    "response": item["response"],
                                    "artifact_arrays": item["artifact_arrays"],
                                    "checkpoint_sha256": item["checkpoint_sha"],
                                    "capture_digest": lane["capture_digest"],
                                    "prepare_receipt": lane["prepare_receipt"],
                                    "status": "prepared",
                                }
                                self.server.window_prepares_by_step[step_identity] = identity
                        details["artifact_bytes"] = 0
                        details["durability"] = "final-window-only"
                else:
                    batch_sha = bytes_sha256(raw)
                    final = self.server.root / "prepared" / f"batch-{batch_id}-{batch_sha[:16]}.npz"
                    temporary = final.with_suffix(".npz.part")
                    packed = {}
                    for index, item in enumerate(computed):
                        prefix = f"lane{index:03d}__"
                        item["artifact_prefix"] = prefix
                        packed.update({
                            prefix + name: value
                            for name, value in item["artifact_arrays"].items()
                        })
                    with self.server.phase_span(
                        computed[0]["key"], "h200_learner_buffer_append",
                        batch_id=batch_id, lane_count=len(computed),
                        checkpoint_sha256=computed[0]["checkpoint_sha"],
                    ) as details:
                        with temporary.open("wb") as stream:
                            np.savez(stream, **packed)
                            stream.flush()
                            os.fsync(stream.fileno())
                        os.replace(temporary, final)
                        details["artifact_bytes"] = final.stat().st_size
                        now = _now()
                        try:
                            with self.server.operation_lock:
                                with self.server.store.transaction() as db:
                                    for item in computed:
                                        key = item["key"]
                                        lane = item["lane"]
                                        artifact_reference = canonical_json({
                                            "path": str(final),
                                            "prefix": item["artifact_prefix"],
                                        })
                                        db.execute(
                                            "INSERT INTO step_prepares VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                            (
                                                lane["idempotency_key"], item["request_sha"], key.run_id,
                                                key.round_id, key.actor_id, key.rollout_window_id,
                                                key.rollout_step_index, key.episode_id, key.episode_step_index,
                                                lane["capture_digest"], lane["prepare_receipt"],
                                                canonical_json(item["metadata"]),
                                                canonical_json(item["response"]), artifact_reference,
                                                item["checkpoint_sha"], "prepared", now, now,
                                            ),
                                        )
                        except Exception:
                            final.unlink(missing_ok=True)
                            raise

            by_identity = dict(existing)
            by_identity.update({item["lane"]["idempotency_key"]: item["response"] for item in computed})
            responses = [by_identity[lane[4]["idempotency_key"]] for lane in parsed]
            response = {
                "contract_version": expected_contract,
                "kind": "prepare_batch_response",
                "batch_id": batch_id,
                "lanes": responses,
            }
            self.server.emit_telemetry({
                "route": "prepare_batch", "status": "prepared",
                "batch_id": batch_id, "lane_count": len(parsed),
                "new_lane_count": len(computed), "request_bytes": len(raw),
                "response_bytes": len(canonical_bytes(response)) + 1,
                "server_total_ms": (time.perf_counter() - batch_started) * 1000.0,
            })
            return response

    def _commit_batch(
        self, payload: dict[str, Any], *, expected_contract: str = BATCH_CONTRACT,
        request_kind: str = "commit_batch", response_kind: str = "commit_batch_response",
    ) -> dict[str, Any]:
        from .contracts import exact_keys, identifier

        exact_keys(payload, {"contract_version", "kind", "batch_id", "lanes"})
        if payload["contract_version"] != expected_contract or payload["kind"] != request_kind:
            raise ContractError("commit batch contract or kind mismatch")
        batch_id = identifier(payload["batch_id"], "batch_id")
        lanes = payload["lanes"]
        if not isinstance(lanes, list) or not 1 <= len(lanes) <= 256:
            raise ContractError("commit batch must contain 1..256 lanes")
        batch_started = time.perf_counter()
        required = {
            "contract_version", "idempotency_key", "step_key", "prepare_receipt",
            "selection_id", "selection_receipt", "commit_receipt", "transition",
            "sealed_window",
        }

        # Validate the entire vector barrier before taking the writer lock.
        parsed = []
        seen_ids = set()
        for lane in lanes:
            exact_keys(lane, required)
            if lane["contract_version"] != expected_contract:
                raise ContractError("commit batch lane contract_version mismatch")
            identity = identifier(lane["idempotency_key"], "idempotency_key")
            if identity in seen_ids:
                raise ContractError("duplicate commit idempotency key in batch")
            seen_ids.add(identity)
            key = StepKey.parse(lane["step_key"])
            for field in (
                "prepare_receipt", "selection_id", "selection_receipt", "commit_receipt"
            ):
                sha256_digest(lane[field], field)
            transition = validate_transition(lane["transition"])
            if key.rollout_step_index == 80:
                if lane["sealed_window"] is None:
                    raise ContractError("rollout step 80 requires sealed_window")
                sealed = validate_sealed_window(
                    lane["sealed_window"], key, expected_contract=expected_contract
                )
            else:
                if lane["sealed_window"] is not None:
                    raise ContractError("sealed_window is allowed only at rollout step 80")
                sealed = None
            raw = canonical_bytes(lane)
            parsed.append({
                "payload": lane, "identity": identity, "key": key,
                "transition": transition, "sealed": sealed,
                "request_sha": bytes_sha256(raw),
            })
        vector_keys = {
            (item["key"].run_id, item["key"].round_id,
             item["key"].rollout_step_index)
            for item in parsed
        }
        if len(vector_keys) != 1:
            raise ContractError("all commit lanes must share one vector rollout step")

        if self.server.ephemeral_window_cache:
            return self._commit_batch_memory(
                payload, batch_id, lanes, parsed, batch_started,
                expected_contract=expected_contract, response_kind=response_kind,
            )

        # One request, one durable transaction.  This preserves per-lane
        # idempotency while eliminating 56 FULL-sync SQLite transactions and
        # their writer-lock convoy (28 inserts + 28 response updates).
        with self.server.batch_commit_lock:
            responses = []
            seals = []
            with self.server.operation_lock:
                with self.server.store.transaction() as db:
                    for item in parsed:
                        lane = item["payload"]
                        key = item["key"]
                        identity = item["identity"]
                        prepared = db.execute(
                            "SELECT * FROM step_prepares WHERE run_id=? AND round_id=? "
                            "AND actor_id=? AND rollout_window_id=? AND rollout_step_index=?",
                            (key.run_id, key.round_id, key.actor_id,
                             key.rollout_window_id, key.rollout_step_index),
                        ).fetchone()
                        if prepared is None:
                            raise ConflictError("commit has no matching prepare")
                        selection = json.loads(prepared["response_json"])
                        echoed = {
                            "step_key": selection["step_key"],
                            "prepare_receipt": selection["prepare_receipt"],
                            "selection_id": selection["selection_id"],
                            "selection_receipt": selection["selection_receipt"],
                        }
                        if any(lane[field] != value for field, value in echoed.items()):
                            raise ConflictError("commit does not identify the prepared selection")
                        commit_body = {
                            "step_key": selection["step_key"],
                            "prepare_receipt": selection["prepare_receipt"],
                            "selection_id": selection["selection_id"],
                            "selected_source": selection["selected_source"],
                            "beta": selection["beta"],
                            "selected_action": selection["selected_action"],
                        }
                        commit_digest = bytes_sha256(canonical_bytes(commit_body))
                        expected_receipt = bytes_sha256(canonical_bytes({
                            "commit_digest": commit_digest,
                            "result": item["transition"],
                        }))
                        if lane["commit_receipt"] != expected_receipt:
                            raise ConflictError(
                                "commit_receipt does not bind the selected action and transition"
                            )

                        existing = db.execute(
                            "SELECT * FROM step_commits WHERE idempotency_key=?", (identity,)
                        ).fetchone()
                        if existing is not None:
                            if existing["prepare_idempotency_key"] != prepared["idempotency_key"]:
                                raise ConflictError("commit idempotency key identifies another prepare")
                            if existing["request_sha256"] != item["request_sha"]:
                                raise ConflictError(
                                    "idempotency key was reused with a changed commit payload"
                                )
                            if existing["response_json"] is not None:
                                result = json.loads(existing["response_json"])
                                responses.append(result)
                                if item["sealed"] is not None:
                                    seals.append((key, item["sealed"], prepared["checkpoint_sha256"]))
                                continue
                        else:
                            prior = db.execute(
                                "SELECT idempotency_key FROM step_commits "
                                "WHERE prepare_idempotency_key=?",
                                (prepared["idempotency_key"],),
                            ).fetchone()
                            if prior is not None:
                                raise ConflictError(
                                    "prepared selection was already committed under another key"
                                )

                        body = {
                            "contract_version": expected_contract,
                            "idempotency_key": identity,
                            "step_key": key.as_dict(),
                            "commit_receipt": lane["commit_receipt"],
                            "status": "committed",
                        }
                        result = {**body, "remote_commit_receipt": response_receipt(body)}
                        now = _now()
                        if existing is None:
                            db.execute(
                                "INSERT INTO step_commits VALUES (?,?,?,?,?,?,?,?)",
                                (identity, item["request_sha"], prepared["idempotency_key"],
                                 canonical_json(item["transition"]),
                                 None if item["sealed"] is None
                                 else canonical_json(item["sealed"]),
                                 canonical_json(result), now, now),
                            )
                        else:
                            db.execute(
                                "UPDATE step_commits SET response_json=?,updated_at=? "
                                "WHERE idempotency_key=?",
                                (canonical_json(result), now, identity),
                            )
                        db.execute(
                            "UPDATE step_prepares SET status='committed',updated_at=? "
                            "WHERE idempotency_key=?",
                            (now, prepared["idempotency_key"]),
                        )
                        responses.append(result)
                        if item["sealed"] is not None:
                            seals.append((key, item["sealed"], prepared["checkpoint_sha256"]))

            # Window sealing remains outside the SQLite writer lock.  It is
            # needed only for step 80 and can safely run lane-parallel because
            # every commit record above is already durable and content-addressed.
            if seals:
                workers = min(len(seals), max(1, getattr(self.server.backend, "size", 8)))

                def seal_lane(pair):
                    key, sealed, _checkpoint_sha = pair
                    self._seal_window(key, sealed)

                with self.server.phase_span(
                    seals[0][0], "h200_window_seal_and_publish",
                    batch_id=batch_id, lane_count=len(seals), worker_count=workers,
                    checkpoint_sha256=seals[0][2],
                ):
                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        list(executor.map(seal_lane, seals))
        response = {
            "contract_version": expected_contract,
            "kind": response_kind,
            "batch_id": batch_id,
            "lanes": responses,
        }
        self.server.emit_telemetry({
            "route": "commit_batch", "status": "committed", "batch_id": batch_id,
            "lane_count": len(lanes), "request_bytes": len(canonical_bytes(payload)),
            "response_bytes": len(canonical_bytes(response)) + 1,
            "server_total_ms": (time.perf_counter() - batch_started) * 1000.0,
        })
        return response

    def _commit_batch_memory(
        self,
        payload: dict[str, Any],
        batch_id: str,
        lanes: list[dict[str, Any]],
        parsed: list[dict[str, Any]],
        batch_started: float,
        *,
        expected_contract: str,
        response_kind: str,
    ) -> dict[str, Any]:
        """Commit into the non-recoverable window cache without hot-path I/O."""

        responses: list[dict[str, Any]] = []
        seals: list[tuple[StepKey, dict[str, Any], str]] = []
        with self.server.batch_commit_lock:
            with self.server.window_cache_lock:
                for item in parsed:
                    lane = item["payload"]
                    key = item["key"]
                    identity = item["identity"]
                    prepare_identity = self.server.window_prepares_by_step.get((
                        key.run_id, key.round_id, key.actor_id,
                        key.rollout_window_id, key.rollout_step_index,
                    ))
                    if prepare_identity is None:
                        raise ConflictError("commit has no matching prepare")
                    prepared = self.server.window_prepares[prepare_identity]
                    selection = prepared["response"]
                    echoed = {
                        "step_key": selection["step_key"],
                        "prepare_receipt": selection["prepare_receipt"],
                        "selection_id": selection["selection_id"],
                        "selection_receipt": selection["selection_receipt"],
                    }
                    if any(lane[field] != value for field, value in echoed.items()):
                        raise ConflictError("commit does not identify the prepared selection")
                    commit_body = {
                        "step_key": selection["step_key"],
                        "prepare_receipt": selection["prepare_receipt"],
                        "selection_id": selection["selection_id"],
                        "selected_source": selection["selected_source"],
                        "beta": selection["beta"],
                        "selected_action": selection["selected_action"],
                    }
                    commit_digest = bytes_sha256(canonical_bytes(commit_body))
                    expected_receipt = bytes_sha256(canonical_bytes({
                        "commit_digest": commit_digest,
                        "result": item["transition"],
                    }))
                    if lane["commit_receipt"] != expected_receipt:
                        raise ConflictError(
                            "commit_receipt does not bind the selected action and transition"
                        )

                    existing = self.server.window_commits.get(identity)
                    if existing is not None:
                        if existing["prepare_idempotency_key"] != prepare_identity:
                            raise ConflictError(
                                "commit idempotency key identifies another prepare"
                            )
                        if existing["request_sha256"] != item["request_sha"]:
                            raise ConflictError(
                                "idempotency key was reused with a changed commit payload"
                            )
                        responses.append(existing["response"])
                        completed_key = (
                            key.run_id, key.round_id, key.actor_id,
                            key.rollout_window_id,
                        )
                        if (
                            item["sealed"] is not None
                            and completed_key not in self.server.completed_window_keys
                        ):
                            seals.append((
                                key, item["sealed"], prepared["checkpoint_sha256"]
                            ))
                        continue
                    prior = self.server.window_commit_by_prepare.get(prepare_identity)
                    if prior is not None:
                        raise ConflictError(
                            "prepared selection was already committed under another key"
                        )
                    body = {
                        "contract_version": expected_contract,
                        "idempotency_key": identity,
                        "step_key": key.as_dict(),
                        "commit_receipt": lane["commit_receipt"],
                        "status": "committed",
                    }
                    result = {**body, "remote_commit_receipt": response_receipt(body)}
                    self.server.window_commits[identity] = {
                        "request_sha256": item["request_sha"],
                        "prepare_idempotency_key": prepare_identity,
                        "transition": item["transition"],
                        "sealed": item["sealed"],
                        "response": result,
                    }
                    self.server.window_commit_by_prepare[prepare_identity] = identity
                    prepared["status"] = "committed"
                    responses.append(result)
                    if item["sealed"] is not None:
                        seals.append((
                            key, item["sealed"], prepared["checkpoint_sha256"]
                        ))

            if seals:
                workers = min(len(seals), max(1, getattr(self.server.backend, "size", 8)))
                with self.server.phase_span(
                    seals[0][0], "h200_window_seal_and_publish",
                    batch_id=batch_id, lane_count=len(seals), worker_count=workers,
                    checkpoint_sha256=seals[0][2],
                ):
                    with ThreadPoolExecutor(max_workers=workers) as executor:
                        list(executor.map(
                            lambda pair: self._seal_window_memory(pair[0], pair[1]),
                            seals,
                        ))

        response = {
            "contract_version": expected_contract,
            "kind": response_kind,
            "batch_id": batch_id,
            "lanes": responses,
        }
        self.server.emit_telemetry({
            "route": "commit_batch", "status": "committed_memory",
            "batch_id": batch_id, "lane_count": len(lanes),
            "request_bytes": len(canonical_bytes(payload)),
            "response_bytes": len(canonical_bytes(response)) + 1,
            "server_total_ms": (time.perf_counter() - batch_started) * 1000.0,
        })
        return response

    def _check_order(self, db, key: StepKey, metadata: dict[str, Any]) -> None:
        if key.rollout_step_index == 0:
            return
        previous = db.execute(
            "SELECT p.*,c.transition_json FROM step_prepares p "
            "LEFT JOIN step_commits c ON c.prepare_idempotency_key=p.idempotency_key "
            "WHERE p.run_id=? AND p.round_id=? AND p.actor_id=? AND "
            "p.rollout_window_id=? AND p.rollout_step_index=?",
            (key.run_id, key.round_id, key.actor_id, key.rollout_window_id,
             key.rollout_step_index - 1),
        ).fetchone()
        if previous is None or previous["transition_json"] is None:
            raise ConflictError("previous rollout step is not committed")
        previous_transition = json.loads(previous["transition_json"])
        terminal = previous_transition["done"] or previous_transition["timeout"]
        if terminal:
            if key.episode_id == previous["episode_id"] or key.episode_step_index != 0:
                raise ConflictError("step after done/timeout must begin a new episode at index zero")
        elif (
            key.episode_id != previous["episode_id"]
            or key.episode_step_index != previous["episode_step_index"] + 1
        ):
            raise ConflictError("nonterminal episode identity/index is discontinuous")
        previous_meta = json.loads(previous["metadata_json"])
        if metadata["env_id"] != previous_meta["env_id"]:
            raise ConflictError("environment changed inside one rollout window")
        if not terminal and metadata["episode_assignment"] != previous_meta["episode_assignment"]:
            raise ConflictError("episode assignment changed before reset")

    def _check_order_memory(self, key: StepKey, metadata: dict[str, Any]) -> None:
        """Apply the same episode/order gate to the in-memory lean window."""

        step_identity = (
            key.run_id, key.round_id, key.actor_id,
            key.rollout_window_id, key.rollout_step_index,
        )
        other = self.server.window_prepares_by_step.get(step_identity)
        if other is not None:
            raise ConflictError("rollout step was already prepared under another key")
        if key.rollout_step_index == 0:
            return
        previous_identity = self.server.window_prepares_by_step.get((
            key.run_id, key.round_id, key.actor_id,
            key.rollout_window_id, key.rollout_step_index - 1,
        ))
        if previous_identity is None:
            raise ConflictError("previous rollout step is not prepared")
        previous = self.server.window_prepares[previous_identity]
        commit_identity = self.server.window_commit_by_prepare.get(previous_identity)
        if commit_identity is None:
            raise ConflictError("previous rollout step is not committed")
        previous_transition = self.server.window_commits[commit_identity]["transition"]
        terminal = previous_transition["done"] or previous_transition["timeout"]
        previous_key = previous["key"]
        if terminal:
            if key.episode_id == previous_key.episode_id or key.episode_step_index != 0:
                raise ConflictError(
                    "step after done/timeout must begin a new episode at index zero"
                )
        elif (
            key.episode_id != previous_key.episode_id
            or key.episode_step_index != previous_key.episode_step_index + 1
        ):
            raise ConflictError("nonterminal episode identity/index is discontinuous")
        previous_meta = previous["metadata"]
        if metadata["env_id"] != previous_meta["env_id"]:
            raise ConflictError("environment changed inside one rollout window")
        if not terminal and metadata["episode_assignment"] != previous_meta["episode_assignment"]:
            raise ConflictError("episode assignment changed before reset")

    def _prepare(self, payload: dict[str, Any], raw: bytes) -> dict[str, Any]:
        prepare_started = time.perf_counter()
        key, beta, metadata, arrays = parse_prepare(payload)
        self.server.admit_observation_sources([(key, metadata)])
        request_sha = bytes_sha256(raw)
        identity = payload["idempotency_key"]
        step_identity = canonical_json(key.as_dict())
        wait_for: threading.Event | None = None
        leader = False
        with self.server.operation_lock:
            # This pass only resolves idempotency/order/checkpoint state.  Do
            # not take SQLite's single writer slot before the expensive
            # render/inference path; persistence has its own short transaction
            # after inference below.  This matters when the v1 compatibility
            # endpoint is served by the same multi-gateway deployment as v3.
            with self.server.store.connection() as db:
                existing = db.execute(
                    "SELECT request_sha256,response_json FROM step_prepares WHERE idempotency_key=?",
                    (identity,),
                ).fetchone()
                if existing:
                    if existing["request_sha256"] != request_sha:
                        raise ConflictError("idempotency key was reused with a changed prepare payload")
                    replay = json.loads(existing["response_json"])
                    self.server.emit_telemetry({
                        "route": "prepare", "status": "idempotent_replay",
                        "idempotency_key": identity, "step_key": key.as_dict(),
                        "request_bytes": len(raw),
                        "response_bytes": len(canonical_bytes(replay)) + 1,
                        "server_total_ms": (time.perf_counter() - prepare_started) * 1000.0,
                    })
                    return replay
                id_inflight = self.server.prepare_inflight_ids.get(identity)
                if id_inflight is not None and id_inflight != (step_identity, request_sha):
                    raise ConflictError("idempotency key already has a different prepare in flight")
                inflight = self.server.prepare_inflight.get(step_identity)
                if inflight is not None:
                    inflight_identity, inflight_sha, wait_for = inflight
                    if inflight_identity != identity or inflight_sha != request_sha:
                        raise ConflictError("rollout step already has a different prepare in flight")
                else:
                    self._check_order(db, key, metadata)
                    wait_for = threading.Event()
                    self.server.prepare_inflight[step_identity] = (
                        identity, request_sha, wait_for,
                    )
                    self.server.prepare_inflight_ids[identity] = (step_identity, request_sha)
                    checkpoint_sha = self.server.active_checkpoint(key)
                    leader = True

        # An identical concurrent retry waits for the leader to persist the
        # response. Inference itself is safe to retry, but one HTTP request
        # never silently becomes a second leader after a failed first attempt.
        if not leader:
            if not wait_for.wait(timeout=300):
                raise RuntimeError("timed out waiting for the in-flight prepare")
            with self.server.store.connection() as db:
                existing = db.execute(
                    "SELECT request_sha256,response_json FROM step_prepares WHERE idempotency_key=?",
                    (identity,),
                ).fetchone()
            if existing is None:
                raise RuntimeError("concurrent prepare failed before persistence; retry is safe")
            if existing["request_sha256"] != request_sha:
                raise ConflictError("idempotency key was reused with a changed prepare payload")
            replay = json.loads(existing["response_json"])
            self.server.emit_telemetry({
                "route": "prepare", "status": "concurrent_replay",
                "idempotency_key": identity, "step_key": key.as_dict(),
                "request_bytes": len(raw),
                "response_bytes": len(canonical_bytes(replay)) + 1,
                "server_total_ms": (time.perf_counter() - prepare_started) * 1000.0,
            })
            return replay

        try:
            inference_started = time.perf_counter()
            with self.server.phase_span(
                key, "h200_image3f_render_and_student_inference",
                observation_source=metadata["observation_source"],
            ) as phase_details:
                result = self.server.backend.infer(metadata, arrays, checkpoint_sha)
                phase_details.update(result.telemetry)
            inference_ms = (time.perf_counter() - inference_started) * 1000.0
            selection_started = time.perf_counter()
            with self.server.phase_span(key, "h200_beta_action_selection", beta=beta) as phase_details:
                student_26 = np.concatenate(
                    (result.student_action_11, arrays["homie_action"]), dtype=np.float32
                )
                selection_id = bytes_sha256(canonical_bytes({
                    "contract_version": STEP_CONTRACT,
                    "idempotency_key": identity,
                    "step_key": key.as_dict(),
                    "capture_digest": payload["capture_digest"],
                    "checkpoint_sha256": checkpoint_sha,
                }))
                selection_draw = int(selection_id[:16], 16) / float(1 << 64)
                use_teacher = selection_draw < beta
                selected = arrays["teacher_action"].copy() if use_teacher else student_26.copy()
                body = {
                    "contract_version": STEP_CONTRACT,
                    "idempotency_key": identity,
                    "step_key": key.as_dict(),
                    "prepare_receipt": payload["prepare_receipt"],
                    "capture_digest": payload["capture_digest"],
                    "beta": beta,
                    "selection_id": selection_id,
                    "selection_draw": selection_draw,
                    "selected_source": "teacher" if use_teacher else "student",
                    "student_action_11": encode_array(result.student_action_11),
                    "selected_action": encode_array(np.ascontiguousarray(selected, dtype=np.float32)),
                    "observation_digest": result.observation_digest,
                }
                response = {**body, "selection_receipt": response_receipt(body)}
                phase_details["selected_source"] = response["selected_source"]
            selection_ms = (time.perf_counter() - selection_started) * 1000.0
            persistence_started = time.perf_counter()
            with self.server.phase_span(key, "h200_prepare_artifact_persistence") as phase_details:
                artifact = self.server.root / "prepared" / f"{identity}.npz"
                temporary = artifact.with_suffix(".npz.part")
                # Persist only tensors required to assemble the learner chunk.
                # The capture digest/RTX audit binds the full request, but
                # recompressing metric depth, raw depth variants, ray-traced
                # source tiles, and the fixed anchor at every lane-step added
                # no learner information and dominated the online critical path.
                artifact_arrays = {
                    name: arrays[name]
                    for name in (
                        "actor_obs", "teacher_obs", "critic_obs", "homie_obs",
                        "gt_obj_pos", "teacher_action", "homie_action",
                    )
                }
                for view in VIEWS:
                    artifact_arrays[f"depth_ctrl__{view}"] = arrays[f"depth_ctrl__{view}"]
                    artifact_arrays[f"rgb_robot__{view}"] = arrays[f"rgb_robot__{view}"]
                artifact_arrays.update({
                    f"rgb_gen__{view}": result.policy_rgb[view] for view in VIEWS
                })
                artifact_arrays.update({
                    "student_action_11": result.student_action_11,
                    "student_action": student_26,
                    "behavior_action": np.ascontiguousarray(selected, dtype=np.float32),
                    "behavior_is_teacher": np.array(use_teacher, dtype=np.bool_),
                    "beta": np.array(beta, dtype=np.float32),
                })
                with temporary.open("wb") as stream:
                    # Online data is already image-like/high entropy. Deflate
                    # consumed most of the critical path for little space
                    # reduction; ZIP_STORED keeps the same NPZ learner format.
                    np.savez(stream, **artifact_arrays)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, artifact)
                phase_details["artifact_bytes"] = artifact.stat().st_size
                now = _now()
                with self.server.operation_lock:
                    with self.server.store.transaction() as db:
                        try:
                            db.execute(
                                "INSERT INTO step_prepares VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                (
                                    identity, request_sha, key.run_id, key.round_id, key.actor_id,
                                    key.rollout_window_id, key.rollout_step_index, key.episode_id,
                                    key.episode_step_index, payload["capture_digest"],
                                    payload["prepare_receipt"], canonical_json(metadata),
                                    canonical_json(response), str(artifact), checkpoint_sha,
                                    "prepared", now, now,
                                ),
                            )
                        except Exception:
                            artifact.unlink(missing_ok=True)
                            raise
            persistence_ms = (time.perf_counter() - persistence_started) * 1000.0
            self.server.emit_telemetry({
                "route": "prepare", "status": "prepared",
                "idempotency_key": identity, "step_key": key.as_dict(),
                "observation_source": metadata["observation_source"],
                "selected_source": response["selected_source"],
                "request_bytes": len(raw),
                "response_bytes": len(canonical_bytes(response)) + 1,
                "inference_ms": inference_ms,
                "selection_ms": selection_ms,
                "persistence_ms": persistence_ms,
                "server_total_ms": (time.perf_counter() - prepare_started) * 1000.0,
                **result.telemetry,
            })
            return response
        finally:
            with self.server.operation_lock:
                current = self.server.prepare_inflight.get(step_identity)
                if current is not None and current[2] is wait_for:
                    del self.server.prepare_inflight[step_identity]
                    self.server.prepare_inflight_ids.pop(identity, None)
                    wait_for.set()

    def _commit(
        self,
        payload: dict[str, Any],
        raw: bytes,
        *,
        expected_contract: str = STEP_CONTRACT,
    ) -> dict[str, Any]:
        commit_started = time.perf_counter()
        required = {
            "contract_version", "idempotency_key", "step_key", "prepare_receipt",
            "selection_id", "selection_receipt", "commit_receipt", "transition",
            "sealed_window",
        }
        from .contracts import exact_keys, identifier
        exact_keys(payload, required)
        if payload["contract_version"] != expected_contract:
            raise ContractError("contract_version mismatch")
        identity = identifier(payload["idempotency_key"], "idempotency_key")
        key = StepKey.parse(payload["step_key"])
        for field in ("prepare_receipt", "selection_id", "selection_receipt", "commit_receipt"):
            sha256_digest(payload[field], field)
        transition = validate_transition(payload["transition"])
        if key.rollout_step_index == 80:
            if payload["sealed_window"] is None:
                raise ContractError("rollout step 80 requires sealed_window")
            sealed = validate_sealed_window(
                payload["sealed_window"], key, expected_contract=expected_contract
            )
        else:
            if payload["sealed_window"] is not None:
                raise ContractError("sealed_window is allowed only at rollout step 80")
            sealed = None
        request_sha = bytes_sha256(raw)
        wait_for: threading.Event | None = None
        leader = False
        with self.server.operation_lock:
            with self.server.store.transaction() as db:
                prepared = db.execute(
                    "SELECT * FROM step_prepares WHERE run_id=? AND round_id=? AND actor_id=? "
                    "AND rollout_window_id=? AND rollout_step_index=?",
                    (key.run_id, key.round_id, key.actor_id, key.rollout_window_id,
                     key.rollout_step_index),
                ).fetchone()
                if prepared is None:
                    raise ConflictError("commit has no matching prepare")
                response = json.loads(prepared["response_json"])
                echoed = {
                    "step_key": response["step_key"],
                    "prepare_receipt": response["prepare_receipt"],
                    "selection_id": response["selection_id"],
                    "selection_receipt": response["selection_receipt"],
                }
                if any(payload[field] != value for field, value in echoed.items()):
                    raise ConflictError("commit does not identify the prepared selection")
                commit_body = {
                    "step_key": response["step_key"],
                    "prepare_receipt": response["prepare_receipt"],
                    "selection_id": response["selection_id"],
                    "selected_source": response["selected_source"],
                    "beta": response["beta"],
                    "selected_action": response["selected_action"],
                }
                commit_digest = bytes_sha256(canonical_bytes(commit_body))
                expected_commit_receipt = bytes_sha256(canonical_bytes({
                    "commit_digest": commit_digest,
                    "result": transition,
                }))
                if payload["commit_receipt"] != expected_commit_receipt:
                    raise ConflictError("commit_receipt does not bind the selected action and transition")
                existing = db.execute(
                    "SELECT * FROM step_commits WHERE idempotency_key=?", (identity,)
                ).fetchone()
                if existing:
                    if existing["prepare_idempotency_key"] != prepared["idempotency_key"]:
                        raise ConflictError("commit idempotency key identifies another prepare")
                    if existing["request_sha256"] != request_sha:
                        raise ConflictError("idempotency key was reused with a changed commit payload")
                    if existing["response_json"] is not None:
                        replay = json.loads(existing["response_json"])
                        self.server.emit_telemetry({
                            "route": "commit", "status": "idempotent_replay",
                            "idempotency_key": identity, "step_key": key.as_dict(),
                            "request_bytes": len(raw),
                            "response_bytes": len(canonical_bytes(replay)) + 1,
                            "server_total_ms": (time.perf_counter() - commit_started) * 1000.0,
                        })
                        return replay
                    wait_for = self.server.commit_inflight.get(identity)
                    if wait_for is None:
                        wait_for = threading.Event()
                        self.server.commit_inflight[identity] = wait_for
                        leader = True
                else:
                    prior = db.execute(
                        "SELECT idempotency_key FROM step_commits WHERE prepare_idempotency_key=?",
                        (prepared["idempotency_key"],),
                    ).fetchone()
                    if prior is not None:
                        raise ConflictError("prepared selection was already committed under another key")
                    now = _now()
                    db.execute(
                        "INSERT INTO step_commits VALUES (?,?,?,?,?,?,?,?)",
                        (identity, request_sha, prepared["idempotency_key"], canonical_json(transition),
                         None if sealed is None else canonical_json(sealed), None, now, now),
                    )
                    db.execute(
                        "UPDATE step_prepares SET status='committed',updated_at=? "
                        "WHERE idempotency_key=?", (now, prepared["idempotency_key"]),
                    )
                    wait_for = threading.Event()
                    self.server.commit_inflight[identity] = wait_for
                    leader = True

        # A same-key retry waits for the leader. If the process crashed after
        # the durable commit row but before its response, the first retry sees
        # no in-memory leader and safely resumes idempotent seal/publication.
        if not leader:
            if not wait_for.wait(timeout=600):
                raise RuntimeError("timed out waiting for the in-flight commit")
            with self.server.store.connection() as db:
                finished = db.execute(
                    "SELECT request_sha256,response_json FROM step_commits WHERE idempotency_key=?",
                    (identity,),
                ).fetchone()
            if finished is None or finished["response_json"] is None:
                raise RuntimeError("concurrent commit failed before response persistence; retry is safe")
            if finished["request_sha256"] != request_sha:
                raise ConflictError("idempotency key was reused with a changed commit payload")
            replay = json.loads(finished["response_json"])
            self.server.emit_telemetry({
                "route": "commit", "status": "concurrent_replay",
                "idempotency_key": identity, "step_key": key.as_dict(),
                "request_bytes": len(raw),
                "response_bytes": len(canonical_bytes(replay)) + 1,
                "server_total_ms": (time.perf_counter() - commit_started) * 1000.0,
            })
            return replay

        try:
            # Independent actor windows deliberately seal outside the global
            # operation lock. The sink is content-addressed/idempotent, while
            # commit_inflight prevents duplicate same-key work in one process.
            if sealed is not None:
                with self.server.phase_span(key, "h200_window_seal_and_publish"):
                    self._seal_window(key, sealed)

            body = {
                "contract_version": expected_contract,
                "idempotency_key": identity,
                "step_key": key.as_dict(),
                "commit_receipt": payload["commit_receipt"],
                "status": "committed",
            }
            result = {**body, "remote_commit_receipt": response_receipt(body)}
            with self.server.operation_lock:
                with self.server.store.transaction() as db:
                    db.execute(
                        "UPDATE step_commits SET response_json=?,updated_at=? WHERE idempotency_key=?",
                        (canonical_json(result), _now(), identity),
                    )
            self.server.emit_telemetry({
                "route": "commit", "status": "committed",
                "idempotency_key": identity, "step_key": key.as_dict(),
                "request_bytes": len(raw),
                "response_bytes": len(canonical_bytes(result)) + 1,
                "persistence_ms": (time.perf_counter() - commit_started) * 1000.0,
                "server_total_ms": (time.perf_counter() - commit_started) * 1000.0,
            })
            return result
        finally:
            with self.server.operation_lock:
                current = self.server.commit_inflight.get(identity)
                if current is wait_for:
                    del self.server.commit_inflight[identity]
                    wait_for.set()

    def _seal_window_memory(self, key: StepKey, sealed: dict[str, Any]) -> None:
        """Publish one complete learner chunk from the lean in-memory ledger."""

        rows: list[dict[str, Any]] = []
        loaded: list[dict[str, np.ndarray]] = []
        with self.server.window_cache_lock:
            for index in range(81):
                prepare_identity = self.server.window_prepares_by_step.get((
                    key.run_id, key.round_id, key.actor_id,
                    key.rollout_window_id, index,
                ))
                if prepare_identity is None:
                    raise ConflictError(
                        "cannot seal until exactly rollout steps 0..80 are prepared"
                    )
                prepared = self.server.window_prepares[prepare_identity]
                commit_identity = self.server.window_commit_by_prepare.get(
                    prepare_identity
                )
                if commit_identity is None:
                    raise ConflictError(
                        "cannot seal until exactly rollout steps 0..80 are committed"
                    )
                committed = self.server.window_commits[commit_identity]
                prepared_key = prepared["key"]
                rows.append({
                    "rollout_step_index": prepared_key.rollout_step_index,
                    "episode_id": prepared_key.episode_id,
                    "episode_step_index": prepared_key.episode_step_index,
                    "round_id": prepared_key.round_id,
                    "metadata_json": canonical_json(prepared["metadata"]),
                    "transition_json": canonical_json(committed["transition"]),
                    "response_json": canonical_json(prepared["response"]),
                    "prepare_receipt": prepared["prepare_receipt"],
                    "capture_digest": prepared["capture_digest"],
                    "checkpoint_sha256": prepared["checkpoint_sha256"],
                })
                loaded.append(prepared["artifact_arrays"])
        self._seal_window_records(key, sealed, rows, loaded)
        with self.server.window_cache_lock:
            self.server.completed_window_keys.add((
                key.run_id, key.round_id, key.actor_id, key.rollout_window_id,
            ))

    def _seal_window(self, key: StepKey, sealed: dict[str, Any]) -> None:
        with self.server.store.connection() as db:
            rows = list(db.execute(
                "SELECT p.*,c.transition_json FROM step_prepares p JOIN step_commits c "
                "ON c.prepare_idempotency_key=p.idempotency_key "
                "WHERE p.run_id=? AND p.round_id=? "
                "AND p.actor_id=? AND p.rollout_window_id=? ORDER BY p.rollout_step_index",
                (key.run_id, key.round_id, key.actor_id, key.rollout_window_id),
            ))
        loaded = [load_prepared_artifact(row["artifact_path"]) for row in rows]
        self._seal_window_records(key, sealed, rows, loaded)

    def _seal_window_records(
        self,
        key: StepKey,
        sealed: dict[str, Any],
        rows: list[Any],
        loaded: list[dict[str, np.ndarray]],
    ) -> None:
        """Validate identity/outcome audit data and publish learner tensors once."""

        if len(rows) != 81 or [row["rollout_step_index"] for row in rows] != list(range(81)):
            raise ConflictError("cannot seal until exactly rollout steps 0..80 are committed")
        episode_ids = [row["episode_id"] for row in rows]
        bindings = sealed["provenance"]["episode_bindings"]
        actual_starts = []
        for index, episode_id in enumerate(episode_ids):
            if index == 0 or episode_id != episode_ids[index - 1]:
                meta = json.loads(rows[index]["metadata_json"])
                actual_starts.append({
                    "episode_id": episode_id,
                    "first_rollout_step_index": index,
                    "use_raytraced": meta["episode_assignment"]["use_raytraced"],
                    "observation_source": meta["observation_source"],
                    "anchor_sha256": meta["episode_assignment"]["anchor_sha256"],
                })
        if actual_starts != bindings:
            raise ConflictError("sealed_window episode bindings differ from committed steps")

        actual_meta = [json.loads(row["metadata_json"]) for row in rows]
        actual_transitions = [json.loads(row["transition_json"]) for row in rows]
        actual_selections = [json.loads(row["response_json"]) for row in rows]
        actual_commit_digests = []
        actual_commit_receipts = []
        for selection, transition in zip(actual_selections, actual_transitions):
            commit_body = {
                "step_key": selection["step_key"],
                "prepare_receipt": selection["prepare_receipt"],
                "selection_id": selection["selection_id"],
                "selected_source": selection["selected_source"],
                "beta": selection["beta"],
                "selected_action": selection["selected_action"],
            }
            commit_digest = bytes_sha256(canonical_bytes(commit_body))
            actual_commit_digests.append(commit_digest)
            actual_commit_receipts.append(bytes_sha256(canonical_bytes({
                "commit_digest": commit_digest,
                "result": transition,
            })))
        per_step_expected = {
            "episode_ids": episode_ids,
            "episode_step_indices": [row["episode_step_index"] for row in rows],
            "env_ids": [item["env_id"] for item in actual_meta],
            "seqs": [item["seq"] for item in actual_meta],
            "round_ids": [row["round_id"] for row in rows],
            "observation_sources": [item["observation_source"] for item in actual_meta],
            "dones": [item["done"] for item in actual_transitions],
            "timeouts": [item["timeout"] for item in actual_transitions],
            "rewards": [float(item["reward"]) for item in actual_transitions],
            "behavior_sources": [item["selected_source"] for item in actual_selections],
            "prepare_receipts": [row["prepare_receipt"] for row in rows],
            "capture_digests": [row["capture_digest"] for row in rows],
            "selection_ids": [item["selection_id"] for item in actual_selections],
            "commit_digests": actual_commit_digests,
            "commit_receipts": actual_commit_receipts,
        }
        for field, expected in per_step_expected.items():
            if sealed[field] != expected:
                raise ConflictError(f"sealed_window {field} differs from committed H200 ledger")

        # Lean learner chunk: these are the only arrays read by
        # raven_learner_worker.load_batch.  The complete identity, selected
        # action, reward/done/timeout and receipt audit remains in SQLite and
        # in the RTX seal manifest; duplicating it in the gradient input was
        # expensive and algorithmically redundant.
        arrays: dict[str, np.ndarray] = {
            "learner_chunk_format": np.array("roborender.dagger.learner.v1"),
            "gt_actions": np.stack([item["teacher_action"] for item in loaded]).astype(np.float32),
            "obs_actor_obs": np.stack([item["actor_obs"] for item in loaded]).astype(np.float32),
        }
        for view in VIEWS:
            arrays[f"rgb_gen__{view}"] = np.stack(
                [item[f"rgb_gen__{view}"] for item in loaded]
            ).astype(np.float32, copy=False)
        chunk_id = f"d{key.round_id}-{key.actor_id}-{key.rollout_window_id}"
        final = self.server.root / "sealed" / f"{chunk_id}.npz"
        if not final.exists():
            temporary = final.with_suffix(".npz.part")
            encoded = io.BytesIO()
            payload = None
            try:
                np.savez(encoded, **arrays)
                payload = encoded.getbuffer()
                digest = bytes_sha256(payload)
                with temporary.open("wb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, final)
            finally:
                if payload is not None:
                    payload.release()
                encoded.close()
        else:
            # This path is only for a process-local idempotent recovery.  The
            # v4 in-memory ledger suppresses normal duplicate finalization.
            digest = file_sha256(final)
        checkpoints = {row["checkpoint_sha256"] for row in rows}
        if len(checkpoints) != 1:
            raise ConflictError("checkpoint changed inside an 81-step barrier")
        reservation = ChunkReservation(
            chunk_id=chunk_id,
            sha256=digest,
            size_bytes=final.stat().st_size,
            run_id=key.run_id,
            round_id=key.round_id,
            actor_id=key.actor_id,
            environment_id=int(actual_meta[0]["env_id"]),
            rollout_window_id=key.rollout_window_id,
            first_step_id=0,
            last_step_id=80,
            checkpoint_sha256=next(iter(checkpoints)),
            teacher_checkpoint_sha256=sealed["provenance"]["teacher_checkpoint_sha256"],
            sealed_window_sha256=bytes_sha256(canonical_bytes(sealed)),
        )
        # The ephemeral fast path drops per-step records after publication.
        # Retain only verified episode/window identities for leakage checks;
        # this performs no additional array load, hash, stack or compression.
        from .mse_protocol import persist_chunk_provenance
        persist_chunk_provenance(self.server.store, reservation, sealed)
        self.server.trajectory_sink.publish(final, reservation)


def main() -> int:
    project = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18620)
    parser.add_argument("--state-root", type=Path, default=project / "runtime/dagger_services")
    parser.add_argument("--token-file", type=Path, default=project / "runtime/secrets/dagger_services.token")
    parser.add_argument("--trajectory-url", default="http://127.0.0.1:18611")
    parser.add_argument("--image3f-port", type=int, default=18500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--image3f-ports",
        help="comma-separated loopback Image3F ports, one per pooled GPU worker",
    )
    parser.add_argument(
        "--devices",
        help="comma-separated CUDA devices, one per pooled GPU worker",
    )
    parser.add_argument("--fake", action="store_true")
    parser.add_argument(
        "--telemetry-jsonl",
        type=Path,
        help="private append-only server timing log; does not change the wire schema",
    )
    parser.add_argument(
        "--phase-telemetry-jsonl",
        type=Path,
        help="append-only start/end/error wall-clock phase events",
    )
    parser.add_argument(
        "--local-test-sink-root",
        type=Path,
        help=(
            "publish sealed chunks to an isolated local directory; allowed only "
            "with --fake so conformance tests cannot enter the production trajectory service"
        ),
    )
    parser.add_argument(
        "--shared-local-trajectory",
        action="store_true",
        help=(
            "publish sealed H200 learner chunks by hard link into the shared "
            "state root; avoids a redundant loopback upload and second full scan"
        ),
    )
    parser.add_argument(
        "--ephemeral-window-cache",
        action="store_true",
        help=(
            "keep incomplete 81-step learner windows and idempotency state in RAM; "
            "publish only at final seal and fail closed on any gateway restart"
        ),
    )
    args = parser.parse_args()
    token = args.token_file.read_text(encoding="utf-8").strip()
    if len(token) < 32:
        raise SystemExit("bearer token must contain at least 32 characters")
    if args.host not in ("127.0.0.1", "::1", "localhost"):
        raise SystemExit("render/student boundary must bind only to loopback")
    if args.local_test_sink_root is not None and not args.fake:
        raise SystemExit("--local-test-sink-root is allowed only with --fake")
    if (args.devices is None) != (args.image3f_ports is None):
        raise SystemExit("--devices and --image3f-ports must be supplied together")
    if args.fake:
        backend = FakeRenderStudentBackend()
        backend.test_checkpoint_sha256 = "0" * 64
    else:
        if args.devices is None:
            worker_specs = [(args.device, args.image3f_port)]
        else:
            devices = [item.strip() for item in args.devices.split(",") if item.strip()]
            try:
                ports = [int(item.strip()) for item in args.image3f_ports.split(",") if item.strip()]
            except ValueError as exc:
                raise SystemExit("--image3f-ports must contain integers") from exc
            if not devices or len(devices) != len(ports):
                raise SystemExit("--devices and --image3f-ports must have equal nonzero lengths")
            if len(set(devices)) != len(devices) or len(set(ports)) != len(ports):
                raise SystemExit("pooled devices and Image3F ports must be unique")
            if any(port < 1024 or port > 65535 for port in ports):
                raise SystemExit("pooled Image3F ports must be 1024..65535")
            worker_specs = list(zip(devices, ports, strict=True))
        workers = [
            LiveRenderStudentBackend(
                checkpoint_resolver=lambda digest: content_addressed_checkpoint(
                    args.state_root, digest
                ),
                policy_spec=project / "checkpoints/students/bootstrap/g1_3view_student_bootstrap_v1.policy_spec.json",
                image3f_port=port,
                device=device,
                # The registry/learner verifies content before publishing a
                # digest. V4 keeps checkpoint activation on the hot path lean;
                # The diagnostic v1/v3 path retains independent per-gateway hashing.
                verify_checkpoint_hash=not args.ephemeral_window_cache,
            )
            for device, port in worker_specs
        ]
        # Always retain the explicit exclusive-slot dispatcher, including the
        # common one-gateway/one-GPU deployment.  The HTTP batch handler may
        # receive several lanes at once, but a CUDA/Image3F backend is not a
        # multiplexable socket or execution context.
        backend = RenderStudentBackendPool(workers)
    if args.local_test_sink_root is not None:
        sink = LocalTrajectorySink(args.local_test_sink_root)
    elif args.shared_local_trajectory:
        sink = SharedFilesystemTrajectorySink(args.state_root)
    else:
        sink = TrajectorySink(args.trajectory_url, token)
    server = RenderStudentHTTPServer(
        (args.host, args.port), Handler, root=args.state_root, token=token,
        backend=backend, trajectory_sink=sink, telemetry_path=args.telemetry_jsonl,
        phase_telemetry_path=args.phase_telemetry_jsonl,
        ephemeral_window_cache=args.ephemeral_window_cache,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
