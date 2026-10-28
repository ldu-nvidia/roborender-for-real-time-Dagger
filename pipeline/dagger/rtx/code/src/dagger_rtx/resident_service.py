"""One main-thread native owner, with durable sequential phase commands.

This module is CPU-only. HTTP threads never call Isaac. A process crash is not
a simulator checkpoint: a used state directory is deliberately non-resumable.
"""
from __future__ import annotations

import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import threading
import time
import uuid

from .contracts import canonical_json
from .durable import atomic_write_json
from .phase_modes import mode_for, resolve

SCHEMA = "roborender.isaac-resident.v1"
MAX_REQUEST = 1 << 20
OPERATIONS = {"capture_reset", "capture_initial_b2", "diagnose_scene_pressure", "probe_training_phase", "probe_continuous_training", "prepare_scene", "collect_round", "collect_mse_round", "collect_production_segment", "collect_mse_validation", "evaluate_pair", "evaluate_explicit_panel", "evaluate_rendered_continuation", "evaluate_scaled_rendered_panel", "evaluate_raven_pooled_rendered", "evaluate_scaled_teacher_panel", "shutdown"}
OPERATIONS.add('run_phase')


def _bounded_evaluation_progress(output):
    """Observe completed trace rows when the scaled evaluator has no progress.json.

    This is metadata only, never scientific result aggregation or simulator I/O.
    Ignore a partial final line, and read at most1MiB from the current cohort.
    """
    root = Path(output)
    if root.is_symlink():
        raise ValueError('evaluation progress root alias forbidden')
    cohorts = sorted(root.glob('cohort-[0-9][0-9][0-9]'))
    if len(cohorts) > 295 or any(path.is_symlink() or not path.is_dir() for path in cohorts):
        raise ValueError('invalid evaluation progress cohort inventory')
    if not cohorts:
        return None
    path = cohorts[-1] / 'steps.jsonl'
    result = dict(phase='evaluation_initialization', cohort_index=int(cohorts[-1].name[-3:]))
    if not path.exists():
        return result
    if path.is_symlink() or not path.is_file():
        raise ValueError('evaluation progress trace alias forbidden')
    size = path.stat().st_size
    offset = max(0, size - MAX_REQUEST)
    with path.open('rb') as stream:
        stream.seek(offset)
        lines = stream.read(MAX_REQUEST).split(b'\n')
    if offset:
        lines = lines[1:]  # The first bounded fragment need not be a whole row.
    lines = lines[:-1]     # Only newline-committed records represent progress.
    if not lines:
        return result
    value = json.loads(lines[-1])
    if not isinstance(value, dict):
        raise ValueError('invalid evaluation progress record')
    step = value.get('step')
    if type(step) is not int or step < 0:
        raise ValueError('invalid evaluation progress step')
    result.update(phase='evaluation_rollout', vector_step=step + 1)
    return result


def bounded_native_progress(output):
    if output is None:
        return None
    path = Path(output) / "progress.json"
    try:
        if path.stat().st_size > MAX_REQUEST:
            raise ValueError("oversize native progress")
        value = json.loads(path.read_bytes())
        return {key: value[key] for key in ("phase", "status", "run_id", "round_id", "vector_step", "total_vector_steps",
            "batch_size", "completed", "total", "counts", "elapsed_ms", "event", "sequence",
            "phase_completion_counts", "updated_at_utc", "arm", "cohorts_completed",
            "completed_episodes", "successes", "expected_episodes") if key in value}
    except FileNotFoundError:
        try:
            return _bounded_evaluation_progress(output)
        except (OSError, ValueError, TypeError) as exc:
            return {"phase": "progress_snapshot_unavailable", "error_type": type(exc).__name__}
    except (OSError, ValueError, TypeError) as exc:
        return {"phase": "progress_snapshot_unavailable", "error_type": type(exc).__name__}


def validate_command(value):
    keys = {"schema", "worker_instance_id", "command_id", "operation", "payload"}
    if not isinstance(value, dict) or set(value) != keys or value["schema"] != SCHEMA:
        raise ValueError("invalid resident command fields/schema")
    if not re.fullmatch(r"[a-f0-9]{32}", value["worker_instance_id"] or ""):
        raise ValueError("invalid worker instance")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,95}", value["command_id"] or ""):
        raise ValueError("invalid phase command identity")
    if value["operation"] not in OPERATIONS or not isinstance(value["payload"], dict):
        raise ValueError("invalid phase operation/payload")
    if value["operation"] == "shutdown" and value["payload"]:
        raise ValueError("shutdown takes no payload")
    if value['operation'] == 'run_phase':
        resolve(value['payload'])
    # Snapshot accepted parameters: a caller must not mutate the queued command
    # after its digest and mode/operation admission have been checked.
    return json.loads(canonical_json(value))


class ResidentOwner:
    """All operations are serialized on the thread that constructed the backend."""
    def __init__(self, root, backend, *, gpu_uuid, source_sha256):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = self.root / "resident-state.json"
        if self.path.exists():
            # Preserve the old file; never re-execute even a queued native job.
            raise RuntimeError("existing resident state is CRASH_UNCERTAIN; use a fresh identity")
        self.backend = backend
        self.main_thread = threading.get_ident()
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
        self.state = {"schema": SCHEMA, "worker_instance_id": uuid.uuid4().hex,
                      "native_pid": os.getpid(), "gpu_uuid": gpu_uuid,
                      "source_sha256": source_sha256, "native_initializations": 1,
                      "phase": "IDLE", "commands": {}, "active_command_id": None,
                      "mode": "IDLE",
                      "created_at_ns": time.time_ns()}
        self._save()

    def _save(self):
        self.state["updated_at_ns"] = time.time_ns()
        atomic_write_json(self.path, self.state, mode=0o644)

    def status(self):
        with self.lock:
            value = {k: v for k, v in self.state.items() if k != "commands"}
            value["current_event"] = getattr(self, "last_event", None)
        if hasattr(self.backend, "progress"):
            value["native_progress"] = self.backend.progress()
        if hasattr(self.backend, 'phase_capabilities'):
            value['phase_capabilities'] = self.backend.phase_capabilities()
        return value

    def submit(self, command):
        value = validate_command(command)
        digest = hashlib.sha256(canonical_json(value)).hexdigest()
        with self.condition:
            if value["worker_instance_id"] != self.state["worker_instance_id"]:
                raise ValueError("resident instance mismatch")
            previous = self.state["commands"].get(value["command_id"])
            if previous is not None:
                if previous["payload_sha256"] != digest:
                    raise ValueError("command idempotency conflict")
                return previous["acceptance"]
            if self.state["phase"] != "IDLE":
                raise ValueError("native owner is busy or uncertain")
            # Backend validates all paths/configs/capabilities before acceptance.
            self.backend.validate(value["operation"], value["payload"])
            if hasattr(self.backend, 'validate_command_identity'):
                self.backend.validate_command_identity(value)
            acceptance = {"schema": SCHEMA, "worker_instance_id": self.state["worker_instance_id"],
                          "command_id": value["command_id"], "payload_sha256": digest,
                          "status": "accepted"}
            self.state["commands"][value["command_id"]] = {
                "command": value, "payload_sha256": digest, "acceptance": acceptance,
                "status": "QUEUED", "accepted_at_ns": time.time_ns()}
            self.state.update(phase="QUEUED", active_command_id=value["command_id"])
            self.state['mode'] = mode_for(value['operation'], value['payload'])
            self._save()
            self.condition.notify_all()
            return acceptance

    def result(self, command_id):
        with self.lock:
            row = self.state["commands"].get(command_id)
            if row is None:
                raise KeyError(command_id)
            return {k: v for k, v in row.items() if k not in {"command", "acceptance"}}

    def run_next(self, timeout=.5):
        if threading.get_ident() != self.main_thread:
            raise RuntimeError("Isaac may run only on its native owner thread")
        with self.condition:
            if self.state["phase"] == "IDLE":
                self.condition.wait(timeout)
            if self.state["phase"] != "QUEUED":
                return self.state["phase"] != "STOPPED"
            cid = self.state["active_command_id"]
            row = self.state["commands"][cid]
            row.update(status="RUNNING", started_at_ns=time.time_ns())
            self.state["phase"] = "RUNNING"
            self._save()
        started = time.monotonic()
        self.report("phase_start", command_id=cid, operation=row["command"]["operation"])
        try:
            receipt = self.backend.run(row["command"]["operation"], row["command"]["payload"], self.report)
            if not isinstance(receipt, dict) or receipt.get("complete") is not True:
                raise RuntimeError("native phase has no complete receipt")
            canonical_json(receipt)
        except BaseException as exc:
            with self.condition:
                row.update(status="CRASH_UNCERTAIN", error_type=type(exc).__name__,
                           finished_at_ns=time.time_ns(), seconds=time.monotonic()-started)
                self.state["phase"] = "CRASH_UNCERTAIN"
                self._save()
            self.report("phase_failed", command_id=cid, error_type=type(exc).__name__)
            # Do not run a following phase or retry a potentially native step.
            raise
        with self.condition:
            row.update(status="COMPLETE", receipt=receipt, finished_at_ns=time.time_ns(),
                       seconds=time.monotonic()-started)
            self.state.update(phase="STOPPED" if row["command"]["operation"] == "shutdown" else "IDLE",
                              active_command_id=None, mode="IDLE")
            self._save()
        self.report("phase_end", command_id=cid, seconds=row["seconds"], profile=receipt.get("profile"), receipt=receipt)
        return self.state["phase"] != "STOPPED"

    def report(self, phase, **fields):
        row = {"schema": SCHEMA, "phase": phase, "native_pid": os.getpid(),
               "worker_instance_id": self.state["worker_instance_id"], "at_ns": time.time_ns(), **fields}
        print(canonical_json(row).decode(), flush=True)
        with self.lock, (self.root / "resident-events.jsonl").open("ab") as stream:
            stream.write(canonical_json(row)+b"\n")
            self.last_event = {key: row[key] for key in ("phase", "mode", "command_id", "operation", "at_ns", "arm",
                "old_profile", "profile", "seconds", "scene_initializations", "native_initializations", "error_type") if key in row}
        atomic_write_json(self.root / "resident-progress.json", row, mode=0o644)


def serve(owner, *, port, token_file):
    path = Path(token_file)
    if path.is_symlink() or path.stat().st_mode & 0o077 or not path.is_file():
        raise RuntimeError("resident bearer token must be an owner-only regular file")
    token = path.read_text().strip()
    if not token or "\n" in token:
        raise RuntimeError("invalid resident token")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def log_message(self, *args):
            pass
        def send(self, code, value):
            data = canonical_json(value)
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)
            self.close_connection = True
        def authorized(self):
            supplied = self.headers.get("Authorization", "")
            if not hmac.compare_digest(supplied, "Bearer "+token):
                self.send(401, {"error": "unauthorized"})
                return False
            return True
        def do_GET(self):
            if not self.authorized():
                return
            if self.path in {"/ready", "/v1/status"}:
                status = owner.status()
                self.send(200, {"ok": status["phase"] not in {"CRASH_UNCERTAIN", "STOPPED"},
                                "role": "isaac_resident", **status})
            elif self.path.startswith("/v1/commands/"):
                try:
                    self.send(200, owner.result(self.path.rsplit("/", 1)[-1]))
                except KeyError:
                    self.send(404, {"error": "unknown_command"})
            else:
                self.send(404, {"error": "unknown_route"})
        def do_POST(self):
            if not self.authorized():
                return
            if self.path != "/v1/commands":
                self.send(404, {"error": "unknown_route"})
                return
            try:
                if self.headers.get("Content-Type") != "application/json" or self.headers.get("Transfer-Encoding"):
                    raise ValueError("JSON Content-Length framing required")
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_REQUEST:
                    raise ValueError("request size limit")
                self.connection.settimeout(10)
                value = json.loads(self.rfile.read(size))
                result = owner.submit(value)
            except (ValueError, TypeError, OSError, RuntimeError):
                self.send(409, {"error": "invalid_or_conflicting_command"})
                return
            self.send(202, result)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="resident-http")
    thread.start()
    try:
        owner.report("worker_ready", port=port, native_initializations=1)
        while owner.run_next():
            pass  # Idle does not integrate physics or make fake GPU work.
    finally:
        server.shutdown()
        server.server_close()
