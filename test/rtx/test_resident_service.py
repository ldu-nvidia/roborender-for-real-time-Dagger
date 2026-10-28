from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
import json
import threading
import socket
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import pytest

from dagger_rtx.resident_service import ResidentOwner, SCHEMA, validate_command, serve, bounded_native_progress


class FakeNative:
    def __init__(self):
        self.calls = []
    def validate(self, operation, payload):
        if payload.get("invalid"):
            raise ValueError("bad native binding")
    def run(self, operation, payload, report):
        self.calls.append((operation, payload, threading.get_ident()))
        if payload.get("fail"):
            raise RuntimeError("native uncertainty")
        return {"complete": True, "operation": operation}


def owner(tmp_path):
    backend = FakeNative()
    value = ResidentOwner(tmp_path, backend, gpu_uuid="GPU-test", source_sha256="a"*64)
    return value, backend


def command(value, cid="phase-one", operation="capture_reset", payload=None):
    return {"schema": SCHEMA, "worker_instance_id": value.state["worker_instance_id"],
            "command_id": cid, "operation": operation, "payload": payload or {}}


def test_commands_run_only_on_main_thread(tmp_path):
    value, backend = owner(tmp_path)
    with ThreadPoolExecutor(1) as pool:
        accepted = pool.submit(value.submit, command(value)).result()
        with pytest.raises(RuntimeError, match="owner thread"):
            pool.submit(value.run_next).result()
    assert not backend.calls
    assert value.run_next()
    assert backend.calls[0][2] == threading.get_ident()
    assert value.result("phase-one")["status"] == "COMPLETE"
    assert accepted == value.submit(command(value))
    assert len(backend.calls) == 1


def test_concurrent_duplicate_submission_once(tmp_path):
    value, backend = owner(tmp_path)
    body = command(value)
    with ThreadPoolExecutor(8) as pool:
        receipts = list(pool.map(lambda _: value.submit(body), range(32)))
    assert all(row == receipts[0] for row in receipts)
    value.run_next()
    assert len(backend.calls) == 1


def test_busy_and_conflict_are_rejected(tmp_path):
    value, backend = owner(tmp_path)
    value.submit(command(value))
    with pytest.raises(ValueError, match="busy"):
        value.submit(command(value, "phase-two"))
    with pytest.raises(ValueError, match="conflict"):
        value.submit(command(value, payload={"seed": 43}))
    value.run_next()
    with pytest.raises(ValueError, match="conflict"):
        value.submit(command(value, payload={"seed": 43}))
    assert len(backend.calls) == 1


def test_old_instance_rejected(tmp_path):
    value, backend = owner(tmp_path)
    body = command(value)
    body["worker_instance_id"] = "b"*32
    with pytest.raises(ValueError, match="instance mismatch"):
        value.submit(body)
    assert not backend.calls


@pytest.mark.parametrize("mutation", [
    {"operation": "env.step"}, {"extra": True}, {"schema": "v0"},
    {"command_id": "../escape"}, {"payload": {"nan": float("nan")}},
    {"operation": "shutdown", "payload": {"force": True}},
])
def test_bad_commands_rejected(tmp_path, mutation):
    value, _ = owner(tmp_path)
    with pytest.raises((ValueError, TypeError)):
        validate_command({**command(value), **mutation})


@pytest.mark.parametrize("when", ["idle", "queued", "complete", "uncertain"])
def test_restart_never_claims_simulator_restored(tmp_path, when):
    value, backend = owner(tmp_path)
    if when != "idle":
        value.submit(command(value, payload={"fail": True} if when == "uncertain" else {}))
    if when == "complete":
        value.run_next()
    if when == "uncertain":
        with pytest.raises(RuntimeError):
            value.run_next()
    saved = value.path.read_bytes()
    with pytest.raises(RuntimeError, match="CRASH_UNCERTAIN"):
        ResidentOwner(tmp_path, FakeNative(), gpu_uuid="GPU-test", source_sha256="a"*64)
    assert value.path.read_bytes() == saved


def test_failure_blocks_later_work(tmp_path):
    value, backend = owner(tmp_path)
    value.submit(command(value, payload={"fail": True}))
    with pytest.raises(RuntimeError):
        value.run_next()
    assert value.status()["phase"] == "CRASH_UNCERTAIN"
    assert value.result("phase-one")["status"] == "CRASH_UNCERTAIN"
    with pytest.raises(ValueError, match="busy or uncertain"):
        value.submit(command(value, "phase-two"))
    assert len(backend.calls) == 1


def test_sequential_training_evaluation_rounds_one_owner(tmp_path):
    value, backend = owner(tmp_path)
    for cid, operation in [("r1", "collect_round"), ("eval1", "evaluate_pair"),
                           ("r2", "collect_round"), ("eval2", "evaluate_pair")]:
        value.submit(command(value, cid, operation))
        assert value.run_next()
    assert value.status()["native_initializations"] == 1
    assert [row[0] for row in backend.calls] == ["collect_round", "evaluate_pair"]*2
    value.submit(command(value, "shutdown", "shutdown"))
    assert not value.run_next()
    assert value.status()["phase"] == "STOPPED"
    assert json.loads(value.path.read_bytes())["commands"]["shutdown"]["status"] == "COMPLETE"


def test_new_production_segment_command_keeps_existing_serial_idempotent_owner(tmp_path):
    value, backend = owner(tmp_path)
    body = command(value, 'production-segment-one', 'collect_production_segment', {'pinned': 'CPU-fixture'})
    first = value.submit(body)
    assert value.submit(body) == first and not backend.calls
    value.run_next()
    assert len(backend.calls) == 1 and backend.calls[0][0] == 'collect_production_segment'
    assert value.submit(body) == first
    assert value.status()['native_initializations'] == 1
    assert value.result('production-segment-one')['status'] == 'COMPLETE'
    with pytest.raises(ValueError, match='conflict'):
        value.submit({**body, 'payload': {'changed': True}})
    assert len(backend.calls) == 1


def test_real_loopback_boundary_and_concurrent_idempotency(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    token = tmp_path / "token"
    token.write_text("unit-test-resident-token")
    token.chmod(0o600)
    shared = {}
    def worker():
        value, backend = owner(tmp_path / "worker")
        shared.update(owner=value, backend=backend)
        serve(value, port=port, token_file=token)
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    def http(route, body=None, auth=True):
        headers = {"Content-Type": "application/json"}
        if auth:
            headers["Authorization"] = "Bearer unit-test-resident-token"
        request = Request(f"http://127.0.0.1:{port}"+route,
            data=json.dumps(body).encode() if body is not None else None, headers=headers)
        with urlopen(request, timeout=3) as response:
            return json.loads(response.read())
    deadline = time.monotonic()+5
    while True:
        try:
            ready = http("/ready")
            break
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(.02)
    assert ready["native_initializations"] == 1
    with pytest.raises(HTTPError) as rejected:
        http("/ready", auth=False)
    assert rejected.value.code == 401
    value = shared["owner"]
    body = command(value)
    with ThreadPoolExecutor(4) as pool:
        accepted = list(pool.map(lambda _: http("/v1/commands", body), range(8)))
    assert all(row == accepted[0] for row in accepted)
    while http("/v1/commands/phase-one")["status"] != "COMPLETE":
        assert time.monotonic() < deadline
        time.sleep(.01)
    assert len(shared["backend"].calls) == 1
    with pytest.raises(HTTPError) as conflict:
        http("/v1/commands", {**body, "payload": {"seed": 99}})
    assert conflict.value.code == 409
    http("/v1/commands", command(value, "shutdown", "shutdown"))
    thread.join(5)
    assert not thread.is_alive()


def test_bounded_progress_is_metadata_only_and_phase_switch_safe(tmp_path):
    assert bounded_native_progress(None) is None
    assert bounded_native_progress(tmp_path) is None
    (tmp_path / "progress.json").write_text(json.dumps({"phase":"capture_teacher","vector_step":5,
        "elapsed_ms":123,"counts":{"complete":6},"teacher_actions":[1,2,3]}))
    row = bounded_native_progress(tmp_path)
    assert row == {"phase":"capture_teacher","vector_step":5,"elapsed_ms":123,"counts":{"complete":6}}
    (tmp_path / "progress.json").write_text("bad JSON")
    assert bounded_native_progress(tmp_path)["phase"] == "progress_snapshot_unavailable"


def test_scaled_progress_reads_only_latest_complete_trace_row(tmp_path):
    first = tmp_path / 'cohort-000'; first.mkdir()
    (first / 'steps.jsonl').write_text('{"step":80,"lanes":[{"action":"private"}]}\n')
    second = tmp_path / 'cohort-001'; second.mkdir()
    assert bounded_native_progress(tmp_path) == dict(phase='evaluation_initialization', cohort_index=1)
    trace = second / 'steps.jsonl'
    trace.write_text('{"step":2,"lanes":[{"action":"private"}]}\n{"step":3')
    assert bounded_native_progress(tmp_path) == dict(phase='evaluation_rollout', cohort_index=1, vector_step=3)
    trace.write_text('{"step":2}\n{"step":3}\n')
    assert bounded_native_progress(tmp_path)['vector_step'] == 4
    trace.write_text('[]\n')
    assert bounded_native_progress(tmp_path)['phase'] == 'progress_snapshot_unavailable'
    assert bounded_native_progress(None) is None  # IDLE cannot expose an old phase.


def test_scaled_progress_bounds_large_trace_and_rejects_alias(tmp_path):
    cohort = tmp_path / 'cohort-000'; cohort.mkdir()
    trace = cohort / 'steps.jsonl'
    trace.write_bytes((b'{"step":0,"unused":"' + b'x' * 100 + b'"}\n') * 10000 + b'{"step":99}\n')
    assert bounded_native_progress(tmp_path)['vector_step'] == 100
    alias = tmp_path / 'cohort-001'; alias.symlink_to(cohort, target_is_directory=True)
    assert bounded_native_progress(tmp_path)['phase'] == 'progress_snapshot_unavailable'
