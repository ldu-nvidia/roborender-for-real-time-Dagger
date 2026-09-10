from __future__ import annotations

import json
from pathlib import Path

from dagger_rtx.batch_backend import DeterministicFakeBatchRaven
from dagger_rtx.batch_orchestrator import BatchStepOrchestrator
from dagger_rtx.batch_owner import IsaacBatchTransitionOwner
from dagger_rtx.boundary import H200_COMMIT_PATH
from dagger_rtx.fake import fake_static_provenance
from dagger_rtx.fake_h200 import FakeH200Server
from dagger_rtx.http_transport import BoundedBearerJsonTransport, TransientTransportError
from dagger_rtx.phase_telemetry import PHASE_BOUNDARY_SCHEMA, PhaseBoundaryTelemetry
from dagger_rtx.window_runner import BatchWindowRunner, TransportRetryPolicy


REQUIRED_EVENT_FIELDS = {
    "schema",
    "sequence",
    "run_id",
    "round_id",
    "vector_step",
    "phase",
    "event",
    "started_at_utc",
    "finished_at_utc",
    "elapsed_ms",
    "completed",
    "total",
    "status",
    "counts",
}


class _Clocks:
    def __init__(self) -> None:
        self.wall = 1_800_000_000_000_000_000
        self.mono = 50_000_000_000

    def clock_ns(self) -> int:
        return self.wall

    def monotonic_ns(self) -> int:
        return self.mono

    def advance_ms(self, value: int) -> None:
        delta = value * 1_000_000
        self.wall += delta
        self.mono += delta


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _reporter(tmp_path: Path, *, run_id: str = "phase-test", steps: int = 1):
    clocks = _Clocks()
    reporter = PhaseBoundaryTelemetry(
        tmp_path / "phase-boundaries.jsonl",
        tmp_path / "progress.json",
        run_id=run_id,
        round_id=1,
        total_vector_steps=steps,
        batch_size=2,
        clock_ns=clocks.clock_ns,
        monotonic_ns=clocks.monotonic_ns,
        fsync_events=False,
    )
    return reporter, clocks


def test_phase_event_order_required_fields_and_atomic_latest_progress(tmp_path):
    reporter, clocks = _reporter(tmp_path)
    try:
        assert reporter.start(0, "capture_teacher", total=2, counts={"requested_lanes": 2})
        clocks.advance_ms(7)
        reporter.progress(
            0,
            "capture_teacher",
            completed=1,
            total=2,
            counts={"captured_lanes": 1},
        )
        clocks.advance_ms(5)
        assert reporter.end(
            0,
            "capture_teacher",
            completed=2,
            total=2,
            counts={"captured_lanes": 2},
        )
    finally:
        reporter.close()

    rows = _rows(tmp_path / "phase-boundaries.jsonl")
    assert [row["event"] for row in rows] == ["start", "progress", "end"]
    assert [row["sequence"] for row in rows] == [1, 2, 3]
    assert all(REQUIRED_EVENT_FIELDS <= set(row) for row in rows)
    assert all(row["schema"] == PHASE_BOUNDARY_SCHEMA for row in rows)
    assert rows[-1]["elapsed_ms"] == 12.0
    latest = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert latest["event"] == "end"
    assert latest["completed"] == latest["total"] == 2
    assert latest["completed_phase_instances"] == 1
    assert latest["phase_completion_counts"] == {"capture_teacher": 1}


def test_phase_exception_emits_error_boundary(tmp_path):
    reporter, clocks = _reporter(tmp_path)
    try:
        reporter.start(0, "prepare_encode_send_wait", total=2)
        clocks.advance_ms(9)
        reporter.error(
            0,
            "prepare_encode_send_wait",
            TimeoutError("injected timeout"),
            completed=0,
            total=2,
        )
    finally:
        reporter.close()

    rows = _rows(tmp_path / "phase-boundaries.jsonl")
    assert [row["event"] for row in rows] == ["start", "error"]
    assert rows[-1]["elapsed_ms"] == 9.0
    assert rows[-1]["status"] == "error"
    assert rows[-1]["counts"]["error_type"] == "TimeoutError"
    assert rows[-1]["counts"]["error_message"] == "injected timeout"


def test_completed_phase_retry_is_reported_but_not_double_counted(tmp_path):
    reporter, clocks = _reporter(tmp_path)
    try:
        reporter.start(0, "commit_encode_send_wait", total=2)
        clocks.advance_ms(3)
        reporter.error(
            0,
            "commit_encode_send_wait",
            TimeoutError("first attempt"),
            completed=0,
            total=2,
        )
        reporter.start(0, "commit_encode_send_wait", total=2)
        clocks.advance_ms(4)
        assert reporter.end(0, "commit_encode_send_wait", total=2)
        assert not reporter.start(0, "commit_encode_send_wait", total=2)
        assert not reporter.end(0, "commit_encode_send_wait", total=2)
    finally:
        reporter.close()

    rows = _rows(tmp_path / "phase-boundaries.jsonl")
    assert sum(row["event"] == "end" for row in rows) == 1
    assert sum(row["status"] == "already_complete" for row in rows) == 2
    latest = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert latest["completed_phase_instances"] == 1
    assert latest["phase_completion_counts"] == {"commit_encode_send_wait": 1}


def test_real_telemetry_axis_is_contiguous_at_window39_and_post_validation40(
    tmp_path,
):
    """The last pre-validation step and first next-epoch step must not collide."""

    reporter = PhaseBoundaryTelemetry(
        tmp_path / "phase-boundaries.jsonl",
        tmp_path / "progress.json",
        run_id="phase-validation-boundary",
        round_id=1,
        total_vector_steps=80 * 81,
        batch_size=28,
        fsync_events=False,
    )
    try:
        reporter.set_vector_step_offset(39 * 81)
        assert reporter.start(80, "vector_step_barrier", total=28)
        assert reporter.end(80, "vector_step_barrier", total=28)

        # A post-validation fresh simulator segment retains the same global
        # telemetry axis even though its window-local index restarts at zero.
        reporter.set_vector_step_offset(40 * 81)
        assert reporter.start(0, "vector_step_barrier", total=28)
        assert reporter.end(0, "vector_step_barrier", total=28)
    finally:
        reporter.close()

    rows = [row for row in _rows(tmp_path / "phase-boundaries.jsonl") if row["event"] == "end"]
    assert [row["vector_step"] for row in rows] == [3239, 3240]
    latest = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert latest["vector_step"] == 3240
    assert latest["completed_phase_instances"] == 2


class _FailFirstCommit:
    def __init__(self, delegate) -> None:
        self.delegate = delegate
        self.failed = False

    def post(self, path, payload, *, max_response_bytes):
        if path == H200_COMMIT_PATH and not self.failed:
            self.failed = True
            raise TransientTransportError("injected retryable commit failure")
        return self.delegate.post(path, payload, max_response_bytes=max_response_bytes)


def _token_file(tmp_path: Path) -> Path:
    path = tmp_path / "token"
    path.write_text("test-token\n", encoding="utf-8")
    path.chmod(0o600)
    return path


def test_b2_full_phase_order_and_commit_retry_never_double_steps(tmp_path, tiny_profile):
    run_id = "phase-b2-run"
    reporter = PhaseBoundaryTelemetry(
        tmp_path / "phase-boundaries.jsonl",
        tmp_path / "progress.json",
        run_id=run_id,
        round_id=1,
        total_vector_steps=1,
        batch_size=2,
        fsync_events=False,
    )
    backend = DeterministicFakeBatchRaven(tiny_profile, num_envs=2)
    owner = IsaacBatchTransitionOwner(
        backend,
        tmp_path / "owner",
        tmp_path / "staging",
        shape_profile=tiny_profile,
        static_provenance=fake_static_provenance(),
        phase_telemetry=reporter,
    )
    try:
        with FakeH200Server(shape_profile=tiny_profile) as remote:
            transport = _FailFirstCommit(
                BoundedBearerJsonTransport(remote.base_url, _token_file(tmp_path))
            )
            orchestrator = BatchStepOrchestrator(
                owner,
                transport,
                tmp_path / "orchestrator",
                num_envs=2,
                phase_telemetry=reporter,
            )
            runner = BatchWindowRunner(
                owner,
                orchestrator,
                run_id=run_id,
                round_id=1,
                actor_prefix="phase-b2",
                retry_policy=TransportRetryPolicy(
                    max_attempts=2,
                    initial_backoff_s=0,
                    max_backoff_s=0,
                ),
                sleep=lambda _: None,
                phase_telemetry=reporter,
            )
            runner.begin_window(0)
            result = runner.run_step(0, 0)
            assert result["phase"] == "COMPLETE"
            assert backend.batch_step_calls == 1
            assert remote.state.prepare_effects == 2
            assert remote.state.commit_effects == 2
    finally:
        reporter.close()

    rows = _rows(tmp_path / "phase-boundaries.jsonl")
    completed = [row["phase"] for row in rows if row["event"] == "end"]
    expected_order = [
        "capture_teacher",
        "prepare_encode_and_persist",
        "prepare_local_total",
        "prepare_encode_send_wait",
        "commit_intent_journal",
        "native_env_step",
        "commit_outcomes_and_persist",
        "commit_local_total",
        "commit_encode_send_wait",
        "recovery",
        "vector_step_barrier",
    ]
    positions = [completed.index(phase) for phase in expected_order]
    assert positions == sorted(positions)
    assert completed.count("native_env_step") == 1
    assert completed.count("commit_encode_send_wait") == 1
    assert completed.count("vector_step_barrier") == 1
    assert any(
        row["phase"] == "commit_encode_send_wait" and row["event"] == "error"
        for row in rows
    )
    assert any(
        row["phase"] == "vector_step_barrier" and row["status"] == "retrying"
        for row in rows
    )
