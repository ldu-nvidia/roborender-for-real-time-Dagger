from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from dagger_rtx.phase_telemetry import PhaseBoundaryTelemetry
from dagger_rtx.wandb_phase_mirror import (
    WandbPhaseMirror,
    WandbPhaseMirrorError,
    authorize_collection_wandb,
)
from conftest import PROJECT_ROOT


@dataclass
class _FakeRun:
    id: str = "wandb-phase-run"
    name: str = "wandb-phase-run"
    project: str = "dagger"
    entity: str = "nvidia"


class _FakeWandb:
    def __init__(self, *, fail_log: bool = False) -> None:
        self.run = _FakeRun()
        self.fail_log = fail_log
        self.logs: list[tuple[dict, int, bool]] = []
        self.finishes: list[tuple[int, bool]] = []

    def log(self, payload, *, step, commit):
        self.logs.append((dict(payload), step, commit))
        if self.fail_log:
            raise RuntimeError("injected W&B failure")

    def finish(self, *, exit_code, quiet):
        self.finishes.append((exit_code, quiet))


def _telemetry(tmp_path: Path, client: _FakeWandb) -> tuple[PhaseBoundaryTelemetry, WandbPhaseMirror]:
    mirror = WandbPhaseMirror(client, run_id="wandb-phase-run")
    telemetry = PhaseBoundaryTelemetry(
        tmp_path / "phase.jsonl",
        tmp_path / "progress.json",
        run_id="wandb-phase-run",
        round_id=1,
        total_vector_steps=2,
        batch_size=2,
        fsync_events=False,
        event_observer=mirror.observe,
    )
    return telemetry, mirror


def _complete(telemetry: PhaseBoundaryTelemetry, step: int, phase: str, **counts) -> None:
    telemetry.start(step, phase, total=2)
    telemetry.end(step, phase, total=2, completed=2, counts=counts)


def test_buffers_end_and_error_then_logs_exactly_once_at_completed_barrier(tmp_path):
    client = _FakeWandb()
    telemetry, mirror = _telemetry(tmp_path, client)
    try:
        _complete(
            telemetry,
            0,
            "capture_teacher",
            lanes=2,
            ignored_string="not-a-metric",
        )
        telemetry.start(0, "prepare_encode_send_wait", total=2)
        telemetry.error(
            0,
            "prepare_encode_send_wait",
            TimeoutError("retry"),
            completed=1,
            total=2,
            counts={"responses": 1},
        )
        _complete(telemetry, 0, "prepare_encode_send_wait", responses=2)
        _complete(telemetry, 0, "prepare_or_exchange", responses=2)
        _complete(telemetry, 0, "env_step", env_step_calls=1)
        _complete(
            telemetry,
            0,
            "vector_step_barrier",
            durable_steps=1,
            completion_fraction=0.5,
        )

        # Completed retry produces only a local already_complete progress row.
        assert not telemetry.start(0, "vector_step_barrier", total=2)
        assert not telemetry.end(0, "vector_step_barrier", total=2)
    finally:
        telemetry.close()
    mirror.finish(exit_code=0)
    mirror.finish(exit_code=1)

    assert len(client.logs) == 1
    payload, step, commit = client.logs[0]
    assert (step, commit) == (0, True)
    assert payload["collection/vector_step_completed"] == 1
    assert payload["collection/vector_step"] == 0
    assert payload["collection/completion_fraction"] == 0.5
    assert "collection/timing/capture_teacher_ms" in payload
    assert "collection/timing/prepare_or_exchange_ms" in payload
    assert "collection/timing/env_step_ms" in payload
    assert payload["collection/errors"] == 1
    assert payload["collection/retries"] == 0
    assert payload["collection/phase/capture_teacher/count/lanes"] == 2
    assert not any("ignored_string" in key for key in payload)
    assert payload["collection/phase/prepare_encode_send_wait/error_count"] == 1
    assert payload["collection/phase/prepare_encode_send_wait/ended"] == 1
    assert payload["collection/phase/vector_step_barrier/ended"] == 1
    assert client.finishes == [(0, True)]
    assert mirror.summary() == {
        "enabled": True,
        "logged_vector_steps": 1,
        "finish_succeeded": True,
        "failure_count": 0,
        "failures": [],
    }


def test_wandb_failure_does_not_escape_or_duplicate_completed_step(tmp_path):
    client = _FakeWandb(fail_log=True)
    telemetry, mirror = _telemetry(tmp_path, client)
    try:
        _complete(telemetry, 0, "vector_step_barrier")
        assert not telemetry.start(0, "vector_step_barrier", total=2)
    finally:
        telemetry.close()
    assert len(client.logs) == 1
    assert mirror.summary()["failure_count"] == 1


def test_rejects_mismatched_preinitialized_raven_run():
    client = _FakeWandb()
    client.run.entity = "someone-else"
    with pytest.raises(WandbPhaseMirrorError, match="entity mismatch"):
        WandbPhaseMirror(client, run_id="wandb-phase-run")


def test_production_collection_wandb_requires_matching_immutable_config():
    logging = {
        "rtx_collection_wandb": True,
        "wandb_project": "dagger",
        "wandb_run_name": "production-run",
    }
    assert authorize_collection_wandb(
        mode="production",
        raw_enabled="1",
        run_id="production-run",
        production_logging=logging,
    )

    with pytest.raises(WandbPhaseMirrorError, match="differs from immutable"):
        authorize_collection_wandb(
            mode="production",
            raw_enabled="0",
            run_id="production-run",
            production_logging=logging,
        )
    with pytest.raises(WandbPhaseMirrorError, match="run name equal to run_id"):
        authorize_collection_wandb(
            mode="production",
            raw_enabled="1",
            run_id="different-run",
            production_logging=logging,
        )


@pytest.mark.parametrize("mode", ["capture-only", "isolated-one-step", "capacity-probe"])
def test_nonproduction_collection_wandb_remains_fail_closed(mode):
    with pytest.raises(WandbPhaseMirrorError, match="restricted"):
        authorize_collection_wandb(
            mode=mode,
            raw_enabled="1",
            run_id="test-run",
        )


def test_single_window_collection_wandb_remains_authorized():
    assert authorize_collection_wandb(
        mode="single-window",
        raw_enabled="1",
        run_id="test-run",
    )


def test_launcher_wandb_is_opt_in_owner_only_and_uses_stable_raven_identity():
    launcher = (PROJECT_ROOT / "scripts" / "launch_native_raven.sh").read_text(
        encoding="utf-8"
    )
    assert "wandb_hydra_args=(use_wandb=false)" in launcher
    assert "--wandb-netrc" in launcher
    assert "W&B netrc must be owned by the invoking user" in launcher
    assert "W&B netrc must be mode 0400 or 0600" in launcher
    assert '[[ $mode == single-window || $mode == production ]]' in launcher
    assert "production W&B flag differs from immutable config" in launcher
    assert "production W&B run name must match run_id" in launcher
    assert "dst=/dagger-runtime/home/.netrc,readonly" in launcher
    assert "DAGGER_COLLECTION_WANDB_ENABLED=\"$wandb_enabled\"" in launcher
    assert "wandb_hydra_args=(\n    use_wandb=true" in launcher
    assert "project_name=dagger" in launcher
    assert '"experiment_dir=/dagger-runtime/logs_rl/dagger/$run_id"' in launcher
    assert "wandb.wandb_entity=nvidia" in launcher
    assert '"wandb.wandb_id=$run_id"' in launcher
    assert '--env WANDB_NAME="$run_id"' in launcher
    assert '--env WANDB_RUN_ID="$run_id"' in launcher


def test_trainer_finishes_wandb_before_native_result_publication_and_on_error():
    source = (
        PROJECT_ROOT / "code" / "src" / "dagger_rtx" / "native_service_trainer.py"
    ).read_text(encoding="utf-8")
    publication = 'atomic_write_json(runtime_root / "native-service-result.json", summary)'
    assert source.index("wandb_mirror.finish(exit_code=0)") < source.index(publication)
    error_finish = source.index("wandb_mirror.finish(exit_code=1)")
    error_raise = source.index("            raise\n", error_finish)
    assert error_finish < error_raise
