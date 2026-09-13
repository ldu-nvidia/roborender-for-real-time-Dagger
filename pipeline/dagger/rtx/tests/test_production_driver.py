from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dagger_rtx.production_driver import (
    BarrierGate,
    H200ContinuationActivityAttestation,
    InitialProductionGate,
    ProductionDriverError,
    ProductionContinuationSegment,
    ProductionMultiWindowDriver,
    ProductionRestartUnsafe,
    ProductionRunSpec,
    WindowAdvanceGate,
)


RUN_ID = "dagger-r1-production-gpu1-b28-20260910-v2"
CONFIG_SHA = "42f6d681a565ffd9146a1787a61e0775d7546367eeeaa8d9f5b512142c570514"
INITIAL_SHA = "4b7fbc3b10c5a3a6381fd6a89bd5ac779e045fcae8789769d2f0c084f4570a42"
ACTOR_PREFIX = f"{RUN_ID}-g1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _spec(*, total_windows: int = 40) -> ProductionRunSpec:
    return ProductionRunSpec(
        run_id=RUN_ID,
        config_sha256=CONFIG_SHA,
        actor_prefix=ACTOR_PREFIX,
        actor_ids=tuple(f"{ACTOR_PREFIX}-lane-{lane:03d}" for lane in range(28)),
        initial_checkpoint_sha256=INITIAL_SHA,
        total_windows=total_windows,
    )


def _segment(start: int, checkpoint: str, reason: str = "crash_uncertain"):
    base = 10_000_000 + start * 100_000
    activity = (
        None
        if reason == "post_validation"
        else H200ContinuationActivityAttestation(
            run_id=RUN_ID,
            config_sha256=CONFIG_SHA,
            round_id=1,
            iteration_id=start + 1,
            rollout_window_id=f"window-{start:06d}",
            prepare_count=0,
            commit_count=0,
            chunk_count=0,
            actor_barrier_count=0,
            observed_at_utc="2026-09-10T09:00:00Z",
        )
    )
    return ProductionContinuationSegment(
        segment_id=f"segment-from-window-{start:06d}",
        run_id=RUN_ID,
        config_sha256=CONFIG_SHA,
        start_global_window=start,
        expected_checkpoint_sha256=checkpoint,
        simulator_seed=42000 + start,
        episode_serial_base=base,
        episode_serial_limit_exclusive=base + 100_000,
        reset_reason=reason,
        predecessor_state_path=f"/tmp/predecessor-{start}.json",
        predecessor_state_sha256=_sha(f"predecessor-{start}"),
        h200_activity_attestation=activity,
        h200_activity_evidence_path=(
            f"/tmp/h200-activity-{start}.json" if activity is not None else None
        ),
        h200_activity_evidence_sha256=(
            _sha(f"h200-activity-{start}") if activity is not None else None
        ),
        prior_continuation_manifest_sha256s=(),
    )


class _Telemetry:
    def __init__(self):
        self.offsets = []

    def set_vector_step_offset(self, value):
        self.offsets.append(value)


class _Lifecycle:
    def __init__(self):
        self.begin_calls = 0
        self.finish_calls = 0

    def begin(self):
        self.begin_calls += 1

    def finish(self):
        self.finish_calls += 1


class _Runner:
    def __init__(self, spec, *, fail_at=None):
        self.num_envs = spec.num_envs
        self.run_id = spec.run_id
        self.round_id = spec.round_id
        self.actor_prefix = spec.actor_prefix
        self.actor_ids = spec.actor_ids
        self.calls = []
        self.fail_at = fail_at
        self.phase_telemetry = _Telemetry()
        self.lifecycle = _Lifecycle()

    def run_window(self, index, *, manage_lifecycle=True):
        assert manage_lifecycle is False
        self.calls.append(index)
        if index == self.fail_at:
            raise RuntimeError("injected native failure")
        window_id = f"window-{index:06d}"
        return {
            "contract_version": "roborender.dagger.rtx.v1",
            "batch_size": 28,
            "window_index": index,
            "transition_batches": 81,
            "physical_transition_rows": 2268,
            "final": {
                "phase": "COMPLETE",
                "step_keys": [
                    {
                        "run_id": self.run_id,
                        "round_id": self.round_id,
                        "actor_id": actor,
                        "rollout_window_id": window_id,
                        "rollout_step_index": 80,
                        "episode_id": f"episode-{lane:03d}-00000001",
                        "episode_step_index": 7,
                    }
                    for lane, actor in enumerate(self.actor_ids)
                ],
                "local_commits": [
                    {
                        "sealed": True,
                        "seal": {"sha256": _sha(f"seal-{index}-{lane}")},
                        "result": {
                            "reward": float(lane),
                            "done": lane == 0 and index == 1,
                            "timeout": False,
                        },
                    }
                    for lane in range(28)
                ],
                "remote_acknowledgements": [
                    {"remote_commit_receipt": _sha(f"ack-{index}-{lane}")}
                    for lane in range(28)
                ],
            },
        }


class _Control:
    def __init__(
        self, spec, *, unchanged_at=None, wrong_status_at=None, bad_barrier_at=None
    ):
        self.spec = spec
        self.unchanged_at = unchanged_at
        self.wrong_status_at = wrong_status_at
        self.bad_barrier_at = bad_barrier_at
        self.initial_calls = 0
        self.continuation_calls = []
        self.barrier_calls = []
        self.advance_calls = []
        self.checkpoint = spec.initial_checkpoint_sha256

    def attest_initial(self, spec):
        self.initial_calls += 1
        assert spec == self.spec
        return InitialProductionGate(
            run_id=spec.run_id,
            config_sha256=spec.config_sha256,
            round_id=1,
            active_iteration_id=1,
            active_checkpoint_sha256=spec.initial_checkpoint_sha256,
            actor_ids=spec.actor_ids,
            status="collecting",
        )

    def attest_continuation(self, spec, segment):
        self.continuation_calls.append(segment)
        self.checkpoint = segment.expected_checkpoint_sha256
        return InitialProductionGate(
            run_id=spec.run_id,
            config_sha256=spec.config_sha256,
            round_id=1,
            active_iteration_id=segment.start_global_window + 1,
            active_checkpoint_sha256=segment.expected_checkpoint_sha256,
            actor_ids=spec.actor_ids,
            status="collecting",
        )

    def post_window_barriers(
        self, spec, *, completed_window_index, input_checkpoint_sha256
    ):
        self.barrier_calls.append((completed_window_index, input_checkpoint_sha256))
        actor_ids = spec.actor_ids
        if completed_window_index == self.bad_barrier_at:
            actor_ids = actor_ids[:-1]
        return BarrierGate(
            run_id=spec.run_id,
            round_id=spec.round_id,
            completed_window_index=completed_window_index,
            iteration_id=completed_window_index + 1,
            checkpoint_sha256=input_checkpoint_sha256,
            actor_ids=actor_ids,
            chunk_ids=tuple(
                f"d{spec.round_id}-{actor}-window-{completed_window_index:06d}"
                for actor in actor_ids
            ),
            arrived=len(actor_ids),
            expected=spec.num_envs,
            ready_for_learning=len(actor_ids) == spec.num_envs,
        )

    def wait_for_window_advance(
        self, spec, *, completed_window_index, previous_checkpoint_sha256
    ):
        self.advance_calls.append((completed_window_index, previous_checkpoint_sha256))
        if completed_window_index != self.unchanged_at:
            self.checkpoint = _sha(f"candidate-{completed_window_index + 1}")
        validation_boundary = (completed_window_index + 1) % 40 == 0
        status = "awaiting_validation" if validation_boundary else "collecting"
        if completed_window_index == self.wrong_status_at:
            status = "learning"
        return WindowAdvanceGate(
            run_id=spec.run_id,
            config_sha256=spec.config_sha256,
            round_id=1,
            completed_window_index=completed_window_index,
            completed_iteration_id=completed_window_index + 1,
            next_iteration_id=(
                None if validation_boundary else completed_window_index + 2
            ),
            output_checkpoint_sha256=self.checkpoint,
            barrier_count=28,
            actor_ids=spec.actor_ids,
            status=status,
        )


def test_runs_exactly_40_sequential_windows_in_one_driver(tmp_path):
    spec = _spec()
    runner = _Runner(spec)
    control = _Control(spec)
    driver = ProductionMultiWindowDriver(
        runner, control, tmp_path, spec, process_instance_id="test-process-a"
    )

    result = driver.run()

    assert runner.calls == list(range(40))
    assert runner.phase_telemetry.offsets == [index * 81 for index in range(40)]
    assert control.initial_calls == 1
    assert [item[0] for item in control.barrier_calls] == list(range(40))
    assert [item[0] for item in control.advance_calls] == list(range(40))
    assert result["phase"] == "COMPLETE"
    assert result["next_window_index"] == 40
    assert result["current_iteration_id"] == 40
    assert result["final_candidate_checkpoint_sha256"] == _sha("candidate-40")
    assert result["simulator_initialization_count"] == 1
    assert result["rollout_lifecycle_initialization_count"] == 1
    assert runner.lifecycle.begin_calls == 1
    assert runner.lifecycle.finish_calls == 1
    assert len(result["completed_windows"]) == 40
    assert [row["rollout_window_id"] for row in result["completed_windows"]] == [
        f"window-{index:06d}" for index in range(40)
    ]
    assert all(row["sealed_lane_count"] == 28 for row in result["completed_windows"])
    assert all(
        row["physical_transition_rows"] == 28 * 81
        for row in result["completed_windows"]
    )


def test_bounded_v4_smoke_stops_after_one_window_at_learner_barrier(tmp_path):
    spec = _spec()
    runner = _Runner(spec)
    control = _Control(spec)
    driver = ProductionMultiWindowDriver(
        runner,
        control,
        tmp_path,
        spec,
        stop_at_learner_barrier=True,
        process_instance_id="test-v4-smoke-process",
    )

    result = driver.run()

    assert runner.calls == [0]
    assert control.initial_calls == 1
    assert control.barrier_calls == [(0, INITIAL_SHA)]
    assert control.advance_calls == []
    assert result["phase"] == "READY_FOR_LEARNING"
    assert result["next_window_index"] == 1
    assert result["pending_barriers"]["arrived"] == 28
    assert result["pending_barriers"]["ready_for_learning"] is True
    assert len(result["completed_windows"]) == 1
    assert result["completed_windows"][0]["output_checkpoint_sha256"] is None
    assert runner.lifecycle.begin_calls == 1
    assert runner.lifecycle.finish_calls == 1


def test_failed_native_window_is_durable_crash_uncertain_and_never_replayed(tmp_path):
    spec = _spec()
    runner = _Runner(spec, fail_at=2)
    driver = ProductionMultiWindowDriver(
        runner, _Control(spec), tmp_path, spec, process_instance_id="test-process-a"
    )

    with pytest.raises(RuntimeError, match="injected native failure"):
        driver.run()
    assert runner.calls == [0, 1, 2]
    state = json.loads((tmp_path / "production-driver-state.json").read_text())
    assert state["phase"] == "CRASH_UNCERTAIN"
    assert state["next_window_index"] == 2

    replacement = _Runner(spec)
    with pytest.raises(ProductionRestartUnsafe, match="no attested Raven/Isaac"):
        ProductionMultiWindowDriver(
            replacement,
            _Control(spec),
            tmp_path,
            spec,
            process_instance_id="test-process-b",
        )
    assert replacement.calls == []


def test_checkpoint_must_advance_before_next_window(tmp_path):
    spec = _spec()
    runner = _Runner(spec)
    driver = ProductionMultiWindowDriver(
        runner,
        _Control(spec, unchanged_at=0),
        tmp_path,
        spec,
        process_instance_id="test-process-a",
    )

    with pytest.raises(ProductionDriverError, match="did not advance"):
        driver.run()
    assert runner.calls == [0]
    assert driver.status()["phase"] == "CRASH_UNCERTAIN"


def test_all_28_exact_actor_barriers_are_required_before_wait_or_next_window(tmp_path):
    spec = _spec()
    runner = _Runner(spec)
    control = _Control(spec, bad_barrier_at=0)
    driver = ProductionMultiWindowDriver(
        runner, control, tmp_path, spec, process_instance_id="test-process-a"
    )

    with pytest.raises(ProductionDriverError, match="actor-barrier identity"):
        driver.run()
    assert runner.calls == [0]
    assert control.advance_calls == []
    assert driver.status()["phase"] == "CRASH_UNCERTAIN"


def test_invalid_controller_status_fails_before_next_native_window(tmp_path):
    spec = _spec()
    runner = _Runner(spec)
    driver = ProductionMultiWindowDriver(
        runner,
        _Control(spec, wrong_status_at=1),
        tmp_path,
        spec,
        process_instance_id="test-process-a",
    )

    with pytest.raises(ProductionDriverError, match="identity mismatch"):
        driver.run()
    assert runner.calls == [0, 1]
    assert driver.status()["next_window_index"] == 1
    assert driver.status()["phase"] == "CRASH_UNCERTAIN"


def test_completed_state_can_be_reopened_for_read_only_status_without_rerun(tmp_path):
    spec = _spec()
    runner = _Runner(spec)
    driver = ProductionMultiWindowDriver(
        runner, _Control(spec), tmp_path, spec, process_instance_id="test-process-a"
    )
    expected = driver.run()
    before = (tmp_path / "production-driver-state.json").read_bytes()

    replacement = _Runner(spec)
    reopened = ProductionMultiWindowDriver(
        replacement,
        _Control(spec),
        tmp_path,
        spec,
        process_instance_id="test-process-b",
    )
    assert reopened.run() == expected
    assert replacement.calls == []
    assert (tmp_path / "production-driver-state.json").read_bytes() == before


def test_crash_continuation_is_fresh_and_runs_global_windows_one_through_39(tmp_path):
    spec = _spec()
    candidate = _sha("candidate-1")
    segment = _segment(1, candidate)
    runner = _Runner(spec)
    control = _Control(spec)
    driver = ProductionMultiWindowDriver(
        runner,
        control,
        tmp_path,
        spec,
        continuation=segment,
        process_instance_id="continuation-process-a",
    )

    result = driver.run()

    assert control.initial_calls == 0
    assert control.continuation_calls == [segment]
    assert runner.calls == list(range(1, 40))
    assert runner.phase_telemetry.offsets == [index * 81 for index in range(1, 40)]
    assert result["start_window_index"] == 1
    assert result["end_window_index_exclusive"] == 40
    assert result["next_window_index"] == 40
    assert len(result["completed_windows"]) == 39
    assert result["execution_segment_sha256"] == segment.digest


def test_post_validation_segment_preserves_global_iterations_41_through_80(tmp_path):
    spec = _spec(total_windows=80)
    candidate = _sha("candidate-40")
    segment = _segment(40, candidate, "post_validation")
    runner = _Runner(spec)
    control = _Control(spec)
    result = ProductionMultiWindowDriver(
        runner,
        control,
        tmp_path,
        spec,
        continuation=segment,
        process_instance_id="post-validation-process-a",
    ).run()

    assert runner.calls == list(range(40, 80))
    assert control.barrier_calls[0][0] == 40
    assert control.advance_calls[-1][0] == 79
    assert result["next_window_index"] == 80
    assert result["current_iteration_id"] == 80
    assert len(result["completed_windows"]) == 40


def test_50_window_budget_stops_first_segment_at_validation_then_runs_41_through_50(
    tmp_path,
):
    spec = _spec(total_windows=50)

    first_runner = _Runner(spec)
    first_result = ProductionMultiWindowDriver(
        first_runner,
        _Control(spec),
        tmp_path / "segment-1",
        spec,
        process_instance_id="fifty-window-first-segment",
    ).run()
    assert first_runner.calls == list(range(40))
    assert first_result["next_window_index"] == 40
    assert first_result["current_iteration_id"] == 40

    checkpoint_40 = _sha("candidate-40")
    segment = _segment(40, checkpoint_40, "post_validation")
    second_runner = _Runner(spec)
    second_control = _Control(spec)
    second_result = ProductionMultiWindowDriver(
        second_runner,
        second_control,
        tmp_path / "segment-2",
        spec,
        continuation=segment,
        process_instance_id="fifty-window-second-segment",
    ).run()

    assert second_runner.calls == list(range(40, 50))
    assert second_control.barrier_calls[0][0] == 40
    assert second_control.barrier_calls[-1][0] == 49
    assert second_control.advance_calls[-1][0] == 49
    assert second_result["end_window_index_exclusive"] == 50
    assert second_result["next_window_index"] == 50
    assert second_result["current_iteration_id"] == 51
    assert len(second_result["completed_windows"]) == 10


@pytest.mark.parametrize("invalid_total", [0, 39, True])
def test_production_spec_rejects_budget_below_validation_interval(invalid_total):
    with pytest.raises(ValueError, match="total_windows"):
        _spec(total_windows=invalid_total)


def test_production_spec_rejects_any_actor_or_gpu_substitution():
    spec = _spec()
    assert spec.wire_contract_version == "roborender.dagger.rtx.v4"
    actors = list(spec.actor_ids)
    actors[-1] = f"{ACTOR_PREFIX}-lane-999"
    with pytest.raises(ValueError, match="actor_ids"):
        ProductionRunSpec(
            run_id=RUN_ID,
            config_sha256=CONFIG_SHA,
            actor_prefix=ACTOR_PREFIX,
            actor_ids=tuple(actors),
            initial_checkpoint_sha256=INITIAL_SHA,
        )
    with pytest.raises(ValueError, match="GPU1 UUID"):
        ProductionRunSpec(
            run_id=RUN_ID,
            config_sha256=CONFIG_SHA,
            actor_prefix=ACTOR_PREFIX,
            actor_ids=spec.actor_ids,
            initial_checkpoint_sha256=INITIAL_SHA,
            physical_gpu_uuid="GPU-00000000-0000-0000-0000-000000000000",
        )
    with pytest.raises(ValueError, match="served lean v3 or v4"):
        ProductionRunSpec(
            run_id=RUN_ID,
            config_sha256=CONFIG_SHA,
            actor_prefix=ACTOR_PREFIX,
            actor_ids=spec.actor_ids,
            initial_checkpoint_sha256=INITIAL_SHA,
            wire_contract_version="roborender.dagger.rtx.v2",
        )


def test_generic_launcher_has_no_historical_one_run_identity_constants():
    launcher = (PROJECT_ROOT / "scripts/launch_native_raven.sh").read_text(
        encoding="utf-8"
    )
    trainer = (
        PROJECT_ROOT / "code/src/dagger_rtx/native_service_trainer.py"
    ).read_text(encoding="utf-8")
    for source in (launcher, trainer):
        assert "36138c34d68378b26265a9ff1f0e854ca1f6bc4c4bb5e582a2a7a9301f9fea33" not in source
        assert "tuple(range(18720, 18728))" not in source
        assert "production_config.get(\"seed\") != 4242" not in source
        assert "I_ACKNOWLEDGE_FRESH_PRODUCTION_CONTINUATION_TO_WINDOW_BUDGET" in source
