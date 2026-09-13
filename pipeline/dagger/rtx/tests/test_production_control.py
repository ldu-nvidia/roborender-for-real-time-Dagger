from __future__ import annotations

import hashlib

import pytest

from dagger_rtx.production_control import CoordinatorHTTPControl
from dagger_rtx.production_driver import (
    H200ContinuationActivityAttestation,
    ProductionContinuationSegment,
    ProductionDriverError,
    ProductionRunSpec,
)


RUN_ID = "dagger-r1-production-gpu1-b28-20260910-v2"
CONFIG_SHA = "42f6d681a565ffd9146a1787a61e0775d7546367eeeaa8d9f5b512142c570514"
INITIAL_SHA = "4b7fbc3b10c5a3a6381fd6a89bd5ac779e045fcae8789769d2f0c084f4570a42"
PREFIX = f"{RUN_ID}-g1"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _spec(*, total_windows: int = 40) -> ProductionRunSpec:
    return ProductionRunSpec(
        run_id=RUN_ID,
        config_sha256=CONFIG_SHA,
        actor_prefix=PREFIX,
        actor_ids=tuple(f"{PREFIX}-lane-{lane:03d}" for lane in range(28)),
        initial_checkpoint_sha256=INITIAL_SHA,
        total_windows=total_windows,
    )


def _status(spec, *, iteration=1, run_status="collecting", checkpoint=INITIAL_SHA,
            completed_status="collecting", candidate=None, barriers=()):
    rows = [{
        "run_id": spec.run_id,
        "round_id": 1,
        "iteration_id": iteration,
        "input_checkpoint_sha256": checkpoint,
        "candidate_checkpoint_sha256": candidate,
        "status": completed_status,
    }]
    return {
        "ok": True,
        "run": {
            "run_id": spec.run_id,
            "config_sha256": spec.config_sha256,
            "current_round": 1,
            "beta": .75,
            "rollout_steps": 81,
            "actor_ids": list(spec.actor_ids),
            "initial_checkpoint_sha256": INITIAL_SHA,
            "active_checkpoint_sha256": checkpoint,
            "validation_interval_iterations": 40,
            "status": run_status,
        },
        "iterations": rows,
        "barriers": list(barriers),
    }


def _control(tmp_path):
    token = tmp_path / "token"
    token.write_text("unit-test-token")
    token.chmod(0o600)
    return CoordinatorHTTPControl("http://127.0.0.1:18610", token, sleep=lambda _: None)


def test_initial_attestation_is_exact_and_read_only(tmp_path, monkeypatch):
    spec = _spec()
    control = _control(tmp_path)
    monkeypatch.setattr(control, "_request", lambda method, path, payload=None: _status(spec))
    gate = control.attest_initial(spec)
    assert gate.active_iteration_id == 1
    assert gate.active_checkpoint_sha256 == INITIAL_SHA
    assert gate.actor_ids == spec.actor_ids


def test_continuation_attestation_requires_advanced_predecessor_and_empty_current(
    tmp_path, monkeypatch
):
    spec = _spec()
    checkpoint = _sha("candidate-1")
    segment = ProductionContinuationSegment(
        segment_id="segment-from-window-000001",
        run_id=spec.run_id,
        config_sha256=spec.config_sha256,
        start_global_window=1,
        expected_checkpoint_sha256=checkpoint,
        simulator_seed=42001,
        episode_serial_base=10_100_000,
        episode_serial_limit_exclusive=10_200_000,
        reset_reason="crash_uncertain",
        predecessor_state_path="/tmp/failed-state.json",
        predecessor_state_sha256=_sha("failed-state"),
        h200_activity_attestation=H200ContinuationActivityAttestation(
            run_id=spec.run_id,
            config_sha256=spec.config_sha256,
            round_id=1,
            iteration_id=2,
            rollout_window_id="window-000001",
            prepare_count=0,
            commit_count=0,
            chunk_count=0,
            actor_barrier_count=0,
            observed_at_utc="2026-09-10T09:00:00Z",
        ),
        h200_activity_evidence_path="/tmp/h200-activity.json",
        h200_activity_evidence_sha256=_sha("activity"),
        prior_continuation_manifest_sha256s=(),
    )
    barriers = [
        {
            "run_id": spec.run_id,
            "round_id": 1,
            "iteration_id": 1,
            "actor_id": actor,
        }
        for actor in spec.actor_ids
    ]
    value = _status(
        spec,
        iteration=1,
        run_status="collecting",
        checkpoint=checkpoint,
        completed_status="advanced",
        candidate=checkpoint,
        barriers=barriers,
    )
    value["iterations"].append(
        {
            "run_id": spec.run_id,
            "round_id": 1,
            "iteration_id": 2,
            "input_checkpoint_sha256": checkpoint,
            "candidate_checkpoint_sha256": None,
            "status": "collecting",
        }
    )
    control = _control(tmp_path)
    monkeypatch.setattr(control, "_status", lambda _: value)

    gate = control.attest_continuation(spec, segment)
    assert gate.active_iteration_id == 2
    assert gate.active_checkpoint_sha256 == checkpoint

    value["barriers"].append(
        {
            "run_id": spec.run_id,
            "round_id": 1,
            "iteration_id": 2,
            "actor_id": spec.actor_ids[0],
        }
    )
    with pytest.raises(ProductionDriverError, match="continuation"):
        control.attest_continuation(spec, segment)


def test_posts_exactly_28_deterministic_actor_barriers(tmp_path, monkeypatch):
    spec = _spec()
    control = _control(tmp_path)
    calls = []
    barrier_rows = []

    def request(method, path, payload=None):
        if method == "POST":
            calls.append(payload)
            barrier_rows.append({
                "run_id": spec.run_id,
                "round_id": 1,
                "iteration_id": 1,
                "actor_id": payload["actor_id"],
                "checkpoint_sha256": INITIAL_SHA,
                "chunk_ids": payload["chunk_ids"],
            })
            return {"ok": True, "iteration_id": 1, "arrived": len(calls),
                    "expected": 28, "ready_for_learning": len(calls) == 28}
        return _status(spec, barriers=barrier_rows)

    monkeypatch.setattr(control, "_request", request)
    gate = control.post_window_barriers(
        spec, completed_window_index=0, input_checkpoint_sha256=INITIAL_SHA
    )
    assert len(calls) == 28
    assert gate.ready_for_learning is True
    assert calls[0]["chunk_ids"] == [f"d1-{spec.actor_ids[0]}-window-000000"]
    assert calls[-1]["chunk_ids"] == [f"d1-{spec.actor_ids[-1]}-window-000000"]


def test_wait_accepts_intermediate_advance_and_final_validation(tmp_path, monkeypatch):
    spec = _spec()
    control = _control(tmp_path)
    candidate = _sha("candidate-1")
    value = _status(
        spec, iteration=1, run_status="collecting", checkpoint=candidate,
        completed_status="advanced", candidate=candidate,
        barriers=[{"round_id": 1, "iteration_id": 1}] * 28,
    )
    value["iterations"].append({
        "run_id": spec.run_id, "round_id": 1, "iteration_id": 2,
        "input_checkpoint_sha256": candidate,
        "candidate_checkpoint_sha256": None, "status": "collecting",
    })
    monkeypatch.setattr(control, "_status", lambda _: value)
    gate = control.wait_for_window_advance(
        spec, completed_window_index=0, previous_checkpoint_sha256=INITIAL_SHA
    )
    assert gate.next_iteration_id == 2
    assert gate.output_checkpoint_sha256 == candidate

    final_candidate = _sha("candidate-40")
    final = _status(
        spec, iteration=40, run_status="validating", checkpoint=candidate,
        completed_status="validating", candidate=final_candidate,
        barriers=[{"round_id": 1, "iteration_id": 40}] * 28,
    )
    monkeypatch.setattr(control, "_status", lambda _: final)
    gate = control.wait_for_window_advance(
        spec, completed_window_index=39, previous_checkpoint_sha256=candidate
    )
    assert gate.next_iteration_id is None
    assert gate.status == "validating"
    assert gate.output_checkpoint_sha256 == final_candidate


def test_50_window_budget_keeps_validation_at_40_and_accepts_final_iteration_50(
    tmp_path, monkeypatch
):
    spec = _spec(total_windows=50)
    control = _control(tmp_path)
    candidate_50 = _sha("candidate-50")
    value = _status(
        spec,
        iteration=50,
        run_status="collecting",
        checkpoint=candidate_50,
        completed_status="advanced",
        candidate=candidate_50,
        barriers=[{"round_id": 1, "iteration_id": 50}] * 28,
    )
    value["iterations"].append(
        {
            "run_id": spec.run_id,
            "round_id": 1,
            "iteration_id": 51,
            "input_checkpoint_sha256": candidate_50,
            "candidate_checkpoint_sha256": None,
            "status": "collecting",
        }
    )
    monkeypatch.setattr(control, "_status", lambda _: value)

    gate = control.wait_for_window_advance(
        spec,
        completed_window_index=49,
        previous_checkpoint_sha256=_sha("candidate-49"),
    )

    assert gate.completed_iteration_id == 50
    assert gate.next_iteration_id == 51
    assert gate.status == "collecting"
    assert gate.output_checkpoint_sha256 == candidate_50
