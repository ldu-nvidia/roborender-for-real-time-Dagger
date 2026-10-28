from __future__ import annotations
from test.deployment.source_paths import RTX_ROOT

import importlib.util
from pathlib import Path
import sys

import pytest


SCRIPT = RTX_ROOT / "scripts" / "follow_production_pipeline.py"
SPEC = importlib.util.spec_from_file_location("follow_production_pipeline", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

RUN_ID = "dagger-r1-production-gpu1-b28-20260910-v2-r3"


def progress(step=0, phase="capture_teacher", event="start", durable=0, status="running"):
    return {
        "schema": MODULE.EXPECTED_PROGRESS_SCHEMA,
        "run_id": RUN_ID,
        "vector_step": step,
        "phase": phase,
        "event": event,
        "status": status,
        "elapsed_ms": None,
        "phase_completion_counts": {"vector_step_barrier": durable},
    }


def driver(phase="WINDOW_RUNNING", iteration=1, completed=0):
    return {
        "schema": MODULE.EXPECTED_DRIVER_SCHEMA,
        "phase": phase,
        "current_iteration_id": iteration,
        "completed_windows": [{} for _ in range(completed)],
        "spec": {
            "run_id": RUN_ID,
            "num_envs": 28,
            "rollout_steps": 81,
            "total_windows": 40,
        },
    }


@pytest.mark.parametrize(
    ("phase", "text"),
    [
        ("capture_teacher", "collect 3-view conditions"),
        ("prepare_encode_send_wait", "Image3F render -> student action"),
        ("native_env_step", "env.step([28,26])"),
        ("commit_encode_send_wait", "send 28 outcomes"),
        ("seal_local_audits", "28 x 81-transition"),
    ],
)
def test_key_phase_labels(phase, text):
    view = MODULE.build_view(RUN_ID, progress(phase=phase), driver())
    assert text in view.stage
    assert view.health == "healthy"


def test_global_step_maps_to_window_and_local_step():
    view = MODULE.build_view(RUN_ID, progress(step=81, durable=81), driver(iteration=2, completed=1))
    assert (view.window, view.local_step, view.iteration) == (2, 1, 2)
    assert view.durable_steps == 81
    rendered = MODULE.format_view(view)
    assert "window 02/40" in rendered
    assert "step 01/81" in rendered
    assert "2268/90720 transitions" in rendered


def test_waiting_driver_uses_h200_learning_status():
    h200 = {"run": {"status": "learning"}, "iterations": []}
    view = MODULE.build_view(
        RUN_ID, progress(step=80, durable=81), driver("WAITING_FOR_CHECKPOINT_ADVANCE"), h200
    )
    assert "optimizing model" in view.stage
    assert not view.terminal


def test_fail_closed_driver_has_precedence_over_stale_progress():
    state = driver("CRASH_UNCERTAIN")
    state["uncertain_reason"] = "test uncertainty"
    view = MODULE.build_view(RUN_ID, progress(), state)
    assert view.failed and view.terminal
    assert "FAIL-CLOSED" in view.stage
    assert "test uncertainty" in view.detail


def test_final_collection_reports_validation_not_experiment_complete():
    h200 = {"run": {"status": "validating"}, "iterations": []}
    view = MODULE.build_view(
        RUN_ID, progress(step=3239, durable=3240), driver("COMPLETE", 40, 40), h200
    )
    assert "validation" in view.stage
    assert not view.terminal


def test_validation_metric_is_terminal():
    h200 = {
        "run": {"status": "complete", "validation_goal_reached_rate": 0.625},
        "iterations": [],
    }
    view = MODULE.build_view(
        RUN_ID, progress(step=3239, durable=3240), driver("COMPLETE", 40, 40), h200
    )
    assert view.terminal and not view.failed
    assert "0.625" in view.detail


def test_validation_metric_can_come_from_iteration_row():
    h200 = {
        "run": {"status": "complete"},
        "iterations": [
            {"iteration_id": 39, "validation_goal_reached_rate": None},
            {"iteration_id": 40, "validation_goal_reached_rate": 0.75},
        ],
    }
    view = MODULE.build_view(
        RUN_ID, progress(step=3239, durable=3240), driver("COMPLETE", 40, 40), h200
    )
    assert view.terminal and not view.failed
    assert "0.75" in view.detail


def test_coordinator_unavailable_during_learning_is_attention():
    view = MODULE.build_view(
        RUN_ID,
        progress(step=80, durable=81),
        driver("WAITING_FOR_CHECKPOINT_ADVANCE"),
        {"_unavailable": "connection refused"},
    )
    assert view.health == "attention"
    assert "connection refused" in view.detail


def test_unknown_phase_is_never_guessed():
    view = MODULE.build_view(RUN_ID, progress(phase="future_phase"), driver())
    assert view.stage == "Pipeline: future_phase"


def test_phase_error_is_visible_but_driver_remains_authoritative():
    view = MODULE.build_view(
        RUN_ID,
        progress(event="error", status="error"),
        driver("WINDOW_RUNNING"),
    )
    assert view.health == "attention"
    assert not view.failed
    assert not view.terminal


def test_identity_mismatch_fails_closed():
    bad = progress()
    bad["run_id"] = "another-run"
    with pytest.raises(MODULE.ProgressError, match="another run"):
        MODULE.build_view(RUN_ID, bad, driver())


def test_script_contains_no_runtime_write_primitives():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "atomic_write" not in source
    assert "os.replace" not in source
    assert 'connection.request(\n                "GET"' in source
    assert 'connection.request(\n                "POST"' not in source


def test_monitor_uses_50_window_budget_independently_of_validation_cadence():
    try:
        assert MODULE.configure_cardinality(
            {
                "execution": {"window_count": 50},
                "validation": {"interval_iterations": 40},
            }
        ) == 50
        state = driver(iteration=41, completed=40)
        state["spec"]["total_windows"] = 50
        view = MODULE.build_view(
            RUN_ID,
            progress(step=40 * 81, durable=40 * 81),
            state,
        )
        rendered = MODULE.format_view(view)
        assert "iter 41/50" in rendered
        assert "window 41/50" in rendered
    finally:
        MODULE.configure_cardinality(
            {
                "execution": {"window_count": 40},
                "validation": {"interval_iterations": 40},
            }
        )


def test_monitor_accepts_future_round_validation_at_50():
    try:
        assert MODULE.configure_cardinality({
            "execution": {"window_count": 50},
            "validation": {"interval_iterations": 50},
        }) == 50
        assert MODULE.TOTAL_VECTOR_STEPS == 50 * 81
    finally:
        MODULE.configure_cardinality({
            "execution": {"window_count": 40},
            "validation": {"interval_iterations": 40},
        })
