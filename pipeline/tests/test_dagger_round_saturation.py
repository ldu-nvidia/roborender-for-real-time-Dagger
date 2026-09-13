from __future__ import annotations

import hashlib
import json
import urllib.error

import pytest

from pipeline.dagger.production_controller import (
    RAVEN_FIDELITY_CONTRACT,
    Controller,
    get_with_transport_retry,
    load_config,
)
from pipeline.dagger.services.server import _validation_due, _validation_plateau


def test_production_validation_boundary_is_every_40_updates():
    assert not _validation_due(1, 40)
    assert not _validation_due(39, 40)
    assert _validation_due(40, 40)
    assert _validation_due(80, 40)
    with pytest.raises(ValueError):
        _validation_due(0, 40)
    with pytest.raises(ValueError):
        _validation_due(1, 0)


def test_all_rounds_use_same_validation_plateau_rule():
    for _round_id in range(1, 5):
        rows = [
            {
                "iteration_id": index,
                "candidate_checkpoint_sha256": f"{index:064x}",
                "evaluation_id": f"evaluation-{index}",
                "validation_goal_reached_rate": 0.50,
            }
            for index in range(1, 11)
        ]
        state = _validation_plateau(rows)
        assert state["saturated"]
        assert state["stop_reason"] == "success_rate_plateau"
        assert state["best_iteration_id"] == 1


def test_plateau_resets_patience_only_for_meaningful_improvement():
    rates = [0.50, 0.505, 0.52, 0.519, 0.515]
    rows = [
        {
            "iteration_id": index,
            "candidate_checkpoint_sha256": f"{index:064x}",
            "evaluation_id": f"evaluation-{index}",
            "validation_goal_reached_rate": rate,
        }
        for index, rate in enumerate(rates, 1)
    ]
    state = _validation_plateau(rows)
    assert not state["saturated"]  # minimum ten iterations applies uniformly
    assert state["best_iteration_id"] == 3
    assert state["bad_validations"] == 2


def test_round_stops_at_hard_maximum_even_when_every_iteration_improves():
    rows = [
        {
            "iteration_id": index,
            "candidate_checkpoint_sha256": f"{index:064x}",
            "evaluation_id": f"evaluation-{index}",
            "validation_goal_reached_rate": index / 50.0,
        }
        for index in range(1, 41)
    ]
    state = _validation_plateau(rows)
    assert state["saturated"]
    assert state["stop_reason"] == "maximum_validation_epochs"
    assert state["best_iteration_id"] == 40


def test_controller_submits_barrier_admitted_chunks_without_split_subprocesses():
    class Learner:
        calls = []

        def post(self, path, payload):
            self.calls.append((path, payload))
            return {"ok": True}

    controller = Controller.__new__(Controller)
    controller.run_id = "fresh-run"
    controller.round_id = 1
    controller.wandb_mode = "disabled"
    controller.actor_ids = [f"actor{index}" for index in range(28)]
    controller.learner = Learner()
    controller.config = {
        "execution": {"window_count": 40},
        "seed": 42,
        "teacher_beta": 0.75,
        "rtx_actor": {"gpu_uuid": "gpu1"},
        "student": {
            "learning_rate": 2e-4,
            "max_gradient_norm": 1.0,
            "mini_batches_per_iteration": 1,
        },
        "h200": {"render_student_worker_count": 8},
        "logging": {"wandb_project": "dagger", "wandb_run_name": "fresh-run"},
        "raven_fidelity": dict(RAVEN_FIDELITY_CONTRACT),
    }

    controller._submit_learning({
        "iteration_id": 1,
        "input_checkpoint_sha256": "a" * 64,
    }, {
        "barriers": [
            {
                "round_id": 1,
                "iteration_id": 1,
                "actor_id": f"actor{index}",
                "chunk_ids": [f"c{index}"],
            }
            for index in range(28)
        ],
    })

    assert len(controller.learner.calls) == 1
    path, payload = controller.learner.calls[0]
    assert path == "/v1/jobs"
    assert payload["train_chunk_ids"] == [f"c{index}" for index in range(28)]
    assert payload["parameters"]["iteration_id"] == 1
    assert payload["parameters"]["microbatch_environments"] == 28
    assert payload["parameters"]["raven_exact_full_batch"] is True


def test_control_get_retries_only_transient_transport_failure(monkeypatch):
    class FlakyClient:
        calls = 0

        def get(self, path):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("transient read timeout")
            return {"path": path, "ok": True}

    delays = []
    monkeypatch.setattr(
        "pipeline.dagger.production_controller.time.sleep", delays.append
    )
    client = FlakyClient()
    assert get_with_transport_retry(
        client, "/status", attempts=2, initial_backoff_seconds=0.25
    ) == {"path": "/status", "ok": True}
    assert client.calls == 2
    assert delays == [0.25]


def test_control_get_does_not_retry_protocol_error(monkeypatch):
    class BrokenClient:
        calls = 0

        def get(self, _path):
            self.calls += 1
            raise RuntimeError("semantic contract failure")

    monkeypatch.setattr(
        "pipeline.dagger.production_controller.time.sleep",
        lambda _delay: pytest.fail("semantic errors must not be retried"),
    )
    client = BrokenClient()
    with pytest.raises(RuntimeError, match="semantic contract failure"):
        get_with_transport_retry(client, "/status")
    assert client.calls == 1


def test_controller_reports_only_current_iteration_barriers():
    class Coordinator:
        def get(self, _path):
            return {
                "run": {"status": "collecting"},
                "iterations": [{"iteration_id": 16}],
                "barriers": [
                    {"round_id": 1, "iteration_id": 15},
                    {"round_id": 1, "iteration_id": 16},
                    {"round_id": 1, "iteration_id": 16},
                ],
            }

    controller = Controller.__new__(Controller)
    controller.coordinator = Coordinator()
    controller.run_id = "run"
    controller.round_id = 1
    controller.actor_ids = ["lane-0", "lane-1"]
    controller.config = {"execution": {"window_count": 40}}
    report = controller.tick()
    assert report["actor_barriers"] == 2
    assert report["expected_actor_barriers"] == 2


def test_controller_stops_cleanly_after_exact_50_window_budget():
    class Coordinator:
        def get(self, _path):
            return {
                "run": {"status": "collecting"},
                "iterations": [
                    {"iteration_id": 50, "status": "advanced"},
                    {"iteration_id": 51, "status": "collecting"},
                ],
                "barriers": [],
            }

    controller = Controller.__new__(Controller)
    controller.coordinator = Coordinator()
    controller.run_id = "fifty-window-run"
    controller.round_id = 1
    controller.actor_ids = [f"lane-{index}" for index in range(28)]
    controller.config = {"execution": {"window_count": 50}}

    report = controller.tick()

    assert report["iteration_id"] == 51
    assert report["action"] == "collection_budget_complete"
    assert report["actor_barriers"] == 0
    assert report["total_windows"] == 50


@pytest.mark.parametrize("invalid_total", [0, 39, True])
def test_controller_config_rejects_budget_below_validation_interval(
    tmp_path, invalid_total
):
    config = {
        "schema": "roborender.cross_gpu_dagger_round.v1",
        "round_id": 1,
        "teacher_beta": 0.75,
        "rtx_actor": {
            "physical_gpu_index": 1,
            "gpu_uuid": "GPU-a28701dd-27fa-67d2-8b76-23be7a57fcf9",
            "forbidden_gpu_indices": [0],
            "concurrent_environments": 28,
            "rollout_steps_per_iteration": 81,
            "transitions_per_iteration": 2268,
        },
        "raven_fidelity": dict(RAVEN_FIDELITY_CONTRACT),
        "validation": {
            "interval_iterations": 40,
            "minimum_validation_epochs": 10,
            "maximum_validation_epochs": 40,
            "patience": 4,
            "minimum_absolute_improvement": 0.01,
            "metric": "raven_goal_reached_rate",
        },
        "execution": {
            "window_count": invalid_total,
            "stop_after_actor_barriers": False,
            "wait_for_checkpoint_advance": True,
            "learner_submission_authorized": True,
            "validation_authorized": True,
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="window_count"):
        load_config(path)


def test_controller_config_accepts_50_budget_with_40_validation_cadence(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    panel = tmp_path / "panel.json"
    panel.write_text("{}", encoding="utf-8")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    panel_sha = hashlib.sha256(panel.read_bytes()).hexdigest()
    monkeypatch.setattr(
        "pipeline.dagger.production_controller.expand_validation_panel",
        lambda _value: [{} for _ in range(256)],
    )
    config = {
        "schema": "roborender.cross_gpu_dagger_round.v1",
        "round_id": 1,
        "teacher_beta": 0.75,
        "rtx_actor": {
            "physical_gpu_index": 1,
            "gpu_uuid": "GPU-a28701dd-27fa-67d2-8b76-23be7a57fcf9",
            "forbidden_gpu_indices": [0],
            "concurrent_environments": 28,
            "rollout_steps_per_iteration": 81,
            "transitions_per_iteration": 2268,
        },
        "raven_fidelity": dict(RAVEN_FIDELITY_CONTRACT),
        "validation": {
            "interval_iterations": 40,
            "minimum_validation_epochs": 10,
            "maximum_validation_epochs": 40,
            "patience": 4,
            "minimum_absolute_improvement": 0.01,
            "metric": "raven_goal_reached_rate",
            "scenario_manifest": str(panel),
            "scenario_manifest_sha256": panel_sha,
            "completed_episodes": 256,
        },
        "execution": {
            "window_count": 50,
            "stop_after_actor_barriers": False,
            "wait_for_checkpoint_advance": True,
            "learner_submission_authorized": True,
            "validation_authorized": True,
        },
        "student": {
            "initial_checkpoint": str(checkpoint),
            "initial_checkpoint_sha256": checkpoint_sha,
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    loaded = load_config(path)

    assert loaded["execution"]["window_count"] == 50
    assert loaded["validation"]["interval_iterations"] == 40
