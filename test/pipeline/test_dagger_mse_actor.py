"""Native MSE barrier tests with fake windows; no Isaac/CUDA imports."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pipeline.bc_training.train_bc_h200 import EarlyStopping
from pipeline.dagger.beta_campaign import make_first_round_config, validate_actor_completion
from pipeline.dagger.mse_campaign import validate_mse_actor_completion
from pipeline.dagger.services.heldout_mse import RoundMSESelection, RESULT_SCHEMA
from pipeline.dagger.rtx.code.src.dagger_rtx.beta_round import canonical, actor_ids
from pipeline.dagger.rtx.code.src.dagger_rtx.mse_round_driver import MseBetaRoundDriver, read_mse_protocol
from pipeline.dagger.rtx.code.src.dagger_rtx.window_runner import RolloutLifecycle


@pytest.fixture
def inputs(tmp_path):
    checkpoint = tmp_path / "parent.pt"
    checkpoint.write_bytes(b"fake CPU fixture")
    panel = tmp_path / "panel.json"
    panel.write_text("{}")
    config = make_first_round_config("dagger-beta-mse-unit", seed=123, checkpoint=checkpoint,
        checkpoint_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), panel=panel,
        panel_sha=hashlib.sha256(panel.read_bytes()).hexdigest(), source_sha="b" * 64, gateway_pids=[101, 102, 103, 104])
    protocol = {"schema": "roborender.dagger-mse-round-protocol.v1", "run_id": config["run_id"], "round_id": 1,
                "initial_checkpoint_sha256": config["student"]["initial_checkpoint_sha256"],
                "validation_manifest_path": "/shared/fixed-validation.json", "validation_manifest_sha256": "a" * 64,
                "interval_iterations": 1, "max_iterations": 50, "min_validation_epochs": 1,
                "patience": 1, "relative_improvement": .005, "absolute_improvement": 1e-6,
                "final_test": {"scenario_manifest_sha256": config["validation"]["scenario_manifest_sha256"],
                               "scenario_ids": ["test-0"], "observation_mode": "raytraced"}}
    path = tmp_path / "mse.json"
    path.write_bytes(canonical(protocol))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return config, protocol, path, sha


class FakeMSECoordinator:
    def __init__(self, config, protocol):
        self.config, self.protocol = config, protocol
        self.parent = config["student"]["initial_checkpoint_sha256"]
        self.rows = [{"round_id": 1, "iteration_id": 1, "status": "collecting",
                      "input_checkpoint_sha256": self.parent, "candidate_checkpoint_sha256": None}]
        self.barriers, self.posts, self.wait_polls = [], 0, 0
        self.selected = RoundMSESelection(1, protocol["validation_manifest_sha256"],
                                          EarlyStopping(min_epochs=1, patience=1))
        self.selected.record(0, self.result(self.parent, 1.0))
        self.state, self.stop_reason = "active", None
        self.bad_protocol, self.bad_selected, self.missing_barrier = False, False, False

    def result(self, checkpoint, mse):
        return {"schema": RESULT_SCHEMA, "purpose": "validation", "round_id": 1,
                "validation_manifest_sha256": self.protocol["validation_manifest_sha256"],
                "checkpoint_sha256": checkpoint, "mse": mse, "per_action_mse": [mse] * 11,
                "frames": 81, "elements": 891, "chunk_count": 1, "inference_only": True,
                "selection_metric": "action_mse", "test_payloads_opened": False}

    def context(self):
        selected = self.selected.state_dict()
        if self.bad_selected:
            selected["best_checkpoint_sha256"] = "f" * 64
        row = self.rows[-1]
        return {"ok": True, "mse_round": {"run_id": self.config["run_id"], "round_id": 1,
            "protocol_sha256": "0" * 64 if self.bad_protocol else hashlib.sha256(canonical(self.protocol)).hexdigest(),
            "protocol": self.protocol, "selection": selected, "status": self.state,
            "stop_reason": self.stop_reason, "iteration_id": row["iteration_id"],
            "checkpoint_sha256": row["candidate_checkpoint_sha256"] or row["input_checkpoint_sha256"]}}

    def __call__(self, method, path, body=None):
        if path.startswith("/v1/mse-rounds/"):
            return copy.deepcopy(self.context())
        if method == "POST":
            self.posts += 1
            row = self.rows[-1]
            iteration = row["iteration_id"]
            assert body["checkpoint_sha256"] == self.parent
            if self.posts % self.config["rtx_actors"][0]["concurrent_environments"] == 0:
                self.barriers.extend({"round_id": 1, "iteration_id": iteration, "actor_id": actor,
                    "checkpoint_sha256": self.parent, "chunk_ids": [f"d1-{actor}-window-{iteration-1:06d}"]}
                    for actor in actor_ids(self.config))
                row.update(status="awaiting_mse_validation", candidate_checkpoint_sha256=str(iteration) * 64)
                self.wait_polls = 0
            return {"ok": True, "expected": len(actor_ids(self.config)), "iteration_id": iteration}
        row = self.rows[-1]
        if row["status"] == "awaiting_mse_validation":
            self.wait_polls += 1
            if self.wait_polls == 3:
                iteration, candidate = row["iteration_id"], row["candidate_checkpoint_sha256"]
                self.selected.record(iteration, self.result(candidate, .9 if iteration == 1 else .91))
                if iteration == 2:
                    row["status"] = "selected_pending_test"
                    self.state, self.stop_reason = "selected_pending_test", "validation_mse_plateau"
                else:
                    row["status"] = "advanced"
                    self.parent = candidate
                    self.rows.append({"round_id": 1, "iteration_id": iteration + 1, "status": "collecting",
                                      "input_checkpoint_sha256": candidate, "candidate_checkpoint_sha256": None})
        return copy.deepcopy({"ok": True, "run": {"config_sha256": hashlib.sha256(canonical(self.config)).hexdigest(),
            "actor_ids": actor_ids(self.config), "current_round": 1, "beta": .75, "rollout_steps": 81,
            "validation_interval_iterations": self.config["validation"]["interval_iterations"], "status": self.rows[-1]["status"],
            "active_checkpoint_sha256": self.parent}, "iterations": self.rows,
            "barriers": self.barriers[:-1] if self.missing_barrier else self.barriers})


def make_driver(tmp_path, inputs, control=None):
    config, protocol, path, sha = inputs
    read_mse_protocol(path, sha, config)
    begin, finish = Mock(), Mock()
    runner = SimpleNamespace(num_envs=6, actor_prefix=config["run_id"] + "-g1", phase_telemetry=None,
                             lifecycle=RolloutLifecycle(begin, finish),
                             run_window=Mock(return_value={"transition_batches": 81, "physical_transition_rows": 486}))
    control = control or FakeMSECoordinator(config, protocol)
    driver = MseBetaRoundDriver(runner, control, tmp_path / "driver", config,
                               hashlib.sha256(canonical(config)).hexdigest(), 1,
                               mse_protocol=protocol, mse_protocol_file_sha256=sha, sleep=lambda _: None)
    return driver, runner, begin, finish, control


def test_actor_waits_for_mse_and_stops_early_only_after_full_window(inputs, tmp_path):
    driver, runner, begin, finish, control = make_driver(tmp_path, inputs)
    result = driver.run()
    assert result["phase"] == "COMPLETE"
    assert result["completed_windows"] == 2 and result["declared_max_iterations"] == 50
    assert result["transition_batches"] == 162 and result["physical_transition_rows"] == 972
    assert result["final_candidate_checkpoint_sha256"] == "2" * 64
    assert result["selected_checkpoint_sha256"] == "1" * 64
    assert runner.run_window.call_count == 2
    begin.assert_called_once()
    finish.assert_called_once()
    assert control.posts == 12


def test_actor_sidecar_and_remote_identity_are_pre_lifecycle_gates(inputs, tmp_path):
    config, protocol, path, sha = inputs
    with pytest.raises(ValueError, match="file digest"):
        read_mse_protocol(path, "0" * 64, config)
    changed = copy.deepcopy(protocol)
    changed["max_iterations"] = 51
    path.write_bytes(canonical(changed))
    with pytest.raises(ValueError, match="maximum window"):
        read_mse_protocol(path, hashlib.sha256(path.read_bytes()).hexdigest(), config)
    path.write_bytes(canonical(protocol))
    control = FakeMSECoordinator(config, protocol)
    control.bad_protocol = True
    driver, runner, begin, finish, _ = make_driver(tmp_path, inputs, control)
    with pytest.raises(RuntimeError, match="protocol/baseline"):
        driver.run()
    begin.assert_not_called()
    runner.run_window.assert_not_called()


@pytest.mark.parametrize("fault", ["missing_barrier", "bad_selected"])
def test_actor_never_advances_through_bad_selection_or_barrier(inputs, tmp_path, fault):
    control = FakeMSECoordinator(*inputs[:2])
    setattr(control, fault, True)
    driver, runner, begin, finish, _ = make_driver(tmp_path, inputs, control)
    with pytest.raises(RuntimeError, match="barrier lineage|selected checkpoint"):
        driver.run()
    assert driver.state["phase"] == "CRASH_UNCERTAIN"
    assert runner.run_window.call_count <= 2
    finish.assert_called_once()


def test_separate_campaign_admits_early_selection_but_legacy_stays_fifty(inputs, tmp_path):
    config, _, _, sha = inputs
    driver, _, _, _, control = make_driver(tmp_path, inputs)
    first = driver.run()
    second = copy.deepcopy(first)
    second["physical_gpu"] = 2
    receipt = {"actors": [first, second], "actor_ids": actor_ids(config), "owned_actors_stopped": True}
    context = control.context()["mse_round"]
    report = validate_mse_actor_completion(config, receipt, control.rows, context, sha)
    assert report["completed_iterations"] == 2
    assert report["selected_checkpoint_sha256"] == "1" * 64
    assert report["last_candidate_checkpoint_sha256"] == "2" * 64
    with pytest.raises(RuntimeError, match="all declared iterations"):
        validate_actor_completion(config, receipt, control.rows)
    second["physical_transition_rows"] -= 1
    with pytest.raises(RuntimeError, match="receipt differs"):
        validate_mse_actor_completion(config, receipt, control.rows, context, sha)
