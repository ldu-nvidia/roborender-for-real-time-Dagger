import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dagger_rtx.qualification_driver import (
    QualificationDriver, CommitAckLossProbe, actor_ids, read_config, validate_config,
)
from dagger_rtx.gateway_topology import require_gateway_count
from dagger_rtx.http_transport import TransientTransportError
from dagger_rtx.window_runner import RolloutLifecycle


def config():
    return {"schema": "roborender.beta-runtime-check.v1", "run_id": "rtx-beta-migration-runtime-test",
            "windows": 3, "num_envs_per_gpu": 6, "physical_rtx_gpus": [1, 2],
            "initial_checkpoint_sha256": "a" * 64, "seed": 610191, "gateway_pids": [1, 2, 3, 4],
            "source_sha256": "b" * 64, "teacher_beta": .75,
            "wandb_run_name": "smoke-dagger-test", "wandb_run_id": "test-runtime", "wandb_mode": "offline",
            "ack_loss_probe": {"gpu": 1, "window_index": 1, "step_index": 5}}


@pytest.mark.parametrize("field,value", [("windows", 50), ("windows", True), ("num_envs_per_gpu", 28),
    ("physical_rtx_gpus", [0, 1]), ("gateway_pids", [1, 1, 2, 3]), ("teacher_beta", .5),
    ("run_id", "production-run"), ("wandb_run_name", "BC"), ("source_sha256", "bad")])
def test_invalid_qualification_fails_closed(field, value):
    c = config()
    c[field] = value
    with pytest.raises((ValueError, RuntimeError)):
        validate_config(c)


def test_only_four_v3_workers_and_no_production_weakening():
    require_gateway_count("v3", "multi-window-check", 4)
    require_gateway_count("v3", "multi-window-check", 1)
    for protocol, mode, count in [("v4", "multi-window-check", 4), ("v3", "multi-window-check", 8),
                                  ("v3", "production", 4), ("v4", "production", 4)]:
        with pytest.raises(RuntimeError):
            require_gateway_count(protocol, mode, count)


def test_v2_single_actor_is_explicit_and_gpu1_only():
    c = config()
    c.update(schema="roborender.beta-runtime-check.v2", num_envs_per_gpu=8,
             physical_rtx_gpus=[1], gateway_pids=[123])
    assert validate_config(c) is c
    assert len(actor_ids(c)) == 8 and all("-g1-" in actor for actor in actor_ids(c))
    for field, value in [("physical_rtx_gpus", [2]), ("physical_rtx_gpus", [1, 2]),
                         ("num_envs_per_gpu", 28), ("gateway_pids", [1, 2, 3, 4])]:
        invalid = {**c, field: value}
        with pytest.raises(ValueError):
            validate_config(invalid)


def test_config_file_digest(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config()))
    with pytest.raises(ValueError, match="digest"):
        read_config(path, "0" * 64)


class FakeCoordinator:
    def __init__(self, c):
        self.c = c
        self.parent = c["initial_checkpoint_sha256"]
        self.rows = [{"round_id": 1, "iteration_id": 1, "status": "collecting",
                      "input_checkpoint_sha256": self.parent, "candidate_checkpoint_sha256": None}]
        self.barriers = []
        self.posts = 0

    def __call__(self, method, path, body=None):
        if method == "POST":
            self.posts += 1
            current = self.rows[-1]
            iteration = current["iteration_id"]
            assert body["checkpoint_sha256"] == current["input_checkpoint_sha256"]
            if self.posts % self.c["num_envs_per_gpu"] == 0:
                self.barriers.extend({"round_id": 1, "iteration_id": iteration, "actor_id": actor,
                                      "checkpoint_sha256": self.parent,
                                      "chunk_ids": [f"d1-{actor}-window-{iteration-1:06d}"]}
                                     for actor in actor_ids(self.c))
                candidate = str(iteration) * 64
                current.update(candidate_checkpoint_sha256=candidate,
                               status="awaiting_validation" if iteration == 3 else "advanced")
                if iteration < 3:
                    self.parent = candidate
                    self.rows.append({"round_id": 1, "iteration_id": iteration + 1, "status": "collecting",
                                      "input_checkpoint_sha256": candidate, "candidate_checkpoint_sha256": None})
            return {"ok": True, "expected": len(actor_ids(self.c)), "iteration_id": iteration}
        return copy.deepcopy({"ok": True, "run": {
            "config_sha256": "c" * 64, "actor_ids": list(actor_ids(self.c)), "current_round": 1,
            "beta": .75, "rollout_steps": 81, "validation_interval_iterations": 3,
            "status": self.rows[-1]["status"], "active_checkpoint_sha256": self.parent},
            "iterations": self.rows, "barriers": self.barriers})


@pytest.mark.parametrize("num_envs", [4, 6, 8])
def test_three_windows_one_lifecycle_with_checkpoint_gates(tmp_path, num_envs):
    c = config()
    c["num_envs_per_gpu"] = num_envs
    if num_envs == 8:
        c.update(schema="roborender.beta-runtime-check.v2", physical_rtx_gpus=[1], gateway_pids=[1])
    begin, finish = Mock(), Mock()
    run_window = Mock(return_value={"transition_batches": 81, "physical_transition_rows": 81 * num_envs})
    runner = SimpleNamespace(num_envs=num_envs, actor_prefix=c["run_id"] + "-g1", phase_telemetry=None,
                             lifecycle=RolloutLifecycle(begin, finish), run_window=run_window)
    control = FakeCoordinator(c)
    driver = QualificationDriver(runner, control, tmp_path, c, "c" * 64, 1)
    result = driver.run()
    assert result["phase"] == "COMPLETE" and len(result["windows"]) == 3
    assert result["transition_batches"] == 243 and control.posts == 3 * num_envs
    assert result["physical_transition_rows"] == 243 * num_envs
    assert [row["input_checkpoint_sha256"] for row in result["windows"]] == ["a" * 64, "1" * 64, "2" * 64]
    begin.assert_called_once()
    finish.assert_called_once()
    assert run_window.call_count == 3
    for index, call in enumerate(run_window.call_args_list):
        assert call.args == (index,) and call.kwargs == {"manage_lifecycle": False}
    with pytest.raises(RuntimeError, match="restart forbidden"):
        QualificationDriver(runner, control, tmp_path, c, "c" * 64, 1)


def test_actor_failure_is_uncertain_no_next_window(tmp_path):
    c = config()
    finish = Mock()
    runner = SimpleNamespace(num_envs=6, actor_prefix=c["run_id"] + "-g2", phase_telemetry=None,
                             lifecycle=RolloutLifecycle(Mock(), finish), run_window=Mock(side_effect=RuntimeError("native error")))
    driver = QualificationDriver(runner, FakeCoordinator(c), tmp_path, c, "c" * 64, 2)
    with pytest.raises(RuntimeError, match="native error"):
        driver.run()
    assert driver.state["phase"] == "CRASH_UNCERTAIN"
    assert runner.run_window.call_count == 1
    finish.assert_called_once()


@pytest.mark.parametrize('gateway_count', [1, 4])
def test_lost_ack_is_after_real_commit_and_only_once(tmp_path, gateway_count):
    closed = [Mock() for _ in range(gateway_count)]
    response = [{"receipt": "exact"}]
    transports = [SimpleNamespace(transport=SimpleNamespace(close=fn)) for fn in closed]
    client = SimpleNamespace(commit_batch=Mock(return_value=response))
    if gateway_count == 1:
        client.transport = transports[0].transport
    else:
        client.gateways = transports
    probe = CommitAckLossProbe(client, tmp_path / "fault.json")
    selections = [{"step_key": {"rollout_window_id": "window-000001", "rollout_step_index": 5}}]
    with pytest.raises(TransientTransportError):
        probe.commit_batch(selections, [{}], ["same-key"])
    assert probe.commit_batch(selections, [{}], ["same-key"]) == response
    assert client.commit_batch.call_count == 2
    assert all(fn.call_count == 1 for fn in closed)
    assert json.loads((tmp_path / "fault.json").read_text())["native_step_replay_allowed"] is False
