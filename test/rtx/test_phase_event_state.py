"""Run the actual pinned Lab3 reset-trigger method without Isaac or GPU work."""
import ast
import hashlib
import inspect
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import pytest
import torch

from dagger_rtx.phase_reset_state import ColdPhaseState

REFERENCE = Path(__file__).resolve().parents[2] / (
    "runtime/migrations/alpha-to-beta-20261008/IsaacLab-3.0.0-beta2/"
    "source/isaaclab/isaaclab/managers/event_manager.py")


def original_apply():
    source = REFERENCE.read_bytes()
    assert hashlib.sha256(source).hexdigest() == "d90c82254a950baa1fbec336557cf4a6c50a2d2b05d61e094a3da50802c55ddc"
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "EventManager")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "apply")
    ns = dict(torch=torch, inspect=inspect, logger=logging.getLogger(__name__), Sequence=Sequence)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(REFERENCE), "exec"), ns)
    return ns["apply"]


def environment(batch=16, minimum=20):
    env = SimpleNamespace(common_step_counter=0, log_dict={}, colors=torch.zeros(batch))

    def randomize(target, ids):
        target.colors[ids] = torch.rand(target.colors[ids].shape)

    manager = SimpleNamespace(
        _mode_term_names={"reset": ["robot_material"]},
        _mode_term_cfgs={"reset": [SimpleNamespace(func=randomize, params={}, min_step_count_between_reset=minimum)]},
        _mode_class_term_cfgs={"reset": []},
        _reset_term_last_triggered_step_id=[torch.zeros(batch, dtype=torch.int32)],
        _reset_term_last_triggered_once=[torch.zeros(batch, dtype=torch.bool)],
        _env=env, _is_scene_entities_resolved=True,
    )
    env.simulator = SimpleNamespace(event_manager=manager)
    return env


@pytest.mark.parametrize("batch", [1, 7, 16, 32])
@pytest.mark.parametrize("minimum", [0, 20])
def test_warm_phase_matches_cold_reset_events_and_rng(batch, minimum):
    saved_rng = torch.get_rng_state()
    try:
        apply = original_apply()
        cold, warm = environment(batch, minimum), environment(batch, minimum)
        snapshot = ColdPhaseState(warm)
        manager = warm.simulator.event_manager
        tensor_ids = [id(manager._reset_term_last_triggered_step_id[0]), id(manager._reset_term_last_triggered_once[0])]
        apply(manager, "reset", global_env_step_count=324)
        before_restore = torch.get_rng_state().clone()
        snapshot.restore(warm)
        assert torch.equal(torch.get_rng_state(), before_restore)
        assert tensor_ids == [id(manager._reset_term_last_triggered_step_id[0]), id(manager._reset_term_last_triggered_once[0])]
        torch.manual_seed(901)
        apply(cold.simulator.event_manager, "reset", global_env_step_count=cold.common_step_counter)
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(901)
        apply(manager, "reset", global_env_step_count=warm.common_step_counter)
        assert torch.equal(cold.colors, warm.colors)
        assert torch.equal(torch.get_rng_state(), expected_rng)
        # Restore is boundary-only: normal reset throttling remains unchanged.
        if minimum:
            state = torch.get_rng_state().clone()
            apply(manager, "reset", global_env_step_count=1)
            assert torch.equal(torch.get_rng_state(), state)
    finally:
        torch.set_rng_state(saved_rng)


def test_old_clock_only_restore_reproduces_suppressed_randomization():
    env = environment()
    apply = original_apply()
    manager = env.simulator.event_manager
    apply(manager, "reset", global_env_step_count=324)
    previous = env.colors.clone()
    before = torch.get_rng_state().clone()
    env.common_step_counter = 0  # Previous warm boundary behavior.
    apply(manager, "reset", global_env_step_count=0)
    assert torch.equal(env.colors, previous)
    assert torch.equal(torch.get_rng_state(), before)


def test_snapshot_copies_actual_cold_values_not_assumed_zero():
    env = environment()
    manager = env.simulator.event_manager
    manager._reset_term_last_triggered_step_id[0][3] = 7
    manager._reset_term_last_triggered_once[0][3] = True
    snapshot = ColdPhaseState(env)
    manager._reset_term_last_triggered_step_id[0].fill_(324)
    snapshot.restore(env)
    assert manager._reset_term_last_triggered_step_id[0].tolist() == [0, 0, 0, 7] + [0] * 12
    assert manager._reset_term_last_triggered_once[0].sum() == 1


@pytest.mark.parametrize("fault", ["manager", "names", "count", "shape", "dtype"])
def test_changed_event_topology_fails_closed_before_mutation(fault):
    env = environment()
    snapshot = ColdPhaseState(env)
    manager = env.simulator.event_manager
    manager._reset_term_last_triggered_step_id[0].fill_(324)
    if fault == "manager":
        env.simulator.event_manager = environment().simulator.event_manager
    elif fault == "names":
        manager._mode_term_names["reset"] = ["different"]
    elif fault == "count":
        manager._reset_term_last_triggered_once.append(torch.zeros(16, dtype=torch.bool))
    elif fault == "shape":
        manager._reset_term_last_triggered_once[0] = torch.zeros(8, dtype=torch.bool)
    else:
        manager._reset_term_last_triggered_once[0] = torch.zeros(16)
    with pytest.raises(RuntimeError, match="cold phase event"):
        snapshot.restore(env)
    assert manager._reset_term_last_triggered_step_id[0].eq(324).all()


@pytest.mark.parametrize("mode", ["startup", "prestartup", "interval", "class"])
def test_unsupported_event_state_is_not_silently_certified(mode):
    env = environment()
    manager = env.simulator.event_manager
    if mode == "class":
        manager._mode_class_term_cfgs["reset"] = [object()]
    else:
        manager._mode_term_names[mode] = ["unknown"]
    with pytest.raises(RuntimeError, match="unsupported"):
        ColdPhaseState(env)
