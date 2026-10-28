from test.deployment.source_paths import SUCCESS_RATE_ROOT
from test.deployment.source_paths import ACTOR_ROOT, REPO_ROOT
import importlib.util
import ast
import json
from pathlib import Path

import pytest

from pipeline.success_rate import episode_metrics as diag


def test_terminal_goal_loss_is_visible_without_changing_legacy():
    accounting = diag.FirstEpisodeAccounting(1)
    accounting.update(0, [True], [False], [True], [False], [{}])
    result = accounting.result()
    assert result["native_successes"] == 1
    assert result["exported_successes"] == 0
    assert result["accounting_disagreements"] == 1


def test_first_episode_only_and_persistent_goal():
    accounting = diag.FirstEpisodeAccounting(2)
    accounting.update(0, [True, False], [True, False], [False, True], [False, False], [{}, {}])
    accounting.update(1, [False, True], [False, True], [True, True], [False, False], [{}, {}])
    assert accounting.result()["complete"]
    assert accounting.result()["native_successes"] == 1
    assert accounting.result()["exported_successes"] == 1
    assert accounting.result()["outcomes"][1]["steps"] == 1


def test_incomplete_not_certified():
    assert not diag.FirstEpisodeAccounting(5).result()["complete"]


def test_matching_state_gate_is_not_seed_only():
    diag.require_paired_states({"x": {"sha256": "a"}}, {"x": {"sha256": "a"}})
    with pytest.raises(ValueError, match="initial states do not match"):
        diag.require_paired_states({"x": {"sha256": "a"}}, {"x": {"sha256": "b"}})
    with pytest.raises(ValueError, match="fields differ"):
        diag.require_paired_states({"x": {}}, {"y": {}})


def test_actual_raven_aggregation_replays_first_completed_cases(tmp_path):
    pytest.importorskip("torch")
    audit_spec = importlib.util.spec_from_file_location(
        "audit_diag", ACTOR_ROOT / "audit_success_diagnostic.py")
    audit = importlib.util.module_from_spec(audit_spec)
    audit_spec.loader.exec_module(audit)
    trace = tmp_path / "steps.jsonl"
    rows = [
        {"exported_goals": [True, False, False, False, False],
         "lanes": [{"reset_buf": False}] * 5},
        {"exported_goals": [False] * 5,
         "lanes": [{"reset_buf": True}] * 5},
    ]
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
    source = (REPO_ROOT / "pipeline/GR00T-VisualSim2Real/gr00t/rl/envs/loco_manip/walk_stand_grasp.py")
    result = audit.replay(trace, source)
    assert result["completed"] == 5
    assert result["successes"] == 1
    assert result["episode_lengths"] == [2] * 5


def test_instrumentation_accepts_native_scalar_ground_height():
    torch = pytest.importorskip("torch")
    import numpy as np
    source = SUCCESS_RATE_ROOT / "evaluation_loop.py"
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "cpu")
    namespace = {"np": np, "torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    assert namespace["cpu"](0.0).shape == ()
    tensor = torch.tensor([True, False])
    copy = namespace["cpu"](tensor)
    tensor[:] = False
    assert copy.tolist() == [True, False]


def test_launcher_preserves_domain_randomization_interpolation():
    # The retired diagnostic CLI has no unreachable launcher body; its shared
    # pure command builder remains available for historical receipt tests.
    source = (ACTOR_ROOT / "run_paired_student_success.py").read_text()
    assert '"domain_rand.image_augmentation.enabled=false"' in source
    assert '"++env.config.domain_rand.image_augmentation.enabled=false"' not in source
    assert '"env._target_=isaac6_success_diagnostic.SuccessAccountingEnvironment"' in source
    assert '"+env._target_=isaac6_success_diagnostic.SuccessAccountingEnvironment"' not in source


def test_paired_student_launcher_pins_both_candidates_and_never_trains():
    source = (ACTOR_ROOT / "run_paired_student_success.py").read_text()
    assert '"--bc-sha256"' in source and '"--dagger-sha256"' in source
    assert 'Legacy ray-traced student evaluation is retired' in source
    assert 'DIAGNOSTIC_CHECKPOINT_SHA256={checkpoint_sha}' in source
    assert 'memory_gib=40' in source
    assert '"domain_rand.image_augmentation.enabled=false"' in source
    assert '"DIAGNOSTIC_REFERENCE_DIR=/paired-reference"' in source


def test_runtime_checkpoint_and_reset_pairing_are_explicit():
    source = (SUCCESS_RATE_ROOT / "evaluation_loop.py").read_text()
    assert 'os.environ.get("DIAGNOSTIC_CHECKPOINT_SHA256", BC_SHA)' in source
    assert 'require_paired_states(' in source
    assert '"checkpoint_sha256": checkpoint_sha' in source


def test_shared_cold_scene_restores_commands_and_refreshes_before_observations():
    from types import SimpleNamespace
    import os
    from initial_robot_refresh_diagnostic import CLEAR_ONCE, ORIGINAL, validate_mode
    torch = pytest.importorskip("torch")
    source = ACTOR_ROOT / "isaac6_panel_evaluator.py"
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PanelResetEnvironment")
    calls = []

    class Base:
        def reset_all(self):
            from initial_reset_contract import begin_initial_reset, complete_initial_reset
            begin_initial_reset(self)
            self.need_to_refresh_envs = torch.ones(self.num_envs, dtype=torch.bool)
            assert self.common_step_counter == 0
            manager = self.simulator.event_manager
            assert all(torch.count_nonzero(c) == 0 for c in manager._reset_term_last_triggered_step_id)
            assert all(not c.any() for c in manager._reset_term_last_triggered_once)
            calls.append("native_reset_states")
            self._refresh_sim_tensors()
            calls.append("native_observations")
            complete_initial_reset(self)
            return "observations"

        def _refresh_sim_tensors(self):
            calls.append("read_native_tensors")

    namespace = {"SuccessAccountingEnvironment": Base, "torch": torch,
                 "quat_rotate": lambda q, v: v, "os": os,
                 "ORIGINAL_INITIAL_REFRESH": ORIGINAL, "CLEAR_ONCE": CLEAR_ONCE,
                 "validate_initial_refresh": validate_mode}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    env = namespace["PanelResetEnvironment"]()
    env.simulator = SimpleNamespace(
        sim=SimpleNamespace(cfg=SimpleNamespace(dt=.002), forward=lambda: calls.append("forward_no_integration")),
        scene=SimpleNamespace(reset=lambda: calls.append("reset_scene_buffers"),
                              update=lambda dt: calls.append(("refresh_scene", dt))),
        _rigid_body_pos=torch.ones(5, 1, 3), base_quat=torch.zeros(5, 4),
        event_manager=SimpleNamespace(
            _reset_term_last_triggered_step_id=[torch.full((5,), 157, dtype=torch.int64)],
            _reset_term_last_triggered_once=[torch.ones(5, dtype=torch.bool)]),
        hold_object_to_hand_contact_forces=torch.ones(5, 1, 3),
        get_task_root_state=lambda name: torch.zeros(5, 13))
    env.history_handler = SimpleNamespace(history={"homie": -torch.zeros(5, 430)})
    env.num_envs, env.pelvis_idx, env.device = 5, 0, "cpu"
    env.config = SimpleNamespace(lift_goal_offset_pelvis=[.1, .2, .3],
                                 task=SimpleNamespace(front_object="front_object"))
    env.lift_goal_buf = torch.full((5, 3), 900.)
    env.last_lift_distance = torch.zeros(5)
    for name in ("_homie_commands", "_homie_commands_unclipped", "_last_homie_commands"):
        setattr(env, name, torch.zeros(5, 7))
    assert env._reset_cold_scene() == "observations"
    for name in env._panel_cold_commands:
        getattr(env, name).fill_(7)
    env.common_step_counter = 5000
    env.simulator.event_manager._reset_term_last_triggered_step_id[0][:] = torch.tensor([4985, 4999, 0, 4980, 4995])
    env.simulator.event_manager._reset_term_last_triggered_once[0].fill_(True)
    calls.clear()
    assert env._reset_cold_scene() == "observations"
    assert calls == ["reset_scene_buffers", "native_reset_states", "forward_no_integration",
                     ("refresh_scene", .002), "read_native_tensors", "native_observations"]
    assert all(torch.count_nonzero(getattr(env, n)) == 0 for n in env._panel_cold_commands)
    assert not torch.signbit(env.history_handler.history["homie"]).any()
    assert torch.count_nonzero(env.simulator.hold_object_to_hand_contact_forces) == 0
    assert torch.allclose(env.lift_goal_buf, torch.tensor([[1.1, 1.2, 1.3]]).repeat(5, 1))
    assert torch.allclose(env.last_lift_distance, env.lift_goal_buf.norm(dim=-1))
    env.simulator.hold_object_to_hand_contact_forces.fill_(7)
    env.lift_goal_buf.fill_(8)
    calls.clear()
    env._refresh_sim_tensors()
    assert calls == ["read_native_tensors"]  # ordinary physics steps unchanged
    assert torch.all(env.simulator.hold_object_to_hand_contact_forces == 7)
    assert torch.all(env.lift_goal_buf == 8)
    env.simulator.event_manager._reset_term_last_triggered_once[0] = torch.ones(4, dtype=torch.bool)
    with pytest.raises(RuntimeError, match="shape/dtype"):
        env._reset_cold_scene()


def test_pair_audit_rejects_duplicate_arms_and_initial_state_mismatch(tmp_path):
    pytest.importorskip("torch")
    audit_spec = importlib.util.spec_from_file_location(
        "audit_pair", ACTOR_ROOT / "audit_success_diagnostic.py")
    audit = importlib.util.module_from_spec(audit_spec)
    audit_spec.loader.exec_module(audit)
    with pytest.raises(ValueError, match="distinct"):
        audit.audit_pair(tmp_path, tmp_path / "unused", ("bc", "bc"))
    for arm, value in (("bc", "original"), ("dagger", "changed")):
        (tmp_path / arm).mkdir()
        (tmp_path / arm / "initial-state.json").write_text(json.dumps({"state": value}))
    with pytest.raises(RuntimeError, match="not byte-identical"):
        audit.audit_pair(tmp_path, tmp_path / "unused", ("bc", "dagger"))


@pytest.mark.parametrize("bad_field,bad_value", [
    ("complete", False), ("completed", 4), ("optimizer_steps", 1),
    ("policy_unchanged", False), ("accounting_disagreements", 1),
])
def test_pair_audit_refuses_invalid_live_evidence(tmp_path, bad_field, bad_value):
    pytest.importorskip("torch")
    audit_spec = importlib.util.spec_from_file_location(
        "audit_bad_evidence", ACTOR_ROOT / "audit_success_diagnostic.py")
    audit = importlib.util.module_from_spec(audit_spec)
    audit_spec.loader.exec_module(audit)
    for arm in ("bc", "dagger"):
        (tmp_path / arm).mkdir()
        (tmp_path / arm / "initial-state.json").write_text('{"state":"same"}')
    result = {"complete": True, "completed": 5, "optimizer_steps": 0,
              "policy_unchanged": True, "accounting_disagreements": 0}
    result[bad_field] = bad_value
    (tmp_path / "bc/result.json").write_text(json.dumps(result))
    with pytest.raises(RuntimeError, match="incomplete or mutated"):
        audit.audit_pair(tmp_path, tmp_path / "unused", ("bc", "dagger"))
