"""CPU regressions for corrected student evaluation, without native services."""
from test.deployment.source_paths import ACTOR_ROOT
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

HERE = ACTOR_ROOT
sys.path.insert(0, str(HERE))
import initial_robot_refresh_diagnostic as refresh
import explicit_motion_panel as panels
from teacher_action_diagnostic import diagnostic_step
from test.deployment.rtx_actor.test_explicit_motion_panel import panel
from test.deployment.rtx_actor.test_initial_robot_refresh_diagnostic import panel_environment, result as teacher_result


MODE = refresh.STUDENT_CLEAR_ONCE


def student_evidence(count=2, arm="bc"):
    env = SimpleNamespace(num_envs=count, need_to_refresh_envs=torch.ones(count, dtype=torch.bool))
    return refresh.clear_already_applied_initial_refresh(env, mode=MODE, arm=arm, teacher_action_mode="distill")


def mixed_panel():
    value = panel()
    value["cohorts"][-1]["assignment"]["lanes"] = value["cohorts"][-1]["assignment"]["lanes"][:1]
    value["cohorts"][-1]["case_ids"] = value["cohorts"][-1]["case_ids"][:1]
    value["episode_count"] = 3
    return value


def student_result(spec, arm="bc"):
    count = len(spec["case_ids"])
    successes = [lane == 0 for lane in range(count)]
    return {"complete": True, "completed": count, "expected": count, "optimizer_steps": 0,
        "policy_unchanged": True, "accounting_disagreements": 0,
        "seed": spec["assignment"]["simulator_seed"], "observation_mode": "raytraced_3view",
        "motion_assignment_sha256": hashlib.sha256(panels.canonical(spec["assignment"])).hexdigest(),
        "checkpoint_sha256": "b" * 64, "runtime_physics_sha256": "c" * 64,
        "arm": arm, "teacher_action_mode": "distill", "goal_accounting_version": panels.GOAL_ACCOUNTING_VERSION,
        "original_raven_metric_completed": count, "original_raven_metric_successes": sum(successes),
        "native_successes": sum(successes), "exported_successes": sum(successes),
        "outcomes": [{"lane": lane, "steps": 123, "native_success": success,
            "exported_success": success, "timeout": not success} for lane, success in enumerate(successes)],
        refresh.FIELD: MODE, "initial_robot_refresh_control": refresh.control_contract(MODE),
        "initial_robot_refresh_attestation": student_evidence(count, arm)}


def expected(arm="bc"):
    return dict(checkpoint_sha256="b" * 64, arm=arm, teacher_action_mode="distill",
                diagnostic_initial_robot_refresh=MODE)


def test_teacher_v1_contract_and_attestation_bytes_are_preserved():
    control = {"schema": "roborender.initial-robot-refresh-control.v1", "mode": "clear_already_applied_once",
        "scope": "independent_panel_initial_reset_only",
        "boundary": "after_original_reset_writes_forward_and_observations_before_first_policy_step",
        "operation": "clear_only_already_applied_pending_robot_refresh_flags_once",
        "ordinary_done_resets_modified": False, "robot_or_object_targets_modified": False,
        "policy_or_rng_modified": False, "extra_physics_steps": 0,
        "success_criteria_modified": False, "production_activation": False}
    env = SimpleNamespace(num_envs=2, need_to_refresh_envs=torch.ones(2, dtype=torch.bool))
    proof = {"schema": "roborender.initial-robot-refresh-attestation.v1", "control": control,
        "num_envs": 2, "pending_before": [True, True], "pending_after": [False, False],
        "cleared_lanes": 2, "clear_calls": 1, "native_steps_added": 0}
    assert refresh.canonical(refresh.control_contract()) == refresh.canonical(control)
    assert refresh.canonical(refresh.clear_already_applied_initial_refresh(env)) == refresh.canonical(proof)
    assert refresh.validate_attestation(proof) == proof


@pytest.mark.parametrize("arm", ["bc", "dagger"])
@pytest.mark.parametrize("count", [1, 2])
def test_student_v2_requires_explicit_mode_and_cannot_be_teacher_v1(arm, count):
    evidence = student_evidence(count, arm)
    assert evidence["schema"] == "roborender.initial-robot-refresh-attestation.v2"
    assert evidence["control"]["schema"] == "roborender.initial-robot-refresh-control.v2"
    assert evidence["control"]["production_activation"] is True
    assert evidence["control"]["mode"] == MODE
    assert refresh.validate_attestation(evidence, num_envs=count, mode=MODE) == evidence
    with pytest.raises(ValueError, match="attestation differs"):
        refresh.validate_attestation(evidence)
    teacher = SimpleNamespace(num_envs=count, need_to_refresh_envs=torch.ones(count, dtype=torch.bool))
    with pytest.raises(ValueError, match="attestation differs"):
        refresh.validate_attestation(refresh.clear_already_applied_initial_refresh(teacher), mode=MODE)


@pytest.mark.parametrize("arm,teacher_mode,count,resume", [
    ("teacher", "distill", 2, False), ("teacher", "raven_ppo", 2, False),
    ("bc", "raven_ppo", 2, False), ("dagger", "raven_ppo", 2, False),
    ("bc", "distill", 0, False), ("bc", "distill", 3, False),
    ("bc", "distill", True, False), ("bc", "distill", 2, True),
])
def test_student_admission_rejects_teacher_wrong_batch_and_resume(arm, teacher_mode, count, resume):
    with pytest.raises(ValueError, match="fresh non-resume BC/DAgger student B1/B2"):
        refresh.validate_mode(MODE, arm=arm, teacher_action_mode=teacher_mode, num_envs=count, has_resume=resume)


@pytest.mark.parametrize("arm,mode", [("teacher", MODE), ("bc", refresh.CLEAR_ONCE), ("bc", refresh.ORIGINAL)])
def test_invalid_low_level_clear_leaves_flags_and_counter_unchanged(arm, mode):
    env = SimpleNamespace(num_envs=2, need_to_refresh_envs=torch.ones(2, dtype=torch.bool))
    with pytest.raises(ValueError):
        refresh.clear_already_applied_initial_refresh(env, mode=mode, arm=arm, teacher_action_mode="distill")
    assert env.need_to_refresh_envs.all()
    assert not hasattr(env, "_diagnostic_initial_refresh_clear_calls")


@pytest.mark.parametrize("arm", ["bc", "dagger"])
def test_corrected_reset_keeps_first_student_action_targets_and_rng(monkeypatch, arm):
    obs = {"teacher_obs": torch.arange(452, dtype=torch.float32).reshape(2, 226),
           "homie_obs": torch.zeros(2, 516)}
    model = torch.nn.Linear(3, 11).eval()
    weights = deepcopy(model.state_dict())
    calls = []
    def step(policy, observations, **kwargs):
        calls.append("student")
        student = torch.cat([policy(observations["teacher_obs"][:, :3]), torch.zeros(2, 15)], dim=-1)
        teacher = student.clone()
        teacher[:, :11] += 100
        return {"gt_actions": teacher, "action_mean": student}
    trainer = SimpleNamespace(policy_model=model, policy_step=step,
        ref_model=SimpleNamespace(rollout=lambda **_: pytest.fail("teacher takeover")))
    monkeypatch.setenv("DIAGNOSTIC_ARM", arm)
    monkeypatch.setenv("DIAGNOSTIC_TEACHER_ACTION_MODE", "distill")
    events = []
    original = panel_environment([], obs)
    corrected = panel_environment(events, obs)
    targets = {key: getattr(corrected, key).clone() for key in
               ("target_robot_root_states", "target_robot_dof_state", "_homie_commands", "_last_homie_commands")}
    rng = torch.get_rng_state().clone()
    # Historical observation fixture, not a callable pre-fix evaluator.
    before = deepcopy(obs)
    monkeypatch.setenv("DIAGNOSTIC_INITIAL_ROBOT_REFRESH", MODE)
    after = corrected.reset_panel()
    with torch.no_grad():
        left, _ = diagnostic_step(trainer, before, None, arm=arm, mode="distill")
        right, _ = diagnostic_step(trainer, after, None, arm=arm, mode="distill")
    assert events[-2:] == ["observations_constructed", "clear_pending_once"]
    assert calls == ["student", "student"]
    assert all(torch.equal(before[key], after[key]) for key in obs)
    assert torch.equal(left["actions"], right["actions"])
    assert torch.equal(right["actions"], right["action_mean"])
    assert not torch.equal(right["actions"][:, :11], right["gt_actions"][:, :11])
    assert all(torch.equal(getattr(corrected, key), value) for key, value in targets.items())
    assert all(torch.equal(model.state_dict()[key], value) for key, value in weights.items())
    assert torch.equal(torch.get_rng_state(), rng)
    assert original.need_to_refresh_envs.all() and not corrected.need_to_refresh_envs.any()
    with pytest.raises(RuntimeError, match="already applied"):
        refresh.clear_already_applied_initial_refresh(corrected, mode=MODE, arm=arm, teacher_action_mode="distill")
    with pytest.raises(ValueError, match="privileged teacher arm"):
        diagnostic_step(trainer, after, None, arm=arm, mode="raven_ppo")


@pytest.mark.parametrize("arm", ["bc", "dagger"])
def test_corrected_b2_b1_panel_receipts_partitions_and_failures_remain_bound(arm):
    p = mixed_panel()
    receipts = []
    result = panels.run_panel(p, execute_cohort=lambda spec, _: student_result(spec, arm),
        expected_checkpoint_sha256="b" * 64, expected_arm=arm,
        diagnostic_initial_robot_refresh=MODE, save_receipt=receipts.append)
    assert result[refresh.FIELD] == MODE
    assert result["successes"] == 2 and result["completed_episodes"] == 3
    assert len(result["outcomes"]) == 3 and result["success_rate"] == 2 / 3
    assert result["action_mode"] == p["action_mode"]
    assert [a["num_envs"] for a in result["initial_robot_refresh_attestations"]] == [2, 1]
    assert all(panels.validate_cohort_receipt(p, row, **expected(arm)) == row for row in receipts)
    assert panels.completed_progress(p, receipts, **expected(arm))[refresh.FIELD] == MODE
    assert panels.validate_result(p, result, "b" * 64, allowed_arms=(arm,),
        diagnostic_initial_robot_refresh=MODE) == result
    with pytest.raises(ValueError, match="differs from requested"):
        panels.validate_result(p, result, "b" * 64, allowed_arms=(arm,))
    partitions = [panels.run_partition(p, panels.make_partition(p, [index]),
        execute_cohort=lambda spec, _: student_result(spec, arm), **expected(arm)) for index in (0, 1)]
    assert panels.aggregate_partitions(p, partitions, **expected(arm)) == result
    with pytest.raises(ValueError, match="differs from requested"):
        panels.aggregate_partitions(p, partitions, checkpoint_sha256="b" * 64, arm=arm)
    with pytest.raises(ValueError, match="non-resume"):
        panels.run_panel(p, execute_cohort=lambda *_: pytest.fail("must not resume"),
            expected_checkpoint_sha256="b" * 64, expected_arm=arm,
            diagnostic_initial_robot_refresh=MODE, completed_receipts=receipts[:1])
    with pytest.raises(ValueError, match="non-resume"):
        panels.run_partition(p, panels.make_partition(p, [0]),
            execute_cohort=lambda *_: pytest.fail("must not resume"), completed_receipts=receipts[:1], **expected(arm))


@pytest.mark.parametrize("kind", ["original", "teacher", "v1_attestation", "v1_control", "missing_cohort", "wrong_tail_count"])
def test_corrected_student_cannot_accept_original_teacher_or_forged_receipts(kind):
    p = panel()
    values = [student_result(spec) for spec in p["cohorts"]]
    if kind == "original":
        for value in values:
            for key in (refresh.FIELD, "initial_robot_refresh_control", "initial_robot_refresh_attestation"):
                value.pop(key)
    elif kind == "teacher":
        values = [teacher_result(spec) for spec in p["cohorts"]]
    elif kind == "v1_attestation":
        values[0]["initial_robot_refresh_attestation"] = teacher_result(p["cohorts"][0])["initial_robot_refresh_attestation"]
    elif kind == "v1_control":
        values[0]["initial_robot_refresh_control"] = refresh.control_contract()
    with pytest.raises(ValueError):
        result = panels.summarize(p, values)
        if kind == "missing_cohort":
            result["initial_robot_refresh_attestations"].pop()
        if kind == "wrong_tail_count":
            result["initial_robot_refresh_attestations"][-1] = student_evidence(1)
        panels.validate_result(p, result, "b" * 64, allowed_arms=("bc",), diagnostic_initial_robot_refresh=MODE)
