from test.deployment.source_paths import SUCCESS_RATE_ROOT
from test.deployment.source_paths import ACTOR_ROOT
import ast
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = ACTOR_ROOT
sys.path.insert(0, str(ROOT))
from explicit_motion_panel import SCHEMA, canonical, run_panel, summarize, validate_panel, validate_result
from teacher_action_diagnostic import DISTILL, RAVEN_PPO, validate_mode, validate_trace_steps
from initial_robot_refresh_diagnostic import ORIGINAL as ORIGINAL_INITIAL_REFRESH, STUDENT_CLEAR_ONCE, validate_mode as validate_initial_refresh


def panel():
    value = {"schema": SCHEMA, "panel_id": "same-support-test", "purpose": "test",
        "motion_inventory_sha256": "a" * 64, "observation_mode": "raytraced_3view",
        "action_mode": "deterministic_action_mean_11_plus_homie_15", "metric": "native_goal_or_until_done",
        "max_vector_steps_per_cohort": 450, "episode_count": 4, "cohorts": []}
    for i in range(2):
        assignment = {"schema": "roborender.native-motion-cohort.v1", "run_id": value["panel_id"],
            "cohort_id": f"cohort-{i}", "purpose": "test", "simulator_seed": 100 + i,
            "motion_inventory_sha256": "a" * 64,
            "lanes": [{"lane": j, "motion_id": f"motion-{j}", "path": f"/motions/{j}.npz",
                       "sha256": str(j + 1) * 64} for j in range(2)]}
        value["cohorts"].append({"assignment": assignment, "case_ids": [f"case-{i}-{j}" for j in range(2)]})
    return value


def result(spec, successes=(True, False)):
    return {"complete": True, "completed": 2, "expected": 2, "optimizer_steps": 0,
        "policy_unchanged": True, "accounting_disagreements": 0,
        "seed": spec["assignment"]["simulator_seed"], "observation_mode": "raytraced_3view",
        "motion_assignment_sha256": hashlib.sha256(canonical(spec["assignment"])).hexdigest(),
        "checkpoint_sha256": "b" * 64, "runtime_physics_sha256": "c" * 64,
        "arm": "bc", "goal_accounting_version": "roborender.raven.pre_reset_goal.v1",
        "original_raven_metric_completed": 2, "original_raven_metric_successes": sum(successes),
        "native_successes": sum(successes), "exported_successes": sum(successes),
        "outcomes": [{"lane": i, "steps": 123, "native_success": success,
                      "exported_success": success, "timeout": False} for i, success in enumerate(successes)]}


def test_cpu_import_has_no_torch_or_native_runtime():
    subprocess.run([sys.executable, "-c", f"import sys;sys.path.insert(0,{str(ROOT)!r});import explicit_motion_panel;assert 'torch' not in sys.modules;assert not any(x.startswith('omni') for x in sys.modules)"], check=True)


def test_exact_same_support_panel_and_success_fraction():
    p = panel()
    assert validate_panel(p) == p
    rows = [result(spec) for spec in p["cohorts"]]
    summary = summarize(p, rows)
    assert summary["successes"] == 2 and summary["completed_episodes"] == 4
    assert summary["success_rate"] == 0.5
    assert not summary["legacy_five_motion_benchmark"]
    assert [row["case_id"] for row in summary["outcomes"]] == ["case-0-0", "case-0-1", "case-1-0", "case-1-1"]
    assert validate_result(p, summary, "b" * 64) == summary


def test_teacher_diagnostic_retains_true_behavior_and_is_not_student_validation():
    p = panel()
    rows = [result(spec) for spec in p["cohorts"]]
    for row in rows:
        row["arm"] = "teacher"
    summary = summarize(p, rows)
    assert summary["arm"] == "teacher"
    assert summary["action_mode"] == "privileged_teacher_plus_homie"
    with pytest.raises(ValueError):
        validate_result(p, summary, "b" * 64)
    assert validate_result(p, summary, "b" * 64, allowed_arms=("teacher",)) == summary
    summary["action_mode"] = p["action_mode"]
    with pytest.raises(ValueError):
        validate_result(p, summary, "b" * 64, allowed_arms=("teacher",))


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(episode_count=256),
    lambda p: p.update(purpose="training"),
    lambda p: p.update(max_vector_steps_per_cohort=451),
    lambda p: p["cohorts"][1]["assignment"].update(simulator_seed=100),
    lambda p: p["cohorts"][1]["case_ids"].__setitem__(0, "case-0-0"),
    lambda p: p["cohorts"][0]["assignment"]["lanes"][0].update(lane=1),
    lambda p: p["cohorts"][0]["assignment"].update(motion_inventory_sha256="d" * 64),
])
def test_wrong_or_ambiguous_panel_rejected(mutation):
    p = panel()
    mutation(p)
    with pytest.raises(ValueError):
        validate_panel(p)


@pytest.mark.parametrize("field,value", [("complete", False), ("completed", 1), ("optimizer_steps", 1),
    ("policy_unchanged", False), ("accounting_disagreements", 1), ("motion_assignment_sha256", "d" * 64),
    ("seed", 99), ("observation_mode", "rendered")])
def test_incomplete_or_changed_native_receipt_is_not_success_rate(field, value):
    p = panel()
    rows = [result(spec) for spec in p["cohorts"]]
    rows[1][field] = value
    with pytest.raises(ValueError):
        summarize(p, rows)


def test_missing_duplicate_lane_and_checkpoint_switch_rejected():
    p = panel()
    rows = [result(spec) for spec in p["cohorts"]]
    with pytest.raises(ValueError, match="missing"):
        summarize(p, rows[:1])
    bad = deepcopy(rows)
    bad[1]["outcomes"][1]["lane"] = 0
    with pytest.raises(ValueError):
        summarize(p, bad)
    bad = deepcopy(rows)
    bad[1]["checkpoint_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="checkpoint"):
        summarize(p, bad)


def test_cohort_order_and_failure_stop_no_partial_metric():
    p = panel()
    seen, reports = [], []
    def execute(spec, i):
        seen.append(i)
        return result(spec)
    assert run_panel(p, execute_cohort=execute, report=reports.append)["success_rate"] == .5
    assert seen == [0, 1]
    assert reports[-1]["completed_episodes"] == 4
    def failed(spec, i):
        if i == 1:
            raise RuntimeError("native lane incomplete")
        return result(spec)
    with pytest.raises(RuntimeError, match="incomplete"):
        run_panel(p, execute_cohort=failed)


def test_legacy_default_remains_five_and_native_optimizer_guard_is_restored():
    source = (SUCCESS_RATE_ROOT / "evaluation_loop.py").read_text()
    module = ast.parse(source)
    cls = next(x for x in module.body if isinstance(x, ast.ClassDef) and x.name == "SuccessDiagnosticTrainer")
    expected = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "_diagnostic_expected_envs")
    assert next(x for x in expected.body if isinstance(x, ast.Return)).value.value == 5
    assert "self.optimizer.step = original_optimizer_step" in source
    assert "FirstEpisodeAccounting(count)" in source
    assert "actions, raw_teacher = select_policy_actions(" in source
    helper = (SUCCESS_RATE_ROOT / "policy_actions.py").read_text()
    assert "student = actions[\"action_mean\"]" in helper


def test_native_receipt_is_preserved_when_explicit_cohort_result_is_published(tmp_path):
    """Exercise the actual wrapper method without importing Torch or Isaac.

    Native train() writes result.json before returning; a second exclusive
    publication to that same path used to fail only after real evaluation.
    """
    from native_artifact_io import publish_json
    from native_motion_assignment import validate_motion_assignment
    from initial_robot_refresh_diagnostic import require_corrected_execution,SMOKE_STUDENT_CLEAR_ONCE

    source = ROOT / "isaac6_motion_evaluator.py"
    module = ast.parse(source.read_text())
    cls = next(x for x in module.body if isinstance(x, ast.ClassDef)
               and x.name == "ExplicitMotionSuccessEvaluator")
    output = tmp_path / "cohort-000"
    raw = {"complete": True, "optimizer_steps": 0, "native_successes": 1}

    class NativeDiagnostic:
        def train(self):
            publish_json(output / "result.json", raw)
            return dict(raw)

    scope = {
        "SuccessDiagnosticTrainer": NativeDiagnostic,
        "validate_motion_assignment": validate_motion_assignment,
        "Path": Path, "deepcopy": deepcopy, "hashlib": hashlib,
        "canonical": canonical, "publish_json": publish_json,
        "OmegaConf": SimpleNamespace(set_struct=lambda *args: None),
        "torch": SimpleNamespace(ones=lambda count, **kwargs: [True] * count, bool=bool),
        "_environment": lambda values: nullcontext(),
        "DISTILL": DISTILL, "RAVEN_PPO": RAVEN_PPO,
        "validate_mode": validate_mode, "validate_trace_steps": validate_trace_steps,
        "ORIGINAL_INITIAL_REFRESH": ORIGINAL_INITIAL_REFRESH,
        "STUDENT_CLEAR_ONCE": STUDENT_CLEAR_ONCE,
        "SMOKE_STUDENT_CLEAR_ONCE": SMOKE_STUDENT_CLEAR_ONCE,
        "validate_initial_refresh": validate_initial_refresh,
        "require_corrected_execution": require_corrected_execution,
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), scope)
    evaluator = scope["ExplicitMotionSuccessEvaluator"]()
    evaluator.config = SimpleNamespace()
    evaluator.accelerator = SimpleNamespace(device="cpu")
    reset = lambda *args, **kwargs: None
    evaluator.env = SimpleNamespace(_reset_task_states=reset,
        max_episode_length_s=8.,dt=.02,sim_dt=.005,max_episode_length=400.)
    assignment = panel()["cohorts"][0]["assignment"]
    # This publication regression does not run a student or image callback;
    # exercise the permitted privileged-teacher path with corrected reset.
    actual = evaluator.evaluate_cohort(assignment, checkpoint_path="/owned/student.pt",
                                      checkpoint_sha256="b" * 64, output=output,
                                      arm="teacher", teacher_action_mode=RAVEN_PPO,
                                      diagnostic_initial_robot_refresh="clear_already_applied_once")
    assert json.loads((output / "result.json").read_bytes()) == raw
    assert json.loads((output / "explicit-result.json").read_bytes()) == actual
    assert actual["motion_assignment_sha256"] == hashlib.sha256(canonical(assignment)).hexdigest()
    assert evaluator.env._reset_task_states is reset
    with pytest.raises(FileExistsError):
        publish_json(output / "explicit-result.json", actual)


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(success_rate=.9),
    lambda r: r.update(checkpoint_sha256="d" * 64),
    lambda r: r.update(arm="teacher"),
    lambda r: r.update(augmentation_applied=True),
    lambda r: r.update(optimizer_steps=1),
    lambda r: r.update(goal_accounting_version="old"),
    lambda r: r["outcomes"][0].update(simulator_seed=999),
    lambda r: r["outcomes"][0].update(motion_sha256="e" * 64),
    lambda r: r["outcomes"][0].update(steps=451),
    lambda r: r["outcomes"][1].update(case_id="case-0-0"),
    lambda r: r["outcomes"][0].update(native_success=False),
])
def test_host_import_rejects_changed_identity_behavior_or_accounting(mutation):
    p = panel()
    value = summarize(p, [result(spec) for spec in p["cohorts"]])
    mutation(value)
    with pytest.raises(ValueError):
        validate_result(p, value, "b" * 64)
