"""CPU-only guards against Raven's eval-time teacher action override."""
from __future__ import annotations
from test.deployment.source_paths import SUCCESS_RATE_ROOT
from test.deployment.source_paths import REPO_ROOT

import ast
from copy import deepcopy
import hashlib
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch

from pipeline.deployment.rtx_actor.success_diagnostic_core import require_student_only_evaluation
from pipeline.deployment.rtx_actor.teacher_action_diagnostic import diagnostic_step


ROOT = REPO_ROOT
SAFE = dict(enforce_teacher_rollout=False, ratio_teacher_rollout=0.0,
            rollout_with_teacher_num_steps=0)


@pytest.mark.parametrize("arm", ["bc", "dagger"])
def test_supported_student_settings_are_accepted_without_mutation(arm):
    config = deepcopy(SAFE)
    require_student_only_evaluation(config, arm=arm)
    assert config == SAFE


def test_omegaconf_native_config_is_supported():
    from omegaconf import OmegaConf
    require_student_only_evaluation(OmegaConf.create(SAFE), arm="bc")


@pytest.mark.parametrize("arm", ["bc", "dagger"])
@pytest.mark.parametrize("changes", [
    {"enforce_teacher_rollout": True}, {"enforce_teacher_rollout": 0},
    {"ratio_teacher_rollout": 0.5}, {"ratio_teacher_rollout": False},
    {"ratio_teacher_rollout": float("nan")}, {"ratio_teacher_rollout": "0"},
    {"rollout_with_teacher_num_steps": 1}, {"rollout_with_teacher_num_steps": False},
])
def test_student_takeover_or_ambiguous_settings_fail_closed(arm, changes):
    with pytest.raises(ValueError, match="student-only evaluation"):
        require_student_only_evaluation({**SAFE, **changes}, arm=arm)


@pytest.mark.parametrize("missing", list(SAFE))
def test_student_settings_must_be_explicit(missing):
    config = deepcopy(SAFE)
    del config[missing]
    with pytest.raises(ValueError, match="student-only evaluation"):
        require_student_only_evaluation(config, arm="bc")


def test_teacher_diagnostic_keeps_its_existing_control_convention():
    require_student_only_evaluation({}, arm="teacher")
    require_student_only_evaluation(dict(enforce_teacher_rollout=True,
        ratio_teacher_rollout=1.0, rollout_with_teacher_num_steps=100), arm="teacher")
    with pytest.raises(ValueError, match="unsupported"):
        require_student_only_evaluation(SAFE, arm="unknown")


def original_policy_step():
    path = ROOT / "runtime/raven_gr00t_collection_94d7985/gr00t/rl/trl/trainer/distill_trainer_obj_pred_homie_api.py"
    if not path.is_file():
        pytest.skip("pinned original Raven source unavailable")
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "3da5a2557ae18a8ba53c4da14d8841b80a609b851b326a21534ee2c5001b7584"
    cls = next(node for node in ast.parse(raw).body if isinstance(node, ast.ClassDef)
               and node.name == "TRLDistillTrainerObjPredHomieAPI")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "policy_step")
    scope = dict(torch=torch, deepcopy=deepcopy, init_actor_critic_dict={"num_one_step_obs": 86})
    exec(compile(ast.fix_missing_locations(ast.Module(body=[deepcopy(method)], type_ignores=[])),
                 "<hash-pinned-original-policy-step>", "exec"), scope)
    return scope["policy_step"]


class Homie:
    def __call__(self, obs):
        mean = torch.full((4, 15), 3.0)
        return dict(actions=mean, action_mean=mean.clone(), action_sigma=torch.ones_like(mean),
                    entropy=torch.zeros(4))

    def get_actions_log_prob(self, actions):
        return torch.zeros(4)


class Student:
    def rollout_with_obj_pred(self, **kwargs):
        return dict(action_mean=torch.full((4, 11), 2.0), action_sigma=torch.ones(4, 11))

    def get_actions_log_prob(self, actions):
        return torch.zeros(4)


@pytest.mark.parametrize("enforced", [False, True])
def test_original_teacher_override_runs_in_eval_and_student_mean_cannot_undo_it(enforced):
    config = {**SAFE, "enforce_teacher_rollout": enforced,
              "ratio_teacher_rollout": 0.5 if enforced else 0.0}
    trainer = SimpleNamespace(config=config, export_homie_model=False, policy_model=Student(),
        unwrapped_model=SimpleNamespace(training=False, homie_walk_model=Homie(),
            homie_stand_model=Homie(), homie_switch_threshold=0.5, opt_homie=False),
        ref_model=SimpleNamespace(act_inference=lambda **kwargs: torch.full((4, 11), 9.0)),
        env=SimpleNamespace(config={"zero_vel": True, "zero_finger": False},
            stage_buf=torch.ones(4), _delta_actions=torch.ones(4, 11)))
    trainer.policy_step = MethodType(original_policy_step(), trainer)
    obs = dict(homie_obs=torch.zeros(4, 516), teacher_obs=torch.zeros(4, 226),
               actor_obs=torch.zeros(4, 113), gt_obj_pos=torch.zeros(4, 3))
    original = trainer.policy_step(trainer.policy_model, obs, cur_dones=torch.zeros(4))
    actions, _ = diagnostic_step(trainer, obs, torch.zeros(4), arm="bc")
    assert torch.equal(actions["actions"], original["action_mean"])
    assert actions["actions"][:, 11:].eq(3).all()
    assert actions["gt_actions"][:, :3].eq(-10).all()
    if enforced:
        assert actions["actions"][:2, :3].eq(-10).all()
        assert actions["actions"][2:, :11].eq(2).all()
        with pytest.raises(ValueError, match="student-only evaluation"):
            require_student_only_evaluation(config, arm="bc")
    else:
        assert actions["actions"][:, :11].eq(2).all()
        require_student_only_evaluation(config, arm="bc")


def test_native_entrypoint_rejects_before_model_reset_or_step_work():
    path = SUCCESS_RATE_ROOT / "evaluation_loop.py"
    cls = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef)
               and node.name == "SuccessDiagnosticTrainer")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "evaluate")

    class FakePath:
        def __init__(self, *_):
            pass

        def resolve(self):
            return self

        def is_relative_to(self, _):
            return True

        def mkdir(self, **kwargs):
            pass

    def forbidden(*args, **kwargs):
        pytest.fail("student control guard must precede subsequent evaluation work")

    scope = dict(Path=FakePath, os=SimpleNamespace(environ={"DIAGNOSTIC_ARM": "bc"}),
                 require_student_only_evaluation=require_student_only_evaluation,
                 validate_mode=forbidden, DISTILL="distill")
    exec(compile(ast.fix_missing_locations(ast.Module(body=[deepcopy(method)], type_ignores=[])),
                 "<current-native-evaluation-entrypoint>", "exec"), scope)
    with pytest.raises(ValueError, match="student-only evaluation"):
        scope["evaluate"](SimpleNamespace(config={**SAFE, "enforce_teacher_rollout": True}))
