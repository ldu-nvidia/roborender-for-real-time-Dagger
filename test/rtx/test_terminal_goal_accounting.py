"""CPU regressions executing the pinned native step/reset ordering unchanged."""
from test.deployment.source_paths import SUCCESS_RATE_ROOT
from test.deployment.source_paths import PIPELINE_ROOT, REPO_ROOT
import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dagger_rtx.terminal_goal_accounting import (
    GOAL_ACCOUNTING_VERSION,
    TerminalGoalAccountingMixin,
)


ENVS = PIPELINE_ROOT / "GR00T-VisualSim2Real/gr00t/rl/envs"


def native_method(relative, class_name, name):
    tree = ast.parse((ENVS / relative).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name))
    method.decorator_list = []
    return method


class NoPhysicsBase:
    def _check_termination(self):
        self.check_calls += 1
        self.reset_buf.copy_(self.requested_done)
        self.time_out_buf.copy_(self.requested_timeout)

    def _reset_tasks_callback(self, env_ids):
        pass


def make_environment(*, corrected=True, count=4):
    methods = [native_method("base_task/staged_task_base.py", "StagedTaskBase", name)
               for name in ("_check_termination", "_reset_tasks_callback")]
    methods.append(native_method("loco_manip/walk_stand_place_grasp_turn_homie.py",
                                 "WalkStandPlaceGraspTurnHomie", "_post_physics_step"))
    scope = {"torch": torch, "NoPhysicsBase": NoPhysicsBase}
    module = ast.Module(body=[ast.ClassDef(name="Native", bases=[ast.Name(id="NoPhysicsBase", ctx=ast.Load())],
                        keywords=[], body=methods, decorator_list=[])], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<pinned-native-callbacks>", "exec"), scope)
    native = scope["Native"]
    cls = type("Corrected", (TerminalGoalAccountingMixin, native), {}) if corrected else native
    env = cls()
    env.num_envs, env.num_stages = count, 5
    env.config = SimpleConfig(reset_on_complete=False, reset_on_overtime=True,
                             simulator=SimpleNamespace(config=SimpleNamespace(
                                 cameras=SimpleNamespace(enable_cameras=True))))
    for name in ("reset_buf", "time_out_buf", "requested_done", "requested_timeout", "goal_now",
                 "current_completed_task_buf", "current_last_stage_completed_task_buf",
                 "last_completed_task_buf", "last_last_stage_completed_task_buf",
                 "just_completed_task_buf", "grasp_goal_reached_buf", "need_to_refresh_envs"):
        setattr(env, name, torch.zeros(count, dtype=torch.bool))
    for name in ("time_in_stage_buf", "time_since_completion_buf", "current_max_stage_buf", "last_max_stage_buf"):
        setattr(env, name, torch.zeros(count, dtype=torch.int64))
    env.episode_length_buf = torch.ones(count, dtype=torch.int64)
    env.stage_buf = torch.full((count,), 4)
    env.max_stage_time = torch.full((5,), 9999)
    env._make_mask = lambda index: env.stage_buf == index
    env.stage_complete_cond_func = lambda: env.goal_now.clone()
    env._update_termination_curriculum = lambda: None
    env.check_calls = env.reset_calls = env.observation_calls = 0
    env.log_dict, env.extras = {}, {}
    env._refresh_sim_tensors = lambda: None
    env._update_counters_each_step = lambda: None
    env._pre_compute_observations_callback = lambda **kw: None
    env._update_tasks_callback = lambda: None
    env._compute_reward = lambda: None
    env._compute_observations = lambda: None
    env.use_past_obs = env.viewer = False
    env.common_step_counter, env.history_save_interval = 1, 1
    env.history_handler = SimpleNamespace(history={})

    callback = native_method("loco_manip/walk_stand_place_grasp_turn_homie.py",
                             "WalkStandPlaceGraspTurnHomie", "_post_compute_observations_callback")
    exports = [n for n in callback.body if ast.unparse(n).startswith(
        ("self.grasp_goal_reached_buf |=", "self.extras['grasp_goal_reached'] ="))]
    assert len(exports) == 2
    code = compile(ast.fix_missing_locations(ast.Module(body=exports, type_ignores=[])), "<native-export>", "exec")

    def post_observations():
        env.observation_calls += 1
        exec(code, {"self": env})

    def reset(ids):
        env.reset_calls += 1
        env._reset_tasks_callback(ids)
        env.grasp_goal_reached_buf[ids] = False
        env.episode_length_buf[ids] = 0

    env._post_compute_observations_callback = post_observations
    env.reset_envs_idx = reset
    return env


class SimpleConfig(dict):
    __getattr__ = dict.__getitem__


def test_unpatched_native_loses_same_step_terminal_success():
    env = make_environment(corrected=False)
    env.goal_now[0] = env.requested_done[0] = True
    env._post_physics_step()
    assert env.last_last_stage_completed_task_buf[0]
    assert not env.extras["grasp_goal_reached"][0]


@pytest.mark.parametrize("timeout", [False, True])
def test_same_step_success_survives_done_or_timeout_without_next_episode_leak(timeout):
    env = make_environment()
    # Lanes: goal+done, goal+ongoing, failure+done, ongoing no goal.
    env.goal_now[:] = torch.tensor([True, True, False, False])
    env.requested_done[:] = torch.tensor([True, False, True, False])
    env.requested_timeout[0] = timeout
    env._post_physics_step()
    exported = env.extras["grasp_goal_reached"]
    assert exported.tolist() == [True, True, False, False]
    assert env.current_last_stage_completed_task_buf.tolist() == [False, True, False, False]
    assert env.grasp_goal_reached_buf.tolist() == [False, True, False, False]
    assert env.reset_buf.tolist() == [True, False, True, False]
    assert env.check_calls == env.reset_calls == env.observation_calls == 1
    assert env.goal_accounting_version == GOAL_ACCOUNTING_VERSION
    # New episodes must not inherit the prior terminal goal; exported tensors
    # already returned to a caller must not change on the next reset/step.
    env.goal_now[:] = env.requested_done[:] = env.requested_timeout[:] = False
    env._post_physics_step()
    assert env.extras["grasp_goal_reached"].tolist() == [False, True, False, False]
    assert exported.tolist() == [True, True, False, False]


def test_prior_success_retained_through_later_failure_then_cleared():
    env = make_environment(count=1)
    env.goal_now[:] = True
    env._post_physics_step()
    env.goal_now[:] = False
    env.requested_done[:] = True
    env._post_physics_step()
    assert env.extras["grasp_goal_reached"].item()
    env.requested_done[:] = False
    env._post_physics_step()
    assert not env.extras["grasp_goal_reached"].item()


def test_manual_reset_does_not_reuse_snapshot():
    env = make_environment(count=1)
    env.goal_now[:] = True
    env._post_physics_step()
    env.reset_envs_idx(torch.tensor([0]))
    env._post_compute_observations_callback()
    env.goal_now[:] = False
    assert not env.extras["grasp_goal_reached"].item()
    env._post_physics_step()
    assert not env.extras["grasp_goal_reached"].item()


@pytest.mark.parametrize("stage,length_before_step,expected", [(3, 5, False), (4, 0, False), (4, 1, True)])
def test_native_last_stage_and_minimum_episode_length_are_unchanged(stage, length_before_step, expected):
    env = make_environment(count=1)
    env.stage_buf[:] = stage
    env.episode_length_buf[:] = length_before_step
    env.goal_now[:] = env.requested_done[:] = True
    env._post_physics_step()
    assert env.extras["grasp_goal_reached"].item() is expected


def test_missing_capture_fails_instead_of_reusing_stale_goal():
    env = make_environment(count=1)
    env.goal_now[:] = True
    env._post_physics_step()
    env._check_termination = lambda: None
    with pytest.raises(RuntimeError, match="pre-reset goal capture"):
        env._post_physics_step()


def test_production_and_diagnostic_use_same_accounting_mixin():
    repo = REPO_ROOT
    for path, class_name in (
        (repo / "pipeline/dagger/rtx/code/src/dagger_rtx/fixed_panel_raven.py", "FixedPanelWalkStandPlaceGraspTurnHomie"),
        (SUCCESS_RATE_ROOT / "evaluation_loop.py", "SuccessAccountingEnvironment"),
    ):
        cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == class_name)
        assert ast.unparse(cls.bases[0]) == "TerminalGoalAccountingMixin"
