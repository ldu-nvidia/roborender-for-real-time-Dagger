"""Matched-input evaluation control comparison, not native qualification.

Execute the COMPLETE hash-pinned Raven eval method and policy_step, and the
unchanged current diagnostic for-loop with its real diagnostic_step/render
guards. Physics, neural networks, generated pixels and framework hooks are CPU
doubles. First-episode and pooled protocols are separately matched; their scores
are never compared as if they described the same experiment. No Isaac import.
"""
from __future__ import annotations
from test.deployment.source_paths import SUCCESS_RATE_ROOT

import ast
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import io
import json
import logging
import os
import sys
import time
from types import ModuleType, SimpleNamespace
from typing import Optional

import numpy as np
import pytest
import torch

from dagger_rtx.terminal_goal_accounting import TerminalGoalAccountingMixin
from pipeline.deployment.rtx_actor.raven_pooled_benchmark import (
    PooledEpisodeAccounting, randomize_initial_clocks,
)
from pipeline.deployment.rtx_actor.success_diagnostic_core import (
    FirstEpisodeAccounting, require_student_only_evaluation,
)
from pipeline.deployment.rtx_actor.teacher_action_diagnostic import diagnostic_step
from test.deployment.source_paths import ACTOR_ROOT, REPO_ROOT


REFERENCE = REPO_ROOT / "runtime/raven_gr00t_collection_94d7985/gr00t/rl"
PINS = {
    "trl/trainer/distill_trainer_obj_pred_homie_api.py":
        "3da5a2557ae18a8ba53c4da14d8841b80a609b851b326a21534ee2c5001b7584",
    "envs/loco_manip/walk_stand_grasp.py":
        "d985d3e74414b6ceabefd917bdaa3b36c67ff9e141882436f931815f16bd3be9",
}
SAFE = dict(enforce_teacher_rollout=False, ratio_teacher_rollout=0.,
            rollout_with_teacher_num_steps=0)


class Config(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


class Staged:
    """Only the isinstance marker imported by original eval; no native code."""


def reference_class(relative, class_name, names):
    path = REFERENCE / relative
    raw = path.read_bytes()  # Required source: fail, do not silently skip coverage.
    assert hashlib.sha256(raw).hexdigest() == PINS[relative]
    cls = next(n for n in ast.parse(raw).body
               if isinstance(n, ast.ClassDef) and n.name == class_name)
    methods = [deepcopy(n) for n in cls.body
               if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in methods} == set(names)
    module = ast.Module(body=[ast.ClassDef(name="Reference", bases=[], keywords=[],
        body=methods, decorator_list=[])], type_ignores=[])
    scope = dict(torch=torch, np=np, os=os, time=time, Optional=Optional,
        logger=logging.getLogger(__name__), deepcopy=deepcopy,
        init_actor_critic_dict={"num_one_step_obs": 86},
        unwrap_model_for_generation=lambda model, *_a, **_k: nullcontext(model))
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope["Reference"]


def metric_class():
    return reference_class("envs/loco_manip/walk_stand_grasp.py", "WalkStandGrasp", (
        "init_eval_metrics_tracking", "update_eval_metrics_per_step",
        "process_eval_episode_completions", "reset_eval_episode_tracking",
        "get_eval_metrics_summary"))


def cpu(value):
    return value.detach().cpu().numpy().copy() if torch.is_tensor(value) else np.asarray(value)


def scripted_env(count, *, current, terminal_loss=False):
    """Identical scripted transitions; no numerical physics equivalence claim."""
    class Env(metric_class(), Staged):
        def __init__(self):
            self.num_envs, self.index, self.reset_calls = count, 0, 0
            self.max_episode_length, self.max_stage_time = 400, [157]
            self.episode_length_buf = torch.zeros(count, dtype=torch.int64)
            self.time_in_stage_buf = torch.zeros(count, dtype=torch.int64)
            self.actual_time_in_stage_buf = torch.zeros(count, dtype=torch.int64)
            self.stage_buf = torch.ones(count)
            self._delta_actions = torch.ones(count, 11)
            self.config = Config(zero_vel=True, zero_finger=False)
            self.actions, self.action_keys, self.captured, self.events = [], [], {}, []
            self.clock_at_policy, self.reset_observation = None, None
            self.grasp_goal_reached_buf = torch.zeros(count, dtype=torch.bool)
            self.current_last_stage_completed_task_buf = self.grasp_goal_reached_buf.clone()
            self.extras = {}

        def get_object_class_names(self):
            return None

        def pixels(self):
            return torch.arange(count, dtype=torch.float32)[:, None] + self.index / 8

        def obs(self):
            homie = torch.zeros(count, 516)
            homie[:, 0] = self.index
            homie[::2, -86] = 1  # Alternate actual original walk/stand selector.
            return dict(vision_obs=torch.full((count, 1), -999.) if current else self.pixels(),
                actor_obs=torch.full((count, 113), float(self.index)),
                teacher_obs=torch.full((count, 226), float(self.index)),
                homie_obs=homie, gt_obj_pos=torch.zeros(count, 3),
                clock_observation=self.time_in_stage_buf.clone())

        def reset_all(self):
            self.reset_calls += 1
            assert self.reset_calls == 1
            self.events.append(("reset", 0))
            self.reset_observation = self.obs()
            return self.reset_observation

        def render_results(self):
            self.events.append(("original_render_hook", self.index))

        def end_render_results(self):
            self.events.append(("original_render_end", self.index))

        def _check_termination(self):
            # A scripted native boundary for the REAL terminal-accounting mixin.
            self.current_last_stage_completed_task_buf = self.grasp_goal_reached_buf.clone()

        def _post_physics_step(self):
            self._check_termination()
            self.grasp_goal_reached_buf[self.done] = False
            self.current_last_stage_completed_task_buf[self.done] = False
            self.extras = {"grasp_goal_reached": self.grasp_goal_reached_buf}

        def step(self, action):
            assert not torch.is_grad_enabled()
            assert self.index < 5, "script exhausted: an extra physical step would occur"
            self.events.append(("step", self.index))
            self.actions.append(action["actions"].clone())
            self.action_keys.append(sorted(action))
            goals = [[1, 0, 0, 1], [0, 1, 1, 0], [0, 0, 0, 0],
                     [0, 0, 0, 0], [1, 1, 0, 1]][self.index]
            dones = [[0, 1, 0, 0], [0, 1, 1, 0], [1, 0, 0, 0],
                     [0, 1, 0, 1], [1, 1, 1, 1]][self.index]
            self.done = torch.tensor(dones * (count // 4), dtype=torch.bool)
            native_goal = torch.tensor(goals * (count // 4), dtype=torch.bool)
            if terminal_loss:
                self.done[:] = True
                native_goal[:] = False
                native_goal[0] = True
                self.grasp_goal_reached_buf = native_goal.clone()
                self._post_physics_step()
                exported = self.extras["grasp_goal_reached"]
            else:
                exported = native_goal.clone()
                self._transition_goal_before_reset = exported.clone()
            self.captured.update(current_last_stage_completed_task_buf=cpu(native_goal),
                reset_buf=cpu(self.done), time_out_buf=cpu(self.done & ~native_goal))
            reward = torch.arange(1, count + 1, dtype=torch.float32) + self.index / 4
            self.index += 1
            return self.obs(), reward, self.done, {"grasp_goal_reached": exported}

    cls = type("CorrectedEnv", (TerminalGoalAccountingMixin, Env), {}) if current and terminal_loss else Env
    return cls()


class Student(torch.nn.Module):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.weight = torch.nn.Parameter(torch.arange(11, dtype=torch.float32) / 11)
        self.inputs, self.dones, self.rollouts = [], [], 0

    def eval_mode(self):
        self.eval()

    def init_rollout(self):
        self.rollouts += 1

    def rollout_with_obj_pred(self, *, obs_dict, cur_dones, episode_attnmask):
        assert not self.training and not torch.is_grad_enabled() and episode_attnmask is None
        self.env.events.append(("policy", self.env.index))
        if self.env.clock_at_policy is None:
            self.env.clock_at_policy = [getattr(self.env, key).clone() for key in
                ("episode_length_buf", "time_in_stage_buf", "actual_time_in_stage_buf")]
        self.inputs.append({k: v.clone() for k, v in obs_dict.items()})
        self.dones.append(cur_dones.clone())
        mean = obs_dict["vision_obs"] + self.weight
        return dict(action_mean=mean, action_sigma=torch.ones_like(mean))

    def get_actions_log_prob(self, actions):
        return torch.zeros(actions.shape[0])


class Homie:
    def __init__(self, bias):
        self.bias = bias

    def __call__(self, obs):
        assert not torch.is_grad_enabled()
        mean = (obs[:, :1] + self.bias).expand(-1, 15).clone()
        return dict(actions=mean, action_mean=mean.clone(), action_sigma=torch.ones_like(mean),
                    entropy=torch.zeros(obs.shape[0]))

    def get_actions_log_prob(self, actions):
        return torch.zeros(actions.shape[0])


def trainer_for(env, output, *, pooled, target):
    original = reference_class("trl/trainer/distill_trainer_obj_pred_homie_api.py",
        "TRLDistillTrainerObjPredHomieAPI", ("eval", "policy_step"))
    trainer = original()
    trainer.env, trainer.policy_model = env, Student(env)
    trainer.config = Config(**SAFE, init_at_random_ep_len=pooled, eval=Config(
        save_trajectories=True, save_videos=False, eval_num_envs_episodes=not pooled,
        num_eval_episodes=target))
    require_student_only_evaluation(trainer.config, arm="dagger")
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
    trainer.args = SimpleNamespace(output_dir=str(output))
    trainer.model = trainer.unwrapped_model = SimpleNamespace(training=False,
        policy=trainer.policy_model, homie_walk_model=Homie(2),
        homie_stand_model=Homie(-2), homie_switch_threshold=.5, opt_homie=False)
    trainer.ref_model = SimpleNamespace(act_inference=lambda obs_dict, **_k:
        torch.full((env.num_envs, 11), 50.))
    trainer._eval_mode = lambda: None
    trainer.export_homie_model = False
    trainer.state, trainer.control = SimpleNamespace(), SimpleNamespace()
    trainer.callback_handler = SimpleNamespace(on_step_end=lambda *_a: None)
    trainer._native_resource_guard = lambda: env.events.append(("guard", env.index))
    return trainer


def current_loop(trainer, *, pooled, target):
    env, count = trainer.env, trainer.env.num_envs
    trainer.policy_model.eval_mode()
    trainer.policy_model.init_rollout()
    obs = env.reset_all()
    if pooled:
        randomize_initial_clocks(env)
    env.init_eval_metrics_tracking("cpu")

    class MatchedPixels:
        """Predeclared CPU pixels, NOT an Image3F renderer or live receipt."""
        def receipt(self):
            return {"evaluation_binding": {"test_only": True}}

        def __call__(self, obs, dones, step):
            env.events.append(("matched_pixels", step))
            return dict(obs, vision_obs=env.pixels())

    trainer._require_rendered_success_observations = True
    trainer._success_observation_transform = MatchedPixels()
    trainer._rendered_success_evaluation_binding = {"test_only": True}
    account = PooledEpisodeAccounting(count, target) if pooled else FirstEpisodeAccounting(count)
    path = SUCCESS_RATE_ROOT / "evaluation_loop.py"
    cls = next(n for n in ast.parse(path.read_bytes()).body
               if isinstance(n, ast.ClassDef) and n.name == "SuccessDiagnosticTrainer")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate")
    loops = [n for n in ast.walk(method) if isinstance(n, ast.For) and ast.unparse(n.target) == "step"]
    assert len(loops) == 1
    scope = dict(self=trainer, pooled={"max_vector_steps": 4500} if pooled else None,
        scaled_teacher=None, teacher_guard_checks=0, torch=torch, np=np, time=time, json=json,
        cycle_timings=None, numeric_steps=0, numeric_active=False, arm="dagger", teacher_mode="distill",
        count=count, obs=obs, dones=torch.zeros(count), select_policy_actions=diagnostic_step,
        episode_reward=torch.zeros(count), episode_length=torch.zeros(count, dtype=torch.int64),
        captured=env.captured, account=account, trace=io.StringIO(), emit=lambda *_a: None,
        started=time.monotonic(), cpu=cpu, lane_details=lambda *_a: {})
    with torch.no_grad():
        exec(compile(ast.fix_missing_locations(ast.Module(body=[deepcopy(loops[0])], type_ignores=[])),
                     str(path), "exec"), scope)
    trainer.completed_rows = [row for line in scope["trace"].getvalue().splitlines()
        for row in json.loads(line)["completed_outcomes"]]
    return account.result()


@pytest.fixture
def cpu_imports(monkeypatch):
    marker = ModuleType("gr00t.rl.envs.base_task.staged_task_base")
    marker.StagedTaskBase = Staged
    monkeypatch.setitem(sys.modules, marker.__name__, marker)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.mark.parametrize("count,pooled,target,steps,completed,successes", [
    (4, False, 4, 4, 4, 3),
    (4, True, 3, 2, 3, 2),       # Two successful unfinished lanes must not count.
    (4, True, 5, 4, 6, 4),       # Keep simultaneous final-step failure and success.
    (4, True, 7, 5, 10, 7),      # Repeated lanes; keep all three overshoot episodes.
    (16, False, 16, 4, 16, 12),
    (16, True, 32, 5, 40, 28),
    (32, False, 32, 4, 32, 24),
    (32, True, 64, 5, 80, 56),
])
def test_full_original_eval_matches_actual_current_loop_on_identical_inputs(
        tmp_path, cpu_imports, count, pooled, target, steps, completed, successes, record_property):
    left, right = (trainer_for(scripted_env(count, current=current), tmp_path / name,
        pooled=pooled, target=target) for current, name in ((False, "original"), (True, "current")))
    initial_weights = deepcopy(left.policy_model.state_dict())
    with torch.random.fork_rng():
        torch.manual_seed(998)
        before = torch.get_rng_state()
        left.eval()
        after = torch.get_rng_state()
        torch.set_rng_state(before)
        result = current_loop(right, pooled=pooled, target=target)
        assert torch.equal(torch.get_rng_state(), after)
    summary = json.loads((tmp_path / "original/metrics_eval.json").read_text())
    assert (left.env.index, right.env.index) == (steps, steps)
    assert left.env.reset_calls == right.env.reset_calls == 1
    assert left.policy_model.rollouts == right.policy_model.rollouts == 1
    assert summary["collected_episodes"] == result["completed"] == completed
    assert result["exported_successes"] == result["native_successes"] == successes
    assert summary["goal_reached_rate"] == successes / completed
    assert [row["exported_success"] for row in right.completed_rows] == left.env.eval_metrics["goal_reached_buffer"]
    assert [row["steps"] for row in right.completed_rows] == left.env.eval_metrics["length_buffer"]
    assert [row["episode_reward"] for row in right.completed_rows] == left.env.eval_metrics["reward_buffer"]
    for key, value in left.env.eval_metrics.items():
        other = right.env.eval_metrics[key]
        if torch.is_tensor(value):
            assert torch.equal(value, other), key
        else:
            assert value == other, key
    assert left.env.action_keys == right.env.action_keys
    for step, (original, current) in enumerate(zip(left.env.actions, right.env.actions)):
        assert torch.equal(original, current)
        expected_student = torch.arange(count, dtype=torch.float32)[:, None] + step / 8 + left.policy_model.weight
        assert torch.equal(current[:, :11], expected_student)
        assert current[::2, 11:].eq(step + 2).all()
        assert current[1::2, 11:].eq(step - 2).all()
    for original, current in zip(left.policy_model.inputs, right.policy_model.inputs):
        assert original.keys() == current.keys()
        for key in original:
            assert torch.equal(original[key], current[key]), key
    assert all(torch.equal(a, b) for a, b in zip(left.policy_model.dones, right.policy_model.dones))
    assert all(torch.equal(a, b) for a, b in zip(left.env.clock_at_policy, right.env.clock_at_policy))
    assert left.policy_model.inputs[0]["clock_observation"].eq(0).all()
    if pooled:
        assert right.env.clock_at_policy[0].gt(0).any()
        assert torch.equal(right.env.clock_at_policy[1], right.env.clock_at_policy[2])
        assert result["overshoot"] == completed - target
    else:
        assert all(clock.eq(0).all() for clock in right.env.clock_at_policy)
    for trainer in (left, right):
        for key, value in trainer.policy_model.state_dict().items():
            assert torch.equal(value, initial_weights[key])
        assert trainer.policy_model.weight.grad is None
        assert [(kind, step) for kind, step in trainer.env.events if kind in {"policy", "step"}] == [
            (kind, step) for step in range(steps) for kind in ("policy", "step")]
    record_property("protocol", "pooled_random_clocks" if pooled else "matched_first_episode_fresh_clocks")
    record_property("outcomes", json.dumps(dict(num_envs=count, vector_steps=steps,
        minimum=target, completed=completed, successes=successes,
        completed_lengths=left.env.eval_metrics["length_buffer"],
        completed_rewards=left.env.eval_metrics["reward_buffer"], physics_tested=False)))


def test_terminal_goal_correction_is_an_explicit_nonparity_not_a_relabel(
        tmp_path, cpu_imports, record_property):
    # Both loops see the same scripted pre-reset goal and done. The unfixed
    # CPU parent clears it during auto-reset; our REAL mixin preserves it.
    # This models the boundary, NOT execution of original Isaac reset/physics.
    original = trainer_for(scripted_env(4, current=False, terminal_loss=True),
                           tmp_path / "original", pooled=False, target=4)
    corrected = trainer_for(scripted_env(4, current=True, terminal_loss=True),
                            tmp_path / "corrected", pooled=False, target=4)
    original.eval()
    result = current_loop(corrected, pooled=False, target=4)
    assert original.env.eval_metrics["goal_reached_buffer"] == [False] * 4
    assert result["completed"] == 4 and result["exported_successes"] == 1
    assert corrected.env.grasp_goal_reached_buf.eq(False).all()
    assert corrected.env.current_last_stage_completed_task_buf.eq(False).all()
    assert original.env.index == corrected.env.index == 1
    record_property("intentional_difference", "scripted_reset_erasure:original0/4;corrected1/4")
