"""Differential checks against Raven's hash-pinned success-accounting methods.

These execute the original metric methods, not a retyped approximation. CPU
transition streams deliberately include failures, terminal successes and later
episodes on already-scored lanes. They do not certify Isaac physics, generated
images, the checkpoint, or equality of two different reset distributions.
"""
from __future__ import annotations
from test.deployment.source_paths import REPO_ROOT

import ast
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pipeline.deployment.rtx_actor.success_diagnostic_core import FirstEpisodeAccounting


ROOT = REPO_ROOT
REFERENCE = ROOT / "runtime/raven_gr00t_collection_94d7985/gr00t/rl"
METRICS = "envs/loco_manip/walk_stand_grasp.py"
TRAINER = "trl/trainer/distill_trainer_obj_pred_homie_api.py"
HASHES = {
    METRICS: "d985d3e74414b6ceabefd917bdaa3b36c67ff9e141882436f931815f16bd3be9",
    TRAINER: "3da5a2557ae18a8ba53c4da14d8841b80a609b851b326a21534ee2c5001b7584",
}


def reference_tree(relative):
    path = REFERENCE / relative
    if not path.is_file():
        pytest.skip(f"pinned original Raven file unavailable: {path}")
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == HASHES[relative], path
    return ast.parse(raw)


def original_metrics(count):
    names = {"init_eval_metrics_tracking", "update_eval_metrics_per_step",
             "process_eval_episode_completions", "reset_eval_episode_tracking"}
    cls = next(node for node in reference_tree(METRICS).body
               if isinstance(node, ast.ClassDef) and node.name == "WalkStandGrasp")
    methods = [copy.deepcopy(node) for node in cls.body
               if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in methods} == names
    module = ast.Module(body=[ast.ClassDef(name="OriginalMetrics", bases=[], keywords=[],
        body=methods, decorator_list=[])], type_ignores=[])
    scope = {"torch": torch}
    exec(compile(ast.fix_missing_locations(module), "<hash-pinned-raven-metrics>", "exec"), scope)
    env = scope["OriginalMetrics"]()
    env.num_envs = count
    env.get_object_class_names = lambda: None
    env.init_eval_metrics_tracking(torch.device("cpu"))
    return env


def compare_stream(stream, count):
    original = original_metrics(count)
    current = FirstEpisodeAccounting(count)
    seen = torch.zeros(count, dtype=torch.bool)
    rewards = torch.zeros(count)
    lengths = torch.zeros(count, dtype=torch.long)
    completion_order = []
    for step, (goals, dones) in enumerate(stream):
        goals, dones = torch.tensor(goals, dtype=torch.bool), torch.tensor(dones, dtype=torch.bool)
        original.update_eval_metrics_per_step({"grasp_goal_reached": goals})
        lengths += 1
        rewards += torch.arange(count, dtype=torch.float32) + 0.5
        # Raven eval_num_envs_episodes=True: score each lane's first done only,
        # while the native environment may continue/reset that lane afterward.
        new_ids = (dones & ~seen).nonzero()
        original.process_eval_episode_completions(new_ids, rewards, lengths)
        seen |= dones
        done_ids = dones.nonzero()
        original.reset_eval_episode_tracking(done_ids)
        rewards[done_ids] = 0
        lengths[done_ids] = 0
        rows = current.update(step, goals.tolist(), goals.tolist(), dones.tolist(),
                              [False] * count, [{} for _ in range(count)])
        completion_order.extend(row["exported_success"] for row in rows)
        assert completion_order == original.eval_metrics["goal_reached_buffer"]
        result = current.result()
        assert result["completed"] == len(completion_order)
        assert result["exported_successes"] == sum(completion_order)
        assert result["native_successes"] == sum(completion_order)
        assert result["accounting_disagreements"] == 0
    return current.result(), original


def test_original_and_current_score_success_only_at_completion_and_preserve_failures():
    result, original = compare_stream([
        ([True, False, False, False], [False, True, False, False]),
        ([False, True, True, False], [False, True, True, False]),
        ([False, True, False, False], [True, True, False, True]),
    ], 4)
    assert result["complete"] is True
    assert result["exported_successes"] == 2
    assert result["completed"] == 4
    # Lane1's later success cannot replace its first completed failure.
    assert original.eval_metrics["goal_reached_buffer"] == [False, True, True, False]


@pytest.mark.parametrize("count", [1, 2, 7, 16, 32])
def test_randomized_transition_stream_matches_original_first_episode_metric(count):
    generator = torch.Generator().manual_seed(530 + count)
    goals = torch.rand(120, count, generator=generator) < 0.08
    dones = torch.rand(120, count, generator=generator) < 0.11
    dones[-1] = True
    result, _ = compare_stream(zip(goals.tolist(), dones.tolist()), count)
    assert result["complete"]
    assert result["completed"] == count


def test_original_goal_does_not_count_an_unfinished_episode():
    result, _ = compare_stream([([True, False], [False, True])], 2)
    assert not result["complete"]
    assert result["completed"] == 1
    assert result["exported_successes"] == 0


def test_original_evaluation_uses_mean_action_and_supports_first_episode_mode():
    cls = next(node for node in reference_tree(TRAINER).body
               if isinstance(node, ast.ClassDef) and node.name == "TRLDistillTrainerObjPredHomieAPI")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "eval")
    statements = [ast.unparse(node) for node in ast.walk(method) if isinstance(node, ast.Assign)]
    assert "action_state['actions'] = action_state['action_mean']" in statements
    assert any("self.config.eval.get('eval_num_envs_episodes', False)" in row for row in statements)
    assert any("max_episodes = self.env.num_envs" == row for row in statements)
    # The default pooled mode is different from the current fixed first-case
    # panel. Do not call those two experiment protocols identical.
    assert any("self.config.eval.get('num_eval_episodes', self.env.num_envs)" in row for row in statements)


def test_original_random_initial_clock_is_a_real_distribution_difference():
    cls = next(node for node in reference_tree(TRAINER).body
               if isinstance(node, ast.ClassDef) and node.name == "TRLDistillTrainerObjPredHomieAPI")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "eval")
    clock = next(node for node in method.body if isinstance(node, ast.If)
                 and ast.unparse(node.test) == "self.config.get('init_at_random_ep_len', False)")
    # Execute the actual first clock assignment; the staged-clock continuation
    # needs native imports and is intentionally not simulated by this check.
    assignment = copy.deepcopy(clock.body[0])
    scope = {"torch": torch, "self": SimpleNamespace(env=SimpleNamespace(
        episode_length_buf=torch.zeros(32, dtype=torch.long), max_episode_length=400))}
    with torch.random.fork_rng():
        torch.manual_seed(17)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[assignment], type_ignores=[])),
                     "<original-random-clock>", "exec"), scope)
    clocks = scope["self"].env.episode_length_buf
    assert (clocks > 0).any() and (clocks < 400).all()
    # Existing historical panel deliberately disables this, so these tests
    # certify accounting only, not equality of the initialization distribution.
    current = (ROOT / "pipeline/deployment/rtx_actor/isaac6_motion_evaluator.py").read_text()
    assert "self.config.init_at_random_ep_len = False" in current
