"""CPU behavioral comparisons, not Isaac/rendering/production qualification.

The reference methods are AST-extracted from the hash-pinned Raven snapshot.
Only their simulator, policy, storage and transport dependencies are test doubles.
Missing reference assets skip explicitly; different reference bytes fail.
"""
from __future__ import annotations
from test.deployment.source_paths import REPO_ROOT

import ast
import hashlib
import logging
import os
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest

from pipeline.dagger.rtx.code.src.dagger_rtx.contracts import RAVEN_VIEW_ORDER
from pipeline.dagger.rtx.code.src.dagger_rtx.native_raven import NativeRavenBatchComposition
from pipeline.dagger.rtx.code.src.dagger_rtx.production_segment import (
    CONTINUOUS_SCHEMA, ProductionSegmentDriver,
)
from pipeline.dagger.rtx.code.src.dagger_rtx.window_runner import (
    BatchWindowRunner, RolloutLifecycle,
)


REFERENCE = REPO_ROOT / "runtime/raven_gr00t_collection_94d7985/gr00t/rl/trl/trainer"
SOURCE_HASHES = {
    "ppo_trainer.py": "f56a1488af7cc007c81e07abd9b1adfc12ab83b5c541ea163381ef9af743ca61",
    "distill_trainer.py": "dc386dcf8b2be0c151403fd0e8841129c939d99dc5e9e7850e92829b3ad50ca2",
    "distill_trainer_videogen.py": "cc7f8aa3de2a9d60324b9294ba5db7ce15f339d59fa168ef1ccab7484d11890b",
}


def _reference_methods(filename, class_name, names):
    source = REFERENCE / filename
    if not source.is_file():
        pytest.skip(f"pinned Raven reference unavailable: {source}")
    raw = source.read_bytes()
    if filename in SOURCE_HASHES:
        assert hashlib.sha256(raw).hexdigest() == SOURCE_HASHES[filename], "Raven reference changed"
    tree = ast.parse(raw, filename=str(source))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {method.name for method in methods} == set(names)
    return methods


@pytest.fixture
def original_classes():
    torch = pytest.importorskip("torch", reason="CPU tensor reference comparison requires torch")
    nodes = [
        ast.ClassDef(name="OriginalPPO", bases=[], keywords=[], decorator_list=[],
                     body=_reference_methods("ppo_trainer.py", "TRLPPOTrainer", {"_rollout_step"})),
        ast.ClassDef(name="OriginalDistill", bases=[ast.Name(id="OriginalPPO", ctx=ast.Load())],
                     keywords=[], decorator_list=[], body=_reference_methods(
                         "distill_trainer.py", "TRLDistillTrainer", {"_rollout_step"})),
        ast.ClassDef(name="OriginalImageDagger", bases=[ast.Name(id="OriginalPPO", ctx=ast.Load())],
                     keywords=[], decorator_list=[], body=_reference_methods(
                         "distill_trainer_videogen.py", "TRLDistillTrainerVideoGen",
                         {"policy_step", "_image_dagger_relabel"})),
    ]
    namespace = dict(torch=torch, np=np, os=os, time=time, F=torch.nn.functional,
                     logger=logging.getLogger(__name__))
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, "hash_pinned_original_raven_methods", "exec"), namespace)
    return SimpleNamespace(torch=torch, **{name: namespace[name] for name in (
        "OriginalPPO", "OriginalDistill", "OriginalImageDagger")})


class TinyPhysicalEnvironment:
    """CPU state machine; lane0 terminates at81, lane1 at401, others continue."""

    def __init__(self, torch, batch):
        self.torch, self.num_envs = torch, batch
        self.positions = torch.zeros(batch)
        self.steps = 0
        self.reset_calls = 0
        self.native_resets = [0] * batch
        self.action_rows = []

    def observe(self):
        return {"actor_obs": self.positions[:, None].clone(),
                "vision_obs": self.torch.zeros(self.num_envs, 2, 2, 3)}

    def reset_all(self):
        self.reset_calls += 1
        self.positions.zero_()
        return self.observe()

    def step(self, state):
        self.action_rows.append((state["actions"].clone(), state["gt_actions"].clone()))
        self.steps += 1
        self.positions += 1
        dones = self.torch.zeros(self.num_envs, dtype=self.torch.bool)
        dones[0] = self.positions[0] >= 81
        if self.num_envs > 1:
            dones[1] = self.positions[1] >= 401
        for lane in self.torch.nonzero(dones).flatten().tolist():
            self.native_resets[lane] += 1
            self.positions[lane] = 0
        return self.observe(), self.torch.zeros(self.num_envs), dones, {
            "episode": {}, "time_outs": self.torch.zeros_like(dones)}


class TinyStorage:
    def __init__(self):
        self.rows = {}
        self.step = 0
        self.clears = 0

    def clear(self):
        self.rows = {}
        self.step = 0
        self.clears += 1

    def update_key(self, key, value):
        self.rows.setdefault(key, []).append(value.clone())

    def increment_step(self):
        self.step += 1


def _policy(lifecycle, prefix):
    return SimpleNamespace(init_rollout=lambda: lifecycle.append(prefix + ":begin"),
                           clear_rollout=lambda: lifecycle.append(prefix + ":finish"))


def _original_trace(classes, windows=5, batch=16):
    torch = classes.torch
    lifecycle = []
    env = TinyPhysicalEnvironment(torch, batch)
    trainer = classes.OriginalDistill()
    trainer.env = env
    trainer.accelerator = SimpleNamespace(device="cpu")
    trainer.num_steps_per_env = 81
    trainer.value_model = None
    trainer.storage = TinyStorage()
    trainer._train_rollout_mode = lambda: None
    trainer._process_env_step = lambda *args: None
    trainer.ep_infos = []
    trainer.cur_reward_sum = torch.zeros(batch)
    trainer.cur_episode_length = torch.zeros(batch)
    trainer.state = SimpleNamespace(rewbuffer=[], lenbuffer=[])
    trainer.ref_model = _policy(lifecycle, "teacher")
    model = SimpleNamespace(policy=_policy(lifecycle, "student"), value_model=None)

    def policy_step(policy, obs, cur_dones=None):
        # Distinct sentinels ensure behavior actions are not mistaken for labels.
        return {"actions": obs["actor_obs"] - 10, "gt_actions": obs["actor_obs"] + 10}

    trainer.policy_step = policy_step
    observation = env.reset_all()
    updates = []
    for _ in range(windows):
        observation = trainer._rollout_step(model, observation)
        assert trainer.storage.step == 81
        updates.append({key: torch.stack(rows) for key, rows in trainer.storage.rows.items()})
    return SimpleNamespace(env=env, lifecycle=lifecycle, updates=updates, storage=trainer.storage)


def _anchor_binder(batch):
    """Exercise the real per-episode binder with tiny CPU anchor arrays."""
    composition = object.__new__(NativeRavenBatchComposition)
    composition.num_envs = batch
    composition.profile = SimpleNamespace(control_h=1, control_w=1)
    composition._anchors = [None] * batch
    composition._prompts = [None] * batch
    composition._raytraced = [None] * batch
    counts = [0] * batch
    trainer = SimpleNamespace(_id_anchors=[None] * batch, _id_prompts=[None] * batch,
                              _id_use_raytraced={})

    def sample(lane):
        counts[lane] += 1
        trainer._id_anchors[lane] = {
            view: np.full((1, 1, 3), counts[lane], np.uint8) for view in RAVEN_VIEW_ORDER}
        trainer._id_prompts[lane] = "CPU reference anchor"
        trainer._id_use_raytraced[lane] = False

    trainer._id_sample_anchor = sample
    composition.trainer = trainer
    composition.entrypoints = SimpleNamespace(prepare_episode_slots=lambda *_: None)
    return composition, counts


def _continuous_trace(torch):
    env = TinyPhysicalEnvironment(torch, 16)
    observation = env.reset_all()
    binder, anchor_counts = _anchor_binder(16)
    lifecycle, recorded_keys, anchor_rows, rows, updates = [], [], [], [], []

    class Owner:
        num_envs = 16

        def begin_windows(self, requests):
            return [{"ok": True} for _ in requests]

    class Orchestrator:
        num_envs = 16

        def run_batch(self, keys):
            nonlocal observation
            fresh = {lane: key for lane, key in enumerate(keys) if key.episode_step_index == 0}
            binder.bind_episodes(fresh)
            recorded_keys.append(keys)
            anchor_rows.append(tuple(anchor_counts))
            labels = observation["actor_obs"] + 10
            actions = observation["actor_obs"] - 10
            rows.append((observation["actor_obs"].clone(), labels.clone(), actions.clone()))
            observation, _, dones, _ = env.step({"actions": actions, "gt_actions": labels})
            return {"phase": "COMPLETE", "local_commits": [
                {"result": {"done": bool(dones[lane]), "timeout": False},
                 "sealed": keys[lane].rollout_step_index == 80} for lane in range(16)],
                "remote_acknowledgements": [{} for _ in range(16)]}

    runner = BatchWindowRunner(Owner(), Orchestrator(), run_id="raven-continuity-cpu",
        round_id=1, actor_prefix="raven-continuity-cpu-g0", episode_namespace="continuous-r1",
        lifecycle=RolloutLifecycle(begin=lambda: lifecycle.append("begin"),
                                   finish=lambda: lifecycle.append("finish")))
    driver = object.__new__(ProductionSegmentDriver)
    driver.segment = dict(schema=CONTINUOUS_SCHEMA, start_iteration_index=0, window_count=100,
        input_checkpoint_sha256="a" * 64, lifecycle_policy="native_episode_done_only",
        qualification_after_updates=4)
    driver.config = dict(run_id="raven-continuity-cpu", initial_checkpoint_sha256="a" * 64,
                         num_envs_per_gpu=16, physical_rtx_gpus=[0, 1])
    driver.round_id = 1
    driver.source_sha256 = "b" * 64
    driver.runner = runner
    driver.state = {"windows": []}
    driver.selected = None
    driver.status = lambda: {"run": {"status": "collecting", "active_checkpoint_sha256": "a" * 64},
        "iterations": [{"round_id": 1, "iteration_id": 1, "status": "collecting",
                        "input_checkpoint_sha256": "a" * 64, "candidate_checkpoint_sha256": None}],
        "barriers": []}
    driver.save = lambda phase, **fields: driver.state.update(status=phase, **fields)
    driver.request = lambda *_: {"ok": True, "iteration_id": len(updates) + 1, "expected": 32}
    gates = []
    driver._wait_continuous_qualification = lambda checkpoint: gates.append((env.steps, checkpoint))

    def accepted_update(index, parent):
        # The remote learner/other RTX are deliberately not simulated. This
        # records the exact fresh local half-batch present at its real barrier.
        assert index == len(updates)
        assert len(rows) == (index + 1) * 81
        assert parent == ("a" * 64 if not index else hashlib.sha256(str(index).encode()).hexdigest())
        updates.append(rows[index * 81:(index + 1) * 81])
        candidate = hashlib.sha256(str(index + 1).encode()).hexdigest()
        if index == 99:
            driver.selected = {"selected_checkpoint_sha256": candidate, "stop_reason": "fixed_100_updates"}
        return candidate

    driver._candidate = accepted_update
    result = driver.run()
    return SimpleNamespace(env=env, updates=updates, lifecycle=lifecycle, keys=recorded_keys,
        anchors=anchor_rows, anchor_counts=anchor_counts, result=result, gates=gates)


def test_reference_collects_fresh_windows_without_162_step_scene_cutoff(original_classes):
    reference = _original_trace(original_classes)
    assert reference.env.reset_calls == 1
    assert reference.env.steps == 405
    assert reference.storage.clears == 5
    assert reference.env.native_resets[:2] == [5, 1]
    assert reference.lifecycle == ["teacher:begin", "student:begin", "student:finish", "teacher:finish"] * 5
    for window, batch in enumerate(reference.updates):
        assert batch["actor_obs"].shape == (81, 16, 1)
        # Lane2 has no episode end: its state crosses the former162 cutoff.
        assert batch["actor_obs"][0, 2, 0].item() == window * 81
        assert original_classes.torch.equal(batch["gt_actions"], batch["actor_obs"] + 10)
        assert original_classes.torch.equal(batch["actions"], batch["actor_obs"] - 10)


def test_actual_continuous_driver_matches_reference_and_keeps_terminal_anchors(original_classes):
    reference = _original_trace(original_classes)
    current = _continuous_trace(original_classes.torch)
    assert current.env.reset_calls == 1
    assert current.env.steps == 8100
    assert current.lifecycle == ["begin", "finish"] * 100
    assert len(current.updates) == 100
    assert current.result["completed_windows"] == 100
    assert current.result["transition_batches"] == 8100
    assert current.result["physical_transition_rows"] == 8100 * 16
    assert len(current.gates) == 1 and current.gates[0][0] == 4 * 81
    for index, expected in enumerate(reference.updates):
        observed = current.updates[index]
        for column, name in enumerate(("actor_obs", "gt_actions", "actions")):
            assert original_classes.torch.equal(
                original_classes.torch.stack([row[column] for row in observed]), expected[name])
    # Unlike the original's zeroed local dones, the actual persistent cursor
    # preserves a terminal at the very last step of a rollout window.
    assert current.keys[80][0].episode_step_index == 80
    assert current.keys[81][0].episode_step_index == 0
    assert current.keys[80][0].episode_id != current.keys[81][0].episode_id
    assert current.anchors[80][0] == 1 and current.anchors[81][0] == 2
    # A nonterminal lane retains its episode/anchor across162 and beyond.
    assert current.keys[162][2].episode_step_index == 162
    assert current.keys[404][2].episode_step_index == 404
    assert current.anchor_counts[2] == 1


def test_original_image_dagger_relabel_does_not_reach_training_storage(original_classes, monkeypatch):
    """Document an intentional exclusion, not a bug to copy for parity."""
    torch = original_classes.torch
    monkeypatch.delenv("IMAGE_DAGGER_DUMP_DIR", raising=False)
    trainer = original_classes.OriginalImageDagger()
    batch = 1
    trainer.env = TinyPhysicalEnvironment(torch, batch)
    views = {"head": {"depth": torch.ones(1, 2, 2, 1),
                       "robot_rgb": torch.zeros(1, 2, 2, 3, dtype=torch.uint8)}}
    trainer.env.simulator = SimpleNamespace(get_conditioning_views=lambda: views)
    trainer.accelerator = SimpleNamespace(device="cpu")
    trainer.num_steps_per_env = 2
    trainer.value_model = None
    trainer.storage = TinyStorage()
    trainer._train_rollout_mode = lambda: None
    trainer._process_env_step = lambda *_: None
    trainer.ep_infos = []
    trainer.cur_reward_sum = torch.zeros(batch)
    trainer.cur_episode_length = torch.zeros(batch)
    trainer.state = SimpleNamespace(rewbuffer=[], lenbuffer=[])
    trainer._in_train_rollout = True
    trainer.vg_enabled = True
    trainer.vg_mode = "image_dagger"
    trainer._depth_chunk, trainer._robot_rgb_chunk = [], []
    trainer._get_depth_frame = lambda: torch.zeros(1, 2, 2, 1)
    trainer._id_anchors = [{"head": np.zeros((2, 2, 3), np.uint8)}]
    trainer._id_prompts = ["CPU sentinel"]
    trainer._id_use_raytraced = {0: False}
    trainer.camera_resolution = (2, 2)
    trainer._id_env_clients = [SimpleNamespace(render_frame=lambda *args, **kwargs: {
        "head": np.full((2, 2, 3), 255, np.uint8)})]
    trainer.vg_view_names = ["head"]
    trainer.vg_control_res = None
    trainer._h5v4_control = lambda depth, view: np.zeros(depth.shape, dtype=np.uint8)
    trainer.vg_seed = 42
    trainer._policy_views = ["head"]
    trainer._id_gen_ms = [0.]
    trainer._normalize_like_env = lambda value: value
    seen = []

    def base_policy(self, policy, observation, cur_dones=None):
        seen.append(observation["vision_obs"].clone())
        # No gt_actions returned: this test isolates observation aliasing, not
        # the independent behavior-mixture branch. Env still records its label.
        return {"actions": torch.zeros(batch, 1)}

    original_step = trainer.env.step
    trainer.env.step = lambda state: original_step({**state, "gt_actions": torch.ones(batch, 1)})
    monkeypatch.setattr(original_classes.OriginalPPO, "policy_step", base_policy, raising=False)
    model = SimpleNamespace(policy=_policy([], "student"), value_model=None)
    caller_observation = trainer.env.reset_all()
    trainer._rollout_step(model, caller_observation)
    assert len(seen) == len(trainer.storage.rows["vision_obs"]) == 2
    assert all(torch.all(value == 1) for value in seen)
    assert all(torch.all(value == 0) for value in trainer.storage.rows["vision_obs"])
    assert torch.all(caller_observation["vision_obs"] == 0)
