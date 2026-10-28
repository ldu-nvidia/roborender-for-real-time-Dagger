"""Coupled CPU comparison, NOT native/render/model/production certification.

Pinned Raven methods execute unchanged, except that their ``torch.rand`` is
given an explicit tape of current hash selections. This is a conditional action
comparison, NOT seeded-RNG parity. Both paths receive declared generated images
as their observations: Raven's native-storage relabel bug is deliberately not
reproduced. The simulator, renderer, teacher, Homie, network and transport are
small, named doubles. Real current window sequencing, selection, artifact seal,
chunk loader and full-population learner methods are exercised together.

Original lane permutation remains enabled. We compare resulting floating-point
updates with tolerances, record the order/RNG difference, and never assert exact
historical optimizer, full policy, physics or distributed-service equivalence.
The original VideoGen._rollout_step and PPO.train are NOT invoked: original
PPO/Distill collection methods and VideoGen.policy_step are joined explicitly.
randperm, AdamW configuration/clipping/step are harness calls. NativeRaven,
learner main/admission/checkpoint serialization and the real backend are outside
this test's scope.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import gc
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest
import torch

from pipeline.dagger.rtx.code.src.dagger_rtx.window_runner import BatchWindowRunner, RolloutLifecycle
from pipeline.dagger.services import raven_learner_worker as learner
from pipeline.dagger.services import render_student_server as service
from pipeline.dagger.services.step_contract import (
    canonical_bytes, bytes_sha256, decode_array, encode_array, response_receipt,
)
from pipeline.dagger.services.store import StateStore
from test.deployment.source_paths import REPO_ROOT
from test.pipeline.test_bc_raven_offline_semantics import _original_student_class


REFERENCE = REPO_ROOT / "runtime/raven_gr00t_collection_94d7985/gr00t/rl/trl"
PINS = {
    "trainer/ppo_trainer.py": "f56a1488af7cc007c81e07abd9b1adfc12ab83b5c541ea163381ef9af743ca61",
    "trainer/distill_trainer.py": "dc386dcf8b2be0c151403fd0e8841129c939d99dc5e9e7850e92829b3ad50ca2",
    "trainer/distill_trainer_obj_pred.py": "f0aad847ba7b42da743f9a983bfa7922d89ba3bfab82b86941efaf4a2b6ca19e",
    "trainer/distill_trainer_obj_pred_homie_api.py": "3da5a2557ae18a8ba53c4da14d8841b80a609b851b326a21534ee2c5001b7584",
    "trainer/distill_trainer_videogen.py": "cc7f8aa3de2a9d60324b9294ba5db7ce15f339d59fa168ef1ccab7484d11890b",
    "utils/rl.py": "efd4dafa49e45fcd553829480d112018fcf2fa91a9e2858b9a96fdde71c36931",
    "modules/vision_actor_critic_modules.py": "dfda777a937f1f8253e0eb8979045784da9731dfea51c9fbf4fea1ed39e00733",
    "modules/vision_actor_critic_modules_obj_pred.py": "86acfbd0cf610aec49e3a11e7049b18d4ca83d3bf28bcb63fc3484683c1cb239",
}
VIEWS = ("head_color", "hand_left", "hand_right")
BATCH = 4
SPEC = {"input_contract": {"views": VIEWS, "rgb_mean": [0., 0., 0.],
                            "rgb_std": [1., 1., 1.]},
        "output_contract": {"policy_action_dim": 11}}


def pinned_tree(relative):
    path = REFERENCE / relative
    assert path.is_file(), f"required pinned Raven reference unavailable: {path}"
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == PINS[relative], f"Raven reference changed: {path}"
    return ast.parse(raw, filename=str(path))


def methods(relative, name, selected):
    cls = next(n for n in pinned_tree(relative).body if isinstance(n, ast.ClassDef) and n.name == name)
    result = [deepcopy(n) for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in selected]
    assert {n.name for n in result} == set(selected)
    return result


class SelectionTape:
    """Only the original Bernoulli call is replaced; all other torch calls real."""
    def __init__(self):
        self.rows = []
        self.index = 0

    def rand(self, count, *, device):
        assert count == BATCH and device == torch.device("cpu")
        row = self.rows[self.index]
        self.index += 1
        return torch.tensor(row, dtype=torch.float64)

    def __getattr__(self, name):
        return getattr(torch, name)


def original_classes(tape):
    definitions = [
        ("PPO", None, "trainer/ppo_trainer.py", "TRLPPOTrainer", {"_rollout_step"}),
        ("Distill", "PPO", "trainer/distill_trainer.py", "TRLDistillTrainer",
         {"_rollout_step", "_get_rollout_data", "_get_mb_rollout_data", "_compute_loss"}),
        ("Obj", "Distill", "trainer/distill_trainer_obj_pred.py", "TRLDistillTrainerObjPred",
         {"_get_rollout_data", "_get_mb_rollout_data", "_forward_model", "_compute_loss", "_compute_obj_pred_loss"}),
        ("Homie", "Obj", "trainer/distill_trainer_obj_pred_homie_api.py", "TRLDistillTrainerObjPredHomieAPI",
         {"policy_step", "_compute_dagger_bc_loss"}),
        ("Video", "Homie", "trainer/distill_trainer_videogen.py", "TRLDistillTrainerVideoGen", {"policy_step"}),
        ("Wrapper", None, "trainer/distill_trainer_obj_pred_homie_api.py", "PolicyAndValueWrapperObjPredHomie",
         {"forward", "forward_component"}),
    ]
    nodes = [deepcopy(next(n for n in pinned_tree("utils/rl.py").body
                          if isinstance(n, ast.FunctionDef) and n.name == "compute_episode_attnmask"))]
    for name, base, file, original, selected in definitions:
        nodes.append(ast.ClassDef(name=name, bases=[] if base is None else [ast.Name(id=base, ctx=ast.Load())],
            keywords=[], decorator_list=[], body=methods(file, original, selected)))
    namespace = dict(torch=tape, np=np, os=os, gc=gc, time=time, deepcopy=deepcopy,
        init_actor_critic_dict={"num_one_step_obs": 4}, logger=logging.getLogger(__name__))
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 "hash_pinned_raven_coupled_methods", "exec"), namespace)
    return SimpleNamespace(**{key: namespace[key] for key in ("Video", "Wrapper", "compute_episode_attnmask")})


@pytest.fixture(autouse=True)
def cpu_only_rng(monkeypatch):
    before, threads = torch.get_rng_state(), torch.get_num_threads()
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.set_num_threads(1)
    yield
    torch.set_rng_state(before)
    torch.set_num_threads(threads)


def tiny_policy():
    """Real original buffer/distribution methods; tiny78-slot feature network."""
    # This coupled gate requires its oracle; the older reusable helper alone
    # permits skips, so verify both required exact sources before calling it.
    pinned_tree("modules/vision_actor_critic_modules.py")
    pinned_tree("modules/vision_actor_critic_modules_obj_pred.py")
    Original = _original_student_class()

    class TinyStudent(Original):
        num_actions = 11

        def __init__(self):
            torch.nn.Module.__init__(self)
            self.bn = torch.nn.SyncBatchNorm(3, momentum=.001)
            self.mlp_module = torch.nn.ModuleList([torch.nn.Linear(3, 3) for _ in range(33)]
                                                + [torch.nn.Linear(4, 11)])
            self.obj_pred_mlp = torch.nn.Sequential(*[torch.nn.Linear(3, 3) for _ in range(4)])
            self.std = torch.nn.Parameter(torch.full((11,), .001), requires_grad=False)
            self.forward_populations = []

        def forward(self, obs_dict, **_):
            vision, actor = obs_dict["vision_obs"], obs_dict["actor_obs"]
            image = vision.reshape(-1, *vision.shape[-3:]).permute(0, 3, 1, 2)
            self.forward_populations.append((self.training, torch.is_grad_enabled(), image.shape[0]))
            features = self.bn(image).mean((2, 3)).reshape(*vision.shape[:-3], 3)
            hidden = features
            for layer in self.mlp_module[:-1]:
                hidden = hidden + .01 * torch.tanh(layer(hidden))
            return {"actions": self.mlp_module[-1](torch.cat((hidden, actor[..., :1]), dim=-1)),
                    "obj_pred": self.obj_pred_mlp(features)}

        def get_actions_log_prob(self, actions):
            return self.distribution.log_prob(actions).sum(-1)

    policy = TinyStudent()
    assert len([p for p in policy.parameters() if p.requires_grad]) == 78
    return policy


class ScriptedEnvironment:
    """Not physics: deterministic input tape, terminals80/81/82/401, logs actions."""
    num_envs = BATCH
    config = {}
    simulator = SimpleNamespace()

    def __init__(self):
        self.age = torch.zeros(BATCH, dtype=torch.long)
        self.steps = self.initial_resets = 0
        self.natural_resets = torch.zeros(BATCH, dtype=torch.long)
        self.actions = []

    def observe(self):
        actor = self.age.float()[:, None] / 1000
        lanes = torch.arange(BATCH).float()[:, None, None, None] / 100
        views = [torch.full((BATCH, 2, 2, 3), (i + 1) / 10) + lanes + actor[:, :, None, None] / 10
                 for i in range(3)]
        homie = torch.zeros(BATCH, 4)
        homie[::2, 0] = 1
        return dict(actor_obs=actor, vision_obs=torch.cat(views, dim=1),
                    teacher_obs=actor + 2, homie_obs=homie, gt_obj_pos=torch.ones(BATCH, 3))

    def reset_all(self):
        self.initial_resets += 1
        self.age.zero_()
        return self.observe()

    def step(self, state):
        self.actions.append({k: v.clone() for k, v in state.items()})
        self.steps += 1
        self.age += 1
        done = self.age >= torch.tensor([80, 81, 82, 401])
        self.natural_resets += done
        self.age[done] = 0
        return self.observe(), torch.zeros(BATCH), done, {"episode": {}, "time_outs": torch.zeros_like(done)}


class RowStorage:
    def clear(self):
        self.rows, self.row, self.step = [], {}, 0

    def update_key(self, key, value):
        self.row[key] = value.clone()

    def increment_step(self):
        self.rows.append(self.row)
        self.row = {}
        self.step += 1

    def __getattr__(self, key):
        if key.startswith("__") or not self.rows or key not in self.rows[0]:
            raise AttributeError(key)
        return torch.stack([row[key] for row in self.rows])


class Config(dict):
    __getattr__ = dict.__getitem__


class HomieDouble:
    def __init__(self, value):
        self.value = value

    def __call__(self, observation):
        actions = torch.full((*observation.shape[:-1], 15), self.value)
        return dict(actions=actions, action_mean=actions, action_sigma=torch.ones_like(actions),
                    entropy=torch.zeros(observation.shape[0]))

    def get_actions_log_prob(self, actions):
        return torch.zeros(actions.shape[0])


def reference_runner(classes, policy):
    runner = classes.Video()
    runner.env = ScriptedEnvironment()
    runner.accelerator = SimpleNamespace(device=torch.device("cpu"))
    runner.num_steps_per_env, runner.value_model = 81, None
    runner.storage = RowStorage()
    runner.policy_model = policy
    runner.learn_normalized_actions = False
    runner.config = Config(enforce_teacher_rollout=False, ratio_teacher_rollout=0,
                           dagger_bc_loss_coef=1., obj_pred_loss_coef=0., use_obj_pred=True)
    runner.bc_loss_fn = runner.obj_pred_loss_fn = torch.nn.MSELoss()
    runner.enable_obj_pred_visualization = runner.export_homie_model = False
    runner._process_env_step = lambda *_: None
    runner.ep_infos = []
    runner.cur_reward_sum, runner.cur_episode_length = torch.zeros(BATCH), torch.zeros(BATCH)
    runner.state = SimpleNamespace(rewbuffer=[], lenbuffer=[], global_step=0)
    runner.ref_model = SimpleNamespace(act_inference=lambda obs_dict, input_key:
        obs_dict[input_key].repeat(1, 11), init_rollout=lambda: None, clear_rollout=lambda: None)
    wrapper = classes.Wrapper()
    wrapper.policy, wrapper.value_model, wrapper.opt_homie, wrapper.training = policy, None, False, False
    wrapper.homie_walk_model, wrapper.homie_stand_model = HomieDouble(.3), HomieDouble(-.2)
    wrapper.homie_switch_threshold = .5
    runner.unwrapped_model = wrapper
    runner._train_rollout_mode = policy.eval
    # Explicit generated-observation adapter: environment already supplies the
    # generated pixels, so reference image_dagger relabel/storage bug is not run.
    runner._in_train_rollout, runner.vg_enabled, runner.vg_mode = True, True, "collect"
    runner._depth_chunk, runner._robot_rgb_chunk = [], []
    runner._get_depth_frame = lambda: torch.zeros(1)
    runner._dagger_beta = lambda: .75
    return runner, wrapper


def current_selection(handler):
    """Execute the UNCHANGED nested function from current production source."""
    path = Path(inspect.getsourcefile(service.Handler))
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Handler")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_prepare_batch")
    selected = [n for n in ast.walk(method) if isinstance(n, ast.FunctionDef) and n.name == "select_lane"]
    assert len(selected) == 1
    namespace = dict(np=np, time=time, bytes_sha256=bytes_sha256, canonical_bytes=canonical_bytes,
        encode_array=encode_array, response_receipt=response_receipt, self=handler,
        expected_contract="roborender.dagger.rtx.v4")
    exec(compile(ast.fix_missing_locations(ast.Module(body=[deepcopy(selected[0])], type_ignores=[])),
                 str(path), "exec"), namespace)
    return namespace["select_lane"]


def seal(handler, records, loaded, key):
    """Transport double builds an actual seal; production reconciles/writes it."""
    transitions = [json.loads(r["transition_json"]) for r in records]
    selections = [json.loads(r["response_json"]) for r in records]
    metadata = [json.loads(r["metadata_json"]) for r in records]
    starts = [dict(episode_id=r["episode_id"], first_rollout_step_index=i,
        use_raytraced=False, observation_source="generated_controls", anchor_sha256="c" * 64)
        for i, r in enumerate(records) if i == 0 or r["episode_id"] != records[i - 1]["episode_id"]]
    digests, receipts = [], []
    for selection, transition in zip(selections, transitions, strict=True):
        body = {k: selection[k] for k in ("step_key", "prepare_receipt", "selection_id",
                                         "selected_source", "beta", "selected_action")}
        digest = bytes_sha256(canonical_bytes(body))
        digests.append(digest)
        receipts.append(bytes_sha256(canonical_bytes(dict(commit_digest=digest, result=transition))))
    value = dict(rollout_window_id=key.rollout_window_id,
        provenance=dict(episode_bindings=starts, teacher_checkpoint_sha256="d" * 64),
        episode_ids=[r["episode_id"] for r in records], episode_step_indices=[r["episode_step_index"] for r in records],
        env_ids=[m["env_id"] for m in metadata], seqs=[m["seq"] for m in metadata], round_ids=[1] * 81,
        observation_sources=["generated_controls"] * 81, dones=[r["done"] for r in transitions],
        timeouts=[r["timeout"] for r in transitions], rewards=[r["reward"] for r in transitions],
        behavior_sources=[s["selected_source"] for s in selections],
        prepare_receipts=[r["prepare_receipt"] for r in records], capture_digests=[r["capture_digest"] for r in records],
        selection_ids=[s["selection_id"] for s in selections], commit_digests=digests, commit_receipts=receipts)
    handler._seal_window_records(key, value, records, loaded)
    return value


class CurrentCollection:
    num_envs = BATCH

    def __init__(self, path, policy, tape):
        self.policy, self.tape, self.env = policy, tape, ScriptedEnvironment()
        self.observation = self.env.reset_all()
        self.keys_seen, self.responses, self.paths, self.seals = [], [], [], []
        self.rows, self.arrays = [[] for _ in range(BATCH)], [[] for _ in range(BATCH)]
        (path / "sealed").mkdir(parents=True)
        self.handler = object.__new__(service.Handler)
        self.handler.server = SimpleNamespace(root=path, store=StateStore(path / "ledger"),
            trajectory_sink=SimpleNamespace(publish=lambda file, reservation: self.paths.append(file)))
        self.select = current_selection(self.handler)
        self.runner = BatchWindowRunner(self, self, run_id="coupled-cpu", round_id=1,
            actor_prefix="cpu-g0", episode_namespace="scripted", lifecycle=RolloutLifecycle(
                begin=policy.eval, finish=lambda: None))

    def begin_windows(self, requests):
        digest = hashlib.sha256()
        for name, value in self.policy.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().numpy().tobytes())
        self.checkpoint_sha256 = digest.hexdigest()
        return [dict(ok=True) for _ in requests]

    def run_batch(self, keys):
        self.keys_seen.append(keys)
        selected, results = [], []
        for lane, key in enumerate(keys):
            actor = self.observation["actor_obs"][lane].numpy()
            teacher = np.concatenate((np.repeat(self.observation["teacher_obs"][lane].numpy(), 11),
                                      np.full(15, .3 if lane % 2 == 0 else -.2, np.float32)))
            vision = self.observation["vision_obs"][lane]
            with torch.no_grad():
                action = self.policy({"vision_obs": vision[None, None],
                                      "actor_obs": torch.from_numpy(actor)[None, None]})["actions"].reshape(11).numpy()
            result = SimpleNamespace(student_action_11=action,
                policy_rgb={v: vision[i * 2:(i + 1) * 2].numpy() for i, v in enumerate(VIEWS)},
                observation_digest="e" * 64, telemetry={})
            arrays = dict(actor_obs=actor, teacher_action=teacher)
            meta = dict(env_id=lane, seq=self.env.steps, observation_source="generated_controls",
                        episode_assignment=dict(use_raytraced=False, anchor_sha256="c" * 64))
            lane_request = dict(idempotency_key=f"step-{self.env.steps}-{lane}",
                               capture_digest="f" * 64, wire_digest="1" * 64, prepare_receipt="2" * 64)
            response = self.select(((key, .75, meta, arrays, lane_request, "3" * 64, self.checkpoint_sha256), result, 0.))
            results.append(response)
            selected.append(decode_array(response["response"]["selected_action"]))
        self.tape.rows.append([r["response"]["selection_draw"] for r in results])
        labels = torch.stack([torch.from_numpy(r["artifact_arrays"]["teacher_action"]) for r in results])
        self.observation, _, dones, _ = self.env.step(dict(actions=torch.from_numpy(np.stack(selected)), gt_actions=labels))
        for lane, (key, result) in enumerate(zip(keys, results, strict=True)):
            response = result["response"]
            row = dict(rollout_step_index=key.rollout_step_index, episode_id=key.episode_id,
                episode_step_index=key.episode_step_index, round_id=1, checkpoint_sha256=self.checkpoint_sha256,
                prepare_receipt=response["prepare_receipt"], capture_digest=response["capture_digest"],
                metadata_json=json.dumps(result["metadata"]), response_json=json.dumps(response),
                transition_json=json.dumps(dict(done=bool(dones[lane]), timeout=False, reward=0.)))
            self.rows[lane].append(row)
            self.arrays[lane].append(result["artifact_arrays"])
            if key.rollout_step_index == 80:
                self.seals.append(seal(self.handler, self.rows[lane], self.arrays[lane], key))
                self.rows[lane], self.arrays[lane] = [], []
        self.responses.extend(results)
        return dict(phase="COMPLETE", local_commits=[dict(result=dict(done=bool(done), timeout=False),
                    sealed=keys[0].rollout_step_index == 80) for done in dones],
                    remote_acknowledgements=[{} for _ in keys])


def assert_close_state(first, second, measurements=None, prefix="", moment_bounds=None):
    assert first.keys() == second.keys()
    for key in first:
        if isinstance(first[key], dict):
            assert_close_state(first[key], second[key], measurements, f"{prefix}{key}.", moment_bounds)
        elif isinstance(first[key], torch.Tensor):
            if measurements is not None:
                left, right = first[key].double(), second[key].double()
                absolute = (left - right).abs()
                relative = absolute / torch.maximum(left.abs(), right.abs()).clamp_min(1e-12)
                for name, value in (("max_absolute", float(absolute.max())),
                                    ("max_relative_floor_1e-12", float(relative.max()))):
                    if value > measurements.get(name, -1.):
                        measurements[name] = value
                        measurements[f"{name}_tensor"] = f"{prefix}{key}"
                measurements["tensors_compared"] = measurements.get("tensors_compared", 0) + 1
                if key in ("exp_avg", "exp_avg_sq"):
                    scales = measurements.setdefault("moment_scales", {}).setdefault(key, {})
                    scales["maximum_absolute_value"] = max(scales.get("maximum_absolute_value", 0.),
                                                           float(torch.maximum(left.abs(), right.abs()).max()))
                    nonzero = torch.cat((left.flatten(), right.flatten())).abs()
                    nonzero = nonzero[nonzero != 0]
                    if nonzero.numel():
                        scales["minimum_nonzero_absolute_value"] = min(
                            scales.get("minimum_nonzero_absolute_value", float("inf")), float(nonzero.min()))
            if key in ("exp_avg", "exp_avg_sq"):
                # Never apply a fixed1e-6 floor to moments. Their permitted
                # difference is derived per element from the already-checked
                # clipped gradients, preceding moments and float32 roundoff.
                assert moment_bounds is not None, "optimizer moments require propagated bounds"
                bound = moment_bounds[f"{prefix}{key}"]
                assert torch.equal(first[key] == 0, second[key] == 0), f"{prefix}{key}: zero support changed"
                difference = (first[key].double() - second[key].double()).abs()
                assert bool(torch.all(difference <= bound)), f"{prefix}{key}: exceeds gradient-derived bound"
                continue
            if key == "step":
                assert torch.equal(first[key], second[key]), f"{prefix}{key}: optimizer step changed"
                continue
            # The full population is identical but reduction order is not.
            # Float32 SyncBN reductions on this tiny fixture differ by ~1e-6;
            # exact tensor equality would falsely assert lane-order parity.
            torch.testing.assert_close(first[key], second[key], rtol=2e-4, atol=1e-6,
                                       msg=lambda message: f"{prefix}{key}: {message}")
        else:
            assert first[key] == second[key]


def before_adamw_step(optimizer):
    before = deepcopy(optimizer.state_dict())
    gradients = {}
    for group, saved in zip(optimizer.param_groups, before["param_groups"], strict=True):
        for parameter, ident in zip(group["params"], saved["params"], strict=True):
            gradients[ident] = parameter.grad.detach().clone()
    return before, gradients


def assert_adamw_transition(before, gradients, after):
    """Independent elementwise moment recurrence; no absolute tolerance floor."""
    assert before["param_groups"] == after["param_groups"]
    assert set(after["state"]) == set(gradients)
    for group in after["param_groups"]:
        beta1, beta2 = group["betas"]
        for ident in group["params"]:
            gradient = gradients[ident].double()
            old = before["state"].get(ident, {})
            actual = after["state"][ident]
            expected_step = int(old.get("step", 0)) + 1
            assert float(actual["step"]) == expected_step, "optimizer step changed"
            for field, beta, term in (("exp_avg", beta1, gradient),
                                      ("exp_avg_sq", beta2, gradient.square())):
                prior = old[field].double() if field in old else torch.zeros_like(gradient)
                expected = prior * beta + term * (1. - beta)
                # Relative error alone is ill-conditioned when first-moment
                # terms cancel. Bound float32 operation rounding by operand
                # magnitudes, not by a fixed absolute floor or the tiny result.
                rounding = 8 * torch.finfo(torch.float32).eps * (
                    beta * prior.abs() + (1. - beta) * term.abs())
                assert bool(torch.all((actual[field].double() - expected).abs() <= rounding)), \
                    f"{ident}.{field}: differs from independent moment recurrence"
                assert torch.equal(actual[field] == 0, expected == 0), f"{ident}.{field}: lost nonzero moments"


def adamw_difference_bounds(first_before, first_gradients, second_before, second_gradients):
    """Bounds follow AdamW's recurrence, including the observed lane-order error."""
    result = {}
    assert first_before["param_groups"] == second_before["param_groups"]
    epsilon = torch.finfo(torch.float32).eps
    for group in first_before["param_groups"]:
        for ident in group["params"]:
            first, second = first_gradients[ident].double(), second_gradients[ident].double()
            old1, old2 = first_before["state"].get(ident, {}), second_before["state"].get(ident, {})
            for field, beta, term1, term2 in (("exp_avg", group["betas"][0], first, second),
                    ("exp_avg_sq", group["betas"][1], first.square(), second.square())):
                prior1 = old1[field].double() if field in old1 else torch.zeros_like(first)
                prior2 = old2[field].double() if field in old2 else torch.zeros_like(second)
                scale = beta * (prior1.abs() + prior2.abs()) + (1. - beta) * (term1.abs() + term2.abs())
                result[f"state.{ident}.{field}"] = (beta * (prior1 - prior2).abs()
                    + (1. - beta) * (term1 - term2).abs() + 8 * epsilon * scale)
    return result


def assert_window_tensors(loaded, rollout):
    vision, actor, target = loaded
    torch.testing.assert_close(vision, rollout["vision_obs"], rtol=0, atol=0)
    torch.testing.assert_close(actor, rollout["actor_obs"], rtol=0, atol=0)
    torch.testing.assert_close(target, rollout["gt_actions"][..., :11], rtol=0, atol=0)
    assert vision.shape[:2] == (BATCH, 81)


@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize('native_learner_order', [False, True])
def test_coupled_three_windows_original_collection_to_current_adamw(tmp_path, recompute, record_property, native_learner_order):
    torch.random.default_generator.manual_seed(512)
    tape = SelectionTape()
    classes = original_classes(tape)
    original = tiny_policy()
    current = deepcopy(original)
    reference, wrapper = reference_runner(classes, original)
    observation = reference.env.reset_all()
    collection = CurrentCollection(tmp_path, current, tape)
    optimizers = [torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=.0002, weight_decay=0.) for model in (original, current)]
    permutations, checkpoint_ids, numeric_report = [], [], []
    for window in range(3):
        rollout_rng = torch.get_rng_state().clone()
        collection.runner.run_window(window)
        assert torch.equal(rollout_rng, torch.get_rng_state())
        checkpoint_ids.append(collection.checkpoint_sha256)
        with torch.no_grad():
            observation = reference._rollout_step(wrapper, observation)
        assert not torch.equal(rollout_rng, torch.get_rng_state())
        assert tape.index == (window + 1) * 81
        for expected, actual in zip(reference.env.actions[-81:], collection.env.actions[-81:], strict=True):
            torch.testing.assert_close(expected["actions"], actual["actions"], rtol=1e-6, atol=1e-6)
            torch.testing.assert_close(expected["gt_actions"], actual["gt_actions"], rtol=0, atol=0)
        rollout = reference._get_rollout_data(observation.keys())
        vision, actor, target = learner.load_batch(sorted(collection.paths[-BATCH:]), torch.device("cpu"), SPEC)
        assert_window_tensors((vision, actor, target), rollout)
        for lane in range(BATCH):
            np.testing.assert_array_equal(collection.seals[window * BATCH + lane]["dones"],
                                          rollout["dones"][lane].numpy())
            assert collection.seals[window * BATCH + lane]["timeouts"] == [False] * 81
        if window == 0:
            # Lane0 terminates after observation79; observation80 is a fresh
            # episode. Raven builds an attention mask but this feed-forward
            # MSE does not remove either terminal or post-reset row.
            mask = classes.compute_episode_attnmask(rollout["dones"])
            assert bool(mask[0, 79, 80]) and not bool(mask[0, 78, 79])
        assert not torch.equal(torch.stack([r["actions"] for r in reference.storage.rows])[..., :11],
                               torch.stack([r["gt_actions"] for r in reference.storage.rows])[..., :11])

        # The actual original permutation is NOT removed or copied into current.
        before_permutation = torch.get_rng_state().clone()
        permutation = torch.randperm(BATCH)
        permutations.append(permutation)
        minibatch = reference._get_mb_rollout_data(rollout, permutation)
        assert minibatch["mb_masks"] is None  # non-recurrent; do not mask terminal rows out of MSE
        assert torch.equal(minibatch["episode_attnmask"], classes.compute_episode_attnmask(rollout["dones"][permutation]))
        original.train(); current.train()
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        forward = reference._forward_model(wrapper, minibatch)
        losses = reference._compute_loss(forward, minibatch)
        losses["loss"].backward()
        original_rng = torch.get_rng_state().clone()
        torch.set_rng_state(before_permutation)
        if native_learner_order:
            from pipeline.dagger.services import learner_order
            request=dict(train_chunk_ids=[f'lane{i}' for i in range(BATCH)], parameters={
                learner_order.PARAMETER:learner_order.PROTOCOL,
                'learner_rng_protocol':'roborender.learner-rng-continuation.v1',
                'raven_exact_full_batch':True,'num_mini_batches':1,'microbatch_environments':BATCH})
            (vision,actor,target),receipt=learner_order.permute_batch(request,vision,actor,target)
            assert receipt['environment_indices']==permutation.tolist()
        actual_loss = learner.backward_global_mse_in_environment_microbatches(current, vision, actor, target,
            microbatch_environments=BATCH, full_environment_batch_per_timestep=True,
            activation_checkpointing=recompute, zero_auxiliary_object_loss=True)
        # Same complete population/loss, but not identical original RNG history.
        assert torch.equal(original_rng, torch.get_rng_state()) is native_learner_order
        assert actual_loss == pytest.approx(float(losses["dagger_bc_loss"].detach()), rel=2e-6)
        measured = dict(window=window, lane_permutation=permutation.tolist(), gradients={}, policy={}, optimizer={},
                        loss_absolute=abs(actual_loss - float(losses["dagger_bc_loss"].detach())))
        assert_close_state({n: p.grad for n, p in original.named_parameters() if p.requires_grad},
                           {n: p.grad for n, p in current.named_parameters() if p.requires_grad}, measured["gradients"])
        optimizer_transitions = []
        for model, optimizer in zip((original, current), optimizers, strict=True):
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            before, gradients = before_adamw_step(optimizer)
            optimizer.step()
            assert_adamw_transition(before, gradients, optimizer.state_dict())
            optimizer_transitions.append((before, gradients))
            assert len(optimizer.state) == 78
            assert all(int(state["step"]) == window + 1 for state in optimizer.state.values())
            for parameter in model.obj_pred_mlp.parameters():
                assert torch.count_nonzero(parameter.grad) == 0
                assert torch.count_nonzero(optimizer.state[parameter]["exp_avg"]) == 0
                assert torch.count_nonzero(optimizer.state[parameter]["exp_avg_sq"]) == 0
        assert_close_state(original.state_dict(), current.state_dict(), measured["policy"])
        if native_learner_order:
            for key,value in original.state_dict().items():
                assert torch.equal(value,current.state_dict()[key]),key
            for first,second in zip(optimizers[0].state.values(),optimizers[1].state.values()):
                for key in first:assert torch.equal(first[key],second[key]),key
        moment_bounds = adamw_difference_bounds(*optimizer_transitions[0], *optimizer_transitions[1])
        assert_close_state(optimizers[0].state_dict(), optimizers[1].state_dict(), measured["optimizer"],
                           moment_bounds=moment_bounds)
        numeric_report.append(measured)
        assert int(original.bn.num_batches_tracked) == int(current.bn.num_batches_tracked) == window + 1
    assert any(not torch.equal(p, torch.arange(BATCH)) for p in permutations)
    assert len(set(checkpoint_ids)) == 3  # actual updated tiny-policy tensors feed the next window
    assert collection.env.initial_resets == reference.env.initial_resets == 1
    assert collection.env.steps == reference.env.steps == 243
    assert torch.equal(collection.env.natural_resets, reference.env.natural_resets)
    assert collection.keys_seen[162][3].episode_step_index == 162
    assert collection.keys_seen[242][3].episode_step_index == 242
    for lane, step in enumerate((80, 81, 82)):
        assert collection.keys_seen[step][lane].episode_step_index == 0
        assert collection.keys_seen[step - 1][lane].episode_id != collection.keys_seen[step][lane].episode_id
    assert len(collection.paths) == 12
    record_property("numeric_difference_report", json.dumps(dict(
        schema="test.dagger-coupled-cpu-numerics.v1", allclose_atol=1e-6,
        allclose_rtol=2e-4, tolerance_rule="abs(a-b)<=atol+rtol*abs(reference)",
        floating_point="float32", recompute=recompute,
        optimizer_moment_check="independent recurrence and cross-path gradient-propagated8eps operand bounds, no fixed atol; exact zero support/step",
        original_lane_permutation_preserved=True, original_behavior_draws="explicit_current_hash_tape",
        global_rng_equivalent=False, network="tiny78-slot-double", windows=numeric_report), sort_keys=True))


def test_generated_artifact_never_substitutes_behavior_for_teacher_or_native_rgb():
    arrays = dict(actor_obs=np.zeros(1, np.float32), teacher_action=np.ones(26, np.float32),
                  rgb_native=np.full((6, 2, 3), -100., np.float32))
    generated = {view: np.full((2, 2, 3), i + 10., np.float32) for i, view in enumerate(VIEWS)}
    result = service.Handler._artifact_arrays(arrays, SimpleNamespace(policy_rgb=generated),
        np.full(26, -99., np.float32), np.full(26, -98., np.float32), False, .75)
    np.testing.assert_array_equal(result["teacher_action"], arrays["teacher_action"])
    assert set(result) == {"teacher_action", "actor_obs", *(f"rgb_gen__{view}" for view in VIEWS)}
    for view in VIEWS:
        np.testing.assert_array_equal(result[f"rgb_gen__{view}"], generated[view])


@pytest.mark.parametrize("fault", ["temporal_order", "behavior_instead_of_label", "native_instead_of_generated"])
def test_exact_handoff_comparison_rejects_semantic_corruption(fault):
    vision = torch.arange(BATCH * 81 * 6 * 2 * 3).reshape(BATCH, 81, 6, 2, 3).float()
    actor = torch.arange(BATCH * 81).reshape(BATCH, 81, 1).float()
    target = torch.ones(BATCH, 81, 11)
    reference = dict(vision_obs=vision.clone(), actor_obs=actor.clone(), gt_actions=target.clone())
    if fault == "temporal_order":
        actor = actor.flip(1)
    elif fault == "behavior_instead_of_label":
        target = -target
    else:
        vision = torch.zeros_like(vision)
    with pytest.raises(AssertionError):
        assert_window_tensors((vision, actor, target), reference)


def test_numeric_tolerance_does_not_accept_wrong_update():
    with pytest.raises(AssertionError):
        assert_close_state({"weight": torch.zeros(3)}, {"weight": torch.full((3,), 1e-4)})


@pytest.mark.parametrize("fault", ["exp_avg", "exp_avg_sq", "step"])
def test_optimizer_checks_reject_zeroed_small_moments_and_changed_step(fault):
    parameter = torch.nn.Parameter(torch.ones(3))
    optimizer = torch.optim.AdamW([parameter], lr=.0002, weight_decay=0.)
    # Moments are far below the old1e-6 comparison floor and must NOT disappear.
    parameter.grad = torch.tensor([1e-5, -2e-5, 3e-5])
    before, gradients = before_adamw_step(optimizer)
    optimizer.step()
    good = deepcopy(optimizer.state_dict())
    assert_adamw_transition(before, gradients, good)
    corrupted = deepcopy(good)
    if fault == "step":
        corrupted["state"][0][fault] += 1
    else:
        assert torch.count_nonzero(corrupted["state"][0][fault]) == 3
        corrupted["state"][0][fault].zero_()
    with pytest.raises(AssertionError):
        assert_adamw_transition(before, gradients, corrupted)
    bounds = adamw_difference_bounds(before, gradients, before, gradients)
    with pytest.raises(AssertionError):
        assert_close_state(good, corrupted, moment_bounds=bounds)


def test_required_oracle_is_not_silently_skipped(tmp_path, monkeypatch):
    monkeypatch.setitem(pinned_tree.__globals__, "REFERENCE", tmp_path)
    with pytest.raises(AssertionError, match="required pinned Raven"):
        pinned_tree("trainer/ppo_trainer.py")


@pytest.mark.parametrize("fault", ["short", "out_of_order", "wrong_done", "checkpoint_change"])
def test_real_seal_rejects_incomplete_reordered_or_mixed_window(tmp_path, fault):
    tape = SelectionTape()
    collection = CurrentCollection(tmp_path, tiny_policy(), tape)
    # Capture a valid real seal invocation; do not publish it until mutated.
    calls = []
    collection.handler._seal_window_records = lambda *args: calls.append(deepcopy(args))
    collection.runner.run_window(0)
    key, value, rows, loaded = calls[0]
    if fault == "short":
        rows.pop(); loaded.pop()
    elif fault == "out_of_order":
        rows[1], rows[2] = rows[2], rows[1]
    elif fault == "wrong_done":
        value["dones"][0] = not value["dones"][0]
    else:
        rows[-1]["checkpoint_sha256"] = "9" * 64
    with pytest.raises(service.ConflictError):
        service.Handler._seal_window_records(collection.handler, key, value, rows, loaded)
    assert collection.paths == []
