"""Record original offline BC semantics without launching Isaac or CUDA.

The ignored reference checkout must be Raven commit94d7985's exact source.
Absent reference files skip explicitly; changed bytes fail. The extracted
offline methods run with small CPU policy/storage/teacher doubles. This proves
the replay/control-flow difference and the versioned repair, not native ResNet
accuracy, GPU numerics, unknown historical optimizer versions, or policy quality.
In particular, a matching full-batch gradient does not imply matching historical
BatchNorm buffers: Raven's offline branch bypasses the normal rollout eval mode.
"""
from __future__ import annotations
from test.deployment.source_paths import REPO_ROOT

import ast
import contextlib
import copy
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from pipeline.bc_training import train_bc_h200 as trainer


ROOT = REPO_ROOT
REFERENCE = ROOT / "runtime/raven_gr00t_collection_94d7985/gr00t/rl/trl/trainer"
SOURCE_HASHES = {
    "ppo_trainer.py": "f56a1488af7cc007c81e07abd9b1adfc12ab83b5c541ea163381ef9af743ca61",
    "distill_trainer_videogen.py": "cc7f8aa3de2a9d60324b9294ba5db7ce15f339d59fa168ef1ccab7484d11890b",
}
VIEWS = ("head_color", "hand_left", "hand_right")


def _reference_class(filename, name):
    path = REFERENCE / filename
    if not path.is_file():
        raise unittest.SkipTest(f"pinned Raven source unavailable: {path}")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_HASHES[filename]:
        raise AssertionError(f"pinned Raven source changed: {path}")
    return next(node for node in ast.parse(raw).body
                if isinstance(node, ast.ClassDef) and node.name == name)


class _ParentPolicyDouble:
    def _rollout_step(self, model, observation):
        raise AssertionError("offline must bypass the parent rollout/eval-mode switch")

    def policy_step(self, policy, observation):
        # The real parent also computes teacher/Homie outputs. The offline method
        # must replace those recomputed labels with the paired dataset labels.
        result = policy(observation)
        result["gt_actions"] = torch.full((16, 26), -777.0)
        return result


def _original_offline_class(parent=_ParentPolicyDouble):
    cls = _reference_class("distill_trainer_videogen.py", "TRLDistillTrainerVideoGen")
    names = {"_rollout_step", "_offline_fill_storage", "_next_offline_batch"}
    methods = [node for node in cls.body
               if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in methods} == names
    extracted = ast.ClassDef(
        name="OriginalOffline", bases=[ast.Name(id="ParentDouble", ctx=ast.Load())],
        keywords=[], body=methods, decorator_list=[])
    namespace = {"torch": torch, "np": np, "ParentDouble": parent}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[extracted], type_ignores=[])),
                 "hash_pinned_original_offline_bc", "exec"), namespace)
    return namespace["OriginalOffline"]


def _current_checkpointed_forward():
    source = ROOT / "pipeline/dagger/services/raven_learner_worker.py"
    tree = ast.parse(source.read_text())
    names = {"_batch_norm_recompute_context", "checkpointed_full_window_actions"}
    nodes = [node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == names
    namespace = {"torch": torch, "contextlib": contextlib, "checkpoint": checkpoint}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(source), "exec"), namespace)
    return namespace["checkpointed_full_window_actions"]


class _TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.SyncBatchNorm(3, momentum=0.001)
        self.head = torch.nn.Linear(3, 11)
        self.forward_populations = []

    def init_rollout(self):
        pass

    def clear_rollout(self):
        pass

    def forward(self, observation):
        vision = observation["vision_obs"]
        image = vision.reshape(-1, *vision.shape[-3:]).permute(0, 3, 1, 2)
        self.forward_populations.append((self.training, image.shape[0]))
        features = self.bn(image).mean((2, 3))
        return {"actions": self.head(features).reshape(*vision.shape[:-3], 11)}


class _ReplayTinyBase(torch.nn.Module):
    """Small feature network; rollout/distribution code comes from pinned Raven."""
    is_recurrent = False
    max_rollout_history = 1
    running_mean_std = None
    state_history_module = None
    use_temporal_attention = False
    clamp_noise_std = True
    max_noise_std = .001

    def __init__(self):
        super().__init__()
        self.bn = torch.nn.SyncBatchNorm(3, momentum=.001)
        self.head = torch.nn.Linear(4, 11)
        self.obj_pred_mlp = torch.nn.Linear(3, 3)
        self.std = torch.nn.Parameter(torch.full((11,), .001), requires_grad=False)
        self.forward_populations = []

    def forward(self, obs_dict, **_):
        vision, actor = obs_dict["vision_obs"], obs_dict["actor_obs"]
        images = vision.reshape(-1, *vision.shape[-3:]).permute(0, 3, 1, 2)
        self.forward_populations.append((self.training, torch.is_grad_enabled(), images.shape[0]))
        features = self.bn(images).mean((2, 3)).reshape(*vision.shape[:-3], 3)
        return {"actions": self.head(torch.cat((features, actor[..., :1]), dim=-1)),
                "obj_pred": self.obj_pred_mlp(features)}


def _original_student_class():
    modules = REFERENCE.parent / "modules"
    files = {
        "vision_actor_critic_modules.py": (
            "dfda777a937f1f8253e0eb8979045784da9731dfea51c9fbf4fea1ed39e00733",
            "VisionActor", {"_update_obs_buffer", "init_rollout", "clear_rollout", "action_mean", "action_std"}),
        "vision_actor_critic_modules_obj_pred.py": (
            "86acfbd0cf610aec49e3a11e7049b18d4ca83d3bf28bcb63fc3484683c1cb239",
            "VisionActorObjPred", {"rollout_with_obj_pred", "action_mean_with_obj_pred"}),
    }
    methods = []
    for filename, (expected_hash, class_name, names) in files.items():
        path = modules / filename
        if not path.is_file():
            raise unittest.SkipTest(f"pinned Raven source unavailable: {path}")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_hash:
            raise AssertionError(f"pinned Raven source changed: {path}")
        cls = next(node for node in ast.parse(raw).body
                   if isinstance(node, ast.ClassDef) and node.name == class_name)
        selected = [node for node in cls.body
                    if isinstance(node, ast.FunctionDef) and node.name in names]
        assert {node.name for node in selected} == names
        methods.extend(selected)
    cls = ast.ClassDef(name="ReferenceStudent", bases=[ast.Name(id="Base", ctx=ast.Load())],
                      keywords=[], body=methods, decorator_list=[])
    namespace = {"torch": torch, "Normal": torch.distributions.Normal, "Base": _ReplayTinyBase}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[])),
                 "hash_pinned_raven_student_methods", "exec"), namespace)
    return namespace["ReferenceStudent"]


class _Storage:
    def clear(self):
        self.rows = []
        self.row = {}

    def update_key(self, key, value):
        self.row[key] = value.clone()

    def increment_step(self):
        self.rows.append(self.row)
        self.row = {}


def _fixture():
    rng = np.random.default_rng(71)
    chunks = []
    for lane in range(16):
        sequence = np.arange(81, dtype=np.float32) + lane * 1000
        chunks.append({
            **{f"rgb_gen__{view}": rng.integers(0, 256, (81, 2, 2, 3), dtype=np.uint8)
               for view in VIEWS},
            "obs_actor_obs": np.repeat(sequence[:, None], 113, axis=1),
            "gt_actions": np.repeat((sequence + 50000)[:, None], 26, axis=1),
            "dones": np.zeros(81, dtype=bool),
        })
    runner = _original_offline_class()()
    runner.vg_mode = "offline"
    runner.accelerator = SimpleNamespace(device=torch.device("cpu"))
    runner.env = SimpleNamespace(num_envs=16)
    runner.storage = _Storage()
    runner._next_offline_batch = lambda batch: chunks
    runner.num_steps_per_env = 81
    runner.camera_resolution = (6, 2)
    runner._policy_views = VIEWS
    runner._normalize_like_env = lambda value: value
    runner.learn_normalized_actions = False
    runner.ep_infos = []
    vision = torch.from_numpy(np.stack([
        np.concatenate([chunk[f"rgb_gen__{view}"] for view in VIEWS], axis=1)
        for chunk in chunks])).float() / 255
    return runner, chunks, vision


class RavenOfflineSemanticsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_versioned_defaults_do_not_change_legacy_or_allow_shuffle(self):
        self.assertEqual(trainer.parse_args([]).student_lifecycle,
                         trainer.REPLAY_STUDENT_LIFECYCLE)
        legacy = trainer.parse_args(["--training-protocol", "legacy_validation"])
        self.assertEqual(legacy.student_lifecycle, trainer.DIRECT_STUDENT_LIFECYCLE)
        self.assertEqual(trainer.RAVEN_BATCHING_CONTRACT,
                         "raven_full_bxt_16x81_activation_checkpoint_v1")
        for flags in (["--experimental-shuffle"],
                      ["--training-protocol", "legacy_validation", "--student-lifecycle",
                       trainer.REPLAY_STUDENT_LIFECYCLE]):
            with self.assertRaises(ValueError):
                trainer.parse_args(flags)

    def test_replay_rejects_unqualified_stateful_or_stochastic_policy(self):
        for field, value in (("is_recurrent", True), ("max_rollout_history", 2),
                             ("running_mean_std", True),
                             ("max_noise_std", .01)):
            with self.subTest(field=field):
                policy = _original_student_class()()
                setattr(policy, field, value)
                with self.assertRaises(ValueError):
                    trainer.configure_raven_offline_student(policy)
        policy = _original_student_class()()
        policy.extra_dropout = torch.nn.Dropout(.1)
        with self.assertRaisesRegex(ValueError, "dropout"):
            trainer.configure_raven_offline_student(policy)

    def test_replay_dataset_requires_real_finite_object_observation_only_in_new_mode(self):
        class Archive(dict):
            @property
            def files(self):
                return list(self)
            def __enter__(self):
                return self
            def __exit__(self, *_):
                pass
        data = Archive({f"rgb_gen__{view}": np.zeros((81, 108, 192, 3), dtype=np.uint8)
                        for view in VIEWS})
        data.update(obs_actor_obs=np.zeros((81, 113), dtype=np.float32),
                    gt_actions=np.zeros((81, 26), dtype=np.float32))
        path = Path("/fixture/rendered.npz")
        legacy = trainer.BCChunks([path], 113, 11)
        replay = trainer.BCChunks([path], 113, 11, include_object_target=True)
        with mock.patch.object(np, "load", return_value=data):
            self.assertNotIn("object_target", legacy[0])
            with self.assertRaisesRegex(ValueError, "requires obs_gt_obj_pos"):
                replay[0]
            data["obs_gt_obj_pos"] = np.arange(81 * 9, dtype=np.float32).reshape(81, 9)
            torch.testing.assert_close(replay[0]["object_target"],
                                       torch.from_numpy(data["obs_gt_obj_pos"][:, :3]))
            data["obs_gt_obj_pos"] = np.zeros((81, 3), dtype=np.float32)
            self.assertEqual(replay[0]["object_target"].shape, (81, 3))
            data["obs_gt_obj_pos"] = np.zeros((81, 2), dtype=np.float32)
            with self.assertRaisesRegex(ValueError, "invalid Raven"):
                replay[0]
            data["obs_gt_obj_pos"] = np.full((81, 9), np.nan, dtype=np.float32)
            with self.assertRaisesRegex(ValueError, "invalid Raven"):
                replay[0]

    def test_replay_requires_train_mode_and_clears_buffer_on_failed_replay(self):
        policy = _original_student_class()()
        trainer.configure_raven_offline_student(policy)
        tensors = (torch.zeros(16, 81, 6, 2, 3), torch.zeros(16, 81, 113),
                   torch.zeros(16, 81, 11), torch.zeros(16, 81, 3))
        policy.eval()
        with self.assertRaisesRegex(ValueError, "train mode"):
            trainer.backward_raven_offline_bc_mse(policy, *tensors)
        policy.train()
        with mock.patch.object(policy, "rollout_with_obj_pred", side_effect=RuntimeError("injected")):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                trainer.backward_raven_offline_bc_mse(policy, *tensors)
        self.assertEqual(policy.obs_dict_buffer, {})

    def test_replay_matches_pinned_student_bn_rng_gradients_and_adamw_for_two_updates(self):
        """Same explicit torch AdamW, not certification of unknown Raven install."""
        torch.manual_seed(20261025)
        reference = _original_student_class()().train()
        current = copy.deepcopy(reference)
        for parameter in current.obj_pred_mlp.parameters():
            parameter.requires_grad_(False)
        trainer.configure_raven_offline_student(current)
        self.assertTrue(all(parameter.requires_grad for parameter in current.obj_pred_mlp.parameters()))
        reference_optimizer = torch.optim.AdamW(
            [p for p in reference.parameters() if p.requires_grad], lr=2e-4, weight_decay=0)
        current_optimizer = torch.optim.AdamW(
            [p for p in current.parameters() if p.requires_grad], lr=2e-4, weight_decay=0)
        fixture, chunks, vision = _fixture()
        for lane, chunk in enumerate(chunks):
            chunk["obs_actor_obs"] /= 5000
            chunk["gt_actions"] /= 50000
            chunk["obs_gt_obj_pos"] = np.full((81, 9), lane / 16, dtype=np.float32)
        actor = torch.from_numpy(np.stack([chunk["obs_actor_obs"] for chunk in chunks]))
        target = torch.from_numpy(np.stack([chunk["gt_actions"][:, :11] for chunk in chunks]))
        object_target = torch.from_numpy(np.stack([chunk["obs_gt_obj_pos"][:, :3] for chunk in chunks]))

        class ActualStudentParent(_ParentPolicyDouble):
            def policy_step(self, policy, observation):
                result = policy.rollout_with_obj_pred(observation)
                result["gt_actions"] = torch.full((16, 26), -777.0)
                return result

        runner = _original_offline_class(ActualStudentParent)()
        runner.__dict__.update(fixture.__dict__)
        initial_objects = copy.deepcopy(current.obj_pred_mlp.state_dict())
        reference_rng = torch.get_rng_state().clone()
        current_rng = reference_rng.clone()
        real_randperm = torch.randperm
        for update in range(2):
            reference_optimizer.zero_grad(set_to_none=True)
            torch.set_rng_state(reference_rng)
            runner._rollout_step(SimpleNamespace(policy=reference), {})
            expected_lanes = real_randperm(16, device=vision.device)
            result = reference.action_mean_with_obj_pred({
                "vision_obs": vision[expected_lanes], "actor_obs": actor[expected_lanes]})
            expected_loss = torch.nn.functional.mse_loss(result["action_mean"], target[expected_lanes])
            auxiliary = torch.nn.functional.mse_loss(result["predicted_obj_pos"], object_target[expected_lanes])
            (expected_loss + 0 * auxiliary).backward()
            reference_rng = torch.get_rng_state().clone()
            reference_grads = {name: p.grad.clone() if p.grad is not None else None
                               for name, p in reference.named_parameters()}
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 1, error_if_nonfinite=True)
            reference_optimizer.step()

            current_optimizer.zero_grad(set_to_none=True)
            torch.set_rng_state(current_rng)
            observed_lanes = []
            def record_permutation(*args, **kwargs):
                result = real_randperm(*args, **kwargs)
                observed_lanes.append(result.clone())
                return result
            with mock.patch.object(torch, "randperm", side_effect=record_permutation):
                loss = trainer.backward_raven_offline_bc_mse(current, vision, actor, target, object_target)
            current_rng = torch.get_rng_state().clone()
            self.assertEqual(loss, float(expected_loss.detach()))
            self.assertEqual(len(observed_lanes), 1)
            self.assertTrue(torch.equal(observed_lanes[0], expected_lanes))
            self.assertFalse(torch.equal(expected_lanes, torch.arange(16)))
            self.assertTrue(torch.equal(current_rng, reference_rng))
            for name, parameter in current.named_parameters():
                if reference_grads[name] is None:
                    self.assertIsNone(parameter.grad)
                else:
                    torch.testing.assert_close(parameter.grad, reference_grads[name], rtol=0, atol=0)
            torch.nn.utils.clip_grad_norm_(current.parameters(), 1, error_if_nonfinite=True)
            current_optimizer.step()
            for name, tensor in reference.state_dict().items():
                torch.testing.assert_close(current.state_dict()[name], tensor, rtol=0, atol=0)
            for left, right in zip(reference_optimizer.state.values(), current_optimizer.state.values(), strict=True):
                self.assertEqual(left.keys(), right.keys())
                for name in left:
                    torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)
            self.assertEqual(int(current.bn.num_batches_tracked), (update + 1) * 82)
            self.assertEqual(current.obs_dict_buffer, {})
            self.assertTrue(current.training)
            for parameter in current.obj_pred_mlp.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertEqual(torch.count_nonzero(parameter.grad).item(), 0)
                self.assertEqual(current_optimizer.state[parameter]["step"].item(), update + 1)
            self.assertNotIn(current.std, current_optimizer.state)
        for name, value in current.obj_pred_mlp.state_dict().items():
            self.assertTrue(torch.equal(value, initial_objects[name]))
        # Recompute executes only the full window, never the no-grad replay.
        self.assertEqual(sum(not grad and population == 16
                             for _, grad, population in current.forward_populations), 162)

    def test_new_fixed_final_loop_replay_checkpoint_and_resume_match(self):
        self._fixed_final_loop_replay_checkpoint_and_resume_match(16)

    def test_nondivisible_fixed_final_loop_replay_checkpoint_and_resume_match(self):
        self._fixed_final_loop_replay_checkpoint_and_resume_match(18)

    def _fixed_final_loop_replay_checkpoint_and_resume_match(self, count):
        """Exercise dispatch, metadata, optimizer construction and saved RNG."""
        student = _original_student_class()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename in ("bootstrap.pt", "policy.json", "train.json"):
                (root / filename).write_text("fixture")
            spec = {"input_contract": {"actor_obs_dim": 113, "rgb_mean": [0, 0, 0],
                        "rgb_std": [1, 1, 1]}, "output_contract": {"policy_action_dim": 11}}
            paths = [root / f"chunk_{lane:03d}.npz" for lane in range(count)]
            chunks = [{"vision": torch.full((81, 6, 2, 3), lane + 40, dtype=torch.uint8),
                       "actor": torch.full((81, 113), lane / 16),
                       "target": torch.full((81, 11), lane / 32),
                       "object_target": torch.full((81, 3), lane / 64)} for lane in range(count)]

            def run(name, resume=None):
                args = trainer.parse_args(["--max-epochs", "2", "--workers", "0",
                    "--chunk-batch-size", "16",
                    "--wandb-mode", "disabled", "--dataset-root", str(root),
                    "--diagnostic-without-production-admission",
                    "--train-split", str(root / "train.json"),
                    "--validation-split", str(root / "missing-validation.json"),
                    "--data-admission-json", str(root / "missing-admission.json"),
                    "--bootstrap", str(root / "bootstrap.pt"),
                    "--policy-spec", str(root / "policy.json"),
                    "--output-root", str(root / "outputs"), "--run-name", name])
                args.resume_checkpoint = resume
                torch.manual_seed(123)
                policy = student()
                for parameter in policy.obj_pred_mlp.parameters():
                    parameter.requires_grad_(False)
                with contextlib.ExitStack() as stack:
                    stack.enter_context(mock.patch.object(trainer, "load_split", return_value=({}, paths)))
                    stack.enter_context(mock.patch.object(trainer, "build_policy", return_value=(policy, spec)))
                    dataset = stack.enter_context(mock.patch.object(trainer, "BCChunks", return_value=chunks))
                    stack.enter_context(mock.patch.object(trainer, "validate", side_effect=AssertionError("no validation")))
                    stack.enter_context(mock.patch.object(trainer, "backward_raven_bc_mse", side_effect=AssertionError("no direct update")))
                    stack.enter_context(mock.patch.object(torch.cuda, "get_device_name", return_value="CPU fixture"))
                    stack.enter_context(mock.patch.object(torch.cuda, "get_rng_state_all", return_value=[]))
                    stack.enter_context(mock.patch.object(torch.cuda, "set_rng_state_all"))
                    stack.enter_context(mock.patch("builtins.print"))
                    trainer.run_fixed_final(args, {"status": "fixture"},
                        trainer.fixed_final_contract(2, budget_explicit=True), device=torch.device("cpu"))
                    dataset.assert_called_once_with(paths, 113, 11, include_object_target=True)
                self.assertFalse(policy.training)
                self.assertTrue(all(not p.requires_grad for p in policy.parameters()))
                return root / "outputs" / name

            uninterrupted = run("smoke-replay")
            steps = (count + 15) // 16
            resumed = run("smoke-replay-resume", uninterrupted / "epoch_checkpoints" / f"epoch_0001_step_{steps:06d}.pt")
            left = torch.load(uninterrupted / "last.pt", weights_only=False)
            right = torch.load(resumed / "last.pt", weights_only=False)
            self.assertEqual(left["training_batch_contract"]["schema"], trainer.RAVEN_REPLAY_BATCHING_CONTRACT)
            self.assertEqual(left["training_batch_contract"]["student_lifecycle"], trainer.REPLAY_STUDENT_LIFECYCLE)
            self.assertIn("obj_pred_mlp.weight", left["optimizer_contract"]["trainable_parameters"])
            self.assertNotIn("std", left["optimizer_contract"]["trainable_parameters"])
            self.assertEqual(int(left["policy_state_dict"]["bn.num_batches_tracked"]), 164 * steps)
            self.assertEqual(left['training_cursor'], trainer.cyclic_sampling.cursor_state(count,16,2*steps))
            self.assertEqual(left['training_cursor'], right['training_cursor'])
            for name, value in left["policy_state_dict"].items():
                self.assertTrue(torch.equal(value, right["policy_state_dict"][name]))
            for parameter, state in left["optimizer_state_dict"]["state"].items():
                for key, value in state.items():
                    self.assertTrue(torch.equal(value, right["optimizer_state_dict"]["state"][parameter][key]))
            for key in ("torch_cpu", "train_generator"):
                self.assertTrue(torch.equal(left["rng_state"][key], right["rng_state"][key]))

    def test_original_train_loop_enters_offline_replay_in_training_mode(self):
        cls = _reference_class("ppo_trainer.py", "TRLPPOTrainer")
        train = next(node for node in cls.body
                     if isinstance(node, ast.FunctionDef) and node.name == "train")
        source = ast.unparse(train)
        self.assertLess(source.index("model.train()"), source.index("self._rollout_step("))
        self.assertNotIn(".eval()", source[:source.index("self._rollout_step(")])
        # Upstream TRL0.17/0.19 unwrap_model_for_generation does not change
        # train/eval mode. The reference declares both versions in different
        # dependency files; its actual installed version is not certified here.
        # This CPU test performs no external library/source fetch.

    def test_original_stores_paired_rows_and_overwrites_recomputed_teacher_labels(self):
        runner, chunks, vision = _fixture()
        policy = _TinyPolicy().train()
        runner._rollout_step(SimpleNamespace(policy=policy), {})
        self.assertEqual(len(runner.storage.rows), 81)
        for step, row in enumerate(runner.storage.rows):
            torch.testing.assert_close(row["vision_obs"], vision[:, step])
            torch.testing.assert_close(row["actor_obs"], torch.from_numpy(np.stack([
                chunk["obs_actor_obs"][step] for chunk in chunks])))
            torch.testing.assert_close(row["gt_actions"], torch.from_numpy(np.stack([
                chunk["gt_actions"][step] for chunk in chunks])))
        self.assertEqual(policy.forward_populations, [(True, 16)] * 81)

    def test_original_replay_changes_bn_buffers_but_not_full_batch_training_gradient(self):
        torch.manual_seed(71)
        runner, _, vision = _fixture()
        original = _TinyPolicy().train()
        current = copy.deepcopy(original)
        runner._rollout_step(SimpleNamespace(policy=original), {})
        self.assertEqual(int(original.bn.num_batches_tracked), 81)
        actor = torch.zeros(16, 81, 113)
        target = torch.zeros(16, 81, 11)
        old_actions = original({"vision_obs": vision, "actor_obs": actor})["actions"]
        torch.nn.functional.mse_loss(old_actions, target).backward()
        new_actions = _current_checkpointed_forward()(current, vision, actor)
        torch.nn.functional.mse_loss(new_actions, target).backward()
        self.assertEqual(int(original.bn.num_batches_tracked), 82)
        self.assertEqual(int(current.bn.num_batches_tracked), 1)
        self.assertTrue(all(population == 16 * 81
                            for _, population in current.forward_populations))
        torch.testing.assert_close(old_actions, new_actions)
        for old, new in zip(original.parameters(), current.parameters(), strict=True):
            torch.testing.assert_close(old.grad, new.grad)
        self.assertGreater(float((original.bn.running_mean - current.bn.running_mean).abs().max()), .01)
        self.assertGreater(float((original.bn.running_var - current.bn.running_var).abs().max()), .01)
        original.eval()
        current.eval()
        with torch.no_grad():
            old_eval = original({"vision_obs": vision})["actions"]
            new_eval = current({"vision_obs": vision})["actions"]
        self.assertGreater(float((old_eval - new_eval).abs().max()), .001)

    def test_original_cursor_wraps_inside_batch_for_nondivisible_population(self):
        runner = _original_offline_class()()
        runner._offline_files = list(range(18))
        runner._offline_cursor = 0

        class Archive(dict):
            @property
            def files(self):
                return list(self)

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

        def load(index):
            return Archive(rgb_gen=np.asarray(index))

        with mock.patch.object(np, "load", side_effect=load):
            first = runner._next_offline_batch(16)
            second = runner._next_offline_batch(16)
        self.assertEqual([int(row["rgb_gen"]) for row in first], list(range(16)))
        self.assertEqual([int(row["rgb_gen"]) for row in second], [16, 17, *range(14)])
        self.assertEqual(runner._offline_cursor, 32)


if __name__ == "__main__":
    unittest.main()
