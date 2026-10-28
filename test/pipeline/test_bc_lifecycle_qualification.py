"""CPU-only tests of the bounded real-BC qualification harness; no GPU launch."""
from __future__ import annotations

from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from pipeline.bc_training import qualify_raven_bc_lifecycle as smoke
from pipeline.bc_training import train_bc_h200 as bc


class TinyStudent(torch.nn.Module):
    is_recurrent = False

    def __init__(self):
        super().__init__()
        self.max_rollout_history = 1
        self.running_mean_std = None
        self.state_history_module = None
        self.use_temporal_attention = False
        self.clamp_noise_std = True
        self.max_noise_std = .001
        self.std = torch.nn.Parameter(torch.full((11,), .001), requires_grad=False)
        self.bn = torch.nn.SyncBatchNorm(3, momentum=.001)
        self.head = torch.nn.Linear(4, 11)
        self.obj_pred_mlp = torch.nn.Linear(3, 3)

    def init_rollout(self):
        self.obs_dict_buffer = {}

    def clear_rollout(self):
        self.obs_dict_buffer = {}

    def action_mean_with_obj_pred(self, observation):
        vision, actor = observation["vision_obs"], observation["actor_obs"]
        images = vision.reshape(-1, *vision.shape[-3:]).permute(0, 3, 1, 2)
        features = self.bn(images).mean((2, 3)).reshape(*vision.shape[:-3], 3)
        actions = self.head(torch.cat((features, actor[..., :1]), dim=-1))
        torch.distributions.Normal(actions, actions * 0 + self.std).sample()
        return {"action_mean": actions, "predicted_obj_pos": self.obj_pred_mlp(features)}

    def rollout_with_obj_pred(self, observation):
        return self.action_mean_with_obj_pred(observation)


def fixture_batches():
    return [{"vision": torch.full((16, 81, 6, 2, 3), 70 + i, dtype=torch.uint8),
             "actor": torch.ones(16, 81, 113), "target": torch.full((16, 81, 11), .2),
             "object_target": torch.full((16, 81, 3), .1)} for i in range(2)]


class BCLifecycleQualificationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_default_main_only_plans_without_cuda_or_output(self):
        self.assertFalse(smoke.parse_args([]).execute)
        with mock.patch.object(smoke, "plan", return_value={"execute": False}) as plan, \
                mock.patch.object(smoke, "execute", side_effect=AssertionError("execution forbidden")), \
                mock.patch.object(torch.cuda, "is_available", side_effect=AssertionError("CUDA forbidden")), \
                mock.patch("builtins.print"):
            smoke.main([])
        plan.assert_called_once()

    def test_plan_uses_exact_admitted_merged_prefix_without_decoding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bootstrap = root / "checkpoints/students/bootstrap/g1_3view_student_bootstrap_v1.pt"
            bootstrap.parent.mkdir(parents=True)
            bootstrap.write_bytes(b"bootstrap fixture")
            bootstrap.with_suffix(".policy_spec.json").write_text(json.dumps({
                "checkpoint_sha256": bc.sha256_file(bootstrap)}))
            for name in ("train.json", "retired.json", "admission.json", "registry.json"):
                (root / name).write_text("{}")
            paths, items = [], []
            for i in reversed(range(40)):
                path = root / f"chunk-{i:03d}.npz"
                path.write_bytes(b"not decoded in dry-run")
                paths.append(path)
                items.append({"chunk_id": f"chunk-{i:03d}", "rendered_npz": path.name})
            selection = {"DATASET_ROOT": str(root), "TRAIN_SPLIT": str(root / "train.json"),
                "VALIDATION_SPLIT": str(root / "retired.json"), "DATA_ADMISSION_JSON": str(root / "admission.json")}
            admitted = {"chunk_count": 2880, "original_chunk_counts": {"train": 2560, "validation": 320},
                "validation_enabled": False, "origin_roles": {row["chunk_id"]: "train" for row in items}}
            args = smoke.parse_args(["--asset-root", str(root), "--registry", str(root / "registry.json")])
            with mock.patch.object(smoke, "resolve_selection", return_value=selection), \
                    mock.patch.object(smoke, "validate_training_admission", return_value=admitted), \
                    mock.patch.object(bc, "load_split", return_value=({"items": items}, paths)), \
                    mock.patch.object(np, "load", side_effect=AssertionError("payload decode forbidden")), \
                    mock.patch.object(torch.cuda, "is_available", side_effect=AssertionError("CUDA forbidden")):
                plan = smoke.plan(args)
                self.assertEqual([row["path"] for row in plan["selected_chunks"]],
                                 [str(root / f"chunk-{i:03d}.npz") for i in range(32)])
                self.assertEqual(plan["optimizer_updates"], 2)
                self.assertFalse(plan["validation_enabled"])
                admitted["chunk_count"] = 2560
                with self.assertRaisesRegex(ValueError, "merged"):
                    smoke.plan(args)

    def test_output_must_be_new_isolated_and_smoke_named(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "registry.json"
            registry.write_text(json.dumps({"hosts": {"cosmos": {"roots": {"run_root": str(root / "runs")}}}}))
            args = SimpleNamespace(registry=registry, asset_root=root, output=root / "runs/smoke-bc")
            self.assertEqual(smoke.output_path(args), args.output)
            self.assertFalse(args.output.exists())
            for bad in (root / "runs/production", root / "checkpoints/smoke-bc", Path("relative")):
                args.output = bad
                with self.assertRaises(ValueError):
                    smoke.output_path(args)
            args.output = root / "runtime/smoke-existing"
            args.output.mkdir(parents=True)
            with self.assertRaises(ValueError):
                smoke.output_path(args)

    def test_cuda_guard_requires_one_h200_uuid_and_free_memory(self):
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(torch.cuda, "is_available", return_value=True))
            count = stack.enter_context(mock.patch.object(torch.cuda, "device_count", return_value=2))
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                smoke.cuda_identity(64)
            count.return_value = 1
            props = SimpleNamespace(name="H200", uuid="GPU-fixture")
            stack.enter_context(mock.patch.object(torch.cuda, "get_device_properties", return_value=props))
            stack.enter_context(mock.patch.object(torch.cuda, "get_device_capability", return_value=(9, 0)))
            stack.enter_context(mock.patch.object(torch.cuda, "mem_get_info", return_value=(63 << 30, 141 << 30)))
            with self.assertRaisesRegex(RuntimeError, "insufficient"):
                smoke.cuda_identity(64)
            props.uuid = None
            with self.assertRaisesRegex(RuntimeError, "UUID"):
                smoke.cuda_identity(64)
            with self.assertRaises(ValueError):
                smoke.cuda_identity(0)

    def test_two_real_cpu_updates_have_expected_bn_optimizer_and_weight_changes(self):
        torch.manual_seed(42)
        policy = TinyStudent()
        bc.configure_raven_offline_student(policy)
        optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=2e-4, weight_decay=0)
        progress = mock.Mock()
        result = smoke.run_updates(policy, optimizer, fixture_batches(), torch.device("cpu"),
                                   torch.zeros(1,1,1,1,3), torch.ones(1,1,1,1,3), progress=progress)
        self.assertEqual(len(result["updates"]), 2)
        self.assertEqual(result["batch_norm_counter_deltas"], {"bn": 164})
        self.assertTrue(result["changed_parameter_names"])
        self.assertEqual(progress.call_count, 2)
        self.assertTrue(all(int(state["step"]) == 2 for state in optimizer.state.values()))

    def test_incomplete_or_excess_batches_do_not_pass(self):
        for batches in (fixture_batches()[:1], fixture_batches() + fixture_batches()[:1]):
            policy = TinyStudent()
            optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=2e-4, weight_decay=0)
            with self.assertRaisesRegex(ValueError, "two"):
                smoke.run_updates(policy, optimizer, batches, torch.device("cpu"),
                                  torch.zeros(1,1,1,1,3), torch.ones(1,1,1,1,3))

    def test_nonfinite_labels_fail_before_an_optimizer_step(self):
        for key in ("target", "object_target"):
            policy = TinyStudent()
            optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=2e-4, weight_decay=0)
            batches = fixture_batches()
            batches[0][key][0, 0, 0] = float("nan")
            with self.assertRaisesRegex(ValueError, "finite"):
                smoke.run_updates(policy, optimizer, batches, torch.device("cpu"),
                                  torch.zeros(1,1,1,1,3), torch.ones(1,1,1,1,3))
            self.assertEqual(len(optimizer.state), 0)

    def test_diagnostic_checkpoint_reloads_model_optimizer_rng_and_rejects_production_resume(self):
        torch.manual_seed(71)
        policy = TinyStudent()
        optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=2e-4, weight_decay=0)
        smoke.run_updates(policy, optimizer, fixture_batches(), torch.device("cpu"),
                          torch.zeros(1,1,1,1,3), torch.ones(1,1,1,1,3))
        generator = torch.Generator().manual_seed(71)
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(torch.cuda, "get_rng_state_all", return_value=[]), \
                mock.patch.object(torch.cuda, "set_rng_state_all"), \
                mock.patch.object(bc, "build_policy", side_effect=lambda *_: (TinyStudent(), {})):
            path = Path(directory) / "diagnostic.pt"
            bc.save_checkpoint(path, policy, optimizer, 0, 2, None, None, {
                "schema": smoke.SCHEMA, "completed_epoch": False, "production_checkpoint": False,
                "training_protocol": {"schema": smoke.SCHEMA}, "validation_enabled": False}, generator)
            result = smoke.reload_checkpoint(path, policy, optimizer, generator, Path("spec"), Path("bootstrap"))
            self.assertTrue(all(result.values()))
            saved = torch.load(path, weights_only=False)
            self.assertIsNone(saved["validation_mse"])
            with self.assertRaisesRegex(ValueError, "exact fixed_final"):
                bc.validate_fixed_final_resume(saved, bc.fixed_final_contract(1, budget_explicit=True))
            changed = copy.deepcopy(saved["policy_state_dict"])
            changed["head.bias"].add_(1)
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                smoke.assert_equal(policy.state_dict(), changed)


if __name__ == "__main__":
    unittest.main()
