"""CPU-only fixed-final BC training/commit/resume regression tests."""
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from pipeline.bc_training import train_bc_h200 as trainer
from pipeline.bc_training.stopping_gate import fixed_final_contract, validate_fixed_final_resume


class FixedFinalBCTest(unittest.TestCase):
    def test_default_disables_validation_but_does_not_invent_budget(self):
        args = trainer.parse_args([])
        self.assertEqual(args.training_protocol, "fixed_final")
        self.assertEqual(args.student_lifecycle, trainer.REPLAY_STUDENT_LIFECYCLE)
        with self.assertRaisesRegex(ValueError, "explicit positive"):
            fixed_final_contract(args.max_epochs, budget_explicit=args.epoch_budget_explicit)
        args = trainer.parse_args(["--max-epochs=3"])
        contract = fixed_final_contract(args.max_epochs, budget_explicit=args.epoch_budget_explicit)
        self.assertEqual(contract["max_epochs"], 3)
        self.assertFalse(contract["validation_enabled"])
        self.assertEqual(contract["checkpoint_selection"], "last_completed_epoch")

    def test_disabled_validation_rejects_uncapped_and_fitting_gates(self):
        for budget in (None, 0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                fixed_final_contract(budget, budget_explicit=True)
        for target in ("epoch_target", "eval_target"):
            with self.assertRaisesRegex(ValueError, "disables validation"):
                fixed_final_contract(3, budget_explicit=True, **{target: .015})

    def test_missing_budget_fails_before_cuda_model_or_training(self):
        with mock.patch.object(trainer, "parse_args", return_value=trainer.parse_args([])), \
                mock.patch.object(trainer, "load_data_admission", return_value={"status": "fixture"}), \
                mock.patch.object(torch.cuda, "is_available") as cuda, \
                mock.patch.object(trainer, "run_fixed_final") as run:
            with self.assertRaisesRegex(ValueError, "explicit positive"):
                trainer.main()
            cuda.assert_not_called()
            run.assert_not_called()

    def test_legacy_mode_is_explicit(self):
        args = trainer.parse_args(["--training-protocol", "legacy_validation"])
        self.assertEqual(args.training_protocol, "legacy_validation")
        self.assertEqual(args.student_lifecycle, trainer.DIRECT_STUDENT_LIFECYCLE)
        self.assertEqual(args.max_epochs, 100)

    def _fixture(self, root, *, run_name="smoke-fixed", resume=None):
        root = Path(root)
        bootstrap = root / "bootstrap.pt"
        bootstrap.write_bytes(b"fixture-bootstrap")
        train = root / "train.json"
        train.write_text("{}")
        spec_file = root / "policy.json"
        spec_file.write_text("{}")
        args = trainer.parse_args(["--max-epochs", "2", "--workers", "0",
            "--chunk-batch-size", "16",
            "--student-lifecycle", trainer.DIRECT_STUDENT_LIFECYCLE,
            "--wandb-mode", "disabled", "--dataset-root", str(root),
            "--diagnostic-without-production-admission", "--data-admission-json", str(root / "absent-admission.json"),
            "--train-split", str(train), "--validation-split", str(root / "nonexistent-validation.json"),
            "--bootstrap", str(bootstrap), "--policy-spec", str(spec_file),
            "--output-root", str(root / "outputs"), "--run-name", run_name])
        args.resume_checkpoint = resume
        protocol = fixed_final_contract(2, budget_explicit=True)
        spec = {"input_contract": {"actor_obs_dim": 2, "rgb_mean": [0, 0, 0],
                    "rgb_std": [1, 1, 1]}, "output_contract": {"policy_action_dim": 1}}
        paths = [root / f"train_{i:03d}.npz" for i in range(16)]
        return args, protocol, spec, paths

    def _run(self, args, protocol, spec, paths, *, fail_second_update=False):
        calls = []
        policy = torch.nn.Linear(2, 1)
        with torch.no_grad():
            policy.weight.fill_(.25)
            policy.bias.fill_(.1)

        def split(path, expected, root):
            calls.append(expected)
            if expected != "train":
                raise AssertionError("validation split payload must never be loaded")
            return {"split": "train"}, paths

        updates = 0
        def backward(model, vision, actor, target):
            nonlocal updates
            updates += 1
            if fail_second_update and updates == 2:
                raise RuntimeError("injected second-epoch failure")
            self.assertEqual(actor.shape[:2], (16, 81))
            loss = (model(actor) - target).square().mean()
            loss.backward()
            return float(loss.detach())

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(trainer, "load_split", side_effect=split))
            stack.enter_context(mock.patch.object(trainer, "build_policy", return_value=(policy, spec)))
            dataset = stack.enter_context(mock.patch.object(trainer, "BCChunks", return_value=list(range(16))))
            stack.enter_context(mock.patch.object(trainer, "install_index_safe_max_pool2d", return_value=0))
            stack.enter_context(mock.patch.object(trainer, "prepare_batch", return_value=(
                torch.zeros(16, 81, 1), torch.ones(16, 81, 2), torch.ones(16, 81, 1))))
            stack.enter_context(mock.patch.object(trainer, "backward_raven_bc_mse", side_effect=backward))
            validation = stack.enter_context(mock.patch.object(trainer, "validate", side_effect=AssertionError("validation forbidden")))
            stack.enter_context(mock.patch.object(torch.cuda, "get_device_name", return_value="CPU fixture"))
            stack.enter_context(mock.patch.object(torch.cuda, "get_rng_state_all", return_value=[]))
            stack.enter_context(mock.patch.object(torch.cuda, "set_rng_state_all"))
            stack.enter_context(mock.patch("builtins.print"))
            trainer.run_fixed_final(args, {"status": "fixture"}, protocol, device=torch.device("cpu"))
            dataset.assert_called_once()
            validation.assert_not_called()
        self.assertEqual(calls, ["train"])
        return args.output_root / args.run_name

    def test_actual_cpu_loop_saves_last_epoch_optimizer_rng_and_no_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            args, protocol, spec, paths = self._fixture(directory)
            output = self._run(args, protocol, spec, paths)
            receipt = json.loads((output / "training-complete.json").read_text())
            self.assertEqual(receipt["status"], "READY_FOR_TEST")
            self.assertEqual((receipt["epoch"], receipt["global_step"]), (2, 2))
            self.assertFalse(receipt["test_execution_started"])
            self.assertFalse((output / "validation_state.json").exists())
            self.assertFalse((output / "best_validation.pt").exists())
            checkpoint = torch.load(output / "last.pt", weights_only=False)
            self.assertIsNone(checkpoint["validation_mse"])
            self.assertIsNone(checkpoint["validation_per_action_mse"])
            self.assertTrue(checkpoint["optimizer_state_dict"]["state"])
            self.assertEqual(set(checkpoint["rng_state"]), {"python", "numpy", "torch_cpu", "torch_cuda", "train_generator"})
            self.assertEqual(checkpoint["selection"], "last_completed_epoch")
            self.assertEqual(trainer.sha256_file(output / "last.pt"), receipt["selected_checkpoint_sha256"])

    def test_partial_epoch_failure_preserves_last_boundary_without_test_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            args, protocol, spec, paths = self._fixture(directory)
            with self.assertRaisesRegex(RuntimeError, "second-epoch"):
                self._run(args, protocol, spec, paths, fail_second_update=True)
            output = args.output_root / args.run_name
            self.assertFalse((output / "training-complete.json").exists())
            checkpoint = torch.load(output / "last.pt", weights_only=False)
            self.assertEqual((checkpoint["epoch"], checkpoint["global_step"]), (1, 1))

    def test_logging_flush_failure_does_not_publish_test_handoff(self):
        with tempfile.TemporaryDirectory() as directory:
            args, protocol, spec, paths = self._fixture(directory)
            args.wandb_mode = "offline"
            wandb = mock.MagicMock()
            run = wandb.init.return_value
            run.id, run.url, run.entity = "fixture-run", "https://invalid.example/fixture", "fixture-entity"
            run.finish.side_effect = RuntimeError("injected logging flush failure")
            with mock.patch.dict("sys.modules", {"wandb": wandb}):
                with self.assertRaisesRegex(RuntimeError, "logging flush"):
                    self._run(args, protocol, spec, paths)
            output = args.output_root / args.run_name
            self.assertFalse((output / "training-complete.json").exists())
            checkpoint = torch.load(output / "last.pt", weights_only=False)
            self.assertEqual((checkpoint["epoch"], checkpoint["global_step"]), (2, 2))

    def test_online_finalize_recovers_same_run_without_learner_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            args, protocol, spec, paths = self._fixture(directory)
            args.wandb_mode = "online"
            wandb = mock.MagicMock()
            run = wandb.init.return_value
            run.id, run.url, run.entity = "fixture-run", "https://invalid.example/fixture", "fixture-entity"
            run.summary = {}
            remote_summary = {}
            wandb.Api.return_value.run.return_value = SimpleNamespace(summary=remote_summary)
            run.finish.side_effect = RuntimeError("injected logging interruption")
            output = args.output_root / args.run_name
            with mock.patch.dict("sys.modules", {"wandb": wandb}):
                with self.assertRaisesRegex(RuntimeError, "logging interruption"):
                    self._run(args, protocol, spec, paths)
                intent = json.loads((output / "training-finalization-intent.json").read_text())
                checkpoint_sha = trainer.sha256_file(output / "last.pt")
                wandb.init.reset_mock()
                run.finish.side_effect = lambda: remote_summary.update(run.summary)
                with ExitStack() as stack:
                    for name in ("DataLoader", "build_policy", "backward_raven_bc_mse", "validate", "load_split"):
                        stack.enter_context(mock.patch.object(trainer, name, side_effect=AssertionError(name)))
                    stack.enter_context(mock.patch.object(torch, "load", side_effect=AssertionError("checkpoint unpickle")))
                    stack.enter_context(mock.patch.object(torch.cuda, "is_available", side_effect=AssertionError("CUDA")))
                    actual = trainer.publish_fixed_final_completion(output)
                    self.assertEqual(actual, intent)
                    wandb.init.assert_called_once_with(id="fixture-run", entity="fixture-entity",
                        project=args.wandb_project, resume="must", mode="online")
                    wandb.init.reset_mock()
                    self.assertEqual(trainer.publish_fixed_final_completion(output), intent)
                    wandb.init.assert_not_called()
                self.assertEqual(trainer.sha256_file(output / "last.pt"), checkpoint_sha)
                self.assertEqual(remote_summary["completion_operation_id"], intent["completion_operation_id"])

    def test_finalize_rejects_partial_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            args, protocol, spec, paths = self._fixture(directory)
            with self.assertRaisesRegex(RuntimeError, "second-epoch"):
                self._run(args, protocol, spec, paths, fail_second_update=True)
            output = args.output_root / args.run_name
            with self.assertRaisesRegex(ValueError, "fully completed"):
                trainer.publish_fixed_final_completion(output)
            self.assertFalse((output / "training-complete.json").exists())

    def test_finalize_rejects_changed_data_source_or_collision(self):
        for mutation in ("data", "source", "receipt"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                args, protocol, spec, paths = self._fixture(directory)
                output = self._run(args, protocol, spec, paths)
                if mutation == "data":
                    args.train_split.write_text("changed")
                elif mutation == "source":
                    state = json.loads((output / "training_state.json").read_text())
                    state["training_source_identity"] = {}
                    trainer.atomic_json(output / "training_state.json", state)
                else:
                    receipt = json.loads((output / "training-complete.json").read_text())
                    receipt["completion_operation_id"] = "different-operation"
                    trainer.atomic_json(output / "training-complete.json", receipt)
                with self.assertRaises(ValueError):
                    trainer.publish_fixed_final_completion(output)

    def test_finalize_never_invents_missing_finalization_intent(self):
        with tempfile.TemporaryDirectory() as directory:
            args, protocol, spec, paths = self._fixture(directory)
            output = self._run(args, protocol, spec, paths)
            (output / "training-finalization-intent.json").unlink()
            with self.assertRaisesRegex(ValueError, "quiescence"):
                trainer.publish_fixed_final_completion(output)

    def test_exact_epoch_resume_matches_uninterrupted_weights_and_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            args, protocol, spec, paths = self._fixture(directory)
            original = self._run(args, protocol, spec, paths)
            args2, protocol2, spec2, paths2 = self._fixture(directory, run_name="smoke-resume",
                resume=original / "epoch_checkpoints" / "epoch_0001_step_000001.pt")
            resumed = self._run(args2, protocol2, spec2, paths2)
            left = torch.load(original / "last.pt", weights_only=False)
            right = torch.load(resumed / "last.pt", weights_only=False)
            for key in left["policy_state_dict"]:
                self.assertTrue(torch.equal(left["policy_state_dict"][key], right["policy_state_dict"][key]))
            for param_id in left["optimizer_state_dict"]["state"]:
                for key in left["optimizer_state_dict"]["state"][param_id]:
                    self.assertTrue(torch.equal(left["optimizer_state_dict"]["state"][param_id][key],
                                                right["optimizer_state_dict"]["state"][param_id][key]))
            self.assertTrue(torch.equal(left["rng_state"]["train_generator"], right["rng_state"]["train_generator"]))

    def test_strict_resume_rejects_historical_changed_budget_or_missing_rng(self):
        contract = fixed_final_contract(3, budget_explicit=True)
        checkpoint = {"training_protocol": contract, "validation_enabled": False,
                      "epoch": 1, "global_step": 1, "rng_state": {}}
        validate_fixed_final_resume(checkpoint, contract)
        for change in ({"training_protocol": None}, {"validation_enabled": True},
                       {"rng_state": None}, {"epoch": 3}, {"global_step": True}):
            with self.assertRaises(ValueError):
                validate_fixed_final_resume({**checkpoint, **change}, contract)
        with self.assertRaisesRegex(ValueError, "exact fixed_final"):
            validate_fixed_final_resume(checkpoint, fixed_final_contract(4, budget_explicit=True))

    def test_resume_rejects_cross_student_lifecycle_before_restoring_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            args, protocol, spec, paths = self._fixture(directory)
            original = self._run(args, protocol, spec, paths)
            checkpoint = torch.load(original / "epoch_checkpoints" / "epoch_0001_step_000001.pt",
                                    weights_only=False)
            checkpoint["training_batch_contract"]["schema"] = trainer.RAVEN_REPLAY_BATCHING_CONTRACT
            changed = Path(directory) / "incompatible-replay.pt"
            torch.save(checkpoint, changed)
            args2, protocol2, spec2, paths2 = self._fixture(
                directory, run_name="smoke-incompatible", resume=changed)
            with self.assertRaisesRegex(ValueError, "training_batch_contract"):
                self._run(args2, protocol2, spec2, paths2)


if __name__ == "__main__":
    unittest.main()
