"""BC selection and resume regressions; all tests run on CPU."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from bc_training.train_bc_h200 import (
    EarlyStopping, load_resume_state, load_split, save_checkpoint,
    validate, validate_split_pair, load_data_admission, parse_args, training_sampling_contract,
    DIRECT_STUDENT_LIFECYCLE, REPLAY_STUDENT_LIFECYCLE,
)


class BCValidationProtocolTest(unittest.TestCase):
    def test_sorted_cyclic_default_and_explicit_experimental_shuffle(self):
        with mock.patch("sys.argv", ["train"]):
            defaults = parse_args()
            self.assertFalse(defaults.experimental_shuffle)
            self.assertEqual(defaults.student_lifecycle, REPLAY_STUDENT_LIFECYCLE)
        with mock.patch("sys.argv", ["train", "--training-protocol", "legacy_validation",
                "--student-lifecycle", DIRECT_STUDENT_LIFECYCLE, "--experimental-shuffle"]):
            legacy = parse_args()
            self.assertTrue(legacy.experimental_shuffle)
            self.assertEqual(legacy.student_lifecycle, DIRECT_STUDENT_LIFECYCLE)
        paths = [Path("/pool/chunk_001.npz"), Path("/pool/chunk_002.npz")]
        contract = training_sampling_contract(paths, Path("/pool"), shuffle=False)
        self.assertEqual(contract["mode"], "raven_sorted_cyclic")
        self.assertEqual(contract["epoch_boundary_cursor"], 0)
        with self.assertRaisesRegex(ValueError, "sorted"):
            training_sampling_contract(paths[::-1], Path("/pool"), shuffle=False)
        data = list(range(32))
        generator = torch.Generator().manual_seed(99)
        loader = torch.utils.data.DataLoader(data, batch_size=16, shuffle=False, generator=generator)
        first = [batch.tolist() for batch in loader]
        saved = generator.get_state()
        uninterrupted = [batch.tolist() for batch in loader]
        resumed = torch.Generator().set_state(saved)
        restored = [batch.tolist() for batch in torch.utils.data.DataLoader(
            data, batch_size=16, shuffle=False, generator=resumed)]
        self.assertEqual(first, [list(range(16)), list(range(16, 32))])
        self.assertEqual(uninterrupted, restored)

    def test_raven_replay_rejects_experimental_shuffle_before_cuda(self):
        for explicit in ([], ["--student-lifecycle", REPLAY_STUDENT_LIFECYCLE]):
            with self.subTest(explicit=explicit), mock.patch.object(torch.cuda, "is_available") as cuda:
                with self.assertRaisesRegex(ValueError, "sorted/cyclic"):
                    parse_args(["--experimental-shuffle", *explicit])
                cuda.assert_not_called()

    def test_production_requires_admission_before_cuda_or_model_access(self):
        from bc_training import train_bc_h200 as trainer
        with mock.patch("sys.argv", ["train", "--require-production-data-admission"]):
            with mock.patch.object(torch.cuda, "is_available") as cuda:
                with self.assertRaisesRegex(ValueError, "requires --data-admission-json"):
                    trainer.main()
                cuda.assert_not_called()

    def test_historical_optout_is_explicit(self):
        with mock.patch("sys.argv", ["train", "--diagnostic-without-production-admission"]):
            with self.assertRaisesRegex(ValueError, "smoke-"):
                load_data_admission(parse_args())
        with mock.patch("sys.argv", ["train", "--diagnostic-without-production-admission",
                "--run-name", "smoke-historical", "--wandb-run-name", "smoke-historical"]):
            report = load_data_admission(parse_args())
        self.assertEqual(report, {"status": "historical_unqualified",
                                  "production_admitted": False})

    def test_small_new_minimum_is_selected_without_resetting_patience(self):
        stop = EarlyStopping(min_epochs=1, patience=2, relative_improvement=.01)
        stop.update(0, 1.0)
        meaningful, saturated = stop.update(1, .995)
        self.assertFalse(meaningful)
        self.assertFalse(saturated)
        self.assertEqual((stop.best_mse, stop.best_epoch, stop.bad_validations), (.995, 1, 1))
        meaningful, saturated = stop.update(2, .994)
        self.assertFalse(meaningful)
        self.assertTrue(saturated)
        self.assertEqual((stop.best_mse, stop.best_epoch), (.994, 2))

    def test_cumulative_small_improvements_reset_patience(self):
        stop = EarlyStopping(min_epochs=1, patience=3, relative_improvement=.01)
        stop.update(0, 1.0)
        stop.update(1, .995)
        self.assertEqual(stop.update(2, .989), (True, False))
        self.assertEqual(stop.bad_validations, 0)
        self.assertEqual(stop.patience_best_mse, .989)

    def test_nonfinite_negative_or_duplicate_validation_is_rejected(self):
        for invalid in (math.nan, math.inf, -math.inf, -.1):
            with self.assertRaises(ValueError):
                EarlyStopping().update(0, invalid)
        stop = EarlyStopping()
        stop.update(0, .1)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            stop.update(0, .09)
        for invalid in (math.nan, math.inf, -.1):
            with self.assertRaises(ValueError):
                EarlyStopping(relative_improvement=invalid)

    def _resume_fixture(self, root: Path):
        policy = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=2e-4, weight_decay=0)
        generator = torch.Generator().manual_seed(7)
        stop = EarlyStopping(min_epochs=1, patience=2, relative_improvement=.01)
        history = []
        for epoch, mse in enumerate((1.0, .995, .994)):
            stop.update(epoch, mse)
            history.append({"epoch": epoch, "global_step": epoch * 4, "mse": mse})
        path = root / "checkpoint.pt"
        metadata = {
            "policy_spec": {"policy": "cpu-test"}, "bootstrap_sha256": "bootstrap",
            "train_split_sha256": "train", "validation_split_sha256": "validation",
            "early_stopping_state": stop.state_dict(), "validation_history": history,
        }
        save_checkpoint(path, policy, optimizer, 2, 8, .994, [.994], metadata, generator)
        return path, metadata, policy

    def _restore(self, path, metadata, *, stopper=None, train_sha="train", sampling=None):
        policy = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=2e-4, weight_decay=0)
        stop = stopper or EarlyStopping(min_epochs=1, patience=2, relative_improvement=.01)
        result = load_resume_state(
            path, path.parent / "absent-validation-state.json", policy=policy,
            optimizer=optimizer, train_generator=torch.Generator(),
            spec=metadata["policy_spec"], bootstrap_sha256="bootstrap",
            train_split_sha256=train_sha, validation_split_sha256="validation",
            stopper=stop, seed=7, expected_learning_rate=2e-4,
            expected_sampling_contract=sampling,
        )
        return result, stop, policy

    def test_resume_rejects_missing_or_changed_sampling_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path, metadata, _ = self._resume_fixture(Path(directory))
            sampling = training_sampling_contract([Path("/pool/a.npz")], Path("/pool"), shuffle=False)
            with self.assertRaisesRegex(ValueError, "sampling contract"):
                self._restore(path, metadata, sampling=sampling)
            checkpoint = torch.load(path, weights_only=False)
            checkpoint["training_sampling_contract"] = sampling
            torch.save(checkpoint, path)
            self._restore(path, metadata, sampling=sampling)
            with self.assertRaisesRegex(ValueError, "sampling contract"):
                self._restore(path, metadata, sampling={**sampling, "mode": "experimental_shuffle"})

    def test_resume_uses_immutable_checkpoint_state_without_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            path, metadata, original = self._resume_fixture(Path(directory))
            (_, history, rng), stop, restored = self._restore(path, metadata)
            self.assertEqual(stop.state_dict(), metadata["early_stopping_state"])
            self.assertTrue(stop.is_saturated(2))
            self.assertEqual(stop.best_epoch, 2)
            self.assertEqual(len(history), 3)
            self.assertEqual(rng, "exact_checkpoint_restore")
            for old, new in zip(original.parameters(), restored.parameters(), strict=True):
                self.assertTrue(torch.equal(old, new))

    def test_resume_rejects_changed_patience_or_split(self):
        with tempfile.TemporaryDirectory() as directory:
            path, metadata, _ = self._resume_fixture(Path(directory))
            with self.assertRaisesRegex(ValueError, "early-stopping state/config"):
                self._restore(path, metadata, stopper=EarlyStopping(
                    min_epochs=1, patience=4, relative_improvement=.01))
            with self.assertRaisesRegex(ValueError, "train_split_sha256"):
                self._restore(path, metadata, train_sha="another-training-split")

    def test_resume_rejects_duplicate_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path, metadata, _ = self._resume_fixture(Path(directory))
            checkpoint = torch.load(path, weights_only=False)
            checkpoint["validation_history"].insert(1, checkpoint["validation_history"][0])
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "global steps"):
                self._restore(path, metadata)

    def test_split_pair_requires_same_provenance_and_disjoint_examples(self):
        train = {"source_manifest_sha256": "a", "items": [
            {"chunk_id": "one", "stream_id": "stream-one"}]}
        validation = {"source_manifest_sha256": "a", "items": [
            {"chunk_id": "two", "stream_id": "stream-two"}]}
        validate_split_pair(train, [Path("one")], validation, [Path("two")])
        with self.assertRaisesRegex(ValueError, "rendered paths overlap"):
            validate_split_pair(train, [Path("one")], validation, [Path("one")])
        with self.assertRaisesRegex(ValueError, "source manifest identities"):
            validate_split_pair(train, [Path("one")],
                                {**validation, "source_manifest_sha256": "b"}, [Path("two")])
        with self.assertRaisesRegex(ValueError, "chunk_id identities overlap"):
            validate_split_pair(train, [Path("one")], train, [Path("two")])

    def test_load_split_rejects_duplicate_chunk_and_outside_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / "train.json"
            row = {"chunk_id": "one", "rendered_npz": "one.npz"}
            split.write_text(json.dumps({"split": "train", "chunk_count": 2,
                                         "items": [row, row]}))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_split(split, "train", root)
            split.write_text(json.dumps({"split": "train", "chunk_count": 1,
                "items": [{**row, "rendered_npz": "../outside.npz"}]}))
            with self.assertRaisesRegex(ValueError, "escapes dataset root"):
                load_split(split, "train", root)

    def test_validation_rejects_empty_or_nonfinite_outputs(self):
        class Policy(torch.nn.Module):
            def forward(self, observation):
                return {"actions": torch.full((1, 1, 1), math.nan)}
        policy = Policy()
        device = torch.device("cpu")
        mean, std = torch.zeros(3), torch.ones(3)
        with self.assertRaisesRegex(ValueError, "no frames"):
            validate(policy, [], device, mean, std, 1)
        batch = {"vision": torch.zeros(1, 1, 1, 1, 3),
                 "actor": torch.zeros(1, 1, 2), "target": torch.zeros(1, 1, 1)}
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate(policy, [batch], device, mean, std, 1)


if __name__ == "__main__":
    unittest.main()
