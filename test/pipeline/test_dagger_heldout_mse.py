"""CPU tests for opt-in held-out MSE validation; no simulator or service launch."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from pipeline.bc_training.train_bc_h200 import EarlyStopping
from pipeline.dagger.services.heldout_mse import (
    MANIFEST_SCHEMA, RESULT_SCHEMA, RoundMSESelection, canonical_digest,
    evaluate_checkpoint, evaluate_policy, load_validation_manifest,
)
from pipeline.dagger.services.contracts import file_sha256


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.bn = torch.nn.BatchNorm1d(2)
        self.dropout = torch.nn.Dropout(.9)

    def forward(self, observation):
        assert not torch.is_grad_enabled()
        assert torch.is_inference_mode_enabled()
        assert not self.training and not self.bn.training
        # Eval-mode normalization leaves values unchanged (eps removed for exact test).
        shape = observation["actor_obs"].shape
        value = self.bn(observation["actor_obs"].reshape(-1, 2)).reshape(shape)
        return {"actions": self.dropout(value) * self.scale}


class HeldoutMSETest(unittest.TestCase):
    def fixture(self, root):
        spec = {"input_contract": {"views": ["head_color", "hand_left", "hand_right"],
                "vision_shape": [6, 2, 3], "actor_obs_dim": 2,
                "rgb_mean": [0, 0, 0], "rgb_std": [1, 1, 1]},
                "output_contract": {"policy_action_dim": 2}}
        rows = []
        for index, (frames, label) in enumerate(((1, 1.), (3, 3.))):
            path = root / f"chunk-{index}.npz"
            arrays = {f"rgb_gen__{view}": np.zeros((frames, 2, 2, 3), np.uint8)
                      for view in spec["input_contract"]["views"]}
            arrays.update(obs_actor_obs=np.zeros((frames, 2), np.float32),
                          gt_actions=np.full((frames, 2), label, np.float32))
            np.savez(path, **arrays)
            rows.append({"chunk_id": f"heldout-{index}", "path": str(path),
                "sha256": file_sha256(path), "size_bytes": path.stat().st_size,
                "frames": frames, "source_episode_ids": [f"heldout-episode-{index}"],
                "source_window_ids": [f"heldout-window-{index}"],
                "observation_sources": ["generated_controls"]})
        manifest = {"schema": MANIFEST_SCHEMA, "purpose": "validation", "round_id": 1,
                    "policy_spec_sha256": canonical_digest(spec),
                    "teacher_checkpoint_sha256": "a" * 64,
                    "view_order": spec["input_contract"]["views"], "chunks": rows}
        manifest_path = root / "validation.json"
        manifest_path.write_text(json.dumps(manifest))
        training = {"purpose": "train", "round_id": 1, "chunk_sha256s": ["b" * 64],
                    "source_episode_ids": ["training-episode"],
                    "source_window_ids": ["training-window"]}
        return spec, manifest, manifest_path, training

    def test_element_weighted_mse_no_grad_no_batchnorm_or_rng_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            spec, manifest, path, training = self.fixture(Path(directory))
            sha = file_sha256(path)
            load_validation_manifest(path, sha, training)
            policy = TinyPolicy().train()
            policy.scale.grad = torch.tensor(17.)
            before = {key: value.clone() for key, value in policy.state_dict().items()}
            rng = torch.get_rng_state().clone()
            result = evaluate_policy(policy, spec, manifest, checkpoint_sha256="c" * 64,
                                     manifest_sha256=sha, device=torch.device("cpu"))
            # (1 frame * 1^2 + 3 frames * 3^2) / 4 = 7, not chunk-average 5.
            self.assertEqual(result["mse"], 7.)
            self.assertEqual(result["per_action_mse"], [7., 7.])
            self.assertEqual((result["frames"], result["elements"]), (4, 8))
            self.assertTrue(policy.training and policy.bn.training)
            self.assertEqual(float(policy.scale.grad), 17.)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            for key, value in policy.state_dict().items():
                self.assertTrue(torch.equal(value, before[key]))

    def test_training_episode_window_or_chunk_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, manifest, path, training = self.fixture(Path(directory))
            for key, rowkey in (("source_episode_ids", "source_episode_ids"),
                                ("source_window_ids", "source_window_ids"),
                                ("chunk_sha256s", "sha256")):
                other = copy.deepcopy(training)
                value = manifest["chunks"][0][rowkey]
                other[key] = value if isinstance(value, list) else [value]
                with self.assertRaisesRegex(ValueError, "overlap"):
                    load_validation_manifest(path, file_sha256(path), other)

    def test_test_purpose_and_changed_manifest_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, manifest, path, training = self.fixture(Path(directory))
            old_hash = file_sha256(path)
            manifest["purpose"] = "test"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "digest changed"):
                load_validation_manifest(path, old_hash, training)
            with self.assertRaisesRegex(ValueError, "test data is forbidden"):
                load_validation_manifest(path, file_sha256(path), training)

    def test_changed_chunk_or_policy_contract_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            spec, manifest, path, _ = self.fixture(Path(directory))
            other = copy.deepcopy(spec)
            other["input_contract"]["rgb_mean"] = [.1, 0, 0]
            with self.assertRaisesRegex(ValueError, "policy specification"):
                evaluate_policy(TinyPolicy(), other, manifest, checkpoint_sha256="c" * 64,
                                manifest_sha256=file_sha256(path), device=torch.device("cpu"))
            chunk_path = Path(manifest["chunks"][0]["path"])
            with chunk_path.open("ab") as stream:
                stream.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "content changed"):
                evaluate_policy(TinyPolicy(), spec, manifest, checkpoint_sha256="c" * 64,
                                manifest_sha256=file_sha256(path), device=torch.device("cpu"))

    def test_checkpoint_hash_and_strict_policy_factory_are_used(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, _, path, training = self.fixture(root)
            checkpoint = root / "checkpoint.pt"
            torch.save({"policy_spec": spec, "policy_state_dict": TinyPolicy().state_dict()}, checkpoint)
            calls = []
            def factory(actual_spec, state, device):
                calls.append(actual_spec)
                policy = TinyPolicy()
                policy.load_state_dict(state, strict=True)
                return policy.to(device)
            with self.assertRaisesRegex(ValueError, "checkpoint hash mismatch"):
                evaluate_checkpoint(checkpoint, "d" * 64, path, file_sha256(path), training,
                                    device=torch.device("cpu"), policy_factory=factory)
            self.assertFalse(calls)
            result = evaluate_checkpoint(checkpoint, file_sha256(checkpoint), path,
                file_sha256(path), training, device=torch.device("cpu"), policy_factory=factory)
            self.assertEqual(result["mse"], 7.)
            self.assertEqual(calls, [spec])

    def test_nonfinite_prediction_fails_and_restores_mode(self):
        class Invalid(TinyPolicy):
            def forward(self, obs):
                return {"actions": torch.full_like(obs["actor_obs"], math.nan)}
        with tempfile.TemporaryDirectory() as directory:
            spec, manifest, path, _ = self.fixture(Path(directory))
            policy = Invalid().train()
            with self.assertRaisesRegex(ValueError, "non-finite"):
                evaluate_policy(policy, spec, manifest, checkpoint_sha256="c" * 64,
                                manifest_sha256=file_sha256(path), device=torch.device("cpu"))
            self.assertTrue(policy.training)

    def receipt(self, mse, checkpoint="c", manifest="d", round_id=1):
        return {"schema": RESULT_SCHEMA, "purpose": "validation", "round_id": round_id,
                "validation_manifest_sha256": manifest * 64, "checkpoint_sha256": checkpoint * 64,
                "mse": mse, "per_action_mse": [mse, mse], "frames": 81, "elements": 162,
                "chunk_count": 1, "inference_only": True, "selection_metric": "action_mse",
                "test_payloads_opened": False}

    def test_round_stopping_best_patience_idempotency_and_resume(self):
        selection = RoundMSESelection(1, "d" * 64,
            EarlyStopping(min_epochs=1, patience=2, relative_improvement=.01))
        selection.record(0, self.receipt(1.))
        first = selection.record(40, self.receipt(.995, checkpoint="e"))
        self.assertTrue(first["absolute_best"])
        self.assertFalse(first["meaningful_improvement"])
        self.assertEqual(selection.record(40, self.receipt(.995, checkpoint="e")), first)
        restored = RoundMSESelection.restore(selection.state_dict())
        final = restored.record(80, self.receipt(.994, checkpoint="f"))
        self.assertTrue(final["saturated"])
        self.assertEqual(restored.best_checkpoint_sha256, "f" * 64)
        with self.assertRaisesRegex(ValueError, "already saturated"):
            restored.record(120, self.receipt(.9))
        tampered = restored.state_dict()
        tampered["best_checkpoint_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "disagrees"):
            RoundMSESelection.restore(tampered)

    def test_round_rejects_changed_population_test_receipts_or_round(self):
        selection = RoundMSESelection(1, "d" * 64)
        with self.assertRaisesRegex(ValueError, "iteration-zero"):
            selection.record(40, self.receipt(1.))
        selection.record(0, self.receipt(1.))
        for receipt in (self.receipt(.9, manifest="e"), self.receipt(.9, round_id=2)):
            with self.assertRaisesRegex(ValueError, "identity changed"):
                selection.record(40, receipt)
        receipt = self.receipt(.9)
        receipt["purpose"] = "test"
        with self.assertRaisesRegex(ValueError, "test/success/training"):
            selection.record(40, receipt)
        receipt = self.receipt(.9)
        receipt.update(frames=82, elements=164)
        with self.assertRaisesRegex(ValueError, "population changed"):
            selection.record(40, receipt)


if __name__ == "__main__":
    unittest.main()
