"""Raven-fidelity tests for the three-view offline BC optimizer batch."""

from __future__ import annotations
from test.deployment.source_paths import PIPELINE_ROOT

import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

from bc_training.train_bc_h200 import (
    FIXED_FINAL_TRAIN_CHUNKS_PER_UPDATE,
    RAVEN_BATCHING_CONTRACT,
    RAVEN_CHUNK_FRAMES,
    RAVEN_TRAIN_CHUNKS_PER_UPDATE,
    backward_raven_bc_mse,
    parse_args,
    validate_resume_training_batch_contract,
    validate_training_batch_population,
)


class _BatchNormPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 1, bias=False)
        self.bn = torch.nn.SyncBatchNorm(4, momentum=0.001)
        self.head = torch.nn.Linear(7, 2)

    def forward(self, observation):
        vision = observation["vision_obs"]
        actor = observation["actor_obs"]
        batch, timesteps, height, width, channels = vision.shape
        image = vision.reshape(batch * timesteps, height, width, channels)
        image = image.permute(0, 3, 1, 2)
        features = self.bn(self.conv(image)).mean(dim=(2, 3))
        features = features.reshape(batch, timesteps, 4)
        return {"actions": self.head(torch.cat((actor, features), dim=-1))}


class BcTrainingBatchingTest(unittest.TestCase):
    def test_new_default_is_thirty_two_chunks_and_legacy_remains_sixteen(self) -> None:
        with mock.patch.object(sys, "argv", ["train_bc_h200.py"]):
            self.assertEqual(
                parse_args().chunk_batch_size, FIXED_FINAL_TRAIN_CHUNKS_PER_UPDATE
            )
        self.assertEqual(FIXED_FINAL_TRAIN_CHUNKS_PER_UPDATE, 32)
        self.assertEqual(parse_args(["--training-protocol", "legacy_validation"]).chunk_batch_size, 16)
        self.assertEqual(parse_args(["--chunk-batch-size", "16"]).chunk_batch_size, 16)
        launcher = (
            PIPELINE_ROOT
            / "bc_training/train_raven_bc_h200.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("CHUNK_BATCH_SIZE=${CHUNK_BATCH_SIZE:-16}", launcher)
        self.assertIn("CHUNK_BATCH_SIZE=${CHUNK_BATCH_SIZE:-32}", launcher)
        self.assertIn("[[ $CHUNK_BATCH_SIZE == 16 || $CHUNK_BATCH_SIZE == 32 ]]", launcher)
        self.assertIn("frames_per_update={batch * 81}", launcher)
        self.assertIn("raven_full_bxt_16x81_activation_checkpoint_v1", RAVEN_BATCHING_CONTRACT)

    def test_partial_or_different_batch_population_fails_closed(self) -> None:
        self.assertEqual(validate_training_batch_population(1600, 16), 100)
        with self.assertRaisesRegex(ValueError, "chunk-batch-size 16"):
            validate_training_batch_population(1600, 4)
        with self.assertRaisesRegex(ValueError, "refusing to drop or pad"):
            validate_training_batch_population(1599, 16)

    def test_resume_rejects_legacy_or_changed_batching_contract(self) -> None:
        expected = {"schema": RAVEN_BATCHING_CONTRACT, "chunks_per_update": 16}
        validate_resume_training_batch_contract(
            {"training_batch_contract": expected}, expected
        )
        with self.assertRaisesRegex(ValueError, "optimizer histories cannot be mixed"):
            validate_resume_training_batch_contract({}, expected)
        with self.assertRaisesRegex(ValueError, "optimizer histories cannot be mixed"):
            validate_resume_training_batch_contract(
                {"training_batch_contract": {**expected, "chunks_per_update": 4}},
                expected,
            )

    def test_full_16x81_batch_matches_direct_batchnorm_gradient_and_state(self) -> None:
        torch.manual_seed(71)
        shape = (RAVEN_TRAIN_CHUNKS_PER_UPDATE, RAVEN_CHUNK_FRAMES)
        vision = torch.randn(*shape, 2, 2, 3)
        actor = torch.randn(*shape, 3)
        target = torch.randn(*shape, 2)

        actual = _BatchNormPolicy()
        reference = _BatchNormPolicy()
        reference.load_state_dict(actual.state_dict())
        actual.train()
        reference.train()

        actual_mse = backward_raven_bc_mse(actual, vision, actor, target)
        expected_loss = torch.nn.functional.mse_loss(
            reference({"vision_obs": vision, "actor_obs": actor})["actions"],
            target,
        )
        expected_loss.backward()

        self.assertAlmostEqual(actual_mse, float(expected_loss.detach()), places=6)
        self.assertTrue(
            torch.equal(actual.bn.num_batches_tracked, reference.bn.num_batches_tracked)
        )
        self.assertTrue(torch.allclose(actual.bn.running_mean, reference.bn.running_mean))
        self.assertTrue(torch.allclose(actual.bn.running_var, reference.bn.running_var))
        for actual_parameter, expected_parameter in zip(
            actual.parameters(), reference.parameters(), strict=True
        ):
            self.assertTrue(
                torch.allclose(
                    actual_parameter.grad, expected_parameter.grad, rtol=1e-5, atol=1e-6
                )
            )

    def test_backward_rejects_incomplete_chunk_batch(self) -> None:
        policy = _BatchNormPolicy()
        vision = torch.randn(15, RAVEN_CHUNK_FRAMES, 2, 2, 3)
        actor = torch.randn(15, RAVEN_CHUNK_FRAMES, 3)
        target = torch.randn(15, RAVEN_CHUNK_FRAMES, 2)
        with self.assertRaisesRegex(ValueError, r"B,T=\(16, 81\)"):
            backward_raven_bc_mse(policy, vision, actor, target)


if __name__ == "__main__":
    unittest.main()
