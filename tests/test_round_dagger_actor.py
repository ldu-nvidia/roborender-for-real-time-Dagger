import os
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from round_dagger import artifacts
from round_dagger.actor import (
    ActorExportError,
    denormalize_policy_rgb,
    export_trainer_rollout,
    runtime_policy_sha256,
    runtime_state_dict_sha256,
    trainer_rollout_arrays,
)


class FakeStorage:
    def __init__(self, values):
        self.values = values
        self.step = int(values["vision_obs"].shape[0])

    def query_key(self, key):
        return self.values[key]


class NumpyPolicy:
    def __init__(self):
        self.weight = np.arange(10, dtype=np.float32).reshape(2, 5)
        self.bias = np.zeros(2, dtype=np.float32)

    def state_dict(self):
        return {"weight": self.weight, "bias": self.bias}


def fake_trainer(*, steps=3, envs=4):
    mean = np.asarray([0.2, 0.3, 0.4], dtype=np.float32)
    std = np.asarray([0.5, 0.4, 0.3], dtype=np.float32)
    rgb = np.arange(steps * envs * 6 * 4 * 3, dtype=np.uint8).reshape(
        steps, envs, 6, 4, 3
    )
    normalized = (rgb.astype(np.float32) / 255.0 - mean) / std
    expert = np.arange(steps * envs * 2, dtype=np.float32).reshape(steps, envs, 2)
    student = expert + np.float32(0.5)
    expert_mask = (np.indices((steps, envs))[0] % 2 == 0)[..., None]
    behavior = np.where(expert_mask, expert, student).astype(np.float32)
    ids = np.broadcast_to(
        np.arange(10, 10 + envs, dtype=np.int64)[None, :, None],
        (steps, envs, 1),
    ).copy()
    episode_steps = np.broadcast_to(
        np.arange(steps, dtype=np.int64)[:, None, None], (steps, envs, 1)
    ).copy()
    values = {
        "vision_obs": normalized,
        "gt_actions": expert,
        "student_actions": student,
        "actions": behavior,
        "expert_mask": expert_mask,
        "dones": np.zeros((steps, envs, 1), dtype=np.bool_),
        "round_observation_is_raytraced": np.zeros(
            (steps, envs, 1), dtype=np.bool_
        ),
        "round_global_env_ids": ids,
        "round_episode_ids": np.zeros((steps, envs, 1), dtype=np.int64),
        "round_episode_steps": episode_steps,
        "actor_obs": np.ones((steps, envs, 5), dtype=np.float32),
    }
    cameras = SimpleNamespace(image_mean=mean.tolist(), image_std=std.tolist())
    env = SimpleNamespace(
        config=SimpleNamespace(
            simulator=SimpleNamespace(config=SimpleNamespace(cameras=cameras))
        )
    )
    trainer = SimpleNamespace(
        storage=FakeStorage(values),
        num_steps_per_env=steps,
        env=env,
        algo_obs_dim_dict={"vision_obs": 72, "actor_obs": 5},
        accelerator=SimpleNamespace(process_index=0),
    )
    return trainer, rgb


class ActorConversionTest(unittest.TestCase):
    def test_denormalization_recovers_uint8_policy_observation(self):
        trainer, expected = fake_trainer()
        arrays = trainer_rollout_arrays(trainer)
        np.testing.assert_array_equal(arrays["policy_rgb"], expected)
        self.assertEqual(arrays["global_env_ids"].shape, (3, 4))
        self.assertEqual(arrays["expert_mask"].shape, (3, 4))
        self.assertEqual(arrays["obs__actor_obs"].shape, (3, 4, 5))

    def test_bad_normalization_and_incomplete_storage_fail_closed(self):
        with self.assertRaisesRegex(ActorExportError, "provided together"):
            denormalize_policy_rgb(
                np.zeros((1, 1, 2, 2, 3), dtype=np.float32),
                image_mean=[0, 0, 0],
                image_std=None,
            )
        trainer, _ = fake_trainer()
        trainer.storage.step -= 1
        with self.assertRaisesRegex(ActorExportError, "incomplete"):
            trainer_rollout_arrays(trainer)

    def test_runtime_digest_is_stable_and_parameter_sensitive(self):
        policy = NumpyPolicy()
        first = runtime_policy_sha256(policy)
        self.assertEqual(first, runtime_policy_sha256(policy))
        policy.weight[0, 0] += 1
        self.assertNotEqual(first, runtime_policy_sha256(policy))

    def test_state_mapping_and_module_runtime_digests_match(self):
        policy = NumpyPolicy()
        self.assertEqual(
            runtime_policy_sha256(policy),
            runtime_state_dict_sha256(policy.state_dict()),
        )
        with self.assertRaisesRegex(ActorExportError, "non-empty mapping"):
            runtime_state_dict_sha256({})

    def test_verified_checkpoint_runtime_hash_cli(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is unavailable")
        from round_dagger.artifacts import sha256_file

        script = Path(__file__).parents[1] / "scripts/round_dagger/checkpoint_runtime_sha256.py"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "candidate.pt"
            state = {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
            torch.save({"policy_state_dict": state}, checkpoint)
            expected = sha256_file(checkpoint)
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--checkpoint",
                    str(checkpoint),
                    "--expected-sha256",
                    expected,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(result.stdout)
            self.assertEqual(report["checkpoint_sha256"], expected)
            self.assertEqual(
                report["runtime_policy_sha256"], runtime_state_dict_sha256(state)
            )
            self.assertEqual(report["state_entries"], 1)

            checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
            rejected = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--checkpoint",
                    str(checkpoint),
                    "--expected-sha256",
                    expected,
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("mismatch", rejected.stderr)

            link = root / "candidate-link.pt"
            link.symlink_to(checkpoint)
            rejected_link = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--checkpoint",
                    str(link),
                    "--expected-sha256",
                    sha256_file(checkpoint),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rejected_link.returncode, 0)
            self.assertIn("symlink", rejected_link.stderr)


class ActorPublicationIntegrationTest(unittest.TestCase):
    def test_one_rank_exports_parts_and_seals_exactly(self):
        trainer, _ = fake_trainer()
        policy = NumpyPolicy()
        runtime_sha = runtime_policy_sha256(policy)
        digests = {
            "policy_sha256": "1" * 64,
            "runtime_policy_sha256": runtime_sha,
            "teacher_sha256": "3" * 64,
            "renderer_sha256": "4" * 64,
            "beta_sha256": "5" * 64,
            "config_sha256": "6" * 64,
        }
        identity = {
            "campaign_id": "actor-test",
            "macro_round": 0,
            "cohort_index": 0,
            "purpose": "train",
        }
        with tempfile.TemporaryDirectory() as temporary:
            cohort = Path(temporary) / "cohort"
            artifacts.create_cohort(
                cohort,
                expected_ranks=[0],
                expected_global_env_ids=[10, 11, 12, 13],
                expected_transitions=12,
                view_contract={
                    "dtype": "uint8",
                    "layout": "vertical",
                    "shape": [6, 4, 3],
                    "views": ["head_color", "hand_left", "hand_right"],
                },
                action_contract={"dtype": "float32", "shape": [2]},
                **identity,
                **digests,
            )
            environment = {
                "ROUND_DAGGER_COHORT_DIR": str(cohort),
                "ROUND_DAGGER_CAMPAIGN_ID": identity["campaign_id"],
                "ROUND_DAGGER_MACRO_ROUND": "0",
                "ROUND_DAGGER_COHORT_INDEX": "0",
                "ROUND_DAGGER_POLICY_SHA256": digests["policy_sha256"],
                "ROUND_DAGGER_RUNTIME_POLICY_SHA256": runtime_sha,
                "ROUND_DAGGER_TEACHER_SHA256": digests["teacher_sha256"],
                "ROUND_DAGGER_RENDERER_SHA256": digests["renderer_sha256"],
                "ROUND_DAGGER_BETA_SHA256": digests["beta_sha256"],
                "ROUND_DAGGER_CONFIG_SHA256": digests["config_sha256"],
                "ROUND_DAGGER_ENVS_PER_SHARD": "2",
                "ROUND_DAGGER_ROLLOUT_ID": "0",
            }
            with mock.patch.dict(os.environ, environment, clear=False):
                reports = export_trainer_rollout(
                    trainer,
                    policy_module=policy,
                    initial_runtime_policy_sha256=runtime_sha,
                )
            self.assertEqual(len(reports), 2)
            sealed = artifacts.seal_cohort(cohort)
            self.assertEqual(sealed["transition_count"], 12)
            self.assertEqual(sealed["global_env_ids"], [10, 11, 12, 13])

    def test_changed_policy_is_rejected_before_publish(self):
        trainer, _ = fake_trainer()
        policy = NumpyPolicy()
        before = runtime_policy_sha256(policy)
        policy.bias += 1
        with self.assertRaisesRegex(ActorExportError, "changed during collection"):
            export_trainer_rollout(
                trainer,
                policy_module=policy,
                initial_runtime_policy_sha256=before,
            )


if __name__ == "__main__":
    unittest.main()
