"""Coordinator-side recovery gates for the immutable legacy actor source."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from round_dagger import artifacts, orchestration
from round_dagger.actor_host_access import (
    HOST_MUTABLE_DIRECTORIES,
    normalize_container_outputs,
)


DIGESTS = {
    "policy_sha256": "1" * 64,
    "runtime_policy_sha256": "2" * 64,
    "teacher_sha256": "3" * 64,
    "renderer_sha256": "4" * 64,
    "beta_sha256": artifacts.canonical_json_sha256({"beta": 0.5}),
    "config_sha256": "6" * 64,
}
IDENTITY = {
    "campaign_id": "dagger_prod_run1",
    "macro_round": 0,
    "cohort_index": 0,
    "purpose": "train",
}


def _arrays() -> dict[str, np.ndarray]:
    steps, environments, actions = 2, 1, 11
    zeros = np.zeros((steps, environments, actions), dtype=np.float32)
    return {
        "policy_rgb": np.zeros((steps, environments, 6, 4, 3), dtype=np.uint8),
        "expert_actions": zeros,
        "student_actions": zeros.copy(),
        "behavior_actions": zeros.copy(),
        "expert_mask": np.zeros((steps, environments), dtype=np.bool_),
        "dones": np.asarray([[False], [True]], dtype=np.bool_),
        "observation_is_raytraced": np.zeros(
            (steps, environments), dtype=np.bool_
        ),
        "global_env_ids": np.asarray([0], dtype=np.int64),
        "episode_ids": np.zeros((steps, environments), dtype=np.int64),
        "episode_steps": np.asarray([[0], [1]], dtype=np.int64),
        "obs__actor_obs": np.zeros((steps, environments, 3), dtype=np.float32),
    }


class CoordinatorActorRecoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.cohort = self.root / "cohort_00"
        self.run = self.root / "cohort_00_run"
        self.run.mkdir()
        artifacts.create_cohort(
            self.cohort,
            expected_ranks=[0],
            expected_global_env_ids=[0],
            expected_transitions=2,
            view_contract={
                "dtype": "uint8",
                "layout": "vertical",
                "shape": [6, 4, 3],
                "views": ["head_color", "hand_left", "hand_right"],
            },
            action_contract={"dtype": "float32", "shape": [11]},
            **IDENTITY,
            **DIGESTS,
        )
        self.environment = {
            "TRAIN_RUN_DIR": str(self.run),
            "ROUND_DAGGER_COHORT_DIR": str(self.cohort),
        }
        self.helper_calls = 0

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _seal(self) -> None:
        artifacts.publish_rollout_shard(
            self.cohort,
            rank=0,
            rollout_id=0,
            part_index=0,
            arrays=_arrays(),
            **IDENTITY,
            **DIGESTS,
        )
        artifacts.seal_cohort(self.cohort)

    def _normalize(self, _environment: object) -> None:
        self.helper_calls += 1
        normalize_container_outputs(
            self.run,
            host_uid=os.geteuid(),
            host_gid=os.getegid(),
        )

    def test_existing_sealed_cohort_is_recovered_with_one_hook_call(self) -> None:
        self._seal()
        orchestration._prepare_actor_host_access(self.environment)
        scalar_dir = self.run / "telemetry/scalars"
        scalar_dir.chmod(0o555)
        seal = self.cohort / artifacts.SEAL_FILENAME
        before = (seal.read_bytes(), seal.stat().st_mtime_ns)
        calls = 0

        def invoke() -> subprocess.CompletedProcess[bytes]:
            nonlocal calls
            calls += 1
            return subprocess.CompletedProcess([], 0)

        with mock.patch.object(
            orchestration, "_actor_host_access_helper", self._normalize
        ):
            result = orchestration._run_collect_with_host_access(
                invoke, self.environment
            )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(calls, 1)
        self.assertEqual(self.helper_calls, 1)
        self.assertEqual((seal.read_bytes(), seal.stat().st_mtime_ns), before)

    def test_post_seal_failure_retries_hook_but_never_rerolls_out(self) -> None:
        hook_calls = 0
        rollout_calls = 0
        seal_before: tuple[bytes, int] | None = None
        state = self.run / "telemetry/.wandb_upload_state.json"

        def invoke() -> subprocess.CompletedProcess[bytes]:
            nonlocal hook_calls, rollout_calls, seal_before
            hook_calls += 1
            if hook_calls == 1:
                # The coordinator must establish the future cohort's complete
                # host-owned layout before the legacy root actor is entered.
                for relative in HOST_MUTABLE_DIRECTORIES:
                    path = self.run / relative
                    self.assertTrue(path.is_dir())
                    self.assertEqual(path.stat().st_uid, os.geteuid())
                rollout_calls += 1
                self._seal()
                state.write_text("{}\n", encoding="utf-8")
                state.chmod(0o600)
                (self.run / "telemetry/scalars").chmod(0o555)
                seal = self.cohort / artifacts.SEAL_FILENAME
                seal_before = (seal.read_bytes(), seal.stat().st_mtime_ns)
                return subprocess.CompletedProcess([], 2)
            self.assertTrue(artifacts.verify_cohort(self.cohort)["sealed"])
            return subprocess.CompletedProcess([], 0)

        with mock.patch.object(
            orchestration, "_actor_host_access_helper", self._normalize
        ):
            result = orchestration._run_collect_with_host_access(
                invoke, self.environment
            )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(hook_calls, 2)
        self.assertEqual(rollout_calls, 1)
        self.assertEqual(self.helper_calls, 1)
        self.assertEqual(state.stat().st_mode & 0o777, 0o444)
        self.assertIsNotNone(seal_before)
        seal = self.cohort / artifacts.SEAL_FILENAME
        self.assertEqual((seal.read_bytes(), seal.stat().st_mtime_ns), seal_before)

    def test_unsealed_failure_is_not_retried(self) -> None:
        calls = 0

        def invoke() -> subprocess.CompletedProcess[bytes]:
            nonlocal calls
            calls += 1
            (self.run / "telemetry/scalars").chmod(0o555)
            return subprocess.CompletedProcess([], 2)

        with mock.patch.object(
            orchestration, "_actor_host_access_helper", self._normalize
        ):
            result = orchestration._run_collect_with_host_access(
                invoke, self.environment
            )
        self.assertEqual(result.returncode, 2)
        self.assertEqual(calls, 1)
        self.assertEqual(self.helper_calls, 0)


if __name__ == "__main__":
    unittest.main()
