"""Contract, launcher, and isolation gates for bounded actor validation pairs."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from round_dagger import artifacts
from round_dagger.actor_pair import ActorPairError, inspect_actor_pair
from round_dagger.validation_pair import (
    ValidationPairError,
    inspect_validation_pair,
    seal_validation_pair,
    validation_contract,
)


ROOT = Path(__file__).resolve().parents[1]
PAIR_SCRIPT = ROOT / "scripts/round_dagger/run_actor_validation_pair.sh"
RANK_SCRIPT = ROOT / "scripts/round_dagger/run_actor_validation.sh"
DOCKER_EXECUTOR = (
    ROOT / "scripts/round_dagger/run_actor_validation_rank_docker.sh"
)
DIGESTS = {
    "policy_sha256": "1" * 64,
    "runtime_policy_sha256": "2" * 64,
    "teacher_sha256": "3" * 64,
    "renderer_sha256": "4" * 64,
    "beta_sha256": "5" * 64,
    "config_sha256": "6" * 64,
}


def _create_validation_cohort(
    path: Path,
    *,
    purpose: str = "eval",
    envs_per_rank: int = 2,
    steps: int = 3,
    offset: int = 10,
) -> None:
    total_envs = 2 * envs_per_rank
    artifacts.create_cohort(
        path,
        campaign_id="bounded-validation-test",
        macro_round=7,
        cohort_index=3,
        purpose=purpose,
        expected_ranks=[0, 1],
        expected_global_env_ids=list(range(offset, offset + total_envs)),
        expected_transitions=total_envs * steps,
        view_contract={
            "dtype": "uint8",
            "layout": "vertical",
            "shape": [6, 4, 3],
            "views": ["head_color", "hand_left", "hand_right"],
        },
        action_contract={"dtype": "float32", "shape": [2]},
        **DIGESTS,
    )


def _arrays(*, global_env_ids: list[int], steps: int) -> dict[str, np.ndarray]:
    envs = len(global_env_ids)
    zeros = np.zeros((steps, envs, 2), dtype=np.float32)
    return {
        "policy_rgb": np.zeros((steps, envs, 6, 4, 3), dtype=np.uint8),
        "expert_actions": np.ones_like(zeros),
        "student_actions": zeros,
        "behavior_actions": zeros.copy(),
        "expert_mask": np.zeros((steps, envs), dtype=np.bool_),
        "dones": np.zeros((steps, envs), dtype=np.bool_),
        "observation_is_raytraced": np.zeros((steps, envs), dtype=np.bool_),
        "global_env_ids": np.asarray(global_env_ids, dtype=np.int64),
        "episode_ids": np.zeros((steps, envs), dtype=np.int64),
        "episode_steps": np.broadcast_to(
            np.arange(steps, dtype=np.int64)[:, None], (steps, envs)
        ).copy(),
        "obs__actor_obs": np.zeros((steps, envs, 3), dtype=np.float32),
    }


class ValidationPairContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_contract_drives_bounded_shape_and_production_gate_stays_fixed(self) -> None:
        cohort = self.root / "cohort"
        _create_validation_cohort(cohort, envs_per_rank=4, steps=17, offset=96)
        shape = validation_contract(cohort)
        self.assertEqual(shape["ranks"], [0, 1])
        self.assertEqual(shape["environments_per_rank"], 4)
        self.assertEqual(shape["steps_per_environment"], 17)
        self.assertEqual(shape["global_env_offset"], 96)
        self.assertEqual(shape["expected_transitions"], 136)

        with self.assertRaisesRegex(ActorPairError, "2x32x512 production"):
            inspect_actor_pair(cohort)

        maximum = self.root / "maximum"
        _create_validation_cohort(maximum, envs_per_rank=32, steps=300, offset=0)
        maximum_shape = validation_contract(maximum)
        self.assertEqual(maximum_shape["environments_per_rank"], 32)
        self.assertEqual(maximum_shape["steps_per_environment"], 300)
        self.assertEqual(maximum_shape["expected_transitions"], 19200)

    def test_bounds_and_consecutive_environment_partition_fail_closed(self) -> None:
        too_many = self.root / "too-many"
        _create_validation_cohort(too_many, envs_per_rank=33)
        with self.assertRaisesRegex(ValidationPairError, "<= 32"):
            validation_contract(too_many)

        too_long = self.root / "too-long"
        _create_validation_cohort(too_long, steps=301)
        with self.assertRaisesRegex(ValidationPairError, "1 through 300"):
            validation_contract(too_long)

        sparse = self.root / "sparse"
        _create_validation_cohort(sparse)
        contract_path = sparse / "contract.json"
        # Immutable publication is not needed for this malformed-fixture gate.
        contract_path.chmod(0o644)
        contract = artifacts.load_strict_json(contract_path)
        contract["expected_global_env_ids"] = [10, 11, 13, 14]
        contract_path.write_text(
            __import__("json").dumps(contract, sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValidationPairError, "consecutive"):
            validation_contract(sparse)

    def test_only_two_exact_rank_partitions_can_seal(self) -> None:
        cohort = self.root / "cohort"
        _create_validation_cohort(cohort)
        artifacts.publish_rollout_shard(
            cohort,
            rank=0,
            rollout_id=0,
            part_index=0,
            arrays=_arrays(global_env_ids=[10, 11], steps=3),
            campaign_id="bounded-validation-test",
            macro_round=7,
            cohort_index=3,
            purpose="eval",
            **DIGESTS,
        )
        self.assertEqual(
            inspect_validation_pair(cohort)["rank_states"],
            {"0": "complete", "1": "empty"},
        )
        with self.assertRaisesRegex(ValidationPairError, "both bounded"):
            seal_validation_pair(cohort)
        artifacts.publish_rollout_shard(
            cohort,
            rank=1,
            rollout_id=0,
            part_index=0,
            arrays=_arrays(global_env_ids=[12, 13], steps=3),
            campaign_id="bounded-validation-test",
            macro_round=7,
            cohort_index=3,
            purpose="eval",
            **DIGESTS,
        )
        sealed = seal_validation_pair(cohort)
        self.assertEqual(sealed["transition_count"], 12)
        self.assertEqual(sealed["purpose"], "eval")


class ValidationPairLauncherTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.run_dir = self.root / "run"
        self.cohort = self.root / "cohort"
        _create_validation_cohort(self.cohort)
        self.record_executor = self.root / "record-executor"
        self.record_executor.write_text(
            "#!/usr/bin/env bash\n"
            "rank=$1\n"
            "{\n"
            "  printf 'rank=%s\\n' \"$rank\"\n"
            "  printf 'script=%s\\n' \"$2\"\n"
            "  printf 'ranks=%s\\n' \"$ROUND_DAGGER_VALIDATION_RANKS\"\n"
            "  printf 'envs=%s\\n' \"$ROUND_DAGGER_VALIDATION_ENVS_PER_RANK\"\n"
            "  printf 'steps=%s\\n' \"$ROUND_DAGGER_VALIDATION_STEPS\"\n"
            "  printf 'purpose=%s\\n' \"$ROUND_DAGGER_VALIDATION_PURPOSE\"\n"
            "  printf 'offset=%s\\n' \"$ROUND_DAGGER_GLOBAL_ENV_OFFSET\"\n"
            f"}} > {self.root!s}/record-rank-$rank\n",
            encoding="utf-8",
        )
        self.record_executor.chmod(0o755)
        self.zero_executor = self.root / "zero-executor"
        self.zero_executor.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        self.zero_executor.chmod(0o755)
        self.fail_and_reap_executor = self.root / "fail-and-reap-executor"
        self.fail_and_reap_executor.write_text(
            "#!/usr/bin/env bash\n"
            "if [ \"$1\" = 0 ]; then\n"
            "  sleep 0.2\n"
            f"  printf reaped > {self.root!s}/rank-0-finished\n"
            "  exit 0\n"
            "fi\n"
            "exit 17\n",
            encoding="utf-8",
        )
        self.fail_and_reap_executor.chmod(0o755)
        self.publisher = self.root / "publish-rank.py"
        self.publisher.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "import numpy as np\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            "from round_dagger.artifacts import load_strict_json, publish_rollout_shard\n"
            "rank = int(sys.argv[1])\n"
            "cohort = Path(os.environ['ROUND_DAGGER_COHORT_DIR'])\n"
            "c = load_strict_json(cohort / 'contract.json')\n"
            "steps = c['expected_steps_per_env']\n"
            "per_rank = len(c['expected_global_env_ids']) // 2\n"
            "ids = c['expected_global_env_ids'][rank*per_rank:(rank+1)*per_rank]\n"
            "envs = len(ids)\n"
            "actions = np.zeros((steps, envs, 2), dtype=np.float32)\n"
            "arrays = {\n"
            " 'policy_rgb': np.zeros((steps, envs, 6, 4, 3), dtype=np.uint8),\n"
            " 'expert_actions': np.ones_like(actions),\n"
            " 'student_actions': actions, 'behavior_actions': actions.copy(),\n"
            " 'expert_mask': np.zeros((steps, envs), dtype=np.bool_),\n"
            " 'dones': np.zeros((steps, envs), dtype=np.bool_),\n"
            " 'observation_is_raytraced': np.zeros((steps, envs), dtype=np.bool_),\n"
            " 'global_env_ids': np.asarray(ids, dtype=np.int64),\n"
            " 'episode_ids': np.zeros((steps, envs), dtype=np.int64),\n"
            " 'episode_steps': np.broadcast_to(np.arange(steps, dtype=np.int64)[:, None], (steps, envs)).copy(),\n"
            " 'obs__actor_obs': np.zeros((steps, envs, 3), dtype=np.float32),\n"
            "}\n"
            "digests = {k: c[k] for k in ('policy_sha256','runtime_policy_sha256','teacher_sha256','renderer_sha256','beta_sha256','config_sha256')}\n"
            "publish_rollout_shard(cohort, rank=rank, rollout_id=0, part_index=0, arrays=arrays, campaign_id=c['campaign_id'], macro_round=c['macro_round'], cohort_index=c['cohort_index'], purpose=c['purpose'], **digests)\n",
            encoding="utf-8",
        )
        self.publisher.chmod(0o755)
        self.publish_executor = self.root / "publish-executor"
        self.publish_executor.write_text(
            "#!/usr/bin/env bash\n"
            f"exec {self.publisher!s} \"$1\"\n",
            encoding="utf-8",
        )
        self.publish_executor.chmod(0o755)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _environment(
        self, *, executor: Path, dry_run: bool, train_ack: bool = False
    ) -> dict[str, str]:
        environment = {
            **os.environ,
            "DRY_RUN": "1" if dry_run else "0",
            "ROUND_DAGGER_COHORT_DIR": str(self.cohort),
            "TRAIN_RUN_DIR": str(self.run_dir),
            "ROUND_DAGGER_CAMPAIGN_ID": "bounded-validation-test",
            "ROUND_DAGGER_MACRO_ROUND": "7",
            "ROUND_DAGGER_COHORT_INDEX": "3",
            "ROUND_DAGGER_VALIDATION_PAIR_EXECUTOR": str(executor),
            "ROUND_DAGGER_VALIDATION_PAIR_STAGGER_SECONDS": "0",
            # Stale ambient topology must not override contract-derived values.
            "ROUND_DAGGER_VALIDATION_RANKS": "1",
            "ROUND_DAGGER_VALIDATION_ENVS_PER_RANK": "31",
            "ROUND_DAGGER_VALIDATION_STEPS": "299",
            "ROUND_DAGGER_GLOBAL_ENV_OFFSET": "999",
            "RANK": "8",
            "WORLD_SIZE": "9",
            "MASTER_ADDR": "must-not-propagate",
        }
        environment.pop("ROUND_DAGGER_VALIDATION_PURPOSE", None)
        environment.pop("ROUND_DAGGER_VALIDATION_TRAIN_ACK", None)
        if train_ack:
            environment["ROUND_DAGGER_VALIDATION_TRAIN_ACK"] = (
                "bounded-train-replay"
            )
        return environment

    def _run(self, environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(PAIR_SCRIPT)],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )

    def test_dry_run_launches_both_contract_derived_ranks_without_sealing(self) -> None:
        result = self._run(
            self._environment(executor=self.record_executor, dry_run=True)
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("DRY_RUN_ACTOR_VALIDATION_PAIR_OK", result.stdout)
        for rank in (0, 1):
            record = (self.root / f"record-rank-{rank}").read_text(encoding="utf-8")
            self.assertIn(f"rank={rank}", record)
            self.assertIn(f"script={RANK_SCRIPT}", record)
            self.assertIn("ranks=2", record)
            self.assertIn("envs=2", record)
            self.assertIn("steps=3", record)
            self.assertIn("purpose=eval", record)
            self.assertIn("offset=10", record)
        self.assertFalse((self.cohort / "seal.json").exists())

    def test_false_zero_rank_exits_cannot_seal_without_exact_shards(self) -> None:
        result = self._run(
            self._environment(executor=self.zero_executor, dry_run=False)
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("returned without exact shard coverage", result.stderr)
        self.assertFalse((self.cohort / "seal.json").exists())

    def test_one_failed_rank_does_not_abandon_the_other_child(self) -> None:
        result = self._run(
            self._environment(
                executor=self.fail_and_reap_executor, dry_run=False
            )
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("rank0=0 rank1=17", result.stderr)
        self.assertEqual(
            (self.root / "rank-0-finished").read_text(encoding="utf-8"),
            "reaped",
        )
        self.assertFalse((self.cohort / "seal.json").exists())

    def test_exact_fake_rank_payloads_are_reaped_verified_and_sealed(self) -> None:
        result = self._run(
            self._environment(executor=self.publish_executor, dry_run=False)
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ACTOR_VALIDATION_PAIR_SEALED", result.stdout)
        report = artifacts.verify_cohort(self.cohort)
        self.assertEqual(report["transition_count"], 12)
        self.assertEqual(report["ranks"], [0, 1])
        self.assertEqual(report["global_env_ids"], [10, 11, 12, 13])

    def test_train_contract_requires_explicit_ack_before_launch(self) -> None:
        self.cohort = self.root / "train-cohort"
        _create_validation_cohort(self.cohort, purpose="train")
        rejected = self._run(
            self._environment(executor=self.record_executor, dry_run=True)
        )
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("bounded train validation requires", rejected.stderr)
        self.assertFalse((self.root / "record-rank-0").exists())

        accepted = self._run(
            self._environment(
                executor=self.record_executor, dry_run=True, train_ack=True
            )
        )
        self.assertEqual(accepted.returncode, 0, accepted.stderr)
        self.assertIn("purpose=train", accepted.stdout)


class ValidationRankDockerIsolationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.argv_file = self.root / "docker.argv"
        self.fake_docker = self.root / "docker"
        self.fake_docker.write_text(
            "#!/usr/bin/env bash\n"
            f"printf '%s\\n' \"$@\" > {self.argv_file!s}\n",
            encoding="utf-8",
        )
        self.fake_docker.chmod(0o755)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _environment(self, *, ipc: str = "private") -> dict[str, str]:
        return {
            **os.environ,
            "ROUND_DAGGER_VALIDATION_ACTOR_RANK": "1",
            "ROUND_DAGGER_VALIDATION_RANKS": "2",
            "ROUND_DAGGER_ACTOR_RANK": "1",
            "ROUND_DAGGER_ACTOR_WORLD_SIZE": "2",
            "ROUND_DAGGER_CAMPAIGN_ID": "validation-docker-test",
            "ROUND_DAGGER_MACRO_ROUND": "7",
            "ROUND_DAGGER_COHORT_INDEX": "3",
            "ROUND_DAGGER_DOCKER_BIN": str(self.fake_docker),
            "ROUND_DAGGER_RTX_IMAGE": "validation:test",
            "ROUND_DAGGER_ACTOR_DOCKER_IPC": ipc,
            "ROUND_DAGGER_ACTOR_DOCKER_SHM_SIZE": "24g",
            "ROUND_DAGGER_ACTOR_DOCKER_GPU_IDS": "4,7",
            "ROUND_DAGGER_ACTOR_DOCKER_MOUNT_ROOTS": str(self.root),
        }

    def test_executor_uses_second_gpu_private_ipc_and_validation_primitive(self) -> None:
        result = subprocess.run(
            ["bash", str(DOCKER_EXECUTOR), "1", str(RANK_SCRIPT)],
            env=self._environment(),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = self.argv_file.read_text(encoding="utf-8").splitlines()
        self.assertIn("--ipc=private", argv)
        self.assertNotIn("--ipc=host", argv)
        self.assertIn("--shm-size=24g", argv)
        self.assertIn('"device=7"', argv)
        self.assertIn(str(RANK_SCRIPT), argv)
        self.assertNotIn(
            str(ROOT / "scripts/round_dagger/run_actor_cohort.sh"), argv
        )

    def test_host_ipc_is_rejected_before_docker_is_invoked(self) -> None:
        result = subprocess.run(
            ["bash", str(DOCKER_EXECUTOR), "1", str(RANK_SCRIPT)],
            env=self._environment(ipc="host"),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("host IPC is forbidden", result.stderr)
        self.assertFalse(self.argv_file.exists())

    def test_production_rank_primitive_is_rejected(self) -> None:
        result = subprocess.run(
            [
                "bash",
                str(DOCKER_EXECUTOR),
                "1",
                str(ROOT / "scripts/round_dagger/run_actor_cohort.sh"),
            ],
            env=self._environment(),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("unknown actor primitive", result.stderr)
        self.assertFalse(self.argv_file.exists())


if __name__ == "__main__":
    unittest.main()
