"""Focused gates for independent two-process actor cohort orchestration."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from round_dagger import artifacts
from round_dagger.actor_pair import (
    ActorPairError,
    inspect_actor_pair,
    require_actor_rank_completion,
    seal_actor_pair,
)
from tests.test_round_dagger_handoff import publish_policy_bundle


ROOT = Path(__file__).resolve().parents[1]
PAIR_SCRIPT = ROOT / "scripts/round_dagger/run_actor_cohort_pair.sh"
DOCKER_EXECUTOR = ROOT / "scripts/round_dagger/run_actor_rank_docker.sh"
WANDB_RANK_SCRIPT = ROOT / "scripts/round_dagger/run_actor_cohort_with_wandb.sh"
DIGESTS = {
    "policy_sha256": "1" * 64,
    "runtime_policy_sha256": "2" * 64,
    "teacher_sha256": "3" * 64,
    "renderer_sha256": "4" * 64,
    "beta_sha256": "5" * 64,
    "config_sha256": "6" * 64,
}


def _arrays(rank: int) -> dict[str, np.ndarray]:
    steps, envs = 300, 32
    global_start = rank * envs
    expert = np.ones((steps, envs, 2), dtype=np.float32)
    student = np.zeros_like(expert)
    mask = np.zeros((steps, envs), dtype=np.bool_)
    return {
        "policy_rgb": np.zeros((steps, envs, 6, 4, 3), dtype=np.uint8),
        "expert_actions": expert,
        "student_actions": student,
        "behavior_actions": student.copy(),
        "expert_mask": mask,
        "dones": np.zeros((steps, envs), dtype=np.bool_),
        "observation_is_raytraced": np.zeros((steps, envs), dtype=np.bool_),
        "global_env_ids": np.arange(global_start, global_start + envs, dtype=np.int64),
        "episode_ids": np.zeros((steps, envs), dtype=np.int64),
        "episode_steps": np.broadcast_to(
            np.arange(steps, dtype=np.int64)[:, None], (steps, envs)
        ).copy(),
        "obs__actor_obs": np.zeros((steps, envs, 3), dtype=np.float32),
    }


def _cohort(path: Path, *, image_shape: list[int] | None = None) -> None:
    artifacts.create_cohort(
        path,
        campaign_id="actor-pair-test",
        macro_round=0,
        cohort_index=0,
        purpose="train",
        expected_ranks=[0, 1],
        expected_global_env_ids=list(range(64)),
        expected_transitions=64 * 300,
        view_contract={
            "dtype": "uint8",
            "layout": "vertical",
            "shape": image_shape or [6, 4, 3],
            "views": ["head_color", "hand_left", "hand_right"],
        },
        action_contract={"dtype": "float32", "shape": [2]},
        **DIGESTS,
    )


class ActorPairArtifactGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.cohort = self.root / "cohort"
        _cohort(self.cohort)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _publish(self, rank: int) -> None:
        artifacts.publish_rollout_shard(
            self.cohort,
            rank=rank,
            rollout_id=0,
            part_index=0,
            arrays=_arrays(rank),
            campaign_id="actor-pair-test",
            macro_round=0,
            cohort_index=0,
            purpose="train",
            **DIGESTS,
        )

    def test_only_exact_joint_rank_coverage_can_publish_seal(self) -> None:
        self.assertEqual(
            inspect_actor_pair(self.cohort)["rank_states"],
            {"0": "empty", "1": "empty"},
        )
        self._publish(0)
        self.assertEqual(
            inspect_actor_pair(self.cohort)["rank_states"],
            {"0": "complete", "1": "empty"},
        )
        with self.assertRaisesRegex(ActorPairError, "both independent"):
            seal_actor_pair(self.cohort)
        self.assertFalse((self.cohort / "seal.json").exists())

        self._publish(1)
        sealed = seal_actor_pair(self.cohort)
        self.assertEqual(sealed["ranks"], [0, 1])
        self.assertEqual(sealed["transition_count"], 64 * 300)
        self.assertTrue(inspect_actor_pair(self.cohort)["sealed"])
        self.assertEqual(
            seal_actor_pair(self.cohort)["cohort_sha256"],
            sealed["cohort_sha256"],
        )

    def test_partial_and_unreferenced_output_fail_closed(self) -> None:
        arrays = _arrays(0)
        partial = {
            name: value[:4] if name == "global_env_ids" else value[:, :4]
            for name, value in arrays.items()
        }
        artifacts.publish_rollout_shard(
            self.cohort,
            rank=0,
            rollout_id=0,
            part_index=0,
            arrays=partial,
            campaign_id="actor-pair-test",
            macro_round=0,
            cohort_index=0,
            purpose="train",
            **DIGESTS,
        )
        self.assertEqual(inspect_actor_pair(self.cohort)["rank_states"]["0"], "partial")
        with self.assertRaisesRegex(ActorPairError, "both independent"):
            seal_actor_pair(self.cohort)

        other = self.root / "stray"
        _cohort(other)
        (other / "shards" / "orphan.npz").write_bytes(b"not committed")
        with self.assertRaisesRegex(ActorPairError, "uncommitted or unreferenced"):
            inspect_actor_pair(other)

    def test_rank_gate_does_not_read_a_peers_in_progress_payload(self) -> None:
        self._publish(0)
        # Payload-first publication is deliberately invisible until its JSON
        # sidecar commits.  Rank zero must be able to finish while rank one is
        # between those two atomic publication phases.
        peer_partial = (
            self.cohort
            / "shards"
            / "rank-0001_rollout-00000000_part-0000.npz"
        )
        peer_partial.write_bytes(b"peer still publishing")
        report = require_actor_rank_completion(self.cohort, 0)
        self.assertEqual(report["state"], "complete")
        self.assertEqual(report["transition_count"], 32 * 300)


class ActorPairLauncherDryRunTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.checkpoint = self.root / "policy.pt"
        self.teacher = self.root / "teacher.pt"
        self.checkpoint.write_bytes(b"policy")
        self.teacher.write_bytes(b"teacher")
        self.policy_sha = artifacts.sha256_file(self.checkpoint)
        self.teacher_sha = artifacts.sha256_file(self.teacher)
        self.runtime_sha = "2" * 64
        self.spec, self.manifest = publish_policy_bundle(
            self.checkpoint, self.runtime_sha
        )
        self.cohort = self.root / "cohort"
        artifacts.create_cohort(
            self.cohort,
            campaign_id="pair-launcher-test",
            macro_round=0,
            cohort_index=0,
            purpose="train",
            expected_ranks=[0, 1],
            expected_global_env_ids=list(range(64)),
            expected_transitions=64 * 300,
            view_contract={
                "dtype": "uint8",
                "layout": "vertical",
                "shape": [324, 192, 3],
                "views": ["head_color", "hand_left", "hand_right"],
            },
            action_contract={"dtype": "float32", "shape": [26]},
            policy_sha256=self.policy_sha,
            runtime_policy_sha256=self.runtime_sha,
            teacher_sha256=self.teacher_sha,
            renderer_sha256="4" * 64,
            beta_sha256="5" * 64,
            config_sha256="6" * 64,
        )
        endpoints = []
        for rank in (0, 1):
            endpoint = self.root / f"endpoint-{rank}.json"
            endpoint.write_text("{}\n", encoding="utf-8")
            endpoints.append(endpoint)
        self.endpoint_manifest = self.root / "endpoints.txt"
        self.endpoint_manifest.write_text(
            "".join(f"{path}\n" for path in endpoints), encoding="utf-8"
        )
        self.anchors = self.root / "anchors"
        self.anchors.mkdir()
        (self.anchors / "anchor.npz").write_bytes(b"anchor")
        self.run_dir = self.root / "run"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _environment(self) -> dict[str, str]:
        return {
            **os.environ,
            "DRY_RUN": "1",
            "GR00T_PYTHON": "/usr/bin/python3",
            "ROUND_DAGGER_LOGICAL_ENV_COUNT": "64",
            "ROUND_DAGGER_STEPS_PER_ENVIRONMENT": "300",
            "ROUND_DAGGER_EXPECTED_TRANSITIONS": "19200",
            "ROUND_DAGGER_COHORT_DIR": str(self.cohort),
            "ROUND_DAGGER_CAMPAIGN_ID": "pair-launcher-test",
            "ROUND_DAGGER_MACRO_ROUND": "0",
            "ROUND_DAGGER_COHORT_INDEX": "0",
            "ROUND_DAGGER_POLICY_CHECKPOINT": str(self.checkpoint),
            "ROUND_DAGGER_POLICY_SHA256": self.policy_sha,
            "ROUND_DAGGER_RUNTIME_POLICY_SHA256": self.runtime_sha,
            "ROUND_DAGGER_TEACHER_CHECKPOINT": str(self.teacher),
            "ROUND_DAGGER_TEACHER_SHA256": self.teacher_sha,
            "ROUND_DAGGER_RENDERER_SHA256": "4" * 64,
            "ROUND_DAGGER_BETA": "0.5",
            "ROUND_DAGGER_BETA_SHA256": "5" * 64,
            "ROUND_DAGGER_CONFIG_SHA256": "6" * 64,
            "ROUND_DAGGER_POLICY_SPEC": str(self.spec),
            "ROUND_DAGGER_POLICY_MANIFEST": str(self.manifest),
            "ROUND_DAGGER_POLICY_SPEC_SHA256": artifacts.sha256_file(self.spec),
            "ROUND_DAGGER_POLICY_MANIFEST_SHA256": artifacts.sha256_file(
                self.manifest
            ),
            "ROUND_DAGGER_MIN_ANCHORS": "1",
            "IMAGE3F_ENDPOINT_MANIFEST": str(self.endpoint_manifest),
            "ANCHORS": str(self.anchors),
            "TRAIN_RUN_DIR": str(self.run_dir),
            "ROUND_DAGGER_ACTOR_RUN_DIR": str(self.run_dir),
            "ROUND_DAGGER_WANDB_ENABLED": "0",
            # Prove the pair launcher removes accidental Torch topology.
            "RANK": "9",
            "WORLD_SIZE": "10",
            "LOCAL_RANK": "9",
            "MASTER_ADDR": "should-not-propagate",
        }

    def test_dry_run_launches_exact_independent_ranks_and_never_seals(self) -> None:
        result = subprocess.run(
            ["bash", str(PAIR_SCRIPT)],
            env=self._environment(),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("actor_rank=0 actor_world_size=2", result.stdout)
        self.assertIn("actor_rank=1 actor_world_size=2", result.stdout)
        self.assertIn("no_torch_distributed=1", result.stdout)
        self.assertNotIn("--multi_gpu", result.stdout)
        self.assertFalse((self.cohort / "seal.json").exists())

    def test_one_failed_executor_is_reaped_without_sealing_partial_pair(self) -> None:
        executor = self.root / "executor"
        executor.write_text(
            "#!/usr/bin/env bash\n"
            "rank=$1\nscript=$2\n"
            "if [ \"$rank\" = 1 ]; then exit 17; fi\n"
            "exec /usr/bin/env bash \"$script\"\n",
            encoding="utf-8",
        )
        executor.chmod(0o755)
        environment = self._environment()
        environment["ROUND_DAGGER_ACTOR_PAIR_EXECUTOR"] = str(executor)
        result = subprocess.run(
            ["bash", str(PAIR_SCRIPT)],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("rank0=0 rank1=17", result.stderr)
        self.assertIn("cohort was not sealed", result.stderr)
        self.assertFalse((self.cohort / "seal.json").exists())

    def test_wandb_enabled_rejects_rank_primitive_without_sidecar(self) -> None:
        environment = self._environment()
        environment["ROUND_DAGGER_WANDB_ENABLED"] = "1"
        environment["ROUND_DAGGER_ACTOR_PAIR_RANK_SCRIPT"] = str(
            ROOT / "scripts/round_dagger/run_actor_cohort.sh"
        )
        result = subprocess.run(
            ["bash", str(PAIR_SCRIPT)],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires the repository W&B rank primitive", result.stderr)
        self.assertFalse((self.cohort / "seal.json").exists())


class ActorPairWandbRetryRecoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.cohort = self.root / "cohort"
        _cohort(self.cohort)
        # Reproduce the exact interrupted state: rank zero is immutable, rank
        # one is absent, and no joint seal has yet been published.
        artifacts.publish_rollout_shard(
            self.cohort,
            rank=0,
            rollout_id=0,
            part_index=0,
            arrays=_arrays(0),
            campaign_id="actor-pair-test",
            macro_round=0,
            cohort_index=0,
            purpose="train",
            **DIGESTS,
        )
        self.run_dir = self.root / "run"
        self.owner_log = self.root / "owner.log"
        self.drained = self.root / "sidecar-drained"
        self.publisher = self.root / "publish_rank.py"
        self.publisher.write_text(
            "import sys\n"
            f"sys.path.insert(0, {str(ROOT)!r})\n"
            "from pathlib import Path\n"
            "from round_dagger import artifacts\n"
            "from tests.test_round_dagger_actor_pair import DIGESTS, _arrays\n"
            "root = Path(sys.argv[1])\n"
            "rank = int(sys.argv[2])\n"
            "artifacts.publish_rollout_shard(\n"
            "    root, rank=rank, rollout_id=0, part_index=0, arrays=_arrays(rank),\n"
            "    campaign_id='actor-pair-test', macro_round=0, cohort_index=0,\n"
            "    purpose='train', **DIGESTS)\n",
            encoding="utf-8",
        )
        self.executor = self.root / "executor"
        self.executor.write_text(
            "#!/usr/bin/env bash\n"
            "set -Eeuo pipefail\n"
            "rank=$1\n"
            "rank_script=$2\n"
            f"printf '%s %s %s %s\\n' \"$rank\" \"$ROUND_DAGGER_WANDB_SIDECAR_OWNER\" \"${{ROUND_DAGGER_WANDB_PAIR_LIFECYCLE:-}}\" \"$rank_script\" >> {self.owner_log!s}\n"
            "[[ \"$rank\" == 1 ]]\n"
            "[[ \"$ROUND_DAGGER_WANDB_SIDECAR_OWNER\" == 1 ]]\n"
            "[[ \"$ROUND_DAGGER_WANDB_PAIR_LIFECYCLE\" == 1 ]]\n"
            f"[[ \"$rank_script\" == {str(WANDB_RANK_SCRIPT)!r} ]]\n"
            f"/usr/bin/python3 {self.publisher!s} \"$ROUND_DAGGER_COHORT_DIR\" \"$rank\"\n"
            "printf '0\\n' > \"$ROUND_DAGGER_WANDB_ACTOR_DONE_MARKER\"\n"
            "finish=\"$TRAIN_RUN_DIR/$ROUND_DAGGER_WANDB_PAIR_FINISH_MARKER\"\n"
            "for _ in $(seq 1 100); do\n"
            "  [ ! -f \"$finish\" ] || break\n"
            "  sleep 0.05\n"
            "done\n"
            "[ -f \"$finish\" ] && [ ! -L \"$finish\" ]\n"
            "[ ! -e \"$ROUND_DAGGER_COHORT_DIR/seal.json\" ]\n"
            f"printf 'drained\\n' > {self.drained!s}\n",
            encoding="utf-8",
        )
        self.executor.chmod(0o755)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _environment(self) -> dict[str, str]:
        return {
            **os.environ,
            "DRY_RUN": "0",
            "ROUND_DAGGER_LOGICAL_ENV_COUNT": "64",
            "ROUND_DAGGER_STEPS_PER_ENVIRONMENT": "300",
            "ROUND_DAGGER_EXPECTED_TRANSITIONS": "19200",
            "ROUND_DAGGER_COHORT_DIR": str(self.cohort),
            "TRAIN_RUN_DIR": str(self.run_dir),
            "ROUND_DAGGER_ACTOR_RUN_DIR": str(self.run_dir),
            "ROUND_DAGGER_ACTOR_PAIR_EXECUTOR": str(self.executor),
            "ROUND_DAGGER_WANDB_ENABLED": "1",
        }

    def test_rank_one_owns_and_drains_sidecar_before_retry_seals(self) -> None:
        result = subprocess.run(
            ["bash", str(PAIR_SCRIPT)],
            env=self._environment(),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("actor rank 0 is already exactly complete", result.stdout)
        self.assertIn("ACTOR_PAIR_SEALED", result.stdout)
        owner_lines = self.owner_log.read_text(encoding="utf-8").splitlines()
        self.assertEqual(owner_lines, [f"1 1 1 {WANDB_RANK_SCRIPT}"])
        self.assertEqual(self.drained.read_text(encoding="utf-8"), "drained\n")
        seal = self.cohort / "seal.json"
        self.assertTrue(seal.is_file())
        self.assertLessEqual(self.drained.stat().st_mtime_ns, seal.stat().st_mtime_ns)
        self.assertTrue(artifacts.verify_cohort(self.cohort)["sealed"])

    def test_both_complete_but_unsealed_fails_without_telemetry_owner(self) -> None:
        artifacts.publish_rollout_shard(
            self.cohort,
            rank=1,
            rollout_id=0,
            part_index=0,
            arrays=_arrays(1),
            campaign_id="actor-pair-test",
            macro_round=0,
            cohort_index=0,
            purpose="train",
            **DIGESTS,
        )
        result = subprocess.run(
            ["bash", str(PAIR_SCRIPT)],
            env=self._environment(),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("no incomplete actor", result.stderr)
        self.assertFalse((self.cohort / "seal.json").exists())
        self.assertFalse(self.owner_log.exists())


class ActorWandbExplicitOwnerFlagTest(unittest.TestCase):
    def _run(self, root: Path, *, rank: str, owner: str) -> subprocess.CompletedProcess:
        run_dir = root / "run"
        return subprocess.run(
            ["bash", str(WANDB_RANK_SCRIPT)],
            env={
                **os.environ,
                "ROUND_DAGGER_ACTOR_RUN_DIR": str(run_dir),
                "TRAIN_RUN_DIR": str(run_dir),
                "ROUND_DAGGER_ACTOR_RANK": rank,
                "ROUND_DAGGER_ACTOR_WORLD_SIZE": "2",
                "ROUND_DAGGER_WANDB_ENABLED": "1",
                "ROUND_DAGGER_WANDB_SIDECAR_OWNER": owner,
            },
            text=True,
            capture_output=True,
            check=False,
        )

    def test_rank_one_can_be_owner_and_rank_zero_can_be_nonowner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            rank_one = self._run(root, rank="1", owner="1")
            rank_zero = self._run(root, rank="0", owner="0")
        self.assertNotEqual(rank_one.returncode, 0)
        self.assertIn("ROUND_DAGGER_WANDB_PYTHON", rank_one.stderr)
        self.assertNotEqual(rank_zero.returncode, 0)
        self.assertIn("GR00T_PYTHON", rank_zero.stderr)
        self.assertNotIn("ROUND_DAGGER_WANDB_PYTHON", rank_zero.stderr)

    def test_owner_flag_is_strict_boolean(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = self._run(Path(temporary).resolve(), rank="1", owner="rank1")
        self.assertEqual(result.returncode, 2)
        self.assertIn("owner flag must be exactly 0 or 1", result.stderr)


class ActorRankDockerIsolationTest(unittest.TestCase):
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
            "ROUND_DAGGER_ACTOR_RANK": "0",
            "ROUND_DAGGER_ACTOR_WORLD_SIZE": "2",
            "ROUND_DAGGER_CAMPAIGN_ID": "docker-isolation-test",
            "ROUND_DAGGER_MACRO_ROUND": "0",
            "ROUND_DAGGER_COHORT_INDEX": "0",
            "ROUND_DAGGER_DOCKER_BIN": str(self.fake_docker),
            "ROUND_DAGGER_RTX_IMAGE": "actor:test",
            "ROUND_DAGGER_ACTOR_DOCKER_IPC": ipc,
            "ROUND_DAGGER_ACTOR_DOCKER_SHM_SIZE": "32g",
            "ROUND_DAGGER_ACTOR_DOCKER_GPU_IDS": "4,7",
            "ROUND_DAGGER_ACTOR_DOCKER_MOUNT_ROOTS": str(self.root),
            "NETRC": str(self.root / ".netrc"),
            "WANDB_API_KEY": "must-not-be-forwarded",
        }

    def test_executor_uses_distinct_gpu_private_ipc_and_large_shm(self) -> None:
        result = subprocess.run(
            [
                "bash",
                str(DOCKER_EXECUTOR),
                "0",
                str(ROOT / "scripts/round_dagger/run_actor_cohort.sh"),
            ],
            env=self._environment(),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = self.argv_file.read_text(encoding="utf-8").splitlines()
        self.assertIn("--ipc=private", argv)
        self.assertNotIn("--ipc=host", argv)
        self.assertIn("--shm-size=32g", argv)
        self.assertIn('"device=4"', argv)
        self.assertIn("--network=host", argv)
        self.assertIn("NETRC", argv)
        self.assertNotIn("WANDB_API_KEY", argv)
        self.assertNotIn("must-not-be-forwarded", argv)

    def test_host_ipc_is_rejected_before_docker_is_invoked(self) -> None:
        result = subprocess.run(
            [
                "bash",
                str(DOCKER_EXECUTOR),
                "0",
                str(ROOT / "scripts/round_dagger/run_actor_cohort.sh"),
            ],
            env=self._environment(ipc="host"),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("host IPC is forbidden", result.stderr)
        self.assertFalse(self.argv_file.exists())

if __name__ == "__main__":
    unittest.main()
