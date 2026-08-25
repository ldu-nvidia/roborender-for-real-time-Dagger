from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from round_dagger import artifacts
from tests.policy_bundle_helpers import publish_policy_bundle


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "scripts/round_dagger/submit_oci_nrt_round_learner_and_wait.sh"
BATCH = ROOT / "scripts/round_dagger/oci_nrt_round_learner.sbatch"
CONTAINER_WRAPPER = ROOT / "scripts/round_dagger/run_oci_nrt_learner_container.sh"


class RemoteLearnerBridgeContractTest(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[dict[str, str], Path]:
        campaign = root / "campaign"
        policy_dir = campaign / "policies/policy_000000"
        snapshot_dir = campaign / "replay/snapshots"
        learner = campaign / "rounds/macro_round_000000/learner"
        policy_dir.mkdir(parents=True)
        snapshot_dir.mkdir(parents=True)
        checkpoint = policy_dir / "checkpoint.pt"
        checkpoint.write_bytes(b"strict-parent-checkpoint")
        runtime_sha = "2" * 64
        spec, manifest = publish_policy_bundle(checkpoint, runtime_sha)
        snapshot = snapshot_dir / "through_macro_round_000000.json"
        snapshot.write_text("{}\n", encoding="utf-8")
        checkpoint_sha = artifacts.sha256_file(checkpoint)
        environment = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "DRY_RUN": "1",
            "ROUND_DAGGER_PYTHON": sys.executable,
            "ROUND_DAGGER_REMOTE_SOURCE_ROOT": "/sealed/source",
            "ROUND_DAGGER_REMOTE_SOURCE_INVENTORY_SHA256": "4" * 64,
            "ROUND_DAGGER_REMOTE_HANDOFF_ROOT": str(root / "handoff"),
            "ROUND_DAGGER_CAMPAIGN_ROOT": str(campaign),
            "ROUND_DAGGER_CAMPAIGN_ID": "bridge-contract",
            "ROUND_DAGGER_MACRO_ROUND": "0",
            "ROUND_DAGGER_POLICY_CHECKPOINT": str(checkpoint),
            "ROUND_DAGGER_POLICY_SPEC": str(spec),
            "ROUND_DAGGER_POLICY_MANIFEST": str(manifest),
            "ROUND_DAGGER_POLICY_SPEC_SHA256": artifacts.sha256_file(spec),
            "ROUND_DAGGER_POLICY_MANIFEST_SHA256": artifacts.sha256_file(manifest),
            "ROUND_DAGGER_POLICY_SHA256": checkpoint_sha,
            "ROUND_DAGGER_RUNTIME_POLICY_SHA256": runtime_sha,
            "ROUND_DAGGER_REPLAY_SNAPSHOT": str(snapshot),
            "ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256": artifacts.sha256_file(snapshot),
            "ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256": "3" * 64,
            "ROUND_DAGGER_LEARNER_OUTPUT_DIR": str(learner),
            "ROUND_DAGGER_LEARNER_RESULT": str(
                learner / "learner_result_round_000000.json"
            ),
            "ROUND_DAGGER_LEARNER_CODE_SHA256": "4" * 64,
            "ROUND_DAGGER_LEARNER_CONFIG_SHA256": "5" * 64,
            "ROUND_DAGGER_LEARNER_MODEL_LINEAGE_SHA256": checkpoint_sha,
            "ROUND_DAGGER_BASE_POLICY_SPEC": str(spec),
            "ROUND_DAGGER_TARGET_ENVIRONMENTS": "256",
            "ROUND_DAGGER_COHORT_COUNT": "4",
            "ROUND_DAGGER_ENVIRONMENTS_PER_COHORT": "64",
            "ROUND_DAGGER_STEPS_PER_ENVIRONMENT": "300",
            "ROUND_DAGGER_TRANSITIONS_PER_MACRO_ROUND": "76800",
        }
        return environment, root / "handoff"

    def _run(self, environment: dict[str, str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(BRIDGE)],
            cwd=cwd,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    def test_dry_run_validates_real_bundle_without_mutation_or_remote_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment, handoff = self._fixture(root)
            # Deliberately run from outside the repository; the hook must set
            # its own import root instead of depending on coordinator cwd.
            completed = self._run(environment, cwd=root)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("gpus=8 updates=500 save_every=5", completed.stdout)
            self.assertIn("OCI_NRT_ROUND_LEARNER_BRIDGE_DRY_RUN_OK", completed.stdout)
            self.assertFalse(handoff.exists())

    def test_dry_run_rejects_topology_and_sidecar_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment, _ = self._fixture(root)
            bad_topology = dict(environment, ROUND_DAGGER_ENVIRONMENTS_PER_COHORT="32")
            completed = self._run(bad_topology, cwd=root)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("production topology", completed.stderr)

            bad_sidecar = dict(environment, ROUND_DAGGER_POLICY_SPEC_SHA256="f" * 64)
            completed = self._run(bad_sidecar, cwd=root)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("policy spec SHA mismatch", completed.stderr)

    def test_scripts_pin_restart_security_and_topology_contracts(self) -> None:
        bridge = BRIDGE.read_text(encoding="utf-8")
        batch = BATCH.read_text(encoding="utf-8")
        container_wrapper = CONTAINER_WRAPPER.read_text(encoding="utf-8")
        self.assertNotIn('exports="ALL"', bridge)
        self.assertNotIn("WANDB_API_KEY", bridge + batch)
        self.assertIn("flock -n 9", bridge)
        self.assertIn("remote-complete", bridge)
        self.assertIn("adopt_remote_success", bridge)
        self.assertIn("learner_completion_attestation.py", bridge)
        self.assertIn("refusing rerun or adoption", bridge)
        self.assertIn("--completion-attestation", bridge)
        self.assertIn("remote_learner_success.py\" verify", bridge)
        self.assertIn("remote_completion_state", bridge)
        self.assertIn("cannot determine remote learner completion state", bridge)
        self.assertIn("remote_learner_success.py\" verify", bridge)
        self.assertIn("LOCAL_SUCCESS_MARKER", bridge)
        self.assertIn("remote learner inventory changed across bridge restart", bridge)
        incoming_verify = bridge.index('--result "$INCOMING_RESULT"')
        atomic_publish = bridge.index('publish --stage "$INCOMING_OUTPUT"')
        self.assertLess(incoming_verify, atomic_publish)
        self.assertIn("PREEMPTED|NODE_FAIL|BOOT_FAIL|TIMEOUT", bridge)
        for non_retryable in ("CANCELLED", "FAILED", "OUT_OF_MEMORY"):
            self.assertNotIn(f"PREEMPTED|{non_retryable}", bridge)
        self.assertIn("--gpus-per-node=8", bridge)
        self.assertIn(
            "PARTITION=${ROUND_DAGGER_REMOTE_PARTITION:-batch_singlenode}",
            bridge,
        )
        self.assertIn("partition must be batch_singlenode", bridge)
        self.assertIn("--time=04:00:00", bridge)
        self.assertNotIn("--time=08:00:00", bridge)
        self.assertIn("/cm/shared/apps/slurm/current/bin", bridge)
        self.assertIn("--gpus-per-task=8", batch)
        self.assertIn("/round_dagger_job", batch)
        self.assertIn("ROUND_DAGGER_WANDB_RUNTIME_DIR=/round_dagger_job/wandb", batch)
        self.assertIn(":ro", batch)
        self.assertIn("/round_dagger_job/wandb", container_wrapper)
        self.assertNotIn('$OUTPUT/wandb', container_wrapper)
        self.assertIn('"256:4:64:300:76800"', bridge)
        self.assertIn('"256:4:64:300:76800"', batch)

    def test_shells_parse_and_are_executable(self) -> None:
        for path in (BRIDGE, BATCH, CONTAINER_WRAPPER):
            self.assertTrue(os.access(path, os.X_OK), path)
            completed = subprocess.run(
                ["bash", "-n", str(path)],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
