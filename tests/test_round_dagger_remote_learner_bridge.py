from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from round_dagger import artifacts
from round_dagger.learner_identity import (
    learner_identity_sha256,
    learner_wandb_run_id,
)
from scripts.round_dagger import remote_campaign_transfer as transfer
from tests.policy_bundle_helpers import publish_policy_bundle


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "scripts/round_dagger/submit_oci_nrt_round_learner_and_wait.sh"
BATCH = ROOT / "scripts/round_dagger/oci_nrt_round_learner.sbatch"
CONTAINER_WRAPPER = ROOT / "scripts/round_dagger/run_oci_nrt_learner_container.sh"


class RemoteLearnerBridgeContractTest(unittest.TestCase):
    @staticmethod
    def _bind_identity(environment: dict[str, str]) -> dict[str, str]:
        identity = learner_identity_sha256(
            campaign_id=environment["ROUND_DAGGER_CAMPAIGN_ID"],
            macro_round=int(environment["ROUND_DAGGER_MACRO_ROUND"]),
            learner_code_sha256=environment["ROUND_DAGGER_LEARNER_CODE_SHA256"],
            learner_config_sha256=environment[
                "ROUND_DAGGER_LEARNER_CONFIG_SHA256"
            ],
            replay_snapshot_file_sha256=environment[
                "ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256"
            ],
            replay_snapshot_sha256=environment[
                "ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256"
            ],
            parent_checkpoint_sha256=environment["ROUND_DAGGER_POLICY_SHA256"],
            parent_runtime_policy_sha256=environment[
                "ROUND_DAGGER_RUNTIME_POLICY_SHA256"
            ],
            runtime_container_sha256=environment[
                "ROUND_DAGGER_REMOTE_CONTAINER_SHA256"
            ],
            learner_contract_sha256=environment[
                "ROUND_DAGGER_LEARNER_CONTRACT_SHA256"
            ],
        )
        environment["ROUND_DAGGER_LEARNER_IDENTITY_SHA256"] = identity
        environment["ROUND_DAGGER_WANDB_RUN_ID"] = learner_wandb_run_id(
            campaign_id=environment["ROUND_DAGGER_CAMPAIGN_ID"],
            macro_round=int(environment["ROUND_DAGGER_MACRO_ROUND"]),
            learner_identity_sha256=identity,
        )
        return environment

    def _fixture(self, root: Path) -> tuple[dict[str, str], Path]:
        campaign = root / "campaign"
        policy_dir = campaign / "policies/policy_000000"
        snapshot_dir = campaign / "replay/snapshots"
        learner = campaign / "rounds/macro_round_000000/learner"
        policy_dir.mkdir(parents=True)
        snapshot_dir.mkdir(parents=True)
        checkpoint = policy_dir / "checkpoint.pt"
        checkpoint.write_bytes(b"strict-parent-checkpoint")
        spec, manifest = publish_policy_bundle(checkpoint, "2" * 64)
        runtime_sha = artifacts.load_strict_json(spec)[
            "runtime_policy_sha256"
        ]
        snapshot = snapshot_dir / "through_macro_round_000000.json"
        snapshot.write_text("{}\n", encoding="utf-8")
        learner_contract = campaign / "learner_contract.json"
        learner_contract.write_text(
            json.dumps({"schema": "fixture.campaign-owned-learner.v1"}) + "\n",
            encoding="utf-8",
        )
        learner_contract.chmod(0o444)
        wandb = campaign / "actors/cohort_00/.wandb_media"
        wandb.mkdir(parents=True)
        (wandb / "latest").symlink_to(snapshot)
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
            "ROUND_DAGGER_LEARNER_CONTRACT": str(learner_contract),
            "ROUND_DAGGER_LEARNER_CONTRACT_SHA256": artifacts.sha256_file(
                learner_contract
            ),
            "ROUND_DAGGER_REMOTE_CONTAINER_SHA256": "7" * 64,
            "ROUND_DAGGER_LEARNER_MODEL_LINEAGE_SHA256": checkpoint_sha,
            "ROUND_DAGGER_BASE_POLICY_SPEC": str(spec),
            "ROUND_DAGGER_TARGET_ENVIRONMENTS": "256",
            "ROUND_DAGGER_COHORT_COUNT": "4",
            "ROUND_DAGGER_ENVIRONMENTS_PER_COHORT": "64",
            "ROUND_DAGGER_STEPS_PER_ENVIRONMENT": "512",
            "ROUND_DAGGER_TRANSITIONS_PER_MACRO_ROUND": "131072",
        }
        return self._bind_identity(environment), root / "handoff"

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
            self.assertIn("gpus=8 updates=1200 save_every=5", completed.stdout)
            self.assertIn("walltime=03:45:00", completed.stdout)
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

            bad_wandb_run = dict(
                environment, ROUND_DAGGER_WANDB_RUN_ID="reused-unbound-run"
            )
            completed = self._run(bad_wandb_run, cwd=root)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "not bound to the exact learner identity",
                completed.stderr,
            )

            bad_contract_path = dict(
                environment,
                ROUND_DAGGER_LEARNER_CONTRACT=str(root / "other.json"),
            )
            completed = self._run(bad_contract_path, cwd=root)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("canonical campaign-owned", completed.stderr)

            contract = Path(environment["ROUND_DAGGER_LEARNER_CONTRACT"])
            contract.chmod(0o400)
            completed = self._run(environment, cwd=root)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("exactly 0444", completed.stderr)

    def test_trusted_dispatch_cannot_disable_mandatory_heldout100_gate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment, _ = self._fixture(root)

            # Untrusted dry-run validation retains its explicit opt-out
            # behavior and therefore remains usable for dependency-light
            # control-plane tests.
            untrusted = dict(
                environment,
                ROUND_DAGGER_REQUIRE_HELDOUT100_GATE="0",
            )
            completed = self._run(untrusted, cwd=root)
            self.assertEqual(completed.returncode, 0, completed.stderr)

            trusted = dict(
                environment,
                ROUND_DAGGER_TRUSTED_TRAIN_DISPATCH="1",
                ROUND_DAGGER_COORDINATOR_SOURCE_ROOT=str(ROOT),
                ROUND_DAGGER_TRUSTED_HOOK_SOURCE_INVENTORY_SHA256="4" * 64,
            )
            inherited_disable = dict(
                trusted,
                ROUND_DAGGER_REQUIRE_HELDOUT100_GATE="0",
            )
            completed = self._run(inherited_disable, cwd=root)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "trusted production dispatch cannot disable the heldout100 gate",
                completed.stderr,
            )

            # With no override, trusted dispatch still forces the gate.  This
            # minimal fixture intentionally has no held-out fields, so it must
            # fail before any remote access rather than silently dry-run the
            # production learner.
            completed = self._run(trusted, cwd=root)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn(
                "trusted production dispatch requires fixed-256 held-out evaluation",
                completed.stderr,
            )

    def test_maximum_campaign_token_keeps_wandb_run_id_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            environment, _ = self._fixture(root)
            environment["ROUND_DAGGER_CAMPAIGN_ID"] = "c" * 128
            self._bind_identity(environment)
            completed = self._run(environment, cwd=root)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            run_id = environment["ROUND_DAGGER_WANDB_RUN_ID"]
            self.assertLessEqual(len(run_id), 128)
            self.assertIn(f"wandb_run_id={run_id}", completed.stdout)

    def test_identity_namespaces_do_not_collide_and_identical_retry_resumes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, handoff_root = self._fixture(root)
            second = self._bind_identity(
                dict(first, ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256="8" * 64)
            )
            third = self._bind_identity(
                dict(first, ROUND_DAGGER_REMOTE_CONTAINER_SHA256="9" * 64)
            )

            first_run = self._run(first, cwd=root)
            identical_retry = self._run(first, cwd=root)
            second_run = self._run(second, cwd=root)
            third_run = self._run(third, cwd=root)
            for completed in (first_run, identical_retry, second_run, third_run):
                self.assertEqual(completed.returncode, 0, completed.stderr)

            first_identity = first["ROUND_DAGGER_LEARNER_IDENTITY_SHA256"]
            second_identity = second["ROUND_DAGGER_LEARNER_IDENTITY_SHA256"]
            third_identity = third["ROUND_DAGGER_LEARNER_IDENTITY_SHA256"]
            first_handoff = (
                handoff_root
                / "bridge-contract/macro_round_000000"
                / f"identity_{first_identity}"
            )
            second_handoff = (
                handoff_root
                / "bridge-contract/macro_round_000000"
                / f"identity_{second_identity}"
            )
            self.assertIn(f"identity={first_identity}", first_run.stdout)
            self.assertIn(f"local_handoff={first_handoff}", first_run.stdout)
            self.assertIn(
                f"wandb_run_id=rrd-m000000-id{first_identity}",
                first_run.stdout,
            )
            self.assertIn(f"job_name=rr-rd-{first_identity}", first_run.stdout)
            self.assertIn(f"identity={second_identity}", second_run.stdout)
            self.assertIn(f"local_handoff={second_handoff}", second_run.stdout)
            self.assertIn(
                f"wandb_run_id=rrd-m000000-id{second_identity}",
                second_run.stdout,
            )
            self.assertEqual(first_run.stdout, identical_retry.stdout)
            self.assertNotEqual(first_handoff, second_handoff)
            self.assertNotEqual(second_identity, third_identity)
            self.assertNotEqual(first_identity, third_identity)
            self.assertIn(f"identity={third_identity}", third_run.stdout)
            self.assertFalse(handoff_root.exists())

    def test_revoked_allocation_reuses_semantic_identity_and_restart_frontier(
        self,
    ) -> None:
        bridge = BRIDGE.read_text(encoding="utf-8")
        learner = (
            ROOT / "scripts/round_dagger/run_h100_learner.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "retryable='^(PREEMPTED|NODE_FAIL|BOOT_FAIL|TIMEOUT|REVOKED)$'",
            bridge,
        )
        self.assertIn(
            "LEARNER_NAMESPACE=identity_${ROUND_DAGGER_LEARNER_IDENTITY_SHA256}",
            bridge,
        )
        self.assertIn(
            'JOB_NAME="rr-rd-${ROUND_DAGGER_LEARNER_IDENTITY_SHA256}"',
            bridge,
        )
        retry_loop = bridge.split("retryable=", 1)[1]
        self.assertNotIn("LEARNER_NAMESPACE=", retry_loop)
        self.assertNotIn("REMOTE_OUTPUT=", retry_loop)
        self.assertIn("$state =~ $retryable", retry_loop)
        self.assertIn('RESTART="$ROUND_DAGGER_LEARNER_OUTPUT_DIR/restart.pt"', learner)
        self.assertIn('COMMAND+=(--resume-from "$RESUME_PATH")', learner)
        self.assertIn("--save-every", learner)

    def test_scripts_pin_restart_security_and_topology_contracts(self) -> None:
        bridge = BRIDGE.read_text(encoding="utf-8")
        batch = BATCH.read_text(encoding="utf-8")
        container_wrapper = CONTAINER_WRAPPER.read_text(encoding="utf-8")
        explicit_exports = bridge.split("export_names=(", 1)[1].split(")", 1)[0]
        self.assertNotIn('exports="ALL"', bridge)
        self.assertIn("ROUND_DAGGER_CAMPAIGN_ROOT", explicit_exports)
        self.assertIn(
            'ROUND_DAGGER_CAMPAIGN_ROOT == "$ROUND_DAGGER_LOCAL_CAMPAIGN_ROOT"',
            batch,
        )
        self.assertNotIn("WANDB_API_KEY", bridge + batch)
        self.assertIn("flock -n 9", bridge)
        self.assertIn("remote-complete", bridge)
        self.assertNotIn("adopt_remote_success", bridge)
        self.assertIn("learner_completion_attestation.py", bridge)
        self.assertIn("attested-result", bridge)
        self.assertIn("result-only", bridge)
        self.assertIn("submitting container finalization", bridge)
        self.assertIn("submitting finalization recovery", bridge)
        self.assertIn("container finalization incomplete", bridge)
        self.assertIn("$state == FAILED || $state =~ $retryable", bridge)
        self.assertIn("remote_restart_state", bridge)
        self.assertIn("FAILED after durable restart", bridge)
        self.assertIn("$state == FAILED && $completion_state == empty", bridge)
        self.assertIn("--completion-attestation", bridge)
        self.assertIn("remote_learner_success.py\" verify", bridge)
        self.assertIn("remote_completion_state", bridge)
        self.assertIn("cannot determine remote learner completion state", bridge)
        self.assertIn("remote_learner_success.py\" verify", bridge)
        self.assertIn("LOCAL_SUCCESS_MARKER", bridge)
        self.assertIn("remote learner inventory changed across bridge restart", bridge)
        self.assertIn("projected remote learner contract SHA mismatch", bridge)
        self.assertIn(
            '"$REMOTE_CAMPAIGN/learner_contract.json"',
            bridge,
        )
        incoming_verify = bridge.index('--result "$INCOMING_RESULT"')
        atomic_publish = bridge.index('publish --stage "$INCOMING_OUTPUT"')
        self.assertLess(incoming_verify, atomic_publish)
        self.assertIn(
            "retryable='^(PREEMPTED|NODE_FAIL|BOOT_FAIL|TIMEOUT|REVOKED)$'",
            bridge,
        )
        for non_retryable in ("CANCELLED", "FAILED", "OUT_OF_MEMORY"):
            self.assertNotIn(f"PREEMPTED|{non_retryable}", bridge)
        self.assertIn("--gpus-per-node=8", bridge)
        self.assertIn(
            "PARTITION=${ROUND_DAGGER_REMOTE_PARTITION:-batch_singlenode}",
            bridge,
        )
        self.assertIn("partition must be batch_singlenode", bridge)
        self.assertIn("LEARNER_WALLTIME=03:45:00", bridge)
        self.assertIn('--time="$LEARNER_WALLTIME"', bridge)
        self.assertNotIn("--time=04:00:00", bridge)
        self.assertNotIn("--time=08:00:00", bridge)
        self.assertIn("#SBATCH --time=03:45:00", batch)
        self.assertNotIn("#SBATCH --time=04:00:00", batch)
        self.assertIn("/cm/shared/apps/slurm/current/bin", bridge)
        self.assertIn("--gpus-per-task=8", batch)
        self.assertIn("/round_dagger_job", batch)
        self.assertIn("ROUND_DAGGER_WANDB_RUNTIME_DIR=/round_dagger_job/wandb", batch)
        self.assertIn(":ro", batch)
        self.assertIn("/round_dagger_job/wandb", container_wrapper)
        self.assertNotIn('$OUTPUT/wandb', container_wrapper)
        self.assertNotIn("python3", batch)
        self.assertNotIn("PYTHONPATH=", batch)
        self.assertNotIn("from round_dagger", batch)
        self.assertIn("validate_policy_bundle_identity", container_wrapper)
        self.assertIn("verify_sealed_source_snapshot.py", container_wrapper)
        self.assertIn("remote_campaign_transfer.py", container_wrapper)
        self.assertIn("learner_completion_attestation.py\" adopt", container_wrapper)
        self.assertIn("remote_learner_success.py\" verify", container_wrapper)
        self.assertIn("ROUND_DAGGER_CONTAINER_SUCCESS_MARKER", batch)
        self.assertIn('"256:4:64:512:131072"', bridge)
        self.assertIn('"256:4:64:512:131072"', batch)
        self.assertIn("LEARNER_NAMESPACE=identity_", bridge)
        self.assertIn('JOB_NAME="rr-rd-${ROUND_DAGGER_LEARNER_IDENTITY_SHA256}"', bridge)
        self.assertIn(
            "LEARNER_NAMESPACE=identity_${ROUND_DAGGER_LEARNER_IDENTITY_SHA256}",
            bridge,
        )
        self.assertIn('$REMOTE_OUTPUT/restart.pt', bridge)
        self.assertIn("restart.pt lineage", bridge)
        self.assertIn("learner_run_identity.v2", bridge)
        self.assertIn("runtime_container_sha256=", bridge)
        self.assertIn("learner_contract_sha256=", bridge)
        self.assertIn(
            "$ROUND_DECIMAL == 0 ]] && ((HELDOUT100_REQUIRED))",
            bridge,
        )
        self.assertIn(
            "HELDOUT100_REQUIRED && ! HELDOUT_ENABLED", bridge
        )
        self.assertIn(
            "trusted production dispatch requires fixed-256 held-out evaluation",
            bridge,
        )
        self.assertIn("HELDOUT100_REQUIRED=1", bridge)
        self.assertIn(
            "trusted production dispatch cannot disable the heldout100 gate",
            bridge,
        )
        self.assertIn("HELDOUT100_BINDING_SHA256", bridge)
        self.assertNotIn(
            "HELDOUT100_PRODUCTION_CONTRACT_SHA256=e4a1d23", bridge
        )
        self.assertIn("submit_oci_nrt_h100_heldout100.sh", bridge)
        self.assertIn("BEGIN_HELDOUT100_GATE_EVIDENCE_VERIFIER", bridge)
        self.assertIn("heldout100 gate evidence adopted", bridge)
        self.assertIn("non-adoptable partial state", bridge)
        self.assertIn("wait_for_heldout100_remote_completion", bridge)
        empty_case = bridge.split('            empty)\n', 1)[1].split(
            '                ;;', 1
        )[0]
        self.assertIn("submit_oci_nrt_h100_heldout100.sh", empty_case)
        self.assertIn('if [[ $smoke_state == active:* ]]', empty_case)
        self.assertIn("wait_for_heldout100_remote_completion", empty_case)
        self.assertIn("sacct", bridge)
        self.assertIn("evidence_state=$smoke_state", bridge)
        self.assertIn("HELDOUT100_REMOTE_CAMPAIGN", bridge)
        self.assertIn("isolated heldout100 campaign projection", bridge)
        self.assertIn('--link-dest="$REMOTE_CAMPAIGN"', bridge)
        self.assertIn(
            'ROUND_DAGGER_HELDOUT100_REMOTE_CAMPAIGN_ROOT="$HELDOUT100_REMOTE_CAMPAIGN"',
            bridge,
        )
        gate = bridge.index("if ((HELDOUT100_GATE_ENABLED)); then", bridge.index("BEGIN_HELDOUT100_GATE_EVIDENCE_VERIFIER"))
        production_submit = bridge.index("sbatch --parsable", gate)
        self.assertLess(gate, production_submit)
        self.assertNotIn("rm -rf", bridge)

    def test_heldout100_gate_evidence_adoption_is_identity_bound(self) -> None:
        bridge = BRIDGE.read_text(encoding="utf-8")
        marked = bridge.split(
            "# BEGIN_HELDOUT100_GATE_EVIDENCE_VERIFIER\n", 1
        )[1].split("# END_HELDOUT100_GATE_EVIDENCE_VERIFIER", 1)[0]
        verifier = marked.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
        sha = {
            "code": "1" * 64,
            "config": "2" * 64,
            "identity": "3" * 64,
            "validation": "4" * 64,
            "binding": "f" * 64,
            "production": "5" * 64,
            "replay_file": "6" * 64,
            "replay": "7" * 64,
            "parent": "8" * 64,
            "runtime": "9" * 64,
            "container": "a" * 64,
            "heldout_file": "b" * 64,
            "heldout": "c" * 64,
        }
        run_id = (
            f"rrd-m000000-id{sha['identity']}-b{sha['binding'][:32]}"
        )
        heldout_path = "/mnt/data2/ldu/roborender_round_dagger_production/dagger_prod_run1/heldout_snapshot.json"
        quality = {
            "initial_action_mse": 2.0,
            "final_action_mse": 1.0,
            "absolute_action_mse_reduction": 1.0,
            "relative_action_mse_reduction": 0.5,
            "action_mae_delta": -0.1,
            "action_rmse_delta": -0.1,
            "action_cosine_delta": 0.1,
        }
        loss = {
            "initial_loss": 2.0,
            "final_loss": 1.0,
            "absolute_loss_reduction": 1.0,
            "relative_loss_reduction": 0.5,
        }
        readback = {
            "schema": "fast_roborender.round_dagger.h100_heldout100_wandb_readback.v1",
            "status": "PASS",
            "account_email": "ldu@nvidia.com",
            "entity": "nvidia",
            "authorization_scope": "team",
            "team_member": True,
            "entity_is_team": True,
            "personal": False,
            "project": "roborender-round-dagger-validation",
            "wandb_run_id": run_id,
            "campaign_id": "dagger_prod_run1",
            "macro_round": 0,
            "learner_code_sha256": sha["code"],
            "learner_config_sha256": sha["config"],
            "learner_identity_sha256": sha["identity"],
            "runtime_container_sha256": sha["container"],
            "replay_snapshot_file_sha256": sha["replay_file"],
            "replay_snapshot_sha256": sha["replay"],
            "model_lineage_sha256": sha["parent"],
            "parent_runtime_policy_sha256": sha["runtime"],
            "learner_contract_sha256": sha["validation"],
            "validation_contract_sha256": sha["validation"],
            "validation_binding_sha256": sha["binding"],
            "production_contract_sha256": sha["production"],
            "heldout_snapshot_file_sha256": sha["heldout_file"],
            "heldout_snapshot_sha256": sha["heldout"],
            "heldout_case_count": 256,
            "heldout_optimizer_steps": [0, 100],
            "histories_identical": True,
            "quality_histories_identical": True,
            "quality_improvement_verified": True,
            "loss_improvement_verified": True,
            "strict_resume_verified": True,
            "single_canonical_wandb_run": True,
            "stale_wandb_history_policy": "fail_closed_no_adoption",
            "quality_improvement": quality,
            "loss_improvement": loss,
            "remote": {
                "state": "finished",
                "training_history_steps": list(range(1, 101)),
                "quality_history_steps": [0, 100],
                "quality_key_count": 49,
                "config_verified": True,
                "quality_improvement_verified": True,
            },
        }
        restart = {
            "schema": "fast_roborender.round_dagger.restart_equivalence.v1",
            "semantic_equivalent": True,
            "expected_completed_steps": 100,
            "uninterrupted_checkpoint": "/output/full/restart.pt",
            "uninterrupted_checkpoint_sha256": "d" * 64,
            "resumed_checkpoint": "/output/resumed/restart.pt",
            "resumed_checkpoint_sha256": "d" * 64,
            "results_equivalent": True,
            "uninterrupted_result": "/output/full/result.json",
            "uninterrupted_result_sha256": "e" * 64,
            "resumed_result": "/output/resumed/result.json",
            "resumed_result_sha256": "e" * 64,
        }
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "evidence"
            evidence.mkdir()
            readback_path = evidence / "heldout100_wandb_readback.json"
            restart_path = evidence / "restart_equivalence.json"
            readback_path.write_text(
                json.dumps(readback, sort_keys=True) + "\n", encoding="utf-8"
            )
            restart_path.write_text(
                json.dumps(restart, sort_keys=True) + "\n", encoding="utf-8"
            )
            attestation = {
                "schema": "fast_roborender.round_dagger.oci_nrt_h100_heldout100_validation.v1",
                "status": "PASS",
                "purpose": "live_h100_fixed256_step100_wandb_restart_gate",
                "campaign_id": "dagger_prod_run1",
                "allocation": {
                    "nodes": 1,
                    "gpus_per_node": 8,
                    "gpu_architecture": "H100-SM90",
                    "backend": "nccl",
                    "batch_size_per_rank": 1,
                    "global_batch_size": 8,
                },
                "optimizer_steps": 100,
                "intentional_stop_step": 50,
                "heldout_optimizer_steps": [0, 100],
                "heldout_case_count": 256,
                "heldout_snapshot_path": heldout_path,
                "heldout_snapshot_file_sha256": sha["heldout_file"],
                "heldout_snapshot_sha256": sha["heldout"],
                "validation_contract_sha256": sha["validation"],
                "validation_binding_sha256": sha["binding"],
                "production_target_contract_sha256": sha["production"],
                "learner_code_sha256": sha["code"],
                "learner_config_sha256": sha["config"],
                "learner_identity_sha256": sha["identity"],
                "runtime_container_sha256": sha["container"],
                "training_replay_snapshot_file_sha256": sha["replay_file"],
                "training_replay_snapshot_sha256": sha["replay"],
                "parent_checkpoint_sha256": sha["parent"],
                "parent_runtime_policy_sha256": sha["runtime"],
                "wandb_run_id": run_id,
                "wandb_quality_key_count": 49,
                "quality_improvement": quality,
                "loss_improvement": loss,
                "wandb_readback_sha256": artifacts.sha256_file(readback_path),
                "restart_report_sha256": artifacts.sha256_file(restart_path),
                "strict_resume_verified": True,
                "uninterrupted_equivalence_verified": True,
                "stale_wandb_history_policy": "fail_closed_no_adoption",
                "quality_improvement_verified": True,
                "loss_improvement_verified": True,
            }
            attestation_path = evidence / "success_attestation.json"
            attestation_path.write_text(
                json.dumps(attestation, sort_keys=True) + "\n", encoding="utf-8"
            )
            for path in evidence.iterdir():
                path.chmod(0o444)
            evidence.chmod(0o555)
            arguments = [
                str(evidence), "dagger_prod_run1", "0", sha["code"],
                sha["config"], sha["identity"], sha["validation"],
                sha["binding"], sha["production"], sha["replay_file"], sha["replay"],
                sha["parent"], sha["runtime"], sha["container"], heldout_path,
                sha["heldout_file"], sha["heldout"], run_id,
            ]
            valid = subprocess.run(
                [sys.executable, "-", *arguments], input=verifier, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            self.assertEqual(valid.returncode, 0, valid.stderr)
            self.assertIn("HELDOUT100_GATE_EVIDENCE_VERIFIED", valid.stdout)

            attestation_path.chmod(0o644)
            attestation["training_replay_snapshot_sha256"] = "f" * 64
            attestation_path.write_text(
                json.dumps(attestation, sort_keys=True) + "\n", encoding="utf-8"
            )
            attestation_path.chmod(0o444)
            tampered = subprocess.run(
                [sys.executable, "-", *arguments], input=verifier, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            self.assertNotEqual(tampered.returncode, 0)
            self.assertIn("training_replay_snapshot_sha256 differs", tampered.stderr)

    def test_container_launcher_supports_idempotent_finalization_recovery(self) -> None:
        launcher = (
            ROOT / "scripts/round_dagger/run_h100_learner.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("RECOVER_EXISTING_RESULT=1", launcher)
        self.assertIn("H100_LEARNER_FINALIZATION_RECOVERY", launcher)
        self.assertIn("existing learner result has no matching durable restart", launcher)
        self.assertIn("existing learner result restart is not at the final frontier", launcher)
        self.assertIn("H100_LEARNER_RECOVERY_CLEANUP", launcher)
        self.assertIn("recovered=attested", launcher)

    def test_batch_host_dry_run_does_not_require_python_or_numpy(self) -> None:
        batch = ROOT / "scripts/round_dagger/oci_nrt_round_learner.sbatch"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "sealed-source"
            source_tools = source / "scripts/round_dagger"
            source_tools.mkdir(parents=True)
            for path in (
                source / ".source_snapshot_manifest.json",
                source_tools / "verify_sealed_source_snapshot.py",
                source_tools / "remote_campaign_transfer.py",
            ):
                path.write_text("{}\n", encoding="utf-8")

            remote_campaign = root / "remote-campaign"
            policy_dir = remote_campaign / "policies"
            policy_dir.mkdir(parents=True)
            checkpoint = policy_dir / "checkpoint.pt"
            specification = Path(f"{checkpoint}.policy_spec.json")
            manifest = Path(f"{checkpoint}.manifest.json")
            checkpoint.write_bytes(b"host-dry-run-checkpoint")
            specification.write_text("{}\n", encoding="utf-8")
            manifest.write_text("{}\n", encoding="utf-8")

            inventory = root / "campaign.inventory.json"
            inventory.write_text("{}\n", encoding="utf-8")
            container = root / "trainer.sqsh"
            container.write_bytes(b"sealed-container")
            netrc = root / "wandb.netrc"
            netrc.write_text("machine api.wandb.ai\n", encoding="utf-8")
            netrc.chmod(0o600)
            remote_output = root / "learner-output"
            job_root = root / "job"

            fake_bin = root / "fake-bin"
            fake_bin.mkdir()
            invoked = root / "bare-python-was-invoked"
            fake_python = fake_bin / "python3"
            fake_python.write_text(
                "#!/bin/sh\n: > \"$PYTHON_INVOKED_MARKER\"\nexit 91\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)

            canonical_campaign = Path("/canonical/round-dagger-campaign")
            canonical_checkpoint = canonical_campaign / "policies/checkpoint.pt"
            canonical_specification = Path(
                f"{canonical_checkpoint}.policy_spec.json"
            )
            canonical_manifest = Path(f"{canonical_checkpoint}.manifest.json")
            canonical_output = canonical_campaign / "rounds/macro_round_000000/learner"
            environment = {
                **os.environ,
                "PATH": f"{fake_bin}:/usr/bin:/bin",
                "PYTHON_INVOKED_MARKER": str(invoked),
                "ROUND_DAGGER_BATCH_DRY_RUN": "1",
                "SLURM_JOB_NUM_NODES": "1",
                "SLURM_GPUS_ON_NODE": "8",
                "ROUND_DAGGER_REMOTE_SOURCE_ROOT": str(source),
                "ROUND_DAGGER_REMOTE_SOURCE_INVENTORY_SHA256": "a" * 64,
                "ROUND_DAGGER_REMOTE_CAMPAIGN_ROOT": str(remote_campaign),
                "ROUND_DAGGER_REMOTE_CAMPAIGN_INVENTORY": str(inventory),
                "ROUND_DAGGER_LOCAL_CAMPAIGN_ROOT": str(canonical_campaign),
                "ROUND_DAGGER_CAMPAIGN_ROOT": str(canonical_campaign),
                "ROUND_DAGGER_REMOTE_LEARNER_OUTPUT": str(remote_output),
                "ROUND_DAGGER_REMOTE_CONTAINER": str(container),
                "ROUND_DAGGER_REMOTE_CONTAINER_SHA256": artifacts.sha256_file(container),
                "ROUND_DAGGER_REMOTE_WANDB_NETRC": str(netrc),
                "ROUND_DAGGER_REMOTE_JOB_ROOT": str(job_root),
                "ROUND_DAGGER_CAMPAIGN_ID": "bare-host-dry-run",
                "ROUND_DAGGER_MACRO_ROUND": "0",
                "ROUND_DAGGER_POLICY_SPEC": str(canonical_specification),
                "ROUND_DAGGER_POLICY_MANIFEST": str(canonical_manifest),
                "ROUND_DAGGER_POLICY_SPEC_SHA256": artifacts.sha256_file(specification),
                "ROUND_DAGGER_POLICY_MANIFEST_SHA256": artifacts.sha256_file(manifest),
                "ROUND_DAGGER_POLICY_CHECKPOINT": str(canonical_checkpoint),
                "ROUND_DAGGER_POLICY_SHA256": artifacts.sha256_file(checkpoint),
                "ROUND_DAGGER_RUNTIME_POLICY_SHA256": "b" * 64,
                "ROUND_DAGGER_REPLAY_SNAPSHOT": str(
                    canonical_campaign / "replay/snapshots/round_000000.json"
                ),
                "ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256": "c" * 64,
                "ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256": "d" * 64,
                "ROUND_DAGGER_LEARNER_OUTPUT_DIR": str(canonical_output),
                "ROUND_DAGGER_LEARNER_RESULT": str(
                    canonical_output / "learner_result_round_000000.json"
                ),
                "ROUND_DAGGER_LEARNER_CODE_SHA256": "a" * 64,
                "ROUND_DAGGER_LEARNER_CONFIG_SHA256": "e" * 64,
                "ROUND_DAGGER_LEARNER_CONTRACT_SHA256": "f" * 64,
                "ROUND_DAGGER_LEARNER_MODEL_LINEAGE_SHA256": artifacts.sha256_file(
                    checkpoint
                ),
                "ROUND_DAGGER_BASE_POLICY_SPEC": str(canonical_specification),
                "ROUND_DAGGER_LEARNER_BATCH_SIZE": "8",
                "ROUND_DAGGER_LEARNER_MAX_STEPS": "1200",
                "ROUND_DAGGER_LEARNER_SAVE_EVERY": "5",
                "ROUND_DAGGER_LEARNER_LEARNING_RATE": "2.5e-5",
                "ROUND_DAGGER_LEARNER_WEIGHT_DECAY": "0.01",
                "ROUND_DAGGER_LEARNER_SEED": "17",
                "ROUND_DAGGER_LEARNER_PARTITION_SEED": "23",
                "ROUND_DAGGER_WANDB_PROJECT": "roborender-round-dagger",
                "ROUND_DAGGER_WANDB_ENTITY": "nvidia",
                "ROUND_DAGGER_WANDB_RUN_ID": "pending-identity",
                "ROUND_DAGGER_WANDB_EXPECTED_EMAIL": "ldu@nvidia.com",
                "ROUND_DAGGER_WANDB_AUTHORIZATION_SCOPE": "team",
                "ROUND_DAGGER_REMOTE_SUCCESS_MARKER": str(
                    job_root / "learner_success.json"
                ),
                "ROUND_DAGGER_TARGET_ENVIRONMENTS": "256",
                "ROUND_DAGGER_COHORT_COUNT": "4",
                "ROUND_DAGGER_ENVIRONMENTS_PER_COHORT": "64",
                "ROUND_DAGGER_STEPS_PER_ENVIRONMENT": "512",
                "ROUND_DAGGER_TRANSITIONS_PER_MACRO_ROUND": "131072",
            }
            self._bind_identity(environment)
            identity = environment["ROUND_DAGGER_LEARNER_IDENTITY_SHA256"]
            identity_root = root / f"identity_{identity}"
            identity_root.mkdir()
            identity_campaign = identity_root / "campaign_input"
            remote_campaign.rename(identity_campaign)
            identity_inventory = identity_root / "campaign_input.inventory.json"
            inventory.rename(identity_inventory)
            identity_output = identity_root / "learner_output"
            identity_job = identity_root / "job"
            environment.update(
                ROUND_DAGGER_REMOTE_CAMPAIGN_ROOT=str(identity_campaign),
                ROUND_DAGGER_REMOTE_CAMPAIGN_INVENTORY=str(identity_inventory),
                ROUND_DAGGER_REMOTE_LEARNER_OUTPUT=str(identity_output),
                ROUND_DAGGER_REMOTE_JOB_ROOT=str(identity_job),
                ROUND_DAGGER_REMOTE_SUCCESS_MARKER=str(
                    identity_job / "learner_success.json"
                ),
                SLURM_JOB_NAME=f"rr-rd-{identity}",
            )
            readonly_directories = [source_tools, source]
            for path in source.rglob("*"):
                path.chmod(0o555 if path.is_dir() else 0o444)
            source.chmod(0o555)
            try:
                completed = subprocess.run(
                    ["bash", str(batch)],
                    cwd=root,
                    env=environment,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=30,
                    check=False,
                )
                missing_identity = dict(environment)
                missing_identity.pop("ROUND_DAGGER_LEARNER_IDENTITY_SHA256")
                omitted = subprocess.run(
                    ["bash", str(batch)],
                    cwd=root,
                    env=missing_identity,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=30,
                    check=False,
                )
                tampered_identity = dict(
                    environment, ROUND_DAGGER_LEARNER_IDENTITY_SHA256="0" * 64
                )
                tampered = subprocess.run(
                    ["bash", str(batch)],
                    cwd=root,
                    env=tampered_identity,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=30,
                    check=False,
                )
            finally:
                for directory in readonly_directories:
                    directory.chmod(0o755)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("HOST_DRY_RUN_OK", completed.stdout)
            self.assertIn("host_python=unused", completed.stdout)
            self.assertFalse(invoked.exists(), "bare host Python was unexpectedly invoked")
            self.assertNotEqual(omitted.returncode, 0)
            self.assertIn("missing required environment", omitted.stderr)
            self.assertNotEqual(tampered.returncode, 0)
            self.assertIn("exact semantic run", tampered.stderr)

    def test_local_projection_filters_mutable_state_before_inventory(self) -> None:
        bridge = BRIDGE.read_text(encoding="utf-8")
        self.assertIn("--delete-excluded", bridge)
        for exclusion in (
            "--exclude='/actors/'",
            "--exclude='/operator/'",
            "--exclude='/.coordinator_state.lock'",
            "--exclude='/rounds/*/orchestration_run/'",
        ):
            self.assertIn(exclusion, bridge)
        self.assertNotIn(
            'validate-structure --root "$CAMPAIGN_ROOT"', bridge
        )
        self.assertNotIn('validate-structure --root "$INCOMING"', bridge)

        local_rsync = bridge.index(
            "rsync --archive --links --delete --delete-excluded"
        )
        incoming_validation = bridge.index(
            'validate-projection --root "$INCOMING"'
        )
        incoming_inventory = bridge.index(
            'inventory --root "$INCOMING" --output "$LOCAL_INVENTORY"'
        )
        self.assertLess(local_rsync, incoming_validation)
        self.assertLess(incoming_validation, incoming_inventory)
        self.assertIn('validate-projection --root "$LOCAL_PROJECTION"', bridge)

    def test_resumable_rsync_does_not_mix_append_with_delayed_updates(self) -> None:
        bridge = BRIDGE.read_text(encoding="utf-8")
        self.assertEqual(bridge.count("--append-verify"), 3)
        self.assertNotIn("--delay-updates", bridge)
        self.assertEqual(bridge.count("--delete-delay"), 3)
        self.assertIn(
            '--stage "$REMOTE_STAGE" --destination "$REMOTE_CAMPAIGN"',
            bridge,
        )
        self.assertIn(
            'publish --stage "$INCOMING_OUTPUT"',
            bridge,
        )

    def test_resumable_rsync_option_set_executes_and_verifies_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            destination.mkdir()
            payload = b"verified-resumable-payload" * 4096
            (source / "artifact.bin").write_bytes(payload)
            (destination / "artifact.bin").write_bytes(payload[:8192])

            completed = subprocess.run(
                [
                    "rsync",
                    "--archive",
                    "--no-links",
                    "--partial",
                    "--append-verify",
                    "--delete-delay",
                    f"{source}/",
                    f"{destination}/",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual((destination / "artifact.bin").read_bytes(), payload)

    def test_projection_rsync_removes_excluded_and_stale_mutable_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = root / "campaign"
            incoming = root / "incoming"
            for relative in (
                "policies",
                "replay",
                "rounds/macro_round_000000/cohorts",
                "actors/cohort_00/.wandb_media",
                "operator",
                "rounds/macro_round_000000/orchestration_run",
            ):
                (campaign / relative).mkdir(parents=True, exist_ok=True)
            (campaign / "campaign.json").write_text("{}\n", encoding="utf-8")
            (campaign / "coordinator_state.json").write_text(
                "{}\n", encoding="utf-8"
            )
            (campaign / ".coordinator_state.lock").write_text(
                "lock\n", encoding="utf-8"
            )
            (campaign / "replay/snapshot.json").write_text(
                "snapshot\n", encoding="utf-8"
            )
            (campaign / "actors/cohort_00/.wandb_media/latest").symlink_to(
                campaign / "campaign.json"
            )
            (campaign / "operator/state.json").write_text(
                "{}\n", encoding="utf-8"
            )
            (campaign / "rounds/macro_round_000000/orchestration_run/log").write_text(
                "telemetry\n", encoding="utf-8"
            )

            (incoming / "actors/stale").mkdir(parents=True)
            (incoming / "operator").mkdir()
            (incoming / "actors/stale/value").write_text(
                "stale\n", encoding="utf-8"
            )
            completed = subprocess.run(
                [
                    "rsync",
                    "--archive",
                    "--links",
                    "--delete",
                    "--delete-excluded",
                    f"--link-dest={campaign}",
                    "--exclude=/actors/",
                    "--exclude=/operator/",
                    "--exclude=/.coordinator_state.lock",
                    "--exclude=/rounds/*/orchestration_run/",
                    f"{campaign}/",
                    f"{incoming}/",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertFalse((incoming / "actors").exists())
            self.assertFalse((incoming / "operator").exists())
            self.assertFalse((incoming / ".coordinator_state.lock").exists())
            self.assertFalse(
                (incoming / "rounds/macro_round_000000/orchestration_run").exists()
            )
            self.assertTrue((incoming / "replay/snapshot.json").is_file())
            transfer.validate_campaign_projection(incoming)

    def test_projection_rsync_preserves_included_link_for_strict_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = root / "campaign"
            incoming = root / "incoming"
            for relative in (
                "policies",
                "replay",
                "rounds/macro_round_000000",
            ):
                (campaign / relative).mkdir(parents=True, exist_ok=True)
            (campaign / "campaign.json").write_text("{}\n", encoding="utf-8")
            (campaign / "coordinator_state.json").write_text(
                "{}\n", encoding="utf-8"
            )
            (campaign / "replay/unsafe-link").symlink_to(
                campaign / "campaign.json"
            )
            incoming.mkdir()

            completed = subprocess.run(
                [
                    "rsync",
                    "--archive",
                    "--links",
                    "--delete",
                    "--delete-excluded",
                    f"--link-dest={campaign}",
                    "--exclude=/actors/",
                    "--exclude=/operator/",
                    "--exclude=/.coordinator_state.lock",
                    "--exclude=/rounds/*/orchestration_run/",
                    f"{campaign}/",
                    f"{incoming}/",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((incoming / "replay/unsafe-link").is_symlink())
            with self.assertRaisesRegex(
                transfer.TransferInventoryError, "non-regular"
            ):
                transfer.validate_campaign_projection(incoming)

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
