from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from round_dagger.artifacts import sha256_file

try:
    import torch
    from scripts.round_dagger import h100_validation_attestation as attestation
except ImportError:
    torch = None
    attestation = None


ROOT = Path(__file__).resolve().parents[1]
SUBMIT = ROOT / "scripts/round_dagger/submit_oci_nrt_h100_validation.sh"
BATCH = ROOT / "scripts/round_dagger/oci_nrt_h100_validation.sbatch"
WRAPPER = ROOT / "scripts/round_dagger/run_oci_nrt_h100_validation_container.sh"
ATTEST = ROOT / "scripts/round_dagger/h100_validation_attestation.py"
READBACK = ROOT / "scripts/round_dagger/h100_validation_wandb_readback.py"


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


class OCIH100ValidationLauncherTest(unittest.TestCase):
    def test_scripts_are_executable_parse_and_keep_production_bridge_unchanged(self) -> None:
        production_bridge = (
            ROOT / "scripts/round_dagger/submit_oci_nrt_round_learner_and_wait.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("8:500:5:2.5e-5:0.01:17:23", production_bridge)
        for path in (SUBMIT, BATCH, WRAPPER, ATTEST, READBACK):
            self.assertTrue(os.access(path, os.X_OK), path)
        for path in (SUBMIT, BATCH, WRAPPER):
            completed = subprocess.run(
                ["bash", "-n", str(path)], text=True, capture_output=True, check=False
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_static_contract_is_one_gpu_preflight_then_exact_eight_rank_restart_gate(self) -> None:
        submit = SUBMIT.read_text(encoding="utf-8")
        batch = BATCH.read_text(encoding="utf-8")
        wrapper = WRAPPER.read_text(encoding="utf-8")
        combined = submit + batch + wrapper
        self.assertIn("--export=NONE", submit)
        self.assertIn("--gpus-per-node=8", submit)
        self.assertIn("--gpus=1 --gpus-per-task=1", batch)
        self.assertIn("--gpus=8 --gpus-per-task=8", batch)
        self.assertIn("--nproc-per-node 8", wrapper)
        self.assertIn("--batch-size 1 --max-steps 2", wrapper)
        self.assertIn("--save-every 1", wrapper)
        self.assertIn("--validation-stop-at-step 1", wrapper)
        self.assertIn("--resume-from", wrapper)
        self.assertIn('--wandb-run-id "$FULL_RUN_ID"', wrapper)
        self.assertGreaterEqual(
            wrapper.count('--wandb-run-id "$RESUMED_RUN_ID"'), 2
        )
        self.assertIn("h100_validation_wandb_readback.py", wrapper)
        self.assertIn("second_invocation_resume_mode\": \"must", wrapper)
        self.assertIn("verify_learner_restart_equivalence.py", wrapper)
        self.assertIn("$REMOTE_CAMPAIGN:$LOCAL_CAMPAIGN:ro", batch)
        self.assertIn("$REMOTE_OUTPUT:$LOCAL_OUTPUT", batch)
        self.assertIn("/round_dagger_validation_job/wandb", wrapper)
        self.assertNotIn('$OUTPUT/wandb', wrapper)
        self.assertIn("/home/ldu/.netrc:ro", batch)
        self.assertNotIn("WANDB_API_KEY", combined)
        self.assertNotIn("cat \"$HOME/.netrc\"", combined)
        self.assertIn("chmod 0444", wrapper)

    def test_dry_run_validates_inputs_without_submitting_or_mutating_outputs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rr_h100_gate_") as name:
            root = Path(name)
            source = root / "sealed_source"
            tools = source / "scripts/round_dagger"
            tools.mkdir(parents=True)
            fake_verifier = tools / "verify_sealed_source_snapshot.py"
            fake_verifier.write_text(
                "#!/usr/bin/env python3\nraise SystemExit(0)\n", encoding="utf-8"
            )
            fake_verifier.chmod(0o755)
            fake_batch = tools / BATCH.name
            fake_batch.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            fake_batch.chmod(0o755)
            fake_sbatch = root / "bin/sbatch"
            fake_sbatch.parent.mkdir()
            fake_sbatch.write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
            fake_sbatch.chmod(0o755)

            container = root / "trainer.sqsh"
            container.write_bytes(b"pinned validation container")
            remote_campaign = root / "campaign_projection"
            inputs = remote_campaign / "inputs"
            inputs.mkdir(parents=True)
            (remote_campaign / "validation/output").mkdir(parents=True)
            snapshot_sha = "3" * 64
            snapshot = inputs / "snapshot.json"
            _json(
                snapshot,
                {
                    "snapshot_sha256": snapshot_sha,
                    "campaign_id": "dry-run-validation",
                    "through_macro_round": 0,
                    "expected_train_cohorts": 4,
                    "cohorts": [
                        {"shards": [{"id": cohort * 2 + offset} for offset in range(2)]}
                        for cohort in range(4)
                    ],
                },
            )
            checkpoint = inputs / "student.pt"
            checkpoint.write_bytes(b"parent")
            spec = Path(f"{checkpoint}.policy_spec.json")
            manifest = Path(f"{checkpoint}.manifest.json")
            _json(spec, {"spec": True})
            _json(manifest, {"manifest": True})
            output = root / "returned_output"
            job = root / "job_scratch"
            logs = root / "scheduler_logs"
            output.mkdir()
            job.mkdir()
            logs.mkdir()
            netrc = root / "wandb.netrc"
            netrc.write_text("machine api.wandb.ai login validation\n", encoding="utf-8")
            netrc.chmod(0o600)
            local_campaign = "/original/immutable/campaign"
            environment = {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "DRY_RUN": "1",
                "ROUND_DAGGER_VALIDATION_SBATCH_BIN": str(fake_sbatch),
                "ROUND_DAGGER_VALIDATION_SOURCE_ROOT": str(source),
                "ROUND_DAGGER_VALIDATION_SOURCE_INVENTORY_SHA256": "1" * 64,
                "ROUND_DAGGER_VALIDATION_CONTAINER": str(container),
                "ROUND_DAGGER_VALIDATION_CONTAINER_SHA256": sha256_file(container),
                "ROUND_DAGGER_VALIDATION_REMOTE_CAMPAIGN_ROOT": str(remote_campaign),
                "ROUND_DAGGER_VALIDATION_LOCAL_CAMPAIGN_ROOT": local_campaign,
                "ROUND_DAGGER_VALIDATION_REMOTE_OUTPUT_ROOT": str(output),
                "ROUND_DAGGER_VALIDATION_LOCAL_OUTPUT_ROOT": (
                    f"{local_campaign}/validation/output"
                ),
                "ROUND_DAGGER_VALIDATION_REMOTE_JOB_ROOT": str(job),
                "ROUND_DAGGER_VALIDATION_REMOTE_LOG_ROOT": str(logs),
                "ROUND_DAGGER_VALIDATION_WANDB_NETRC": str(netrc),
                "ROUND_DAGGER_VALIDATION_REPLAY_SNAPSHOT": (
                    f"{local_campaign}/inputs/snapshot.json"
                ),
                "ROUND_DAGGER_VALIDATION_REPLAY_SNAPSHOT_FILE_SHA256": sha256_file(
                    snapshot
                ),
                "ROUND_DAGGER_VALIDATION_REPLAY_SNAPSHOT_SHA256": snapshot_sha,
                "ROUND_DAGGER_VALIDATION_POLICY_CHECKPOINT": (
                    f"{local_campaign}/inputs/student.pt"
                ),
                "ROUND_DAGGER_VALIDATION_POLICY_SHA256": sha256_file(checkpoint),
                "ROUND_DAGGER_VALIDATION_POLICY_SPEC_SHA256": sha256_file(spec),
                "ROUND_DAGGER_VALIDATION_POLICY_MANIFEST_SHA256": sha256_file(
                    manifest
                ),
                "ROUND_DAGGER_VALIDATION_RUNTIME_POLICY_SHA256": "4" * 64,
                "ROUND_DAGGER_VALIDATION_LEARNER_CONFIG_SHA256": "5" * 64,
                "ROUND_DAGGER_VALIDATION_CAMPAIGN_ID": "dry-run-validation",
                "ROUND_DAGGER_VALIDATION_MACRO_ROUND": "0",
            }
            completed = subprocess.run(
                [str(SUBMIT)],
                cwd=root,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("--export=NONE", completed.stdout)
            self.assertIn("OCI_NRT_H100_VALIDATION_DRY_RUN_OK", completed.stdout)
            self.assertEqual(list(output.iterdir()), [])
            self.assertEqual(list(job.iterdir()), [])
            self.assertEqual(list(logs.iterdir()), [])


@unittest.skipIf(torch is None, "attestation tests require PyTorch")
class H100ValidationAttestationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="rr_h100_attestation_")
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        _json(self.source / ".source_snapshot_manifest.json", {"sealed": True})
        self.container = self.root / "container.sqsh"
        self.container.write_bytes(b"container")
        self.campaign = self.root / "campaign"
        self.campaign.mkdir()
        self.snapshot = self.campaign / "snapshot.json"
        _json(self.snapshot, {"snapshot": True})
        self.policy = self.campaign / "parent.pt"
        self.policy.write_bytes(b"parent")
        self.spec = Path(f"{self.policy}.policy_spec.json")
        self.manifest = Path(f"{self.policy}.manifest.json")
        _json(self.spec, {"spec": True})
        _json(self.manifest, {"manifest": True})
        self.bundle = self.root / "bundle.json"
        _json(self.bundle, {"config": True})
        self.import_report = self.root / "import.json"
        _json(
            self.import_report,
            {
                "status": "PASS",
                "visible_cuda_devices": 1,
                "compute_capability": [9, 0],
                "nccl_available": True,
                "campaign_mount": "ro",
                "learner_output_mount": "rw",
            },
        )
        self.wandb_report = self.root / "wandb.json"
        _json(
            self.wandb_report,
            {
                "email": "ldu@nvidia.com",
                "entity": "nvidia",
                "authorization_scope": "team",
                "team_member": True,
                "entity_is_team": True,
                "personal": False,
                "mode": "online",
            },
        )
        self.source_sha = "1" * 64
        self.snapshot_sha = "2" * 64
        self.config_sha = sha256_file(self.bundle)
        self.full = self.root / "full"
        self.resumed = self.root / "resumed"
        self.full.mkdir()
        self.resumed.mkdir()
        self.full_restart = self.full / "restart.pt"
        self.resumed_restart = self.resumed / "restart.pt"
        self.interrupted = self.root / "stop1.pt"
        self._checkpoint(self.full_restart, 2)
        self._checkpoint(self.resumed_restart, 2)
        self._checkpoint(self.interrupted, 1)
        self.full_result = self._result(self.full, b"candidate-full")
        self.resumed_result = self._result(self.resumed, b"candidate-resumed")
        self.restart_report = self.root / "restart_equivalence.json"
        _json(
            self.restart_report,
            {
                "schema": "fast_roborender.round_dagger.restart_equivalence.v1",
                "semantic_equivalent": True,
                "results_equivalent": True,
                "expected_completed_steps": 2,
                "uninterrupted_checkpoint_sha256": sha256_file(self.full_restart),
                "resumed_checkpoint_sha256": sha256_file(self.resumed_restart),
                "uninterrupted_result_sha256": sha256_file(self.full_result),
                "resumed_result_sha256": sha256_file(self.resumed_result),
            },
        )
        self.wandb_readback = self.root / "wandb_readback.json"
        _json(
            self.wandb_readback,
            {
                "schema": "fast_roborender.round_dagger.h100_validation_wandb_readback.v1",
                "status": "PASS",
                "account_email": "ldu@nvidia.com",
                "entity": "nvidia",
                "authorization_scope": "team",
                "team_member": True,
                "entity_is_team": True,
                "personal": False,
                "campaign_id": "validation",
                "macro_round": 0,
                "learner_config_sha256": self.config_sha,
                "model_lineage_sha256": sha256_file(self.policy),
                "full_result_sha256": sha256_file(self.full_result),
                "resumed_result_sha256": sha256_file(self.resumed_result),
                "initial_loss": 1.0,
                "final_loss": 0.5,
                "histories_identical": True,
                "resumed_history_contiguous": True,
                "strict_resume_mode": "must",
                "strict_resume_verified": True,
                "full_run": {"run_id": "full-run", "history_steps": [1, 2]},
                "resumed_run": {
                    "run_id": "resumed-run",
                    "history_steps": [1, 2],
                },
            },
        )
        self.wandb_readback.chmod(0o444)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _checkpoint(self, path: Path, steps: int) -> None:
        torch.save(
            {
                "schema": "fast_roborender.round_dagger.learner_checkpoint.v1",
                "completed_steps": steps,
                "world_size": 8,
                "rng_states": [{} for _ in range(8)],
            },
            path,
        )

    def _result(self, directory: Path, candidate_bytes: bytes) -> Path:
        candidate = directory / "candidate_round_000000.pt"
        candidate.write_bytes(candidate_bytes)
        result = directory / "learner_result_round_000000.json"
        _json(
            result,
            {
                "schema": "fast_roborender.round_dagger.learner_result.v2",
                "campaign_id": "validation",
                "macro_round": 0,
                "code_sha256": self.source_sha,
                "learner_config_sha256": self.config_sha,
                "replay_snapshot_sha256": self.snapshot_sha,
                "optimizer_steps": 2,
                "checkpoint_path": candidate.name,
                "checkpoint_sha256": sha256_file(candidate),
                "initial_loss": 1.0,
                "final_loss": 0.5,
            },
        )
        return result

    def _kwargs(self) -> dict:
        return {
            "source_root": self.source,
            "source_inventory_sha256": self.source_sha,
            "container": self.container,
            "container_sha256": sha256_file(self.container),
            "campaign_root": self.campaign,
            "snapshot": self.snapshot,
            "snapshot_file_sha256": sha256_file(self.snapshot),
            "snapshot_sha256": self.snapshot_sha,
            "replay_shards": 8,
            "policy_checkpoint": self.policy,
            "policy_checkpoint_sha256": sha256_file(self.policy),
            "policy_spec": self.spec,
            "policy_spec_sha256": sha256_file(self.spec),
            "policy_manifest": self.manifest,
            "policy_manifest_sha256": sha256_file(self.manifest),
            "bundle": self.bundle,
            "config_sha256": self.config_sha,
            "import_report": self.import_report,
            "wandb_report": self.wandb_report,
            "wandb_readback": self.wandb_readback,
            "uninterrupted_checkpoint": self.full_restart,
            "uninterrupted_result": self.full_result,
            "interrupted_checkpoint": self.interrupted,
            "resumed_checkpoint": self.resumed_restart,
            "resumed_result": self.resumed_result,
            "restart_report": self.restart_report,
            "campaign_id": "validation",
            "macro_round": 0,
        }

    def test_attestation_binds_all_live_gate_evidence(self) -> None:
        report = attestation.build_attestation(**self._kwargs())
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["learner_contract"]["backend"], "nccl")
        self.assertEqual(report["learner_contract"]["world_size"], 8)
        self.assertTrue(report["restart_semantic_equivalent"])
        self.assertEqual(report["source_inventory_sha256"], self.source_sha)

    def test_attestation_rejects_too_few_shards_and_tampered_restart_evidence(self) -> None:
        with self.assertRaisesRegex(
            attestation.H100ValidationAttestationError, "at least 8 shards"
        ):
            attestation.build_attestation(**dict(self._kwargs(), replay_shards=7))
        report = json.loads(self.restart_report.read_text(encoding="utf-8"))
        report["results_equivalent"] = False
        _json(self.restart_report, report)
        with self.assertRaisesRegex(
            attestation.H100ValidationAttestationError,
            "restart equivalence evidence mismatch",
        ):
            attestation.build_attestation(**self._kwargs())

    def test_attestation_rejects_tampered_or_writable_wandb_readback(self) -> None:
        self.wandb_readback.chmod(0o644)
        with self.assertRaisesRegex(
            attestation.H100ValidationAttestationError, "must be read-only"
        ):
            attestation.build_attestation(**self._kwargs())
        self.wandb_readback.chmod(0o444)
        report = json.loads(self.wandb_readback.read_text(encoding="utf-8"))
        report["resumed_history_contiguous"] = False
        self.wandb_readback.chmod(0o644)
        _json(self.wandb_readback, report)
        self.wandb_readback.chmod(0o444)
        with self.assertRaisesRegex(
            attestation.H100ValidationAttestationError,
            "W&B learner readback evidence mismatch",
        ):
            attestation.build_attestation(**self._kwargs())


if __name__ == "__main__":
    unittest.main()
