from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from round_dagger.artifacts import sha256_file
from round_dagger.learner import BUNDLE_SCHEMA, LearnerBundleConfig
from round_dagger.learner_identity import (
    learner_identity_sha256,
    learner_wandb_run_id,
)
from tests.policy_bundle_helpers import publish_policy_bundle

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
CONTRACT = ROOT / "configs/round_dagger/validation_learner_1x8h100_2step.json"


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
        self.assertIn(
            "PARTITION=${ROUND_DAGGER_VALIDATION_PARTITION:-batch_singlenode}",
            submit,
        )
        self.assertIn("partition must be batch_singlenode", submit)
        self.assertIn("--gpus-per-node=8", submit)
        self.assertIn("--gpus=1 --gpus-per-task=1", batch)
        self.assertIn("--gpus=8 --gpus-per-task=8", batch)
        self.assertIn("--nproc-per-node 8", wrapper)
        self.assertIn("--batch-size 1 --max-steps 2", wrapper)
        self.assertIn("--save-every 1", wrapper)
        self.assertIn("--validation-stop-at-step 1", wrapper)
        self.assertIn("--resume-from", wrapper)
        self.assertNotIn("FULL_RUN_ID", wrapper)
        self.assertNotIn("RESUMED_RUN_ID", wrapper)
        self.assertEqual(wrapper.count('--wandb-run-id "$WANDB_RUN_ID"'), 2)
        comparator = wrapper[
            wrapper.index("--master-port 29741") : wrapper.index("--master-port 29742")
        ]
        self.assertNotIn("--wandb-project", comparator)
        self.assertNotIn("--wandb-run-id", comparator)
        self.assertIn('--learner-identity-sha256 "$LEARNER_IDENTITY_SHA"', wrapper)
        self.assertIn('--runtime-container-sha256 "$CONTAINER_SHA"', wrapper)
        self.assertIn('--learner-contract-sha256 "$CONTRACT_SHA"', wrapper)
        self.assertIn('--run-id "$WANDB_RUN_ID"', wrapper)
        self.assertIn('--validation-contract "$CONTRACT"', wrapper)
        self.assertIn('$SOURCE/round_dagger/learner_identity.py', submit)
        self.assertIn('$REMOTE_SOURCE/round_dagger/learner_identity.py', batch)
        self.assertNotIn('PYTHONPATH="$SOURCE" python3 -', submit)
        self.assertNotIn('PYTHONPATH="$REMOTE_SOURCE" python3 -', batch)
        self.assertIn("validation_learner_1x8h100_2step.json", combined)
        self.assertIn("h100_validation_wandb_readback.py", wrapper)
        self.assertIn("second_invocation_resume_mode\": \"must", wrapper)
        self.assertIn("verify_learner_restart_equivalence.py", wrapper)
        self.assertIn("$REMOTE_CAMPAIGN:$LOCAL_CAMPAIGN:ro", batch)
        self.assertIn("$REMOTE_OUTPUT:$LOCAL_OUTPUT", batch)
        self.assertIn("/round_dagger_validation_job/wandb", wrapper)
        self.assertNotIn('$OUTPUT/wandb', wrapper)
        self.assertIn('${PYTHONPATH:+:$PYTHONPATH}', wrapper)
        self.assertIn('expected NumPy 1.26.0', wrapper)
        self.assertIn('"numpy_version": np.__version__', wrapper)
        self.assertIn('"isaac_pip_archive_on_sys_path"', wrapper)
        self.assertIn("/home/ldu/.netrc:ro", batch)
        self.assertNotIn("WANDB_API_KEY", combined)
        self.assertNotIn("cat \"$HOME/.netrc\"", combined)
        self.assertIn("chmod 0444", wrapper)

    def test_host_identity_gate_works_when_numpy_import_is_blocked(self) -> None:
        inputs = {
            "campaign_id": "host-no-numpy",
            "macro_round": 3,
            "learner_code_sha256": "1" * 64,
            "learner_config_sha256": "2" * 64,
            "replay_snapshot_file_sha256": "3" * 64,
            "replay_snapshot_sha256": "4" * 64,
            "parent_checkpoint_sha256": "5" * 64,
            "parent_runtime_policy_sha256": "6" * 64,
            "runtime_container_sha256": "7" * 64,
            "learner_contract_sha256": "8" * 64,
        }
        expected_identity = learner_identity_sha256(**inputs)
        expected_run_id = learner_wandb_run_id(
            campaign_id=inputs["campaign_id"],
            macro_round=inputs["macro_round"],
            learner_identity_sha256=expected_identity,
        )
        identity_script = ROOT / "round_dagger/learner_identity.py"
        with tempfile.TemporaryDirectory(prefix="rr_no_numpy_") as name:
            blocker = Path(name)
            (blocker / "sitecustomize.py").write_text(
                """\
import builtins
_original_import = builtins.__import__
def _blocked_import(name, *args, **kwargs):
    if name == \"numpy\" or name.startswith(\"numpy.\"):
        raise ModuleNotFoundError(\"NumPy deliberately blocked by host-gate test\")
    return _original_import(name, *args, **kwargs)
builtins.__import__ = _blocked_import
""",
                encoding="utf-8",
            )
            environment = dict(os.environ, PYTHONPATH=str(blocker))
            probe = subprocess.run(
                [sys.executable, "-c", "import numpy"],
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(probe.returncode, 0)
            self.assertIn("NumPy deliberately blocked", probe.stderr)

            completed = subprocess.run(
                [
                    sys.executable,
                    str(identity_script),
                    inputs["campaign_id"],
                    str(inputs["macro_round"]),
                    inputs["learner_code_sha256"],
                    inputs["learner_config_sha256"],
                    inputs["replay_snapshot_file_sha256"],
                    inputs["replay_snapshot_sha256"],
                    inputs["parent_checkpoint_sha256"],
                    inputs["parent_runtime_policy_sha256"],
                    inputs["runtime_container_sha256"],
                    inputs["learner_contract_sha256"],
                ],
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(
            completed.stdout.splitlines(), [expected_identity, expected_run_id]
        )

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
            package = source / "round_dagger"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "learner_identity.py").write_bytes(
                (ROOT / "round_dagger/learner_identity.py").read_bytes()
            )
            contract = source / "configs/round_dagger/validation_learner_1x8h100_2step.json"
            contract_document = json.loads(CONTRACT.read_text(encoding="utf-8"))
            contract_document["container"]["path"] = str(container)
            contract_document["container"]["sha256"] = sha256_file(container)
            _json(contract, contract_document)
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
                "ROUND_DAGGER_VALIDATION_CONTRACT_SHA256": sha256_file(contract),
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

            wrong_contract = dict(environment)
            wrong_contract["ROUND_DAGGER_VALIDATION_CONTRACT_SHA256"] = "0" * 64
            rejected = subprocess.run(
                [str(SUBMIT)],
                cwd=root,
                env=wrong_contract,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("validation contract digest mismatch", rejected.stderr)
            self.assertEqual(list(output.iterdir()), [])
            self.assertEqual(list(job.iterdir()), [])
            self.assertEqual(list(logs.iterdir()), [])


@unittest.skipIf(torch is None, "attestation tests require PyTorch")
class H100ValidationAttestationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="rr_h100_attestation_")
        self.root = Path(self.temporary.name)
        self.campaign_id = "validation"
        self.macro_round = 0
        self.source_sha = "1" * 64
        self.snapshot_sha = "2" * 64
        self.runtime_sha = "3" * 64

        self.source = self.root / "source"
        self.source.mkdir()
        _json(
            self.source / ".source_snapshot_manifest.json",
            {
                "schema": "fast_roborender.hsg_source_snapshot.v2",
                "status": "sealed_tracked_hydrated_sources",
                "snapshot_name": "fast_roborender_aaaaaaa_bbbbbbb_sealed",
                "fast_roborender_revision": "a" * 40,
                "gr00t_revision": "b" * 40,
                "tracked_only": True,
                "symlinks_rejected": True,
                "bytecode_rejected": True,
                "lfs_pointer_oids_verified": True,
            },
        )
        self.container = self.root / "container.sqsh"
        self.container.write_bytes(b"container")
        self.container_sha = sha256_file(self.container)

        self.contract = (
            self.source
            / "configs"
            / "round_dagger"
            / "validation_learner_1x8h100_2step.json"
        )
        _json(
            self.contract,
            {
                "schema": (
                    "fast_roborender.round_dagger."
                    "validation_learner_contract.v1"
                ),
                "purpose": "live_h100_restart_and_telemetry_gate",
                "allocation": {
                    "nodes": 1,
                    "gpus_per_node": 8,
                    "gpu_architecture": "H100-SM90",
                    "backend": "nccl",
                    "batch_size_per_rank": 1,
                    "global_batch_size": 8,
                },
                "optimization": {
                    "optimizer_steps": 2,
                    "save_every_steps": 1,
                    "intentional_stop_step": 1,
                    "learning_rate": 2.5e-5,
                    "weight_decay": 0.01,
                    "seed": 17,
                    "partition_seed": 23,
                },
                "restart_validation": {
                    "uninterrupted_comparator": "local_only",
                    "wandb_lineage": "single_canonical_run",
                    "first_invocation_resume": "allow",
                    "second_invocation_resume": "must",
                },
                "cluster": {
                    "host": "oci-nrt",
                    "account": "general_cs_infra__ngcsre",
                    "partition": "batch_singlenode",
                    "qos": "normal",
                },
                "container": {
                    "path": (
                        "/lustre/fs1/portfolios/general/projects/"
                        "general_cs_infra__ngcsre/users/ldu/"
                        "roborender_h100_build/containers/"
                        "roborender-isaac-gr00t-x86_64-sm90-isaac5.1.sqsh"
                    ),
                    "sha256": self.container_sha,
                    "python": "/opt/viral-h100/bin/python",
                },
                "wandb": {
                    "mode": "online",
                    "expected_email": "ldu@nvidia.com",
                    "entity": "nvidia",
                    "authorization_scope": "team",
                    "project": "roborender-round-dagger-validation",
                },
                "production_target_contract_sha256": (
                    "b044ba7d5bd5e3c1645e79616f19729438d00f66e2a40b19"
                    "b4e955ae8d630d90"
                ),
            },
        )
        self.contract.chmod(0o444)
        self.contract_sha = sha256_file(self.contract)

        self.campaign = self.root / "campaign"
        self.campaign.mkdir()
        self.snapshot = self.campaign / "snapshot.json"
        _json(
            self.snapshot,
            {
                "snapshot_sha256": self.snapshot_sha,
                "campaign_id": self.campaign_id,
                "through_macro_round": self.macro_round,
                "expected_train_cohorts": 4,
            },
        )
        self.snapshot_file_sha = sha256_file(self.snapshot)
        self.policy = self.campaign / "parent.pt"
        self.policy.write_bytes(b"parent")
        self.policy_sha = sha256_file(self.policy)
        self.spec = Path(f"{self.policy}.policy_spec.json")
        self.manifest = Path(f"{self.policy}.manifest.json")
        _json(self.spec, {"spec": True})
        _json(self.manifest, {"manifest": True})

        self.bundle = self.root / "validation_learner_bundle.json"
        bundle = LearnerBundleConfig.from_mapping(
            {
                "schema": BUNDLE_SCHEMA,
                "campaign_id": self.campaign_id,
                "macro_round": self.macro_round,
                "model_lineage_sha256": self.policy_sha,
                "code_sha256": self.source_sha,
                "rgb_mean": [0.5, 0.5, 0.5],
                "rgb_std": [0.5, 0.5, 0.5],
                "batch_size": 1,
                "max_steps": 2,
                "save_every": 1,
                "learning_rate": 2.5e-5,
                "weight_decay": 0.01,
                "seed": 17,
                "partition_seed": 23,
                "initial_checkpoint_sha256": self.policy_sha,
                "model_config": {
                    "runtime_policy_sha256": self.runtime_sha,
                },
            }
        )
        self.bundle.write_text(
            json.dumps(
                bundle.to_mapping(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        self.config_sha = bundle.sha256
        self.assertEqual(sha256_file(self.bundle), self.config_sha)

        self.identity_sha = learner_identity_sha256(
            campaign_id=self.campaign_id,
            macro_round=self.macro_round,
            learner_code_sha256=self.source_sha,
            learner_config_sha256=self.config_sha,
            replay_snapshot_file_sha256=self.snapshot_file_sha,
            replay_snapshot_sha256=self.snapshot_sha,
            parent_checkpoint_sha256=self.policy_sha,
            parent_runtime_policy_sha256=self.runtime_sha,
            runtime_container_sha256=self.container_sha,
            learner_contract_sha256=self.contract_sha,
        )
        self.run_id = learner_wandb_run_id(
            campaign_id=self.campaign_id,
            macro_round=self.macro_round,
            learner_identity_sha256=self.identity_sha,
        )
        self.semantic = {
            "campaign_id": self.campaign_id,
            "macro_round": self.macro_round,
            "model_lineage_sha256": self.policy_sha,
            "code_sha256": self.source_sha,
            "learner_config_sha256": self.config_sha,
            "learner_identity_sha256": self.identity_sha,
            "runtime_container_sha256": self.container_sha,
            "learner_contract_sha256": self.contract_sha,
            "replay_snapshot_sha256": self.snapshot_sha,
            "replay_snapshot_file_sha256": self.snapshot_file_sha,
        }

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

        self.rows = [
            {
                "schema": (
                    "fast_roborender.round_dagger.learner_telemetry.v2"
                ),
                "campaign_id": self.campaign_id,
                "macro_round": self.macro_round,
                "optimizer_step": step,
                "loss": loss,
                "replay_transitions_rank": 9600,
                "replay_transitions_per_rank": [9600] * 8,
                "replay_transitions_global": 76800,
                "world_size": 8,
            }
            for step, loss in ((1, 1.0), (2, 0.75))
        ]
        payload = "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in self.rows
        )
        self.full_telemetry = self.root / "full_telemetry.jsonl"
        self.resumed_telemetry = self.root / "resumed_telemetry.jsonl"
        self.full_telemetry.write_text(payload, encoding="utf-8")
        self.resumed_telemetry.write_text(payload, encoding="utf-8")

        self.full = self.root / "full"
        self.resumed = self.root / "resumed"
        self.full.mkdir()
        self.resumed.mkdir()
        self.full_restart = self.full / "restart.pt"
        self.resumed_restart = self.resumed / "restart.pt"
        self.interrupted = self.root / "intentional_stop_step_1.pt"
        self._checkpoint(self.full_restart, 2)
        self._checkpoint(self.resumed_restart, 2)
        self._checkpoint(self.interrupted, 1)
        self.interrupted.chmod(0o444)

        self.full_result = self._result(self.full, self.full_restart)
        self.resumed_result = self._result(
            self.resumed, self.resumed_restart
        )
        self.resume_evidence = self.root / "strict_resume_invocation.json"
        _json(
            self.resume_evidence,
            {
                "schema": (
                    "fast_roborender.round_dagger."
                    "h100_validation_strict_resume.v2"
                ),
                "status": "PASS",
                "run_id": self.run_id,
                "first_invocation_resume_mode": "allow",
                "second_invocation_resume_mode": "must",
                "stop_step": 1,
                "target_step": 2,
                "learner_config_sha256": self.config_sha,
                "learner_identity_sha256": self.identity_sha,
                "learner_code_sha256": self.source_sha,
                "replay_snapshot_file_sha256": self.snapshot_file_sha,
                "replay_snapshot_sha256": self.snapshot_sha,
                "parent_checkpoint_sha256": self.policy_sha,
                "parent_runtime_policy_sha256": self.runtime_sha,
                "runtime_container_sha256": self.container_sha,
                "learner_contract_sha256": self.contract_sha,
                "uninterrupted_comparator_wandb": False,
                "canonical_wandb_lineage": True,
                "stopped_checkpoint_sha256": sha256_file(self.interrupted),
                "resumed_checkpoint_sha256": sha256_file(
                    self.resumed_restart
                ),
            },
        )
        self.resume_evidence.chmod(0o444)

        self.restart_report = self.root / "restart_equivalence.json"
        _json(
            self.restart_report,
            {
                "schema": (
                    "fast_roborender.round_dagger.restart_equivalence.v1"
                ),
                "semantic_equivalent": True,
                "results_equivalent": True,
                "expected_completed_steps": 2,
                "uninterrupted_checkpoint": str(
                    self.full_restart.resolve()
                ),
                "uninterrupted_checkpoint_sha256": sha256_file(
                    self.full_restart
                ),
                "resumed_checkpoint": str(self.resumed_restart.resolve()),
                "resumed_checkpoint_sha256": sha256_file(
                    self.resumed_restart
                ),
                "uninterrupted_result": str(self.full_result.resolve()),
                "uninterrupted_result_sha256": sha256_file(
                    self.full_result
                ),
                "resumed_result": str(self.resumed_result.resolve()),
                "resumed_result_sha256": sha256_file(self.resumed_result),
            },
        )

        self.wandb_readback = self.root / "wandb_readback.json"
        self._write_readback()
        self.wandb_readback.chmod(0o444)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _checkpoint(self, path: Path, steps: int) -> None:
        state = {"weight": torch.tensor([1.0])}
        torch.save(
            {
                "schema": (
                    "fast_roborender.round_dagger.learner_checkpoint.v2"
                ),
                "checkpoint_schema": "gr00t.trl.full_state.v2",
                "policy_state_dict": state,
                "policy_parameter_names": ["weight"],
                **self.semantic,
                "completed_steps": steps,
                "initial_loss": 1.0,
                "world_size": 8,
                "model_state_dict": {
                    "weight": torch.tensor([1.0])
                },
                "optimizer_state_dict": {},
                "lr_scheduler_state_dict": None,
                "value_state_dict": None,
                "rng_states": [{} for _ in range(8)],
            },
            path,
        )

    def _result(self, directory: Path, restart: Path) -> Path:
        candidate = directory / "candidate_round_000000.pt"
        shutil.copyfile(restart, candidate)
        spec, manifest = publish_policy_bundle(candidate, self.runtime_sha)
        result = directory / "learner_result_round_000000.json"
        _json(
            result,
            {
                "schema": (
                    "fast_roborender.round_dagger.learner_result.v2"
                ),
                "campaign_id": self.campaign_id,
                "macro_round": self.macro_round,
                "checkpoint_path": candidate.name,
                "checkpoint_sha256": sha256_file(candidate),
                "runtime_policy_sha256": self.runtime_sha,
                "policy_spec_path": spec.name,
                "policy_spec_sha256": sha256_file(spec),
                "policy_bundle_manifest_path": manifest.name,
                "policy_bundle_manifest_sha256": sha256_file(manifest),
                "model_lineage_sha256": self.policy_sha,
                "replay_snapshot_sha256": self.snapshot_sha,
                "code_sha256": self.source_sha,
                "learner_config_sha256": self.config_sha,
                "optimizer_steps": 2,
                "initial_loss": 1.0,
                "final_loss": 0.5,
            },
        )
        return result

    def _write_readback(self) -> None:
        full_candidate = self.full / "candidate_round_000000.pt"
        resumed_candidate = self.resumed / "candidate_round_000000.pt"
        _json(
            self.wandb_readback,
            {
                "schema": (
                    "fast_roborender.round_dagger."
                    "h100_validation_wandb_readback.v2"
                ),
                "status": "PASS",
                "account_email": "ldu@nvidia.com",
                "entity": "nvidia",
                "authorization_scope": "team",
                "team_member": True,
                "entity_is_team": True,
                "personal": False,
                "project": "roborender-round-dagger-validation",
                "campaign_id": self.campaign_id,
                "macro_round": self.macro_round,
                "learner_code_sha256": self.source_sha,
                "learner_config_sha256": self.config_sha,
                "model_lineage_sha256": self.policy_sha,
                "replay_snapshot_file_sha256": self.snapshot_file_sha,
                "replay_snapshot_sha256": self.snapshot_sha,
                "parent_runtime_policy_sha256": self.runtime_sha,
                "learner_identity_sha256": self.identity_sha,
                "runtime_container_sha256": self.container_sha,
                "learner_contract_sha256": self.contract_sha,
                "validation_contract_sha256": self.contract_sha,
                "wandb_run_id": self.run_id,
                "canonical_run": {
                    "run_id": self.run_id,
                    "history": self.rows,
                    "history_steps": [1, 2],
                    "history_contiguous": True,
                    "remote_state": "finished",
                    "config_verified": True,
                },
                "uninterrupted_comparator": {
                    "wandb_enabled": False,
                    "history_steps": [1, 2],
                    "telemetry_sha256": sha256_file(
                        self.full_telemetry
                    ),
                    "result_sha256": sha256_file(self.full_result),
                    "checkpoint_sha256": sha256_file(self.full_restart),
                    "candidate_checkpoint_sha256": sha256_file(
                        full_candidate
                    ),
                },
                "resumed_execution": {
                    "wandb_enabled": True,
                    "history_steps": [1, 2],
                    "telemetry_sha256": sha256_file(
                        self.resumed_telemetry
                    ),
                    "result_sha256": sha256_file(self.resumed_result),
                    "stopped_checkpoint_sha256": sha256_file(
                        self.interrupted
                    ),
                    "checkpoint_sha256": sha256_file(
                        self.resumed_restart
                    ),
                    "candidate_checkpoint_sha256": sha256_file(
                        resumed_candidate
                    ),
                },
                "uninterrupted_telemetry_sha256": sha256_file(
                    self.full_telemetry
                ),
                "resumed_telemetry_sha256": sha256_file(
                    self.resumed_telemetry
                ),
                "uninterrupted_result_sha256": sha256_file(
                    self.full_result
                ),
                "resumed_result_sha256": sha256_file(
                    self.resumed_result
                ),
                "uninterrupted_checkpoint_sha256": sha256_file(
                    self.full_restart
                ),
                "stopped_checkpoint_sha256": sha256_file(
                    self.interrupted
                ),
                "resumed_checkpoint_sha256": sha256_file(
                    self.resumed_restart
                ),
                "strict_resume_evidence_sha256": sha256_file(
                    self.resume_evidence
                ),
                "initial_loss": 1.0,
                "final_loss": 0.5,
                "histories_identical": True,
                "resumed_history_contiguous": True,
                "strict_resume_mode": "must",
                "strict_resume_verified": True,
                "single_canonical_wandb_run": True,
            },
        )

    def _kwargs(self) -> dict:
        return {
            "source_root": self.source,
            "source_inventory_sha256": self.source_sha,
            "container": self.container,
            "container_sha256": self.container_sha,
            "validation_contract": self.contract,
            "validation_contract_sha256": self.contract_sha,
            "learner_identity_sha256": self.identity_sha,
            "parent_runtime_policy_sha256": self.runtime_sha,
            "campaign_root": self.campaign,
            "snapshot": self.snapshot,
            "snapshot_file_sha256": self.snapshot_file_sha,
            "snapshot_sha256": self.snapshot_sha,
            "replay_shards": 64,
            "policy_checkpoint": self.policy,
            "policy_checkpoint_sha256": self.policy_sha,
            "policy_spec": self.spec,
            "policy_spec_sha256": sha256_file(self.spec),
            "policy_manifest": self.manifest,
            "policy_manifest_sha256": sha256_file(self.manifest),
            "bundle": self.bundle,
            "config_sha256": self.config_sha,
            "import_report": self.import_report,
            "wandb_report": self.wandb_report,
            "wandb_readback": self.wandb_readback,
            "uninterrupted_telemetry": self.full_telemetry,
            "resumed_telemetry": self.resumed_telemetry,
            "resume_evidence": self.resume_evidence,
            "uninterrupted_checkpoint": self.full_restart,
            "uninterrupted_result": self.full_result,
            "interrupted_checkpoint": self.interrupted,
            "resumed_checkpoint": self.resumed_restart,
            "resumed_result": self.resumed_result,
            "restart_report": self.restart_report,
            "campaign_id": self.campaign_id,
            "macro_round": self.macro_round,
        }

    def test_attestation_binds_exact_v2_execution_evidence(self) -> None:
        report = attestation.build_attestation(**self._kwargs())
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(
            report["learner_execution_contract"]["world_size"], 8
        )
        self.assertTrue(
            report["learner_execution_contract"][
                "single_canonical_wandb_run"
            ]
        )
        self.assertEqual(
            report["canonical_wandb_run_id"], self.run_id
        )
        self.assertEqual(
            report["uninterrupted_telemetry_sha256"],
            sha256_file(self.full_telemetry),
        )
        self.assertEqual(
            report["strict_resume_evidence_sha256"],
            sha256_file(self.resume_evidence),
        )

    def test_attestation_rejects_too_few_shards_and_restart_tampering(self) -> None:
        with self.assertRaisesRegex(
            attestation.H100ValidationAttestationError, "exactly 64 shards"
        ):
            attestation.build_attestation(
                **dict(self._kwargs(), replay_shards=7)
            )
        report = json.loads(
            self.restart_report.read_text(encoding="utf-8")
        )
        report["results_equivalent"] = False
        _json(self.restart_report, report)
        with self.assertRaisesRegex(
            attestation.H100ValidationAttestationError,
            "restart equivalence evidence mismatch",
        ):
            attestation.build_attestation(**self._kwargs())

    def test_attestation_rejects_writable_or_tampered_readback(self) -> None:
        self.wandb_readback.chmod(0o644)
        with self.assertRaisesRegex(
            attestation.H100ValidationAttestationError, "must be read-only"
        ):
            attestation.build_attestation(**self._kwargs())
        report = json.loads(
            self.wandb_readback.read_text(encoding="utf-8")
        )
        report["single_canonical_wandb_run"] = False
        _json(self.wandb_readback, report)
        self.wandb_readback.chmod(0o444)
        with self.assertRaisesRegex(
            attestation.H100ValidationAttestationError,
            "W&B learner readback evidence mismatch",
        ):
            attestation.build_attestation(**self._kwargs())

    def test_attestation_hash_binds_actual_telemetry_and_resume_evidence(self) -> None:
        self.full_telemetry.write_text(
            self.full_telemetry.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            attestation.H100ValidationAttestationError,
            "W&B learner readback evidence mismatch",
        ):
            attestation.build_attestation(**self._kwargs())

    def test_attestation_rejects_contract_or_checkpoint_lineage_drift(self) -> None:
        with self.assertRaisesRegex(
            attestation.H100ValidationAttestationError,
            "semantic identity mismatch",
        ):
            attestation.build_attestation(
                **dict(
                    self._kwargs(),
                    learner_identity_sha256="f" * 64,
                )
            )
        document = torch.load(
            self.resumed_restart,
            map_location="cpu",
            weights_only=False,
        )
        document["runtime_container_sha256"] = "e" * 64
        torch.save(document, self.resumed_restart)
        with self.assertRaisesRegex(
            attestation.H100ValidationAttestationError,
            "v2 lineage mismatch",
        ):
            attestation.build_attestation(**self._kwargs())


if __name__ == "__main__":
    unittest.main()
