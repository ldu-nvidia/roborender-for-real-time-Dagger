from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from round_dagger.artifacts import sha256_file
from round_dagger.learner_identity import (
    learner_identity_sha256,
    learner_wandb_run_id,
)
from scripts.round_dagger import h100_validation_wandb_readback as readback
from tests.policy_bundle_helpers import HOMIE_CONTROLLER_SHA256

try:
    import torch
except ImportError:
    torch = None


def _json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )


class FakeRun:
    def __init__(self, run_id: str, rows: list[dict], config: dict):
        self.id = run_id
        self.state = "finished"
        self.config = config
        self.rows = rows

    def scan_history(self, *, keys, page_size):
        del keys, page_size
        return list(self.rows)


class FakeApi:
    def __init__(self, run: FakeRun):
        self.viewer = {
            "email": "ldu@nvidia.com",
            "username": "leo",
            "teams": ["nvidia"],
        }
        self.expected_run = run
        self.run_queries: list[str] = []

    def run(self, path: str):
        self.run_queries.append(path)
        if path != f"nvidia/validation-project/{self.expected_run.id}":
            raise KeyError(path)
        return self.expected_run


class FakeWandb:
    def __init__(self, run: FakeRun):
        self.api = FakeApi(run)

    def Api(self, timeout):
        del timeout
        return self.api


@unittest.skipIf(torch is None, "W&B readback tests require PyTorch")
class H100ValidationWandbReadbackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="rr_wandb_readback_v2_")
        self.root = Path(self.temporary.name)
        self.campaign_id = "validation"
        self.macro_round = 0
        self.code_sha = "1" * 64
        self.config_sha = "2" * 64
        self.lineage_sha = "3" * 64
        self.snapshot_file_sha = "4" * 64
        self.snapshot_sha = "5" * 64
        self.parent_runtime_sha = "6" * 64
        self.container_sha = "7" * 64
        self.contract_sha = "8" * 64
        self.identity_sha = learner_identity_sha256(
            campaign_id=self.campaign_id,
            macro_round=self.macro_round,
            learner_code_sha256=self.code_sha,
            learner_config_sha256=self.config_sha,
            replay_snapshot_file_sha256=self.snapshot_file_sha,
            replay_snapshot_sha256=self.snapshot_sha,
            parent_checkpoint_sha256=self.lineage_sha,
            parent_runtime_policy_sha256=self.parent_runtime_sha,
            runtime_container_sha256=self.container_sha,
            learner_contract_sha256=self.contract_sha,
        )
        self.run_id = learner_wandb_run_id(
            campaign_id=self.campaign_id,
            macro_round=self.macro_round,
            learner_identity_sha256=self.identity_sha,
        )
        self.rows = [
            {
                "schema": readback.TELEMETRY_SCHEMA,
                "campaign_id": self.campaign_id,
                "macro_round": self.macro_round,
                "optimizer_step": 1,
                "loss": 1.0,
                "replay_transitions_rank": 16384,
                "replay_transitions_per_rank": [16384] * 8,
                "replay_transitions_global": 131072,
                "world_size": 8,
            },
            {
                "schema": readback.TELEMETRY_SCHEMA,
                "campaign_id": self.campaign_id,
                "macro_round": self.macro_round,
                "optimizer_step": 2,
                "loss": 0.75,
                "replay_transitions_rank": 16384,
                "replay_transitions_per_rank": [16384] * 8,
                "replay_transitions_global": 131072,
                "world_size": 8,
            },
        ]
        # Optimizer telemetry is deliberately a training-batch loss.  The
        # result/checkpoint baseline is the separate deterministic eval probe
        # and therefore need not equal the first optimizer row.
        self.probe_initial_loss = 1.25
        self.full_telemetry = self.root / "full.jsonl"
        self.resumed_telemetry = self.root / "resumed.jsonl"
        payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in self.rows)
        self.full_telemetry.write_text(payload, encoding="utf-8")
        self.resumed_telemetry.write_text(payload, encoding="utf-8")

        self.full_checkpoint = self.root / "full_restart.pt"
        self.stopped_checkpoint = self.root / "stopped_restart.pt"
        self.resumed_checkpoint = self.root / "resumed_restart.pt"
        self._save_checkpoint(self.full_checkpoint, completed_steps=2)
        shutil.copyfile(self.full_checkpoint, self.resumed_checkpoint)
        self._save_checkpoint(self.stopped_checkpoint, completed_steps=1)
        self.stopped_checkpoint.chmod(0o444)

        self.full_result = self._publish_result(
            "full",
            self.full_checkpoint,
            initial_loss=self.probe_initial_loss,
            final_loss=0.75,
        )
        self.resumed_result = self._publish_result(
            "resumed",
            self.resumed_checkpoint,
            initial_loss=self.probe_initial_loss,
            final_loss=0.75,
        )
        self.resume_evidence = self.root / "strict_resume.json"
        self._publish_resume_evidence()

        self.remote_config = {
            "campaign_id": self.campaign_id,
            "macro_round": self.macro_round,
            "learner_code_sha256": self.code_sha,
            "learner_config_sha256": self.config_sha,
            "model_lineage_sha256": self.lineage_sha,
            "replay_snapshot_file_sha256": self.snapshot_file_sha,
            "replay_snapshot_sha256": self.snapshot_sha,
            "parent_runtime_policy_sha256": self.parent_runtime_sha,
            "learner_identity_sha256": self.identity_sha,
            "runtime_container_sha256": self.container_sha,
            "learner_contract_sha256": self.contract_sha,
        }
        self.remote_run = FakeRun(self.run_id, self.rows, self.remote_config)
        self.wandb = FakeWandb(self.remote_run)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _checkpoint_document(self, completed_steps: int) -> dict:
        policy_state = {"weight": torch.tensor([1.0], dtype=torch.float32)}
        return {
            "schema": "fast_roborender.round_dagger.learner_checkpoint.v3",
            "checkpoint_schema": "gr00t.trl.full_state.v2",
            "policy_state_dict": policy_state,
            "policy_parameter_names": ["weight"],
            "homie_controller_sha256": HOMIE_CONTROLLER_SHA256,
            "campaign_id": self.campaign_id,
            "macro_round": self.macro_round,
            "model_lineage_sha256": self.lineage_sha,
            "code_sha256": self.code_sha,
            "learner_config_sha256": self.config_sha,
            "learner_identity_sha256": self.identity_sha,
            "runtime_container_sha256": self.container_sha,
            "learner_contract_sha256": self.contract_sha,
            "replay_snapshot_sha256": self.snapshot_sha,
            "replay_snapshot_file_sha256": self.snapshot_file_sha,
            "completed_steps": completed_steps,
            "initial_loss": self.probe_initial_loss,
            "world_size": 8,
            "model_state_dict": policy_state,
            "optimizer_state_dict": {"state": {}, "param_groups": []},
            "lr_scheduler_state_dict": None,
            "value_state_dict": None,
            "rng_states": [{} for _ in range(8)],
        }

    def _save_checkpoint(self, path: Path, *, completed_steps: int) -> None:
        torch.save(self._checkpoint_document(completed_steps), path)

    def _publish_result(
        self,
        name: str,
        restart: Path,
        *,
        initial_loss: float,
        final_loss: float,
    ) -> Path:
        directory = self.root / name
        directory.mkdir()
        candidate = directory / "candidate_round_000000.pt"
        shutil.copyfile(restart, candidate)
        policy_spec = directory / f"{candidate.name}.policy_spec.json"
        policy_manifest = directory / f"{candidate.name}.manifest.json"
        _json(policy_spec, {"fixture": "portable-policy-spec"})
        _json(policy_manifest, {"fixture": "portable-policy-manifest"})
        result = directory / "learner_result_round_000000.json"
        _json(
            result,
            {
                "schema": "fast_roborender.round_dagger.learner_result.v2",
                "campaign_id": self.campaign_id,
                "macro_round": self.macro_round,
                "checkpoint_path": candidate.name,
                "checkpoint_sha256": sha256_file(candidate),
                "runtime_policy_sha256": "a" * 64,
                "policy_spec_path": policy_spec.name,
                "policy_spec_sha256": sha256_file(policy_spec),
                "policy_bundle_manifest_path": policy_manifest.name,
                "policy_bundle_manifest_sha256": sha256_file(policy_manifest),
                "model_lineage_sha256": self.lineage_sha,
                "replay_snapshot_sha256": self.snapshot_sha,
                "code_sha256": self.code_sha,
                "learner_config_sha256": self.config_sha,
                "optimizer_steps": 2,
                "initial_loss": initial_loss,
                "final_loss": final_loss,
            },
        )
        return result

    def _publish_resume_evidence(self) -> None:
        _json(
            self.resume_evidence,
            {
                "schema": readback.RESUME_EVIDENCE_SCHEMA,
                "status": "PASS",
                "run_id": self.run_id,
                "first_invocation_resume_mode": "allow",
                "second_invocation_resume_mode": "must",
                "stop_step": 1,
                "target_step": 2,
                "learner_config_sha256": self.config_sha,
                "learner_identity_sha256": self.identity_sha,
                "learner_code_sha256": self.code_sha,
                "replay_snapshot_file_sha256": self.snapshot_file_sha,
                "replay_snapshot_sha256": self.snapshot_sha,
                "parent_checkpoint_sha256": self.lineage_sha,
                "parent_runtime_policy_sha256": self.parent_runtime_sha,
                "runtime_container_sha256": self.container_sha,
                "learner_contract_sha256": self.contract_sha,
                "uninterrupted_comparator_wandb": False,
                "canonical_wandb_lineage": True,
                "stopped_checkpoint_sha256": sha256_file(self.stopped_checkpoint),
                "resumed_checkpoint_sha256": sha256_file(self.resumed_checkpoint),
            },
        )
        self.resume_evidence.chmod(0o444)

    def _kwargs(self) -> dict:
        return {
            "run_id": self.run_id,
            "uninterrupted_telemetry": self.full_telemetry,
            "resumed_telemetry": self.resumed_telemetry,
            "uninterrupted_result": self.full_result,
            "resumed_result": self.resumed_result,
            "resume_evidence": self.resume_evidence,
            "uninterrupted_checkpoint": self.full_checkpoint,
            "stopped_checkpoint": self.stopped_checkpoint,
            "resumed_checkpoint": self.resumed_checkpoint,
            "campaign_id": self.campaign_id,
            "macro_round": self.macro_round,
            "learner_code_sha256": self.code_sha,
            "learner_config_sha256": self.config_sha,
            "model_lineage_sha256": self.lineage_sha,
            "replay_snapshot_file_sha256": self.snapshot_file_sha,
            "replay_snapshot_sha256": self.snapshot_sha,
            "parent_runtime_policy_sha256": self.parent_runtime_sha,
            "learner_identity_sha256": self.identity_sha,
            "runtime_container_sha256": self.container_sha,
            "validation_contract_sha256": self.contract_sha,
            "project": "validation-project",
            "expected_email": "ldu@nvidia.com",
            "entity": "nvidia",
            "authorization_scope": "team",
            "timeout_s": 0.02,
            "poll_interval_s": 0.001,
            "wandb_module": self.wandb,
        }

    def test_single_canonical_run_binds_restart_and_comparator_evidence(self) -> None:
        report = readback.readback(**self._kwargs())
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["schema"], readback.SCHEMA)
        self.assertEqual(report["wandb_run_id"], self.run_id)
        self.assertTrue(report["single_canonical_wandb_run"])
        self.assertFalse(report["uninterrupted_comparator"]["wandb_enabled"])
        self.assertTrue(report["resumed_execution"]["wandb_enabled"])
        self.assertEqual(report["canonical_run"]["history_steps"], [1, 2])
        self.assertEqual(report["initial_loss"], self.probe_initial_loss)
        self.assertNotEqual(report["initial_loss"], self.rows[0]["loss"])
        self.assertEqual(
            report["stopped_checkpoint_sha256"],
            sha256_file(self.stopped_checkpoint),
        )
        self.assertEqual(
            report["resumed_checkpoint_sha256"],
            sha256_file(self.resumed_checkpoint),
        )
        self.assertEqual(
            self.wandb.api.run_queries,
            [f"nvidia/validation-project/{self.run_id}"],
        )

    def test_tampered_semantic_identity_fails_before_remote_query(self) -> None:
        arguments = self._kwargs()
        arguments["learner_identity_sha256"] = "9" * 64
        with self.assertRaisesRegex(
            readback.H100ValidationReadbackError,
            "does not bind the exact validation semantics",
        ):
            readback.readback(**arguments)
        self.assertEqual(self.wandb.api.run_queries, [])

    def test_tampered_remote_config_or_history_fails_closed(self) -> None:
        self.remote_run.config["learner_config_sha256"] = "9" * 64
        with self.assertRaisesRegex(
            readback.H100ValidationReadbackError, "readback timed out"
        ):
            readback.readback(**self._kwargs())

        self.remote_run.config["learner_config_sha256"] = self.config_sha
        self.remote_run.rows = self.rows[:1]
        with self.assertRaisesRegex(
            readback.H100ValidationReadbackError, "readback timed out"
        ):
            readback.readback(**self._kwargs())

    def test_tampered_checkpoint_lineage_fails_closed(self) -> None:
        document = torch.load(
            self.resumed_checkpoint, map_location="cpu", weights_only=False
        )
        document["learner_contract_sha256"] = "9" * 64
        torch.save(document, self.resumed_checkpoint)
        with self.assertRaisesRegex(
            readback.H100ValidationReadbackError,
            "resumed checkpoint v2 lineage mismatch",
        ):
            readback.readback(**self._kwargs())

    def test_tampered_strict_resume_evidence_fails_closed(self) -> None:
        document = json.loads(self.resume_evidence.read_text(encoding="utf-8"))
        document["second_invocation_resume_mode"] = "allow"
        self.resume_evidence.chmod(0o644)
        _json(self.resume_evidence, document)
        self.resume_evidence.chmod(0o444)
        with self.assertRaisesRegex(
            readback.H100ValidationReadbackError,
            "strict resume invocation evidence mismatch",
        ):
            readback.readback(**self._kwargs())


if __name__ == "__main__":
    unittest.main()
