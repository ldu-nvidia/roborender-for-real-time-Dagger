from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.round_dagger import h100_validation_wandb_readback as readback


def _json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
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
    def __init__(self, runs: dict[str, FakeRun]):
        self.viewer = {
            "email": "ldu@nvidia.com",
            "username": "leo",
            "teams": ["nvidia"],
        }
        self.runs = runs

    def run(self, path: str):
        return self.runs[path.rsplit("/", 1)[-1]]


class FakeWandb:
    def __init__(self, runs: dict[str, FakeRun]):
        self.api = FakeApi(runs)

    def Api(self, timeout):
        del timeout
        return self.api


class H100ValidationWandbReadbackTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="rr_wandb_readback_")
        self.root = Path(self.temporary.name)
        self.campaign_id = "validation"
        self.config_sha = "1" * 64
        self.lineage_sha = "2" * 64
        self.full_run_id = "validation-full"
        self.resumed_run_id = "validation-resumed"
        self.rows = [
            {
                "schema": readback.TELEMETRY_SCHEMA,
                "campaign_id": self.campaign_id,
                "macro_round": 0,
                "optimizer_step": 1,
                "loss": 1.0,
                "replay_transitions_rank": 1,
                "replay_transitions_per_rank": [1] * 8,
                "replay_transitions_global": 8,
                "world_size": 8,
            },
            {
                "schema": readback.TELEMETRY_SCHEMA,
                "campaign_id": self.campaign_id,
                "macro_round": 0,
                "optimizer_step": 2,
                "loss": 0.75,
                "replay_transitions_rank": 1,
                "replay_transitions_per_rank": [1] * 8,
                "replay_transitions_global": 8,
                "world_size": 8,
            },
        ]
        self.full_telemetry = self.root / "full.jsonl"
        self.resumed_telemetry = self.root / "resumed.jsonl"
        payload = "".join(json.dumps(row, sort_keys=True) + "\n" for row in self.rows)
        self.full_telemetry.write_text(payload, encoding="utf-8")
        self.resumed_telemetry.write_text(payload, encoding="utf-8")
        self.full_result = self.root / "full_result.json"
        self.resumed_result = self.root / "resumed_result.json"
        result = {
            "schema": "fast_roborender.round_dagger.learner_result.v2",
            "campaign_id": self.campaign_id,
            "macro_round": 0,
            "learner_config_sha256": self.config_sha,
            "model_lineage_sha256": self.lineage_sha,
            "optimizer_steps": 2,
            "initial_loss": 1.0,
            "final_loss": 0.5,
        }
        _json(self.full_result, result)
        _json(self.resumed_result, result)
        self.resume_evidence = self.root / "strict_resume.json"
        _json(
            self.resume_evidence,
            {
                "schema": readback.RESUME_EVIDENCE_SCHEMA,
                "status": "PASS",
                "run_id": self.resumed_run_id,
                "first_invocation_resume_mode": "allow",
                "second_invocation_resume_mode": "must",
                "stop_step": 1,
                "target_step": 2,
                "learner_config_sha256": self.config_sha,
                "stopped_checkpoint_sha256": "3" * 64,
                "resumed_checkpoint_sha256": "4" * 64,
            },
        )
        self.resume_evidence.chmod(0o444)
        config = {
            "campaign_id": self.campaign_id,
            "macro_round": 0,
            "learner_config_sha256": self.config_sha,
            "model_lineage_sha256": self.lineage_sha,
        }
        self.runs = {
            self.full_run_id: FakeRun(self.full_run_id, self.rows, config),
            self.resumed_run_id: FakeRun(self.resumed_run_id, self.rows, config),
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _kwargs(self) -> dict:
        return {
            "full_run_id": self.full_run_id,
            "resumed_run_id": self.resumed_run_id,
            "full_telemetry": self.full_telemetry,
            "resumed_telemetry": self.resumed_telemetry,
            "full_result": self.full_result,
            "resumed_result": self.resumed_result,
            "resume_evidence": self.resume_evidence,
            "campaign_id": self.campaign_id,
            "macro_round": 0,
            "learner_config_sha256": self.config_sha,
            "model_lineage_sha256": self.lineage_sha,
            "project": "validation-project",
            "expected_email": "ldu@nvidia.com",
            "entity": "nvidia",
            "authorization_scope": "team",
            "timeout_s": 0.02,
            "poll_interval_s": 0.001,
            "wandb_module": FakeWandb(self.runs),
        }

    def test_readback_binds_two_distinct_runs_and_contiguous_strict_resume(self) -> None:
        report = readback.readback(**self._kwargs())
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["runs_distinct"])
        self.assertEqual(report["full_run"]["history_steps"], [1, 2])
        self.assertEqual(report["resumed_run"]["history_steps"], [1, 2])
        self.assertTrue(report["strict_resume_verified"])
        self.assertEqual(report["strict_resume_mode"], "must")

    def test_missing_remote_resumed_step_fails_closed(self) -> None:
        self.runs[self.resumed_run_id].rows = self.rows[:1]
        with self.assertRaisesRegex(
            readback.H100ValidationReadbackError, "readback timed out"
        ):
            readback.readback(**self._kwargs())

    def test_local_transition_balance_tampering_fails_closed(self) -> None:
        cases = {
            "wrong rank count": {
                "replay_transitions_rank": 2,
            },
            "wrong vector length": {
                "replay_transitions_per_rank": [1] * 7,
            },
            "wrong global total": {
                "replay_transitions_global": 9,
            },
        }
        for label, change in cases.items():
            with self.subTest(label=label):
                rows = [dict(row) for row in self.rows]
                rows[0].update(change)
                path = self.root / f"{label.replace(' ', '_')}.jsonl"
                payload = "".join(
                    json.dumps(row, sort_keys=True) + "\n" for row in rows
                )
                path.write_text(payload, encoding="utf-8")
                with self.assertRaisesRegex(
                    readback.H100ValidationReadbackError,
                    "transition balance is invalid",
                ):
                    readback._telemetry(
                        path, campaign_id=self.campaign_id, macro_round=0
                    )

    def test_remote_config_and_strict_resume_evidence_tampering_fail(self) -> None:
        self.runs[self.full_run_id].config["learner_config_sha256"] = "9" * 64
        with self.assertRaisesRegex(
            readback.H100ValidationReadbackError, "readback timed out"
        ):
            readback.readback(**self._kwargs())

        self.runs[self.full_run_id].config["learner_config_sha256"] = self.config_sha
        evidence = json.loads(self.resume_evidence.read_text(encoding="utf-8"))
        evidence["second_invocation_resume_mode"] = "allow"
        self.resume_evidence.chmod(0o644)
        _json(self.resume_evidence, evidence)
        self.resume_evidence.chmod(0o444)
        with self.assertRaisesRegex(
            readback.H100ValidationReadbackError,
            "strict resume invocation evidence mismatch",
        ):
            readback.readback(**self._kwargs())


if __name__ == "__main__":
    unittest.main()
