from __future__ import annotations

import shutil
import tempfile
import types
import unittest
from pathlib import Path

from round_dagger import lepton_job
from round_dagger.lepton_monitor import (
    HEALTH_SCHEMA,
    LeptonMonitorError,
    collect_health,
)
from tests.test_round_dagger_lepton_job import _control_deployment


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONTRACT = (
    ROOT / "configs/round_dagger/production_learner_1x8h200_lepton.json"
)


class _Jobs:
    def __init__(self, jobs=()):
        self.jobs = list(jobs)
        self.calls: list[dict[str, object]] = []

    def list_all(self, **kwargs):
        self.calls.append(dict(kwargs))
        return list(self.jobs)


class _Deployments:
    def __init__(self, deployment):
        self.deployment = deployment

    def get(self, name):
        if name != lepton_job.CONTROL_DEPLOYMENT:
            raise AssertionError("unexpected deployment")
        return self.deployment


class _API:
    def __init__(self, *, workspace="b5k2m9x7", jobs=()):
        self.workspace = workspace
        self.deployment = _Deployments(_control_deployment("Ready", minimum=1))
        self.job = _Jobs(jobs)

    def get_workspace_id(self):
        return self.workspace


class LeptonMonitorTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="rr-lepton-monitor-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.contract = self.root / "contract.json"
        shutil.copyfile(SOURCE_CONTRACT, self.contract)
        self.contract.chmod(0o444)
        self.contract_sha = lepton_job.load_contract(self.contract).sha256
        self.handoff = self.root / "handoff"
        self.handoff.mkdir()

    def _collect(self, api):
        return collect_health(
            backend_contract=self.contract,
            backend_contract_sha256=self.contract_sha,
            handoff_root=self.handoff,
            campaign_id="campaign",
            macro_round=0,
            api_client=api,
        )

    def test_idle_reserved_node_is_healthy_and_queries_are_read_only(self) -> None:
        api = _API()
        report = self._collect(api)
        self.assertEqual(report["schema"], HEALTH_SCHEMA)
        self.assertTrue(report["reachable"])
        self.assertTrue(report["healthy"])
        self.assertTrue(report["read_only"])
        self.assertEqual(report["control_state"], "Ready")
        self.assertEqual(report["active_job_count"], 0)
        self.assertEqual(report["reason_codes"], [])
        self.assertEqual(
            api.job.calls,
            [{"job_query_mode": "alive_only", "status": sorted(
                {
                    "Starting", "Running", "Queueing", "Awaiting",
                    "PendingRetry", "Restarting",
                }
            )}],
        )

    def test_workspace_mismatch_fails_closed(self) -> None:
        with self.assertRaisesRegex(LeptonMonitorError, "control plane"):
            self._collect(_API(workspace="other"))

    def test_orphan_reserved_job_is_unhealthy_and_sanitized(self) -> None:
        job = types.SimpleNamespace(
            metadata=types.SimpleNamespace(name="not-an-owned-job", labels={}),
            spec=types.SimpleNamespace(
                reservation_config=types.SimpleNamespace(
                    reservation_id=lepton_job.RESERVATION_ID,
                    allow_burst_to_other_reservations=False,
                ),
                affinity=types.SimpleNamespace(
                    allowed_dedicated_node_groups=[lepton_job.NODE_GROUP]
                ),
                resource_shape=lepton_job.RESOURCE_SHAPE,
            ),
            status=types.SimpleNamespace(state="Running"),
        )
        report = self._collect(_API(jobs=[job]))
        self.assertFalse(report["healthy"])
        self.assertIn("reserved_job_identity_invalid", report["reason_codes"])
        self.assertEqual(report["malformed_active_jobs"][0]["name"], "<malformed>")


if __name__ == "__main__":
    unittest.main()
