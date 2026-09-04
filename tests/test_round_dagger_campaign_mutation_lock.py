"""Regression tests for the campaign-wide orchestration mutation lease."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from round_dagger import evaluation_recovery, learner_recovery, orchestration


ROOT = Path(__file__).resolve().parents[1]
SHA = "1" * 64


class CampaignMutationLeaseTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(
            prefix="round_dagger_mutation_lock_"
        )
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "campaign"
        self.root.mkdir()

    @property
    def lock_path(self) -> Path:
        return self.root / "operator" / ".campaign_mutation.lock"

    def _hold_in_thread(
        self,
    ) -> tuple[threading.Event, threading.Event, threading.Thread]:
        entered = threading.Event()
        release = threading.Event()

        def hold() -> None:
            with orchestration.campaign_mutation_lease(
                self.root, "thread_holder"
            ):
                entered.set()
                release.wait(timeout=10)

        thread = threading.Thread(target=hold, daemon=True)
        thread.start()
        self.assertTrue(entered.wait(timeout=5), "holder did not acquire lease")
        return entered, release, thread

    def test_cross_process_contention_is_nonblocking_and_reports_owner(self) -> None:
        program = """
import sys
from round_dagger.orchestration import (
    CampaignMutationBusyError,
    campaign_mutation_lease,
)
try:
    with campaign_mutation_lease(sys.argv[1], "child_contender"):
        pass
except CampaignMutationBusyError as exc:
    print(str(exc))
    raise SystemExit(23)
raise SystemExit(0)
"""
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(ROOT), environment.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        with orchestration.campaign_mutation_lease(
            self.root, "parent_holder"
        ) as owner:
            completed = subprocess.run(
                [sys.executable, "-c", program, str(self.root)],
                cwd=ROOT,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=10,
            )
            self.assertEqual(completed.returncode, 23, completed.stderr)
            self.assertIn("campaign mutation lease is busy", completed.stdout)
            self.assertIn("operation=parent_holder", completed.stdout)
            self.assertIn(f"pid={os.getpid()}", completed.stdout)
            persisted = json.loads(self.lock_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted, dict(owner))
            self.assertEqual(persisted["liveness_authority"], "kernel_flock")
            self.assertIn("pid_start_ticks", persisted)
            self.assertIn("boot_id", persisted)

    def test_exception_releases_lease_and_same_thread_nesting_is_reentrant(
        self,
    ) -> None:
        with self.assertRaisesRegex(RuntimeError, "injected body failure"):
            with orchestration.campaign_mutation_lease(
                self.root, "failing_outer"
            ) as outer:
                with orchestration.campaign_mutation_lease(
                    self.root, "nested_helper"
                ) as nested:
                    self.assertEqual(outer["lease_id"], nested["lease_id"])
                raise RuntimeError("injected body failure")

        released = json.loads(self.lock_path.read_text(encoding="utf-8"))
        self.assertEqual(released["state"], "released")
        self.assertEqual(released["operation"], "failing_outer")
        self.assertIn("released_at", released)

        with orchestration.campaign_mutation_lease(
            self.root, "post_failure_retry"
        ):
            pass
        retried = json.loads(self.lock_path.read_text(encoding="utf-8"))
        self.assertEqual(retried["state"], "released")
        self.assertEqual(retried["operation"], "post_failure_retry")
        self.assertFalse(self.lock_path.is_symlink())
        self.assertEqual(self.lock_path.stat().st_nlink, 1)
        self.assertEqual(self.lock_path.stat().st_mode & 0o777, 0o600)

    def test_symlink_lock_is_rejected_without_following_it(self) -> None:
        self.lock_path.parent.mkdir(mode=0o700)
        target = self.root / "outside"
        target.write_text("must-not-change\n", encoding="utf-8")
        self.lock_path.symlink_to(target)
        with self.assertRaisesRegex(
            orchestration.CampaignMutationLockError,
            "campaign root|mutation lock|symlink",
        ):
            with orchestration.campaign_mutation_lease(
                self.root, "unsafe_lock_probe"
            ):
                self.fail("unsafe symlink lease was acquired")
        self.assertEqual(target.read_text(encoding="utf-8"), "must-not-change\n")

    def test_all_campaign_mutators_fail_before_body_when_lease_is_busy(self) -> None:
        _, release, thread = self._hold_in_thread()
        try:
            runner = object.__new__(orchestration.MacroRoundOrchestrator)
            runner.config = SimpleNamespace(
                campaign_root=self.root,
                macro_round=0,
            )
            with mock.patch.object(
                runner,
                "_run_with_mutation_lease",
                side_effect=AssertionError("macro body must not run"),
            ) as macro_body:
                with self.assertRaisesRegex(
                    orchestration.OrchestrationError, "lease is busy"
                ):
                    runner.run()
            macro_body.assert_not_called()

            learner_arguments = {
                "campaign_root": self.root,
                "macro_round": 0,
                "expected_state_revision": 0,
                "expected_campaign_sha256": SHA,
                "expected_replay_snapshot_file_sha256": SHA,
                "expected_replay_snapshot_sha256": SHA,
                "expected_parent_checkpoint_sha256": SHA,
                "expected_parent_runtime_policy_sha256": SHA,
                "campaign_contract_path": self.root / "missing-campaign.json",
                "expected_campaign_contract_sha256": SHA,
                "learner_contract_path": self.root / "missing-learner.json",
                "expected_learner_contract_sha256": SHA,
                "sealed_source_root": "/missing/source",
                "sealed_source_inventory_sha256": SHA,
                "evaluation_sealed_source_root": self.root / "missing-eval",
                "evaluation_sealed_source_inventory_sha256": SHA,
                "expected_bridge_sha256": SHA,
            }
            with self.assertRaisesRegex(
                learner_recovery.LearnerRecoveryError, "lease is busy"
            ):
                learner_recovery.recover_ingested_learner(**learner_arguments)

            evaluation_arguments = {
                "root": self.root,
                "repository_root": self.root / "missing-source",
                "gr00t_root": self.root / "missing-source/gr00t",
                "hold_attestation_path": self.root / "missing-hold.json",
                "expected_hold_attestation_sha256": SHA,
                "scenario_cases_path": self.root / "missing-scenarios.json",
                "thresholds_path": self.root / "missing-thresholds.json",
                "evaluation_config_path": self.root / "missing-eval.json",
                "renderer_runtime_sha256": SHA,
                "expected_renderer_count": 1,
                "learner_recovery_receipt_path": self.root / "missing-receipt.json",
                "expected_learner_recovery_receipt_sha256": SHA,
                "evaluation_source_inventory_sha256": SHA,
                "anchor_pool_path": self.root / "missing-anchors",
                "anchor_count": 1,
                "anchor_index": 0,
                "anchor_inventory_sha256": SHA,
            }
            with self.assertRaisesRegex(
                evaluation_recovery.EvaluationRecoveryError, "lease is busy"
            ):
                evaluation_recovery.run_trusted_recovery(**evaluation_arguments)
        finally:
            release.set()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive(), "holder thread did not exit")


if __name__ == "__main__":
    unittest.main()
