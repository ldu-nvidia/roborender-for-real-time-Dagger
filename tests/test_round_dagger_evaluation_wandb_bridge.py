"""Focused tests for the read-only production evaluation W&B bridge."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from round_dagger.evaluation import EvaluationError
from scripts.round_dagger import evaluation_wandb_bridge as bridge

try:
    from test_round_dagger_state import _campaign
except ImportError:  # pragma: no cover - package-style unittest invocation
    from tests.test_round_dagger_state import _campaign  # type: ignore[no-redef]


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class FakeRun:
    def __init__(self) -> None:
        self.logs: list[dict[str, int | float]] = []
        self.steps: list[int] = []

    def log(
        self, payload: dict[str, int | float], *, step: int | None = None
    ) -> None:
        self.logs.append(dict(payload))
        if step is not None:
            self.steps.append(step)


def _inventory(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class EvaluationWandbBridgeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.tmp = Path(self.temporary.name)
        self.root = _campaign(self.tmp)
        self.evaluation = self.root / "rounds/macro_round_000000/evaluation"
        (self.evaluation / "results").mkdir(parents=True)
        (self.evaluation / "contract.json").write_text("{}\n", encoding="utf-8")
        (self.evaluation / "result.json").write_text("{}\n", encoding="utf-8")
        self.state_dir = self.tmp / "observer-state"
        self.driver_status = self.tmp / "driver_status.json"
        self.driver_status.write_text(
            json.dumps(
                {
                    "schema": bridge.DRIVER_STATUS_SCHEMA,
                    "campaign_root": str(self.root),
                    "state": "LAUNCHING",
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _verified(self, result_sha: str | None = None) -> dict[str, object]:
        return {
            "macro_round": 0,
            "sha256": result_sha or _sha("result-v1"),
            "evidence_bundle_sha256": _sha("bundle"),
            "passed": True,
            "metrics": {
                "paired_cases": 256.0,
                "parent_success_rate": 0.50,
                "candidate_success_rate": 0.625,
                "paired_success_rate_delta": 0.125,
                "parent_safety_violation_rate": 0.10,
                "candidate_safety_violation_rate": 0.05,
                "paired_safety_violation_rate_delta": -0.05,
                "parent_action_imitation_mse": 0.20,
                "candidate_action_imitation_mse": 0.10,
                "paired_action_imitation_mse_delta": -0.10,
                "paired_action_imitation_mse_relative_change": -0.50,
                "parent_episode_return_mean": 10.0,
                "candidate_episode_return_mean": 12.5,
                "paired_episode_return_mean_delta": 2.5,
                "parent_episode_length_mean": 300.0,
                "candidate_episode_length_mean": 280.0,
                "paired_episode_length_mean_delta": -20.0,
                "parent_lag_boundary_cases": 256.0,
                "candidate_lag_boundary_cases": 256.0,
                "paired_initial_condition_matches": 256.0,
                "paired_initial_generated_matches": 256.0,
                "renderer_errors": 0.0,
            },
        }

    @staticmethod
    def _contract() -> dict[str, object]:
        return {
            "candidate_manifest_path": "policies/policy_000001/candidate.json",
            "policies": [
                {"role": "parent", "version": 0},
                {"role": "candidate", "version": 1},
            ],
        }

    def _patch_evidence(self, *, result_sha: str | None = None):
        return (
            mock.patch.object(
                bridge,
                "verify_evaluator_result_evidence",
                return_value=self._verified(result_sha),
            ),
            mock.patch.object(bridge, "load_contract", return_value=self._contract()),
            mock.patch.object(
                bridge,
                "load_candidate_manifest",
                return_value={"optimizer_steps": 1200},
            ),
        )

    def test_uploads_clear_quality_metrics_once_and_never_mutates_campaign(self) -> None:
        before = _inventory(self.root)
        observer = bridge.EvaluationWandbBridge(
            campaign_root=self.root,
            state_dir=self.state_dir,
            entity="nvidia",
            project="roborender-round-dagger",
        )
        fake_run = FakeRun()
        patches = self._patch_evidence()
        with patches[0] as verify, patches[1], patches[2]:
            self.assertEqual(observer.poll_once(fake_run), 1)
            self.assertEqual(observer.poll_once(fake_run), 0)
        self.assertFalse(observer.state_path.exists())
        observer.commit_session()

        self.assertEqual(verify.call_count, 2, "every poll must re-verify evidence")
        self.assertEqual(len(fake_run.logs), 1)
        self.assertEqual(fake_run.steps, [])
        payload = fake_run.logs[0]
        self.assertEqual(payload["eval/macro_round"], 0)
        self.assertEqual(payload["eval/case_count"], 256)
        self.assertEqual(payload["eval/pass"], 1)
        self.assertEqual(payload["eval/parent/success_rate"], 0.50)
        self.assertEqual(payload["eval/candidate/success_rate"], 0.625)
        self.assertEqual(payload["eval/delta/success_rate"], 0.125)
        self.assertEqual(payload["eval/parent/action_imitation_mse"], 0.20)
        self.assertEqual(payload["eval/candidate/action_imitation_mse"], 0.10)
        self.assertEqual(payload["eval/delta/action_imitation_mse"], -0.10)
        self.assertEqual(payload["eval/parent/episode_return_mean"], 10.0)
        self.assertEqual(payload["eval/candidate/episode_return_mean"], 12.5)
        self.assertEqual(payload["eval/delta/episode_return_mean"], 2.5)
        self.assertEqual(payload["eval/parent/episode_length_mean"], 300.0)
        self.assertEqual(payload["eval/candidate/episode_length_mean"], 280.0)
        self.assertEqual(payload["eval/delta/episode_length_mean"], -20.0)
        self.assertEqual(payload["eval/candidate/optimizer_steps"], 1200)
        self.assertEqual(_inventory(self.root), before)
        self.assertTrue(observer.state_path.is_file())
        self.assertNotEqual(self.state_dir, self.root)

    def test_invalid_result_fails_before_wandb_or_cursor_update(self) -> None:
        observer = bridge.EvaluationWandbBridge(
            campaign_root=self.root,
            state_dir=self.state_dir,
            entity="nvidia",
            project="roborender-round-dagger",
        )
        fake_run = FakeRun()
        with mock.patch.object(
            bridge,
            "verify_evaluator_result_evidence",
            side_effect=EvaluationError("bundle changed"),
        ):
            with self.assertRaisesRegex(bridge.BridgeError, "verification failed"):
                observer.poll_once(fake_run)
        self.assertEqual(fake_run.logs, [])
        self.assertFalse(observer.state_path.exists())

    def test_resume_identity_is_deterministic_and_conflicts_fail_closed(self) -> None:
        first = bridge.EvaluationWandbBridge(
            campaign_root=self.root,
            state_dir=self.state_dir,
            entity="nvidia",
            project="roborender-round-dagger",
        )
        second = bridge.EvaluationWandbBridge(
            campaign_root=self.root,
            state_dir=self.state_dir,
            entity="nvidia",
            project="roborender-round-dagger",
        )
        self.assertEqual(first.run_id, second.run_id)
        self.assertLessEqual(len(first.run_id), 128)
        fake_run = FakeRun()
        patches = self._patch_evidence(result_sha=_sha("result-v1"))
        with patches[0], patches[1], patches[2]:
            self.assertEqual(first.poll_once(fake_run), 1)
        first.commit_session()

        patches = self._patch_evidence(result_sha=_sha("result-v2"))
        with patches[0], patches[1], patches[2]:
            with self.assertRaisesRegex(bridge.BridgeError, "changed after upload"):
                second.poll_once(fake_run)
        self.assertEqual(len(fake_run.logs), 1)

    def test_cursor_is_committed_only_after_wandb_drain(self) -> None:
        first = bridge.EvaluationWandbBridge(
            campaign_root=self.root,
            state_dir=self.state_dir,
            entity="nvidia",
            project="roborender-round-dagger",
        )
        first_run = FakeRun()
        patches = self._patch_evidence()
        with patches[0], patches[1], patches[2]:
            self.assertEqual(first.poll_once(first_run), 1)
        self.assertFalse(first.state_path.exists())

        # Simulate SIGKILL before W&B's finish/drain boundary: a new process
        # must replay the record rather than trusting an unacknowledged cursor.
        restarted = bridge.EvaluationWandbBridge(
            campaign_root=self.root,
            state_dir=self.state_dir,
            entity="nvidia",
            project="roborender-round-dagger",
        )
        restarted_run = FakeRun()
        patches = self._patch_evidence()
        with patches[0], patches[1], patches[2]:
            self.assertEqual(restarted.poll_once(restarted_run), 1)
        restarted.commit_session()
        persisted = bridge.load_bridge_state(
            restarted.state_path, restarted.expected_state
        )
        self.assertEqual(
            persisted["uploaded_result_sha256_by_macro_round"],
            {"0": _sha("result-v1")},
        )

    def test_external_state_lock_is_exclusive(self) -> None:
        observer = bridge.EvaluationWandbBridge(
            campaign_root=self.root,
            state_dir=self.state_dir,
            entity="nvidia",
            project="roborender-round-dagger",
        )
        with bridge.exclusive_bridge_lock(observer.state_dir):
            with self.assertRaisesRegex(bridge.BridgeError, "another evaluation"):
                with bridge.exclusive_bridge_lock(observer.state_dir):
                    self.fail("a second observer unexpectedly acquired the lock")

    def test_driver_lifecycle_is_bound_to_the_exact_campaign(self) -> None:
        self.assertEqual(
            bridge.load_driver_state(
                self.driver_status, campaign_root=self.root
            ),
            "LAUNCHING",
        )
        document = json.loads(self.driver_status.read_text(encoding="utf-8"))
        document["campaign_root"] = str(self.tmp / "another-campaign")
        self.driver_status.write_text(json.dumps(document) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(bridge.BridgeError, "mismatched"):
            bridge.load_driver_state(self.driver_status, campaign_root=self.root)

    def test_ready_marker_is_written_only_after_first_successful_poll(self) -> None:
        ready = self.state_dir / "EVALUATION_WANDB_READY.json"

        class Run:
            def __init__(self) -> None:
                self.summary: dict[str, object] = {}
                self.finished: list[int] = []

            def define_metric(self, *_args, **_kwargs) -> None:
                return None

            def log(self, *_args, **_kwargs) -> None:
                return None

            def finish(self, *, exit_code: int) -> None:
                self.finished.append(exit_code)

        run = Run()
        wandb = mock.Mock()
        wandb.init.return_value = run
        arguments = [
            "--campaign-root", str(self.root),
            "--state-dir", str(self.state_dir),
            "--project", "roborender-round-dagger",
            "--entity", "nvidia",
            "--expected-email", "ldu@nvidia.com",
            "--authorization-scope", "team",
            "--driver-status", str(self.driver_status),
            "--ready-marker", str(ready),
        ]
        with (
            mock.patch.object(bridge, "_load_wandb", return_value=wandb),
            mock.patch.object(bridge, "query_identity"),
            mock.patch.object(
                bridge.EvaluationWandbBridge, "poll_once", return_value=0
            ) as poll,
        ):
            self.assertEqual(bridge.main(arguments), 0)
        poll.assert_called_once_with(run)
        document = json.loads(ready.read_text(encoding="utf-8"))
        self.assertEqual(document["schema"], bridge.READY_SCHEMA)
        self.assertEqual(document["process_id"], os.getpid())
        self.assertEqual(document["wandb_run_id"], wandb.init.call_args.kwargs["id"])
        wandb.Settings.assert_called_once_with(x_disable_stats=True)
        self.assertEqual(run.finished, [0])

    def test_ready_marker_must_be_direct_child_of_external_state(self) -> None:
        observer = bridge.EvaluationWandbBridge(
            campaign_root=self.root,
            state_dir=self.state_dir,
            entity="nvidia",
            project="roborender-round-dagger",
        )
        with self.assertRaisesRegex(bridge.BridgeError, "direct child"):
            bridge.ready_marker_path(
                observer.state_dir, self.tmp / "somewhere" / "ready.json"
            )

    def test_main_acquires_state_lock_before_loading_wandb(self) -> None:
        observer = bridge.EvaluationWandbBridge(
            campaign_root=self.root,
            state_dir=self.state_dir,
            entity="nvidia",
            project="roborender-round-dagger",
        )
        arguments = [
            "--campaign-root",
            str(self.root),
            "--state-dir",
            str(self.state_dir),
            "--project",
            "roborender-round-dagger",
            "--entity",
            "nvidia",
            "--expected-email",
            "ldu@nvidia.com",
            "--authorization-scope",
            "team",
            "--driver-status",
            str(self.driver_status),
        ]
        with bridge.exclusive_bridge_lock(observer.state_dir):
            with mock.patch.object(
                bridge,
                "_load_wandb",
                side_effect=AssertionError("W&B must not initialize before the lock"),
            ) as load_wandb:
                self.assertEqual(bridge.main(arguments), 2)
        load_wandb.assert_not_called()

    def test_state_inside_campaign_and_ambiguous_recoveries_are_rejected(self) -> None:
        with self.assertRaisesRegex(bridge.BridgeError, "outside the campaign"):
            bridge.EvaluationWandbBridge(
                campaign_root=self.root,
                state_dir=self.root / "telemetry",
                entity="nvidia",
                project="roborender-round-dagger",
            )
        recovery_parent = self.root / "rounds/macro_round_000000/evaluation_recovery"
        for name in (
            "hardened-0123456789abcdef0123",
            "hardened-abcdef0123456789abcd",
        ):
            recovery = recovery_parent / name
            recovery.mkdir(parents=True)
            (recovery / "result.json").write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(bridge.BridgeError, "multiple complete hardened"):
            bridge.discover_verified_evaluations(self.root)


if __name__ == "__main__":
    unittest.main()
