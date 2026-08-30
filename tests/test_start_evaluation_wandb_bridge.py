from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts.round_dagger import start_evaluation_wandb_bridge as starter


class StartEvaluationWandbBridgeTest(unittest.TestCase):
    def test_detached_command_is_lifecycle_bound_ready_and_pid_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            campaign = root / "campaign"
            campaign.mkdir()
            state_dir = root / "observer"
            driver = root / "driver_status.json"
            driver.write_text("{}\n", encoding="utf-8")
            pidfile = root / "evaluation.pid.json"
            child = SimpleNamespace(pid=os.getpid())
            ready = state_dir / "EVALUATION_WANDB_READY.json"

            def poll():
                ready.write_text(
                    json.dumps(
                        {
                            "schema": starter.READY_SCHEMA,
                            "campaign_id": "campaign",
                            "wandb_run_id": "stable_eval_run",
                            "process_id": child.pid,
                            "ready_unix_ns": 1,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return None

            child.poll = mock.Mock(side_effect=poll)
            with mock.patch.object(starter.subprocess, "Popen", return_value=child) as popen:
                report = starter.start_bridge(
                    python=Path(os.sys.executable),
                    campaign_root=campaign,
                    state_dir=state_dir,
                    driver_status=driver,
                    project="roborender-round-dagger",
                    entity="nvidia",
                    expected_email="ldu@nvidia.com",
                    poll_seconds=60.0,
                    pidfile=pidfile,
                    logfile=root / "evaluation.log",
                )

            command = popen.call_args.args[0]
            self.assertEqual(command[command.index("--driver-status") + 1], str(driver))
            self.assertEqual(command[command.index("--ready-marker") + 1], str(ready))
            self.assertTrue(report["ready"])
            self.assertTrue(report["lifecycle_aware"])
            self.assertEqual(report["wandb_run_id"], "stable_eval_run")
            payload = json.loads(pidfile.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], starter.PID_SCHEMA)
            self.assertEqual(payload["start_ticks"], report["start_ticks"])

    def test_missing_driver_and_nonpositive_poll_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            campaign = root / "campaign"
            campaign.mkdir()
            common = {
                "python": Path(os.sys.executable),
                "campaign_root": campaign,
                "state_dir": root / "observer",
                "project": "roborender-round-dagger",
                "entity": "nvidia",
                "expected_email": "ldu@nvidia.com",
                "pidfile": root / "evaluation.pid.json",
                "logfile": root / "evaluation.log",
            }
            with self.assertRaisesRegex(starter.StarterError, "driver status is missing"):
                starter.start_bridge(
                    **common,
                    driver_status=root / "missing.json",
                    poll_seconds=60.0,
                )
            driver = root / "driver_status.json"
            driver.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(starter.StarterError, "must be positive"):
                starter.start_bridge(
                    **common,
                    driver_status=driver,
                    poll_seconds=0.0,
                )

    def test_readiness_timeout_reaps_exact_detached_child(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            campaign = root / "campaign"
            campaign.mkdir()
            driver = root / "driver_status.json"
            driver.write_text("{}\n", encoding="utf-8")
            child = SimpleNamespace(pid=12345)
            child.poll = mock.Mock(return_value=None)
            child.terminate = mock.Mock()
            child.kill = mock.Mock()
            child.wait = mock.Mock(return_value=2)
            with mock.patch.object(starter.subprocess, "Popen", return_value=child):
                with self.assertRaisesRegex(starter.StarterError, "readiness timed out"):
                    starter.start_bridge(
                        python=Path(os.sys.executable),
                        campaign_root=campaign,
                        state_dir=root / "observer",
                        driver_status=driver,
                        project="roborender-round-dagger",
                        entity="nvidia",
                        expected_email="ldu@nvidia.com",
                        poll_seconds=60.0,
                        pidfile=root / "evaluation.pid.json",
                        logfile=root / "evaluation.log",
                        ready_timeout_seconds=0.01,
                    )
            child.terminate.assert_called_once()
            child.wait.assert_called()


if __name__ == "__main__":
    unittest.main()
