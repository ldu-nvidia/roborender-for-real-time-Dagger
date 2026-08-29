from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts.round_dagger import start_student_quality_wandb_bridge as starter


class StartStudentQualityWandbBridgeTest(unittest.TestCase):
    def test_detached_command_always_binds_driver_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            campaign = root / "campaign"
            campaign.mkdir()
            state_dir = root / "observer"
            driver = root / "control" / "driver_status.json"
            driver.parent.mkdir()
            driver.write_text("{}\n", encoding="utf-8")
            pidfile = root / "quality.pid.json"
            logfile = root / "quality.log"
            child = SimpleNamespace(pid=os.getpid())
            ready = state_dir / "STUDENT_QUALITY_WANDB_READY.json"

            def poll():
                ready.write_text(
                    json.dumps(
                        {
                            "schema": starter.READY_SCHEMA,
                            "campaign_id": "campaign",
                            "wandb_run_id": "stable_run",
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
                    logfile=logfile,
                )

            command = popen.call_args.args[0]
            status_index = command.index("--driver-status")
            self.assertEqual(command[status_index + 1], str(driver))
            poll_index = command.index("--poll-seconds")
            self.assertEqual(command[poll_index + 1], "60.0")
            self.assertTrue(report["lifecycle_aware"])
            self.assertTrue(report["ready"])
            self.assertEqual(report["wandb_run_id"], "stable_run")
            self.assertEqual(report["driver_status"], str(driver))
            payload = json.loads(pidfile.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema"], starter.PID_SCHEMA)

    def test_missing_driver_status_and_nonpositive_poll_fail_closed(self) -> None:
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
                "pidfile": root / "quality.pid.json",
                "logfile": root / "quality.log",
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

    def test_startup_waits_for_ready_marker_and_rolls_back_on_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            campaign = root / "campaign"
            campaign.mkdir()
            driver = root / "driver.json"
            driver.write_text("{}\n", encoding="utf-8")
            child = SimpleNamespace(pid=os.getpid())
            child.poll = mock.Mock(return_value=None)
            child.terminate = mock.Mock()
            child.kill = mock.Mock()
            child.wait = mock.Mock(return_value=2)
            with mock.patch.object(starter.subprocess, "Popen", return_value=child):
                with self.assertRaisesRegex(starter.StarterError, "readiness timed out"):
                    starter.start_bridge(
                        python=Path(os.sys.executable),
                        campaign_root=campaign,
                        state_dir=root / "state",
                        driver_status=driver,
                        project="dagger",
                        entity="nvidia",
                        expected_email="ldu@nvidia.com",
                        poll_seconds=60.0,
                        pidfile=root / "bridge.pid",
                        logfile=root / "bridge.log",
                        ready_timeout_seconds=0.01,
                    )
            child.terminate.assert_called_once()

    def test_invalid_ready_marker_rolls_back_detached_child(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            campaign = root / "campaign"
            campaign.mkdir()
            state_dir = root / "state"
            driver = root / "driver.json"
            driver.write_text("{}\n", encoding="utf-8")
            child = mock.Mock(pid=12345)
            ready = state_dir / "STUDENT_QUALITY_WANDB_READY.json"

            def poll():
                ready.write_text('{"schema":"wrong"}\n', encoding="utf-8")
                return None

            child.poll.side_effect = poll
            child.wait.return_value = 0
            with mock.patch.object(starter.subprocess, "Popen", return_value=child):
                with self.assertRaisesRegex(starter.StarterError, "does not match"):
                    starter.start_bridge(
                        python=Path(os.sys.executable),
                        campaign_root=campaign,
                        state_dir=state_dir,
                        driver_status=driver,
                        project="dagger",
                        entity="nvidia",
                        expected_email="ldu@nvidia.com",
                        poll_seconds=60.0,
                        pidfile=root / "bridge.pid",
                        logfile=root / "bridge.log",
                    )
            child.terminate.assert_called_once()
            child.wait.assert_called()

    def test_pidfile_publication_failure_rolls_back_detached_child(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            campaign = root / "campaign"
            campaign.mkdir()
            state_dir = root / "state"
            driver = root / "driver.json"
            driver.write_text("{}\n", encoding="utf-8")
            child = mock.Mock(pid=12345)
            ready = state_dir / "STUDENT_QUALITY_WANDB_READY.json"

            def poll():
                ready.write_text(
                    json.dumps(
                        {
                            "schema": starter.READY_SCHEMA,
                            "wandb_run_id": "stable",
                            "process_id": child.pid,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                return None

            child.poll.side_effect = poll
            child.wait.return_value = 0
            with (
                mock.patch.object(starter.subprocess, "Popen", return_value=child),
                mock.patch.object(starter, "_process_start_ticks", return_value=99),
                mock.patch.object(starter, "_atomic_json", side_effect=OSError("disk")),
            ):
                with self.assertRaises(OSError):
                    starter.start_bridge(
                        python=Path(os.sys.executable),
                        campaign_root=campaign,
                        state_dir=state_dir,
                        driver_status=driver,
                        project="dagger",
                        entity="nvidia",
                        expected_email="ldu@nvidia.com",
                        poll_seconds=60.0,
                        pidfile=root / "bridge.pid",
                        logfile=root / "bridge.log",
                    )
            child.terminate.assert_called_once()
            child.wait.assert_called()


if __name__ == "__main__":
    unittest.main()
