from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/round_dagger/start_production_campaign_monitor.py"
SPEC = importlib.util.spec_from_file_location("start_production_campaign_monitor", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
starter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(starter)


class StartProductionCampaignMonitorTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="rr-monitor-starter-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.python = self.base / "python"
        self.python.write_text("#!/bin/sh\n", encoding="utf-8")
        self.python.chmod(0o500)

    def test_detaches_and_records_bound_process_identity(self) -> None:
        child = mock.Mock(pid=321)
        child.wait.side_effect = __import__("subprocess").TimeoutExpired("monitor", 0.5)
        with (
            mock.patch.object(starter.subprocess, "Popen", return_value=child) as popen,
            mock.patch.object(starter, "_process_start_ticks", return_value=987),
        ):
            report = starter.start_monitor(
                python=self.python,
                pidfile=self.base / "monitor.pid.json",
                logfile=self.base / "monitor.log",
                monitor_arguments=["--campaign-root", "/campaign"],
            )
        self.assertEqual(report["pid"], 321)
        self.assertTrue(report["read_only"])
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertTrue((self.base / "monitor.pid.json").is_file())

    def test_rejects_bounded_test_mode_for_durable_start(self) -> None:
        with self.assertRaisesRegex(starter.StarterError, "bounded/test"):
            starter.start_monitor(
                python=self.python,
                pidfile=self.base / "monitor.pid.json",
                logfile=self.base / "monitor.log",
                monitor_arguments=["--once"],
            )

    def test_rejects_credential_shaped_argument(self) -> None:
        with self.assertRaisesRegex(starter.StarterError, "credential"):
            starter.start_monitor(
                python=self.python,
                pidfile=self.base / "monitor.pid.json",
                logfile=self.base / "monitor.log",
                monitor_arguments=["--api-key=do-not-store"],
            )

    def test_identity_binding_failure_reaps_detached_monitor(self) -> None:
        child = mock.Mock(pid=321)
        child.wait.side_effect = [
            __import__("subprocess").TimeoutExpired("monitor", 0.5),
            0,
        ]
        child.poll.return_value = None
        with (
            mock.patch.object(starter.subprocess, "Popen", return_value=child),
            mock.patch.object(starter, "_process_start_ticks", return_value=None),
        ):
            with self.assertRaisesRegex(starter.StarterError, "cannot bind"):
                starter.start_monitor(
                    python=self.python,
                    pidfile=self.base / "monitor.pid.json",
                    logfile=self.base / "monitor.log",
                    monitor_arguments=["--campaign-root", "/campaign"],
                )
        child.terminate.assert_called_once()
        child.wait.assert_called()

    def test_pidfile_publication_failure_reaps_detached_monitor(self) -> None:
        child = mock.Mock(pid=321)
        child.wait.side_effect = [
            __import__("subprocess").TimeoutExpired("monitor", 0.5),
            0,
        ]
        child.poll.return_value = None
        with (
            mock.patch.object(starter.subprocess, "Popen", return_value=child),
            mock.patch.object(starter, "_process_start_ticks", return_value=987),
            mock.patch.object(starter, "_atomic_pidfile", side_effect=OSError("disk")),
        ):
            with self.assertRaises(OSError):
                starter.start_monitor(
                    python=self.python,
                    pidfile=self.base / "monitor.pid.json",
                    logfile=self.base / "monitor.log",
                    monitor_arguments=["--campaign-root", "/campaign"],
                )
        child.terminate.assert_called_once()
        child.wait.assert_called()

    def test_production_caller_requires_ready_quality_bridge_and_monitors_it(self) -> None:
        quality_report = {
            "status": "STARTED",
            "pid": 400,
            "start_ticks": 500,
            "ready": True,
        }
        monitor_report = {"status": "STARTED", "pid": 401, "start_ticks": 501}
        with (
            mock.patch.object(
                starter.quality_starter, "start_bridge", return_value=quality_report
            ) as start_bridge,
            mock.patch.object(
                starter, "start_monitor", return_value=monitor_report
            ) as start_monitor,
        ):
            report = starter.start_production_observers(
                python=self.python,
                pidfile=self.base / "monitor.pid",
                logfile=self.base / "monitor.log",
                monitor_arguments=["--campaign-root", "/campaign"],
                quality_campaign_root=self.base / "campaign",
                quality_state_dir=self.base / "quality-state",
                quality_driver_status=self.base / "driver.json",
                quality_project="dagger",
                quality_entity="nvidia",
                quality_expected_email="ldu@nvidia.com",
                quality_poll_seconds=60.0,
                quality_pidfile=self.base / "quality.pid",
                quality_logfile=self.base / "quality.log",
                quality_ready_timeout_seconds=45.0,
            )
        self.assertTrue(report["quality_bridge_ready"])
        self.assertTrue(report["quality_bridge_monitored"])
        start_bridge.assert_called_once()
        monitor_arguments = start_monitor.call_args.kwargs["monitor_arguments"]
        self.assertIn("--required-process-pidfile", monitor_arguments)
        bridge_index = monitor_arguments.index("--required-process-pidfile")
        self.assertEqual(
            monitor_arguments[bridge_index + 1], str(self.base / "quality.pid")
        )


if __name__ == "__main__":
    unittest.main()
