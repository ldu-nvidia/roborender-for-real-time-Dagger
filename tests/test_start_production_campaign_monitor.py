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


if __name__ == "__main__":
    unittest.main()
