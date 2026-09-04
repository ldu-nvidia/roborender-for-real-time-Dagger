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
        self.output = self.base / "output"
        self.output.mkdir()

    def _monitor_arguments(self) -> list[str]:
        return [
            "--campaign-root", str(self.base / "campaign"),
            "--output-root", str(self.output),
            "--expected-driver-status", str(self.base / "driver.json"),
            "--learner-backend", "oci_nrt",
        ]

    def test_detaches_and_records_bound_process_identity(self) -> None:
        child = mock.Mock(pid=321)
        child.poll.return_value = None
        with (
            mock.patch.object(starter.subprocess, "Popen", return_value=child) as popen,
            mock.patch.object(starter, "_process_start_ticks", return_value=987),
            mock.patch.object(starter, "_read_ready_marker", return_value={}),
        ):
            report = starter.start_monitor(
                python=self.python,
                pidfile=self.base / "monitor.pid.json",
                logfile=self.base / "monitor.log",
                monitor_arguments=self._monitor_arguments(),
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
        child.poll.return_value = None
        with (
            mock.patch.object(starter.subprocess, "Popen", return_value=child),
            mock.patch.object(starter, "_process_start_ticks", return_value=None),
            mock.patch.object(starter, "_read_ready_marker", return_value={}),
        ):
            with self.assertRaisesRegex(starter.StarterError, "cannot bind"):
                starter.start_monitor(
                    python=self.python,
                    pidfile=self.base / "monitor.pid.json",
                    logfile=self.base / "monitor.log",
                    monitor_arguments=self._monitor_arguments(),
                )
        child.terminate.assert_called_once()
        child.wait.assert_called()

    def test_pidfile_publication_failure_reaps_detached_monitor(self) -> None:
        child = mock.Mock(pid=321)
        child.poll.return_value = None
        with (
            mock.patch.object(starter.subprocess, "Popen", return_value=child),
            mock.patch.object(starter, "_process_start_ticks", return_value=987),
            mock.patch.object(starter, "_read_ready_marker", return_value={}),
            mock.patch.object(starter, "_atomic_pidfile", side_effect=OSError("disk")),
        ):
            with self.assertRaises(OSError):
                starter.start_monitor(
                    python=self.python,
                    pidfile=self.base / "monitor.pid.json",
                    logfile=self.base / "monitor.log",
                    monitor_arguments=self._monitor_arguments(),
                )
        child.terminate.assert_called_once()
        child.wait.assert_called()

    def test_production_caller_requires_and_monitors_both_ready_bridges(self) -> None:
        quality_report = {
            "status": "STARTED",
            "pid": 400,
            "start_ticks": 500,
            "ready": True,
        }
        evaluation_report = {
            "status": "STARTED",
            "pid": 402,
            "start_ticks": 502,
            "ready": True,
        }
        monitor_report = {"status": "STARTED", "pid": 401, "start_ticks": 501}
        with (
            mock.patch.object(
                starter.quality_starter, "start_bridge", return_value=quality_report
            ) as start_bridge,
            mock.patch.object(
                starter.evaluation_starter,
                "start_bridge",
                return_value=evaluation_report,
            ) as start_evaluation,
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
                evaluation_campaign_root=self.base / "campaign",
                evaluation_state_dir=self.base / "evaluation-state",
                evaluation_driver_status=self.base / "driver.json",
                evaluation_project="dagger",
                evaluation_entity="nvidia",
                evaluation_expected_email="ldu@nvidia.com",
                evaluation_poll_seconds=60.0,
                evaluation_pidfile=self.base / "evaluation.pid",
                evaluation_logfile=self.base / "evaluation.log",
                evaluation_ready_timeout_seconds=45.0,
            )
        self.assertTrue(report["quality_bridge_ready"])
        self.assertTrue(report["quality_bridge_monitored"])
        self.assertTrue(report["evaluation_bridge_ready"])
        self.assertTrue(report["evaluation_bridge_monitored"])
        start_bridge.assert_called_once()
        start_evaluation.assert_called_once()
        monitor_arguments = start_monitor.call_args.kwargs["monitor_arguments"]
        required = [
            monitor_arguments[index + 1]
            for index, value in enumerate(monitor_arguments[:-1])
            if value == "--required-process-pidfile"
        ]
        self.assertEqual(
            required,
            [str(self.base / "quality.pid"), str(self.base / "evaluation.pid")],
        )

    def _observer_arguments(self) -> dict[str, object]:
        return {
            "python": self.python,
            "pidfile": self.base / "monitor.pid",
            "logfile": self.base / "monitor.log",
            "monitor_arguments": ["--campaign-root", "/campaign"],
            "quality_campaign_root": self.base / "campaign",
            "quality_state_dir": self.base / "quality-state",
            "quality_driver_status": self.base / "driver.json",
            "quality_project": "dagger",
            "quality_entity": "nvidia",
            "quality_expected_email": "ldu@nvidia.com",
            "quality_poll_seconds": 60.0,
            "quality_pidfile": self.base / "quality.pid",
            "quality_logfile": self.base / "quality.log",
            "quality_ready_timeout_seconds": 45.0,
            "evaluation_campaign_root": self.base / "campaign",
            "evaluation_state_dir": self.base / "evaluation-state",
            "evaluation_driver_status": self.base / "driver.json",
            "evaluation_project": "dagger",
            "evaluation_entity": "nvidia",
            "evaluation_expected_email": "ldu@nvidia.com",
            "evaluation_poll_seconds": 60.0,
            "evaluation_pidfile": self.base / "evaluation.pid",
            "evaluation_logfile": self.base / "evaluation.log",
            "evaluation_ready_timeout_seconds": 45.0,
        }

    def test_evaluation_start_failure_rolls_back_quality_child(self) -> None:
        quality = {"pid": 400, "start_ticks": 500}
        with (
            mock.patch.object(
                starter.quality_starter, "start_bridge", return_value=quality
            ),
            mock.patch.object(
                starter.evaluation_starter,
                "start_bridge",
                side_effect=starter.StarterError("evaluation failed"),
            ),
            mock.patch.object(starter, "_stop_bound_process") as stop,
            self.assertRaisesRegex(starter.StarterError, "evaluation failed"),
        ):
            starter.start_production_observers(**self._observer_arguments())
        stop.assert_called_once_with(quality)

    def test_monitor_start_failure_rolls_back_both_exact_bridge_children(self) -> None:
        quality = {"pid": 400, "start_ticks": 500}
        evaluation = {"pid": 402, "start_ticks": 502}
        with (
            mock.patch.object(
                starter.quality_starter, "start_bridge", return_value=quality
            ),
            mock.patch.object(
                starter.evaluation_starter,
                "start_bridge",
                return_value=evaluation,
            ),
            mock.patch.object(
                starter, "start_monitor", side_effect=starter.StarterError("monitor failed")
            ),
            mock.patch.object(starter, "_stop_bound_process") as stop,
            self.assertRaisesRegex(starter.StarterError, "monitor failed"),
        ):
            starter.start_production_observers(**self._observer_arguments())
        self.assertEqual(stop.call_args_list, [mock.call(evaluation), mock.call(quality)])

    def test_bridge_campaign_or_driver_mismatch_fails_before_launch(self) -> None:
        arguments = self._observer_arguments()
        arguments["evaluation_campaign_root"] = self.base / "other-campaign"
        with (
            mock.patch.object(starter.quality_starter, "start_bridge") as quality,
            self.assertRaisesRegex(starter.StarterError, "campaign roots must match"),
        ):
            starter.start_production_observers(**arguments)
        quality.assert_not_called()

        arguments = self._observer_arguments()
        arguments["evaluation_driver_status"] = self.base / "other-driver.json"
        with (
            mock.patch.object(starter.quality_starter, "start_bridge") as quality,
            self.assertRaisesRegex(starter.StarterError, "status paths must match"),
        ):
            starter.start_production_observers(**arguments)
        quality.assert_not_called()

    def test_rollback_signals_only_the_exact_pid_start_tick_identity(self) -> None:
        report = {"pid": 400, "start_ticks": 500}
        with (
            mock.patch.object(
                starter, "_process_start_ticks", side_effect=[500, None]
            ),
            mock.patch.object(starter.os, "kill") as kill,
        ):
            starter._stop_bound_process(report)
        kill.assert_called_once_with(400, starter.signal.SIGINT)

        with (
            mock.patch.object(starter, "_process_start_ticks", return_value=501),
            mock.patch.object(starter.os, "kill") as kill,
        ):
            starter._stop_bound_process(report)
        kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
