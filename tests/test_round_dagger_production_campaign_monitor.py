from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from round_dagger import artifacts
from round_dagger.production_campaign_monitor import (
    MONITOR_INTERVAL_SECONDS,
    ProductionMonitorError,
    _gpu_health,
    _gpu_health_burst,
    _slurm_health,
    _tree_snapshot,
    collect_report,
    run_monitor,
)


class ProductionCampaignMonitorTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="rr-production-monitor-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.campaign = self.base / "campaign"
        self.campaign.mkdir()
        self.output = self.base / "monitor-output"
        self.contract = self.base / "runtime-contract.json"
        self.contract.write_text('{"campaign_id":"campaign"}\n', encoding="utf-8")
        self.contract.chmod(0o444)
        self.contract_sha = artifacts.sha256_file(self.contract)

    def test_monitor_interval_is_exactly_thirty_minutes(self) -> None:
        self.assertEqual(MONITOR_INTERVAL_SECONDS, 1800.0)

    def test_bounded_test_poll_writes_atomic_latest_and_history(self) -> None:
        counter = 0

        def report(**_kwargs):
            nonlocal counter
            counter += 1
            return {
                "schema": "fixture",
                "observed_unix_ns": counter,
                "file_movement": {"telemetry": [], "wandb": []},
            }

        with mock.patch(
            "round_dagger.production_campaign_monitor.collect_report", side_effect=report
        ):
            result = run_monitor(
                campaign_root=self.campaign,
                campaign_contract=self.contract,
                campaign_contract_sha256=self.contract_sha,
                output_root=self.output,
                actor_patterns=(),
                fleet_patterns=(),
                telemetry_roots=(),
                wandb_roots=(),
                disk_paths=(self.base,),
                oci_nrt_host="oci-nrt",
                slurm_user="ldu",
                slurm_job_prefix="rrd-",
                max_polls=2,
                test_poll_interval_seconds=0,
            )
        self.assertEqual(result["poll_index"], 2)
        latest = json.loads((self.output / "latest.json").read_text())
        self.assertEqual(latest["poll_index"], 2)
        self.assertEqual(len(list((self.output / "snapshots").glob("*.json"))), 2)

    def test_monitor_output_inside_campaign_is_forbidden(self) -> None:
        with self.assertRaisesRegex(ProductionMonitorError, "outside the campaign"):
            run_monitor(
                campaign_root=self.campaign,
                campaign_contract=self.contract,
                campaign_contract_sha256=self.contract_sha,
                output_root=self.campaign / "monitor",
                actor_patterns=(),
                fleet_patterns=(),
                telemetry_roots=(),
                wandb_roots=(),
                disk_paths=(self.base,),
                oci_nrt_host="oci-nrt",
                slurm_user="ldu",
                slurm_job_prefix="rrd-",
                max_polls=1,
                test_poll_interval_seconds=0,
            )

    def test_tree_snapshot_reports_telemetry_movement_inputs(self) -> None:
        telemetry = self.base / "telemetry"
        telemetry.mkdir()
        (telemetry / "scalars.jsonl").write_bytes(b"one\n")
        observed = _tree_snapshot(telemetry)
        self.assertEqual(observed["file_count"], 1)
        self.assertEqual(observed["total_bytes"], 4)
        self.assertEqual(observed["newest_relative_path"], "scalars.jsonl")

    def test_slurm_queries_are_read_only(self) -> None:
        calls: list[list[str]] = []

        def run(argv, **_kwargs):
            calls.append(list(argv))
            if "squeue" in argv[-1]:
                output = "123|RUNNING|00:10|node1|rrd-round-0\n"
            else:
                output = "123|RUNNING|00:10|0:0|rrd-round-0\n"
            return subprocess.CompletedProcess(argv, 0, output, "")

        with mock.patch("subprocess.run", side_effect=run):
            report = _slurm_health("oci-nrt", "ldu", "rrd-")
        self.assertTrue(report["reachable"])
        self.assertEqual(report["active_jobs"][0]["state"], "RUNNING")
        joined = " ".join(" ".join(call) for call in calls)
        for mutating in ("sbatch", "scancel", "scontrol", "srun"):
            self.assertNotIn(mutating, joined)

    def test_gpu_health_includes_foreign_compute_processes_without_argv(self) -> None:
        responses = [
            subprocess.CompletedProcess(
                ["nvidia-smi"],
                0,
                "0, NVIDIA RTX PRO 6000, GPU-aaa, 97887, 3584, 4, 41\n",
                "",
            ),
            subprocess.CompletedProcess(
                ["nvidia-smi"],
                0,
                "4421, GPU-aaa, tesseract_server, 3584\n",
                "",
            ),
        ]
        with (
            mock.patch(
                "round_dagger.production_campaign_monitor.shutil.which",
                return_value="/usr/bin/nvidia-smi",
            ),
            mock.patch("subprocess.run", side_effect=responses),
        ):
            report = _gpu_health()
        self.assertEqual(report["gpus"][0]["memory_used_mib"], 3584)
        self.assertEqual(report["gpus"][0]["utilization_percent"], 4)
        self.assertEqual(report["gpus"][0]["temperature_c"], 41)
        self.assertEqual(report["gpus"][0]["memory_free_mib"], 94303)
        self.assertTrue(report["gpus"][0]["memory_headroom_ok"])
        self.assertTrue(report["gpus"][0]["temperature_ok"])
        self.assertEqual(report["compute_processes"][0]["pid"], 4421)
        self.assertEqual(report["compute_processes"][0]["process_name"], "tesseract_server")
        self.assertNotIn("argv", report["compute_processes"][0])

    def test_process_match_does_not_count_monitor_argv_or_ancestors(self) -> None:
        pattern = "rr_monitor_self_match_probe_9f4c2a"
        code = (
            "import json,sys; "
            "from round_dagger.production_campaign_monitor import _process_matches; "
            "print(json.dumps(_process_matches([sys.argv[1]])))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code, pattern],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        report = json.loads(completed.stdout)
        self.assertEqual(report[pattern]["count"], 0)
        self.assertEqual(report[pattern]["pids"], [])

    def test_gpu_burst_classifies_tesseract_and_thresholds(self) -> None:
        sample = {
            "available": True,
            "gpus": [
                {
                    "index": 0,
                    "name": "RTX PRO 6000",
                    "uuid": "GPU-aaa",
                    "memory_total_mib": 97887,
                    "memory_used_mib": 3584,
                    "memory_free_mib": 94303,
                    "memory_headroom_ok": True,
                    "utilization_percent": 4,
                    "temperature_c": 41,
                    "temperature_ok": True,
                }
            ],
            "compute_processes": [
                {
                    "pid": 4421,
                    "gpu_uuid": "GPU-aaa",
                    "process_name": "tesseract_server",
                    "used_memory_mib": 3584,
                }
            ],
        }
        with (
            mock.patch(
                "round_dagger.production_campaign_monitor._gpu_health",
                return_value=sample,
            ),
            mock.patch("round_dagger.production_campaign_monitor.time.sleep"),
        ):
            report = _gpu_health_burst()
        self.assertEqual(report["sample_count"], 12)
        self.assertEqual(report["window_seconds"], 60.0)
        self.assertEqual(report["severity"], "healthy")
        self.assertEqual(
            report["compute_processes"][0]["classification"],
            "known_foreign_tesseract",
        )

    def test_report_contains_all_required_health_surfaces(self) -> None:
        coordinator = {
            "campaign_id": "campaign",
            "phase": "COLLECTING",
            "next_macro_round": 1,
            "open_macro_round": 1,
            "macro_round_count": 5,
            "completed": False,
            "revision": 9,
            "active_policy_version": 1,
            "active_policy_checkpoint_sha256": "a" * 64,
        }
        with (
            mock.patch(
                "round_dagger.production_campaign_monitor._coordinator",
                return_value=coordinator,
            ),
            mock.patch(
                "round_dagger.production_campaign_monitor._process_matches",
                side_effect=lambda patterns: {
                    pattern: {"count": 1, "pids": [10]} for pattern in patterns
                },
            ),
            mock.patch(
                "round_dagger.production_campaign_monitor._slurm_health",
                return_value={"reachable": True, "active_jobs": [], "recent_jobs": []},
            ),
            mock.patch(
                "round_dagger.production_campaign_monitor._gpu_health_burst",
                return_value={
                    "available": True,
                    "gpus": [],
                    "alerts": [],
                    "compute_processes": [
                        {
                            "pid": 12,
                            "gpu_uuid": "GPU-a",
                            "process_name": "tesseract_server",
                        }
                    ],
                },
            ),
        ):
            report = collect_report(
                campaign_root=self.campaign,
                campaign_contract=self.contract,
                campaign_contract_sha256=self.contract_sha,
                actor_patterns=("actor",),
                fleet_patterns=("fleet",),
                telemetry_roots=(self.base / "telemetry",),
                wandb_roots=(self.base / "wandb",),
                disk_paths=(self.base,),
                oci_nrt_host="oci-nrt",
                slurm_user="ldu",
                slurm_job_prefix="rrd-",
            )
        self.assertTrue(report["read_only"])
        self.assertEqual(report["interval_seconds"], 1800.0)
        self.assertIn("local_process_health", report)
        self.assertIn("oci_nrt_slurm", report)
        self.assertIn("file_movement", report)
        self.assertIn("disk", report)
        self.assertIn("gpu", report)
        self.assertEqual(
            report["gpu"]["compute_processes"][0]["temporal_classification"],
            "baseline_at_monitor_start",
        )

    def test_collecting_warns_for_missing_actor_and_two_stale_polls(self) -> None:
        telemetry = self.base / "telemetry"
        wandb = self.base / "wandb"
        snapshot = {
            "exists": True,
            "file_count": 1,
            "total_bytes": 10,
            "latest_mtime_ns": 100,
            "newest_relative_path": "event.json",
            "truncated": False,
        }
        previous = {
            "campaign": {
                "phase": "COLLECTING",
                "next_macro_round": 0,
                "revision": 1,
            },
            "file_movement": {
                "telemetry": [{"root": str(telemetry), **snapshot}],
                "wandb": [{"root": str(wandb), **snapshot}],
            },
            "progress_health": {
                "telemetry_stale_polls": 1,
                "local_wandb_stale_polls": 1,
            },
            "gpu": {"compute_processes": []},
        }
        coordinator = {
            "campaign_id": "campaign",
            "phase": "COLLECTING",
            "next_macro_round": 0,
            "open_macro_round": 0,
            "macro_round_count": 5,
            "completed": False,
            "revision": 1,
            "active_policy_version": 0,
            "active_policy_checkpoint_sha256": "a" * 64,
        }
        with (
            mock.patch(
                "round_dagger.production_campaign_monitor._coordinator",
                return_value=coordinator,
            ),
            mock.patch(
                "round_dagger.production_campaign_monitor._tree_snapshot",
                side_effect=lambda root: {"root": str(root), **snapshot},
            ),
            mock.patch(
                "round_dagger.production_campaign_monitor._process_matches",
                side_effect=lambda patterns: {
                    pattern: {"count": 0, "pids": []} for pattern in patterns
                },
            ),
            mock.patch(
                "round_dagger.production_campaign_monitor._slurm_health",
                return_value={"reachable": True, "active_jobs": [], "recent_jobs": []},
            ),
            mock.patch(
                "round_dagger.production_campaign_monitor._gpu_health_burst",
                return_value={
                    "available": True,
                    "gpus": [],
                    "alerts": [],
                    "compute_processes": [],
                },
            ),
        ):
            report = collect_report(
                campaign_root=self.campaign,
                campaign_contract=self.contract,
                campaign_contract_sha256=self.contract_sha,
                actor_patterns=("actor",),
                fleet_patterns=("fleet",),
                telemetry_roots=(telemetry,),
                wandb_roots=(wandb,),
                disk_paths=(self.base,),
                oci_nrt_host="oci-nrt",
                slurm_user="ldu",
                slurm_job_prefix="rrd-",
                previous=previous,
            )
        self.assertIn(
            "COLLECTING phase has no configured local actor process visible",
            report["warnings"],
        )
        self.assertEqual(report["progress_health"]["telemetry_stale_polls"], 2)
        self.assertEqual(report["progress_health"]["local_wandb_stale_polls"], 2)
        self.assertTrue(any("telemetry has not moved" in item for item in report["warnings"]))
        self.assertTrue(any("W&B artifacts have not moved" in item for item in report["warnings"]))


if __name__ == "__main__":
    unittest.main()
