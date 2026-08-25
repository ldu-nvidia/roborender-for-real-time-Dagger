from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import time
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

from scripts.server import local_image3f_fleet as fleet_module
from scripts.server.renderer_identity import (
    canonical_json_sha256,
    regular_file_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
FLEET = ROOT / "scripts/server/local_image3f_fleet.py"
FAKE_SERVER = ROOT / "tests/fixtures/fake_image3f_fleet_server.py"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class LocalImage3FFleetTests(unittest.TestCase):
    def test_virtualenv_python_symlink_is_validated_but_not_resolved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "base-python"
            target.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            target.chmod(0o755)
            venv_python = root / "venv/bin/python"
            venv_python.parent.mkdir(parents=True)
            venv_python.symlink_to(target)
            selected = fleet_module._require_executable(
                venv_python, label="Image3F Python"
            )
            self.assertEqual(selected, venv_python.absolute())
            self.assertNotEqual(selected, target.resolve())

    def test_incompatible_service_is_cleanly_stopped_but_run_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            for name in ("release", "models"):
                (base / name).mkdir()
            fleet = base / "fleet"
            result = subprocess.run(
                [
                    sys.executable,
                    str(FLEET),
                    "run",
                    "--fleet-dir",
                    str(fleet),
                    "--gpus",
                    "3",
                    "--image3f-root",
                    str(base / "release"),
                    "--model-root",
                    str(base / "models"),
                    "--python",
                    sys.executable,
                    "--server-script",
                    str(FAKE_SERVER),
                    "--run-id",
                    "bad-fleet",
                    "--expected-release-id",
                    "intentionally-wrong",
                    "--startup-timeout-s",
                    "0.25",
                    "--shutdown-timeout-s",
                    "5",
                    "--connect-timeout-s",
                    "0.1",
                    "--poll-s",
                    "0.05",
                    "--stop-poll-s",
                    "0.05",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            self.assertEqual(result.returncode, 2, result.stdout)
            self.assertIn("never passed the readiness gate", result.stderr)
            self.assertEqual(read_json(fleet / "supervisor.json")["status"], "failed")
            clean = read_json(fleet / "clean_stop.json")
            self.assertFalse(clean["readiness_succeeded"])
            self.assertEqual(clean["exit_codes"], [0])
            self.assertTrue((fleet / "stop_acks/renderer_000.json").is_file())
            self.assertFalse((fleet / "endpoints/renderer_000.txt").exists())

    def test_readiness_rejects_mixed_runtime_identity_and_wrong_server_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            endpoints = Path(temporary)
            paths = []
            for index in range(2):
                path = endpoints / f"renderer_{index:03d}.txt"
                path.write_text(f"127.0.0.1 {19000 + index}\n", encoding="utf-8")
                paths.append(path)
            with mock.patch.dict(
                "os.environ",
                {
                    "IMAGE3F_SERVER_ID": "fleet:000",
                    "IMAGE3F_MAX_REFERENCE_LAG": "512",
                },
                clear=False,
            ):
                from tests.fixtures.fake_image3f_fleet_server import ping_response

                first = ping_response()
            second = deepcopy(first)
            second["server_id"] = "wrong:001"
            second["runtime_identity"]["release_id"] = "mixed-release"
            rows = [
                {
                    "file": str(path.resolve()),
                    "host": "127.0.0.1",
                    "port": 19000 + index,
                    "rtt_ms": 0.1,
                    "response": response,
                }
                for index, (path, response) in enumerate(zip(paths, (first, second)))
            ]
            with mock.patch.object(fleet_module, "probe", return_value=(rows, [])):
                report, ready = fleet_module._build_probe_report(
                    endpoints,
                    expected=2,
                    timeout_s=0.1,
                    required_reference_lag=256,
                    expected_release_id=None,
                    expected_adapter_sha256=None,
                    expected_engine_sha256=None,
                    expected_inference_steps=5,
                    expected_teacache_threshold=0.3,
                    expected_cfg_scale=1.0,
                    expected_cuda_graph=False,
                    expected_server_ids={"fleet:000", "fleet:001"},
                )
            self.assertFalse(ready)
            self.assertFalse(report["ready"])
            self.assertIsNone(report["runtime_identity_sha256"])
            self.assertEqual(report["unique_endpoints"], 2)

    def test_new_readiness_requires_content_bound_identity_but_legacy_status_is_readable(self):
        with tempfile.TemporaryDirectory() as temporary:
            endpoints = Path(temporary)
            endpoint = endpoints / "renderer_000.txt"
            endpoint.write_text("127.0.0.1 19000\n", encoding="utf-8")
            script_sha = regular_file_sha256(FAKE_SERVER)
            environment = {
                "IMAGE3F_SERVER_ID": "fleet:000",
                "IMAGE3F_MAX_REFERENCE_LAG": "512",
                "IMAGE3F_EXPECT_SERVER_SCRIPT_SHA256": script_sha,
            }
            with mock.patch.dict("os.environ", environment, clear=False):
                from tests.fixtures.fake_image3f_fleet_server import ping_response

                response = ping_response()
            rows = [{
                "file": str(endpoint.resolve()),
                "host": "127.0.0.1",
                "port": 19000,
                "rtt_ms": 0.1,
                "response": response,
            }]
            common = dict(
                expected=1,
                timeout_s=0.1,
                required_reference_lag=256,
                expected_release_id=None,
                expected_adapter_sha256=None,
                expected_engine_sha256=None,
                expected_inference_steps=5,
                expected_teacache_threshold=0.3,
                expected_cfg_scale=1.0,
                expected_cuda_graph=False,
                expected_server_ids={"fleet:000"},
            )
            with mock.patch.object(fleet_module, "probe", return_value=(rows, [])):
                strict, strict_ready = fleet_module._build_probe_report(
                    endpoints,
                    expected_server_script_sha256=script_sha,
                    **common,
                )
            self.assertTrue(strict_ready)
            self.assertEqual(strict["renderer_identity_mode"], "content_bound_v1")
            self.assertEqual(
                strict["renderer_identity_sha256"],
                canonical_json_sha256(response["renderer_identity"]),
            )

            legacy_response = deepcopy(response)
            del legacy_response["renderer_identity"]
            del legacy_response["renderer_identity_sha256"]
            rows[0]["response"] = legacy_response
            with mock.patch.object(
                fleet_module, "probe", side_effect=lambda *_args: (rows, [])
            ):
                rejected, rejected_ready = fleet_module._build_probe_report(
                    endpoints,
                    expected_server_script_sha256=script_sha,
                    **common,
                )
                legacy, legacy_ready = fleet_module._build_probe_report(
                    endpoints,
                    expected_server_script_sha256=None,
                    **common,
                )
            self.assertFalse(rejected_ready)
            self.assertTrue(any("renderer identity" in item for item in rejected["errors"]))
            self.assertTrue(legacy_ready)
            self.assertEqual(legacy["renderer_identity_mode"], "legacy_runtime_only")
            self.assertIsNone(legacy["renderer_identity_sha256"])

    def test_two_gpu_lifecycle_has_dynamic_unique_endpoints_and_clean_stop(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            image3f_root = base / "release"
            model_root = base / "models"
            image3f_root.mkdir()
            model_root.mkdir()
            fleet_dir = base / "fleet"
            command = [
                sys.executable,
                str(FLEET),
                "run",
                "--fleet-dir",
                str(fleet_dir),
                "--gpus",
                "7,9",
                "--image3f-root",
                str(image3f_root),
                "--model-root",
                str(model_root),
                "--python",
                sys.executable,
                "--server-script",
                str(FAKE_SERVER),
                "--run-id",
                "unit-fleet",
                "--startup-timeout-s",
                "10",
                "--shutdown-timeout-s",
                "10",
                "--connect-timeout-s",
                "0.5",
                "--poll-s",
                "0.05",
                "--stop-poll-s",
                "0.05",
                "--expected-release-id",
                "fake-qualified-release",
                "--expected-adapter-sha256",
                "a" * 64,
                "--expected-engine-sha256",
                "b" * 64,
            ]
            supervisor = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    if (fleet_dir / "readiness.json").is_file():
                        break
                    if supervisor.poll() is not None:
                        stdout, stderr = supervisor.communicate()
                        self.fail(f"supervisor exited early: {stdout}\n{stderr}")
                    time.sleep(0.05)
                else:
                    self.fail("fleet did not become ready")

                readiness = read_json(fleet_dir / "readiness.json")
                plan = read_json(fleet_dir / "plan.json")
                self.assertTrue(readiness["ready"])
                self.assertEqual(readiness["capable"], 2)
                self.assertEqual(readiness["unique_endpoints"], 2)
                self.assertEqual(
                    readiness["server_ids"], ["unit-fleet:000", "unit-fleet:001"]
                )
                self.assertEqual(
                    plan["server_script_sha256"], regular_file_sha256(FAKE_SERVER)
                )
                self.assertEqual(readiness["renderer_identity_mode"], "content_bound_v1")
                self.assertEqual(
                    readiness["server_script_sha256"], plan["server_script_sha256"]
                )
                self.assertRegex(
                    readiness["renderer_identity_sha256"], r"^[0-9a-f]{64}$"
                )
                self.assertNotEqual(
                    readiness["renderer_identity_sha256"],
                    readiness["runtime_identity_sha256"],
                )
                for row in readiness["endpoints"]:
                    self.assertEqual(
                        canonical_json_sha256(row["response"]["renderer_identity"]),
                        readiness["renderer_identity_sha256"],
                    )
                endpoints = sorted((fleet_dir / "endpoints").glob("renderer_*.txt"))
                self.assertEqual(len(endpoints), 2)
                ports = [int(path.read_text().split()[1]) for path in endpoints]
                self.assertEqual(len(set(ports)), 2)
                self.assertTrue(all(port > 0 for port in ports))

                status = subprocess.run(
                    [
                        sys.executable,
                        str(FLEET),
                        "status",
                        "--fleet-dir",
                        str(fleet_dir),
                        "--connect-timeout-s",
                        "0.5",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(status.returncode, 0, status.stderr)
                self.assertTrue(json.loads(status.stdout)["live_readiness"]["ready"])

                stopped = subprocess.run(
                    [
                        sys.executable,
                        str(FLEET),
                        "stop",
                        "--fleet-dir",
                        str(fleet_dir),
                        "--wait-timeout-s",
                        "10",
                        "--poll-s",
                        "0.05",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(stopped.returncode, 0, stopped.stderr)
                stdout, stderr = supervisor.communicate(timeout=10)
                self.assertEqual(supervisor.returncode, 0, f"{stdout}\n{stderr}")
            finally:
                if supervisor.poll() is None:
                    stop = fleet_dir / "STOP"
                    if fleet_dir.is_dir() and not stop.exists():
                        stop.write_text("{}\n", encoding="utf-8")
                    supervisor.wait(timeout=10)

            self.assertEqual(read_json(fleet_dir / "supervisor.json")["status"], "stopped")
            clean = read_json(fleet_dir / "clean_stop.json")
            self.assertEqual(clean["exit_codes"], [0, 0])
            self.assertEqual(clean["renderer_count"], 2)
            stop_sha = hashlib.sha256((fleet_dir / "STOP").read_bytes()).hexdigest()
            for index, gpu in enumerate(("7", "9")):
                ack = read_json(
                    fleet_dir / "stop_acks" / f"renderer_{index:03d}.json"
                )
                self.assertEqual(ack["cuda_visible_devices"], gpu)
                self.assertEqual(ack["stop_sentinel_sha256"], stop_sha)
                self.assertFalse(
                    (fleet_dir / "endpoints" / f"renderer_{index:03d}.txt").exists()
                )
                self.assertFalse(
                    (fleet_dir / "queues" / f"renderer_{index:03d}" / "READY").exists()
                )

    def test_duplicate_gpu_is_rejected_without_creating_fleet(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            for name in ("release", "models"):
                (base / name).mkdir()
            fleet = base / "must-not-exist"
            result = subprocess.run(
                [
                    sys.executable,
                    str(FLEET),
                    "run",
                    "--fleet-dir",
                    str(fleet),
                    "--gpus",
                    "0,0",
                    "--image3f-root",
                    str(base / "release"),
                    "--model-root",
                    str(base / "models"),
                    "--python",
                    sys.executable,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("must not contain duplicates", result.stderr)
            self.assertFalse(fleet.exists())

    def test_supervisor_source_has_no_forced_process_shutdown(self):
        source = FLEET.read_text(encoding="utf-8")
        self.assertNotIn(".terminate(", source)
        self.assertNotIn(".kill(", source)
        self.assertIn('"IMAGE3F_PORT": "0"', source)
        self.assertIn("_validate_acknowledgements", source)


if __name__ == "__main__":
    unittest.main()
