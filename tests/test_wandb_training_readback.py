from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts.scaling import wandb_training_readback as readback


class RemoteRun:
    def scan_history(self, *, keys, page_size):
        del page_size
        return iter([{key: 1 for key in keys}])


class FakeApi:
    def __init__(self):
        self.paths = []

    def run(self, path):
        self.paths.append(path)
        return RemoteRun()


class FakeWandb:
    def __init__(self):
        self.api = FakeApi()

    def Api(self, timeout):
        self.timeout = timeout
        return self.api


class WandbTrainingReadbackTest(unittest.TestCase):
    def test_authorization_scope_is_required_and_team_only(self):
        required = [
            "--run-dir",
            "/tmp/run",
            "--project",
            "project",
            "--entity",
            "nvidia",
            "--expected-email",
            "ldu@nvidia.com",
            "--output",
            "/tmp/report.json",
        ]
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit):
                readback._parser().parse_args(required)
            with self.assertRaises(SystemExit):
                readback._parser().parse_args(
                    required + ["--authorization-scope", "personal"]
                )

        args = readback._parser().parse_args(
            required + ["--authorization-scope", "team"]
        )
        self.assertEqual(args.authorization_scope, "team")

    def test_main_queries_and_emits_canonical_team_identity(self):
        identity = SimpleNamespace(
            email="ldu@nvidia.com",
            entity="nvidia",
            authorization_scope="team",
            team_member=True,
            entity_is_team=True,
            personal=False,
        )
        fake_wandb = FakeWandb()
        query_calls = []

        def query_identity(
            wandb_module,
            *,
            expected_email,
            requested_entity,
            authorization_scope,
        ):
            query_calls.append(
                {
                    "wandb_module": wandb_module,
                    "expected_email": expected_email,
                    "requested_entity": requested_entity,
                    "authorization_scope": authorization_scope,
                }
            )
            return identity

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "validation" / "wandb_training_readback.json"
            argv = [
                "wandb_training_readback.py",
                "--run-dir",
                str(root),
                "--project",
                "training-project",
                "--entity",
                "nvidia",
                "--expected-email",
                "ldu@nvidia.com",
                "--authorization-scope",
                "team",
                "--output",
                str(output),
                "--timeout-s",
                "1",
            ]
            with mock.patch.object(
                readback, "_load_wandb", return_value=fake_wandb
            ), mock.patch.object(
                readback, "query_identity", side_effect=query_identity
            ), mock.patch.object(
                readback, "_trainer_run_id", return_value="trainer-run"
            ), mock.patch.object(
                readback, "_run_id", return_value="sidecar-run"
            ), mock.patch.object(
                readback,
                "_expected_sidecar_keys",
                return_value=(
                    {"trainer_step", "scalar/loss", "media/image", "media/video"},
                    {"scalar": 1, "image": 1, "video": 1},
                ),
            ), mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(
                io.StringIO()
            ):
                code = readback.main()

            self.assertEqual(code, 0)
            self.assertEqual(
                query_calls,
                [
                    {
                        "wandb_module": fake_wandb,
                        "expected_email": "ldu@nvidia.com",
                        "requested_entity": "nvidia",
                        "authorization_scope": "team",
                    }
                ],
            )
            self.assertEqual(
                fake_wandb.api.paths,
                [
                    "nvidia/training-project/trainer-run",
                    "nvidia/training-project/sidecar-run",
                ],
            )
            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(
            report["schema"], "fast_roborender.wandb_training_readback.v2"
        )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["account_email"], "ldu@nvidia.com")
        self.assertEqual(report["entity"], "nvidia")
        self.assertEqual(report["authorization_scope"], "team")
        self.assertTrue(report["team_member"])
        self.assertTrue(report["entity_is_team"])
        self.assertFalse(report["personal"])


if __name__ == "__main__":
    unittest.main()
