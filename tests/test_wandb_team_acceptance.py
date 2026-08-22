from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.scaling import promote_wandb_acceptance
from scripts.scaling import wandb_online_smoke


ROOT = Path(__file__).resolve().parents[1]
AUTHORIZATION = {
    "expected_email": "ldu@nvidia.com",
    "expected_entity": "nvidia",
    "authorization_scope": "team",
}


def passing_report(**overrides):
    report = {
        "schema": "fast_roborender.wandb_online_smoke.v2",
        "status": "PASS",
        "account_email": "ldu@nvidia.com",
        "entity": "nvidia",
        "authorization_scope": "team",
        "team_member": True,
        "entity_is_team": True,
        "personal": False,
        "project": "validation-project",
        "run_id": "validation-run",
        "observed_keys": [
            "validation/scalar",
            "validation/image",
            "validation/video",
        ],
    }
    report.update(overrides)
    return report


class FakeRun:
    def __init__(self):
        self.logged = []
        self.finished = False

    def log(self, payload):
        self.logged.append(payload)

    def finish(self):
        self.finished = True


class IdentityApi:
    viewer = {
        "email": "ldu@nvidia.com",
        "username": "ldu",
        "teams": {"edges": [{"node": {"name": "nvidia"}}]},
    }

    def team(self, name):
        return {"name": name, "isTeam": True}


class RemoteRun:
    def scan_history(self, **_kwargs):
        return iter(
            [
                {
                    "validation/scalar": 1.0,
                    "validation/image": {"_type": "image-file"},
                    "validation/video": {"_type": "video-file"},
                }
            ]
        )


class ReadbackApi:
    def __init__(self, owner):
        self.owner = owner

    def run(self, path):
        self.owner.read_paths.append(path)
        return RemoteRun()


class FakeWandb:
    def __init__(self):
        self.api_calls = 0
        self.init_kwargs = []
        self.read_paths = []
        self.run = FakeRun()

    def Api(self, timeout):
        self.api_calls += 1
        if self.api_calls == 1:
            return IdentityApi()
        return ReadbackApi(self)

    def init(self, **kwargs):
        self.init_kwargs.append(kwargs)
        return self.run

    @staticmethod
    def Image(value, **_kwargs):
        return value

    @staticmethod
    def Video(value, **_kwargs):
        return value


class WandbTeamAcceptanceTest(unittest.TestCase):
    def test_online_smoke_report_binds_team_authorization_and_readback(self):
        fake = FakeWandb()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "report.json"
            argv = [
                "wandb_online_smoke.py",
                "--expected-email",
                "ldu@nvidia.com",
                "--entity",
                "nvidia",
                "--authorization-scope",
                "team",
                "--project",
                "validation-project",
                "--run-id",
                "validation-run",
                "--output",
                str(output),
            ]
            stdout = io.StringIO()
            with mock.patch.object(
                wandb_online_smoke, "_load_wandb", return_value=fake
            ), mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(stdout):
                code = wandb_online_smoke.main()

            self.assertEqual(code, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["schema"], "fast_roborender.wandb_online_smoke.v2")
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["account_email"], "ldu@nvidia.com")
            self.assertEqual(report["entity"], "nvidia")
            self.assertEqual(report["authorization_scope"], "team")
            self.assertTrue(report["team_member"])
            self.assertTrue(report["entity_is_team"])
            self.assertFalse(report["personal"])
            self.assertEqual(
                report["observed_keys"],
                ["validation/image", "validation/scalar", "validation/video"],
            )
            self.assertEqual(fake.init_kwargs[0]["entity"], "nvidia")
            self.assertEqual(
                fake.read_paths, ["nvidia/validation-project/validation-run"]
            )
            self.assertTrue(fake.run.finished)

    def test_promotion_is_create_only_and_accepts_exact_v2_team_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "job.json"
            destination = root / "wandb_online_smoke_nvidia_acceptance.json"
            source.write_text(
                json.dumps(passing_report(), sort_keys=True) + "\n", encoding="utf-8"
            )

            digest = promote_wandb_acceptance.promote(
                source, destination, **AUTHORIZATION
            )
            self.assertEqual(digest, hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            self.assertEqual(destination.stat().st_mode & 0o777, 0o444)
            self.assertEqual(
                promote_wandb_acceptance.promote(
                    source, destination, **AUTHORIZATION
                ),
                digest,
            )

            different = root / "different.json"
            different.write_text(
                json.dumps(passing_report(run_id="different")) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                promote_wandb_acceptance.PromotionError,
                "refusing to replace different",
            ):
                promote_wandb_acceptance.promote(
                    different, destination, **AUTHORIZATION
                )

    def test_promotion_rejects_personal_or_unattested_team_reports(self):
        invalid = (
            {"schema": "fast_roborender.wandb_online_smoke.v1"},
            {"entity": "ldu"},
            {"authorization_scope": "personal"},
            {"team_member": False},
            {"entity_is_team": False},
            {"personal": True},
            {"observed_keys": ["validation/scalar"]},
            {
                "observed_keys": [
                    "validation/scalar",
                    "validation/image",
                    "validation/video",
                    "validation/video",
                ]
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, change in enumerate(invalid):
                with self.subTest(change=change):
                    source = root / f"invalid-{index}.json"
                    source.write_text(
                        json.dumps(passing_report(**change)) + "\n",
                        encoding="utf-8",
                    )
                    with self.assertRaises(promote_wandb_acceptance.PromotionError):
                        promote_wandb_acceptance.promote(
                            source, root / f"acceptance-{index}.json", **AUTHORIZATION
                        )

    def test_promotion_rejects_duplicate_json_keys_and_unapproved_callers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "duplicate.json"
            source.write_text(
                '{"schema":"fast_roborender.wandb_online_smoke.v2",'
                '"schema":"fast_roborender.wandb_online_smoke.v2"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                promote_wandb_acceptance.PromotionError, "duplicate JSON key"
            ):
                promote_wandb_acceptance.promote(
                    source, root / "acceptance.json", **AUTHORIZATION
                )

            valid = root / "valid.json"
            valid.write_text(json.dumps(passing_report()), encoding="utf-8")
            invalid_requests = (
                {**AUTHORIZATION, "expected_email": "other@example.com"},
                {**AUTHORIZATION, "expected_entity": "hwinf_dcm"},
                {**AUTHORIZATION, "authorization_scope": "personal"},
            )
            for request in invalid_requests:
                with self.subTest(request=request):
                    with self.assertRaises(promote_wandb_acceptance.PromotionError):
                        promote_wandb_acceptance.promote(
                            valid, root / "other.json", **request
                        )

    def test_validator_passes_explicit_team_scope_to_both_gates(self):
        validator = (
            ROOT / "scripts/scaling/hsg_gb200_wandb_validate.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("EXPECTED_EMAIL=${WANDB_EXPECTED_EMAIL:-ldu@nvidia.com}", validator)
        self.assertIn("EXPECTED_ENTITY=${WANDB_ENTITY:-nvidia}", validator)
        self.assertIn("AUTHORIZATION_SCOPE=${WANDB_AUTHORIZATION_SCOPE:-team}", validator)
        self.assertIn("wandb_online_smoke_nvidia_acceptance.json", validator)
        self.assertEqual(validator.count('--authorization-scope "$AUTHORIZATION_SCOPE"'), 2)
        self.assertIn('--entity "$EXPECTED_ENTITY"', validator)
        self.assertIn('--expected-entity "$EXPECTED_ENTITY"', validator)

    def test_packed_job_passes_team_scope_to_preflight_watcher_and_readback(self):
        packed = (
            ROOT / "scripts/scaling/hsg_gb200_packed_smoke.sbatch"
        ).read_text(encoding="utf-8")
        self.assertEqual(packed.count("--authorization-scope team"), 3)
        watcher_call = packed.index('"$ROOT/scripts/scaling/wandb_run_watcher.py"')
        watcher_scope = packed.index("--authorization-scope team", watcher_call)
        readback_call = packed.index('"$ROOT/scripts/scaling/wandb_training_readback.py"')
        readback_scope = packed.index("--authorization-scope team", readback_call)
        self.assertLess(watcher_scope, readback_call)
        self.assertGreater(readback_scope, readback_call)


if __name__ == "__main__":
    unittest.main()
