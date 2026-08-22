import contextlib
import io
import json
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from scripts.scaling import wandb_preflight


EXPECTED = {
    "expected_email": "ldu@nvidia.com",
    "requested_entity": "nvidia",
    "authorization_scope": "team",
}


def valid_viewer(**overrides):
    values = {
        "email": "ldu@nvidia.com",
        "username": "ldu",
        "entity": "nvidia",
        "teams": ["nvidia"],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class FakeApi:
    def __init__(self, viewer):
        self.viewer = viewer


class FakeWandb:
    def __init__(self, viewer=None, api_error=None):
        self.viewer = viewer
        self.api_error = api_error
        self.timeouts = []
        self.apis = []

    def Api(self, timeout):
        self.timeouts.append(timeout)
        if self.api_error is not None:
            raise self.api_error
        api = FakeApi(self.viewer)
        self.apis.append(api)
        return api


class WandbPreflightTest(unittest.TestCase):
    def test_accepts_only_verified_nvidia_team_membership(self):
        fake = FakeWandb(
            valid_viewer(email="LDU@NVIDIA.COM"),
        )
        identity = wandb_preflight.query_identity(
            fake,
            **EXPECTED,
            timeout_s=7.5,
            environment={"WANDB_MODE": "online"},
        )

        self.assertEqual(identity.email, "ldu@nvidia.com")
        self.assertEqual(identity.username, "ldu")
        self.assertEqual(identity.entity, "nvidia")
        self.assertEqual(identity.authorization_scope, "team")
        self.assertTrue(identity.team_member)
        self.assertTrue(identity.entity_is_team)
        self.assertFalse(identity.personal)
        self.assertEqual(identity.mode, "online")
        self.assertEqual(fake.timeouts, [7.5])
        self.assertEqual(len(fake.apis), 1)

    def test_accepts_raw_graphql_viewer_team_membership(self):
        viewer = {
            "email": "ldu@nvidia.com",
            "username": "ldu",
            "teams": {"edges": [{"node": {"name": "nvidia"}}]},
        }
        identity = wandb_preflight.validate_identity(
            viewer,
            **EXPECTED,
        )
        self.assertTrue(identity.team_member)
        self.assertTrue(identity.entity_is_team)

    def test_username_equality_is_not_team_membership(self):
        with self.assertRaisesRegex(
            wandb_preflight.WandbPreflightError, "team memberships"
        ):
            wandb_preflight.validate_identity(
                valid_viewer(username="nvidia", teams=[]),
                **EXPECTED,
            )

    def test_default_entity_is_not_team_membership(self):
        with self.assertRaisesRegex(
            wandb_preflight.WandbPreflightError, "team memberships"
        ):
            wandb_preflight.validate_identity(
                valid_viewer(entity="nvidia", teams=["another-team"]),
                **EXPECTED,
            )

    def test_rejects_every_request_outside_pinned_authorization(self):
        invalid = (
            {**EXPECTED, "expected_email": "other@example.com"},
            {**EXPECTED, "requested_entity": "ldu"},
            {**EXPECTED, "requested_entity": "hwinf_dcm"},
            {**EXPECTED, "authorization_scope": "personal"},
        )
        for request in invalid:
            with self.subTest(request=request):
                with self.assertRaises(wandb_preflight.WandbPreflightError):
                    wandb_preflight.validate_identity(
                        valid_viewer(), **request
                    )

    def test_rejects_wrong_authenticated_account_without_echoing_addresses(self):
        viewer = valid_viewer(email="someone@example.com")
        with self.assertRaises(wandb_preflight.WandbPreflightError) as caught:
            wandb_preflight.validate_identity(viewer, **EXPECTED)
        message = str(caught.exception)
        self.assertNotIn("someone@example.com", message)
        self.assertNotIn("ldu@nvidia.com", message)

    def test_rejects_missing_or_malformed_team_memberships(self):
        for teams in (None, "nvidia", {}, {"edges": "nvidia"}, [object()]):
            with self.subTest(teams=teams):
                with self.assertRaises(wandb_preflight.WandbPreflightError):
                    wandb_preflight.validate_identity(
                        valid_viewer(teams=teams), **EXPECTED
                    )

    def test_rejects_non_online_modes(self):
        for mode in ("offline", "dryrun", "disabled", "shared"):
            with self.subTest(mode=mode):
                with self.assertRaises(wandb_preflight.WandbPreflightError):
                    wandb_preflight.validate_mode({"WANDB_MODE": mode})

    def test_auth_or_team_failure_does_not_expose_exception_text_or_api_key(self):
        secret = "not-a-real-secret-key"
        failures = (FakeWandb(api_error=RuntimeError(f"request failed with key {secret}")),)
        for fake in failures:
            with self.subTest(fake=fake):
                with self.assertRaises(wandb_preflight.WandbPreflightError) as caught:
                    wandb_preflight.query_identity(
                        fake,
                        **EXPECTED,
                        environment={
                            "WANDB_MODE": "online",
                            "WANDB_API_KEY": secret,
                        },
                    )
                self.assertNotIn(secret, str(caught.exception))
                self.assertIn("RuntimeError", str(caught.exception))

    def test_main_json_reports_team_attestation_and_never_api_key(self):
        secret = "not-a-real-secret-key"
        fake = FakeWandb(valid_viewer())
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(
            wandb_preflight, "_load_wandb", return_value=fake
        ), mock.patch.dict(
            os.environ,
            {"WANDB_API_KEY": secret, "WANDB_MODE": "online"},
            clear=False,
        ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = wandb_preflight.main(
                [
                    "--expected-email",
                    "ldu@nvidia.com",
                    "--entity",
                    "nvidia",
                    "--authorization-scope",
                    "team",
                    "--format",
                    "json",
                ]
            )
        self.assertEqual(code, 0)
        combined = stdout.getvalue() + stderr.getvalue()
        self.assertNotIn(secret, combined)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["entity"], "nvidia")
        self.assertEqual(report["authorization_scope"], "team")
        self.assertTrue(report["team_member"])
        self.assertTrue(report["entity_is_team"])
        self.assertFalse(report["personal"])


if __name__ == "__main__":
    unittest.main()
