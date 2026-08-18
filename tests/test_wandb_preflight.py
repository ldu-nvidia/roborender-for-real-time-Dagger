import contextlib
import io
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from scripts.scaling import wandb_preflight


class FakeApi:
    def __init__(self, viewer):
        self.viewer = viewer


class FakeWandb:
    def __init__(self, viewer=None, error=None):
        self.viewer = viewer
        self.error = error
        self.timeouts = []

    def Api(self, timeout):
        self.timeouts.append(timeout)
        if self.error is not None:
            raise self.error
        return FakeApi(self.viewer)


class WandbPreflightTest(unittest.TestCase):
    def test_derives_personal_entity_and_matches_email_case_insensitively(self):
        viewer = SimpleNamespace(
            email="LDU@NVIDIA.COM", username="ldu-personal", entity=None
        )
        identity = wandb_preflight.query_identity(
            FakeWandb(viewer),
            expected_email="ldu@nvidia.com",
            environment={"WANDB_MODE": "online"},
        )
        self.assertEqual(identity.entity, "ldu-personal")
        self.assertEqual(identity.username, "ldu-personal")
        self.assertTrue(identity.personal)

    def test_ignores_viewer_default_entity_and_uses_username(self):
        viewer = {
            "email": "ldu@nvidia.com",
            "username": "ldu",
            "entity": "nvidia-team",
        }
        identity = wandb_preflight.validate_identity(
            viewer,
            expected_email="ldu@nvidia.com",
        )
        self.assertEqual(identity.entity, "ldu")
        self.assertEqual(identity.username, "ldu")
        self.assertTrue(identity.personal)

    def test_default_team_entity_is_not_proof_of_personal_ownership(self):
        viewer = {
            "email": "ldu@nvidia.com",
            "username": "ldu",
            "entity": "nvidia-team",
        }
        with self.assertRaisesRegex(
            wandb_preflight.WandbPreflightError, "not the authenticated user's personal"
        ):
            wandb_preflight.validate_identity(
                viewer,
                expected_email="ldu@nvidia.com",
                requested_entity="nvidia-team",
            )

        identity = wandb_preflight.validate_identity(
            viewer,
            expected_email="ldu@nvidia.com",
            requested_entity="nvidia-team",
            require_personal=False,
        )
        self.assertEqual(identity.entity, "nvidia-team")
        self.assertFalse(identity.personal)

    def test_requires_username_even_when_viewer_has_default_entity(self):
        viewer = {
            "email": "ldu@nvidia.com",
            "username": None,
            "entity": "nvidia-team",
        }
        with self.assertRaisesRegex(
            wandb_preflight.WandbPreflightError, "personal username"
        ):
            wandb_preflight.validate_identity(
                viewer,
                expected_email="ldu@nvidia.com",
            )

    def test_rejects_wrong_authenticated_account_without_echoing_addresses(self):
        viewer = SimpleNamespace(email="someone@example.com", username="someone")
        with self.assertRaises(wandb_preflight.WandbPreflightError) as caught:
            wandb_preflight.validate_identity(
                viewer,
                expected_email="ldu@nvidia.com",
            )
        message = str(caught.exception)
        self.assertNotIn("someone@example.com", message)
        self.assertNotIn("ldu@nvidia.com", message)

    def test_rejects_team_entity_by_default(self):
        viewer = SimpleNamespace(email="ldu@nvidia.com", username="ldu", entity=None)
        with self.assertRaisesRegex(
            wandb_preflight.WandbPreflightError, "not the authenticated user's personal"
        ):
            wandb_preflight.validate_identity(
                viewer,
                expected_email="ldu@nvidia.com",
                requested_entity="some-team",
            )

        identity = wandb_preflight.validate_identity(
            viewer,
            expected_email="ldu@nvidia.com",
            requested_entity="some-team",
            require_personal=False,
        )
        self.assertFalse(identity.personal)

    def test_rejects_non_online_modes(self):
        for mode in ("offline", "dryrun", "disabled", "shared"):
            with self.subTest(mode=mode):
                with self.assertRaises(wandb_preflight.WandbPreflightError):
                    wandb_preflight.validate_mode({"WANDB_MODE": mode})

    def test_auth_failure_does_not_expose_exception_text_or_api_key(self):
        secret = "not-a-real-secret-key"
        fake = FakeWandb(error=RuntimeError(f"request failed with key {secret}"))
        with self.assertRaises(wandb_preflight.WandbPreflightError) as caught:
            wandb_preflight.query_identity(
                fake,
                expected_email="ldu@nvidia.com",
                environment={"WANDB_MODE": "online", "WANDB_API_KEY": secret},
            )
        self.assertNotIn(secret, str(caught.exception))
        self.assertIn("RuntimeError", str(caught.exception))

    def test_main_json_never_serializes_api_key(self):
        secret = "not-a-real-secret-key"
        fake = FakeWandb(
            SimpleNamespace(email="ldu@nvidia.com", username="ldu", entity=None)
        )
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(wandb_preflight, "_load_wandb", return_value=fake), mock.patch.dict(
            os.environ,
            {"WANDB_API_KEY": secret, "WANDB_MODE": "online"},
            clear=False,
        ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = wandb_preflight.main(
                ["--expected-email", "ldu@nvidia.com", "--format", "json"]
            )
        self.assertEqual(code, 0)
        combined = stdout.getvalue() + stderr.getvalue()
        self.assertNotIn(secret, combined)
        self.assertIn('"entity": "ldu"', stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
