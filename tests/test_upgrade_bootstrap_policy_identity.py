import json
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from round_dagger.actor import runtime_state_dict_sha256
from round_dagger.artifacts import sha256_file
from round_dagger.gr00t_policy import POLICY_SPEC_SCHEMA, load_policy_spec
from round_dagger.policy_bundle import validate_policy_bundle_identity
from round_dagger.policy_identity import runtime_policy_identity_sha256
from scripts.round_dagger.upgrade_bootstrap_policy_identity import (
    LEGACY_POLICY_SPEC_SCHEMA,
    BootstrapUpgradeError,
    upgrade_bundle,
)
from tests.policy_bundle_helpers import portable_policy_spec


def _different_sha256(value: str) -> str:
    return value[:-1] + ("0" if value[-1] != "0" else "1")


class BootstrapIdentityUpgradeTest(unittest.TestCase):
    def _legacy_fixture(
        self,
        root: Path,
        *,
        checkpoint_controllers: dict[str, str] | None = None,
    ) -> dict[str, object]:
        checkpoint = root / "legacy-student.pt"
        checkpoint_document = {
            "policy_state_dict": {
                "linear.bias": torch.tensor([0.25, -0.5], dtype=torch.float32),
                "linear.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            },
            "round": 0,
        }
        if checkpoint_controllers is not None:
            checkpoint_document["homie_controller_sha256"] = checkpoint_controllers
        torch.save(checkpoint_document, checkpoint)
        checkpoint_sha256 = sha256_file(checkpoint)
        policy_state_sha256 = runtime_state_dict_sha256(
            checkpoint_document["policy_state_dict"]
        )

        # A v2 spec had the same architecture and I/O contract, but its
        # runtime identity was only the serialization-independent policy-state
        # digest.  The migration adds both Homie controller identities.
        legacy_spec = portable_policy_spec(checkpoint_sha256, policy_state_sha256)
        legacy_spec["schema"] = LEGACY_POLICY_SPEC_SCHEMA
        legacy_spec["runtime_policy_sha256"] = policy_state_sha256
        legacy_spec.pop("policy_state_sha256")
        legacy_spec.pop("homie_controller_sha256")
        policy_spec = root / "legacy-student.pt.policy_spec.json"
        policy_spec.write_text(
            json.dumps(
                legacy_spec,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        walk_controller = root / "walk-controller.pt"
        stand_controller = root / "stand-controller.pt"
        walk_controller.write_bytes(b"unit-test-homie-walk-controller-v1")
        stand_controller.write_bytes(b"unit-test-homie-stand-controller-v1")
        return {
            "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint_sha256,
            "policy_spec": policy_spec,
            "policy_spec_sha256": sha256_file(policy_spec),
            "walk_controller": walk_controller,
            "walk_controller_sha256": sha256_file(walk_controller),
            "stand_controller": stand_controller,
            "stand_controller_sha256": sha256_file(stand_controller),
            "policy_state_sha256": policy_state_sha256,
        }

    @staticmethod
    def _upgrade(fixture: dict[str, object], output_dir: Path) -> dict[str, object]:
        return upgrade_bundle(
            checkpoint=fixture["checkpoint"],
            checkpoint_sha256=fixture["checkpoint_sha256"],
            policy_spec=fixture["policy_spec"],
            policy_spec_sha256=fixture["policy_spec_sha256"],
            walk_controller=fixture["walk_controller"],
            walk_controller_sha256=fixture["walk_controller_sha256"],
            stand_controller=fixture["stand_controller"],
            stand_controller_sha256=fixture["stand_controller_sha256"],
            output_dir=output_dir,
        )

    def test_upgrade_binds_policy_state_and_both_controllers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._legacy_fixture(root)
            source_hashes = {
                key: sha256_file(fixture[key])
                for key in (
                    "checkpoint",
                    "policy_spec",
                    "walk_controller",
                    "stand_controller",
                )
            }
            output_dir = root / "upgraded"

            report = self._upgrade(fixture, output_dir)

            checkpoint = output_dir / "student-v0.pt"
            policy_spec = Path(f"{checkpoint}.policy_spec.json")
            manifest = Path(f"{checkpoint}.manifest.json")
            migrated = torch.load(checkpoint, map_location="cpu", weights_only=False)
            controller_sha256 = {
                "walk": fixture["walk_controller_sha256"],
                "stand": fixture["stand_controller_sha256"],
            }
            expected_runtime = runtime_policy_identity_sha256(
                fixture["policy_state_sha256"], controller_sha256
            )

            self.assertEqual(
                migrated["homie_controller_sha256"], controller_sha256
            )
            spec = load_policy_spec(policy_spec)
            self.assertEqual(spec["schema"], POLICY_SPEC_SCHEMA)
            self.assertEqual(
                spec["policy_state_sha256"], fixture["policy_state_sha256"]
            )
            self.assertEqual(spec["runtime_policy_sha256"], expected_runtime)
            self.assertEqual(report["runtime_policy_sha256"], expected_runtime)
            self.assertEqual(report["checkpoint_sha256"], sha256_file(checkpoint))
            self.assertEqual(report["policy_spec_sha256"], sha256_file(policy_spec))
            self.assertEqual(report["manifest_sha256"], sha256_file(manifest))
            validated = validate_policy_bundle_identity(
                checkpoint=checkpoint,
                policy_spec=policy_spec,
                manifest=manifest,
                expected_checkpoint_sha256=report["checkpoint_sha256"],
                expected_runtime_policy_sha256=expected_runtime,
            )
            self.assertEqual(validated, report["preflight"])
            self.assertEqual(
                source_hashes,
                {
                    key: sha256_file(fixture[key])
                    for key in source_hashes
                },
            )

    def test_output_is_read_only_and_destination_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._legacy_fixture(root)
            output_dir = root / "upgraded"
            first = self._upgrade(fixture, output_dir)
            output_paths = (
                output_dir / "student-v0.pt",
                output_dir / "student-v0.pt.policy_spec.json",
                output_dir / "student-v0.pt.manifest.json",
            )
            before = {path: sha256_file(path) for path in output_paths}

            for path in output_paths:
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o444)
                self.assertFalse(path.is_symlink())
            with self.assertRaisesRegex(
                BootstrapUpgradeError, "output directory already exists"
            ):
                self._upgrade(fixture, output_dir)
            self.assertEqual(before, {path: sha256_file(path) for path in output_paths})
            self.assertEqual(first["checkpoint_sha256"], before[output_paths[0]])

    def test_input_digest_mismatches_fail_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._legacy_fixture(root)
            digest_fields = (
                "checkpoint_sha256",
                "policy_spec_sha256",
                "walk_controller_sha256",
                "stand_controller_sha256",
            )
            for index, field in enumerate(digest_fields):
                with self.subTest(field=field):
                    changed = dict(fixture)
                    changed[field] = _different_sha256(changed[field])
                    output_dir = root / f"digest-mismatch-{index}"
                    with self.assertRaisesRegex(
                        BootstrapUpgradeError, "SHA-256 mismatch"
                    ):
                        self._upgrade(changed, output_dir)
                    self.assertFalse(output_dir.exists())

    def test_internal_identity_mismatches_fail_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._legacy_fixture(root)
            source_spec = json.loads(fixture["policy_spec"].read_text("utf-8"))
            source_spec["checkpoint_sha256"] = _different_sha256(
                source_spec["checkpoint_sha256"]
            )
            fixture["policy_spec"].write_text(
                json.dumps(source_spec, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            fixture["policy_spec_sha256"] = sha256_file(fixture["policy_spec"])
            output_dir = root / "spec-checkpoint-mismatch"
            with self.assertRaisesRegex(
                BootstrapUpgradeError, "legacy policy spec/checkpoint mismatch"
            ):
                self._upgrade(fixture, output_dir)
            self.assertFalse(output_dir.exists())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._legacy_fixture(root)
            source_spec = json.loads(fixture["policy_spec"].read_text("utf-8"))
            source_spec["runtime_policy_sha256"] = _different_sha256(
                source_spec["runtime_policy_sha256"]
            )
            fixture["policy_spec"].write_text(
                json.dumps(source_spec, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            fixture["policy_spec_sha256"] = sha256_file(fixture["policy_spec"])
            output_dir = root / "runtime-state-mismatch"
            with self.assertRaisesRegex(
                BootstrapUpgradeError,
                "runtime identity differs from checkpoint state",
            ):
                self._upgrade(fixture, output_dir)
            self.assertFalse(output_dir.exists())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wrong_controllers = {"walk": "a" * 64, "stand": "b" * 64}
            fixture = self._legacy_fixture(
                root, checkpoint_controllers=wrong_controllers
            )
            output_dir = root / "controller-identity-mismatch"
            with self.assertRaisesRegex(
                BootstrapUpgradeError,
                "legacy checkpoint already binds different Homie controllers",
            ):
                self._upgrade(fixture, output_dir)
            self.assertFalse(output_dir.exists())

    def test_post_write_failure_removes_staging_and_destination(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._legacy_fixture(root)
            output_dir = root / "upgraded"
            with mock.patch(
                "scripts.round_dagger.upgrade_bootstrap_policy_identity."
                "validate_policy_bundle_identity",
                side_effect=RuntimeError("injected final validation failure"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "injected final validation failure"
                ):
                    self._upgrade(fixture, output_dir)
            self.assertFalse(output_dir.exists())
            self.assertEqual(
                [], list(root.glob(f".{output_dir.name}.staging.*"))
            )


if __name__ == "__main__":
    unittest.main()
