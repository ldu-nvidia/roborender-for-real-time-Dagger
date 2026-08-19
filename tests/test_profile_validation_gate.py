import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/scaling/profile_validation_gate.py"
SPEC = importlib.util.spec_from_file_location("profile_validation_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)

FAST_REVISION = "1" * 40
GR00T_REVISION = "2" * 40


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ProfileValidationGateTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.validation = self.root / "validation"
        self.validation.mkdir()
        self.manifest = self.root / "profile_validation_manifest.txt"
        self.initial = self.root / "initial_policy.pt"
        self.final = self.root / "model_step_000001.pt"
        self.finite = self.validation / "dagger_bc_loss_step_000000001.json"
        self.ddp = self.validation / "ddp_policy_sync_step_000000001.json"
        self.initial.write_bytes(b"untrained-policy")
        self.final.write_bytes(b"trained-policy")

    def tearDown(self):
        self.temporary.cleanup()

    def write_json(self, path: Path, document):
        path.write_text(
            json.dumps(document, sort_keys=True) + "\n", encoding="utf-8"
        )

    def write_manifest(self, profile="smoke_32x300", overrides=None):
        distributed = profile == "smoke_256x300"
        nodes = 8 if distributed else 1
        envs = 256 if distributed else 32
        self.write_json(
            self.finite,
            {
                "schema": gate.FINITE_SCHEMA,
                "update": 1,
                "world_size": nodes,
                "loss": 0.25,
                "finite_verified": True,
                "nonnegative_verified": True,
            },
        )
        if distributed:
            self.write_json(
                self.ddp,
                {
                    "schema": gate.DDP_SCHEMA,
                    "verified": True,
                    "optimizer_update": 1,
                    "world_size": 8,
                    "policy_parameter_count": 123,
                    "policy_parameters_sha256": "3" * 64,
                },
            )
        fields = {
            "run_profile": profile,
            "fast_roborender_revision": FAST_REVISION,
            "gr00t_revision": GR00T_REVISION,
            "global_envs": str(envs),
            "rollout_steps": "300",
            "trainer_nodes": str(nodes),
            "policy_views": "head_color,hand_left,hand_right",
            "policy_view_count": "3",
            "policy_view_layout": "vertical",
            "policy_image_shape": "324x192x3",
            "initial_policy_checkpoint": str(self.initial),
            "initial_policy_sha256": sha256(self.initial),
            "final_checkpoint": str(self.final),
            "final_checkpoint_sha256": sha256(self.final),
            "finite_bc_loss_marker": str(self.finite),
            "finite_bc_loss_marker_sha256": sha256(self.finite),
            "ddp_sync_marker": str(self.ddp) if distributed else "",
            "ddp_sync_marker_sha256": sha256(self.ddp) if distributed else "",
            "camera_calibration_manifest": "",
            "camera_calibration_manifest_sha256": "",
            "rollout_iterations": "1",
            "status": "validated",
        }
        fields.update(overrides or {})
        self.manifest.write_text(
            "".join(f"{key}={value}\n" for key, value in fields.items()),
            encoding="utf-8",
        )
        return fields

    def validate(self, profile="smoke_32x300", **kwargs):
        return gate.validate_profile(
            self.manifest,
            profile,
            FAST_REVISION,
            GR00T_REVISION,
            **kwargs,
        )

    def test_accepts_exact_single_rank_smoke_and_empty_ddp_fields(self):
        self.write_manifest()
        report = self.validate(expected_manifest_sha256=sha256(self.manifest))
        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["global_envs"], 32)
        self.assertEqual(report["finite_bc_loss"], 0.25)
        self.assertEqual(report["ddp_sync_marker"], "")

    def test_accepts_exact_distributed_smoke_and_ddp_attestation(self):
        self.write_manifest("smoke_256x300")
        report = self.validate("smoke_256x300")
        self.assertEqual(report["trainer_nodes"], 8)
        self.assertEqual(report["ddp_policy_parameters_sha256"], "3" * 64)

    def test_rejects_duplicate_unknown_or_missing_manifest_keys(self):
        self.write_manifest()
        with self.manifest.open("a", encoding="utf-8") as handle:
            handle.write("status=validated\n")
        with self.assertRaisesRegex(gate.GateError, "duplicate manifest key"):
            self.validate()

        self.write_manifest()
        with self.manifest.open("a", encoding="utf-8") as handle:
            handle.write("unexpected=value\n")
        with self.assertRaisesRegex(gate.GateError, "unexpected"):
            self.validate()

        self.write_manifest()
        lines = self.manifest.read_text(encoding="utf-8").splitlines()
        self.manifest.write_text(
            "\n".join(line for line in lines if not line.startswith("status=")) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(gate.GateError, "missing"):
            self.validate()

    def test_rejects_profile_revision_topology_and_policy_tampering(self):
        cases = [
            ({"status": "awaiting"}, "status mismatch"),
            ({"global_envs": "31"}, "global_envs mismatch"),
            ({"fast_roborender_revision": "4" * 40}, "revision mismatch"),
            ({"policy_views": "head_color,hand_right,hand_left"}, "policy_views mismatch"),
            ({"policy_image_shape": "192x324x3"}, "policy_image_shape mismatch"),
        ]
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                self.write_manifest(overrides=overrides)
                with self.assertRaisesRegex(gate.GateError, message):
                    self.validate()

    def test_rejects_artifact_path_hash_and_content_substitution(self):
        self.write_manifest()
        self.final.write_bytes(b"substituted")
        with self.assertRaisesRegex(gate.GateError, "SHA-256 mismatch"):
            self.validate()

        self.write_manifest(
            overrides={"initial_policy_checkpoint": str(self.final)}
        )
        with self.assertRaisesRegex(gate.GateError, "must resolve to"):
            self.validate()

        self.write_manifest(
            overrides={"initial_policy_sha256": "uppercase-not-a-hash"}
        )
        with self.assertRaisesRegex(gate.GateError, "64-hex"):
            self.validate()

    def test_rejects_invalid_finite_loss_marker(self):
        invalid_markers = [
            ({"loss": float("nan")}, "finite and nonnegative"),
            ({"loss": -0.01}, "finite and nonnegative"),
            ({"world_size": 8}, "world_size mismatch"),
            ({"finite_verified": False}, "finite_verified mismatch"),
            ({"update": 2}, "update mismatch"),
        ]
        for marker_override, message in invalid_markers:
            with self.subTest(marker_override=marker_override):
                self.write_manifest()
                marker = json.loads(self.finite.read_text(encoding="utf-8"))
                marker.update(marker_override)
                self.write_json(self.finite, marker)
                fields = self.manifest.read_text(encoding="utf-8").replace(
                    f"finite_bc_loss_marker_sha256={sha256(self.finite)}", "unused"
                )
                # Rebuild the manifest so its marker hash remains accurate; the
                # semantic marker checks, not the outer hash, must reject it.
                self.write_manifest()
                marker = json.loads(self.finite.read_text(encoding="utf-8"))
                marker.update(marker_override)
                self.write_json(self.finite, marker)
                lines = []
                for line in self.manifest.read_text(encoding="utf-8").splitlines():
                    if line.startswith("finite_bc_loss_marker_sha256="):
                        line = f"finite_bc_loss_marker_sha256={sha256(self.finite)}"
                    lines.append(line)
                self.manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(gate.GateError, message):
                    self.validate()

    def test_rejects_missing_or_invalid_distributed_ddp_attestation(self):
        self.write_manifest("smoke_256x300")
        marker = json.loads(self.ddp.read_text(encoding="utf-8"))
        marker["verified"] = False
        self.write_json(self.ddp, marker)
        lines = []
        for line in self.manifest.read_text(encoding="utf-8").splitlines():
            if line.startswith("ddp_sync_marker_sha256="):
                line = f"ddp_sync_marker_sha256={sha256(self.ddp)}"
            lines.append(line)
        self.manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(gate.GateError, "verified mismatch"):
            self.validate("smoke_256x300")

        self.write_manifest("smoke_256x300", {"ddp_sync_marker": ""})
        with self.assertRaisesRegex(gate.GateError, "must not be empty"):
            self.validate("smoke_256x300")

    def test_single_rank_requires_both_ddp_fields_empty(self):
        self.write_manifest(overrides={"ddp_sync_marker_sha256": "0" * 64})
        with self.assertRaisesRegex(gate.GateError, "ddp_sync_marker_sha256 mismatch"):
            self.validate()

    def test_rejects_duplicate_marker_json_key_and_wrong_manifest_digest(self):
        self.write_manifest()
        self.finite.write_text(
            '{"schema":"one","schema":"two"}\n', encoding="utf-8"
        )
        lines = []
        for line in self.manifest.read_text(encoding="utf-8").splitlines():
            if line.startswith("finite_bc_loss_marker_sha256="):
                line = f"finite_bc_loss_marker_sha256={sha256(self.finite)}"
            lines.append(line)
        self.manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(gate.GateError, "duplicate JSON key"):
            self.validate()

        self.write_manifest()
        with self.assertRaisesRegex(gate.GateError, "manifest SHA-256 mismatch"):
            self.validate(expected_manifest_sha256="0" * 64)

    def test_cli_emits_machine_readable_report_and_fail_closed_exit(self):
        self.write_manifest()
        command = [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(self.manifest),
            "--expected-profile",
            "smoke_32x300",
            "--expected-fast-revision",
            FAST_REVISION,
            "--expected-gr00t-revision",
            GR00T_REVISION,
            "--expected-manifest-sha256",
            sha256(self.manifest),
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["status"], "verified")

        self.final.write_bytes(b"tampered")
        failed = subprocess.run(command, text=True, capture_output=True, check=False)
        self.assertEqual(failed.returncode, 2)
        self.assertIn("FATAL:", failed.stderr)


if __name__ == "__main__":
    unittest.main()
