import hashlib
import importlib.util
import json
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/scaling/camera_calibration_gate.py"
SPEC = importlib.util.spec_from_file_location("camera_calibration_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)

FAST_REVISION = "1" * 40
GR00T_REVISION = "2" * 40


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(payload, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def write_rgb_png(path: Path, width: int = 1248, height: int = 720) -> None:
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    scanline = b"\x00" + b"\x00\x00\x00" * width
    payload = zlib.compress(scanline * height, level=9)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"IDAT", payload)
        + png_chunk(b"IEND", b"")
    )


class CameraCalibrationGateTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.panels = [
            self.root / "step000_env0.png",
            self.root / "step001_env0.png",
        ]
        for panel in self.panels:
            write_rgb_png(panel)
        self.manifest = self.root / "camera_calibration_manifest.json"
        self.acceptance = self.root / "camera_calibration_acceptance.json"
        self.write_manifest()

    def tearDown(self):
        self.temporary.cleanup()

    def manifest_document(self):
        return {
            "schema": gate.TECHNICAL_SCHEMA,
            "status": gate.TECHNICAL_STATUS,
            "fast_roborender_revision": FAST_REVISION,
            "gr00t_revision": GR00T_REVISION,
            "policy_views": list(gate.EXPECTED_VIEWS),
            "columns": list(gate.EXPECTED_VIEWS),
            "rows": list(gate.EXPECTED_ROWS),
            "tile_shape_hwc": list(gate.EXPECTED_TILE_SHAPE_HWC),
            "panel_shape_hwc": list(gate.EXPECTED_PANEL_SHAPE_HWC),
            "panels": [
                {"path": panel.name, "sha256": sha256(panel)} for panel in self.panels
            ],
        }

    def write_manifest(self, document=None):
        document = self.manifest_document() if document is None else document
        self.manifest.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def accept(self):
        return gate.accept_manifest(
            self.manifest,
            self.acceptance,
            "reviewer@example.com",
            gate.CONFIRMATION,
            FAST_REVISION,
            GR00T_REVISION,
        )

    def verify(self, **kwargs):
        return gate.verify_acceptance(
            self.acceptance,
            FAST_REVISION,
            GR00T_REVISION,
            manifest_path=self.manifest,
            **kwargs,
        )

    def test_accept_and_verify_bind_exact_artifacts_and_revisions(self):
        accepted = self.accept()
        report = self.verify(expected_acceptance_sha256=sha256(self.acceptance))
        self.assertEqual(accepted["status"], "accepted")
        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["policy_views"], list(gate.EXPECTED_VIEWS))
        self.assertEqual(report["panels_verified"], 2)
        self.assertEqual(report["fast_roborender_revision"], FAST_REVISION)
        self.assertEqual(report["gr00t_revision"], GR00T_REVISION)

    def test_accept_requires_exact_confirmation_and_explicit_reviewer(self):
        with self.assertRaisesRegex(gate.GateError, "explicit confirmation"):
            gate.accept_manifest(
                self.manifest,
                self.acceptance,
                "reviewer@example.com",
                "yes",
                FAST_REVISION,
                GR00T_REVISION,
            )
        with self.assertRaisesRegex(gate.GateError, "reviewer"):
            gate.accept_manifest(
                self.manifest,
                self.acceptance,
                "   ",
                gate.CONFIRMATION,
                FAST_REVISION,
                GR00T_REVISION,
            )

    def test_rejects_wrong_ordered_views(self):
        document = self.manifest_document()
        document["columns"] = ["head_color", "hand_right", "hand_left"]
        self.write_manifest(document)
        with self.assertRaisesRegex(gate.GateError, "columns must equal"):
            self.accept()

    def test_rejects_declared_panel_hash_mismatch(self):
        document = self.manifest_document()
        document["panels"][0]["sha256"] = "0" * 64
        self.write_manifest(document)
        with self.assertRaisesRegex(gate.GateError, "SHA-256 mismatch"):
            self.accept()

    def test_rejects_wrong_panel_dimensions(self):
        write_rgb_png(self.panels[1], width=1247)
        document = self.manifest_document()
        self.write_manifest(document)
        with self.assertRaisesRegex(gate.GateError, "shape must be"):
            self.accept()

    def test_verify_detects_panel_substitution(self):
        self.accept()
        write_rgb_png(self.panels[0], width=1248, height=719)
        with self.assertRaisesRegex(gate.GateError, "SHA-256 mismatch"):
            self.verify()

    def test_verify_detects_source_manifest_mutation(self):
        self.accept()
        document = self.manifest_document()
        document["unexpected"] = True
        self.write_manifest(document)
        with self.assertRaisesRegex(gate.GateError, "technical manifest SHA-256 mismatch"):
            self.verify()

    def test_verify_rejects_wrong_revision_and_acceptance_hash(self):
        self.accept()
        with self.assertRaisesRegex(gate.GateError, "fast_roborender_revision mismatch"):
            gate.verify_acceptance(
                self.acceptance,
                "3" * 40,
                GR00T_REVISION,
                manifest_path=self.manifest,
            )
        with self.assertRaisesRegex(gate.GateError, "acceptance SHA-256 mismatch"):
            self.verify(expected_acceptance_sha256="0" * 64)

    def test_refuses_to_overwrite_prior_acceptance(self):
        self.accept()
        with self.assertRaisesRegex(gate.GateError, "refusing to replace"):
            self.accept()

    def test_rejects_duplicate_json_keys(self):
        self.manifest.write_text(
            '{"schema": "first", "schema": "second"}\n', encoding="utf-8"
        )
        with self.assertRaisesRegex(gate.GateError, "duplicate JSON key"):
            self.accept()

    def test_command_line_round_trip(self):
        command = [
            sys.executable,
            str(SCRIPT),
            "accept",
            "--manifest",
            str(self.manifest),
            "--output",
            str(self.acceptance),
            "--reviewer",
            "reviewer@example.com",
            "--confirm",
            gate.CONFIRMATION,
            "--expected-fast-roborender-revision",
            FAST_REVISION,
            "--expected-gr00t-revision",
            GR00T_REVISION,
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(report["status"], "verified")


if __name__ == "__main__":
    unittest.main()
