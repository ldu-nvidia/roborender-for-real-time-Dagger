import hashlib
import importlib.util
import json
import os
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import unittest
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/scaling/camera_calibration_gate.py"
SPEC = importlib.util.spec_from_file_location("camera_calibration_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)

FAST_REVISION = "1" * 40
GR00T_REVISION = "2" * 40
RUN_ID = "calibration_3view_fixture"
PROVENANCE = {
    "fast_roborender_source_tree_sha256": "3" * 64,
    "gr00t_source_tree_sha256": "4" * 64,
    "source_snapshot_attestation_sha256": "5" * 64,
    "source_snapshot_inventory_sha256": "6" * 64,
    "immutable_asset_contract_sha256": "7" * 64,
}


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
            "run_id": RUN_ID,
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

    def write_publishable_manifest(self):
        document = self.manifest_document()
        document.update(PROVENANCE)
        self.write_manifest(document)

    def publish(self, destination=None):
        destination = destination or self.root / f"camera_calibration_manifest_{RUN_ID}.json"
        return gate.publish_technical_manifest_once(
            self.manifest,
            destination,
            self.root,
            RUN_ID,
            FAST_REVISION,
            GR00T_REVISION,
            PROVENANCE["fast_roborender_source_tree_sha256"],
            PROVENANCE["gr00t_source_tree_sha256"],
            PROVENANCE["source_snapshot_attestation_sha256"],
            PROVENANCE["source_snapshot_inventory_sha256"],
            PROVENANCE["immutable_asset_contract_sha256"],
        )

    def publish_command(self, destination):
        return [
            sys.executable,
            str(SCRIPT),
            "publish",
            "--manifest",
            str(self.manifest),
            "--destination",
            str(destination),
            "--validation-directory",
            str(self.root),
            "--expected-run-id",
            RUN_ID,
            "--expected-fast-roborender-revision",
            FAST_REVISION,
            "--expected-gr00t-revision",
            GR00T_REVISION,
            "--expected-fast-source-tree-sha256",
            PROVENANCE["fast_roborender_source_tree_sha256"],
            "--expected-gr00t-source-tree-sha256",
            PROVENANCE["gr00t_source_tree_sha256"],
            "--expected-source-snapshot-attestation-sha256",
            PROVENANCE["source_snapshot_attestation_sha256"],
            "--expected-source-snapshot-inventory-sha256",
            PROVENANCE["source_snapshot_inventory_sha256"],
            "--expected-asset-contract-sha256",
            PROVENANCE["immutable_asset_contract_sha256"],
        ]

    def accept(self):
        return gate.accept_manifest(
            self.manifest,
            self.acceptance,
            "reviewer@example.com",
            gate.CONFIRMATION,
            FAST_REVISION,
            GR00T_REVISION,
            expected_run_id=RUN_ID,
        )

    def verify(self, **kwargs):
        return gate.verify_acceptance(
            self.acceptance,
            FAST_REVISION,
            GR00T_REVISION,
            manifest_path=self.manifest,
            expected_run_id=RUN_ID,
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
        self.assertEqual(report["run_id"], RUN_ID)

    def test_technical_manifest_run_id_must_match_exact_expected_run(self):
        with self.assertRaisesRegex(gate.GateError, "run_id mismatch"):
            gate.validate_technical_manifest(
                self.manifest,
                FAST_REVISION,
                GR00T_REVISION,
                expected_run_id="another_calibration_run",
            )

    def test_accept_and_verify_bind_full_sealed_source_provenance(self):
        document = self.manifest_document()
        document.update(PROVENANCE)
        self.write_manifest(document)
        accepted = gate.accept_manifest(
            self.manifest,
            self.acceptance,
            "reviewer@example.com",
            gate.CONFIRMATION,
            FAST_REVISION,
            GR00T_REVISION,
            PROVENANCE["fast_roborender_source_tree_sha256"],
            PROVENANCE["gr00t_source_tree_sha256"],
            PROVENANCE["source_snapshot_attestation_sha256"],
            PROVENANCE["source_snapshot_inventory_sha256"],
            PROVENANCE["immutable_asset_contract_sha256"],
        )
        report = self.verify(
            expected_fast_source_tree_sha256=PROVENANCE[
                "fast_roborender_source_tree_sha256"
            ],
            expected_gr00t_source_tree_sha256=PROVENANCE[
                "gr00t_source_tree_sha256"
            ],
            expected_source_snapshot_attestation_sha256=PROVENANCE[
                "source_snapshot_attestation_sha256"
            ],
            expected_source_snapshot_inventory_sha256=PROVENANCE[
                "source_snapshot_inventory_sha256"
            ],
            expected_asset_contract_sha256=PROVENANCE[
                "immutable_asset_contract_sha256"
            ],
        )
        for key, value in PROVENANCE.items():
            self.assertEqual(accepted[key], value)
            self.assertEqual(report[key], value)

        with self.assertRaisesRegex(
            gate.GateError, "source_snapshot_attestation_sha256 mismatch"
        ):
            self.verify(
                expected_fast_source_tree_sha256=PROVENANCE[
                    "fast_roborender_source_tree_sha256"
                ],
                expected_gr00t_source_tree_sha256=PROVENANCE[
                    "gr00t_source_tree_sha256"
                ],
                expected_source_snapshot_attestation_sha256="8" * 64,
                expected_source_snapshot_inventory_sha256=PROVENANCE[
                    "source_snapshot_inventory_sha256"
                ],
                expected_asset_contract_sha256=PROVENANCE[
                    "immutable_asset_contract_sha256"
                ],
            )

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

    def test_technical_manifest_symlink_is_rejected(self):
        real_manifest = self.root / "real-camera-calibration-manifest.json"
        self.manifest.rename(real_manifest)
        self.manifest.symlink_to(real_manifest.name)
        with self.assertRaisesRegex(gate.GateError, "regular non-symlink"):
            self.accept()

    def test_panel_symlink_is_rejected(self):
        real_panel = self.root / "real-step000_env0.png"
        self.panels[0].rename(real_panel)
        self.panels[0].symlink_to(real_panel.name)
        with self.assertRaisesRegex(gate.GateError, "regular non-symlink"):
            self.accept()

    def test_verify_rejects_wrong_revision_and_acceptance_hash(self):
        self.accept()
        with self.assertRaisesRegex(gate.GateError, "fast_roborender_revision mismatch"):
            gate.verify_acceptance(
                self.acceptance,
                "3" * 40,
                GR00T_REVISION,
                manifest_path=self.manifest,
                expected_run_id=RUN_ID,
            )
        with self.assertRaisesRegex(gate.GateError, "acceptance SHA-256 mismatch"):
            self.verify(expected_acceptance_sha256="0" * 64)

    def test_acceptance_symlink_is_rejected(self):
        self.accept()
        real_acceptance = self.root / "real-camera-calibration-acceptance.json"
        self.acceptance.rename(real_acceptance)
        self.acceptance.symlink_to(real_acceptance.name)
        with self.assertRaisesRegex(gate.GateError, "regular non-symlink"):
            self.verify()

    def test_refuses_to_overwrite_prior_acceptance(self):
        self.accept()
        with self.assertRaisesRegex(gate.GateError, "refusing to replace"):
            self.accept()

    def test_verify_requires_external_run_id_binding(self):
        self.accept()
        with self.assertRaisesRegex(gate.GateError, "expected run_id is required"):
            gate.verify_acceptance(
                self.acceptance,
                FAST_REVISION,
                GR00T_REVISION,
                manifest_path=self.manifest,
            )

    def test_acceptance_create_rejects_symlink_redirect_without_creating_target(self):
        redirected = self.root / "redirected-acceptance.json"
        self.acceptance.symlink_to(redirected)
        with self.assertRaisesRegex(gate.GateError, "not a safe regular file"):
            self.accept()
        self.assertTrue(self.acceptance.is_symlink())
        self.assertFalse(redirected.exists())
        self.assertFalse(list(self.root.glob(f".{self.acceptance.name}.tmp.*")))

    def test_verify_acceptance_hash_and_parse_use_one_held_snapshot(self):
        self.accept()
        accepted_bytes = self.acceptance.read_bytes()
        accepted_hash = hashlib.sha256(accepted_bytes).hexdigest()
        changed = json.loads(accepted_bytes)
        changed["reviewer"] = "swapped@example.com"
        replacement = self.root / "acceptance-replacement.json"
        replacement.write_text(
            json.dumps(changed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        load_json_bytes = gate.load_json_bytes
        swapped = False

        def swap_after_read(payload, label):
            nonlocal swapped
            if not swapped and Path(label) == self.acceptance:
                os.replace(replacement, self.acceptance)
                swapped = True
            return load_json_bytes(payload, label)

        with mock.patch.object(gate, "load_json_bytes", side_effect=swap_after_read):
            report = self.verify(expected_acceptance_sha256=accepted_hash)
        self.assertTrue(swapped)
        self.assertEqual(report["reviewer"], "reviewer@example.com")
        self.assertEqual(report["acceptance_sha256"], accepted_hash)
        self.assertEqual(
            json.loads(self.acceptance.read_text(encoding="utf-8"))["reviewer"],
            "swapped@example.com",
        )

    def test_verify_rejects_acceptance_path_swap_during_held_read(self):
        self.accept()
        replacement = self.root / "acceptance-read-race.json"
        changed = json.loads(self.acceptance.read_text(encoding="utf-8"))
        changed["reviewer"] = "swapped@example.com"
        replacement.write_text(
            json.dumps(changed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        acceptance_inode = self.acceptance.stat().st_ino
        read_descriptor = gate._read_descriptor
        swapped = False

        def swap_visible_path(descriptor):
            nonlocal swapped
            if not swapped and os.fstat(descriptor).st_ino == acceptance_inode:
                os.replace(replacement, self.acceptance)
                swapped = True
            return read_descriptor(descriptor)

        with mock.patch.object(
            gate, "_read_descriptor", side_effect=swap_visible_path
        ):
            with self.assertRaisesRegex(gate.GateError, "changed while it was read"):
                self.verify()
        self.assertTrue(swapped)

    def test_verify_source_manifest_hash_and_parse_use_one_held_snapshot(self):
        self.accept()
        manifest_bytes = self.manifest.read_bytes()
        manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
        replacement = self.root / "manifest-replacement.json"
        replacement.write_bytes(manifest_bytes + b"\n")
        load_json_bytes = gate.load_json_bytes
        swapped = False

        def swap_after_read(payload, label):
            nonlocal swapped
            if not swapped and Path(label) == self.manifest:
                os.replace(replacement, self.manifest)
                swapped = True
            return load_json_bytes(payload, label)

        with mock.patch.object(gate, "load_json_bytes", side_effect=swap_after_read):
            report = self.verify()
        self.assertTrue(swapped)
        self.assertEqual(report["source_manifest_sha256"], manifest_hash)
        self.assertNotEqual(hashlib.sha256(self.manifest.read_bytes()).hexdigest(), manifest_hash)

    def test_verify_rejects_manifest_path_swap_during_held_read(self):
        self.accept()
        replacement = self.root / "manifest-read-race.json"
        replacement.write_bytes(self.manifest.read_bytes() + b"\n")
        manifest_inode = self.manifest.stat().st_ino
        read_descriptor = gate._read_descriptor
        swapped = False

        def swap_visible_path(descriptor):
            nonlocal swapped
            if not swapped and os.fstat(descriptor).st_ino == manifest_inode:
                os.replace(replacement, self.manifest)
                swapped = True
            return read_descriptor(descriptor)

        with mock.patch.object(
            gate, "_read_descriptor", side_effect=swap_visible_path
        ):
            with self.assertRaisesRegex(gate.GateError, "changed while it was read"):
                self.verify()
        self.assertTrue(swapped)

    def test_atomic_publication_is_idempotent_only_for_identical_bytes(self):
        self.write_publishable_manifest()
        destination = self.root / f"camera_calibration_manifest_{RUN_ID}.json"
        source_bytes = self.manifest.read_bytes()
        first = self.publish(destination)
        second = self.publish(destination)
        self.assertEqual(first["publication"], "published")
        self.assertEqual(second["publication"], "already_present")
        self.assertEqual(destination.read_bytes(), source_bytes)
        self.assertEqual(destination.stat().st_mode & 0o777, 0o444)

        # A different but still valid source manifest must never replace it.
        self.manifest.write_bytes(source_bytes + b"\n")
        with self.assertRaisesRegex(gate.GateError, "refusing to replace different"):
            self.publish(destination)
        self.assertEqual(destination.read_bytes(), source_bytes)
        self.assertFalse(list(self.root.glob(f".{destination.name}.tmp.*")))

    def test_atomic_publication_rejects_a_symlink_installed_at_link_race(self):
        destination = self.root / f"camera_calibration_manifest_{RUN_ID}.json"
        victim = self.root / "victim.json"
        victim.write_bytes(b"do-not-touch")

        def install_symlink_then_collide(*_args, **kwargs):
            os.symlink(
                victim.name,
                destination.name,
                dir_fd=kwargs["dst_dir_fd"],
            )
            raise FileExistsError(destination)

        with mock.patch.object(gate.os, "link", side_effect=install_symlink_then_collide):
            with self.assertRaisesRegex(gate.GateError, "not a safe regular file"):
                gate.atomic_publish_bytes_once(b"validated", destination, self.root)
        self.assertTrue(destination.is_symlink())
        self.assertEqual(victim.read_bytes(), b"do-not-touch")
        self.assertFalse(list(self.root.glob(f".{destination.name}.tmp.*")))

    def test_atomic_publication_rechecks_private_inode_mutated_at_link(self):
        destination = self.root / f"camera_calibration_manifest_{RUN_ID}.json"
        payload = b"complete-validated-document"
        corrupted = b"corrupted-after-fsync"
        real_link = gate.os.link

        def mutate_then_link(source, target, **kwargs):
            os.chmod(
                source,
                0o600,
                dir_fd=kwargs["src_dir_fd"],
                follow_symlinks=False,
            )
            descriptor = os.open(
                source,
                os.O_WRONLY | os.O_TRUNC,
                dir_fd=kwargs["src_dir_fd"],
            )
            try:
                os.write(descriptor, corrupted)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            return real_link(source, target, **kwargs)

        with mock.patch.object(gate.os, "link", side_effect=mutate_then_link):
            with self.assertRaisesRegex(
                gate.GateError, "held calibration publication inode changed"
            ):
                gate.atomic_publish_bytes_once(payload, destination, self.root)
        self.assertEqual(destination.read_bytes(), corrupted)
        self.assertFalse(list(self.root.glob(f".{destination.name}.tmp.*")))

    def test_idempotent_publication_rejects_mutable_existing_mode(self):
        destination = self.root / f"camera_calibration_manifest_{RUN_ID}.json"
        payload = b"complete-validated-document"
        destination.write_bytes(payload)
        destination.chmod(0o600)
        with self.assertRaisesRegex(gate.GateError, "mode must be exactly 0444"):
            gate.atomic_publish_bytes_once(payload, destination, self.root)
        self.assertEqual(destination.read_bytes(), payload)
        self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o600)
        self.assertFalse(list(self.root.glob(f".{destination.name}.tmp.*")))

    def test_idempotent_publication_rechecks_existing_inode_after_read(self):
        destination = self.root / f"camera_calibration_manifest_{RUN_ID}.json"
        payload = b"complete-validated-document"
        corrupted = b"corrupted-after-existing-read"
        self.assertEqual(
            gate.atomic_publish_bytes_once(payload, destination, self.root),
            "published",
        )
        read_existing = gate._read_existing_publication

        def read_then_mutate(*args, **kwargs):
            snapshot = read_existing(*args, **kwargs)
            destination.chmod(0o600)
            destination.write_bytes(corrupted)
            return snapshot

        with mock.patch.object(
            gate, "_read_existing_publication", side_effect=read_then_mutate
        ):
            with self.assertRaisesRegex(
                gate.GateError, "held calibration publication inode changed"
            ):
                gate.atomic_publish_bytes_once(payload, destination, self.root)
        self.assertEqual(destination.read_bytes(), corrupted)
        self.assertFalse(list(self.root.glob(f".{destination.name}.tmp.*")))

    def test_atomic_publication_fifo_collision_returns_promptly_and_cleans_temp(self):
        self.write_publishable_manifest()
        destination = self.root / f"camera_calibration_manifest_{RUN_ID}.json"
        os.mkfifo(destination)
        completed = subprocess.run(
            self.publish_command(destination),
            text=True,
            capture_output=True,
            check=False,
            timeout=3,
        )
        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertIn("regular", completed.stderr)
        self.assertTrue(destination.is_fifo())
        self.assertFalse(list(self.root.glob(f".{destination.name}.tmp.*")))

    def test_source_manifest_fifo_returns_promptly_without_publication(self):
        self.write_publishable_manifest()
        destination = self.root / f"camera_calibration_manifest_{RUN_ID}.json"
        self.manifest.unlink()
        os.mkfifo(self.manifest)
        completed = subprocess.run(
            self.publish_command(destination),
            text=True,
            capture_output=True,
            check=False,
            timeout=3,
        )
        self.assertEqual(completed.returncode, 2, completed.stderr)
        self.assertIn("regular non-symlink", completed.stderr)
        self.assertFalse(destination.exists())
        self.assertFalse(list(self.root.glob(f".{destination.name}.tmp.*")))

    def test_atomic_publication_interruption_leaves_no_partial_or_private_temp(self):
        destination = self.root / f"camera_calibration_manifest_{RUN_ID}.json"
        with mock.patch.object(gate.os, "link", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                gate.atomic_publish_bytes_once(b"complete-bytes", destination, self.root)
        self.assertFalse(destination.exists())
        self.assertFalse(list(self.root.glob(f".{destination.name}.tmp.*")))

    def test_atomic_publication_concurrent_collision_never_replaces_winner(self):
        destination = self.root / f"camera_calibration_manifest_{RUN_ID}.json"
        payloads = (b"first-complete-document", b"second-complete-document")

        def publish(payload):
            try:
                return gate.atomic_publish_bytes_once(payload, destination, self.root)
            except gate.GateError as exc:
                return str(exc)

        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(publish, payloads))
        self.assertIn("published", outcomes)
        self.assertEqual(sum("refusing to replace different" in item for item in outcomes), 1)
        self.assertIn(destination.read_bytes(), payloads)
        self.assertFalse(list(self.root.glob(f".{destination.name}.tmp.*")))

    def test_atomic_publication_concurrent_identical_retry_is_accepted(self):
        destination = self.root / f"camera_calibration_manifest_{RUN_ID}.json"
        payload = b"one-complete-document"
        with ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(
                executor.map(
                    lambda _: gate.atomic_publish_bytes_once(
                        payload, destination, self.root
                    ),
                    range(2),
                )
            )
        self.assertEqual(sorted(outcomes), ["already_present", "published"])
        self.assertEqual(destination.read_bytes(), payload)

    def test_successful_retry_survives_first_publishers_post_link_error(self):
        destination = self.root / f"camera_calibration_manifest_{RUN_ID}.json"
        payload = b"one-complete-document"
        final_linked = threading.Event()
        retry_returned = threading.Event()
        first_error = []
        real_link = gate.os.link
        real_verify = gate._verify_visible_entry

        def link_then_signal(*args, **kwargs):
            result = real_link(*args, **kwargs)
            final_linked.set()
            return result

        def pause_first_after_link(*args, **kwargs):
            if threading.current_thread().name == "first-publisher":
                self.assertTrue(retry_returned.wait(timeout=3))
                raise gate.GateError("synthetic post-link failure")
            return real_verify(*args, **kwargs)

        def first_publish():
            try:
                gate.atomic_publish_bytes_once(payload, destination, self.root)
            except gate.GateError as exc:
                first_error.append(str(exc))

        with mock.patch.object(gate.os, "link", side_effect=link_then_signal), \
             mock.patch.object(
                 gate, "_verify_visible_entry", side_effect=pause_first_after_link
             ):
            first = threading.Thread(target=first_publish, name="first-publisher")
            first.start()
            self.assertTrue(final_linked.wait(timeout=3))
            outcome = gate.atomic_publish_bytes_once(payload, destination, self.root)
            retry_returned.set()
            first.join(timeout=3)

        self.assertFalse(first.is_alive())
        self.assertEqual(outcome, "already_present")
        self.assertEqual(first_error, ["synthetic post-link failure"])
        self.assertEqual(destination.read_bytes(), payload)
        self.assertFalse(list(self.root.glob(f".{destination.name}.tmp.*")))

    def test_atomic_publication_rejects_directory_swap_without_false_success(self):
        publication_dir = self.root / "publication"
        publication_dir.mkdir()
        detached_dir = self.root / "publication-detached"
        destination = publication_dir / "record.json"
        real_link = gate.os.link

        def swap_directory_then_link(*args, **kwargs):
            publication_dir.rename(detached_dir)
            publication_dir.mkdir()
            return real_link(*args, **kwargs)

        with mock.patch.object(
            gate.os, "link", side_effect=swap_directory_then_link
        ):
            with self.assertRaisesRegex(gate.GateError, "directory changed"):
                gate.atomic_publish_bytes_once(
                    b"complete-record", destination, publication_dir
                )
        self.assertFalse(destination.exists())
        self.assertEqual(
            (detached_dir / destination.name).read_bytes(), b"complete-record"
        )
        self.assertFalse(list(publication_dir.glob(".*.tmp.*")))
        self.assertFalse(list(detached_dir.glob(".*.tmp.*")))

    def test_publication_rejects_manifest_mutation_after_validation(self):
        self.write_publishable_manifest()
        destination = self.root / f"camera_calibration_manifest_{RUN_ID}.json"
        capture = gate._capture_technical_snapshot

        def capture_then_mutate(*args, **kwargs):
            snapshot = capture(*args, **kwargs)
            self.manifest.write_bytes(self.manifest.read_bytes() + b"\n")
            return snapshot

        with mock.patch.object(
            gate, "_capture_technical_snapshot", side_effect=capture_then_mutate
        ):
            with self.assertRaisesRegex(gate.GateError, "changed before publication"):
                self.publish(destination)
        self.assertFalse(destination.exists())

    def test_publication_rejects_panel_mutation_after_validation(self):
        self.write_publishable_manifest()
        destination = self.root / f"camera_calibration_manifest_{RUN_ID}.json"
        capture = gate._capture_technical_snapshot

        def capture_then_mutate(*args, **kwargs):
            snapshot = capture(*args, **kwargs)
            self.panels[0].write_bytes(b"mutated-panel")
            return snapshot

        with mock.patch.object(
            gate, "_capture_technical_snapshot", side_effect=capture_then_mutate
        ):
            with self.assertRaisesRegex(gate.GateError, "panel 0 changed"):
                self.publish(destination)
        self.assertFalse(destination.exists())

    def test_calibration_run_root_reservation_rejects_reused_run_id(self):
        runtime = self.root / "runtime"
        runtime.mkdir()
        run_root = runtime / RUN_ID
        self.assertEqual(
            gate.reserve_calibration_run_root(runtime, run_root, RUN_ID), run_root
        )
        with self.assertRaisesRegex(gate.GateError, "refuses to reuse existing RUN_ROOT"):
            gate.reserve_calibration_run_root(runtime, run_root, RUN_ID)
        self.assertTrue(run_root.is_dir())

    def test_calibration_run_root_reservation_rejects_runtime_directory_swap(self):
        runtime = self.root / "runtime"
        runtime.mkdir()
        detached = self.root / "runtime-detached"
        run_root = runtime / RUN_ID
        real_mkdir = gate.os.mkdir

        def reserve_then_swap(path, mode=0o777, *, dir_fd=None):
            real_mkdir(path, mode, dir_fd=dir_fd)
            runtime.rename(detached)
            real_mkdir(runtime, 0o755)

        with mock.patch.object(gate.os, "mkdir", side_effect=reserve_then_swap):
            with self.assertRaisesRegex(gate.GateError, "directory changed"):
                gate.reserve_calibration_run_root(runtime, run_root, RUN_ID)
        self.assertFalse(run_root.exists())
        self.assertFalse((detached / RUN_ID).exists())

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
            "--expected-run-id",
            RUN_ID,
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(report["status"], "verified")


if __name__ == "__main__":
    unittest.main()
