from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.round_dagger import verify_sealed_source_snapshot as verifier


FAST_REVISION = "1" * 40
GR00T_REVISION = "2" * 40
FAST_INDEX = "3" * 40
GR00T_INDEX = "4" * 40


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _make_writable(root: Path) -> None:
    if not root.exists():
        return
    for directory, directories, files in os.walk(root):
        os.chmod(directory, 0o755)
        for name in directories:
            path = Path(directory) / name
            if not path.is_symlink():
                os.chmod(path, 0o755)
        for name in files:
            path = Path(directory) / name
            if not path.is_symlink():
                os.chmod(path, 0o644)


class SealedSourceFixture:
    def __init__(self, owner: unittest.TestCase):
        self.base = Path(tempfile.mkdtemp(prefix="rr_sealed_verify_"))
        owner.addCleanup(self.cleanup)
        self.name = (
            f"fast_roborender_{FAST_REVISION[:7]}_{GR00T_REVISION[:7]}_sealed"
        )
        self.root = self.base / self.name
        self.gr00t = self.root / "GR00T-VisualSim2Real"
        self.gr00t.mkdir(parents=True)
        (self.root / ".source_revision").write_text(FAST_REVISION + "\n")
        (self.gr00t / ".source_revision").write_text(GR00T_REVISION + "\n")
        (self.root / "module.py").write_text("VALUE = 7\n", encoding="utf-8")
        script = self.gr00t / "runner.sh"
        script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        script.chmod(0o755)
        self.manifest = {
            "schema": verifier.MANIFEST_SCHEMA,
            "status": "sealed_tracked_hydrated_sources",
            "snapshot_name": self.name,
            "fast_roborender_revision": FAST_REVISION,
            "gr00t_revision": GR00T_REVISION,
            "fast_roborender_index_tree": FAST_INDEX,
            "gr00t_index_tree": GR00T_INDEX,
            "export_method": "stage0-git-blobs-plus-verified-lfs-v2",
            "lfs_file_count": 0,
            "lfs_total_bytes": 0,
            "tracked_only": True,
            "lfs_pointer_oids_verified": True,
            "symlinks_rejected": True,
            "bytecode_rejected": True,
        }
        _write_json(self.root / ".source_snapshot_manifest.json", self.manifest)
        self.inventory_path = self.base / f"{self.name}.inventory.json"
        self.attestation_path = self.base / f"{self.name}.snapshot_attestation.json"
        self.reseal()

    def cleanup(self) -> None:
        _make_writable(self.base)
        shutil.rmtree(self.base, ignore_errors=True)

    def unseal(self) -> None:
        _make_writable(self.root)
        for path in (self.inventory_path, self.attestation_path):
            if path.exists() and not path.is_symlink():
                path.chmod(0o644)

    def reseal(self) -> None:
        self.unseal()
        for path in sorted(self.root.rglob("*"), reverse=True):
            if path.is_dir() and not path.is_symlink():
                path.chmod(0o555)
            elif path.is_file() and not path.is_symlink():
                path.chmod(0o555 if path.name == "runner.sh" else 0o444)
        self.root.chmod(0o555)
        records, fast_tree, gr00t_tree = verifier._tree_records(self.root)
        inventory = {
            "schema": verifier.INVENTORY_SCHEMA,
            "file_count": len(records),
            "total_file_bytes": sum(record["bytes"] for record in records),
            "files": records,
        }
        _write_json(self.inventory_path, inventory)
        self.inventory_sha = hashlib.sha256(self.inventory_path.read_bytes()).hexdigest()
        attestation = {
            "schema": verifier.ATTESTATION_SCHEMA,
            "status": "PASS",
            "snapshot_name": self.name,
            "fast_roborender_revision": FAST_REVISION,
            "gr00t_revision": GR00T_REVISION,
            "fast_roborender_index_tree": FAST_INDEX,
            "gr00t_index_tree": GR00T_INDEX,
            "fast_roborender_source_tree_sha256": fast_tree,
            "gr00t_source_tree_sha256": gr00t_tree,
            "inventory_sha256": self.inventory_sha,
            "deterministic_archive_sha256": "5" * 64,
            "file_count": inventory["file_count"],
            "total_file_bytes": inventory["total_file_bytes"],
        }
        _write_json(self.attestation_path, attestation)
        self.inventory_path.chmod(0o444)
        self.attestation_path.chmod(0o444)


class SealedSourceVerifierTests(unittest.TestCase):
    def test_valid_snapshot_and_cli_report_are_content_bound(self) -> None:
        fixture = SealedSourceFixture(self)
        report = verifier.verify(fixture.root, fixture.inventory_sha)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["inventory_sha256"], fixture.inventory_sha)
        self.assertEqual(report["fast_roborender_revision"], FAST_REVISION)
        self.assertEqual(report["gr00t_revision"], GR00T_REVISION)

        command = [
            sys.executable,
            str(
                Path(__file__).resolve().parents[1]
                / "scripts/round_dagger/verify_sealed_source_snapshot.py"
            ),
            "--root",
            str(fixture.root),
            "--expected-inventory-sha256",
            fixture.inventory_sha,
        ]
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["schema"], verifier.REPORT_SCHEMA)

    def test_inventory_sha_must_be_well_formed_and_exactly_pinned(self) -> None:
        fixture = SealedSourceFixture(self)
        with self.assertRaisesRegex(verifier.VerificationError, "malformed"):
            verifier.verify(fixture.root, "not-a-hash")
        with self.assertRaisesRegex(verifier.VerificationError, "SHA-256 mismatch"):
            verifier.verify(fixture.root, "f" * 64)

    def test_modified_missing_and_extra_source_files_are_rejected(self) -> None:
        for mutation in ("modified", "missing", "extra"):
            with self.subTest(mutation=mutation):
                fixture = SealedSourceFixture(self)
                fixture.root.chmod(0o755)
                module = fixture.root / "module.py"
                if mutation == "modified":
                    module.chmod(0o644)
                    module.write_text("VALUE = 8\n", encoding="utf-8")
                    module.chmod(0o444)
                elif mutation == "missing":
                    module.unlink()
                else:
                    extra = fixture.root / "unexpected.txt"
                    extra.write_text("unexpected\n", encoding="utf-8")
                    extra.chmod(0o444)
                fixture.root.chmod(0o555)
                with self.assertRaisesRegex(verifier.VerificationError, "does not exactly match"):
                    verifier.verify(fixture.root, fixture.inventory_sha)

    def test_writable_or_wrong_mode_file_directory_and_evidence_are_rejected(self) -> None:
        cases = ("file", "directory", "inventory", "attestation", "special_mode")
        for case in cases:
            with self.subTest(case=case):
                fixture = SealedSourceFixture(self)
                if case == "file":
                    (fixture.root / "module.py").chmod(0o644)
                elif case == "directory":
                    fixture.gr00t.chmod(0o755)
                elif case == "inventory":
                    fixture.inventory_path.chmod(0o644)
                elif case == "attestation":
                    fixture.attestation_path.chmod(0o644)
                else:
                    (fixture.root / "module.py").chmod(0o444 | stat.S_ISUID)
                with self.assertRaisesRegex(
                    verifier.VerificationError, "writable|mode must be exactly|mode is invalid"
                ):
                    verifier.verify(fixture.root, fixture.inventory_sha)

    def test_symlink_fifo_and_bytecode_are_rejected_before_inventory_trust(self) -> None:
        cases = ("symlink", "fifo", "pyc", "pycache", "empty")
        for case in cases:
            with self.subTest(case=case):
                fixture = SealedSourceFixture(self)
                fixture.root.chmod(0o755)
                if case == "symlink":
                    (fixture.root / "link").symlink_to("module.py")
                    pattern = "symlink"
                elif case == "fifo":
                    os.mkfifo(fixture.root / "pipe", 0o444)
                    pattern = "special file"
                elif case == "pyc":
                    bytecode = fixture.root / "bad.pyc"
                    bytecode.write_bytes(b"bytecode")
                    bytecode.chmod(0o444)
                    pattern = "bytecode"
                elif case == "pycache":
                    cache = fixture.root / "__pycache__"
                    cache.mkdir(mode=0o555)
                    cache.chmod(0o555)
                    pattern = "bytecode directory"
                else:
                    empty = fixture.root / "untracked-empty"
                    empty.mkdir(mode=0o555)
                    empty.chmod(0o555)
                    pattern = "untracked empty directory"
                fixture.root.chmod(0o555)
                with self.assertRaisesRegex(verifier.VerificationError, pattern):
                    verifier.verify(fixture.root, fixture.inventory_sha)

    def test_manifest_semantics_and_revision_markers_are_checked(self) -> None:
        fixture = SealedSourceFixture(self)
        fixture.unseal()
        fixture.manifest["tracked_only"] = False
        _write_json(fixture.root / ".source_snapshot_manifest.json", fixture.manifest)
        fixture.reseal()
        with self.assertRaisesRegex(verifier.VerificationError, "tracked_only mismatch"):
            verifier.verify(fixture.root, fixture.inventory_sha)

        fixture = SealedSourceFixture(self)
        fixture.unseal()
        (fixture.root / ".source_revision").write_text("9" * 40 + "\n")
        fixture.reseal()
        with self.assertRaisesRegex(verifier.VerificationError, "revision marker mismatch"):
            verifier.verify(fixture.root, fixture.inventory_sha)

    def test_attestation_is_exact_and_bound_to_manifest_inventory_and_tree(self) -> None:
        fixture = SealedSourceFixture(self)
        fixture.attestation_path.chmod(0o644)
        attestation = json.loads(fixture.attestation_path.read_text())
        attestation["fast_roborender_source_tree_sha256"] = "a" * 64
        _write_json(fixture.attestation_path, attestation)
        fixture.attestation_path.chmod(0o444)
        with self.assertRaisesRegex(verifier.VerificationError, "attestation .* mismatch"):
            verifier.verify(fixture.root, fixture.inventory_sha)

        fixture = SealedSourceFixture(self)
        fixture.attestation_path.chmod(0o644)
        attestation = json.loads(fixture.attestation_path.read_text())
        attestation["unexpected"] = True
        _write_json(fixture.attestation_path, attestation)
        fixture.attestation_path.chmod(0o444)
        with self.assertRaisesRegex(verifier.VerificationError, "field set mismatch"):
            verifier.verify(fixture.root, fixture.inventory_sha)

    def test_external_evidence_symlinks_and_duplicate_json_keys_fail_closed(self) -> None:
        fixture = SealedSourceFixture(self)
        fixture.inventory_path.unlink()
        fixture.inventory_path.symlink_to(fixture.attestation_path.name)
        with self.assertRaisesRegex(verifier.VerificationError, "safely open"):
            verifier.verify(fixture.root, fixture.inventory_sha)

        fixture = SealedSourceFixture(self)
        fixture.inventory_path.chmod(0o644)
        malformed = (
            '{"schema":"fast_roborender.hsg_source_inventory.v1",'
            '"schema":"fast_roborender.hsg_source_inventory.v1",'
            '"file_count":0,"total_file_bytes":0,"files":[]}\n'
        ).encode()
        fixture.inventory_path.write_bytes(malformed)
        fixture.inventory_path.chmod(0o444)
        digest = hashlib.sha256(malformed).hexdigest()
        with self.assertRaisesRegex(verifier.VerificationError, "duplicate JSON key"):
            verifier.verify(fixture.root, digest)


if __name__ == "__main__":
    unittest.main()
