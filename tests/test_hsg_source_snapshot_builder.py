from pathlib import Path
import os
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class HsgSourceSnapshotBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = (ROOT / "scripts/scaling/stage_hsg_source_snapshot.sh").read_text()

    def test_payload_uses_committed_blobs_and_verified_lfs_objects(self):
        self.assertIn('git -C "$repo" ls-files --stage -z', self.script)
        self.assertIn('git -C "$repo" cat-file blob "$object"', self.script)
        self.assertIn('pointer_oid_line', self.script)
        self.assertIn('sha256sum "$source"', self.script)
        self.assertIn('LFS working file size does not match committed pointer', self.script)
        self.assertIn('assert_repo_state "$FAST_REPO"', self.script)
        self.assertIn('assert_repo_state "$GR00T_REPO"', self.script)
        self.assertIn("rsync -a --checksum --delete --no-owner --no-group", self.script)

    def test_untracked_source_is_fail_closed_except_explicit_allowlist(self):
        self.assertIn('"ls-files", "--others", "--exclude-standard", "-z"', self.script)
        self.assertIn("ALLOW_UNTRACKED_FAST", self.script)
        self.assertIn("ALLOW_UNTRACKED_GR00T", self.script)
        self.assertIn("nonignored untracked source is not explicitly allowed", self.script)

    def test_snapshot_rejects_unsafe_entries_and_bytecode(self):
        for token in ("-type l", "-type b", "-type c", "-type p", "-type s"):
            self.assertIn(token, self.script)
        self.assertIn("__pycache__", self.script)
        self.assertIn("'*.pyc'", self.script)
        self.assertIn("'*.pyo'", self.script)
        self.assertIn('find "$LOCAL_TEMPORARY" -type f -perm /111', self.script)
        self.assertIn('find "$LOCAL_TEMPORARY" -type d -exec chmod 0555', self.script)

    def test_local_and_remote_content_digests_are_required(self):
        self.assertGreaterEqual(self.script.count("tree-hash"), 4)
        self.assertIn("deterministic_archive_sha256", self.script)
        self.assertIn('[ "$archive_actual" = "$archive_expected" ]', self.script)
        self.assertIn('[ -z "$(find "$temporary" -perm /222', self.script)
        self.assertIn("inventory_sha256", self.script)
        self.assertIn("REMOTE_VERIFY", self.script)

    def test_remote_transport_is_stream_compressed_and_fail_closed(self):
        self.assertIn("ssh stat awk wc cmp dd grep unlink zstd", self.script)
        self.assertIn("[ -x /usr/bin/zstd ]", self.script)
        self.assertIn("zstd -q -1 -T0 -c", self.script)
        self.assertIn("/usr/bin/zstd -q -d -c | tar -xf -", self.script)
        self.assertIn("REMOTE_SEED_ROOT", self.script)
        self.assertIn('cp -a --reflink=auto "$canonical_seed/."', self.script)
        self.assertIn(
            'case "$canonical_seed" in "$output_root"/fast_roborender_*_sealed',
            self.script,
        )

    def test_publication_is_atomic_and_never_overwrites(self):
        self.assertIn('[ ! -e "$LOCAL_PUBLICATION" ]', self.script)
        self.assertIn('for path in "$destination" "$temporary"', self.script)
        self.assertIn('mv "$LOCAL_PUBLICATION_TEMP" "$LOCAL_PUBLICATION"', self.script)
        self.assertIn('chmod 0555 "$LOCAL_PUBLICATION_TEMP"', self.script)
        self.assertIn('mv "$temporary" "$destination"', self.script)
        self.assertLess(
            self.script.index('mv "$attestation_temporary" "$attestation"'),
            self.script.index('mv "$temporary" "$destination"'),
        )
        self.assertIn("LOCAL_DESTINATION_OWNED", self.script)
        self.assertIn("PUBLISH_COMPLETE=1", self.script)

    def test_preexisting_snapshot_survives_failed_create(self):
        with tempfile.TemporaryDirectory(prefix="rr_snapshot_test_") as temporary:
            root = Path(temporary)
            fast = root / "fast"
            gr00t = fast / "GR00T-VisualSim2Real"
            output = root / "output"
            gr00t.mkdir(parents=True)
            for repo in (fast, gr00t):
                subprocess.run(["git", "init", "-q", str(repo)], check=True)
                subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
                subprocess.run(["git", "-C", str(repo), "config", "user.name", "test"], check=True)
            (gr00t / "nested.txt").write_text("nested\n")
            subprocess.run(["git", "-C", str(gr00t), "add", "nested.txt"], check=True)
            subprocess.run(["git", "-C", str(gr00t), "commit", "-q", "-m", "nested"], check=True)
            (fast / ".gitignore").write_text("GR00T-VisualSim2Real/\n")
            (fast / "top.txt").write_text("top\n")
            subprocess.run(["git", "-C", str(fast), "add", ".gitignore", "top.txt"], check=True)
            subprocess.run(["git", "-C", str(fast), "commit", "-q", "-m", "top"], check=True)
            fast_sha = subprocess.check_output(["git", "-C", str(fast), "rev-parse", "HEAD"], text=True).strip()
            gr00t_sha = subprocess.check_output(["git", "-C", str(gr00t), "rev-parse", "HEAD"], text=True).strip()
            name = f"fast_roborender_{fast_sha[:7]}_{gr00t_sha[:7]}_sealed"
            publication = output / f"{name}.publication"
            destination = publication / name
            destination.mkdir(parents=True)
            sentinel = destination / "sentinel"
            sentinel.write_text("do-not-delete\n")
            attestation = publication / f"{name}.snapshot_attestation.json"
            inventory = publication / f"{name}.inventory.json"
            attestation.write_text("existing-attestation\n")
            inventory.write_text("existing-inventory\n")
            environment = os.environ.copy()
            environment.update({
                "FAST_REPO": str(fast),
                "GR00T_REPO": str(gr00t),
                "LOCAL_OUTPUT_ROOT": str(output),
                "STAGE_ROOT": "/tmp/unused_hsg_stage",
                "REMOTE_OUTPUT_ROOT": "/tmp/unused_hsg_stage/src_snapshots",
                "LOCAL_ONLY": "1",
            })
            result = subprocess.run(
                ["bash", str(ROOT / "scripts/scaling/stage_hsg_source_snapshot.sh")],
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(sentinel.read_text(), "do-not-delete\n")
            self.assertEqual(attestation.read_text(), "existing-attestation\n")
            self.assertEqual(inventory.read_text(), "existing-inventory\n")
            lock = output / f".{name}.publish.lock"
            self.assertFalse(lock.exists(), "this invocation must release only its own lock")

            shutil.rmtree(publication)
            lock.mkdir()
            lock_sentinel = lock / "owner"
            lock_sentinel.write_text("other-publisher\n")
            locked = subprocess.run(
                ["bash", str(ROOT / "scripts/scaling/stage_hsg_source_snapshot.sh")],
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertNotEqual(locked.returncode, 0)
            self.assertEqual(lock_sentinel.read_text(), "other-publisher\n")

    def test_small_multiline_non_lfs_blob_is_not_parsed_as_pointer(self):
        with tempfile.TemporaryDirectory(prefix="rr_snapshot_non_lfs_") as temporary:
            root = Path(temporary)
            fast = root / "fast"
            gr00t = fast / "GR00T-VisualSim2Real"
            output = root / "output"
            gr00t.mkdir(parents=True)
            for repo in (fast, gr00t):
                subprocess.run(["git", "init", "-q", str(repo)], check=True)
                subprocess.run(
                    ["git", "-C", str(repo), "config", "user.email", "test@example.com"],
                    check=True,
                )
                subprocess.run(
                    ["git", "-C", str(repo), "config", "user.name", "test"],
                    check=True,
                )
            (gr00t / "nested.txt").write_text("nested\n")
            subprocess.run(
                ["git", "-C", str(gr00t), "add", "nested.txt"], check=True
            )
            subprocess.run(
                ["git", "-C", str(gr00t), "commit", "-q", "-m", "nested"],
                check=True,
            )
            normal_blob = "one\ntwo\nthree\nfour\nfive\n"
            (fast / ".gitignore").write_text(
                "GR00T-VisualSim2Real/\n__pycache__/\n*.pyc\n*.pyo\n"
            )
            (fast / "normal.txt").write_text(normal_blob)
            gate = fast / "scripts/scaling/hsg_gb200_smoke_gate.py"
            gate.parent.mkdir(parents=True)
            shutil.copy2(
                ROOT / "scripts/scaling/hsg_gb200_smoke_gate.py", gate
            )
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(fast),
                    "add",
                    ".gitignore",
                    "normal.txt",
                    "scripts/scaling/hsg_gb200_smoke_gate.py",
                ],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(fast), "commit", "-q", "-m", "top"],
                check=True,
            )
            environment = os.environ.copy()
            environment.update(
                {
                    "FAST_REPO": str(fast),
                    "GR00T_REPO": str(gr00t),
                    "LOCAL_OUTPUT_ROOT": str(output),
                    "STAGE_ROOT": "/tmp/unused_hsg_stage",
                    "REMOTE_OUTPUT_ROOT": "/tmp/unused_hsg_stage/src_snapshots",
                    "LOCAL_ONLY": "1",
                }
            )
            result = subprocess.run(
                ["bash", str(ROOT / "scripts/scaling/stage_hsg_source_snapshot.sh")],
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            fast_sha = subprocess.check_output(
                ["git", "-C", str(fast), "rev-parse", "HEAD"], text=True
            ).strip()
            gr00t_sha = subprocess.check_output(
                ["git", "-C", str(gr00t), "rev-parse", "HEAD"], text=True
            ).strip()
            snapshot = output / (
                f"fast_roborender_{fast_sha[:7]}_{gr00t_sha[:7]}_sealed.publication"
            ) / (
                f"fast_roborender_{fast_sha[:7]}_{gr00t_sha[:7]}_sealed"
            )
            self.assertEqual((snapshot / "normal.txt").read_text(), normal_blob)
            self.assertEqual(snapshot.parent.stat().st_mode & 0o777, 0o555)
            self.assertTrue(
                (snapshot.parent / f"{snapshot.name}.inventory.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
