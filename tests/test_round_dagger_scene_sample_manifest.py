from __future__ import annotations

import hashlib
import os
import stat
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from scripts.round_dagger import build_scene_sample_manifest as scene


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack("!I", len(payload))
        + kind
        + payload
        + struct.pack("!I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _png(width: int = scene.EXPECTED_WIDTH, height: int = scene.EXPECTED_HEIGHT, value: int = 7) -> bytes:
    ihdr = struct.pack("!IIBBBBB", width, height, 8, 2, 0, 0, 0)
    row = bytes((value, value + 1, value + 2)) * width
    image = zlib.compress(b"".join(b"\x00" + row for _ in range(height)), level=9)
    return scene._PNG_SIGNATURE + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", image) + _chunk(b"IEND", b"")


class SceneSampleManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="scene_samples_")
        self.root = Path(self.temporary.name)
        self.telemetry = self.root / "actor_rank_0000" / "telemetry"
        self.images = self.telemetry / "images"
        self.images.mkdir(parents=True)
        self.destination = self.root / "handoff" / "scene_samples"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write(self, step: int, *, width: int = scene.EXPECTED_WIDTH, value: int | None = None) -> Path:
        path = self.images / f"step_{step:09d}_conditions_generated.png"
        path.write_bytes(_png(width=width, value=step % 200 if value is None else value))
        return path

    def package(self, *, rollout_steps: int = 16):
        return scene.package_scene_samples(
            actor_telemetry_dir=self.telemetry,
            destination=self.destination,
            campaign_id="mini_prod_2",
            actor_rank=0,
            rollout_steps=rollout_steps,
        )

    def test_publishes_hash_pinned_immutable_manifest_and_samples(self) -> None:
        sources = {step: self.write(step) for step in (0, 4, 8, 12)}
        manifest = self.package()

        self.assertEqual(manifest["schema"], scene.SCHEMA)
        self.assertEqual(manifest["campaign_id"], "mini_prod_2")
        self.assertEqual(manifest["actor_rank"], 0)
        self.assertEqual(manifest["rollout_steps"], 16)
        self.assertEqual(manifest["sampled_rollout_steps"], [0, 4, 8, 12])
        self.assertEqual(manifest["view_labels"], list(scene.VIEW_LABELS))
        self.assertEqual(manifest["row_labels"], list(scene.ROW_LABELS))
        self.assertEqual(manifest["sampling_policy"]["max_samples"], 4)
        self.assertEqual(len(manifest["samples"]), 4)
        for record in manifest["samples"]:
            copied = self.destination / record["path"]
            source = sources[record["rollout_step"]]
            expected = hashlib.sha256(source.read_bytes()).hexdigest()
            self.assertEqual(record["sha256"], expected)
            self.assertEqual(hashlib.sha256(copied.read_bytes()).hexdigest(), expected)
            self.assertEqual((record["width"], record["height"]), (1252, 1046))
            self.assertEqual(stat.S_IMODE(copied.stat().st_mode), 0o444)
        self.assertEqual(stat.S_IMODE((self.destination / "manifest.json").stat().st_mode), 0o444)
        core = dict(manifest)
        digest = core.pop("sample_set_sha256")
        self.assertEqual(digest, scene.canonical_json_sha256(core))
        # An identical invocation is an immutable adoption, not a rewrite.
        self.assertEqual(self.package(), manifest)

    def test_more_than_four_candidates_are_endpoint_spaced(self) -> None:
        for step in range(6):
            self.write(step)
        manifest = self.package(rollout_steps=6)
        self.assertEqual(manifest["sampled_rollout_steps"], [0, 2, 3, 5])
        self.assertEqual(manifest["source"]["candidate_count"], 6)
        self.assertEqual(len(manifest["samples"]), 4)

    def test_rejects_wrong_dimensions_before_manifest_publication(self) -> None:
        self.write(0, width=1251)
        with self.assertRaisesRegex(scene.SceneSampleError, "dimensions"):
            self.package()
        self.assertFalse((self.destination / "manifest.json").exists())

    def test_rejects_ambiguous_png_inventory(self) -> None:
        self.write(0)
        (self.images / "preview.png").write_bytes(_png())
        with self.assertRaisesRegex(scene.SceneSampleError, "ambiguous PNG inventory"):
            self.package()

    def test_rejects_symlinks_and_overlapping_destination(self) -> None:
        source = self.write(0)
        source.unlink()
        source.symlink_to(self.root / "outside.png")
        (self.root / "outside.png").write_bytes(_png())
        with self.assertRaisesRegex(scene.SceneSampleError, "non-regular"):
            self.package()

        source.unlink()
        source.write_bytes(_png())
        with self.assertRaisesRegex(scene.SceneSampleError, "disjoint"):
            scene.package_scene_samples(
                actor_telemetry_dir=self.telemetry,
                destination=self.telemetry / "scene_samples",
                campaign_id="mini_prod_2",
                actor_rank=0,
                rollout_steps=1,
            )

    def test_rejects_tampered_destination_and_invalid_identity(self) -> None:
        self.write(0)
        manifest = self.package(rollout_steps=1)
        copied = self.destination / manifest["samples"][0]["path"]
        os.chmod(copied, 0o644)
        with self.assertRaisesRegex(scene.SceneSampleError, "writable"):
            self.package(rollout_steps=1)
        with self.assertRaisesRegex(scene.SceneSampleError, "campaign_id"):
            scene.package_scene_samples(
                actor_telemetry_dir=self.telemetry,
                destination=self.root / "other" / "scene_samples",
                campaign_id="../bad",
                actor_rank=0,
                rollout_steps=1,
            )


if __name__ == "__main__":
    unittest.main()
