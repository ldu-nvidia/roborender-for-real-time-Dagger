from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.round_dagger import lepton_campaign_archive_transfer as campaign
from scripts.round_dagger import lepton_web_terminal_transfer as webtransfer
from scripts.round_dagger import remote_campaign_transfer as tree_transfer


class LeptonCampaignArchiveTransferTest(unittest.TestCase):
    def _projection(self, root: Path) -> None:
        (root / "policies").mkdir(parents=True)
        (root / "replay/nested").mkdir(parents=True)
        (root / "rounds/macro_round_000000/cohorts").mkdir(parents=True)
        (root / "campaign.json").write_text("{}\n", encoding="utf-8")
        (root / "coordinator_state.json").write_text("{}\n", encoding="utf-8")
        (root / "replay/nested/payload.bin").write_bytes(b"payload" * 31)

    def _pack(self, base: Path, name: str = "one") -> tuple[Path, Path, Path, dict]:
        projection = base / "projection"
        if not projection.exists():
            self._projection(projection)
        output = base / name
        output.mkdir()
        archive = output / "campaign.tar.gz"
        inventory = output / "campaign.inventory.json"
        manifest = output / "campaign.manifest.json"
        report = campaign.pack_projection(projection, archive, inventory, manifest)
        return archive, inventory, manifest, report

    def test_pack_is_deterministic_and_binds_existing_inventory_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            first = self._pack(base, "one")
            second = self._pack(base, "two")

            self.assertEqual(first[0].read_bytes(), second[0].read_bytes())
            self.assertEqual(first[1].read_bytes(), second[1].read_bytes())
            self.assertEqual(first[2].read_bytes(), second[2].read_bytes())
            document = json.loads(first[2].read_text(encoding="utf-8"))
            inventory = tree_transfer.load_inventory(first[1])
            self.assertEqual(document["inventory"]["tree_sha256"], inventory["tree_sha256"])
            self.assertEqual(
                document["members"]["files"],
                [record["path"] for record in inventory["files"]],
            )
            self.assertEqual(document["archive"]["format"], "tar+gzip")
            self.assertEqual(document["archive"]["gzip_mtime"], 0)
            self.assertEqual(document["archive"]["compression_level"], 1)
            with gzip.open(first[0], "rb") as handle:
                self.assertTrue(handle.read(512))

    def test_extract_verifies_inventory_then_atomically_publishes_and_adopts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            archive, inventory, manifest, _ = self._pack(base)
            manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
            destination_parent = base / "remote"
            destination_parent.mkdir()
            destination = destination_parent / "campaign"

            first = campaign.extract_publish(
                archive,
                inventory,
                manifest,
                destination,
                expected_manifest_sha256=manifest_sha,
            )
            self.assertEqual(first["publication"], "published")
            self.assertEqual(
                tree_transfer.verify_inventory(destination, inventory)["tree_sha256"],
                first["tree_sha256"],
            )
            self.assertEqual(
                tree_transfer.validate_campaign_projection(destination)["file_count"], 3
            )
            self.assertEqual(
                list(destination_parent.glob(f"{campaign._STAGE_PREFIX}*")), []
            )
            self.assertEqual(
                (destination / "replay/nested/payload.bin").read_bytes(),
                b"payload" * 31,
            )
            self.assertEqual(
                (destination / "replay/nested/payload.bin").stat().st_mode
                & 0o777,
                0o444,
            )
            self.assertEqual((destination / "replay").stat().st_mode & 0o777, 0o755)

            second = campaign.extract_publish(
                archive,
                inventory,
                manifest,
                destination,
                expected_manifest_sha256=manifest_sha,
            )
            self.assertEqual(second["publication"], "adopted")
            self.assertEqual(second["tree_sha256"], first["tree_sha256"])

    def test_wrong_manifest_pin_fails_before_creating_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            archive, inventory, manifest, _ = self._pack(base)
            parent = base / "remote"
            parent.mkdir()
            destination = parent / "campaign"
            with self.assertRaisesRegex(campaign.CampaignArchiveError, "caller-pinned"):
                campaign.extract_publish(
                    archive,
                    inventory,
                    manifest,
                    destination,
                    expected_manifest_sha256="0" * 64,
                )
            self.assertFalse(destination.exists())
            self.assertEqual(list(parent.iterdir()), [])

    def test_link_member_is_rejected_and_private_stage_is_cleaned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            _, inventory_path, manifest_path, _ = self._pack(base)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            inventory = tree_transfer.load_inventory(inventory_path)
            evil_archive = base / "evil.tar.gz"
            with evil_archive.open("wb") as raw:
                with gzip.GzipFile(
                    filename="",
                    mode="wb",
                    fileobj=raw,
                    mtime=0,
                    compresslevel=1,
                ) as compressed:
                    with tarfile.open(
                        fileobj=compressed,
                        mode="w|",
                        format=tarfile.USTAR_FORMAT,
                    ) as archive:
                        for relative in manifest["members"]["directories"]:
                            archive.addfile(campaign._tar_info(relative, directory=True))
                        for index, record in enumerate(inventory["files"]):
                            if index == 0:
                                item = campaign._tar_info(record["path"], directory=False, size=0)
                                item.type = tarfile.SYMTYPE
                                item.linkname = "../escape"
                                archive.addfile(item)
                            else:
                                archive.addfile(
                                    campaign._tar_info(
                                        record["path"],
                                        directory=False,
                                        size=record["size"],
                                    ),
                                    io.BytesIO((base / "projection" / record["path"]).read_bytes()),
                                )
            archive_identity = campaign._regular_identity(evil_archive)
            manifest["archive"]["size"] = archive_identity[0]
            manifest["archive"]["sha256"] = archive_identity[1]
            evil_manifest = base / "evil.manifest.json"
            evil_manifest.write_bytes(campaign._canonical_json(manifest, pretty=True))
            manifest_sha = hashlib.sha256(evil_manifest.read_bytes()).hexdigest()
            parent = base / "remote"
            parent.mkdir()
            destination = parent / "campaign"

            with self.assertRaisesRegex(campaign.CampaignArchiveError, "metadata is unsafe"):
                campaign.extract_publish(
                    evil_archive,
                    inventory_path,
                    evil_manifest,
                    destination,
                    expected_manifest_sha256=manifest_sha,
                )
            self.assertFalse(destination.exists())
            self.assertEqual(list(parent.iterdir()), [])

    def test_existing_different_destination_is_never_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            archive, inventory, manifest, _ = self._pack(base)
            manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
            parent = base / "remote"
            parent.mkdir()
            destination = parent / "campaign"
            campaign.extract_publish(
                archive,
                inventory,
                manifest,
                destination,
                expected_manifest_sha256=manifest_sha,
            )
            changed = destination / "campaign.json"
            # Published extraction trees intentionally contain read-only
            # evidence.  Simulate external tampering explicitly before
            # asserting that adoption still detects the changed bytes.
            changed.chmod(0o644)
            changed.write_text("changed\n", encoding="utf-8")

            with self.assertRaisesRegex(campaign.CampaignArchiveError, "conflicts"):
                campaign.extract_publish(
                    archive,
                    inventory,
                    manifest,
                    destination,
                    expected_manifest_sha256=manifest_sha,
                )
            self.assertEqual(changed.read_text(encoding="utf-8"), "changed\n")

    def test_pack_rejects_mutable_actor_projection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            projection = base / "projection"
            self._projection(projection)
            (projection / "actors/cohort_00").mkdir(parents=True)
            output = base / "output"
            output.mkdir()
            with self.assertRaisesRegex(campaign.CampaignArchiveError, "validation failed"):
                campaign.pack_projection(
                    projection,
                    output / "a.tar.gz",
                    output / "i.json",
                    output / "m.json",
                )

    def test_learner_output_bundle_round_trips_candidate_and_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            output = base / "learner_output"
            (output / "checkpoints").mkdir(parents=True)
            (output / "candidate_round_000000.pt").write_bytes(os.urandom(8192))
            (output / "learner_result.json").write_text(
                '{"status":"PASS"}\n', encoding="utf-8"
            )
            (output / "checkpoints/step_000002.pt").write_bytes(b"checkpoint")
            bundle = base / "bundle"
            bundle.mkdir()
            archive = bundle / "result.tar.gz"
            inventory = bundle / "result.inventory.json"
            manifest = bundle / "result.manifest.json"
            report = campaign.pack_projection(
                output,
                archive,
                inventory,
                manifest,
                tree_kind="learner_output",
            )
            self.assertEqual(report["tree_kind"], "learner_output")
            manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
            destination_parent = base / "rtx"
            destination_parent.mkdir()
            destination = destination_parent / "learner_output"
            published = campaign.extract_publish(
                archive,
                inventory,
                manifest,
                destination,
                expected_manifest_sha256=manifest_sha,
            )
            self.assertEqual(published["tree_kind"], "learner_output")
            self.assertEqual(
                (destination / "candidate_round_000000.pt").read_bytes(),
                (output / "candidate_round_000000.pt").read_bytes(),
            )
            self.assertEqual(
                tree_transfer.verify_inventory(destination, inventory)["tree_sha256"],
                report["inventory"]["tree_sha256"],
            )

    def test_upload_uses_streaming_transport_and_remote_verified_extractor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            archive, inventory, manifest_path, _ = self._pack(base)
            manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            class FakeTransport:
                closed = False

                def close(self):
                    self.closed = True

            class FakeEngine:
                def __init__(self):
                    self.calls = []

                def upload_streaming(self, local, remote, streamer, **kwargs):
                    self.calls.append((local, remote, kwargs))
                    return {"status": "published", "remote": remote}

            class FakeShell:
                def __init__(self, report):
                    self.report = report
                    self.scripts = []

                def run(self, script, **kwargs):
                    self.scripts.append((script, kwargs))
                    return webtransfer.CommandResult(
                        pid=1,
                        worker_pid=2,
                        data=(json.dumps(self.report).encode("utf-8"),),
                    )

            transport = FakeTransport()
            engine = FakeEngine()
            remote_report = {
                "schema": campaign.REPORT_SCHEMA,
                "operation": "extract-publish",
                "status": "PASS",
                "tree_kind": "campaign_projection",
                "manifest_sha256": manifest_sha,
                "archive_sha256": manifest["archive"]["sha256"],
                "inventory_sha256": manifest["inventory"]["sha256"],
                "tree_sha256": manifest["inventory"]["tree_sha256"],
            }
            shell = FakeShell(remote_report)
            arguments = argparse.Namespace(
                archive=str(archive),
                inventory=str(inventory),
                manifest=str(manifest_path),
                manifest_sha256=manifest_sha,
                remote_bundle_root="/workspace/projects/roborender/handoff",
                remote_destination="/workspace/projects/roborender/campaigns/test",
                remote_python="/workspace/projects/roborender/envs/h200/bin/python",
                remote_tool=(
                    "/workspace/projects/roborender/src/sealed/scripts/"
                    "round_dagger/lepton_campaign_archive_transfer.py"
                ),
                deployment="test",
                replica=None,
                heartbeat_seconds=15.0,
                heartbeat_timeout_seconds=45.0,
                command_timeout_seconds=300.0,
                extract_timeout_seconds=900.0,
                chunk_bytes=128 * 1024,
            )
            with (
                mock.patch.object(
                    campaign.webtransfer, "discover_connection", return_value=object()
                ),
                mock.patch.object(
                    campaign.webtransfer,
                    "NodeWebTerminalTransport",
                    return_value=transport,
                ),
                mock.patch.object(campaign.webtransfer, "RemoteShell", return_value=shell),
                mock.patch.object(
                    campaign.webtransfer, "ShellRemoteFileStore", return_value=object()
                ),
                mock.patch.object(campaign.webtransfer, "TransferEngine", return_value=engine),
                mock.patch.object(campaign.webtransfer, "StreamingUploader", return_value=object()),
            ):
                report = campaign.upload_publish(arguments)

            self.assertEqual(report["status"], "PASS")
            self.assertTrue(transport.closed)
            self.assertEqual(
                [Path(call[0]).name for call in engine.calls],
                [archive.name, inventory.name, manifest_path.name],
            )
            self.assertTrue(
                all(
                    call[2] == {"adopt_existing": True, "mode": 0o444}
                    for call in engine.calls
                )
            )
            self.assertNotIn("--delete", json.dumps(engine.calls))
            self.assertIn("extract-publish", shell.scripts[0][0])
            self.assertIn(manifest_sha, shell.scripts[0][0])
            self.assertIn("PYTHONDONTWRITEBYTECODE=1", shell.scripts[0][0])

    def test_remote_bundle_paths_are_content_addressed_and_confined(self) -> None:
        digest = "a" * 64
        paths = campaign._remote_bundle_paths(
            "/workspace/projects/roborender/handoff", digest
        )
        self.assertEqual(len(set(paths.values())), 3)
        self.assertTrue(all(digest[:24] in path for path in paths.values()))
        with self.assertRaises(webtransfer.TransferError):
            campaign._remote_bundle_paths("/tmp/outside", digest)

    def test_download_publish_streams_pinned_result_then_atomically_materializes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            learner = base / "remote_learner_output"
            learner.mkdir()
            (learner / "candidate_round_000000.pt").write_bytes(b"candidate" * 1000)
            (learner / "learner_result.json").write_text("{}\n", encoding="utf-8")
            remote_bundle = base / "remote_bundle"
            remote_bundle.mkdir()
            archive = remote_bundle / "result.tar.gz"
            inventory = remote_bundle / "result.inventory.json"
            manifest_path = remote_bundle / "result.manifest.json"
            campaign.pack_projection(
                learner,
                archive,
                inventory,
                manifest_path,
                tree_kind="learner_output",
            )
            manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            remote_paths = {
                "/workspace/projects/roborender/results/result.tar.gz": archive,
                "/workspace/projects/roborender/results/result.inventory.json": inventory,
                "/workspace/projects/roborender/results/result.manifest.json": manifest_path,
            }

            class FakeTransport:
                closed = False

                def close(self):
                    self.closed = True

            class FakeEngine:
                def __init__(self):
                    self.calls = []

                def download_streaming(self, remote, local, downloader, **kwargs):
                    self.calls.append((remote, str(local), kwargs))
                    Path(local).write_bytes(remote_paths[remote].read_bytes())
                    return {"status": "published", "remote": remote}

            transport = FakeTransport()
            engine = FakeEngine()
            local_bundle = base / "local_bundle"
            local_bundle.mkdir()
            destination_parent = base / "rtx_result"
            destination_parent.mkdir()
            destination = destination_parent / "learner_output"
            arguments = argparse.Namespace(
                remote_archive="/workspace/projects/roborender/results/result.tar.gz",
                remote_inventory="/workspace/projects/roborender/results/result.inventory.json",
                remote_manifest="/workspace/projects/roborender/results/result.manifest.json",
                manifest_sha256=manifest_sha,
                local_bundle_root=str(local_bundle),
                destination=str(destination),
                deployment="test",
                replica=None,
                sdk_python=None,
                heartbeat_seconds=15.0,
                heartbeat_timeout_seconds=45.0,
                command_timeout_seconds=900.0,
                chunk_bytes=128 * 1024,
            )
            with (
                mock.patch.object(
                    campaign.webtransfer, "discover_connection", return_value=object()
                ),
                mock.patch.object(
                    campaign.webtransfer,
                    "NodeWebTerminalTransport",
                    return_value=transport,
                ),
                mock.patch.object(campaign.webtransfer, "RemoteShell", return_value=object()),
                mock.patch.object(
                    campaign.webtransfer, "ShellRemoteFileStore", return_value=object()
                ),
                mock.patch.object(campaign.webtransfer, "TransferEngine", return_value=engine),
                mock.patch.object(
                    campaign.webtransfer, "StreamingDownloader", return_value=object()
                ),
            ):
                report = campaign.download_publish(arguments)

            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["publication"]["tree_kind"], "learner_output")
            self.assertEqual(
                [call[0] for call in engine.calls],
                [arguments.remote_manifest, arguments.remote_inventory, arguments.remote_archive],
            )
            self.assertTrue(
                all(call[2] == {"adopt_existing": True, "mode": 0o444} for call in engine.calls)
            )
            self.assertEqual(
                (destination / "candidate_round_000000.pt").read_bytes(),
                b"candidate" * 1000,
            )
            self.assertTrue(transport.closed)


if __name__ == "__main__":
    unittest.main()
