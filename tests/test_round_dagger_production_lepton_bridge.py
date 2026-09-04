from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from unittest import mock

from scripts.round_dagger import production_lepton_learner_bridge as bridge
from scripts.round_dagger import remote_campaign_transfer


ROOT = Path(__file__).resolve().parents[1]
SDK_PYTHON = Path("/mnt/data2/ldu/roborender_lepton_control/env/bin/python")
IDENTITY = "c" * 64


def _paths(root: Path) -> bridge.BridgePaths:
    output = root / "campaign/rounds/macro_round_000000/learner"
    return bridge.BridgePaths(
        local_handoff=root / "handoff",
        local_projection=root / "handoff/campaign_input",
        local_archive=root / "handoff/campaign_input.tar.gz",
        local_inventory=root / "handoff/campaign_input.inventory.json",
        local_manifest=root / "handoff/campaign_input.manifest.json",
        local_request=root / "handoff/request.json",
        local_lifecycle=root / "handoff/lifecycle.json",
        local_marker=root / "handoff/marker.json",
        local_checksum=root / "handoff/marker.json.sha256",
        local_return_bundle=root / "handoff/returned_bundle",
        local_output=output,
        local_result=output / "learner_result_round_000000.json",
        remote_bundle_root=PurePosixPath(
            f"/workspace/projects/roborender/handoff/{IDENTITY}/campaign_bundle"
        ),
        remote_campaign=PurePosixPath(
            f"/workspace/projects/roborender/campaigns/test/"
            f"macro_round_000000/identity_{IDENTITY}/campaign_input"
        ),
        remote_output=PurePosixPath(
            f"/workspace/projects/roborender/campaigns/test/"
            f"macro_round_000000/identity_{IDENTITY}/campaign_input/"
            "rounds/macro_round_000000/learner"
        ),
        remote_request=PurePosixPath(
            f"/workspace/projects/roborender/handoff/{IDENTITY}/request.json"
        ),
    )


def _contract() -> SimpleNamespace:
    return SimpleNamespace(
        project_root=PurePosixPath("/workspace/projects/roborender"),
        python=PurePosixPath(
            "/workspace/projects/roborender/envs/h200_learner/bin/python"
        ),
        source_root=PurePosixPath(
            "/workspace/projects/roborender/src/fast_roborender"
        ),
        document={"orchestration": {"control_deployment": "control-deployment"}},
    )


def _retirement_fixture(root: Path):
    campaign_id = "retirement-test"
    identity = IDENTITY
    campaign = root / "campaign"
    campaign.mkdir()
    CampaignProjectionTests()._campaign(campaign)
    handoff = (
        root
        / "handoff-root/lepton_job_v1"
        / campaign_id
        / "macro_round_000000"
        / f"identity_{identity}"
    )
    handoff.mkdir(parents=True)
    output = campaign / "rounds/macro_round_000000/learner"
    paths = bridge.BridgePaths(
        local_handoff=handoff,
        local_projection=handoff / "campaign_input",
        local_archive=handoff / "campaign_input.tar.gz",
        local_inventory=handoff / "campaign_input.inventory.json",
        local_manifest=handoff / "campaign_input.manifest.json",
        local_request=handoff / "lepton_learner_request.json",
        local_lifecycle=handoff / "exclusive_lifecycle.json",
        local_marker=handoff / "remote_learner_success.json",
        local_checksum=handoff / "remote_learner_success.json.sha256",
        local_return_bundle=handoff / "returned_bundle",
        local_output=output,
        local_result=output / "learner_result_round_000000.json",
        remote_bundle_root=PurePosixPath("/workspace/projects/roborender/handoff/x"),
        remote_campaign=PurePosixPath("/workspace/projects/roborender/campaigns/x"),
        remote_output=PurePosixPath("/workspace/projects/roborender/campaigns/x/output"),
        remote_request=PurePosixPath("/workspace/projects/roborender/handoff/x/request"),
    )
    bridge.build_campaign_projection(campaign, paths.local_projection)
    pack = bridge.archive_transfer.pack_projection(
        paths.local_projection,
        paths.local_archive,
        paths.local_inventory,
        paths.local_manifest,
    )
    manifest = pack["manifest"]
    upload_report = {
        "schema": bridge.archive_transfer.REPORT_SCHEMA,
        "operation": "upload-publish",
        "status": "PASS",
        "manifest_sha256": manifest["sha256"],
        "publication": {
            "manifest_sha256": manifest["sha256"],
            "archive_sha256": pack["archive"]["sha256"],
            "inventory_sha256": pack["inventory"]["sha256"],
            "tree_sha256": pack["inventory"]["tree_sha256"],
        },
    }
    output.mkdir(parents=True)
    paths.local_result.write_bytes(b"validated-learner-result")
    result_sha = hashlib.sha256(paths.local_result.read_bytes()).hexdigest()
    evidence = {
        "status": "PASS",
        "campaign_id": campaign_id,
        "macro_round": 0,
        "learner_identity_sha256": identity,
        "result_sha256": result_sha,
        "learner_output_manifest_sha256": "8" * 64,
        "learner_output_tree_sha256": "9" * 64,
    }
    paths.local_marker.write_text(
        json.dumps(evidence, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths.local_marker.chmod(0o444)
    marker_sha = hashlib.sha256(paths.local_marker.read_bytes()).hexdigest()
    paths.local_checksum.write_text(
        f"{marker_sha}  lepton_learner_success.json\n",
        encoding="ascii",
    )
    paths.local_checksum.chmod(0o444)
    paths.local_request.write_text("{}\n", encoding="utf-8")
    paths.local_request.chmod(0o444)
    paths.local_return_bundle.mkdir()
    (paths.local_return_bundle / "audit.txt").write_text("retained\n")
    result = {
        "manifest_path": str(paths.local_result),
        "checkpoint_sha256": "7" * 64,
        "optimizer_steps": 1200,
    }
    environment = {
        "ROUND_DAGGER_CAMPAIGN_ROOT": str(campaign),
        "ROUND_DAGGER_CAMPAIGN_ID": campaign_id,
        "ROUND_DAGGER_MACRO_ROUND": "0",
        "ROUND_DAGGER_LEARNER_IDENTITY_SHA256": identity,
    }
    return paths, upload_report, evidence, result, environment


class CampaignProjectionTests(unittest.TestCase):
    def _campaign(self, root: Path) -> None:
        (root / "policies").mkdir(parents=True)
        (root / "replay").mkdir()
        (root / "rounds/macro_round_000000/orchestration_run").mkdir(
            parents=True
        )
        (root / "rounds/macro_round_000000/input").mkdir()
        (root / "actors/cohort_00").mkdir(parents=True)
        (root / "operator").mkdir()
        (root / "campaign.json").write_text("{}\n", encoding="utf-8")
        (root / "coordinator_state.json").write_text("{}\n", encoding="utf-8")
        (root / "policies/base.pt").write_bytes(b"policy")
        (root / "replay/snapshot.json").write_text("{}\n", encoding="utf-8")
        (root / "rounds/macro_round_000000/input/item").write_bytes(b"input")
        (root / "rounds/macro_round_000000/orchestration_run/log").write_bytes(
            b"mutable"
        )
        (root / "actors/cohort_00/rollout.npz").write_bytes(b"large")
        (root / "operator/status.json").write_text("{}\n", encoding="utf-8")
        (root / ".coordinator_state.lock").write_bytes(b"lock")

    def test_projection_excludes_all_mutable_producer_state_and_is_adoptable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            campaign = base / "campaign"
            campaign.mkdir()
            self._campaign(campaign)
            handoff = base / "handoff"
            handoff.mkdir()
            projection = handoff / "campaign_input"

            self.assertEqual(
                bridge.build_campaign_projection(campaign, projection), "published"
            )
            self.assertEqual(
                bridge.build_campaign_projection(campaign, projection), "adopted"
            )
            self.assertFalse((projection / "actors").exists())
            self.assertFalse((projection / "operator").exists())
            self.assertFalse((projection / ".coordinator_state.lock").exists())
            self.assertFalse(
                (
                    projection
                    / "rounds/macro_round_000000/orchestration_run"
                ).exists()
            )
            self.assertEqual(
                remote_campaign_transfer.validate_campaign_projection(projection)[
                    "file_count"
                ],
                5,
            )
            self.assertEqual(
                (projection / "policies/base.pt").stat().st_mode & 0o222, 0
            )

    def test_projection_rejects_included_symlink_but_ignores_excluded_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            campaign = base / "campaign"
            campaign.mkdir()
            self._campaign(campaign)
            (campaign / "actors/ignored-link").symlink_to("/tmp")
            (campaign / "policies/unsafe").symlink_to("/tmp")
            handoff = base / "handoff"
            handoff.mkdir()
            with self.assertRaisesRegex(
                bridge.ProductionLeptonBridgeError, "unsafe"
            ):
                bridge.build_campaign_projection(
                    campaign, handoff / "campaign_input"
                )

    def test_projection_must_not_be_nested_in_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            campaign = Path(temporary) / "campaign"
            campaign.mkdir()
            self._campaign(campaign)
            with self.assertRaisesRegex(
                bridge.ProductionLeptonBridgeError, "outside"
            ):
                bridge.build_campaign_projection(
                    campaign, campaign / "handoff/campaign_input"
                )

    def test_projection_excludes_exact_archive_extraction_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            campaign = base / "campaign"
            campaign.mkdir()
            self._campaign(campaign)
            private = (
                campaign
                / "rounds/macro_round_000000"
                / f".incoming-inventory-tree-{'a' * 16}-{'b' * 24}"
            )
            private.mkdir()
            (private / "partial-candidate.pt").write_bytes(b"partial")
            handoff = base / "handoff"
            handoff.mkdir()
            projection = handoff / "campaign_input"
            self.assertEqual(
                bridge.build_campaign_projection(campaign, projection), "published"
            )
            self.assertFalse(
                (
                    projection
                    / "rounds/macro_round_000000"
                    / private.name
                ).exists()
            )

    def test_projection_rejects_malformed_archive_extraction_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            campaign = base / "campaign"
            campaign.mkdir()
            self._campaign(campaign)
            private = (
                campaign
                / "rounds/macro_round_000000"
                / ".incoming-inventory-tree-not-destination-bound"
            )
            private.mkdir()
            handoff = base / "handoff"
            handoff.mkdir()
            with self.assertRaisesRegex(
                bridge.ProductionLeptonBridgeError,
                "malformed private extraction stage",
            ):
                bridge.build_campaign_projection(
                    campaign, handoff / "campaign_input"
                )


class ProductionLeptonBridgeTests(unittest.TestCase):
    def test_fetch_rejects_noncanonical_remote_marker_before_download(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _paths(Path(temporary))
            request = SimpleNamespace(
                paths={
                    "success_marker": paths.remote_output / "wrong.json",
                    "success_checksum": paths.remote_output / "wrong.json.sha256",
                }
            )
            with mock.patch.object(bridge, "_download_regular") as download:
                with self.assertRaisesRegex(
                    bridge.ProductionLeptonBridgeError,
                    "path is non-canonical",
                ):
                    bridge._fetch_and_validate_marker(
                        _contract(),
                        request,
                        paths,
                        sdk_python=SDK_PYTHON,
                    )
                download.assert_not_called()

    def test_transferred_checksum_preserves_remote_marker_basename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = _paths(Path(temporary))
            paths.local_handoff.mkdir(parents=True)
            paths.local_marker.write_text("{}\n", encoding="utf-8")
            paths.local_marker.chmod(0o444)
            marker_sha = hashlib.sha256(
                paths.local_marker.read_bytes()
            ).hexdigest()
            paths.local_checksum.write_text(
                f"{marker_sha}  lepton_learner_success.json\n",
                encoding="ascii",
            )
            paths.local_checksum.chmod(0o444)
            self.assertEqual(
                bridge._verify_transferred_completion_checksum(paths),
                marker_sha,
            )

            paths.local_checksum.chmod(0o644)
            paths.local_checksum.write_text(
                f"{marker_sha}  {paths.local_marker.name}\n",
                encoding="ascii",
            )
            paths.local_checksum.chmod(0o444)
            with self.assertRaisesRegex(
                bridge.ProductionLeptonBridgeError,
                "does not bind the remote marker",
            ):
                bridge._verify_transferred_completion_checksum(paths)

    def test_projection_upload_sets_explicit_extraction_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = root / "campaign"
            campaign.mkdir()
            paths = _paths(root)
            paths.local_handoff.mkdir(parents=True)
            manifest_sha256 = "a" * 64
            upload_report = {
                "schema": bridge.archive_transfer.REPORT_SCHEMA,
                "operation": "upload-publish",
                "status": "PASS",
                "manifest_sha256": manifest_sha256,
            }
            with mock.patch.dict(
                os.environ,
                {"ROUND_DAGGER_CAMPAIGN_ROOT": str(campaign)},
                clear=True,
            ), mock.patch.object(
                bridge, "build_campaign_projection", return_value="published"
            ), mock.patch.object(
                bridge, "validate_projected_inputs"
            ), mock.patch.object(
                bridge, "validate_projected_replay_snapshots"
            ), mock.patch.object(
                bridge.archive_transfer,
                "pack_projection",
                return_value={"manifest": {"sha256": manifest_sha256}},
            ), mock.patch.object(
                bridge, "_run_json", return_value=upload_report
            ) as run:
                status, observed = bridge._upload_projection(
                    _contract(), paths, sdk_python=SDK_PYTHON
                )
            self.assertEqual(status, "published")
            self.assertEqual(observed, upload_report)
            command = run.call_args.args[0]
            self.assertEqual(
                command[command.index("--extract-timeout-seconds") + 1], "3600"
            )

    def test_main_prunes_sigkill_projection_stage_before_request_or_remote_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = _paths(root)
            paths.local_handoff.mkdir(parents=True)
            stage = paths.local_handoff / ".incoming-campaign-deadbeef"
            nested = stage / "replay/cohort"
            nested.mkdir(parents=True)
            (nested / "partial.npz").write_bytes(b"sigkill-partial-copy")
            (nested / "partial.npz").chmod(0o444)
            nested.chmod(0o555)
            nested.parent.chmod(0o555)
            stage.chmod(0o555)
            archive_incoming = paths.local_handoff / (
                f".{paths.local_archive.name}.{'a' * 32}.incoming"
            )
            archive_incoming.write_bytes(b"sigkill-partial-archive")
            inventory_incoming = paths.local_handoff / (
                f".{paths.local_inventory.name}.abc123_4.incoming"
            )
            inventory_incoming.write_bytes(b"sigkill-partial-inventory")
            manifest_incoming = paths.local_handoff / (
                f".{paths.local_manifest.name}.def567_8.incoming"
            )
            manifest_incoming.write_bytes(b"sigkill-partial-manifest")
            handoff_download = paths.local_handoff / f".incoming.{'b' * 32}"
            handoff_download.write_bytes(b"sigkill-partial-marker")
            paths.local_return_bundle.mkdir()
            result_download = (
                paths.local_return_bundle / f".incoming.{'c' * 32}"
            )
            result_download.write_bytes(b"sigkill-partial-result-archive")
            paths.local_output.parent.mkdir(parents=True)
            destination_id = hashlib.sha256(
                str(paths.local_output).encode("utf-8")
            ).hexdigest()[:16]
            output_stage = paths.local_output.parent / (
                f".incoming-inventory-tree-{destination_id}-{'d' * 24}"
            )
            (output_stage / "partial").mkdir(parents=True)
            (output_stage / "partial/candidate.pt").write_bytes(
                b"sigkill-partial-candidate"
            )
            backend = (
                root / "contract.json",
                _contract(),
                SDK_PYTHON,
                Path("/mnt/data1/ldu/lepton_pod_exec.py"),
            )
            with mock.patch.object(
                bridge, "_load_backend", return_value=backend
            ), mock.patch.object(
                bridge, "derive_paths", return_value=paths
            ), mock.patch.object(
                bridge,
                "_materialize_request",
                side_effect=bridge.ProductionLeptonBridgeError("stop after prune"),
            ) as materialize, mock.patch.object(
                bridge, "_ensure_remote_directories"
            ) as ensure_remote, mock.patch.object(
                bridge, "_upload_projection"
            ) as upload_projection, self.assertRaisesRegex(
                bridge.ProductionLeptonBridgeError, "stop after prune"
            ):
                bridge.main()
            self.assertFalse(stage.exists())
            self.assertFalse(archive_incoming.exists())
            self.assertFalse(inventory_incoming.exists())
            self.assertFalse(manifest_incoming.exists())
            self.assertFalse(handoff_download.exists())
            self.assertFalse(result_download.exists())
            self.assertFalse(output_stage.exists())
            materialize.assert_called_once_with(backend[1], paths)
            ensure_remote.assert_not_called()
            upload_projection.assert_not_called()

    def test_main_rejects_unsafe_orphan_file_before_mutating_any_candidate(self):
        for unsafe_kind in (
            "archive_fifo",
            "handoff_download_symlink",
            "result_download_fifo",
            "output_stage_symlink",
        ):
            with self.subTest(unsafe_kind=unsafe_kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                paths = _paths(root)
                paths.local_handoff.mkdir(parents=True)
                safe_stage = paths.local_handoff / ".incoming-campaign-aaaaaaaa"
                safe_stage.mkdir()
                (safe_stage / "partial").write_bytes(b"safe-but-stale")
                safe_archive = paths.local_handoff / (
                    f".{paths.local_archive.name}.{'1' * 32}.incoming"
                )
                safe_archive.write_bytes(b"safe-archive-but-stale")
                paths.local_return_bundle.mkdir()
                paths.local_output.parent.mkdir(parents=True)
                if unsafe_kind == "archive_fifo":
                    os.mkfifo(
                        paths.local_handoff
                        / f".{paths.local_archive.name}.{'2' * 32}.incoming"
                    )
                elif unsafe_kind == "handoff_download_symlink":
                    (
                        paths.local_handoff / f".incoming.{'3' * 32}"
                    ).symlink_to(root)
                elif unsafe_kind == "result_download_fifo":
                    os.mkfifo(
                        paths.local_return_bundle / f".incoming.{'4' * 32}"
                    )
                else:
                    destination_id = hashlib.sha256(
                        str(paths.local_output).encode("utf-8")
                    ).hexdigest()[:16]
                    (
                        paths.local_output.parent
                        / f".incoming-inventory-tree-{destination_id}-{'5' * 24}"
                    ).symlink_to(root)
                backend = (
                    root / "contract.json",
                    _contract(),
                    SDK_PYTHON,
                    Path("/mnt/data1/ldu/lepton_pod_exec.py"),
                )
                with mock.patch.object(
                    bridge, "_load_backend", return_value=backend
                ), mock.patch.object(
                    bridge, "derive_paths", return_value=paths
                ), mock.patch.object(
                    bridge, "_materialize_request"
                ) as materialize, mock.patch.object(
                    bridge, "_ensure_remote_directories"
                ) as ensure_remote, mock.patch.object(
                    bridge, "_upload_projection"
                ) as upload_projection, self.assertRaisesRegex(
                    bridge.ProductionLeptonBridgeError,
                    "no-follow regular file|extraction stage is unsafe",
                ):
                    bridge.main()
                self.assertTrue(safe_stage.is_dir())
                self.assertTrue((safe_stage / "partial").is_file())
                self.assertTrue(safe_archive.is_file())
                materialize.assert_not_called()
                ensure_remote.assert_not_called()
                upload_projection.assert_not_called()

    def test_main_rejects_malformed_incoming_name_before_mutation(self):
        for malformed in (
            ".campaign_input.tar.gz.not-a-digest.incoming",
            ".campaign_input.inventory.json.not-a-name.incoming",
            ".campaign_input.manifest.json.not-a-name.incoming",
            ".incoming.not-a-digest",
        ):
            with self.subTest(malformed=malformed), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                paths = _paths(root)
                paths.local_handoff.mkdir(parents=True)
                safe = paths.local_handoff / ".incoming-campaign-aaaaaaaa"
                safe.mkdir()
                (safe / "partial").write_bytes(b"safe-but-stale")
                (paths.local_handoff / malformed).write_bytes(b"ambiguous")
                backend = (
                    root / "contract.json",
                    _contract(),
                    SDK_PYTHON,
                    Path("/mnt/data1/ldu/lepton_pod_exec.py"),
                )
                with mock.patch.object(
                    bridge, "_load_backend", return_value=backend
                ), mock.patch.object(
                    bridge, "derive_paths", return_value=paths
                ), mock.patch.object(
                    bridge, "_materialize_request"
                ) as materialize, self.assertRaisesRegex(
                    bridge.ProductionLeptonBridgeError, "malformed"
                ):
                    bridge.main()
                self.assertTrue(safe.is_dir())
                self.assertTrue((safe / "partial").is_file())
                materialize.assert_not_called()

    def test_main_rejects_unsafe_orphan_stage_before_mutating_any_stage(self):
        for unsafe_kind in ("symlink", "special"):
            with self.subTest(unsafe_kind=unsafe_kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                paths = _paths(root)
                paths.local_handoff.mkdir(parents=True)
                safe = paths.local_handoff / ".incoming-campaign-aaaaaaaa"
                safe.mkdir()
                (safe / "partial").write_bytes(b"safe-but-stale")
                unsafe = paths.local_handoff / ".incoming-campaign-bbbbbbbb"
                unsafe.mkdir()
                if unsafe_kind == "symlink":
                    (unsafe / "escape").symlink_to(root)
                else:
                    os.mkfifo(unsafe / "fifo")
                backend = (
                    root / "contract.json",
                    _contract(),
                    SDK_PYTHON,
                    Path("/mnt/data1/ldu/lepton_pod_exec.py"),
                )
                with mock.patch.object(
                    bridge, "_load_backend", return_value=backend
                ), mock.patch.object(
                    bridge, "derive_paths", return_value=paths
                ), mock.patch.object(
                    bridge, "_materialize_request"
                ) as materialize, mock.patch.object(
                    bridge, "_ensure_remote_directories"
                ) as ensure_remote, mock.patch.object(
                    bridge, "_upload_projection"
                ) as upload_projection, self.assertRaisesRegex(
                    bridge.ProductionLeptonBridgeError, "symlink or special"
                ):
                    bridge.main()
                self.assertTrue(safe.is_dir())
                self.assertTrue((safe / "partial").is_file())
                self.assertTrue(os.path.lexists(unsafe))
                materialize.assert_not_called()
                ensure_remote.assert_not_called()
                upload_projection.assert_not_called()

    def test_success_retirement_preserves_canonical_campaign_and_audit_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, upload, evidence, result, environment = _retirement_fixture(root)
            campaign_root = Path(environment["ROUND_DAGGER_CAMPAIGN_ROOT"])
            canonical_before = {
                path.relative_to(campaign_root).as_posix(): (
                    path.stat().st_size,
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
                for path in campaign_root.rglob("*")
                if path.is_file()
            }
            with mock.patch.dict(os.environ, environment, clear=True):
                first = bridge.retire_local_handoff_inputs(
                    paths,
                    upload_report=upload,
                    evidence=evidence,
                    result=result,
                )
                second = bridge.retire_local_handoff_inputs(
                    paths,
                    upload_report=upload,
                    evidence=evidence,
                    result=result,
                )
            self.assertEqual(first["status"], "published")
            self.assertEqual(second["status"], "adopted")
            self.assertFalse(paths.local_projection.exists())
            self.assertFalse(paths.local_archive.exists())
            self.assertTrue(paths.local_inventory.is_file())
            self.assertTrue(paths.local_manifest.is_file())
            self.assertTrue(paths.local_request.is_file())
            self.assertTrue(paths.local_marker.is_file())
            self.assertTrue(paths.local_checksum.is_file())
            self.assertTrue(paths.local_result.is_file())
            self.assertTrue((paths.local_return_bundle / "audit.txt").is_file())
            canonical_after = {
                path.relative_to(campaign_root).as_posix(): (
                    path.stat().st_size,
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
                for path in campaign_root.rglob("*")
                if path.is_file()
            }
            self.assertEqual(canonical_before, canonical_after)

    def test_retirement_resumes_after_archive_and_partial_projection_removal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, upload, evidence, result, environment = _retirement_fixture(root)
            original = bridge._retire_projection_subset

            def interrupt_after_one_file(projection, *, inventory, allowed_directories):
                victim = next(path for path in projection.rglob("*") if path.is_file())
                victim.unlink()
                raise RuntimeError("simulated interruption")

            with mock.patch.dict(os.environ, environment, clear=True):
                with mock.patch.object(
                    bridge,
                    "_retire_projection_subset",
                    side_effect=interrupt_after_one_file,
                ), self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    bridge.retire_local_handoff_inputs(
                        paths,
                        upload_report=upload,
                        evidence=evidence,
                        result=result,
                    )
                self.assertFalse(paths.local_archive.exists())
                self.assertTrue(paths.local_projection.exists())
                self.assertTrue(
                    (paths.local_handoff / "local_input_retirement_intent.json").is_file()
                )
                with mock.patch.object(
                    bridge, "_retire_projection_subset", side_effect=original
                ):
                    resumed = bridge.retire_local_handoff_inputs(
                        paths,
                        upload_report=upload,
                        evidence=evidence,
                        result=result,
                    )
            self.assertEqual(resumed["status"], "published")
            self.assertFalse(paths.local_projection.exists())
            self.assertTrue(paths.local_result.exists())

    def test_main_resumes_intent_committed_partial_retirement_without_remote_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, upload, evidence, result, environment = _retirement_fixture(root)
            campaign_root = Path(environment["ROUND_DAGGER_CAMPAIGN_ROOT"])
            canonical_before = {
                path.relative_to(campaign_root).as_posix(): (
                    path.stat().st_size,
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
                for path in campaign_root.rglob("*")
                if path.is_file()
            }

            def interrupt_after_one_file(projection, *, inventory, allowed_directories):
                victim = next(path for path in projection.rglob("*") if path.is_file())
                victim.unlink()
                raise RuntimeError("simulated SIGKILL frontier")

            with mock.patch.dict(os.environ, environment, clear=True):
                with mock.patch.object(
                    bridge,
                    "_retire_projection_subset",
                    side_effect=interrupt_after_one_file,
                ), self.assertRaisesRegex(RuntimeError, "SIGKILL frontier"):
                    bridge.retire_local_handoff_inputs(
                        paths,
                        upload_report=upload,
                        evidence=evidence,
                        result=result,
                    )
            self.assertFalse(paths.local_archive.exists())
            self.assertTrue(paths.local_projection.exists())
            intent_path = paths.local_handoff / "local_input_retirement_intent.json"
            self.assertTrue(intent_path.is_file())
            request_sha256 = hashlib.sha256(paths.local_request.read_bytes()).hexdigest()
            request = SimpleNamespace(
                sha256=request_sha256,
                campaign_id=evidence["campaign_id"],
                macro_round=evidence["macro_round"],
                identity_sha256=evidence["learner_identity_sha256"],
                paths={
                    "campaign_root": paths.remote_campaign,
                    "learner_output": paths.remote_output,
                },
            )
            backend = (
                root / "contract.json",
                _contract(),
                SDK_PYTHON,
                Path("/mnt/data1/ldu/lepton_pod_exec.py"),
            )
            with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(
                bridge, "_load_backend", return_value=backend
            ), mock.patch.object(
                bridge, "derive_paths", return_value=paths
            ), mock.patch.object(
                bridge, "load_request", return_value=request
            ) as load_request, mock.patch.object(
                bridge, "validate_completion_marker", return_value=evidence
            ), mock.patch.object(
                bridge, "validate_local_result", return_value=result
            ), mock.patch.object(
                bridge,
                "publish_local_acceptance",
                return_value={"content_sha256": "a" * 64},
            ), mock.patch.object(
                bridge, "_emit_success_report"
            ), mock.patch.object(
                bridge, "_materialize_request"
            ) as materialize, mock.patch.object(
                bridge, "_ensure_remote_directories"
            ) as ensure_remote, mock.patch.object(
                bridge, "_upload_projection"
            ) as upload_projection, mock.patch.object(
                bridge, "_upload_regular"
            ) as upload_request, mock.patch.object(
                bridge, "_run"
            ) as lifecycle, mock.patch.object(
                bridge, "_fetch_and_validate_marker"
            ) as fetch_marker, mock.patch.object(
                bridge, "_download_output"
            ) as download_output, mock.patch("builtins.print"):
                self.assertEqual(bridge.main(), 0)
            load_request.assert_called_once_with(
                paths.local_request, contract=backend[1], immutable=True
            )
            materialize.assert_not_called()
            ensure_remote.assert_not_called()
            upload_projection.assert_not_called()
            upload_request.assert_not_called()
            lifecycle.assert_not_called()
            fetch_marker.assert_not_called()
            download_output.assert_not_called()
            self.assertFalse(paths.local_projection.exists())
            self.assertFalse(paths.local_archive.exists())
            self.assertTrue(
                (paths.local_handoff / "local_input_retirement_complete.json").is_file()
            )
            self.assertTrue(paths.local_result.is_file())
            canonical_after = {
                path.relative_to(campaign_root).as_posix(): (
                    path.stat().st_size,
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
                for path in campaign_root.rglob("*")
                if path.is_file()
            }
            self.assertEqual(canonical_before, canonical_after)

    def test_retirement_rejects_target_outside_identity_handoff(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, upload, evidence, result, environment = _retirement_fixture(root)
            unsafe = bridge.BridgePaths(
                **{
                    **paths.__dict__,
                    "local_projection": Path(environment["ROUND_DAGGER_CAMPAIGN_ROOT"]),
                }
            )
            with mock.patch.dict(os.environ, environment, clear=True), self.assertRaisesRegex(
                bridge.ProductionLeptonBridgeError, "outside"
            ):
                bridge.retire_local_handoff_inputs(
                    unsafe,
                    upload_report=upload,
                    evidence=evidence,
                    result=result,
                )
            self.assertTrue(Path(environment["ROUND_DAGGER_CAMPAIGN_ROOT"]).is_dir())

    def test_request_validation_precedes_every_remote_side_effect(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = _paths(root)
            paths.local_handoff.mkdir(parents=True)
            backend = (
                root / "contract.json",
                _contract(),
                SDK_PYTHON,
                Path("/mnt/data1/ldu/lepton_pod_exec.py"),
            )
            with mock.patch.object(
                bridge, "_load_backend", return_value=backend
            ), mock.patch.object(
                bridge, "derive_paths", return_value=paths
            ), mock.patch.object(
                bridge,
                "_materialize_request",
                side_effect=bridge.ProductionLeptonBridgeError(
                    "incomplete dashboard contract"
                ),
            ) as materialize, mock.patch.object(
                bridge, "_ensure_remote_directories"
            ) as ensure_remote, mock.patch.object(
                bridge, "_upload_projection"
            ) as upload_projection, mock.patch.object(
                bridge, "_upload_regular"
            ) as upload_request, mock.patch.object(bridge, "_run") as lifecycle:
                with self.assertRaisesRegex(
                    bridge.ProductionLeptonBridgeError,
                    "incomplete dashboard contract",
                ):
                    bridge.main()
            materialize.assert_called_once_with(backend[1], paths)
            ensure_remote.assert_not_called()
            upload_projection.assert_not_called()
            upload_request.assert_not_called()
            lifecycle.assert_not_called()

    def test_lifecycle_command_is_fully_bound_and_has_no_backend_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = _paths(root)
            contract = root / "contract.json"
            contract.write_text("{}\n", encoding="utf-8")
            request = SimpleNamespace(sha256="b" * 64)
            environment = {
                "ROUND_DAGGER_LEPTON_BACKEND_CONTRACT_SHA256": "a" * 64,
                "ROUND_DAGGER_LEPTON_REMOTE_BACKEND_CONTRACT": (
                    "/workspace/projects/roborender/manifests/backend.json"
                ),
            }
            with mock.patch.dict(os.environ, environment, clear=True):
                command = bridge.build_lifecycle_command(
                    contract,
                    request,
                    paths,
                    sdk_python=SDK_PYTHON,
                    control_exec=Path("/mnt/data1/ldu/lepton_pod_exec.py"),
                )
            self.assertEqual(command[0], str(SDK_PYTHON))
            self.assertIn("--exclusive-node-lifecycle", command)
            self.assertEqual(
                command[command.index("--request-sha256") + 1], "b" * 64
            )
            self.assertEqual(
                command[command.index("--remote-request") + 1],
                str(paths.remote_request),
            )
            self.assertNotIn("oci", " ".join(command).lower())
            self.assertNotIn("slurm", " ".join(command).lower())

    def test_request_upload_is_create_only_and_uses_explicit_sdk(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            local = Path(temporary) / "request.json"
            local.write_text("{}\n", encoding="utf-8")
            local.chmod(0o444)
            with mock.patch.object(
                bridge,
                "_run_json",
                return_value={"operation": "upload", "status": "published"},
            ) as run:
                bridge._upload_regular(
                    local,
                    PurePosixPath(
                        f"/workspace/projects/roborender/handoff/{IDENTITY}/request.json"
                    ),
                    deployment="control",
                    sdk_python=SDK_PYTHON,
                )
            command = run.call_args.args[0]
            self.assertEqual(command[0], str(SDK_PYTHON))
            self.assertIn("--sdk-python", command)
            self.assertIn("--adopt-existing", command)
            self.assertEqual(command[command.index("--mode") + 1], "0444")

    def test_request_upload_rejects_writable_local_request(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            local = Path(temporary) / "request.json"
            local.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(
                bridge.ProductionLeptonBridgeError, "read-only"
            ):
                bridge._upload_regular(
                    local,
                    PurePosixPath(
                        f"/workspace/projects/roborender/handoff/{IDENTITY}/request.json"
                    ),
                    deployment="control",
                    sdk_python=SDK_PYTHON,
                )

    def test_v4_result_bundle_uses_canonical_siblings_and_exact_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = _paths(root)
            paths.local_output.parent.mkdir(parents=True)
            evidence = {
                "learner_output_manifest_path": (
                    f"/workspace/projects/roborender/results/{IDENTITY}/"
                    "learner_output.manifest.json"
                ),
                "learner_output_manifest_sha256": "1" * 64,
                "learner_output_archive_sha256": "2" * 64,
                "learner_output_inventory_sha256": "3" * 64,
                "learner_output_tree_sha256": "4" * 64,
            }
            report = {
                "schema": bridge.archive_transfer.REPORT_SCHEMA,
                "operation": "download-publish",
                "status": "PASS",
                "manifest_sha256": "1" * 64,
                "publication": {
                    "archive_sha256": "2" * 64,
                    "inventory_sha256": "3" * 64,
                    "tree_sha256": "4" * 64,
                },
            }
            with mock.patch.dict(
                os.environ,
                {"ROUND_DAGGER_LEARNER_IDENTITY_SHA256": IDENTITY},
                clear=True,
            ), mock.patch.object(bridge, "_run_json", return_value=report) as run:
                bridge._download_output(
                    _contract(), evidence, paths, sdk_python=SDK_PYTHON
                )
            command = run.call_args.args[0]
            self.assertIn(
                f"/workspace/projects/roborender/results/{IDENTITY}/learner_output.tar.gz",
                command,
            )
            self.assertIn(
                f"/workspace/projects/roborender/results/{IDENTITY}/learner_output.inventory.json",
                command,
            )
            self.assertEqual(
                command[command.index("--destination") + 1],
                str(paths.local_output),
            )
            self.assertIn("--sdk-python", command)

    def test_v4_result_bundle_rejects_noncanonical_manifest_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = _paths(root)
            evidence = {
                "learner_output_manifest_path": (
                    "/workspace/projects/roborender/results/wrong/"
                    "learner_output.manifest.json"
                ),
                "learner_output_manifest_sha256": "1" * 64,
            }
            with mock.patch.dict(
                os.environ,
                {"ROUND_DAGGER_LEARNER_IDENTITY_SHA256": IDENTITY},
                clear=True,
            ), self.assertRaisesRegex(
                bridge.ProductionLeptonBridgeError, "non-canonical"
            ):
                bridge._download_output(
                    _contract(), evidence, paths, sdk_python=SDK_PYTHON
                )

    def test_local_result_must_match_v4_marker_before_candidate_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = _paths(root)
            paths.local_result.parent.mkdir(parents=True)
            paths.local_result.write_bytes(b"result")
            with self.assertRaisesRegex(
                bridge.ProductionLeptonBridgeError, "differs from marker"
            ):
                bridge.validate_local_result(
                    paths, {"result_sha256": "0" * 64}
                )

    def test_standard_result_loader_is_required_after_marker_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = _paths(root)
            paths.local_result.parent.mkdir(parents=True)
            paths.local_result.write_bytes(b"result")
            digest = hashlib.sha256(b"result").hexdigest()
            campaign = root / "campaign"
            replay_snapshot = campaign / "replay/snapshot.json"
            replay_snapshot.parent.mkdir(parents=True, exist_ok=True)
            replay_snapshot.write_text("{}\n", encoding="utf-8")
            environment = {
                "ROUND_DAGGER_MACRO_ROUND": "0",
                "ROUND_DAGGER_CAMPAIGN_ROOT": str(campaign),
                "ROUND_DAGGER_LEARNER_CODE_SHA256": "1" * 64,
                "ROUND_DAGGER_LEARNER_CONFIG_SHA256": "2" * 64,
                "ROUND_DAGGER_LEARNER_MODEL_LINEAGE_SHA256": "3" * 64,
                "ROUND_DAGGER_REPLAY_SNAPSHOT": str(replay_snapshot),
                "ROUND_DAGGER_REPLAY_SNAPSHOT_FILE_SHA256": "4" * 64,
            }
            backend = mock.Mock()
            backend.load_learner_result.return_value = {
                "manifest_path": str(paths.local_result),
                "checkpoint_sha256": "5" * 64,
            }
            with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(
                bridge.orchestration, "ProductionBackend", return_value=backend
            ), mock.patch.object(bridge.state, "load_campaign", return_value={}), mock.patch.object(
                bridge.replay, "load_snapshot", return_value={}
            ):
                result = bridge.validate_local_result(
                    paths, {"result_sha256": digest}
                )
            self.assertEqual(result["checkpoint_sha256"], "5" * 64)
            backend.load_learner_result.assert_called_once()

    def test_trusted_hook_rejects_missing_orchestrator_environment_before_submit(self) -> None:
        environment = {
            "PATH": os.environ.get("PATH", ""),
            "ROUND_DAGGER_TRUSTED_TRAIN_DISPATCH": "1",
            "ROUND_DAGGER_LEARNER_BACKEND": "lepton_job_v1",
            "ROUND_DAGGER_TRUSTED_HOOK_SOURCE_ROOT": str(ROOT),
            "ROUND_DAGGER_TRUSTED_HOOK_SOURCE_INVENTORY_SHA256": "a" * 64,
        }
        completed = subprocess.run(
            [str(ROOT / "scripts/round_dagger/run_production_lepton_learner_hook.sh")],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("ROUND_DAGGER_CAMPAIGN_ROOT", completed.stderr)


if __name__ == "__main__":
    unittest.main()
