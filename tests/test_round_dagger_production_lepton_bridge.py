from __future__ import annotations

import hashlib
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


class ProductionLeptonBridgeTests(unittest.TestCase):
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
