from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from round_dagger import artifacts, learner_recovery, lepton_job, state
from round_dagger.orchestration import (
    LEARNER_BACKEND_LEPTON_JOB_V1,
    LEARNER_BACKEND_OCI_NRT,
)
from round_dagger.promotion import stage_bootstrap_policy
from scripts.round_dagger import production_launch
from tests.policy_bundle_helpers import (
    complete_runtime_policy_sha256,
    publish_policy_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_CONTRACT = ROOT / "configs/round_dagger/production_campaign_256x512.json"
LEGACY_OCI_LEARNER_CONTRACT = (
    ROOT / "configs/round_dagger/production_learner_8xh100_1200.json"
)
LEPTON_BACKEND_TEMPLATE = (
    ROOT / "configs/round_dagger/production_learner_1x8h200_lepton.json"
)
SOURCE_SHA = "7" * 64
SNAPSHOT_SEMANTIC_SHA = "8" * 64
HELDOUT_SEMANTIC_SHA = "a" * 64
CANDIDATE_POLICY_STATE_SHA = "9" * 64
CANDIDATE_RUNTIME_SHA = complete_runtime_policy_sha256(
    CANDIDATE_POLICY_STATE_SHA
)


class IngestedLearnerRecoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="rr_learner_recovery_")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        campaign_contract = artifacts.load_strict_json(CAMPAIGN_CONTRACT)
        self.campaign_contract_sha = artifacts.sha256_file(CAMPAIGN_CONTRACT)
        self.root = self.base / "campaign"
        self.root.mkdir()
        self.heldout_path = self.root / "heldout_snapshot.json"
        self.heldout_path.write_text("{}\n", encoding="utf-8")
        self.heldout_file_sha = artifacts.sha256_file(self.heldout_path)
        legacy = artifacts.load_strict_json(LEGACY_OCI_LEARNER_CONTRACT)
        portable_contract = {
            "schema": "fast_roborender.round_dagger.production_learner_contract.v2",
            "allocation": {
                "world_size": 8,
                "batch_size_per_rank": legacy["allocation"][
                    "batch_size_per_rank"
                ],
                "global_batch_size": legacy["allocation"]["global_batch_size"],
            },
            "optimization": legacy["optimization"],
            "handoff": {"local_root": legacy["handoff"]["local_root"]},
            "wandb": legacy["wandb"],
            "source": legacy["source"],
            "heldout_evaluation": {
                "snapshot_path": str(self.heldout_path),
                "snapshot_file_sha256": self.heldout_file_sha,
                "snapshot_sha256": HELDOUT_SEMANTIC_SHA,
                "interval_steps": 100,
            },
        }
        self.learner_contract = self.root / "learner_contract.json"
        artifacts.publish_immutable_json(
            self.learner_contract, portable_contract
        )
        self.learner_contract_sha = artifacts.sha256_file(
            self.learner_contract
        )
        source = self.base / "bootstrap.pt"
        source.write_bytes(b"fresh-base-policy")
        publish_policy_bundle(source, "1" * 64)
        bootstrap = stage_bootstrap_policy(
            self.root,
            checkpoint_source=source,
            code_sha256="2" * 64,
            policy_config_sha256=self.learner_contract_sha,
        )
        self.assertEqual(
            artifacts.sha256_file(self.learner_contract),
            self.learner_contract_sha,
        )
        state.initialize_campaign(
            self.root,
            campaign_id=campaign_contract["campaign_id"],
            bootstrap_policy_sha256=bootstrap["checkpoint_sha256"],
            bootstrap_policy_path=bootstrap["checkpoint_path"],
            target_environments=256,
            active_environments=64,
            steps_per_environment=512,
            view_names=campaign_contract["observation_contract"]["policy_views"],
            policy_image_shape=campaign_contract["observation_contract"][
                "policy_image_shape"
            ],
            action_dim=campaign_contract["action_contract"]["replay_action_dim"],
            teacher_sha256=campaign_contract["artifacts"]["teacher"][
                "checkpoint_sha256"
            ],
            renderer_sha256=campaign_contract["artifacts"]["renderer"][
                "cohort_identity_sha256"
            ],
            camera_mapping_sha256=campaign_contract["artifacts"][
                "camera_acceptance"
            ]["sha256"],
            training_config_sha256=campaign_contract["evaluation"][
                "hydra_config_sha256"
            ],
            beta_schedule=campaign_contract["beta_schedule"],
            macro_round_count=campaign_contract["topology"][
                "macro_round_count"
            ],
        )
        state.open_macro_round(
            self.root, expected_revision=state.load_state(self.root)["revision"]
        )
        self.snapshot_path = (
            self.root
            / "replay/snapshots/through_macro_round_000000.json"
        )
        self.snapshot_path.parent.mkdir(parents=True)
        self.snapshot_path.write_text("{}\n", encoding="utf-8")
        self.snapshot_file_sha = artifacts.sha256_file(self.snapshot_path)
        collecting = state.load_state(self.root)
        ingested = {
            **collecting,
            "phase": "INGESTED",
            "last_collected_macro_round": 0,
            "last_ingested_macro_round": 0,
            "collection_manifest_sha256": "3" * 64,
            "replay_snapshot_sha256": self.snapshot_file_sha,
        }
        state._replace_state(self.root, ingested, collecting["revision"])
        self.ingested = state.load_state(self.root)
        self.campaign = state.load_campaign(self.root)
        self.campaign_sha = artifacts.canonical_json_sha256(self.campaign)
        self.active = self.ingested["active_policy"]
        self.scene_sample_manifest = (
            self.root
            / "rounds/macro_round_000000/scene_samples/manifest.json"
        )
        artifacts.publish_immutable_json(
            self.scene_sample_manifest,
            {
                "schema": "fast_roborender.round_dagger.scene_samples.v1",
                "campaign_id": self.campaign["campaign_id"],
                "rollout_steps": 512,
            },
        )
        self.snapshot = {
            "path": str(self.snapshot_path),
            "file_sha256": self.snapshot_file_sha,
            "snapshot_sha256": SNAPSHOT_SEMANTIC_SHA,
            "campaign_id": self.campaign["campaign_id"],
            "through_macro_round": 0,
            "expected_train_cohorts": 4,
            "cohort_count": 4,
            "transition_count": 131072,
        }
        self.heldout = {
            "path": str(self.heldout_path),
            "file_sha256": self.heldout_file_sha,
            "snapshot_sha256": HELDOUT_SEMANTIC_SHA,
            "campaign_id": self.campaign["campaign_id"],
            "view_contract": {
                "dtype": "uint8",
                "layout": "vertical",
                "shape": [324, 192, 3],
                "views": ["head_color", "hand_left", "hand_right"],
            },
            "action_contract": {"dtype": "float32", "shape": [26]},
        }
        backend_document = artifacts.load_strict_json(
            LEPTON_BACKEND_TEMPLATE
        )
        backend_document["supply_chain"]["source_inventory_sha256"] = SOURCE_SHA
        backend_document["supply_chain"]["python_freeze_sha256"] = "6" * 64
        self.lepton_backend_contract = self.base / "lepton_backend.json"
        artifacts.publish_immutable_json(
            self.lepton_backend_contract, backend_document
        )
        self.lepton_backend_contract_sha = artifacts.sha256_file(
            self.lepton_backend_contract
        )
        self.lepton_source_root = backend_document["supply_chain"][
            "source_root"
        ]
        self.lepton_remote_backend_contract = (
            "/workspace/projects/roborender/manifests/lepton_backend.json"
        )
        self.lepton_control_exec = Path("/mnt/data1/ldu/lepton_pod_exec.py")
        self.lepton_sdk_python = production_launch.LEPTON_SDK_PYTHON

        seal_parent = self.base / "sealed-coordinator"
        self.evaluation_source_root = (
            seal_parent / "fast_roborender_deadbee_cafef00_sealed"
        )
        relative_hook = (
            "scripts/round_dagger/run_production_lepton_learner_hook.sh"
        )
        sealed_hook = self.evaluation_source_root / relative_hook
        sealed_hook.parent.mkdir(parents=True)
        sealed_hook.write_bytes(learner_recovery.LEPTON_BRIDGE.read_bytes())
        sealed_hook.chmod(0o555)
        hook_bytes = sealed_hook.read_bytes()
        inventory = {
            "schema": "fast_roborender.hsg_source_inventory.v1",
            "file_count": 1,
            "total_file_bytes": len(hook_bytes),
            "files": [
                {
                    "path": relative_hook,
                    "mode": "0o555",
                    "bytes": len(hook_bytes),
                    "sha256": hashlib.sha256(hook_bytes).hexdigest(),
                }
            ],
        }
        self.evaluation_inventory = (
            seal_parent / f"{self.evaluation_source_root.name}.inventory.json"
        )
        self.evaluation_inventory.write_text(
            json.dumps(inventory, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        self.evaluation_inventory.chmod(0o444)
        for directory in (
            sealed_hook.parent,
            sealed_hook.parent.parent,
            self.evaluation_source_root,
            seal_parent,
        ):
            directory.chmod(0o555)
        self.evaluation_source_sha = artifacts.sha256_file(
            self.evaluation_inventory
        )
        self.evaluation_seal_report = {
            "inventory_path": str(self.evaluation_inventory)
        }
        self.addCleanup(self._restore_seal_permissions)

        telemetry = (
            self.root
            / "rounds/macro_round_000000/orchestration_run/telemetry/scalars"
            / "step_000000020_macro_round.json"
        )
        telemetry.parent.mkdir(parents=True)
        telemetry.write_bytes(b"immutable-existing-round-telemetry\n")
        self.telemetry = telemetry
        self.telemetry_before = telemetry.read_bytes()

    def _restore_seal_permissions(self) -> None:
        for directory in sorted(
            (
                path
                for path in (self.base / "sealed-coordinator").rglob("*")
                if path.is_dir()
            ),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            directory.chmod(0o755)
        parent = self.base / "sealed-coordinator"
        if parent.exists():
            parent.chmod(0o755)

    @contextmanager
    def _validated_inputs(self):
        with (
            mock.patch.object(
                learner_recovery.replay,
                "load_snapshot",
                return_value=self.snapshot,
            ),
            mock.patch.object(
                production_launch.replay,
                "load_heldout_snapshot",
                return_value=self.heldout,
            ),
            mock.patch.object(
                learner_recovery.orchestration.source_seal,
                "verify",
                return_value=self.evaluation_seal_report,
            ),
        ):
            yield

    def _kwargs(self) -> dict[str, object]:
        return {
            "campaign_root": self.root,
            "macro_round": 0,
            "expected_state_revision": self.ingested["revision"],
            "expected_campaign_sha256": self.campaign_sha,
            "expected_replay_snapshot_file_sha256": self.snapshot_file_sha,
            "expected_replay_snapshot_sha256": SNAPSHOT_SEMANTIC_SHA,
            "expected_parent_checkpoint_sha256": self.active["checkpoint_sha256"],
            "expected_parent_runtime_policy_sha256": self.active[
                "runtime_policy_sha256"
            ],
            "campaign_contract_path": CAMPAIGN_CONTRACT,
            "expected_campaign_contract_sha256": self.campaign_contract_sha,
            "learner_contract_path": self.learner_contract,
            "expected_learner_contract_sha256": self.learner_contract_sha,
            "sealed_source_root": self.lepton_source_root,
            "sealed_source_inventory_sha256": SOURCE_SHA,
            "evaluation_sealed_source_root": self.evaluation_source_root,
            "evaluation_sealed_source_inventory_sha256": (
                self.evaluation_source_sha
            ),
            "expected_bridge_sha256": artifacts.sha256_file(
                learner_recovery.LEPTON_BRIDGE
            ),
            "learner_backend": LEARNER_BACKEND_LEPTON_JOB_V1,
            "learner_backend_contract_path": self.lepton_backend_contract,
            "expected_learner_backend_contract_sha256": (
                self.lepton_backend_contract_sha
            ),
            "lepton_remote_backend_contract": (
                self.lepton_remote_backend_contract
            ),
            "lepton_control_exec_program": self.lepton_control_exec,
            "lepton_sdk_python": self.lepton_sdk_python,
        }

    def _publish_learner_result(self, environment: dict[str, str]) -> None:
        output = Path(environment["ROUND_DAGGER_LEARNER_OUTPUT_DIR"])
        output.mkdir(parents=True, exist_ok=True)
        checkpoint = output / "candidate_round_000000.pt"
        checkpoint.write_bytes(b"balanced-recovery-candidate")
        specification, manifest = publish_policy_bundle(
            checkpoint, CANDIDATE_POLICY_STATE_SHA
        )
        result = {
            "schema": "fast_roborender.round_dagger.learner_result.v2",
            "campaign_id": self.campaign["campaign_id"],
            "macro_round": 0,
            "checkpoint_path": checkpoint.name,
            "checkpoint_sha256": artifacts.sha256_file(checkpoint),
            "runtime_policy_sha256": CANDIDATE_RUNTIME_SHA,
            "policy_spec_path": specification.name,
            "policy_spec_sha256": artifacts.sha256_file(specification),
            "policy_bundle_manifest_path": manifest.name,
            "policy_bundle_manifest_sha256": artifacts.sha256_file(manifest),
            "model_lineage_sha256": self.active["checkpoint_sha256"],
            "replay_snapshot_sha256": SNAPSHOT_SEMANTIC_SHA,
            "code_sha256": environment["ROUND_DAGGER_LEARNER_CODE_SHA256"],
            "learner_config_sha256": environment[
                "ROUND_DAGGER_LEARNER_CONFIG_SHA256"
            ],
            "optimizer_steps": 1200,
            "initial_loss": 1.0,
            "final_loss": 0.25,
        }
        artifacts.publish_immutable_json(
            Path(environment["ROUND_DAGGER_LEARNER_RESULT"]), result
        )

    def _publish_acceptance_placeholder(self) -> Path:
        path = learner_recovery.learner_acceptance.acceptance_path(self.root, 0)
        artifacts.publish_immutable_json(path, {"status": "PASS"})
        return path

    def test_exact_recovery_marks_trained_without_replaying_round_telemetry(self) -> None:
        calls: list[dict[str, str]] = []

        def bridge_runner(
            bridge: Path, environment: dict[str, str], workdir: Path, timeout: int
        ) -> None:
            self.assertEqual(bridge, learner_recovery.LEPTON_BRIDGE)
            self.assertEqual(workdir, self.root)
            self.assertEqual(timeout, 172800)
            self.assertEqual(environment["ROUND_DAGGER_LEARNER_MAX_STEPS"], "1200")
            self.assertEqual(
                environment["ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256"],
                SNAPSHOT_SEMANTIC_SHA,
            )
            self.assertEqual(
                environment["ROUND_DAGGER_LEARNER_CODE_SHA256"], SOURCE_SHA
            )
            self.assertEqual(
                environment["ROUND_DAGGER_LEARNER_BACKEND"],
                LEARNER_BACKEND_LEPTON_JOB_V1,
            )
            self.assertEqual(environment["ROUND_DAGGER_BETA"], "1.0")
            self.assertEqual(
                environment["ROUND_DAGGER_BETA_SHA256"],
                artifacts.canonical_json_sha256({"beta": 1.0}),
            )
            self.assertEqual(
                environment["ROUND_DAGGER_SCENE_SAMPLE_MANIFEST"],
                str(self.scene_sample_manifest),
            )
            calls.append(environment)
            self._publish_learner_result(environment)
            self._publish_acceptance_placeholder()

        with self._validated_inputs(), mock.patch.object(
            learner_recovery.learner_acceptance,
            "validate_acceptance",
            return_value={"status": "PASS"},
        ):
            result = learner_recovery.recover_ingested_learner(
                **self._kwargs(), bridge_runner=bridge_runner
            )
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["status"], "TRAINED")
        trained = state.load_state(self.root)
        self.assertEqual(trained["phase"], "TRAINED")
        self.assertEqual(trained["revision"], self.ingested["revision"] + 1)
        self.assertEqual(
            trained["candidate_policy"]["manifest_sha256"],
            result["candidate_manifest_sha256"],
        )
        receipt = artifacts.load_strict_json(self.root / result["receipt_path"])
        self.assertEqual(receipt["schema"], learner_recovery.RECEIPT_SCHEMA)
        self.assertEqual(receipt["trained_revision"], trained["revision"])
        config = artifacts.load_strict_json(
            self.root
            / "rounds/macro_round_000000/learner_recovery/"
            "orchestration_config.json"
        )
        self.assertEqual(
            config["schema"],
            "fast_roborender.round_dagger.orchestration_config.v4",
        )
        self.assertEqual(
            config["learner_backend"], LEARNER_BACKEND_LEPTON_JOB_V1
        )
        self.assertEqual(
            config["learner_backend_contract_sha256"],
            self.lepton_backend_contract_sha,
        )
        self.assertEqual(
            config["hooks"]["train"]["argv"],
            [str(learner_recovery.LEPTON_BRIDGE)],
        )
        self.assertEqual(self.telemetry.read_bytes(), self.telemetry_before)

    def test_existing_v4_recovery_metadata_retries_after_bridge_interruption(self) -> None:
        """A durable pre-V5 recovery plan must be byte-for-byte reusable.

        The recovery metadata is published before the remote bridge begins.  A
        bridge interruption therefore leaves an INGESTED campaign with the
        exact immutable V4 config that a later retry must reopen rather than
        replacing with a new orchestration schema.
        """

        def interrupted_bridge(
            _bridge: Path,
            _environment: dict[str, str],
            _workdir: Path,
            _timeout: int,
        ) -> None:
            raise RuntimeError("injected bridge interruption")

        with (
            self._validated_inputs(),
            self.assertRaisesRegex(RuntimeError, "injected bridge interruption"),
        ):
            learner_recovery.recover_ingested_learner(
                **self._kwargs(), bridge_runner=interrupted_bridge
            )

        recovery_dir = self.root / "rounds/macro_round_000000/learner_recovery"
        config_path = recovery_dir / "orchestration_config.json"
        plan_path = recovery_dir / "plan.json"
        environment_path = recovery_dir / "bridge_environment.json"
        original_config = config_path.read_bytes()
        original_plan = plan_path.read_bytes()
        original_environment = environment_path.read_bytes()
        self.assertEqual(
            artifacts.load_strict_json(config_path)["schema"],
            "fast_roborender.round_dagger.orchestration_config.v4",
        )
        self.assertEqual(state.load_state(self.root)["phase"], "INGESTED")

        def completed_bridge(
            _bridge: Path,
            environment: dict[str, str],
            _workdir: Path,
            _timeout: int,
        ) -> None:
            self._publish_learner_result(environment)
            self._publish_acceptance_placeholder()

        with (
            self._validated_inputs(),
            mock.patch.object(
                learner_recovery.learner_acceptance,
                "validate_acceptance",
                return_value={"status": "PASS"},
            ),
        ):
            result = learner_recovery.recover_ingested_learner(
                **self._kwargs(), bridge_runner=completed_bridge
            )

        self.assertEqual(result["status"], "TRAINED")
        self.assertEqual(config_path.read_bytes(), original_config)
        self.assertEqual(plan_path.read_bytes(), original_plan)
        self.assertEqual(environment_path.read_bytes(), original_environment)

    def test_dry_run_is_read_only_and_preexisting_artifacts_fail_closed(self) -> None:
        with self._validated_inputs():
            result = learner_recovery.recover_ingested_learner(
                **self._kwargs(), dry_run=True
            )
        self.assertEqual(result["status"], "validated_no_submission")
        self.assertFalse(
            (self.root / "rounds/macro_round_000000/learner_recovery").exists()
        )
        learner = self.root / "rounds/macro_round_000000/learner"
        learner.mkdir()
        (learner / "unexpected.bin").write_bytes(b"stale")
        with (
            self._validated_inputs(),
            self.assertRaisesRegex(
                learner_recovery.LearnerRecoveryError, "learner output"
            ),
        ):
            learner_recovery.recover_ingested_learner(**self._kwargs())

    def test_default_environment_is_lepton_only_and_backend_bound(self) -> None:
        with self._validated_inputs():
            plan = learner_recovery.recover_ingested_learner(
                **self._kwargs(), dry_run=True
            )
        environment = plan["bridge_environment"]
        self.assertEqual(
            environment["ROUND_DAGGER_LEARNER_BACKEND"],
            LEARNER_BACKEND_LEPTON_JOB_V1,
        )
        self.assertEqual(
            environment["ROUND_DAGGER_LEARNER_LEARNING_RATE"], "2.5e-5"
        )
        self.assertEqual(
            environment["ROUND_DAGGER_WANDB_RUN_ID"],
            f"rrd-m000000-id{environment['ROUND_DAGGER_LEARNER_IDENTITY_SHA256']}",
        )
        self.assertEqual(
            environment["ROUND_DAGGER_LEARNER_CONTRACT_SHA256"],
            self.learner_contract_sha,
        )
        self.assertEqual(environment["ROUND_DAGGER_BETA"], "1.0")
        self.assertEqual(
            environment["ROUND_DAGGER_BETA_SHA256"],
            artifacts.canonical_json_sha256({"beta": 1.0}),
        )
        self.assertEqual(
            environment["ROUND_DAGGER_SCENE_SAMPLE_MANIFEST"],
            str(self.scene_sample_manifest),
        )
        self.assertEqual(
            environment["ROUND_DAGGER_LEPTON_BACKEND_CONTRACT_SHA256"],
            self.lepton_backend_contract_sha,
        )
        self.assertEqual(
            environment["ROUND_DAGGER_REMOTE_CONTAINER_SHA256"],
            artifacts.load_strict_json(self.lepton_backend_contract)[
                "container"
            ]["image"].split("@sha256:", 1)[1],
        )
        for stale in (
            "ROUND_DAGGER_REMOTE_HOST",
            "ROUND_DAGGER_REMOTE_ACCOUNT",
            "ROUND_DAGGER_REMOTE_PARTITION",
            "ROUND_DAGGER_REMOTE_QOS",
            "ROUND_DAGGER_REMOTE_SLURM_BIN",
            "ROUND_DAGGER_REMOTE_CONTAINER",
        ):
            self.assertNotIn(stale, environment)
        self.assertEqual(plan["learner_backend"], LEARNER_BACKEND_LEPTON_JOB_V1)
        self.assertEqual(
            plan["learner_backend_contract_sha256"],
            self.lepton_backend_contract_sha,
        )

    def test_recovery_requires_existing_bound_scene_sample_manifest(self) -> None:
        self.scene_sample_manifest.unlink()
        with (
            self._validated_inputs(),
            self.assertRaisesRegex(
                learner_recovery.LearnerRecoveryError,
                "scene sample manifest is missing or unsafe",
            ),
        ):
            learner_recovery.recover_ingested_learner(
                **self._kwargs(), dry_run=True
            )

    def test_sealed_hook_preflights_every_required_lepton_environment(self) -> None:
        text = learner_recovery.LEPTON_BRIDGE.read_text(encoding="utf-8")
        match = re.search(r"required=\(\n(.*?)\n\)", text, re.DOTALL)
        self.assertIsNotNone(match)
        declared = set(re.findall(r"ROUND_DAGGER_[A-Z0-9_]+", match.group(1)))
        expected = set(lepton_job._REQUIRED_ENV) | set(lepton_job._DASHBOARD_ENV)
        self.assertFalse(expected - declared)

    def test_state_revision_race_after_bridge_blocks_candidate_and_cas(self) -> None:
        def racing_bridge(
            _bridge: Path,
            environment: dict[str, str],
            _workdir: Path,
            _timeout: int,
        ) -> None:
            self._publish_learner_result(environment)
            self._publish_acceptance_placeholder()
            current = state.load_state(self.root)
            state._replace_state(self.root, current, current["revision"])

        with (
            self._validated_inputs(),
            mock.patch.object(
                learner_recovery.learner_acceptance,
                "validate_acceptance",
                return_value={"status": "PASS"},
            ),
            self.assertRaisesRegex(
                learner_recovery.LearnerRecoveryError,
                "changed while learner bridge was running",
            ),
        ):
            learner_recovery.recover_ingested_learner(
                **self._kwargs(), bridge_runner=racing_bridge
            )
        self.assertEqual(state.load_state(self.root)["phase"], "INGESTED")
        self.assertFalse((self.root / "policies/policy_000001").exists())

    def test_lepton_result_without_acceptance_cannot_advance(self) -> None:
        def incomplete_bridge(
            _bridge: Path,
            environment: dict[str, str],
            _workdir: Path,
            _timeout: int,
        ) -> None:
            self._publish_learner_result(environment)

        with (
            self._validated_inputs(),
            self.assertRaisesRegex(
                learner_recovery.LearnerRecoveryError,
                "acceptance is invalid: acceptance receipt is unavailable",
            ),
        ):
            learner_recovery.recover_ingested_learner(
                **self._kwargs(), bridge_runner=incomplete_bridge
            )
        self.assertEqual(state.load_state(self.root)["phase"], "INGESTED")
        self.assertFalse((self.root / "policies/policy_000001").exists())

    def test_lepton_inputs_and_source_identity_fail_closed(self) -> None:
        missing = self._kwargs()
        missing["lepton_remote_backend_contract"] = None
        with (
            self._validated_inputs(),
            self.assertRaisesRegex(
                learner_recovery.LearnerRecoveryError,
                "Lepton backend inputs are incomplete",
            ),
        ):
            learner_recovery.recover_ingested_learner(**missing, dry_run=True)

        mismatched = self._kwargs()
        mismatched["sealed_source_inventory_sha256"] = "b" * 64
        with (
            self._validated_inputs(),
            self.assertRaisesRegex(
                learner_recovery.LearnerRecoveryError,
                "differs from the sealed learner source",
            ),
        ):
            learner_recovery.recover_ingested_learner(
                **mismatched, dry_run=True
            )

        wrong_coordinator_seal = self._kwargs()
        wrong_coordinator_seal[
            "evaluation_sealed_source_inventory_sha256"
        ] = "c" * 64
        with (
            self._validated_inputs(),
            self.assertRaisesRegex(
                learner_recovery.LearnerRecoveryError,
                "sealed learner bridge verification failed",
            ),
        ):
            learner_recovery.recover_ingested_learner(
                **wrong_coordinator_seal, dry_run=True
            )

    def test_recovery_requires_immutable_campaign_owned_learner_contract(self) -> None:
        alternate = self.base / "alternate_learner_contract.json"
        alternate.write_bytes(self.learner_contract.read_bytes())
        alternate.chmod(0o444)
        alternate_kwargs = self._kwargs()
        alternate_kwargs["learner_contract_path"] = alternate
        with (
            self._validated_inputs(),
            self.assertRaisesRegex(
                learner_recovery.LearnerRecoveryError,
                "campaign-owned learner contract is invalid",
            ),
        ):
            learner_recovery.recover_ingested_learner(
                **alternate_kwargs, dry_run=True
            )

        self.learner_contract.chmod(0o644)
        self.addCleanup(self.learner_contract.chmod, 0o444)
        with (
            self._validated_inputs(),
            self.assertRaisesRegex(
                learner_recovery.LearnerRecoveryError,
                "campaign-owned learner contract is invalid",
            ),
        ):
            learner_recovery.recover_ingested_learner(
                **self._kwargs(), dry_run=True
            )

    def test_oci_compatibility_requires_explicit_backend(self) -> None:
        legacy = production_launch.load_learner_contract(
            LEGACY_OCI_LEARNER_CONTRACT
        )
        legacy_sha = artifacts.sha256_file(LEGACY_OCI_LEARNER_CONTRACT)
        binding = learner_recovery._validate_backend_binding(
            learner_backend=LEARNER_BACKEND_OCI_NRT,
            learner_contract=legacy,
            learner_contract_path=LEGACY_OCI_LEARNER_CONTRACT,
            learner_contract_sha256=legacy_sha,
            learner_backend_contract_path=None,
            expected_learner_backend_contract_sha256=None,
            sealed_source_root="/legacy/oci/sealed/source",
            sealed_source_inventory_sha256=SOURCE_SHA,
            lepton_remote_backend_contract=None,
            lepton_control_exec_program=None,
            lepton_sdk_python=None,
        )
        self.assertEqual(binding["backend"], LEARNER_BACKEND_OCI_NRT)
        self.assertEqual(
            binding["runtime_container_sha256"],
            legacy["container"]["sha256"],
        )
        with self.assertRaisesRegex(
            learner_recovery.LearnerRecoveryError,
            "portable/Lepton learner contract",
        ):
            learner_recovery._validate_backend_binding(
                learner_backend=LEARNER_BACKEND_LEPTON_JOB_V1,
                learner_contract=legacy,
                learner_contract_path=LEGACY_OCI_LEARNER_CONTRACT,
                learner_contract_sha256=legacy_sha,
                learner_backend_contract_path=self.lepton_backend_contract,
                expected_learner_backend_contract_sha256=(
                    self.lepton_backend_contract_sha
                ),
                sealed_source_root=self.lepton_source_root,
                sealed_source_inventory_sha256=SOURCE_SHA,
                lepton_remote_backend_contract=(
                    self.lepton_remote_backend_contract
                ),
                lepton_control_exec_program=self.lepton_control_exec,
                lepton_sdk_python=self.lepton_sdk_python,
            )


if __name__ == "__main__":
    unittest.main()
