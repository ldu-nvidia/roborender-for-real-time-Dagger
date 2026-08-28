from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from round_dagger import artifacts, learner_recovery, state
from round_dagger.promotion import stage_bootstrap_policy
from scripts.round_dagger import production_launch
from tests.policy_bundle_helpers import (
    complete_runtime_policy_sha256,
    publish_policy_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_CONTRACT = ROOT / "configs/round_dagger/production_campaign_256x512.json"
LEARNER_CONTRACT = ROOT / "configs/round_dagger/production_learner_8xh100_1200.json"
SOURCE_SHA = "7" * 64
SNAPSHOT_SEMANTIC_SHA = "8" * 64
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
        self.learner_contract_sha = artifacts.sha256_file(LEARNER_CONTRACT)
        self.campaign_contract_sha = artifacts.sha256_file(CAMPAIGN_CONTRACT)
        source = self.base / "bootstrap.pt"
        source.write_bytes(b"fresh-base-policy")
        publish_policy_bundle(source, "1" * 64)
        bootstrap = stage_bootstrap_policy(
            self.base / "campaign",
            checkpoint_source=source,
            code_sha256="2" * 64,
            policy_config_sha256=self.learner_contract_sha,
        )
        self.root = self.base / "campaign"
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
        telemetry = (
            self.root
            / "rounds/macro_round_000000/orchestration_run/telemetry/scalars"
            / "step_000000020_macro_round.json"
        )
        telemetry.parent.mkdir(parents=True)
        telemetry.write_bytes(b"immutable-existing-round-telemetry\n")
        self.telemetry = telemetry
        self.telemetry_before = telemetry.read_bytes()

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
            "learner_contract_path": LEARNER_CONTRACT,
            "expected_learner_contract_sha256": self.learner_contract_sha,
            "sealed_source_root": "/sealed/recovery/source",
            "sealed_source_inventory_sha256": SOURCE_SHA,
            "expected_bridge_sha256": artifacts.sha256_file(learner_recovery.BRIDGE),
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

    def test_exact_recovery_marks_trained_without_replaying_round_telemetry(self) -> None:
        calls: list[dict[str, str]] = []

        def bridge_runner(
            bridge: Path, environment: dict[str, str], workdir: Path, timeout: int
        ) -> None:
            self.assertEqual(bridge, learner_recovery.BRIDGE)
            self.assertEqual(workdir, self.root)
            self.assertEqual(timeout, 172800)
            self.assertEqual(environment["ROUND_DAGGER_LEARNER_MAX_STEPS"], "1200")
            self.assertEqual(environment["ROUND_DAGGER_REPLAY_SNAPSHOT_SHA256"], SNAPSHOT_SEMANTIC_SHA)
            self.assertEqual(environment["ROUND_DAGGER_LEARNER_CODE_SHA256"], SOURCE_SHA)
            calls.append(environment)
            self._publish_learner_result(environment)

        with mock.patch.object(
            learner_recovery.replay, "load_snapshot", return_value=self.snapshot
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
        self.assertEqual(self.telemetry.read_bytes(), self.telemetry_before)

    def test_dry_run_is_read_only_and_preexisting_artifacts_fail_closed(self) -> None:
        with mock.patch.object(
            learner_recovery.replay, "load_snapshot", return_value=self.snapshot
        ):
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
            mock.patch.object(
                learner_recovery.replay, "load_snapshot", return_value=self.snapshot
            ),
            self.assertRaisesRegex(
                learner_recovery.LearnerRecoveryError, "learner output"
            ),
        ):
            learner_recovery.recover_ingested_learner(**self._kwargs())

    def test_generated_environment_passes_real_bridge_dry_run(self) -> None:
        with mock.patch.object(
            learner_recovery.replay, "load_snapshot", return_value=self.snapshot
        ):
            plan = learner_recovery.recover_ingested_learner(
                **self._kwargs(), dry_run=True
            )
        environment = dict(os.environ)
        environment.update(plan["bridge_environment"])
        environment["DRY_RUN"] = "1"
        self.assertEqual(
            environment["ROUND_DAGGER_PYTHON"], os.path.abspath(sys.executable)
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
        completed = subprocess.run(
            [str(learner_recovery.BRIDGE)],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}",
        )
        self.assertIn(
            "OCI_NRT_ROUND_LEARNER_BRIDGE_DRY_RUN_OK", completed.stdout
        )

    def test_state_revision_race_after_bridge_blocks_candidate_and_cas(self) -> None:
        def racing_bridge(
            _bridge: Path,
            environment: dict[str, str],
            _workdir: Path,
            _timeout: int,
        ) -> None:
            self._publish_learner_result(environment)
            current = state.load_state(self.root)
            state._replace_state(self.root, current, current["revision"])

        with (
            mock.patch.object(
                learner_recovery.replay, "load_snapshot", return_value=self.snapshot
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


if __name__ == "__main__":
    unittest.main()
