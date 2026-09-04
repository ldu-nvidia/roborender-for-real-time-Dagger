from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from round_dagger import artifacts
from round_dagger import fresh_lepton_campaign as fresh
from scripts.round_dagger import production_launch


def _write(path: Path, payload: bytes) -> str:
    path.write_bytes(payload)
    return artifacts.sha256_file(path)


class FreshLeptonCampaignTest(unittest.TestCase):
    def test_happy_path_keeps_learner_and_evaluation_seals_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            protected = parent / "old-oci"
            protected.mkdir()
            evaluation_root = parent / "accepted-12b9c60"
            evaluation_root.mkdir()
            output = parent / "fresh-lepton"
            files = {}
            for name in (
                "profile",
                "runtime",
                "aggregate",
                "plan",
                "motion",
                "scenarios",
                "base",
                "bootstrap",
            ):
                path = parent / f"{name}.json"
                files[name] = (path, _write(path, name.encode()))

            evaluation_sha = "d" * 64
            learner_sha = "e" * 64
            bootstrap_sha = files["bootstrap"][1]
            beta = {
                "kind": "linear",
                "start": 1.0,
                "end": 0.0,
                "decay_rounds": 4,
                "rate": 1.0,
            }
            lineage = {"schema": "fixture-lineage"}
            materialized = {
                "campaign_id": "dagger_prod_run1",
                "topology": {
                    "target_environments": 256,
                    "active_environments": 64,
                    "cohort_count": 4,
                    "environments_per_actor_rank": 32,
                    "actor_ranks_per_cohort": 2,
                    "macro_round_count": 5,
                    "steps_per_environment": 512,
                    "transitions_per_cohort": 32768,
                    "transitions_per_macro_round": 131072,
                    "transitions_per_campaign": 655360,
                },
                "observation_contract": {
                    "policy_views": ["head_color", "hand_left", "hand_right"],
                    "policy_image_shape": [324, 192, 3],
                    "policy_view_layout": "vertical",
                    "reference_lag_frames": 256,
                },
                "action_contract": {"replay_action_dim": 26},
                "motion_split": {
                    "heldout_plan_sha256": files["plan"][1],
                    "manifest_sha256": files["motion"][1],
                },
                "evaluation": {
                    "scenario_contract_sha256": files["scenarios"][1],
                    "hydra_config_sha256": "1" * 64,
                },
                "artifacts": {
                    "bootstrap_bundle": {
                        "checkpoint_path": str(files["bootstrap"][0]),
                        "checkpoint_sha256": bootstrap_sha,
                        "runtime_policy_sha256": "2" * 64,
                    },
                    "teacher": {"checkpoint_sha256": "3" * 64},
                    "renderer": {"cohort_identity_sha256": "4" * 64},
                    "camera_acceptance": {"sha256": "5" * 64},
                },
                "beta_schedule": beta,
                "data_lineage_sha256": "6" * 64,
            }
            runtime = {
                "contract_sha256": files["runtime"][1],
                "heldout_plan_path": str(files["plan"][0]),
                "split_manifest_path": str(files["motion"][0]),
                "scenario_path": str(files["scenarios"][0]),
                "capture_provenance": {
                    "evaluation_source_inventory_sha256": evaluation_sha
                },
            }
            campaign = {
                "campaign_id": "dagger_prod_run1",
                "target_environments": 256,
                "active_environments": 64,
                "cohort_count": 4,
                "steps_per_environment": 512,
                "macro_round_count": 5,
                "view_names": ["head_color", "hand_left", "hand_right"],
                "policy_image_shape": [324, 192, 3],
                "action_dim": 26,
                "teacher_sha256": "3" * 64,
                "renderer_sha256": "4" * 64,
                "camera_mapping_sha256": "5" * 64,
                "training_config_sha256": "1" * 64,
                "code_sha256": evaluation_sha,
                "beta_schedule": beta,
                "data_lineage": lineage,
                "data_lineage_sha256": "6" * 64,
            }
            active = {
                "version": 0,
                "checkpoint_path": "policies/policy_000000/checkpoint.pt",
                "checkpoint_sha256": bootstrap_sha,
                "runtime_policy_sha256": "2" * 64,
                "manifest_sha256": "7" * 64,
                "code_sha256": evaluation_sha,
            }
            coordinator = {
                "campaign_sha256": "8" * 64,
                "revision": 0,
                "phase": "READY",
                "next_macro_round": 0,
                "open_macro_round": None,
                "last_collected_macro_round": -1,
                "last_ingested_macro_round": -1,
                "last_trained_macro_round": -1,
                "last_evaluated_macro_round": -1,
                "candidate_policy": None,
                "active_policy": active,
            }

            def bind_learner(**kwargs):
                path = Path(kwargs["output_path"])
                path.write_text("{}\n", encoding="utf-8")
                path.chmod(0o444)
                digest = artifacts.sha256_file(path)
                active["policy_config_sha256"] = digest
                campaign["policy_config_sha256"] = digest
                return {"output_file_sha256": digest}

            def convert(*args, **kwargs):
                return {"case_count": 256, "cohort_count": 8, "file_sha256": "9" * 64}

            base_contract = {"schema": production_launch.LEARNER_CONTRACT_SCHEMA}
            learner_contract = dict(base_contract)
            with (
                mock.patch.object(
                    fresh.production_launch,
                    "materialize_campaign_contract",
                    return_value=(materialized, runtime, lineage),
                ),
                mock.patch.object(
                    fresh.production_launch,
                    "_verify_local_evaluation_seal",
                    return_value={
                        "root": str(evaluation_root),
                        "inventory_sha256": evaluation_sha,
                        "snapshot_name": evaluation_root.name,
                    },
                ),
                mock.patch.object(
                    fresh, "verify_capture_aggregate", return_value={
                        "file_sha256": files["aggregate"][1],
                        "plan_sha256": files["plan"][1],
                    }
                ),
                mock.patch.object(
                    fresh.production_launch,
                    "load_learner_contract",
                    return_value=base_contract,
                ),
                mock.patch.object(fresh, "convert_capture_aggregate", side_effect=convert),
                mock.patch.object(
                    fresh,
                    "verify_conversion_bundle",
                    return_value={
                        "file_sha256": "9" * 64,
                        "snapshot_path": str(output / "heldout.json"),
                        "snapshot_sha256": "a" * 64,
                        "case_count": 256,
                        "cohort_count": 8,
                    },
                ),
                mock.patch.object(
                    fresh,
                    "materialize_heldout_learner_contract",
                    side_effect=bind_learner,
                ),
                mock.patch.object(
                    fresh.production_launch,
                    "load_campaign_owned_learner_contract",
                    side_effect=lambda path, **kwargs: (
                        learner_contract,
                        artifacts.sha256_file(path),
                    ),
                ),
                mock.patch.object(fresh, "stage_bootstrap_policy", return_value=active),
                mock.patch.object(fresh.state, "initialize_campaign"),
                mock.patch.object(fresh.state, "load_campaign", return_value=campaign),
                mock.patch.object(fresh.state, "load_state", return_value=coordinator),
            ):
                report = fresh.materialize_fresh_lepton_campaign(
                    campaign_root=output,
                    protected_oci_campaign_root=protected,
                    invariant_profile_path=files["profile"][0],
                    invariant_profile_sha256=files["profile"][1],
                    runtime_contract_path=files["runtime"][0],
                    runtime_contract_sha256=files["runtime"][1],
                    capture_aggregate_path=files["aggregate"][0],
                    capture_aggregate_sha256=files["aggregate"][1],
                    heldout_plan_path=files["plan"][0],
                    heldout_plan_sha256=files["plan"][1],
                    motion_split_path=files["motion"][0],
                    motion_split_sha256=files["motion"][1],
                    scenarios_path=files["scenarios"][0],
                    scenarios_sha256=files["scenarios"][1],
                    base_learner_contract_path=files["base"][0],
                    base_learner_contract_sha256=files["base"][1],
                    learner_sealed_source_root="/users/ldu/projects/roborender/sealed/new",
                    learner_source_inventory_sha256=learner_sha,
                    evaluation_sealed_source_root=evaluation_root,
                    evaluation_source_inventory_sha256=evaluation_sha,
                )

            learner_contract_sha = artifacts.sha256_file(output / "learner_contract.json")
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["source_seals"]["learner"]["inventory_sha256"], learner_sha)
            self.assertEqual(report["source_seals"]["evaluation"]["inventory_sha256"], evaluation_sha)
            self.assertEqual(report["bootstrap"]["code_sha256"], evaluation_sha)
            self.assertEqual(report["bootstrap"]["policy_config_sha256"], learner_contract_sha)
            self.assertTrue((output / "fresh_lepton_campaign_materialization.json").is_file())
            self.assertFalse(any("monitor" in item.name or "telemetry" in item.name for item in output.rglob("*")))

    def test_existing_even_empty_campaign_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            protected = parent / "old-oci"
            protected.mkdir()
            output = parent / "already-exists"
            output.mkdir()
            with self.assertRaisesRegex(
                fresh.FreshLeptonCampaignError, "already exists"
            ):
                fresh.materialize_fresh_lepton_campaign(
                    campaign_root=output,
                    protected_oci_campaign_root=protected,
                    invariant_profile_path=parent / "missing",
                    invariant_profile_sha256="0" * 64,
                    runtime_contract_path=parent / "missing",
                    runtime_contract_sha256="0" * 64,
                    capture_aggregate_path=parent / "missing",
                    capture_aggregate_sha256="0" * 64,
                    heldout_plan_path=parent / "missing",
                    heldout_plan_sha256="0" * 64,
                    motion_split_path=parent / "missing",
                    motion_split_sha256="0" * 64,
                    scenarios_path=parent / "missing",
                    scenarios_sha256="0" * 64,
                    base_learner_contract_path=parent / "missing",
                    base_learner_contract_sha256="0" * 64,
                    learner_sealed_source_root="/users/ldu/projects/roborender/sealed/new",
                    learner_source_inventory_sha256="1" * 64,
                    evaluation_sealed_source_root=parent,
                    evaluation_source_inventory_sha256="2" * 64,
                )


if __name__ == "__main__":
    unittest.main()
