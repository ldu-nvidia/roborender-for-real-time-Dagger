from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from round_dagger import artifacts, replay
from round_dagger import production_data_lineage
from round_dagger.orchestration import MacroRoundConfig
from round_dagger.promotion import stage_bootstrap_policy
from round_dagger.state import (
    beta_for_macro_round,
    initialize_campaign,
    load_campaign,
    load_state,
)
from scripts.round_dagger import production_launch
from tests.policy_bundle_helpers import publish_policy_bundle


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_CONTRACT = ROOT / "configs/round_dagger/production_campaign_256x512.json"
LEARNER_CONTRACT = ROOT / "configs/round_dagger/production_learner_8xh100_1200.json"
SOURCE_SHA = "a" * 64
RUNTIME_SHA = "b" * 64
EVALUATION_CONFIG_SHA = (
    "bd3d61082149e3b87855ea3ab396e7cae7479d9bfe9d9d3aab99c839353ed7a2"
)
class ProductionLaunchTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="rr_production_launch_")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        scratch = Path(
            "/mnt/data2/ldu/roborender_round_dagger_validation/unit_tests"
        )
        scratch.mkdir(parents=True, exist_ok=True)
        local_temporary = tempfile.TemporaryDirectory(
            prefix="rr_local_seal_", dir=scratch
        )
        self.addCleanup(local_temporary.cleanup)
        self.local_base = Path(local_temporary.name)

    @staticmethod
    def _heldout_rollout_arrays(global_env_id: int) -> dict[str, np.ndarray]:
        expert = np.full((1, 1, 26), 0.25, dtype=np.float32)
        return {
            "policy_rgb": np.full(
                (1, 1, 324, 192, 3), global_env_id, dtype=np.uint8
            ),
            "expert_actions": expert,
            "student_actions": np.zeros_like(expert),
            "behavior_actions": expert.copy(),
            "expert_mask": np.ones((1, 1), dtype=np.bool_),
            "dones": np.zeros((1, 1), dtype=np.bool_),
            "observation_is_raytraced": np.ones((1, 1), dtype=np.bool_),
            "global_env_ids": np.asarray([global_env_id], dtype=np.int64),
            "episode_ids": np.asarray([[global_env_id]], dtype=np.int64),
            "episode_steps": np.zeros((1, 1), dtype=np.int64),
            "obs__actor_obs": np.zeros((1, 1, 113), dtype=np.float32),
        }

    def _materialize_campaign_learner_contract(
        self, campaign_root: Path
    ) -> Path:
        """Create a self-contained heldout-bound learner contract fixture."""

        campaign_root.mkdir(parents=True, exist_ok=True)
        cohort = campaign_root / "heldout" / "cohort_00"
        provenance = {
            "campaign_id": "dagger_prod_run1",
            "macro_round": 0,
            "cohort_index": 0,
            "purpose": "eval",
            "policy_sha256": "1" * 64,
            "runtime_policy_sha256": "2" * 64,
            "teacher_sha256": "3" * 64,
            "renderer_sha256": "4" * 64,
            "beta_sha256": "5" * 64,
            "config_sha256": "6" * 64,
        }
        artifacts.create_cohort(
            cohort,
            expected_ranks=[0],
            expected_global_env_ids=[0],
            expected_transitions=1,
            view_contract={
                "dtype": "uint8",
                "layout": "vertical",
                "shape": [324, 192, 3],
                "views": ["head_color", "hand_left", "hand_right"],
            },
            action_contract={"dtype": "float32", "shape": [26]},
            **provenance,
        )
        artifacts.publish_rollout_shard(
            cohort,
            rank=0,
            rollout_id=0,
            part_index=0,
            arrays=self._heldout_rollout_arrays(0),
            **provenance,
        )
        artifacts.seal_cohort(cohort)
        heldout_path = campaign_root / "heldout_snapshot.json"
        heldout = replay.build_heldout_snapshot([cohort], output=heldout_path)

        contract = artifacts.load_strict_json(LEARNER_CONTRACT)
        contract["heldout_evaluation"] = {
            "snapshot_path": str(heldout_path),
            "snapshot_file_sha256": heldout["file_sha256"],
            "snapshot_sha256": heldout["snapshot_sha256"],
            "interval_steps": 100,
        }
        contract_path = campaign_root / "learner_contract.json"
        artifacts.publish_immutable_json(contract_path, contract)
        return contract_path

    def _materialized_campaign_fixture(
        self,
        *,
        evaluation_source_root: Path = ROOT,
        evaluation_source_inventory_sha256: str = "d" * 64,
    ) -> tuple[dict, dict, Path, str]:
        """Return a schema-valid materialized profile for isolated launch tests.

        Full live-capture byte verification is covered by the scenario-capture
        and production-data-lineage suites.  These launch tests need the
        normalized, already-verified shape at the boundary so they can test
        durable orchestration binding without reviving the legacy contract.
        """

        runtime_path = self.base / "runtime-campaign-contract.json"
        if not runtime_path.exists():
            runtime_path.write_text(
                json.dumps({"fixture": "external-runtime-contract"}, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            runtime_path.chmod(0o444)
        runtime_sha = artifacts.sha256_file(runtime_path)
        profile = json.loads(CAMPAIGN_CONTRACT.read_text(encoding="utf-8"))
        scenario_path = ROOT / profile["evaluation"]["scenario_contract"]
        split_path = (
            evaluation_source_root
            / profile["motion_split"]["manifest_relative_path"]
        )
        heldout_plan_path = (
            evaluation_source_root
            / profile["motion_split"]["heldout_plan_relative_path"]
        )
        bundle_root = self.base / "first-condition-bundles"
        bundle_root.mkdir(exist_ok=True)
        first_condition_bundles = []
        for seed in (17, 29, 43, 59, 71, 89, 101, 127):
            bundle_path = bundle_root / f"seed_{seed}.npz"
            if not bundle_path.exists():
                bundle_path.write_bytes(f"first-condition:{seed}\n".encode())
                bundle_path.chmod(0o444)
            first_condition_bundles.append(
                {
                    "seed": seed,
                    "path": str(bundle_path),
                    "sha256": artifacts.sha256_file(bundle_path),
                    "size": bundle_path.stat().st_size,
                    "case_count": 32,
                    "source_pass_index": 1,
                }
            )
        first_condition_inventory_sha = artifacts.canonical_json_sha256(
            first_condition_bundles
        )
        provenance = {
            "fast_roborender_revision": "1" * 40,
            "gr00t_revision": "2" * 40,
            "actor_source_inventory_sha256": evaluation_source_inventory_sha256,
            "evaluation_source_inventory_sha256": (
                evaluation_source_inventory_sha256
            ),
            "gr00t_source_tree_sha256": "4" * 64,
            "camera_acceptance_sha256": profile["artifacts"][
                "camera_acceptance"
            ]["sha256"],
            "renderer_runtime_sha256": profile["artifacts"]["renderer"][
                "runtime_identity_sha256"
            ],
            "renderer_cohort_sha256": profile["artifacts"]["renderer"][
                "cohort_identity_sha256"
            ],
            "anchor_inventory_sha256": profile["artifacts"]["anchor_pool"][
                "inventory_sha256"
            ],
            "evaluation_config_sha256": profile["evaluation"][
                "hydra_config_sha256"
            ],
            "motion_split_manifest_sha256": profile["motion_split"][
                "manifest_sha256"
            ],
            "heldout_plan_sha256": profile["motion_split"][
                "heldout_plan_sha256"
            ],
        }
        runtime = {
            "schema": production_data_lineage.RUNTIME_CONTRACT_SCHEMA,
            "status": "release_eligible",
            "contract_path": str(runtime_path),
            "contract_sha256": runtime_sha,
            "invariant_profile_sha256": artifacts.sha256_file(CAMPAIGN_CONTRACT),
            "live_capture": {
                "split_manifest_path": str(split_path),
                "split_manifest_sha256": profile["motion_split"][
                    "manifest_sha256"
                ],
                "heldout_plan_path": str(heldout_plan_path),
                "heldout_plan_sha256": profile["motion_split"][
                    "heldout_plan_sha256"
                ],
                "attestation_path": str(self.base / "capture-attestation.json"),
                "attestation_sha256": "5" * 64,
                "evidence_path": str(self.base / "capture-evidence.json"),
                "evidence_sha256": "6" * 64,
                "scenario_path": str(scenario_path),
                "scenario_sha256": profile["evaluation"][
                    "scenario_contract_sha256"
                ],
                "first_condition_bundles": first_condition_bundles,
                "first_condition_bundle_inventory_sha256": (
                    first_condition_inventory_sha
                ),
            },
            "capture_provenance": provenance,
        }
        lineage = production_data_lineage.derive_data_lineage(
            runtime,
            motion_split_contract=profile["motion_split"],
            topology=profile["topology"],
        )
        profile["production_runtime_contract"] = runtime
        profile["data_lineage"] = lineage
        profile["data_lineage_sha256"] = (
            production_data_lineage.data_lineage_sha256(lineage)
        )
        profile["invariant_profile_sha256"] = artifacts.sha256_file(
            CAMPAIGN_CONTRACT
        )
        return profile, runtime, runtime_path, runtime_sha

    def _local_seal_fixture(
        self, name: str = "fast_roborender_deadbee_cafef00_sealed"
    ) -> tuple[Path, dict, dict, dict]:
        publication = self.local_base / f"{name}.publication"
        publication.mkdir()
        root = publication / name
        teacher = root / production_launch.TEACHER_CHECKPOINT_RELATIVE
        teacher.parent.mkdir(parents=True)
        teacher.write_bytes(b"sealed teacher")
        teacher.chmod(0o444)
        configured = self.local_base / f"configured-{name}.pt"
        configured.write_bytes(b"sealed teacher")
        teacher_sha = artifacts.sha256_file(teacher)
        scenarios = root / "configs/round_dagger/production_paired_scenarios_2x8.json"
        thresholds = root / "configs/round_dagger/production_paired_thresholds.json"
        scenarios.parent.mkdir(parents=True, exist_ok=True)
        scenarios.write_bytes(b"sealed scenarios")
        thresholds.write_bytes(b"sealed thresholds")
        split_manifest = root / "GR00T-VisualSim2Real/split.json"
        heldout_plan = root / "configs/round_dagger/heldout-plan.json"
        split_manifest.parent.mkdir(parents=True, exist_ok=True)
        split_manifest.write_bytes(b"sealed split")
        heldout_plan.write_bytes(b"sealed plan")
        campaign = {
            "artifacts": {
                "teacher": {
                    "checkpoint_path": str(configured),
                    "checkpoint_sha256": teacher_sha,
                },
                "anchor_pool": {
                    "path": "/mnt/data2/ldu/fixture-anchor-pool",
                    "anchor_count": 2048,
                    "inventory_sha256": "c" * 64,
                    "inventory_algorithm": (
                        "len64be(path_utf8)+path_utf8+size64be+sha256_binary"
                    ),
                    "usage": "compatibility_fallback_only",
                },
            },
            "evaluation": {
                "scenario_contract": scenarios.relative_to(root).as_posix(),
                "scenario_contract_sha256": artifacts.sha256_file(scenarios),
                "threshold_contract": thresholds.relative_to(root).as_posix(),
                "threshold_contract_sha256": artifacts.sha256_file(thresholds),
            },
            "motion_split": {
                "manifest_relative_path": split_manifest.relative_to(root).as_posix(),
                "manifest_sha256": artifacts.sha256_file(split_manifest),
                "heldout_plan_relative_path": heldout_plan.relative_to(root).as_posix(),
                "heldout_plan_sha256": artifacts.sha256_file(heldout_plan),
            },
        }
        sealed_campaign = root / production_launch.PRODUCTION_CAMPAIGN_CONTRACT_RELATIVE
        sealed_campaign.write_text(
            json.dumps(campaign, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        remote = {
            "root": f"/lustre/stage/src_snapshots/{name}",
            "snapshot_name": name,
            "inventory_sha256": SOURCE_SHA,
            "fast_roborender_revision": "1" * 40,
            "gr00t_revision": "2" * 40,
            "gr00t_source_tree_sha256": "4" * 64,
        }
        local = {
            "schema": production_launch.SEALED_REPORT_SCHEMA,
            "status": "PASS",
            "root": str(root),
            "snapshot_name": name,
            "inventory_sha256": SOURCE_SHA,
            "fast_roborender_revision": "1" * 40,
            "gr00t_revision": "2" * 40,
        }
        publication.chmod(0o555)
        self.addCleanup(publication.chmod, 0o755)
        return root, campaign, remote, local

    def _campaign_and_orchestration(
        self,
        *,
        learner_code_sha: str = SOURCE_SHA,
        learner_config_sha: str | None = None,
        evaluation_source_root: Path = ROOT,
    ) -> tuple[Path, Path]:
        campaign_contract, _, _, runtime_contract_sha = (
            self._materialized_campaign_fixture(
                evaluation_source_root=evaluation_source_root
            )
        )
        root = self.base / "campaign"
        campaign_learner_contract = self._materialize_campaign_learner_contract(
            root
        )
        self.campaign_learner_contract = campaign_learner_contract
        learner_deployment_sha = artifacts.sha256_file(
            campaign_learner_contract
        )
        learner_contract = production_launch.load_learner_contract(
            campaign_learner_contract
        )
        source = self.base / "bootstrap-source.pt"
        source.write_bytes(b"fresh-untrained-student")
        publish_policy_bundle(source, RUNTIME_SHA)
        bootstrap = stage_bootstrap_policy(
            root,
            checkpoint_source=source,
            code_sha256=SOURCE_SHA,
            policy_config_sha256=learner_deployment_sha,
        )
        self.assertEqual(
            artifacts.sha256_file(campaign_learner_contract),
            learner_deployment_sha,
        )
        initialize_campaign(
            root,
            campaign_id="dagger_prod_run1",
            bootstrap_policy_sha256=bootstrap["checkpoint_sha256"],
            bootstrap_policy_path=bootstrap["checkpoint_path"],
            target_environments=256,
            active_environments=64,
            steps_per_environment=512,
            view_names=("head_color", "hand_left", "hand_right"),
            policy_image_shape=(324, 192, 3),
            action_dim=26,
            teacher_sha256=campaign_contract["artifacts"]["teacher"]["checkpoint_sha256"],
            renderer_sha256=campaign_contract["artifacts"]["renderer"]["cohort_identity_sha256"],
            camera_mapping_sha256=campaign_contract["artifacts"]["camera_acceptance"]["sha256"],
            training_config_sha256=EVALUATION_CONFIG_SHA,
            macro_round_count=5,
            data_lineage=campaign_contract["data_lineage"],
            beta_schedule={
                "kind": "linear",
                "start": 1.0,
                "end": 0.0,
                "decay_rounds": 4,
                "rate": 1.0,
            },
        )
        active = load_state(root)["active_policy"]
        checkpoint = root / active["checkpoint_path"]
        derived_bundle_sha = production_launch.derive_learner_bundle_sha256(
            base_policy_spec=Path(f"{checkpoint}.policy_spec.json"),
            parent_checkpoint=checkpoint,
            expected_parent_checkpoint_sha256=active["checkpoint_sha256"],
            runtime_policy_sha256=active["runtime_policy_sha256"],
            campaign_id="dagger_prod_run1",
            macro_round=0,
            code_sha256=SOURCE_SHA,
            learner_contract=learner_contract,
        )
        hooks = {
            "collect": {
                "argv": [str(ROOT / "scripts/round_dagger/run_production_actor_cohort_hook.sh")],
                "timeout_seconds": 14400,
            },
            "transfer": None,
            "ingest": None,
            "train": {
                "argv": [str(ROOT / "scripts/round_dagger/submit_oci_nrt_round_learner_and_wait.sh")],
                "timeout_seconds": 172800,
            },
            "evaluate": {
                "argv": [str(evaluation_source_root / "scripts/round_dagger/run_production_paired_evaluation_hook.sh")],
                "timeout_seconds": 28800,
            },
            "promote": None,
        }
        orchestration = {
            "schema": "fast_roborender.round_dagger.orchestration_config.v3",
            "campaign_root": str(root),
            "actor_root": str(root / "actors"),
            "macro_round": 0,
            "macro_round_count": 5,
            "campaign_contract_sha256": runtime_contract_sha,
            "runtime_policy_sha256": active["runtime_policy_sha256"],
            "learner_code_sha256": learner_code_sha,
            "learner_config_sha256": learner_config_sha or derived_bundle_sha,
            "learner_model_lineage_sha256": active["checkpoint_sha256"],
            "evaluator_config_sha256": EVALUATION_CONFIG_SHA,
            "hooks": hooks,
        }
        config_path = self.base / "macro-round-000000.json"
        config_path.write_text(json.dumps(orchestration, indent=2, sort_keys=True) + "\n")
        config_path.chmod(0o444)
        return root, config_path

    def _git(self, repository: Path, *arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    def _clean_source_fixture(self, name: str) -> tuple[Path, dict[str, str]]:
        root = self.base / name
        nested = root / "GR00T-VisualSim2Real"
        nested.mkdir(parents=True)
        self._git(root, "init", "--quiet")
        self._git(root, "config", "user.email", "tests@example.invalid")
        self._git(root, "config", "user.name", "Production Launch Test")
        (root / ".gitignore").write_text(
            "/GR00T-VisualSim2Real/\n/runtime/\n", encoding="utf-8"
        )
        (root / "actor.py").write_text("ACTOR = 1\n", encoding="utf-8")
        self._git(root, "add", ".gitignore", "actor.py")
        self._git(root, "commit", "--quiet", "-m", "root source")

        self._git(nested, "init", "--quiet")
        self._git(nested, "config", "user.email", "tests@example.invalid")
        self._git(nested, "config", "user.name", "Production Launch Test")
        (nested / "policy.py").write_text("POLICY = 1\n", encoding="utf-8")
        self._git(nested, "add", "policy.py")
        self._git(nested, "commit", "--quiet", "-m", "nested source")

        root_revision = self._git(root, "rev-parse", "HEAD")
        nested_revision = self._git(nested, "rev-parse", "HEAD")
        (root / "-").write_text("non-source sentinel\n", encoding="utf-8")
        (root / ".source_revision").write_text("0" * 40 + "\n", encoding="ascii")
        (nested / ".source_revision").write_text(
            "0" * 40 + "\n", encoding="ascii"
        )
        # Ignored runtime/cache/credential state must not become a source
        # blocker or enter the allowed-untracked accounting.
        ignored = root / "runtime/cache"
        ignored.mkdir(parents=True)
        (ignored / "credential-cache").write_text("ignored\n", encoding="utf-8")
        return root, {
            "fast_roborender_revision": root_revision,
            "gr00t_revision": nested_revision,
        }

    def test_checked_in_contracts_pin_exact_production_topology_and_learner(self) -> None:
        campaign = production_launch.load_campaign_contract(CAMPAIGN_CONTRACT)
        learner = production_launch.load_learner_contract(LEARNER_CONTRACT)
        self.assertEqual(campaign["topology"]["transitions_per_macro_round"], 131072)
        self.assertEqual(
            campaign["schema"],
            "fast_roborender.round_dagger.production_campaign_contract.v2",
        )
        self.assertEqual(campaign["motion_split"]["heldout_motion_count"], 256)
        self.assertEqual(campaign["motion_split"]["collection_motion_count"], 39)
        self.assertEqual(
            campaign["artifacts"]["bootstrap_bundle"]["checkpoint_sha256"],
            "720a975812f6083192234d650033511c821b6902da2ef7734d87943f6c0a359a",
        )
        self.assertEqual(
            campaign["artifacts"]["bootstrap_bundle"]["runtime_policy_sha256"],
            "cd6f70aa9d589e810b7ab22999941930742a94621870457aadd068d2394deb9a",
        )
        self.assertEqual(campaign["topology"]["macro_round_count"], 5)
        self.assertEqual(campaign["topology"]["transitions_per_campaign"], 655360)
        self.assertEqual(campaign["topology"]["steps_per_environment"], 512)
        self.assertEqual(campaign["topology"]["transitions_per_cohort"], 32768)
        self.assertEqual(
            [
                beta_for_macro_round(campaign["beta_schedule"], macro_round)
                for macro_round in range(5)
            ],
            [1.0, 0.75, 0.5, 0.25, 0.0],
        )
        self.assertEqual(campaign["observation_contract"]["reference_lag_frames"], 256)
        self.assertEqual(
            campaign["observation_contract"]["bootstrap_anchor_source"],
            "simulator_rgb",
        )
        self.assertEqual(
            campaign["artifacts"]["anchor_pool"]["usage"],
            "compatibility_fallback_only",
        )
        self.assertEqual(campaign["action_contract"]["replay_action_dim"], 26)
        self.assertEqual(campaign["telemetry"]["clip_every_steps"], 300)
        self.assertEqual(campaign["wandb"]["expected_email"], "ldu@nvidia.com")
        self.assertEqual(campaign["wandb"]["entity"], "nvidia")
        self.assertEqual(
            campaign["artifacts"]["teacher"]["checkpoint_path"],
            str(ROOT / production_launch.TEACHER_CHECKPOINT_RELATIVE),
        )
        self.assertEqual(
            campaign["actor_runtime"]["gr00t_root"],
            str(ROOT / "GR00T-VisualSim2Real"),
        )
        self.assertEqual(
            campaign["actor_runtime"]["pair_executor"],
            str(ROOT / "scripts/round_dagger/run_actor_rank_docker.sh"),
        )
        self.assertEqual(
            campaign["actor_runtime"]["pair_rank_script"],
            str(ROOT / "scripts/round_dagger/run_actor_cohort_with_wandb.sh"),
        )
        thresholds = artifacts.load_strict_json(
            ROOT / campaign["evaluation"]["threshold_contract"]
        )
        self.assertEqual(thresholds["minimum_paired_cases"], 256)
        self.assertIs(thresholds["allow_bounded_nonregression"], False)
        self.assertEqual(
            campaign["wandb"]["netrc"],
            str(production_launch.WANDB_NETRC),
        )
        self.assertEqual(learner["allocation"]["gpus_per_node"], 8)
        self.assertEqual(learner["allocation"]["global_batch_size"], 64)
        self.assertEqual(learner["optimization"]["optimizer_steps"], 1200)
        self.assertEqual(learner["optimization"]["save_every_steps"], 5)
        self.assertEqual(learner["optimization"]["learning_rate"], 2.5e-5)
        self.assertEqual(learner["optimization"]["seed"], 17)
        self.assertEqual(learner["optimization"]["partition_seed"], 23)
        self.assertEqual(learner["cluster"]["account"], "general_cs_infra__ngcsre")
        self.assertEqual(learner["cluster"]["partition"], "batch_singlenode")
        self.assertEqual(learner["cluster"]["qos"], "normal")

    def test_campaign_owned_learner_contract_is_canonical_immutable_and_hashed(
        self,
    ) -> None:
        campaign_root = self.base / "contract-campaign"
        contract_path = self._materialize_campaign_learner_contract(
            campaign_root
        )
        expected_sha = artifacts.sha256_file(contract_path)
        loaded, observed_sha = production_launch.load_campaign_owned_learner_contract(
            contract_path,
            campaign_root=campaign_root,
            expected_sha256=expected_sha,
        )
        self.assertEqual(observed_sha, expected_sha)
        self.assertEqual(
            loaded, production_launch.load_learner_contract(contract_path)
        )

        alternate = campaign_root / "alternate.json"
        alternate.write_bytes(contract_path.read_bytes())
        alternate.chmod(0o444)
        with self.assertRaisesRegex(
            production_launch.ProductionLaunchError, "canonical campaign-owned"
        ):
            production_launch.load_campaign_owned_learner_contract(
                alternate,
                campaign_root=campaign_root,
                expected_sha256=expected_sha,
            )

        contract_path.chmod(0o400)
        with self.assertRaisesRegex(
            production_launch.ProductionLaunchError, "exactly 0444"
        ):
            production_launch.load_campaign_owned_learner_contract(
                contract_path,
                campaign_root=campaign_root,
                expected_sha256=expected_sha,
            )
        contract_path.chmod(0o444)
        with self.assertRaisesRegex(
            production_launch.ProductionLaunchError, "SHA-256 mismatch"
        ):
            production_launch.load_campaign_owned_learner_contract(
                contract_path,
                campaign_root=campaign_root,
                expected_sha256="f" * 64,
            )

        unbound_root = self.base / "unbound-contract-campaign"
        unbound_root.mkdir()
        unbound_path = unbound_root / "learner_contract.json"
        unbound = artifacts.load_strict_json(contract_path)
        unbound.pop("heldout_evaluation")
        artifacts.publish_immutable_json(unbound_path, unbound)
        with self.assertRaisesRegex(
            production_launch.ProductionLaunchError,
            "must bind fixed-256 held-out evaluation",
        ):
            production_launch.load_campaign_owned_learner_contract(
                unbound_path,
                campaign_root=unbound_root,
                expected_sha256=artifacts.sha256_file(unbound_path),
            )

    def test_source_owned_paths_rebind_to_the_executing_repository(self) -> None:
        raw = artifacts.load_strict_json(CAMPAIGN_CONTRACT)
        source_paths = (
            (
                raw["artifacts"]["teacher"]["checkpoint_path"],
                production_launch.TEACHER_CHECKPOINT_RELATIVE,
            ),
            (
                raw["actor_runtime"]["gr00t_root"],
                production_launch.ACTOR_GR00T_ROOT_RELATIVE,
            ),
            (
                raw["actor_runtime"]["pair_executor"],
                production_launch.ACTOR_PAIR_EXECUTOR_RELATIVE,
            ),
            (
                raw["actor_runtime"]["pair_rank_script"],
                production_launch.ACTOR_PAIR_RANK_SCRIPT_RELATIVE,
            ),
        )
        roots = {
            Path(value).parents[len(relative.parts) - 1]
            for value, relative in source_paths
        }
        self.assertEqual(len(roots), 1)

        relocated = self.base / "relocated-fast-roborender"
        with mock.patch.object(
            production_launch, "REPOSITORY_ROOT", relocated
        ):
            materialized = production_launch.load_campaign_contract(
                CAMPAIGN_CONTRACT
            )
        self.assertEqual(
            materialized["artifacts"]["teacher"]["checkpoint_path"],
            str(relocated / production_launch.TEACHER_CHECKPOINT_RELATIVE),
        )
        self.assertEqual(
            materialized["actor_runtime"]["gr00t_root"],
            str(relocated / production_launch.ACTOR_GR00T_ROOT_RELATIVE),
        )
        self.assertEqual(
            materialized["actor_runtime"]["pair_executor"],
            str(relocated / production_launch.ACTOR_PAIR_EXECUTOR_RELATIVE),
        )
        self.assertEqual(
            materialized["actor_runtime"]["pair_rank_script"],
            str(relocated / production_launch.ACTOR_PAIR_RANK_SCRIPT_RELATIVE),
        )

    def test_evaluation_capacity_gate_rejects_legacy_sixteen_case_contract(self) -> None:
        scenarios = self.base / "scenarios.json"
        thresholds = self.base / "thresholds.json"
        scenarios.write_text(
            json.dumps([{} for _ in range(16)]),
            encoding="utf-8",
        )
        thresholds.write_text(
            json.dumps({"minimum_paired_cases": 256}), encoding="utf-8"
        )
        with self.assertRaisesRegex(
            production_launch.ProductionLaunchError,
            "contains 16 cases but production requires 256",
        ):
            production_launch._require_evaluation_case_capacity(
                scenarios, thresholds
            )

    def test_evaluation_capacity_gate_accepts_canonical_256_case_list(self) -> None:
        scenarios = self.base / "scenarios.json"
        thresholds = self.base / "thresholds.json"
        scenarios.write_text(
            json.dumps([{} for _ in range(256)]), encoding="utf-8"
        )
        thresholds.write_text(
            json.dumps({"minimum_paired_cases": 256}), encoding="utf-8"
        )
        production_launch._require_evaluation_case_capacity(
            scenarios, thresholds
        )

    def test_camera_acceptance_is_bound_to_both_nested_source_trees(self) -> None:
        acceptance = self.base / "camera-acceptance.json"
        accepted_revision = "1" * 40
        accepted_tree = "2" * 64
        acceptance.write_text(
            json.dumps(
                {
                    "gr00t_revision": accepted_revision,
                    "gr00t_source_tree_sha256": accepted_tree,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        campaign = {
            "artifacts": {
                "camera_acceptance": {
                    "path": str(acceptance),
                    "sha256": artifacts.sha256_file(acceptance),
                }
            }
        }
        report = {
            "gr00t_revision": accepted_revision,
            "gr00t_source_tree_sha256": accepted_tree,
        }
        production_launch._verify_camera_acceptance_source_bindings(
            campaign,
            learner_source_report=report,
            evaluation_source_report=report,
        )

        for label, field, value in (
            ("revision", "gr00t_revision", "3" * 40),
            ("tree", "gr00t_source_tree_sha256", "4" * 64),
        ):
            with self.subTest(label=label):
                changed = dict(report)
                changed[field] = value
                with self.assertRaises(production_launch.ProductionLaunchError):
                    production_launch._verify_camera_acceptance_source_bindings(
                        campaign,
                        learner_source_report=changed,
                        evaluation_source_report=report,
                    )

    def test_build_environment_forces_legacy_runtime_adapter(self) -> None:
        # Read the checked-in documents directly so this unit gate remains
        # runnable from a clean isolated worktree whose absolute path differs
        # from the canonical production checkout pinned by the full validator.
        campaign, _, _, _ = self._materialized_campaign_fixture()
        learner = json.loads(LEARNER_CONTRACT.read_text(encoding="utf-8"))
        environment = production_launch.build_environment(
            campaign,
            learner,
            sealed_source_root="/remote/sealed/source",
            sealed_source_inventory_sha256="a" * 64,
            evaluation_source_root="/local/sealed/evaluation",
            evaluation_source_inventory_sha256="b" * 64,
            sealed_teacher_checkpoint="/local/sealed/evaluation/teacher.pt",
            learner_contract_sha256="c" * 64,
        )
        self.assertEqual(
            environment["ROUND_DAGGER_RUNTIME_ADAPTER_MODE"], "legacy"
        )
        self.assertEqual(
            environment["IMAGE_DAGGER_EXPECT_ANCHOR_SOURCE"], "simulator_rgb"
        )
        self.assertEqual(
            environment["ROUND_DAGGER_MOTION_SPLIT_MANIFEST_SHA256"],
            campaign["motion_split"]["manifest_sha256"],
        )
        self.assertEqual(
            environment["ROUND_DAGGER_HELDOUT_EVALUATION_PLAN_SHA256"],
            campaign["motion_split"]["heldout_plan_sha256"],
        )
        self.assertEqual(
            environment[
                "ROUND_DAGGER_FIRST_CONDITION_BUNDLE_INVENTORY_SHA256"
            ],
            campaign["production_runtime_contract"]["live_capture"][
                "first_condition_bundle_inventory_sha256"
            ],
        )
        self.assertNotIn("ROUND_DAGGER_RUNTIME_ADAPTER_SHADOW_ACK", environment)
        self.assertEqual(
            [
                name
                for name in environment
                if name.startswith("ROUND_DAGGER_RUNTIME_ADAPTER_")
                or name.startswith("ROUND_DAGGER_RUNTIME_AB_")
            ],
            ["ROUND_DAGGER_RUNTIME_ADAPTER_MODE"],
        )

    def test_validate_launch_binds_campaign_source_config_and_static_environment(self) -> None:
        evaluation_inventory_sha = "d" * 64
        local_sealed_root = Path(
            "/mnt/data2/ldu/roborender_source_snapshots/fast_roborender_source"
        )
        _, orchestration = self._campaign_and_orchestration(
            evaluation_source_root=local_sealed_root
        )
        materialized, runtime_contract, runtime_path, runtime_sha = (
            self._materialized_campaign_fixture(
                evaluation_source_root=local_sealed_root,
                evaluation_source_inventory_sha256=evaluation_inventory_sha,
            )
        )
        source_report = {
            "schema": production_launch.SEALED_REPORT_SCHEMA,
            "status": "PASS",
            "root": "/lustre/fs1/portfolios/general/projects/general_cs_infra__ngcsre/users/ldu/roborender_h100_build/src_snapshots/fast_roborender_source",
            "inventory_sha256": SOURCE_SHA,
            "fast_roborender_revision": "1" * 40,
            "gr00t_revision": "2" * 40,
            "gr00t_source_tree_sha256": "4" * 64,
        }
        local_report = {
            "schema": "fast_roborender.round_dagger.local_source_checkout.v1",
            "status": "clean_and_revision_matched",
            "fast_roborender": {
                "revision": source_report["fast_roborender_revision"],
                "index_tree": "3" * 40,
                "allowed_untracked": ["-", ".source_revision"],
            },
            "gr00t": {
                "revision": source_report["gr00t_revision"],
                "index_tree": "4" * 40,
                "allowed_untracked": [".source_revision", "core"],
            },
        }
        sealed_teacher = (
            local_sealed_root / production_launch.TEACHER_CHECKPOINT_RELATIVE
        )
        local_seal_report = {
            "root": str(local_sealed_root),
            "sealed_teacher_checkpoint": str(sealed_teacher),
            "sealed_teacher_sha256": "943a34fa8b6031dc2f8e3cd3b3424c641f8ba02d680ebb74743e22b31a2bbe38",
        }
        with (
            mock.patch.object(production_launch, "_verify_static_inputs"),
            mock.patch.object(production_launch, "_verify_remote_source", return_value=source_report),
            mock.patch.object(
                production_launch,
                "_verify_local_evaluation_seal",
                return_value=local_seal_report,
            ) as verify_local_seal,
            mock.patch.object(
                production_launch,
                "_verify_local_source_checkout",
                return_value=local_report,
            ),
            mock.patch.object(
                production_launch,
                "_verify_camera_acceptance_source_bindings",
            ),
            mock.patch.object(
                production_launch,
                "materialize_campaign_contract",
                return_value=(
                    materialized,
                    runtime_contract,
                    materialized["data_lineage"],
                ),
            ),
        ):
            environment, plan = production_launch.validate_launch(
                orchestration_config=orchestration,
                campaign_contract_path=runtime_path,
                campaign_contract_sha256=runtime_sha,
                learner_contract_path=(
                    self.base / "campaign" / "learner_contract.json"
                ),
                learner_contract_sha256=artifacts.sha256_file(
                    self.campaign_learner_contract
                ),
                sealed_source_root=source_report["root"],
                sealed_source_inventory_sha256=SOURCE_SHA,
                evaluation_sealed_source_root=local_sealed_root,
                evaluation_sealed_source_inventory_sha256=(
                    evaluation_inventory_sha
                ),
            )
        verify_local_seal.assert_called_once()
        self.assertEqual(plan["status"], "validated_no_submission")
        self.assertEqual(
            plan["schema"],
            "fast_roborender.round_dagger.production_launch_plan.v2",
        )
        self.assertEqual(plan["coordinator_phase"], "READY")
        self.assertEqual(plan["macro_round_count"], 5)
        self.assertEqual(plan["completed_macro_rounds"], 0)
        self.assertEqual(plan["remaining_macro_rounds"], 5)
        self.assertFalse(plan["credentials_in_plan"])
        self.assertEqual(environment["ROUND_DAGGER_REMOTE_SOURCE_INVENTORY_SHA256"], SOURCE_SHA)
        self.assertEqual(
            environment["ROUND_DAGGER_EVALUATION_SEALED_TEACHER"],
            str(sealed_teacher),
        )
        self.assertEqual(
            plan["evaluation_sealed_source_root"], str(local_sealed_root)
        )
        self.assertEqual(plan["learner_source_inventory_sha256"], SOURCE_SHA)
        self.assertEqual(
            plan["evaluation_source_inventory_sha256"],
            evaluation_inventory_sha,
        )
        self.assertNotEqual(
            plan["learner_source_inventory_sha256"],
            plan["evaluation_source_inventory_sha256"],
        )
        self.assertEqual(environment["ROUND_DAGGER_LEARNER_MAX_STEPS"], "1200")
        self.assertEqual(environment["ROUND_DAGGER_MACRO_ROUND_COUNT"], "5")
        self.assertEqual(
            environment["ROUND_DAGGER_TRANSITIONS_PER_CAMPAIGN"], "655360"
        )
        self.assertEqual(
            environment["ROUND_DAGGER_MOTION_SPLIT_MANIFEST_SHA256"],
            "6a600b4dbdb7cb076d203d2440ad41d14acf4f1cad8a9417a76b51dc62ed4834",
        )
        self.assertEqual(
            environment["ROUND_DAGGER_HELDOUT_EVALUATION_PLAN_SHA256"],
            "394ba77b2c8ecbdb4904ef24894cda0b88bb5b606ab75e578ee0a91756630f19",
        )
        self.assertEqual(
            environment["IMAGE_DAGGER_EXPECT_ANCHOR_SOURCE"], "simulator_rgb"
        )
        self.assertEqual(plan["bootstrap_anchor_source"], "simulator_rgb")
        self.assertEqual(environment["ROUND_DAGGER_LEARNER_BATCH_SIZE"], "8")
        self.assertEqual(environment["ROUND_DAGGER_LEARNER_LEARNING_RATE"], "2.5e-5")
        self.assertEqual(environment["ROUND_DAGGER_WANDB_EXPECTED_EMAIL"], "ldu@nvidia.com")
        self.assertEqual(environment["ROUND_DAGGER_WANDB_ENTITY"], "nvidia")
        self.assertEqual(
            environment["ROUND_DAGGER_EVALUATION_SCENARIOS_SHA256"],
            "99081ebee907ef36c388cf62a1d134c015d289776526cacc1db114a87074e724",
        )
        self.assertEqual(
            environment["ROUND_DAGGER_EVALUATION_THRESHOLDS_SHA256"],
            "05d9fbb64c41311e586f2d14f889902a3ea7fe56e31bd8b16cd539a65c035563",
        )
        self.assertEqual(
            environment["ROUND_DAGGER_WANDB_NETRC"],
            str(production_launch.WANDB_NETRC),
        )
        self.assertEqual(environment["IMAGE_DAGGER_TELEMETRY_REQUIRE_VIDEO"], "1")
        self.assertEqual(
            environment["ROUND_DAGGER_RUNTIME_ADAPTER_MODE"], "legacy"
        )
        self.assertNotIn("ROUND_DAGGER_RUNTIME_ADAPTER_SHADOW_ACK", environment)
        self.assertEqual(
            environment["ROUND_DAGGER_FAST_ROBORENDER_REVISION"], "1" * 40
        )
        self.assertEqual(environment["ROUND_DAGGER_GR00T_REVISION"], "2" * 40)
        self.assertEqual(
            environment["ROUND_DAGGER_GR00T_SOURCE_TREE_SHA256"], "4" * 64
        )
        self.assertEqual(
            environment["ROUND_DAGGER_CONTROL_PYTHON"],
            "/mnt/data2/ldu/lacwm_runtime/envs/lacwm-b200-py310/bin/python",
        )
        self.assertEqual(
            environment["ROUND_DAGGER_LAUNCH_ORCHESTRATION_SHA256"],
            artifacts.sha256_file(orchestration),
        )
        deployment_sha = artifacts.sha256_file(
            self.campaign_learner_contract
        )
        orchestration_contract = MacroRoundConfig.load(orchestration)
        self.assertEqual(
            plan["learner_deployment_contract_sha256"], deployment_sha
        )
        self.assertEqual(
            plan["learner_contract_path"],
            str(self.base / "campaign" / "learner_contract.json"),
        )
        self.assertEqual(
            plan["learner_contract_projection_relative_path"],
            "learner_contract.json",
        )
        self.assertEqual(
            environment["ROUND_DAGGER_LEARNER_CONTRACT"],
            str(self.base / "campaign" / "learner_contract.json"),
        )
        self.assertEqual(
            plan["derived_learner_bundle_sha256"],
            orchestration_contract.learner_config_sha256,
        )
        self.assertNotEqual(
            plan["learner_deployment_contract_sha256"],
            plan["derived_learner_bundle_sha256"],
        )
        self.assertEqual(load_campaign(self.base / "campaign")["policy_config_sha256"], deployment_sha)
        self.assertNotIn("learner_contract_sha256", plan)
        self.assertFalse(any("KEY" in name or "TOKEN" in name for name in environment))

    def test_first_condition_bundle_inventory_is_rehashed_at_launch_boundary(self) -> None:
        campaign, _, _, _ = self._materialized_campaign_fixture()
        rows = production_launch._verify_first_condition_bundles(campaign)
        self.assertEqual([row["seed"] for row in rows], [17, 29, 43, 59, 71, 89, 101, 127])

        bundle = Path(rows[0]["path"])
        bundle.chmod(0o644)
        bundle.write_bytes(b"mutated first condition\n")
        bundle.chmod(0o444)
        with self.assertRaisesRegex(
            production_launch.ProductionLaunchError,
            "size differs|bytes differ",
        ):
            production_launch._verify_first_condition_bundles(campaign)

    def test_static_deployment_contract_sha_is_rejected_as_round_bundle_sha(self) -> None:
        _, orchestration = self._campaign_and_orchestration()
        deployment_sha = artifacts.sha256_file(
            self.campaign_learner_contract
        )
        orchestration_document = artifacts.load_strict_json(orchestration)
        orchestration_document["learner_config_sha256"] = deployment_sha
        orchestration.chmod(0o644)
        orchestration.write_text(
            json.dumps(orchestration_document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        orchestration.chmod(0o444)
        materialized, _, _, _ = self._materialized_campaign_fixture()
        with self.assertRaisesRegex(
            production_launch.ProductionLaunchError,
            "derived learner bundle identity",
        ):
            production_launch._validate_orchestration_and_campaign(
                orchestration,
                materialized,
                production_launch.load_learner_contract(
                    self.campaign_learner_contract
                ),
                deployment_sha,
                SOURCE_SHA,
                ROOT,
            )

    def test_production_rejects_legacy_unbounded_orchestration_config(self) -> None:
        _, orchestration = self._campaign_and_orchestration()
        document = json.loads(orchestration.read_text(encoding="utf-8"))
        document["schema"] = "fast_roborender.round_dagger.orchestration_config.v1"
        document.pop("macro_round_count")
        document.pop("campaign_contract_sha256")
        legacy = self.base / "legacy-unbounded-orchestration.json"
        legacy.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        legacy.chmod(0o444)
        with self.assertRaisesRegex(
            production_launch.ProductionLaunchError,
            "bind exactly five macro-rounds",
        ):
            production_launch._validate_orchestration_and_campaign(
                legacy,
                production_launch.load_campaign_contract(CAMPAIGN_CONTRACT),
                production_launch.load_learner_contract(LEARNER_CONTRACT),
                artifacts.sha256_file(LEARNER_CONTRACT),
                SOURCE_SHA,
                ROOT,
            )

    def test_v1_campaign_relabel_cannot_carry_v2_motion_split_shape(self) -> None:
        document = json.loads(CAMPAIGN_CONTRACT.read_text(encoding="utf-8"))
        document["schema"] = (
            "fast_roborender.round_dagger.production_campaign_contract.v1"
        )
        path = self.base / "v1-relabeled-campaign.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(
            production_launch.ProductionLaunchError, "schema mismatch"
        ):
            production_launch.load_campaign_contract(path)

    def test_active_policy_spec_identity_cannot_be_rewritten_during_preflight(
        self,
    ) -> None:
        root, _ = self._campaign_and_orchestration()
        active = load_state(root)["active_policy"]
        checkpoint = root / active["checkpoint_path"]
        with self.assertRaisesRegex(
            production_launch.ProductionLaunchError,
            "active base policy spec/runtime identity",
        ):
            production_launch.derive_learner_bundle_sha256(
                base_policy_spec=Path(f"{checkpoint}.policy_spec.json"),
                parent_checkpoint=checkpoint,
                expected_parent_checkpoint_sha256=active["checkpoint_sha256"],
                runtime_policy_sha256="f" * 64,
                campaign_id="dagger_prod_run1",
                macro_round=0,
                code_sha256=SOURCE_SHA,
                learner_contract=production_launch.load_learner_contract(
                    LEARNER_CONTRACT
                ),
            )

    def test_wrong_sealed_source_identity_fails_before_orchestration(self) -> None:
        _, orchestration = self._campaign_and_orchestration(learner_code_sha="c" * 64)
        with self.assertRaisesRegex(production_launch.ProductionLaunchError, "learner/source identity"):
            production_launch._validate_orchestration_and_campaign(
                orchestration,
                production_launch.load_campaign_contract(CAMPAIGN_CONTRACT),
                production_launch.load_learner_contract(LEARNER_CONTRACT),
                artifacts.sha256_file(LEARNER_CONTRACT),
                SOURCE_SHA,
                ROOT,
            )

    def test_remote_source_verifier_binds_report_root_and_inventory(self) -> None:
        learner = production_launch.load_learner_contract(LEARNER_CONTRACT)
        root = (
            learner["cluster"]["stage_root"]
            + "/src_snapshots/fast_roborender_deadbee_cafef00_sealed"
        )
        report = {
            "schema": production_launch.SEALED_REPORT_SCHEMA,
            "status": "PASS",
            "root": root,
            "inventory_sha256": SOURCE_SHA,
            "fast_roborender_revision": "1" * 40,
            "gr00t_revision": "2" * 40,
        }
        with (
            mock.patch.object(production_launch.shutil, "which", return_value="/usr/bin/ssh"),
            mock.patch.object(
                production_launch.subprocess,
                "run",
                return_value=mock.Mock(
                    returncode=0, stdout=json.dumps(report), stderr=""
                ),
            ) as run,
        ):
            observed = production_launch._verify_remote_source(
                root,
                SOURCE_SHA,
                learner_contract=learner,
            )
        self.assertEqual(observed["root"], root)
        self.assertEqual(run.call_count, 1)
        flattened = " ".join(str(argument) for call in run.call_args_list for argument in call.args[0])
        self.assertNotIn("wandb_v1_", flattened)
        self.assertIn(root, flattened)

    def test_local_seal_derives_inventory_bound_teacher_for_rtx(self) -> None:
        root, campaign, remote, local = self._local_seal_fixture()
        with mock.patch.object(
            production_launch,
            "verify_sealed_source_snapshot",
            return_value=local,
        ) as verifier:
            observed = production_launch._verify_local_evaluation_seal(
                root,
                SOURCE_SHA,
                campaign_contract=campaign,
            )
        verifier.assert_called_once_with(root, SOURCE_SHA)
        self.assertEqual(observed["root"], str(root))
        self.assertEqual(
            observed["sealed_teacher_checkpoint"],
            str(root / production_launch.TEACHER_CHECKPOINT_RELATIVE),
        )
        self.assertEqual(
            observed["sealed_teacher_sha256"],
            campaign["artifacts"]["teacher"]["checkpoint_sha256"],
        )

    def test_local_seal_rejects_report_root_snapshot_and_inventory_mismatch(self) -> None:
        for mismatch in ("root", "snapshot", "inventory", "publication_mode"):
            with self.subTest(mismatch=mismatch):
                root, campaign, remote, local = self._local_seal_fixture(
                    f"fast_roborender_{mismatch}_cafef00_sealed"
                )
                if mismatch == "root":
                    local["root"] += "_different"
                elif mismatch == "snapshot":
                    local["snapshot_name"] += "_different"
                else:
                    if mismatch == "inventory":
                        local["inventory_sha256"] = "b" * 64
                    else:
                        root.parent.chmod(0o755)
                with mock.patch.object(
                    production_launch,
                    "verify_sealed_source_snapshot",
                    return_value=local,
                ):
                    with self.assertRaises(production_launch.ProductionLaunchError):
                        production_launch._verify_local_evaluation_seal(
                            root,
                            SOURCE_SHA,
                            campaign_contract=campaign,
                        )

    def test_evaluation_seal_revision_is_independent_of_learner_revision(self) -> None:
        root, campaign, _, local = self._local_seal_fixture(
            "fast_roborender_independent_revision_cafef00_sealed"
        )
        local["fast_roborender_revision"] = "a" * 40
        local["gr00t_revision"] = "b" * 40
        with mock.patch.object(
            production_launch,
            "verify_sealed_source_snapshot",
            return_value=local,
        ):
            observed = production_launch._verify_local_evaluation_seal(
                root,
                SOURCE_SHA,
                campaign_contract=campaign,
            )
        self.assertEqual(observed["fast_roborender_revision"], "a" * 40)

    def test_local_sealed_teacher_mode_link_and_hash_fail_closed(self) -> None:
        for mismatch in ("mode", "hardlink", "hash"):
            with self.subTest(mismatch=mismatch):
                root, campaign, remote, local = self._local_seal_fixture(
                    f"fast_roborender_{mismatch}_deadbee_sealed"
                )
                teacher = root / production_launch.TEACHER_CHECKPOINT_RELATIVE
                if mismatch == "mode":
                    teacher.chmod(0o644)
                elif mismatch == "hardlink":
                    os.link(teacher, self.local_base / f"{mismatch}-alias.pt")
                else:
                    teacher.chmod(0o644)
                    teacher.write_bytes(b"wrong teacher")
                    teacher.chmod(0o444)
                with mock.patch.object(
                    production_launch,
                    "verify_sealed_source_snapshot",
                    return_value=local,
                ):
                    with self.assertRaises(production_launch.ProductionLaunchError):
                        production_launch._verify_local_evaluation_seal(
                            root,
                            SOURCE_SHA,
                            campaign_contract=campaign,
                        )

    def test_clean_local_root_and_nested_revisions_match_remote_snapshot(self) -> None:
        root, report = self._clean_source_fixture("clean-source")
        observed = production_launch._verify_local_source_checkout(
            report, repository_root=root
        )
        self.assertEqual(observed["status"], "clean_and_revision_matched")
        self.assertEqual(
            observed["fast_roborender"]["allowed_untracked"],
            ["-", ".source_revision"],
        )
        self.assertEqual(
            observed["gr00t"]["allowed_untracked"], [".source_revision"]
        )

    def test_local_tracked_drift_and_unexpected_untracked_source_fail_closed(self) -> None:
        root, report = self._clean_source_fixture("dirty-root")
        (root / "actor.py").write_text("ACTOR = 2\n", encoding="utf-8")
        with self.assertRaisesRegex(
            production_launch.ProductionLaunchError, "tracked worktree is dirty"
        ):
            production_launch._verify_local_source_checkout(
                report, repository_root=root
            )

        root, report = self._clean_source_fixture("dirty-nested")
        (root / "GR00T-VisualSim2Real" / "new_policy.py").write_text(
            "UNTRACKED = True\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(
            production_launch.ProductionLaunchError, "unexpected untracked GR00T source"
        ):
            production_launch._verify_local_source_checkout(
                report, repository_root=root
            )

    def test_local_staged_drift_revision_mismatch_and_unsafe_core_fail_closed(self) -> None:
        root, report = self._clean_source_fixture("staged-source")
        staged = root / "staged.py"
        staged.write_text("STAGED = True\n", encoding="utf-8")
        self._git(root, "add", "staged.py")
        with self.assertRaisesRegex(
            production_launch.ProductionLaunchError, "tracked index is dirty"
        ):
            production_launch._verify_local_source_checkout(
                report, repository_root=root
            )

        root, report = self._clean_source_fixture("wrong-revision")
        report["gr00t_revision"] = "f" * 40
        with self.assertRaisesRegex(
            production_launch.ProductionLaunchError, "local/remote GR00T revision"
        ):
            production_launch._verify_local_source_checkout(
                report, repository_root=root
            )

        root, report = self._clean_source_fixture("unsafe-core")
        (root / "GR00T-VisualSim2Real" / "core").write_bytes(b"not root owned")
        with self.assertRaisesRegex(
            production_launch.ProductionLaunchError, "root-owned"
        ):
            production_launch._verify_local_source_checkout(
                report, repository_root=root
            )

    def test_contract_mutation_and_unsafe_source_path_fail_closed(self) -> None:
        for label, mutate, expected_error in (
            (
                "teacher",
                lambda document: document["artifacts"]["teacher"].__setitem__(
                    "checkpoint_path", "/tmp/substituted-teacher.pt"
                ),
                "teacher checkpoint path repository path binding mismatch",
            ),
            (
                "actor",
                lambda document: document["actor_runtime"].__setitem__(
                    "pair_executor", "/tmp/substituted-actor.sh"
                ),
                "actor pair executor repository path binding mismatch",
            ),
        ):
            with self.subTest(source_owned_path=label):
                campaign = artifacts.load_strict_json(CAMPAIGN_CONTRACT)
                mutate(campaign)
                path = self.base / f"campaign-{label}.json"
                path.write_text(
                    json.dumps(campaign, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    production_launch.ProductionLaunchError, expected_error
                ):
                    production_launch.load_campaign_contract(path)

        learner = artifacts.load_strict_json(LEARNER_CONTRACT)
        learner["optimization"]["optimizer_steps"] = 499
        mutated = self.base / "learner.json"
        mutated.write_text(json.dumps(learner, indent=2, sort_keys=True) + "\n")
        with self.assertRaisesRegex(production_launch.ProductionLaunchError, "learner optimization mismatch"):
            production_launch.load_learner_contract(mutated)

        with self.assertRaisesRegex(production_launch.ProductionLaunchError, "unsafe"):
            production_launch._verify_remote_source(
                "/lustre/sealed/../other;touch-x",
                SOURCE_SHA,
                learner_contract=production_launch.load_learner_contract(LEARNER_CONTRACT),
            )
        with self.assertRaisesRegex(production_launch.ProductionLaunchError, "accepted OCI source area"):
            production_launch._verify_remote_source(
                "/lustre/sealed/otherwise-safe-source",
                SOURCE_SHA,
                learner_contract=production_launch.load_learner_contract(LEARNER_CONTRACT),
            )

    def test_source_owned_path_binding_rejects_mixed_roots_traversal_and_suffixes(
        self,
    ) -> None:
        cases = (
            (
                "mixed-roots",
                lambda document: document["artifacts"]["teacher"].__setitem__(
                    "checkpoint_path",
                    "/mnt/other/source/"
                    + production_launch.TEACHER_CHECKPOINT_RELATIVE.as_posix(),
                ),
                "do not share one canonical source root",
            ),
            (
                "traversal",
                lambda document: document["actor_runtime"].__setitem__(
                    "gr00t_root",
                    "/mnt/data2/ldu/source/../GR00T-VisualSim2Real",
                ),
                "must be a normalized absolute path",
            ),
            (
                "wrong-suffix",
                lambda document: document["actor_runtime"].__setitem__(
                    "pair_rank_script",
                    "/mnt/data2/ldu/source/scripts/round_dagger/not-the-rank-script.sh",
                ),
                "actor pair rank script repository path binding mismatch",
            ),
        )
        for name, mutate, expected_error in cases:
            with self.subTest(name=name):
                campaign = artifacts.load_strict_json(CAMPAIGN_CONTRACT)
                mutate(campaign)
                path = self.base / f"campaign-binding-{name}.json"
                path.write_text(
                    json.dumps(campaign, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    production_launch.ProductionLaunchError, expected_error
                ):
                    production_launch.load_campaign_contract(path)

    def test_shell_launcher_has_non_mutating_dry_run_and_no_secret_config(self) -> None:
        script = (ROOT / "scripts/round_dagger/launch_production_macro_round.sh").read_text()
        self.assertIn("PRODUCTION_MACRO_ROUND_DRY_RUN_OK", script)
        self.assertIn("--evaluation-sealed-source-root", script)
        self.assertIn("--evaluation-sealed-source-inventory-sha256", script)
        self.assertIn("--learner-contract", script)
        self.assertIn("--learner-contract-sha256", script)
        self.assertIn("canonical campaign-owned learner_contract.json", script)
        self.assertIn("learner contract mode must be exactly 0444", script)
        self.assertIn("learner contract SHA-256 mismatch", script)
        self.assertIn("unset WANDB_API_KEY", script)
        self.assertIn('exec "$PYTHON" "$ORCHESTRATOR"', script)
        self.assertIn('PYTHON_REAL=$(realpath -e -- "$PYTHON")', script)
        self.assertNotIn("PYTHON=$PYTHON_REAL", script)
        self.assertNotIn("wandb_v1_", script)
        for contract in (CAMPAIGN_CONTRACT, LEARNER_CONTRACT):
            text = contract.read_text()
            self.assertNotIn("wandb_v1_", text)
            self.assertNotIn("WANDB_API_KEY", text)


if __name__ == "__main__":
    unittest.main()
