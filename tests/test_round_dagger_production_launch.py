from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from round_dagger import artifacts
from round_dagger.orchestration import MacroRoundConfig
from round_dagger.promotion import stage_bootstrap_policy
from round_dagger.state import initialize_campaign, load_campaign, load_state
from scripts.round_dagger import production_launch
from tests.policy_bundle_helpers import publish_policy_bundle


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_CONTRACT = ROOT / "configs/round_dagger/production_campaign_256x300.json"
LEARNER_CONTRACT = ROOT / "configs/round_dagger/production_learner_8xh100_500.json"
SOURCE_SHA = "a" * 64
RUNTIME_SHA = "b" * 64
EVALUATION_CONFIG_SHA = (
    "bd3d61082149e3b87855ea3ab396e7cae7479d9bfe9d9d3aab99c839353ed7a2"
)
# Independently cross-checked against prepare_gr00t_learner_bundle.prepare_bundle
# in the Torch-enabled learner environment for this deterministic fixture.
EXPECTED_DERIVED_BUNDLE_SHA = (
    "6a46f66ea9ec1e8f9a936b35389631f14717ebd4406020eac25be80331144320"
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
                },
            },
            "evaluation": {
                "scenario_contract": scenarios.relative_to(root).as_posix(),
                "scenario_contract_sha256": artifacts.sha256_file(scenarios),
                "threshold_contract": thresholds.relative_to(root).as_posix(),
                "threshold_contract_sha256": artifacts.sha256_file(thresholds),
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
        learner_deployment_sha = artifacts.sha256_file(LEARNER_CONTRACT)
        learner_contract = production_launch.load_learner_contract(LEARNER_CONTRACT)
        root = self.base / "campaign"
        source = self.base / "bootstrap-source.pt"
        source.write_bytes(b"fresh-untrained-student")
        publish_policy_bundle(source, RUNTIME_SHA)
        bootstrap = stage_bootstrap_policy(
            root,
            checkpoint_source=source,
            code_sha256=SOURCE_SHA,
            policy_config_sha256=learner_deployment_sha,
        )
        initialize_campaign(
            root,
            campaign_id="roborender-dagger-256x300-v1",
            bootstrap_policy_sha256=bootstrap["checkpoint_sha256"],
            bootstrap_policy_path=bootstrap["checkpoint_path"],
            target_environments=256,
            active_environments=64,
            steps_per_environment=300,
            view_names=("head_color", "hand_left", "hand_right"),
            policy_image_shape=(324, 192, 3),
            action_dim=26,
            teacher_sha256="943a34fa8b6031dc2f8e3cd3b3424c641f8ba02d680ebb74743e22b31a2bbe38",
            renderer_sha256="a82f639a9038789db220793475dfce3eb5651ff3145287be748f387a20caeb9f",
            camera_mapping_sha256="53234dfe2eab9b1ae02df21e143f996ecde831aadb8aacb2e5f2df632d3d8d27",
            training_config_sha256=EVALUATION_CONFIG_SHA,
            beta_schedule={
                "kind": "linear",
                "start": 1.0,
                "end": 0.0,
                "decay_rounds": 10,
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
            campaign_id="roborender-dagger-256x300-v1",
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
            "schema": "fast_roborender.round_dagger.orchestration_config.v1",
            "campaign_root": str(root),
            "actor_root": str(root / "actors"),
            "macro_round": 0,
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
        self.assertEqual(campaign["topology"]["transitions_per_macro_round"], 76800)
        self.assertEqual(campaign["observation_contract"]["reference_lag_frames"], 256)
        self.assertEqual(campaign["action_contract"]["replay_action_dim"], 26)
        self.assertEqual(campaign["telemetry"]["clip_every_steps"], 300)
        self.assertEqual(campaign["wandb"]["expected_email"], "ldu@nvidia.com")
        self.assertEqual(campaign["wandb"]["entity"], "nvidia")
        thresholds = artifacts.load_strict_json(
            ROOT / campaign["evaluation"]["threshold_contract"]
        )
        self.assertEqual(thresholds["minimum_paired_cases"], 16)
        self.assertIs(thresholds["allow_bounded_nonregression"], False)
        self.assertEqual(
            campaign["wandb"]["netrc"],
            str(production_launch.WANDB_NETRC),
        )
        self.assertEqual(learner["allocation"]["gpus_per_node"], 8)
        self.assertEqual(learner["allocation"]["global_batch_size"], 64)
        self.assertEqual(learner["optimization"]["optimizer_steps"], 500)
        self.assertEqual(learner["optimization"]["save_every_steps"], 5)
        self.assertEqual(learner["optimization"]["learning_rate"], 2.5e-5)
        self.assertEqual(learner["optimization"]["seed"], 17)
        self.assertEqual(learner["optimization"]["partition_seed"], 23)
        self.assertEqual(learner["cluster"]["account"], "general_cs_infra__ngcsre")
        self.assertEqual(learner["cluster"]["partition"], "batch_singlenode")
        self.assertEqual(learner["cluster"]["qos"], "normal")

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
        campaign = json.loads(CAMPAIGN_CONTRACT.read_text(encoding="utf-8"))
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
        ):
            environment, plan = production_launch.validate_launch(
                orchestration_config=orchestration,
                campaign_contract_path=CAMPAIGN_CONTRACT,
                learner_contract_path=LEARNER_CONTRACT,
                sealed_source_root=source_report["root"],
                sealed_source_inventory_sha256=SOURCE_SHA,
                evaluation_sealed_source_root=local_sealed_root,
                evaluation_sealed_source_inventory_sha256=(
                    evaluation_inventory_sha
                ),
            )
        verify_local_seal.assert_called_once()
        self.assertEqual(plan["status"], "validated_no_submission")
        self.assertEqual(plan["coordinator_phase"], "READY")
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
        self.assertEqual(environment["ROUND_DAGGER_LEARNER_MAX_STEPS"], "500")
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
            "1079e2289a8afc93a8cdffd7ed060c1749c9a3067a27f0ba1213c0cd65148d96",
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
        deployment_sha = artifacts.sha256_file(LEARNER_CONTRACT)
        orchestration_contract = MacroRoundConfig.load(orchestration)
        self.assertEqual(
            plan["learner_deployment_contract_sha256"], deployment_sha
        )
        self.assertEqual(
            plan["derived_learner_bundle_sha256"],
            orchestration_contract.learner_config_sha256,
        )
        self.assertEqual(
            plan["derived_learner_bundle_sha256"], EXPECTED_DERIVED_BUNDLE_SHA
        )
        self.assertNotEqual(
            plan["learner_deployment_contract_sha256"],
            plan["derived_learner_bundle_sha256"],
        )
        self.assertEqual(load_campaign(self.base / "campaign")["policy_config_sha256"], deployment_sha)
        self.assertNotIn("learner_contract_sha256", plan)
        self.assertFalse(any("KEY" in name or "TOKEN" in name for name in environment))

    def test_static_deployment_contract_sha_is_rejected_as_round_bundle_sha(self) -> None:
        deployment_sha = artifacts.sha256_file(LEARNER_CONTRACT)
        _, orchestration = self._campaign_and_orchestration(
            learner_config_sha=deployment_sha
        )
        with self.assertRaisesRegex(
            production_launch.ProductionLaunchError,
            "derived learner bundle identity",
        ):
            production_launch._validate_orchestration_and_campaign(
                orchestration,
                production_launch.load_campaign_contract(CAMPAIGN_CONTRACT),
                production_launch.load_learner_contract(LEARNER_CONTRACT),
                deployment_sha,
                SOURCE_SHA,
                ROOT,
            )

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
                campaign_id="roborender-dagger-256x300-v1",
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

    def test_remote_source_verifier_binds_report_root_inventory_and_contract(self) -> None:
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
        responses = [
            mock.Mock(returncode=0, stdout=json.dumps(report), stderr=""),
            mock.Mock(
                returncode=0,
                stdout=f"{artifacts.sha256_file(LEARNER_CONTRACT)}  contract.json\n",
                stderr="",
            ),
        ]
        with (
            mock.patch.object(production_launch.shutil, "which", return_value="/usr/bin/ssh"),
            mock.patch.object(production_launch.subprocess, "run", side_effect=responses) as run,
        ):
            observed = production_launch._verify_remote_source(
                root,
                SOURCE_SHA,
                learner_contract=learner,
                learner_contract_path=LEARNER_CONTRACT,
            )
        self.assertEqual(observed["root"], root)
        self.assertEqual(run.call_count, 2)
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
                learner_contract_path=LEARNER_CONTRACT,
            )
        with self.assertRaisesRegex(production_launch.ProductionLaunchError, "accepted OCI source area"):
            production_launch._verify_remote_source(
                "/lustre/sealed/otherwise-safe-source",
                SOURCE_SHA,
                learner_contract=production_launch.load_learner_contract(LEARNER_CONTRACT),
                learner_contract_path=LEARNER_CONTRACT,
            )

    def test_shell_launcher_has_non_mutating_dry_run_and_no_secret_config(self) -> None:
        script = (ROOT / "scripts/round_dagger/launch_production_macro_round.sh").read_text()
        self.assertIn("PRODUCTION_MACRO_ROUND_DRY_RUN_OK", script)
        self.assertIn("--evaluation-sealed-source-root", script)
        self.assertIn("--evaluation-sealed-source-inventory-sha256", script)
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
