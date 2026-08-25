from __future__ import annotations

import json
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
# Independently cross-checked against prepare_gr00t_learner_bundle.prepare_bundle
# in the Torch-enabled learner environment for this deterministic fixture.
EXPECTED_DERIVED_BUNDLE_SHA = (
    "8ea3904e0b49d9123ec1259f3fbf1fec4acd21dba61032eb23a07e4de34a6dc3"
)


class ProductionLaunchTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="rr_production_launch_")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)

    def _campaign_and_orchestration(
        self,
        *,
        learner_code_sha: str = SOURCE_SHA,
        learner_config_sha: str | None = None,
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
            training_config_sha256="e29857c6e4b51bc2241c6e1101c0bff12204ca6dc8d81070169d6b8927fe7933",
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
                "argv": [str(ROOT / "scripts/round_dagger/run_production_paired_evaluation_hook.sh")],
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
            "evaluator_config_sha256": "e29857c6e4b51bc2241c6e1101c0bff12204ca6dc8d81070169d6b8927fe7933",
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
        self.assertEqual(
            campaign["wandb"]["netrc"],
            str(ROOT / "runtime/round_dagger_credentials/.netrc"),
        )
        self.assertEqual(learner["allocation"]["gpus_per_node"], 8)
        self.assertEqual(learner["allocation"]["global_batch_size"], 64)
        self.assertEqual(learner["optimization"]["optimizer_steps"], 500)
        self.assertEqual(learner["optimization"]["save_every_steps"], 5)
        self.assertEqual(learner["optimization"]["learning_rate"], 2.5e-5)
        self.assertEqual(learner["optimization"]["seed"], 17)
        self.assertEqual(learner["optimization"]["partition_seed"], 23)

    def test_validate_launch_binds_campaign_source_config_and_static_environment(self) -> None:
        _, orchestration = self._campaign_and_orchestration()
        source_report = {
            "schema": production_launch.SEALED_REPORT_SCHEMA,
            "status": "PASS",
            "root": "/lustre/fs1/portfolios/general/projects/general_cs_infra__ngcsre/users/ldu/roborender_h100_build/src_snapshots/fast_roborender_source",
            "inventory_sha256": SOURCE_SHA,
            "fast_roborender_revision": "1" * 40,
            "gr00t_revision": "2" * 40,
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
        with (
            mock.patch.object(production_launch, "_verify_static_inputs"),
            mock.patch.object(production_launch, "_verify_remote_source", return_value=source_report),
            mock.patch.object(
                production_launch,
                "_verify_local_source_checkout",
                return_value=local_report,
            ),
        ):
            environment, plan = production_launch.validate_launch(
                orchestration_config=orchestration,
                campaign_contract_path=CAMPAIGN_CONTRACT,
                learner_contract_path=LEARNER_CONTRACT,
                sealed_source_root=source_report["root"],
                sealed_source_inventory_sha256=SOURCE_SHA,
            )
        self.assertEqual(plan["status"], "validated_no_submission")
        self.assertEqual(plan["coordinator_phase"], "READY")
        self.assertFalse(plan["credentials_in_plan"])
        self.assertEqual(environment["ROUND_DAGGER_REMOTE_SOURCE_INVENTORY_SHA256"], SOURCE_SHA)
        self.assertEqual(environment["ROUND_DAGGER_LEARNER_MAX_STEPS"], "500")
        self.assertEqual(environment["ROUND_DAGGER_LEARNER_BATCH_SIZE"], "8")
        self.assertEqual(environment["ROUND_DAGGER_LEARNER_LEARNING_RATE"], "2.5e-5")
        self.assertEqual(environment["ROUND_DAGGER_WANDB_EXPECTED_EMAIL"], "ldu@nvidia.com")
        self.assertEqual(environment["ROUND_DAGGER_WANDB_ENTITY"], "nvidia")
        self.assertEqual(
            environment["ROUND_DAGGER_WANDB_NETRC"],
            str(ROOT / "runtime/round_dagger_credentials/.netrc"),
        )
        self.assertEqual(environment["IMAGE_DAGGER_TELEMETRY_REQUIRE_VIDEO"], "1")
        self.assertEqual(
            environment["ROUND_DAGGER_FAST_ROBORENDER_REVISION"], "1" * 40
        )
        self.assertEqual(environment["ROUND_DAGGER_GR00T_REVISION"], "2" * 40)
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
