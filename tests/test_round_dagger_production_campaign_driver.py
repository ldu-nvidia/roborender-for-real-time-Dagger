from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from round_dagger import artifacts
from round_dagger.production_campaign_driver import (
    DriverInputs,
    ExclusiveCampaignLock,
    ProductionDriverError,
    _launcher_argv,
    materialize_orchestration_config,
    run_driver,
)


SHA = "a" * 64
RUNTIME = "b" * 64
CHECKPOINT = "c" * 64
EVALUATOR = "d" * 64
ROOT = Path(__file__).resolve().parents[1]


def coordinator(round_index: int, phase: str) -> dict:
    return {
        "phase": phase,
        "next_macro_round": round_index,
        "open_macro_round": None if phase == "READY" else round_index,
        "active_policy": {
            "version": round_index,
            "checkpoint_sha256": CHECKPOINT,
            "checkpoint_path": f"policies/policy_{round_index:06d}/student.pt",
            "runtime_policy_sha256": RUNTIME,
        },
    }


class FiveRoundProductionDriverTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="rr-five-round-driver-")
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.campaign_root = self.base / "campaign"
        self.campaign_root.mkdir()
        self.control_root = self.base / "control"
        self.contract = self.base / "runtime-campaign.json"
        self.contract.write_text(
            json.dumps(
                {
                    "campaign_id": "campaign",
                    "topology": {"macro_round_count": 5},
                    "evaluation": {"hydra_config_sha256": EVALUATOR},
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        self.contract.chmod(0o444)
        self.learner = self.base / "learner.json"
        self.learner.write_text("{}\n", encoding="utf-8")
        self.learner.chmod(0o444)
        self.evaluation_source = self.base / "evaluation-source"
        self.evaluation_source.mkdir()
        self.launcher = self.base / "launcher.sh"
        self.launcher.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        self.launcher.chmod(0o555)
        self.inputs = DriverInputs(
            campaign_root=self.campaign_root,
            campaign_contract=self.contract,
            campaign_contract_sha256=artifacts.sha256_file(self.contract),
            learner_contract=self.learner,
            learner_contract_sha256=artifacts.sha256_file(self.learner),
            sealed_source_root=Path("/remote/sealed/source"),
            sealed_source_inventory_sha256=SHA,
            evaluation_sealed_source_root=self.evaluation_source,
            evaluation_sealed_source_inventory_sha256="e" * 64,
            launcher=self.launcher,
            launcher_sha256=artifacts.sha256_file(self.launcher),
            control_root=self.control_root,
            heartbeat_seconds=0.01,
        )
        self.runtime_contract = {
            "campaign_id": "campaign",
            "topology": {"macro_round_count": 5},
            "evaluation": {"hydra_config_sha256": EVALUATOR},
        }
        self.campaign = {"campaign_id": "campaign", "macro_round_count": 5}

    def _run_restart_phase(self, phase: str) -> dict:
        initial = coordinator(4, phase)
        complete = coordinator(5, "READY")
        snapshots = [
            (self.campaign, initial),
            (self.campaign, initial),  # heartbeat
            (self.campaign, complete),
            (self.campaign, complete),
        ]
        config = self.control_root / f"{phase}.json"
        binding = self.control_root / f"{phase}.binding.json"

        def runner(_argv, *, on_heartbeat, **_kwargs):
            on_heartbeat(999_999, None)
            return 0

        with (
            mock.patch.object(
                DriverInputs,
                "validate",
                return_value=(self.runtime_contract, {}, self.campaign),
            ),
            mock.patch(
                "round_dagger.production_campaign_driver._snapshot",
                side_effect=snapshots,
            ),
            mock.patch(
                "round_dagger.production_campaign_driver.materialize_orchestration_config",
                return_value=(config, SHA, binding),
            ) as materialize,
        ):
            result = run_driver(self.inputs, launcher_runner=runner)
        materialize.assert_called_once()
        return result

    def test_restart_from_every_durable_phase(self) -> None:
        for phase in (
            "READY",
            "COLLECTING",
            "COLLECTED",
            "INGESTED",
            "TRAINED",
            "EVALUATED",
        ):
            with self.subTest(phase=phase):
                result = self._run_restart_phase(phase)
                self.assertEqual(result["state"], "COMPLETE")
                self.assertEqual(result["next_macro_round"], 5)

    def test_completed_restart_does_not_launch(self) -> None:
        complete = coordinator(5, "READY")
        with (
            mock.patch.object(
                DriverInputs,
                "validate",
                return_value=(self.runtime_contract, {}, self.campaign),
            ),
            mock.patch(
                "round_dagger.production_campaign_driver._snapshot",
                return_value=(self.campaign, complete),
            ),
            mock.patch(
                "round_dagger.production_campaign_driver.materialize_orchestration_config"
            ) as materialize,
        ):
            result = run_driver(self.inputs)
        materialize.assert_not_called()
        self.assertTrue(result["completed"])

    def test_all_five_rounds_are_sequential_and_fresh(self) -> None:
        current = coordinator(0, "READY")
        materialized: list[Path] = []

        def snapshot(_root):
            return self.campaign, dict(current)

        def materialize(_inputs, _contract, _campaign, observed):
            path = self.control_root / f"round-{observed['next_macro_round']}-{len(materialized)}.json"
            materialized.append(path)
            return path, SHA, path.with_suffix(".binding.json")

        def runner(_argv, **_kwargs):
            next_round = current["next_macro_round"] + 1
            current.clear()
            current.update(coordinator(next_round, "READY"))
            return 0

        with (
            mock.patch.object(
                DriverInputs,
                "validate",
                return_value=(self.runtime_contract, {}, self.campaign),
            ),
            mock.patch("round_dagger.production_campaign_driver._snapshot", side_effect=snapshot),
            mock.patch(
                "round_dagger.production_campaign_driver.materialize_orchestration_config",
                side_effect=materialize,
            ),
        ):
            result = run_driver(self.inputs, launcher_runner=runner)
        self.assertEqual(result["next_macro_round"], 5)
        self.assertEqual(len(materialized), 5)
        self.assertEqual(len(set(materialized)), 5)

    def test_nonzero_launcher_stops_fail_closed(self) -> None:
        initial = coordinator(1, "COLLECTED")
        config = self.control_root / "failed.json"
        with (
            mock.patch.object(
                DriverInputs,
                "validate",
                return_value=(self.runtime_contract, {}, self.campaign),
            ),
            mock.patch(
                "round_dagger.production_campaign_driver._snapshot",
                side_effect=[(self.campaign, initial), (self.campaign, initial)],
            ),
            mock.patch(
                "round_dagger.production_campaign_driver.materialize_orchestration_config",
                return_value=(config, SHA, config.with_suffix(".binding.json")),
            ),
        ):
            with self.assertRaisesRegex(ProductionDriverError, "exited 7"):
                run_driver(self.inputs, launcher_runner=lambda *_args, **_kwargs: 7)
        status = json.loads((self.control_root / "driver_status.json").read_text())
        self.assertEqual(status["state"], "FAILED")

    def test_dry_run_validates_one_round_without_state_change(self) -> None:
        initial = coordinator(2, "TRAINED")
        config = self.control_root / "dry.json"
        captured: list[list[str]] = []

        def runner(argv, **_kwargs):
            captured.append(list(argv))
            return 0

        with (
            mock.patch.object(
                DriverInputs,
                "validate",
                return_value=(self.runtime_contract, {}, self.campaign),
            ),
            mock.patch(
                "round_dagger.production_campaign_driver._snapshot",
                side_effect=[(self.campaign, initial), (self.campaign, initial)],
            ),
            mock.patch(
                "round_dagger.production_campaign_driver.materialize_orchestration_config",
                return_value=(config, SHA, config.with_suffix(".binding.json")),
            ),
        ):
            result = run_driver(self.inputs, dry_run=True, launcher_runner=runner)
        self.assertEqual(result["state"], "DRY_RUN_VALIDATED")
        self.assertIn("--dry-run", captured[0])

    def test_fresh_immutable_config_for_same_policy(self) -> None:
        observed = coordinator(0, "READY")
        with mock.patch(
            "round_dagger.production_campaign_driver._derive_learner_bundle_sha",
            return_value="f" * 64,
        ):
            first, _, _ = materialize_orchestration_config(
                self.inputs, self.runtime_contract, self.campaign, observed
            )
            second, _, _ = materialize_orchestration_config(
                self.inputs, self.runtime_contract, self.campaign, observed
            )
        self.assertNotEqual(first, second)
        self.assertEqual(first.stat().st_mode & 0o222, 0)
        self.assertEqual(second.stat().st_mode & 0o222, 0)
        self.assertEqual(json.loads(first.read_text())["macro_round_count"], 5)
        self.assertEqual(
            json.loads(first.read_text())["campaign_contract_sha256"],
            artifacts.sha256_file(self.contract),
        )

    def test_launcher_binds_external_runtime_contract_path_and_sha(self) -> None:
        argv = _launcher_argv(self.inputs, self.base / "config.json", dry_run=False)
        self.assertEqual(argv[argv.index("--campaign-contract") + 1], str(self.contract))
        self.assertEqual(
            argv[argv.index("--campaign-contract-sha256") + 1],
            artifacts.sha256_file(self.contract),
        )

    def test_validate_consumes_real_runtime_evidence_schema_via_materializer(self) -> None:
        runtime_evidence = {
            "schema": "fast_roborender.round_dagger.production_runtime_contract.v1",
            "status": "release_eligible",
            "invariant_profile_sha256": "1" * 64,
            "camera_acceptance": {"path": "/camera.json", "sha256": "2" * 64},
            "renderer_fleet": {
                "endpoint_manifest_path": "/endpoints.txt",
                "endpoint_manifest_sha256": "3" * 64,
                "endpoint_directory": "/endpoints",
                "renderer_count": 2,
                "cohort_identity_sha256": "4" * 64,
                "runtime_identity_sha256": "5" * 64,
            },
            "live_capture": {
                "split_manifest_path": "/split.json",
                "split_manifest_sha256": "6" * 64,
                "heldout_plan_path": "/plan.json",
                "heldout_plan_sha256": "7" * 64,
                "pass_artifacts": [],
                "attestation_path": "/attestation.json",
                "attestation_sha256": "8" * 64,
                "evidence_path": "/evidence.json",
                "evidence_sha256": "9" * 64,
                "scenario_path": "/scenarios.json",
                "scenario_sha256": "0" * 64,
            },
        }
        self.contract.chmod(0o600)
        self.contract.write_text(json.dumps(runtime_evidence) + "\n", encoding="utf-8")
        self.contract.chmod(0o444)
        materialized = {
            "campaign_id": "campaign",
            "topology": {"macro_round_count": 5},
            "evaluation": {"hydra_config_sha256": EVALUATOR},
        }
        aligned = DriverInputs(
            campaign_root=self.campaign_root,
            campaign_contract=self.contract,
            campaign_contract_sha256=artifacts.sha256_file(self.contract),
            learner_contract=ROOT / "configs/round_dagger/production_learner_8xh100_1200.json",
            learner_contract_sha256="a" * 64,
            sealed_source_root=Path("/remote/sealed/source"),
            sealed_source_inventory_sha256=SHA,
            evaluation_sealed_source_root=ROOT,
            evaluation_sealed_source_inventory_sha256="e" * 64,
            launcher=ROOT / "scripts/round_dagger/launch_production_macro_round.sh",
            launcher_sha256="b" * 64,
            control_root=self.control_root,
        )
        with (
            mock.patch("round_dagger.production_campaign_driver._verify_file"),
            mock.patch(
                "round_dagger.production_campaign_driver.state.load_campaign",
                return_value={"campaign_id": "campaign", "macro_round_count": 5},
            ),
            mock.patch(
                "scripts.round_dagger.production_launch.materialize_campaign_contract",
                return_value=(materialized, runtime_evidence, {"schema": "lineage"}),
            ) as materializer,
        ):
            observed, _, _ = aligned.validate()
        self.assertEqual(observed, materialized)
        materializer.assert_called_once_with(
            ROOT / "configs/round_dagger/production_campaign_256x512.json",
            runtime_contract_path=self.contract,
            runtime_contract_sha256=artifacts.sha256_file(self.contract),
        )

    def test_campaign_global_lock_is_exclusive(self) -> None:
        with ExclusiveCampaignLock(self.campaign_root):
            with self.assertRaisesRegex(ProductionDriverError, "another production driver"):
                with ExclusiveCampaignLock(self.campaign_root):
                    pass


if __name__ == "__main__":
    unittest.main()
