from __future__ import annotations

import json
import signal
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from round_dagger import artifacts
from round_dagger.production_campaign_driver import (
    DriverInputs,
    ExclusiveCampaignLock,
    ExclusiveLauncherExecutionLock,
    ProductionDriverError,
    _check_path_chain,
    _collection_retry_streak,
    _is_operator_cancellation,
    _load_collection_attempts,
    _load_collection_failures,
    _publish_collection_attempt,
    _recoverable_collection_evidence,
    _reconcile_unmatched_attempt,
    _launcher_argv,
    _publish_immutable,
    materialize_orchestration_config,
    run_driver,
)
from round_dagger.production_data_lineage import (
    DATA_LINEAGE_SCHEMA,
    RUNTIME_CONTRACT_SCHEMA,
)
from round_dagger.scenario_capture import EXPECTED_SEEDS


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
        self.learner = self.campaign_root / "learner_contract.json"
        self.learner.write_bytes(
            (
                ROOT
                / "configs/round_dagger/production_learner_8xh100_1200.json"
            ).read_bytes()
        )
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
        self.campaign = {
            "campaign_id": "campaign",
            "macro_round_count": 5,
            "policy_config_sha256": artifacts.sha256_file(self.learner),
        }

    def _source_bound_inputs(self) -> DriverInputs:
        launcher = ROOT / "scripts/round_dagger/launch_production_macro_round.sh"
        return replace(
            self.inputs,
            launcher=launcher,
            launcher_sha256=artifacts.sha256_file(launcher),
        )

    def _publish_unmatched_attempt_fixture(
        self, inputs: DriverInputs, observed: dict
    ) -> tuple[Path, str, Path, dict]:
        active = observed["active_policy"]
        config_directory = inputs.control_root / "orchestration_configs"
        config_directory.mkdir(parents=True, exist_ok=True)
        config = config_directory / (
            f"macro_round_{observed['next_macro_round']:06d}"
            f"_policy_{active['version']:06d}"
            f"_{active['checkpoint_sha256'][:12]}_attempt_000001.json"
        )
        document = {
            "schema": "fast_roborender.round_dagger.orchestration_config.v3",
            "campaign_root": str(inputs.campaign_root),
            "actor_root": str(inputs.campaign_root / "actors"),
            "macro_round": observed["next_macro_round"],
            "macro_round_count": 5,
            "campaign_contract_sha256": inputs.campaign_contract_sha256,
            "runtime_policy_sha256": active["runtime_policy_sha256"],
            "learner_code_sha256": inputs.sealed_source_inventory_sha256,
            "learner_config_sha256": SHA,
            "learner_model_lineage_sha256": active["checkpoint_sha256"],
            "evaluator_config_sha256": EVALUATOR,
            "hooks": {
                "collect": {"argv": [str(self.launcher)], "timeout_seconds": 1},
                "transfer": None,
                "ingest": None,
                "train": {"argv": [str(self.launcher)], "timeout_seconds": 1},
                "evaluate": {"argv": [str(self.launcher)], "timeout_seconds": 1},
                "promote": None,
            },
        }
        config_sha = _publish_immutable(config, document)
        binding = config.with_suffix(".binding.json")
        _publish_immutable(
            binding,
            {
                "schema": "fast_roborender.round_dagger.production_driver_binding.v1",
                "campaign_id": self.campaign["campaign_id"],
                "campaign_contract_path": str(inputs.campaign_contract),
                "campaign_contract_sha256": inputs.campaign_contract_sha256,
                "learner_contract_path": str(inputs.learner_contract),
                "learner_contract_sha256": inputs.learner_contract_sha256,
                "launcher_path": str(inputs.launcher),
                "launcher_sha256": inputs.launcher_sha256,
                "macro_round": observed["next_macro_round"],
                "active_policy_version": active["version"],
                "active_policy_checkpoint_sha256": active["checkpoint_sha256"],
                "orchestration_config_path": str(config),
                "orchestration_config_sha256": config_sha,
            },
        )
        _publish_collection_attempt(
            inputs=inputs,
            campaign=self.campaign,
            coordinator=observed,
            attempt_index=0,
            config_path=config,
            config_sha256=config_sha,
        )
        return config, config_sha, binding, document

    def _run_restart_phase(self, phase: str) -> dict:
        initial = coordinator(4, phase)
        complete = coordinator(5, "READY")
        snapshots = [
            (self.campaign, initial),
            (self.campaign, initial),  # heartbeat
            (self.campaign, complete),
            (self.campaign, complete),
        ]
        phase_control = self.control_root / phase
        inputs = replace(self.inputs, control_root=phase_control)
        config = phase_control / f"{phase}.json"
        binding = phase_control / f"{phase}.binding.json"

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
            result = run_driver(inputs, launcher_runner=runner)
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

    def test_unmatched_attempt_resumes_from_every_durable_phase(self) -> None:
        for phase in (
            "READY",
            "COLLECTING",
            "COLLECTED",
            "INGESTED",
            "TRAINED",
            "EVALUATED",
        ):
            with self.subTest(phase=phase):
                observed = coordinator(4, phase)
                complete = coordinator(5, "READY")
                control_root = self.control_root / f"unmatched-{phase.lower()}"
                inputs = replace(self.inputs, control_root=control_root)
                config, _config_sha, _binding, document = (
                    self._publish_unmatched_attempt_fixture(inputs, observed)
                )
                launched: list[tuple[list[str], dict]] = []

                def runner(argv, **kwargs):
                    launched.append((list(argv), dict(kwargs)))
                    return 0

                with (
                    mock.patch.object(
                        DriverInputs,
                        "validate",
                        return_value=(self.runtime_contract, {}, self.campaign),
                    ),
                    mock.patch(
                        "round_dagger.production_campaign_driver._snapshot",
                        side_effect=[
                            (self.campaign, observed),
                            (self.campaign, complete),
                            (self.campaign, complete),
                        ],
                    ),
                    mock.patch(
                        "round_dagger.production_campaign_driver._orchestration_document",
                        return_value=document,
                    ),
                    mock.patch(
                        "round_dagger.production_campaign_driver."
                        "materialize_orchestration_config"
                    ) as materialize,
                ):
                    result = run_driver(inputs, launcher_runner=runner)
                materialize.assert_not_called()
                self.assertEqual(result["state"], "COMPLETE")
                self.assertEqual(len(launched), 1)
                self.assertIn(str(config), launched[0][0])
                self.assertIsInstance(
                    launched[0][1].get("execution_lock_fd"), int
                )
                self.assertFalse(
                    list(
                        (control_root / "collection_attempt_failures").glob(
                            "*.json"
                        )
                    )
                )

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
            path = self.control_root / (
                f"round-{observed['next_macro_round']}-{len(materialized)}.json"
            )
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
        inputs = replace(self.inputs, collection_retry_limit=0)
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
                run_driver(inputs, launcher_runner=lambda *_args, **_kwargs: 7)
        status = json.loads((self.control_root / "driver_status.json").read_text())
        self.assertEqual(status["state"], "FAILED")

    def test_recoverable_collection_failure_retries_only_with_proof(self) -> None:
        initial = coordinator(1, "COLLECTING")
        complete = coordinator(2, "READY")
        finished = coordinator(5, "READY")
        configs = [self.control_root / "retry-1.json", self.control_root / "retry-2.json"]
        returns = iter((2, 0))
        snapshots = iter(
            (
                (self.campaign, initial),
                (self.campaign, initial),
                (self.campaign, initial),
                (self.campaign, complete),
                (self.campaign, finished),
            )
        )
        inputs = replace(
            self.inputs,
            collection_retry_limit=1,
            collection_retry_backoff_seconds=0,
        )
        with (
            mock.patch.object(
                DriverInputs,
                "validate",
                return_value=(self.runtime_contract, {}, self.campaign),
            ),
            mock.patch(
                "round_dagger.production_campaign_driver._snapshot",
                side_effect=lambda _root: next(snapshots),
            ),
            mock.patch(
                "round_dagger.production_campaign_driver.materialize_orchestration_config",
                side_effect=[
                    (configs[0], SHA, configs[0].with_suffix(".binding.json")),
                    (configs[1], SHA, configs[1].with_suffix(".binding.json")),
                ],
            ) as materialize,
            mock.patch(
                "round_dagger.production_campaign_driver._recoverable_collection_evidence",
                return_value={"safe_to_retry": True},
            ),
        ):
            result = run_driver(
                inputs,
                launcher_runner=lambda *_args, **_kwargs: next(returns),
            )
        self.assertEqual(result["state"], "COMPLETE")
        self.assertEqual(materialize.call_count, 2)

    def test_recoverable_collection_failure_stops_after_bounded_retries(self) -> None:
        initial = coordinator(1, "COLLECTING")
        configs = [self.control_root / "retry-1.json", self.control_root / "retry-2.json"]
        inputs = replace(
            self.inputs,
            collection_retry_limit=1,
            collection_retry_backoff_seconds=0,
        )
        with (
            mock.patch.object(
                DriverInputs,
                "validate",
                return_value=(self.runtime_contract, {}, self.campaign),
            ),
            mock.patch(
                "round_dagger.production_campaign_driver._snapshot",
                return_value=(self.campaign, initial),
            ),
            mock.patch(
                "round_dagger.production_campaign_driver.materialize_orchestration_config",
                side_effect=[
                    (configs[0], SHA, configs[0].with_suffix(".binding.json")),
                    (configs[1], SHA, configs[1].with_suffix(".binding.json")),
                ],
            ),
            mock.patch(
                "round_dagger.production_campaign_driver._recoverable_collection_evidence",
                return_value={"safe_to_retry": True},
            ),
        ):
            with self.assertRaisesRegex(ProductionDriverError, "exited 2"):
                run_driver(inputs, launcher_runner=lambda *_args, **_kwargs: 2)
        status = json.loads((self.control_root / "driver_status.json").read_text())
        self.assertEqual(status["state"], "FAILED")
        self.assertEqual(status["collection_retry_count"], 1)
        self.assertIn("retry limit", status["detail"])

    def test_retry_budget_resets_after_a_cohort_is_durably_sealed(self) -> None:
        empty = {
            "safe_to_retry": True,
            "cohorts": [
                {
                    "cohort_index": 0,
                    "state": "unsealed",
                    "rank_states": {"0": "empty", "1": "empty"},
                },
                *(
                    {"cohort_index": index, "state": "absent"}
                    for index in range(1, 4)
                ),
            ],
        }
        sealed = {
            "safe_to_retry": True,
            "cohorts": [
                {"cohort_index": 0, "state": "sealed"},
                *(
                    {"cohort_index": index, "state": "absent"}
                    for index in range(1, 4)
                ),
            ],
        }
        failures = [{"recovery": empty} for _ in range(5)]
        self.assertEqual(_collection_retry_streak(failures), 5)
        self.assertEqual(
            _collection_retry_streak([*failures, {"recovery": sealed}]),
            1,
        )
        self.assertEqual(
            _collection_retry_streak(
                [*failures, {"recovery": sealed}, {"recovery": sealed}]
            ),
            2,
        )

    def test_retry_budget_survives_driver_crash_and_restart(self) -> None:
        initial = coordinator(1, "COLLECTING")
        inputs = replace(
            self.inputs,
            collection_retry_limit=1,
            collection_retry_backoff_seconds=1,
        )
        first_config = self.control_root / "restart-first.json"
        with (
            mock.patch.object(
                DriverInputs,
                "validate",
                return_value=(self.runtime_contract, {}, self.campaign),
            ),
            mock.patch(
                "round_dagger.production_campaign_driver._snapshot",
                return_value=(self.campaign, initial),
            ),
            mock.patch(
                "round_dagger.production_campaign_driver.materialize_orchestration_config",
                return_value=(
                    first_config,
                    SHA,
                    first_config.with_suffix(".binding.json"),
                ),
            ),
            mock.patch(
                "round_dagger.production_campaign_driver._recoverable_collection_evidence",
                return_value={"safe_to_retry": True},
            ),
            mock.patch(
                "round_dagger.production_campaign_driver.time.sleep",
                side_effect=RuntimeError("injected driver crash after durable failure"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected driver crash"):
                run_driver(inputs, launcher_runner=lambda *_args, **_kwargs: 2)

        ledger = sorted(
            (self.control_root / "collection_attempt_failures").glob("*.json")
        )
        self.assertEqual(len(ledger), 1)
        self.assertEqual(ledger[0].stat().st_mode & 0o222, 0)
        self.assertEqual(json.loads(ledger[0].read_text())["failure_index"], 1)

        second_config = self.control_root / "restart-second.json"
        launches = 0

        def fail_retry(*_args, **_kwargs):
            nonlocal launches
            launches += 1
            return 2

        with (
            mock.patch.object(
                DriverInputs,
                "validate",
                return_value=(self.runtime_contract, {}, self.campaign),
            ),
            mock.patch(
                "round_dagger.production_campaign_driver._snapshot",
                return_value=(self.campaign, initial),
            ),
            mock.patch(
                "round_dagger.production_campaign_driver.materialize_orchestration_config",
                return_value=(
                    second_config,
                    SHA,
                    second_config.with_suffix(".binding.json"),
                ),
            ),
            mock.patch(
                "round_dagger.production_campaign_driver._recoverable_collection_evidence",
                return_value={"safe_to_retry": True},
            ),
        ):
            with self.assertRaisesRegex(ProductionDriverError, "exited 2"):
                run_driver(inputs, launcher_runner=fail_retry)
        self.assertEqual(launches, 1)

    def test_unmatched_attempt_resumes_without_consuming_retry_budget(self) -> None:
        initial = coordinator(4, "COLLECTING")
        complete = coordinator(5, "READY")
        inputs = replace(
            self.inputs,
            collection_retry_limit=1,
            collection_retry_backoff_seconds=0,
        )
        crashed_config = self.control_root / "crashed-runner.json"
        with (
            mock.patch.object(
                DriverInputs,
                "validate",
                return_value=(self.runtime_contract, {}, self.campaign),
            ),
            mock.patch(
                "round_dagger.production_campaign_driver._snapshot",
                return_value=(self.campaign, initial),
            ),
            mock.patch(
                "round_dagger.production_campaign_driver.materialize_orchestration_config",
                return_value=(
                    crashed_config,
                    SHA,
                    crashed_config.with_suffix(".binding.json"),
                ),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "runner process vanished"):
                run_driver(
                    inputs,
                    launcher_runner=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                        RuntimeError("runner process vanished")
                    ),
                )
        self.assertEqual(
            len(list((self.control_root / "collection_launcher_attempts").glob("*.json"))),
            1,
        )
        self.assertFalse(
            list((self.control_root / "collection_attempt_failures").glob("*.json"))
        )

        launches = 0

        def finish_resume(*_args, **_kwargs):
            nonlocal launches
            launches += 1
            return 0

        with (
            mock.patch.object(
                DriverInputs,
                "validate",
                return_value=(self.runtime_contract, {}, self.campaign),
            ),
            mock.patch(
                "round_dagger.production_campaign_driver._snapshot",
                side_effect=[
                    (self.campaign, initial),
                    (self.campaign, complete),
                    (self.campaign, complete),
                ],
            ),
            mock.patch(
                "round_dagger.production_campaign_driver._reconcile_unmatched_attempt",
                return_value=(
                    crashed_config,
                    SHA,
                    crashed_config.with_suffix(".binding.json"),
                    {"safe_to_retry": True},
                ),
            ),
            mock.patch(
                "round_dagger.production_campaign_driver.materialize_orchestration_config"
            ) as materialize,
        ):
            result = run_driver(inputs, launcher_runner=finish_resume)
        materialize.assert_not_called()
        self.assertEqual(launches, 1)
        self.assertEqual(result["state"], "COMPLETE")
        self.assertFalse(
            list((self.control_root / "collection_attempt_failures").glob("*.json"))
        )

    def test_operator_signal_is_never_automatically_retried(self) -> None:
        for return_code in (-15, 129, 130, 137, 143, 192):
            self.assertTrue(_is_operator_cancellation(return_code))
        for return_code in (1, 2, 7, 128, 193, 255):
            self.assertFalse(_is_operator_cancellation(return_code))
        initial = coordinator(0, "COLLECTING")
        config = self.control_root / "cancelled.json"
        inputs = replace(
            self.inputs,
            collection_retry_limit=5,
            collection_retry_backoff_seconds=0,
        )
        with (
            mock.patch.object(
                DriverInputs,
                "validate",
                return_value=(self.runtime_contract, {}, self.campaign),
            ),
            mock.patch(
                "round_dagger.production_campaign_driver._snapshot",
                return_value=(self.campaign, initial),
            ),
            mock.patch(
                "round_dagger.production_campaign_driver.materialize_orchestration_config",
                return_value=(config, SHA, config.with_suffix(".binding.json")),
            ) as materialize,
            mock.patch(
                "round_dagger.production_campaign_driver._recoverable_collection_evidence"
            ) as recovery,
        ):
            with self.assertRaisesRegex(ProductionDriverError, "exited 143"):
                run_driver(inputs, launcher_runner=lambda *_args, **_kwargs: 143)
        materialize.assert_called_once()
        recovery.assert_not_called()
        failure = json.loads(
            next(
                (self.control_root / "collection_attempt_failures").glob("*.json")
            ).read_text()
        )
        self.assertTrue(failure["operator_cancelled"])
        self.assertIsNone(failure["recovery"])
        status = json.loads((self.control_root / "driver_status.json").read_text())
        self.assertIn("automatic retry is forbidden", status["detail"])

    def test_operator_signal_after_promotion_stops_without_next_launch(self) -> None:
        initial = coordinator(4, "EVALUATED")
        complete = coordinator(5, "READY")
        config = self.control_root / "promoted-then-cancelled.json"
        launches = 0

        def cancelled_after_promotion(*_args, **_kwargs):
            nonlocal launches
            launches += 1
            return 143

        with (
            mock.patch.object(
                DriverInputs,
                "validate",
                return_value=(self.runtime_contract, {}, self.campaign),
            ),
            mock.patch(
                "round_dagger.production_campaign_driver._snapshot",
                side_effect=[
                    (self.campaign, initial),
                    (self.campaign, complete),
                ],
            ),
            mock.patch(
                "round_dagger.production_campaign_driver."
                "materialize_orchestration_config",
                return_value=(config, SHA, config.with_suffix(".binding.json")),
            ) as materialize,
        ):
            with self.assertRaisesRegex(ProductionDriverError, "exited 143"):
                run_driver(
                    self.inputs,
                    launcher_runner=cancelled_after_promotion,
                )
        self.assertEqual(launches, 1)
        materialize.assert_called_once()
        status = json.loads((self.control_root / "driver_status.json").read_text())
        self.assertEqual(status["state"], "FAILED")
        self.assertEqual(status["next_macro_round"], 5)
        self.assertIn("cancelled by an operator signal", status["detail"])

    def test_collection_recovery_evidence_requires_clean_sequential_frontier(self) -> None:
        initial = coordinator(0, "COLLECTING")
        actor_round = (
            self.campaign_root / "actors" / "campaign" / "macro_round_000000"
        )
        cohort = actor_round / "cohort_00"
        cohort.mkdir(parents=True)
        clean = {
            "sealed": False,
            "rank_states": {"0": "complete", "1": "resumable"},
        }
        with mock.patch(
            "round_dagger.actor_pair.inspect_actor_pair_for_recovery",
            return_value=clean,
        ):
            evidence = _recoverable_collection_evidence(
                inputs=self.inputs,
                campaign=self.campaign,
                before=initial,
                after=initial,
            )
        self.assertIsNotNone(evidence)
        self.assertEqual(
            evidence["cohorts"][0]["rank_states"],
            {"0": "complete", "1": "resumable"},
        )

        partial = {"sealed": False, "rank_states": {"0": "partial", "1": "empty"}}
        with mock.patch(
            "round_dagger.actor_pair.inspect_actor_pair_for_recovery",
            return_value=partial,
        ):
            self.assertIsNone(
                _recoverable_collection_evidence(
                    inputs=self.inputs,
                    campaign=self.campaign,
                    before=initial,
                    after=initial,
                )
            )

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

    def test_launcher_binds_external_runtime_and_learner_contract_paths_and_shas(
        self,
    ) -> None:
        argv = _launcher_argv(self.inputs, self.base / "config.json", dry_run=False)
        self.assertEqual(argv[argv.index("--campaign-contract") + 1], str(self.contract))
        self.assertEqual(
            argv[argv.index("--campaign-contract-sha256") + 1],
            artifacts.sha256_file(self.contract),
        )
        self.assertEqual(
            argv[argv.index("--learner-contract") + 1],
            str(self.inputs.learner_contract),
        )
        self.assertEqual(
            argv[argv.index("--learner-contract-sha256") + 1],
            self.inputs.learner_contract_sha256,
        )

    def test_validate_consumes_real_runtime_evidence_from_distinct_evaluation_seal(
        self,
    ) -> None:
        bundles = [
            {
                "seed": seed,
                "path": f"/first-conditions/seed-{seed}.npz",
                "sha256": artifacts.canonical_json_sha256(
                    {"fixture": "first-condition-bundle", "seed": seed}
                ),
                "size": 1,
                "case_count": 32,
                "source_pass_index": 1,
            }
            for seed in EXPECTED_SEEDS
        ]
        bundle_inventory_sha256 = artifacts.canonical_json_sha256(bundles)
        runtime_evidence = {
            "schema": RUNTIME_CONTRACT_SCHEMA,
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
                "first_condition_bundles": bundles,
                "first_condition_bundle_inventory_sha256": (
                    bundle_inventory_sha256
                ),
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
        lineage = {
            "schema": DATA_LINEAGE_SCHEMA,
            "heldout_evaluation": {
                "first_condition_bundles": bundles,
                "first_condition_bundle_inventory_sha256": (
                    bundle_inventory_sha256
                ),
            },
        }
        aligned = DriverInputs(
            campaign_root=self.campaign_root,
            campaign_contract=self.contract,
            campaign_contract_sha256=artifacts.sha256_file(self.contract),
            learner_contract=self.learner,
            learner_contract_sha256=artifacts.sha256_file(self.learner),
            sealed_source_root=Path("/remote/sealed/source"),
            sealed_source_inventory_sha256=SHA,
            evaluation_sealed_source_root=self.evaluation_source,
            evaluation_sealed_source_inventory_sha256="e" * 64,
            launcher=ROOT / "scripts/round_dagger/launch_production_macro_round.sh",
            launcher_sha256="b" * 64,
            control_root=self.control_root,
        )
        with (
            mock.patch("round_dagger.production_campaign_driver._verify_file"),
            mock.patch(
                "round_dagger.production_campaign_driver.state.load_campaign",
                return_value=self.campaign,
            ),
            mock.patch(
                "scripts.round_dagger.production_launch.materialize_campaign_contract",
                return_value=(materialized, runtime_evidence, lineage),
            ) as materializer,
        ):
            observed, _, _ = aligned.validate()
        self.assertEqual(observed, materialized)
        materializer.assert_called_once_with(
            ROOT / "configs/round_dagger/production_campaign_256x512.json",
            runtime_contract_path=self.contract,
            runtime_contract_sha256=artifacts.sha256_file(self.contract),
        )

    def test_validate_rejects_noncanonical_or_mutable_learner_contract(self) -> None:
        inputs = self._source_bound_inputs()
        alternate = self.base / "alternate-learner-contract.json"
        alternate.write_bytes(self.learner.read_bytes())
        alternate.chmod(0o444)
        with mock.patch("round_dagger.production_campaign_driver._verify_file"):
            with self.assertRaisesRegex(
                ProductionDriverError, "canonical campaign-owned"
            ):
                replace(
                    inputs,
                    learner_contract=alternate,
                    learner_contract_sha256=artifacts.sha256_file(alternate),
                ).validate()

        self.learner.chmod(0o400)
        with mock.patch("round_dagger.production_campaign_driver._verify_file"):
            with self.assertRaisesRegex(ProductionDriverError, "exactly 0444"):
                inputs.validate()

    def test_validate_rejects_campaign_policy_contract_identity_mismatch(self) -> None:
        inputs = self._source_bound_inputs()
        with (
            mock.patch("round_dagger.production_campaign_driver._verify_file"),
            mock.patch(
                "round_dagger.production_campaign_driver.state.load_campaign",
                return_value={**self.campaign, "policy_config_sha256": "f" * 64},
            ),
        ):
            with self.assertRaisesRegex(
                ProductionDriverError, "policy_config_sha256 differs"
            ):
                inputs.validate()

    def test_validate_rejects_relative_evaluation_seal(self) -> None:
        relative = DriverInputs(
            campaign_root=self.campaign_root,
            campaign_contract=self.contract,
            campaign_contract_sha256=artifacts.sha256_file(self.contract),
            learner_contract=self.learner,
            learner_contract_sha256=artifacts.sha256_file(self.learner),
            sealed_source_root=Path("/remote/sealed/source"),
            sealed_source_inventory_sha256=SHA,
            evaluation_sealed_source_root=Path("relative-evaluation-seal"),
            evaluation_sealed_source_inventory_sha256="e" * 64,
            launcher=ROOT / "scripts/round_dagger/launch_production_macro_round.sh",
            launcher_sha256="b" * 64,
            control_root=self.control_root,
        )
        with self.assertRaisesRegex(
            ProductionDriverError, "evaluation sealed source root must be absolute"
        ):
            relative.validate()

    def test_campaign_global_lock_is_exclusive(self) -> None:
        with ExclusiveCampaignLock(self.campaign_root):
            with self.assertRaisesRegex(ProductionDriverError, "another production driver"):
                with ExclusiveCampaignLock(self.campaign_root):
                    pass

    def test_replacement_driver_refuses_orphan_child_holding_launcher_lock(
        self,
    ) -> None:
        child = None
        with ExclusiveLauncherExecutionLock(self.campaign_root) as lock:
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                pass_fds=(lock.descriptor,),
            )
        try:
            with (
                mock.patch.object(
                    DriverInputs,
                    "validate",
                    return_value=(self.runtime_contract, {}, self.campaign),
                ),
                self.assertRaisesRegex(
                    ProductionDriverError,
                    "prior launcher/orchestrator is still alive",
                ),
            ):
                run_driver(self.inputs)
        finally:
            if child is not None:
                child.terminate()
                child.wait(timeout=5)
        with ExclusiveLauncherExecutionLock(self.campaign_root):
            pass

    def test_signal_during_first_heartbeat_is_forwarded_to_launcher(self) -> None:
        code = """
import signal
import sys
import time
from pathlib import Path
from round_dagger.production_campaign_driver import _invoke_launcher

def heartbeat(_pid, _ticks):
    print("HEARTBEAT_ENTERED", flush=True)
    time.sleep(1)

result = _invoke_launcher(
    [sys.executable, "-c", "import time; time.sleep(30)"],
    log_path=Path(sys.argv[1]),
    heartbeat_seconds=0.05,
    on_heartbeat=heartbeat,
)
print(f"RETURN={result}", flush=True)
"""
        worker = subprocess.Popen(
            [sys.executable, "-c", code, str(self.base / "signal-window.log")],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            self.assertIsNotNone(worker.stdout)
            self.assertEqual(worker.stdout.readline().strip(), "HEARTBEAT_ENTERED")
            worker.send_signal(signal.SIGTERM)
            stdout, stderr = worker.communicate(timeout=5)
        except BaseException:
            worker.kill()
            worker.wait(timeout=5)
            raise
        self.assertEqual(worker.returncode, 0, stderr)
        self.assertIn("RETURN=143", stdout)

    def test_unmatched_attempt_rejects_binding_to_another_policy(self) -> None:
        observed = coordinator(0, "READY")
        inputs = replace(self.inputs, control_root=self.control_root / "bad-binding")
        config, config_sha, binding, document = (
            self._publish_unmatched_attempt_fixture(inputs, observed)
        )
        binding_document = json.loads(binding.read_text(encoding="utf-8"))
        binding.chmod(0o600)
        binding_document["active_policy_version"] = 7
        binding.write_text(
            json.dumps(binding_document, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        binding.chmod(0o444)
        attempt = _load_collection_attempts(
            inputs=inputs,
            campaign=self.campaign,
            coordinator=observed,
        )[0]
        self.assertEqual(attempt["orchestration_config_sha256"], config_sha)
        self.assertEqual(Path(attempt["orchestration_config_path"]), config)
        with (
            mock.patch(
                "round_dagger.production_campaign_driver._orchestration_document",
                return_value=document,
            ),
            self.assertRaisesRegex(
                ProductionDriverError, "binding no longer matches"
            ),
        ):
            _reconcile_unmatched_attempt(
                inputs=inputs,
                runtime_contract=self.runtime_contract,
                campaign=self.campaign,
                coordinator=observed,
                attempt=attempt,
            )

    def test_driver_rejects_symlink_in_any_existing_path_ancestor(self) -> None:
        real = self.base / "real-control-parent"
        real.mkdir()
        linked = self.base / "linked-control-parent"
        linked.symlink_to(real, target_is_directory=True)
        with self.assertRaisesRegex(ProductionDriverError, "symbolic link"):
            _check_path_chain(linked / "nested/driver_status.json", "driver output")

    def test_sigkill_truncated_staging_is_cleaned_before_atomic_publication(self) -> None:
        parent = self.control_root / "publication"
        parent.mkdir(parents=True)
        target = parent / "attempt.json"
        staging = parent / ".attempt.json.staging"
        killed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import os,signal,sys; "
                    "fd=os.open(sys.argv[1],os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400); "
                    "os.write(fd,b'truncated'); os.fsync(fd); "
                    "os.kill(os.getpid(),signal.SIGKILL)"
                ),
                str(staging),
            ],
            check=False,
        )
        self.assertEqual(killed.returncode, -9)
        self.assertEqual(staging.read_bytes(), b"truncated")

        document = {"schema": "test", "value": 7}
        digest = _publish_immutable(target, document)
        expected = (
            json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
        self.assertEqual(target.read_bytes(), expected)
        self.assertEqual(digest, artifacts.sha256_file(target))
        self.assertEqual(target.stat().st_mode & 0o222, 0)
        self.assertFalse(staging.exists())

    def test_truncated_final_artifact_is_never_overwritten_or_accepted(self) -> None:
        parent = self.control_root / "truncated-final"
        parent.mkdir(parents=True)
        target = parent / "attempt.json"
        target.write_bytes(b'{"partial":')
        target.chmod(0o444)
        before = target.read_bytes()
        with self.assertRaisesRegex(ProductionDriverError, "conflicts"):
            _publish_immutable(target, {"complete": True})
        self.assertEqual(target.read_bytes(), before)

    def test_unsafe_staging_symlink_is_rejected_without_unlinking(self) -> None:
        parent = self.control_root / "unsafe-staging"
        parent.mkdir(parents=True)
        target = parent / "attempt.json"
        victim = parent / "victim"
        victim.write_bytes(b"do not remove")
        staging = parent / ".attempt.json.staging"
        staging.symlink_to(victim)
        with self.assertRaisesRegex(ProductionDriverError, "staging artifact is unsafe"):
            _publish_immutable(target, {"complete": True})
        self.assertTrue(staging.is_symlink())
        self.assertEqual(victim.read_bytes(), b"do not remove")
        self.assertFalse(target.exists())

    def test_ledger_loaders_prune_sigkill_truncated_staging(self) -> None:
        observed = coordinator(0, "COLLECTING")
        cases = (
            (
                "collection_launcher_attempts",
                (
                    ".macro_round_000000_policy_000000_cccccccccccc_"
                    "attempt_000000.json.staging"
                ),
                _load_collection_attempts,
            ),
            (
                "collection_attempt_failures",
                (
                    ".macro_round_000000_policy_000000_cccccccccccc_"
                    "failure_000001.json.staging"
                ),
                _load_collection_failures,
            ),
        )
        for directory_name, staging_name, loader in cases:
            with self.subTest(ledger=directory_name):
                directory = self.control_root / directory_name
                directory.mkdir(parents=True, exist_ok=True)
                staging = directory / staging_name
                killed = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        (
                            "import os,signal,sys; "
                            "fd=os.open(sys.argv[1],"
                            "os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400); "
                            "os.write(fd,b'{\"truncated\":'); os.fsync(fd); "
                            "os.kill(os.getpid(),signal.SIGKILL)"
                        ),
                        str(staging),
                    ],
                    check=False,
                )
                self.assertEqual(killed.returncode, -9)
                self.assertEqual(staging.read_bytes(), b'{"truncated":')
                self.assertEqual(
                    loader(
                        inputs=self.inputs,
                        campaign=self.campaign,
                        coordinator=observed,
                    ),
                    [],
                )
                self.assertFalse(staging.exists())

    def test_ledger_loaders_reject_unsafe_exact_staging_entries(self) -> None:
        observed = coordinator(0, "COLLECTING")
        cases = (
            (
                "attempt-symlink",
                "collection_launcher_attempts",
                (
                    ".macro_round_000000_policy_000000_cccccccccccc_"
                    "attempt_000000.json.staging"
                ),
                "symlink",
                _load_collection_attempts,
            ),
            (
                "failure-directory",
                "collection_attempt_failures",
                (
                    ".macro_round_000000_policy_000000_cccccccccccc_"
                    "failure_000001.json.staging"
                ),
                "directory",
                _load_collection_failures,
            ),
        )
        for case_name, directory_name, staging_name, kind, loader in cases:
            with self.subTest(case=case_name):
                control_root = self.control_root / case_name
                inputs = replace(self.inputs, control_root=control_root)
                directory = control_root / directory_name
                directory.mkdir(parents=True, exist_ok=True)
                staging = directory / staging_name
                victim = control_root / "victim"
                victim.write_bytes(b"do not remove")
                if kind == "symlink":
                    staging.symlink_to(victim)
                else:
                    staging.mkdir()
                with self.assertRaisesRegex(
                    ProductionDriverError, "staging artifact is unsafe"
                ):
                    loader(
                        inputs=inputs,
                        campaign=self.campaign,
                        coordinator=observed,
                    )
                self.assertTrue(staging.exists() or staging.is_symlink())
                self.assertEqual(victim.read_bytes(), b"do not remove")

    def test_ledger_loader_does_not_prune_unknown_hidden_name(self) -> None:
        observed = coordinator(0, "COLLECTING")
        directory = self.control_root / "collection_launcher_attempts"
        directory.mkdir(parents=True, exist_ok=True)
        unknown = directory / ".operator-owned.json.staging"
        unknown.write_bytes(b'{"not":"a publisher staging artifact"}\n')
        unknown.chmod(0o444)
        with self.assertRaises(ProductionDriverError):
            _load_collection_attempts(
                inputs=self.inputs,
                campaign=self.campaign,
                coordinator=observed,
            )
        self.assertTrue(unknown.exists())
        self.assertEqual(
            unknown.read_bytes(), b'{"not":"a publisher staging artifact"}\n'
        )


if __name__ == "__main__":
    unittest.main()
