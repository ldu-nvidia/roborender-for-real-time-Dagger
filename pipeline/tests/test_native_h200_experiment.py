from __future__ import annotations

import copy
import hashlib
import json
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from round_dagger import native_h200_experiment as protocol


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT / "configs/round_dagger/native_h200_bc_dagger4_protocol.v1.json"
)
GROUP_REGISTRY_PATH = ROOT / "configs/round_dagger/native_h200_group_registry.json"


def canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def registry() -> dict[str, object]:
    return json.loads(GROUP_REGISTRY_PATH.read_text(encoding="utf-8"))


def aggregate(total: int, won: int, unsafe: int = 0) -> dict[str, object]:
    first = total // 2
    first_won = won // 2
    first_unsafe = unsafe // 2
    return {
        "near": {
            "episode_count": first,
            "success_count": first_won,
            "safety_violations": first_unsafe,
        },
        "far": {
            "episode_count": total - first,
            "success_count": won - first_won,
            "safety_violations": unsafe - first_unsafe,
        },
    }


class NativeH200ExperimentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "campaign"
        root_patch = mock.patch.object(protocol, "CANONICAL_CAMPAIGN_ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)

        self.contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        self.contract["campaign_root"] = str(self.root)
        self.split = protocol.build_group_split(self.contract, registry())
        self.test_registry = {
            "schema": protocol.TEST_REGISTRY_SCHEMA,
            "groups": self.split["assignments"]["test"],
            "case_count": 40,
            "cases_sha256": sha("sealed-test-cases"),
            "stratum_case_counts": {"near": 20, "far": 20},
        }
        self.test_path = Path(self.temporary.name) / "sealed-test.json"
        self.test_registry_file_sha256 = protocol._publish(
            self.test_path, self.test_registry
        )
        self.controller = protocol.Controller(self.root)
        self.checkpoints = self.root.parent / "checkpoints/students"
        bootstrap = self.checkpoints / "bootstrap/bootstrap.pt"
        bootstrap.parent.mkdir(parents=True, exist_ok=True)
        bootstrap.write_bytes(b"initial-student")
        bootstrap_spec = Path(f"{bootstrap}.policy_spec.json")
        bootstrap_spec.write_bytes(canonical({"fixture": "policy-spec"}))
        bootstrap_spec.chmod(0o444)
        test_contracts = self.root / "test/heldout/contracts"
        test_contracts.mkdir(parents=True, exist_ok=True)
        (test_contracts / "test_registry.v1.json").write_bytes(
            canonical(self.test_registry)
        )
        (test_contracts / "test_cases.private.v1.json").write_bytes(
            b"sealed-test-cases"
        )
        (test_contracts / "motion_partition.v1.json").write_bytes(
            canonical({"fixture": "test-motion-partition"})
        )

        def admit_snapshot(
            _controller: protocol.Controller,
            *,
            path: Path,
            file_sha256: str,
            snapshot_sha256: str,
            round_number: int | None,
        ) -> dict[str, object]:
            return {
                "path": str(path),
                "file_sha256": file_sha256,
                "snapshot_sha256": snapshot_sha256,
                "through_macro_round": 0 if round_number is None else round_number,
            }

        snapshot_patch = mock.patch.object(
            protocol.Controller,
            "_validated_train_replay_snapshot",
            autospec=True,
            side_effect=admit_snapshot,
        )
        snapshot_patch.start()
        self.addCleanup(snapshot_patch.stop)

    def snapshot_arguments(self, round_number: int) -> dict[str, object]:
        generation = protocol.generation_id(None if round_number == 0 else round_number)
        return {
            "train_replay_snapshot_path": (
                self.root
                / protocol.train_round_relative_root(generation)
                / "replay/manifests/train.snapshot.json"
            ),
            "train_replay_snapshot_file_sha256": sha(
                f"train-snapshot-file-{round_number}"
            ),
            "train_replay_snapshot_sha256": sha(
                f"train-snapshot-semantic-{round_number}"
            ),
        }

    def initialize(self) -> dict[str, object]:
        return self.controller.initialize(
            contract=self.contract,
            split=self.split,
            test_registry_sha256=self.test_registry_file_sha256,
            bc_replay_sha256=sha("bc-train-replay"),
            initial_checkpoint_sha256=sha("initial-student"),
            **self.snapshot_arguments(0),
        )

    def validation_result(
        self,
        *,
        prefix: str,
        index: int,
        mse: float,
        checkpoint_sha256: str,
        overrides: dict[str, object] | None = None,
    ) -> tuple[Path, str, dict[str, object]]:
        state = self.controller.status()
        dagger = state["phase"] == "dagger_running"
        macro_round = state["current_round"] if dagger else 0
        generation = protocol.generation_id(macro_round if dagger else None)
        newest_population = 2048 if dagger else 4096
        prior_population = 2048 if dagger else 0
        examples = index * (newest_population // 4)
        newest_examples = examples
        prior_examples = 0
        unit_examples = newest_population
        document: dict[str, object] = {
            "schema": protocol.VALIDATION_MSE_RESULT_SCHEMA,
            "campaign_id": protocol.CANONICAL_CAMPAIGN_ID,
            "generation_id": generation,
            "macro_round": macro_round,
            "checkpoint_sha256": checkpoint_sha256,
            "validation_artifact_sha256": sha("validation-artifact"),
            "validation_snapshot_sha256": sha("validation-snapshot"),
            "split_sha256": self.split["split_sha256"],
            "sampling_policy": "current_round_only_uniform_v1",
            "optimizer_examples": examples,
            "newest_source_examples": newest_examples,
            "prior_source_examples": prior_examples,
            "newest_source_population": newest_population,
            "prior_source_population": prior_population,
            "balanced_exposure_unit_examples": unit_examples,
            "balanced_exposure_units": examples / unit_examples,
            "newest_expected_passes": newest_examples / newest_population,
            "prior_expected_passes": (
                prior_examples / prior_population if prior_population else 0.0
            ),
            "optimizer_examples_per_step": 128,
            "validation_examples": 512,
            "validation_mse": mse,
        }
        if overrides:
            document.update(overrides)
        path = (
            self.root
            / "validation/results"
            / protocol.train_round_relative_root(generation)
            / "mse"
            / f"{prefix}-{index}-{len(list(self.controller.events.glob('*.json')))}.json"
        )
        file_sha256 = protocol._publish(path, document)
        return path, file_sha256, document

    def saturate(self, prefix: str, best_mse: float) -> str:
        best_checkpoint = ""
        values = [best_mse + 0.2, best_mse, *([best_mse] * 6)]
        for index, mse in enumerate(values, 1):
            checkpoint = sha(f"{prefix}-checkpoint-{index}")
            state = self.controller.status()
            macro_round = state["current_round"] if state["phase"] == "dagger_running" else 0
            generation = protocol.generation_id(macro_round if macro_round else None)
            checkpoint_path = (
                self.checkpoints
                / ("BC" if generation == protocol.GENERATION_IDS[0] else str(
                    protocol.train_round_relative_root(generation)
                ))
                / "training_steps"
                / f"{prefix}-checkpoint-{index}.pt"
            )
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_bytes(f"{prefix}-checkpoint-{index}".encode())
            policy_spec = Path(f"{checkpoint_path}.policy_spec.json")
            policy_spec.write_bytes(canonical({"fixture": "policy-spec"}))
            policy_spec.chmod(0o444)
            if index == 2:
                best_checkpoint = checkpoint
            path, file_sha256, _document = self.validation_result(
                prefix=prefix,
                index=index,
                mse=mse,
                checkpoint_sha256=checkpoint,
            )
            self.controller.record_validation(
                result_path=path,
                expected_file_sha256=file_sha256,
            )
        return best_checkpoint

    def final_test_result(
        self, *, checkpoint: str, generation: str, extra: dict[str, object] | None = None
    ) -> tuple[Path, str, dict[str, object]]:
        document: dict[str, object] = {
            "schema": protocol.FINAL_TEST_RESULT_SCHEMA,
            "campaign_id": protocol.CANONICAL_CAMPAIGN_ID,
            "protocol_id": self.contract["protocol_id"],
            "generation_id": generation,
            "checkpoint_sha256": checkpoint,
            "test_registry_sha256": self.test_registry_file_sha256,
            "test_cases_sha256": self.test_registry["cases_sha256"],
            "split_sha256": self.split["split_sha256"],
            "episode_count": 40,
            "success_count": 30,
            "safety_violations": 1,
            "per_stratum": aggregate(40, 30, 1),
        }
        if extra:
            document.update(extra)
        path = self.root / "test/heldout/results" / f"result-{len(extra or {})}.json"
        return path, protocol._publish(path, document), document

    def complete_training(self) -> tuple[str, str]:
        self.initialize()
        self.saturate("bc", 0.8)
        train_group = self.split["assignments"]["train"][0]
        last_checkpoint = ""
        for round_index, best_mse in enumerate((0.7, 0.7005, 0.6995), 1):
            self.controller.start_round(
                new_replay_sha256=sha(f"round-{round_index}-replay"),
                collection_groups=[train_group],
                **self.snapshot_arguments(round_index),
            )
            last_checkpoint = self.saturate(f"round-{round_index}", best_mse)
        state = self.controller.status()
        self.assertEqual(state["phase"], "training_complete")
        self.assertEqual(state["outer_stop_reason"], "no_progress")
        self.assertEqual(state["no_progress_rounds"], 2)
        self.assertEqual(state["selected_checkpoint_sha256"], last_checkpoint)
        self.assertEqual(state["selected_validation_mse"], 0.6995)
        self.controller.select_final()
        return last_checkpoint, protocol.GENERATION_IDS[3]

    def test_split_is_exact_balanced_and_rejects_duplicate_group_identity(self) -> None:
        again = protocol.build_group_split(self.contract, registry())
        self.assertEqual(self.split, again)
        self.assertEqual(
            self.split["counts"], {"train": 23, "validation": 5, "test": 5}
        )
        assigned = [
            item
            for name in protocol.SPLITS
            for item in self.split["assignments"][name]
        ]
        self.assertEqual(len(assigned), 33)
        self.assertEqual(len(set(assigned)), 33)
        self.assertTrue(self.split["balance_report"]["strata"])

        for field in ("group_id", "relative_path", "motion_sha256"):
            bad = copy.deepcopy(registry())
            bad["groups"][1][field] = bad["groups"][0][field]
            with self.subTest(field=field), self.assertRaises(
                protocol.NativeH200ExperimentError
            ):
                protocol.build_group_split(self.contract, bad)

    def test_contract_is_mse_only_and_distinguishes_virtual_from_physical(self) -> None:
        checked = protocol.validate_contract(self.contract)
        self.assertEqual(
            checked["topology"],
            {
                "macro_round_count": 4,
                "actor_process_count": 8,
                "gpus_per_actor_process": 1,
                "environments_per_actor_process": 32,
                "collection_wave_count": 1,
                "total_environment_streams": 256,
                "steps_per_environment": 512,
                "nominal_bc_target_transitions": 131072,
            },
        )
        self.assertEqual(
            [row["beta"] for row in checked["generations"]],
            [1.0, 0.75, 0.5, 0.25, 0.0],
        )
        self.assertEqual(
            [protocol.train_round_relative_root(generation)
             for generation in protocol.GENERATION_IDS],
            [
                "BC/isaac",
                "dagger_1",
                "dagger_2",
                "dagger_3",
                "dagger_4",
            ],
        )
        self.assertEqual(
            checked["collection_budget"]["probe_environments_per_actor_process"],
            [32],
        )
        self.assertEqual(
            {
                key: checked["collection_budget"][key]
                for key in (
                    "host_physical_gpu_count",
                    "admitted_physical_gpu_indices",
                    "physical_execution_concurrency",
                    "virtual_actor_world_size",
                    "actor_wave_scheduling",
                )
            },
            {
                "host_physical_gpu_count": 8,
                "admitted_physical_gpu_indices": list(range(8)),
                "physical_execution_concurrency": 8,
                "virtual_actor_world_size": 8,
                "actor_wave_scheduling": "one_actor_per_gpu_concurrent_v1",
            },
        )
        self.assertNotIn("require_all_gpus", checked["collection_budget"])
        self.assertEqual(
            checked["topology"]["total_environment_streams"]
            * checked["topology"]["steps_per_environment"],
            checked["topology"]["nominal_bc_target_transitions"],
        )
        expected_entrypoints = {
            "actor": "scripts/round_dagger/run_native_h200_actor_fleet.py",
            "learner": "scripts/round_dagger/run_native_h200_learner_fleet.py",
            "validation": "scripts/round_dagger/run_native_h200_mse_fleet.py",
            "test": "scripts/round_dagger/run_native_h200_test_fleet.py",
        }
        for role, entrypoint in expected_entrypoints.items():
            launch = checked["launch"][role]
            self.assertEqual(launch["entrypoint"], entrypoint)
            if role == "actor":
                self.assertEqual(launch["parallel_processes"], 8)
                self.assertEqual(launch["gpus_per_process"], 1)
            else:
                self.assertEqual(launch["parallel_processes"], 1)
                self.assertEqual(launch["gpus_per_process"], 8)
        self.assertEqual(
            checked["launch"]["actor"]["fixed_environment"][
                "ROUND_DAGGER_ACTOR_WORLD_SIZE"
            ],
            "8",
        )
        for role in ("validation", "test"):
            self.assertEqual(checked["launch"][role]["gpus_per_process"], 8)
        self.assertEqual(
            checked["launch"]["validation"]["fixed_environment"][
                "ROUND_DAGGER_EVALUATION_PURPOSE"
            ],
            "validation_mse",
        )
        self.assertEqual(
            checked["launch"]["test"]["fixed_environment"][
                "ROUND_DAGGER_EVALUATION_PURPOSE"
            ],
            "test_success",
        )
        self.assertNotIn("closed_loop", json.dumps(checked).lower())
        self.initialize()
        self.assertFalse(any(self.root.rglob("validation/closed-loop")))
        self.assertFalse(hasattr(self.controller, "record_closed_loop_validation"))
        self.assertFalse(hasattr(self.controller, "finalize_round"))

    def test_contract_rejects_contradictory_bc_physical_pool_policy(self) -> None:
        mutations = {
            "host_physical_gpu_count": 7,
            "admitted_physical_gpu_indices": [1],
            "physical_execution_concurrency": 2,
            "virtual_actor_world_size": 2,
            "actor_wave_scheduling": "all_at_once",
        }
        for key, value in mutations.items():
            bad = copy.deepcopy(self.contract)
            bad["collection_budget"][key] = value
            with self.subTest(key=key), self.assertRaises(
                protocol.NativeH200ExperimentError
            ):
                protocol.validate_contract(bad)

    def test_launch_plans_bind_stage_inputs_without_heldout_training_data(self) -> None:
        state = self.initialize()
        self.assertEqual(state["phase"], "bc_running")
        learner = self.controller.launch_plan("learner")
        self.assertEqual(
            set(learner["bound_inputs"]),
            {"train_replay_manifest", "train_replay_snapshot", "parent_checkpoint"},
        )
        self.assertEqual(
            learner["bound_inputs"]["train_replay_snapshot"],
            {
                "path": str(self.snapshot_arguments(0)["train_replay_snapshot_path"]),
                "file_sha256": self.snapshot_arguments(0)[
                    "train_replay_snapshot_file_sha256"
                ],
                "snapshot_sha256": self.snapshot_arguments(0)[
                    "train_replay_snapshot_sha256"
                ],
                "through_macro_round": 0,
            },
        )
        self.assertEqual(
            learner["bound_inputs"]["parent_checkpoint"]["sha256"],
            sha("initial-student"),
        )
        self.assertNotIn("validation", json.dumps(learner["bound_inputs"]))
        self.assertNotIn("test", json.dumps(learner["bound_inputs"]))
        self.assertEqual(
            learner["forbidden_data_bindings"],
            ["validation_registry", "test_registry"],
        )
        self.assertEqual(len(learner["commands"]), 1)
        self.assertEqual(
            learner["commands"][0]["environment"]["CUDA_VISIBLE_DEVICES"],
            "0,1,2,3,4,5,6,7",
        )

        candidate = self.checkpoints / "BC/training_steps/candidate.pt"
        candidate.write_bytes(b"candidate")
        validation = self.controller.launch_plan(
            "validation", checkpoint_sha256=sha("candidate")
        )
        self.assertEqual(len(validation["commands"]), 1)
        self.assertEqual(
            validation["bound_inputs"]["validation_registry"]["group_ids"],
            self.split["assignments"]["validation"],
        )
        self.assertEqual(
            validation["bound_inputs"]["validation_result_output"]["schema"],
            protocol.VALIDATION_MSE_RESULT_SCHEMA,
        )

        opened: list[tuple[Path, str]] = []
        original = protocol._load_pinned_canonical_json

        def spy(path: Path, digest: str, *, label: str) -> object:
            opened.append((Path(path), label))
            return original(path, digest, label=label)

        with mock.patch.object(protocol, "_load_pinned_canonical_json", side_effect=spy):
            with self.assertRaises(protocol.NativeH200ExperimentError):
                self.controller.record_test(
                    registry_path=self.test_path,
                    result_path=Path(self.temporary.name) / "not-yet.json",
                    expected_result_file_sha256=sha("not-yet"),
                )
        self.assertEqual(opened, [])
        with self.assertRaises(protocol.NativeH200ExperimentError):
            self.controller.launch_plan("test")

    def test_plateau_uses_min_delta_only_for_patience_and_literal_mse_for_best(self) -> None:
        rule = self.contract["optimization"]["stopping_rule"]
        points: list[dict[str, object]] = []
        values = [1.1, 1.0, 0.9995, 0.9995, 0.9995, 0.9995, 0.9995, 0.9995]
        for index, mse in enumerate(values, 1):
            points.append(
                {
                    "optimizer_examples": index * 1024,
                    "newest_source_examples": index * 1024,
                    "prior_source_examples": 0,
                    "newest_source_population": 4096,
                    "prior_source_population": 0,
                    "balanced_exposure_unit_examples": 4096,
                    "balanced_exposure_units": index * 0.25,
                    "optimizer_examples_per_step": 128,
                    "validation_mse": mse,
                    "checkpoint_sha256": sha(f"point-{index}"),
                }
            )
        decision = protocol.plateau_decision(points, rule, stage="bc")
        self.assertTrue(decision["stop"])
        self.assertEqual(decision["reason"], "plateau")
        self.assertEqual(decision["best_checkpoint_sha256"], sha("point-3"))
        self.assertEqual(decision["best_validation_mse"], 0.9995)
        self.assertEqual(decision["non_improving_evaluations"], 6)

        malformed = copy.deepcopy(points[0])
        malformed["optimizer_examples"] = 2048
        malformed["newest_source_examples"] = 2048
        with self.assertRaises(protocol.NativeH200ExperimentError):
            protocol.plateau_decision([malformed], rule, stage="bc")

    def test_validation_admits_only_hash_pinned_immutable_mse_results(self) -> None:
        self.initialize()
        checkpoint = sha("admitted-checkpoint")
        path, file_sha256, _ = self.validation_result(
            prefix="wrong-digest",
            index=1,
            mse=1.0,
            checkpoint_sha256=checkpoint,
        )
        with self.assertRaisesRegex(
            protocol.NativeH200ExperimentError, "file SHA-256 differs"
        ):
            self.controller.record_validation(
                result_path=path,
                expected_file_sha256=sha("wrong-digest"),
            )

        mutable, mutable_sha256, _ = self.validation_result(
            prefix="mutable",
            index=1,
            mse=1.0,
            checkpoint_sha256=checkpoint,
        )
        mutable.chmod(0o644)
        with self.assertRaisesRegex(protocol.NativeH200ExperimentError, "not immutable"):
            self.controller.record_validation(
                result_path=mutable,
                expected_file_sha256=mutable_sha256,
            )

        invalid, invalid_sha256, _ = self.validation_result(
            prefix="invalid-lineage",
            index=1,
            mse=1.0,
            checkpoint_sha256=checkpoint,
            overrides={"split_sha256": sha("other-split")},
        )
        with self.assertRaises(protocol.NativeH200ExperimentError):
            self.controller.record_validation(
                result_path=invalid,
                expected_file_sha256=invalid_sha256,
            )

        admitted, admitted_sha256, document = self.validation_result(
            prefix="admitted",
            index=1,
            mse=1.0,
            checkpoint_sha256=checkpoint,
        )
        state = self.controller.record_validation(
            result_path=admitted,
            expected_file_sha256=admitted_sha256,
        )
        self.assertEqual(len(state["history"]), 1)
        event = protocol._load_json(sorted(self.controller.events.glob("*.json"))[-1])
        self.assertEqual(event["payload"]["result"], document)
        with self.assertRaises(TypeError):
            self.controller.record_validation(  # type: ignore[call-arg]
                optimizer_examples=1024,
                validation_mse=1.0,
            )

    def test_every_stage_best_is_next_parent_and_global_argmin_stops_outer_loop(self) -> None:
        self.initialize()
        bc_checkpoint = self.saturate("bc", 0.8)
        state = self.controller.status()
        self.assertEqual(state["phase"], "bc_complete")
        self.assertEqual(state["active_checkpoint_sha256"], bc_checkpoint)
        self.assertEqual(state["selected_checkpoint_sha256"], bc_checkpoint)

        actor = self.controller.launch_plan("actor")
        self.assertEqual(len(actor["commands"]), 8)
        self.assertEqual(
            actor["bound_inputs"]["checkpoint"]["sha256"], bc_checkpoint
        )
        self.assertEqual(
            actor["bound_inputs"]["train_group_split"]["group_ids"],
            self.split["assignments"]["train"],
        )

        train_group = self.split["assignments"]["train"][0]
        expected_selected_mse = (0.7, 0.7, 0.6995)
        previous_parent = bc_checkpoint
        final_checkpoint = ""
        for round_index, best_mse in enumerate((0.7, 0.7005, 0.6995), 1):
            self.controller.start_round(
                new_replay_sha256=sha(f"round-{round_index}-replay"),
                collection_groups=[train_group],
                **self.snapshot_arguments(round_index),
            )
            learner = self.controller.launch_plan("learner")
            self.assertEqual(
                learner["bound_inputs"]["parent_checkpoint"]["sha256"],
                previous_parent,
            )
            round_checkpoint = self.saturate(f"round-{round_index}", best_mse)
            state = self.controller.status()
            self.assertEqual(state["active_checkpoint_sha256"], round_checkpoint)
            self.assertEqual(state["active_validation_mse"], best_mse)
            self.assertEqual(
                state["selected_validation_mse"], expected_selected_mse[round_index - 1]
            )
            previous_parent = round_checkpoint
            final_checkpoint = round_checkpoint

        self.assertEqual(state["phase"], "training_complete")
        self.assertEqual(state["outer_stop_reason"], "no_progress")
        self.assertEqual(state["no_progress_rounds"], 2)
        self.assertEqual(state["selected_checkpoint_sha256"], final_checkpoint)
        final = self.controller.select_final()
        self.assertEqual(final["final_checkpoint_sha256"], final_checkpoint)
        self.assertEqual(final["final_generation_id"], protocol.GENERATION_IDS[3])

    def test_final_test_is_success_only_hash_pinned_and_admitted_once(self) -> None:
        checkpoint, generation = self.complete_training()
        plan = self.controller.launch_plan("test")
        self.assertEqual(len(plan["commands"]), 1)
        self.assertEqual(
            plan["commands"][0]["environment"]["CUDA_VISIBLE_DEVICES"],
            "0,1,2,3,4,5,6,7",
        )
        self.assertEqual(
            plan["bound_inputs"]["checkpoint"]["sha256"], checkpoint
        )
        self.assertEqual(
            plan["bound_inputs"]["test_result_output"]["schema"],
            protocol.FINAL_TEST_RESULT_SCHEMA,
        )

        bad_path, bad_sha256, _ = self.final_test_result(
            checkpoint=checkpoint,
            generation=generation,
            extra={"validation_mse": 0.1},
        )
        with self.assertRaises(protocol.NativeH200ExperimentError):
            self.controller.record_test(
                registry_path=self.test_path,
                result_path=bad_path,
                expected_result_file_sha256=bad_sha256,
            )

        result_path, result_sha256, result = self.final_test_result(
            checkpoint=checkpoint,
            generation=generation,
        )
        with self.assertRaisesRegex(
            protocol.NativeH200ExperimentError, "file SHA-256 differs"
        ):
            self.controller.record_test(
                registry_path=self.test_path,
                result_path=result_path,
                expected_result_file_sha256=sha("wrong-result"),
            )

        opened: list[tuple[Path, str]] = []
        original = protocol._load_pinned_canonical_json

        def spy(path: Path, digest: str, *, label: str) -> object:
            opened.append((Path(path), label))
            return original(path, digest, label=label)

        with mock.patch.object(protocol, "_load_pinned_canonical_json", side_effect=spy):
            state = self.controller.record_test(
                registry_path=self.test_path,
                result_path=result_path,
                expected_result_file_sha256=result_sha256,
            )
        self.assertEqual(
            opened,
            [(self.test_path, "test registry"), (result_path, "final test result")],
        )
        self.assertEqual(state["phase"], "tested")
        self.assertEqual(state["test_metrics"]["success_rate"], 0.75)
        self.assertNotIn("mse", json.dumps(state["test_metrics"]).lower())
        event = protocol._load_json(sorted(self.controller.events.glob("*.json"))[-1])
        self.assertEqual(event["payload"]["result"], result)
        self.assertEqual(event["payload"]["test_result_file_sha256"], result_sha256)

        opened.clear()
        with mock.patch.object(protocol, "_load_pinned_canonical_json", side_effect=spy):
            with self.assertRaises(protocol.NativeH200ExperimentError):
                self.controller.record_test(
                    registry_path=self.test_path,
                    result_path=result_path,
                    expected_result_file_sha256=result_sha256,
                )
        self.assertEqual(opened, [])

        with self.assertRaises(TypeError):
            self.controller.record_test(  # type: ignore[call-arg]
                registry_path=self.test_path,
                success_count=30,
                safety_violations=1,
            )
        for path in (
            self.controller.contract_path,
            self.controller.split_path,
            *self.controller.events.iterdir(),
        ):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o444)


if __name__ == "__main__":
    unittest.main()
