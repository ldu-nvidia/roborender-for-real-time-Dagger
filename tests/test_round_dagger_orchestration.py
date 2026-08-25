import json
import sys
import tempfile
import unittest
from pathlib import Path

from round_dagger.artifacts import publish_immutable_json, sha256_file
from round_dagger.orchestration import (
    COHORT_COUNT,
    ENVIRONMENTS_PER_COHORT,
    EVALUATOR_RESULT_SCHEMA,
    HookSpec,
    MacroRoundConfig,
    MacroRoundOrchestrator,
    ProductionBackend,
    ORCHESTRATION_CONFIG_SCHEMA,
    OrchestrationError,
    STEPS_PER_ENVIRONMENT,
    TARGET_ENVIRONMENTS,
    TRANSITIONS_PER_COHORT,
    TRANSITIONS_PER_MACRO_ROUND,
    WANDB_PUBLIC_CONFIG_SCHEMA,
    _public_wandb_config,
    validate_production_topology,
)
from scripts.scaling import wandb_run_watcher
from tests.policy_bundle_helpers import publish_policy_bundle


SHA = {
    "runtime": "1" * 64,
    "learner_code": "2" * 64,
    "learner_config": "3" * 64,
    "model_lineage": "4" * 64,
    "evaluator_config": "5" * 64,
    "policy": "6" * 64,
    "teacher": "7" * 64,
    "renderer": "8" * 64,
    "beta": "9" * 64,
    "camera": "a" * 64,
    "training": "b" * 64,
    "orchestration": "c" * 64,
    "snapshot_file": "d" * 64,
    "snapshot_semantic": "e" * 64,
    "candidate": "f" * 64,
}


def campaign():
    return {
        "campaign_id": "production-campaign",
        "target_environments": TARGET_ENVIRONMENTS,
        "active_environments": ENVIRONMENTS_PER_COHORT,
        "cohort_count": COHORT_COUNT,
        "steps_per_environment": STEPS_PER_ENVIRONMENT,
    }


def contract():
    return {
        "macro_round": 0,
        "policy_version": 0,
        "policy_checkpoint_path": "policies/policy_000000/checkpoint.pt",
        "policy_checkpoint_sha256": SHA["policy"],
        "policy_runtime_sha256": SHA["runtime"],
        "policy_spec_sha256": "d" * 64,
        "policy_bundle_manifest_sha256": "e" * 64,
        "teacher_sha256": SHA["teacher"],
        "renderer_sha256": SHA["renderer"],
        "camera_mapping_sha256": SHA["camera"],
        "training_config_sha256": SHA["training"],
        "beta": 0.5,
        "beta_sha256": SHA["beta"],
        "steps_per_environment": STEPS_PER_ENVIRONMENT,
        "policy_image_shape": [324, 192, 3],
        "view_names": ["head_color", "hand_left", "hand_right"],
        "action_dim": 26,
        "cohorts": [
            {
                "cohort_index": index,
                "environment_start": index * ENVIRONMENTS_PER_COHORT,
                "environment_count": ENVIRONMENTS_PER_COHORT,
            }
            for index in range(COHORT_COUNT)
        ],
    }


def hook() -> HookSpec:
    return HookSpec((str(Path(sys.executable).resolve()), "-c", "pass"), 30)


def config(root: Path, *, transfer=True, ingest=True, promote=True):
    hooks = {
        "collect": hook(),
        "transfer": hook() if transfer else None,
        "ingest": hook() if ingest else None,
        "train": hook(),
        "evaluate": hook(),
        "promote": hook() if promote else None,
    }
    return MacroRoundConfig(
        campaign_root=root,
        actor_root=root / "actor",
        macro_round=0,
        runtime_policy_sha256=SHA["runtime"],
        learner_code_sha256=SHA["learner_code"],
        learner_config_sha256=SHA["learner_config"],
        learner_model_lineage_sha256=SHA["policy"],
        evaluator_config_sha256=SHA["evaluator_config"],
        hooks=hooks,
        sha256=SHA["orchestration"],
    )


class FakeBackend:
    def __init__(self, configuration):
        self.config = configuration
        self.campaign = campaign()
        self.contract = contract()
        self.phase = "READY"
        self.revision = 0
        self.next_round = 0
        self.open_round_number = None
        self.cohorts = {}
        self.collection = None
        self.snapshot = None
        self.candidate = None
        self.evaluation = None
        self.collection_sealed = False
        self.ingest_started_after_collection = False

    def load_campaign(self):
        return dict(self.campaign)

    def load_state(self):
        result = {
            "phase": self.phase,
            "revision": self.revision,
            "next_macro_round": self.next_round,
            "open_macro_round": self.open_round_number,
            "active_policy": {"version": self.next_round},
            "replay_snapshot_sha256": (
                self.snapshot["file_sha256"] if self.snapshot else None
            ),
        }
        if self.candidate is not None:
            result["candidate_policy"] = {
                "manifest_path": self.candidate["manifest_path"]
            }
        return result

    def open_round(self):
        self.phase = "COLLECTING"
        self.open_round_number = 0
        self.revision += 1
        return dict(self.contract)

    def load_contract(self):
        return dict(self.contract)

    def ensure_cohort_contract(self, path, **_kwargs):
        path.parent.mkdir(parents=True, exist_ok=True)

    def _report(self, index):
        start = index * ENVIRONMENTS_PER_COHORT
        return {
            "campaign_id": self.campaign["campaign_id"],
            "macro_round": 0,
            "cohort_index": index,
            "purpose": "train",
            "sealed": True,
            "expected_ranks": [0, 1],
            "ranks": [0, 1],
            "expected_global_env_ids": list(
                range(start, start + ENVIRONMENTS_PER_COHORT)
            ),
            "global_env_ids": list(range(start, start + ENVIRONMENTS_PER_COHORT)),
            "expected_transitions": TRANSITIONS_PER_COHORT,
            "transition_count": TRANSITIONS_PER_COHORT,
            "policy_sha256": SHA["policy"],
            "runtime_policy_sha256": SHA["runtime"],
            "teacher_sha256": SHA["teacher"],
            "renderer_sha256": SHA["renderer"],
            "beta_sha256": SHA["beta"],
            "config_sha256": SHA["training"],
            "cohort_sha256": f"{index + 1:064x}",
        }

    def complete_cohort(self, path):
        return self.cohorts.get(str(path))

    def seal_cohort(self, path):
        index = int(path.name.rsplit("_", 1)[1])
        report = self._report(index)
        self.cohorts[str(path)] = report
        return report

    def publish_transfer(self, source, destination):
        report = dict(self.cohorts[str(source)])
        self.cohorts[str(destination)] = report

    def transfer_local(self, source, destination):
        self.publish_transfer(source, destination)
        return self.cohorts[str(destination)]

    def seal_collection(self, paths):
        if len(paths) != COHORT_COUNT:
            raise AssertionError("orchestrator did not provide exactly four cohorts")
        self.collection_sealed = True
        self.collection = {
            "manifest_path": "rounds/macro_round_000000/collection.json",
            "manifest_sha256": "0" * 64,
            "cohort_count": COHORT_COUNT,
            "transition_count": TRANSITIONS_PER_MACRO_ROUND,
            "cohorts": [
                {
                    "cohort_index": index,
                    "cohort_manifest_path": (
                        f"rounds/macro_round_000000/cohorts/cohort_{index:02d}"
                    ),
                }
                for index in range(COHORT_COUNT)
            ],
        }
        return dict(self.collection)

    def mark_collected(self, _collection):
        if not self.collection_sealed:
            raise AssertionError("COLLECTED advanced before strict aggregate seal")
        self.phase = "COLLECTED"
        self.revision += 1
        return self.load_state()

    def load_collection(self):
        return dict(self.collection)

    def ingest_collection(self, _collection):
        self.ingest_started_after_collection = self.phase == "COLLECTED"
        if not self.ingest_started_after_collection:
            raise AssertionError("ingest ran before collection state transition")
        self.snapshot = {
            "path": str(self.config.campaign_root / "replay" / "snapshot.json"),
            "file_sha256": SHA["snapshot_file"],
            "snapshot_sha256": SHA["snapshot_semantic"],
            "expected_train_cohorts": COHORT_COUNT,
            "transition_count": TRANSITIONS_PER_MACRO_ROUND,
            "cohort_count": COHORT_COUNT,
        }
        return dict(self.snapshot)

    def mark_ingested(self, _snapshot):
        self.phase = "INGESTED"
        self.revision += 1
        return self.load_state()

    def load_snapshot(self, **_kwargs):
        return dict(self.snapshot)

    def learner_output_dir(self):
        return self.config.campaign_root / "learner"

    def learner_result_path(self):
        return self.learner_output_dir() / "learner_result_round_000000.json"

    def load_learner_result(self, **_kwargs):
        return {
            "absolute_checkpoint_path": str(self.learner_output_dir() / "candidate.pt"),
            "optimizer_steps": 12,
            "initial_loss": 2.0,
            "final_loss": 1.0,
        }

    def stage_candidate(self, _learner_result, _snapshot):
        self.candidate = {
            "policy_version": 1,
            "macro_round": 0,
            "manifest_path": "policies/policy_000001/candidate.json",
            "manifest_sha256": SHA["candidate"],
            "optimizer_steps": 12,
        }
        return dict(self.candidate)

    def mark_trained(self, _candidate):
        self.phase = "TRAINED"
        self.revision += 1
        return self.load_state()

    def load_candidate(self):
        return dict(self.candidate)

    def evaluator_result_path(self):
        return self.config.campaign_root / "evaluation" / "result.json"

    def load_evaluator_result(self, **_kwargs):
        return {
            "schema": EVALUATOR_RESULT_SCHEMA,
            "passed": True,
            "metrics": {"success_rate": 0.75},
        }

    def publish_evaluation(self, _candidate, _result):
        self.evaluation = {
            "manifest_path": "policies/policy_000001/evaluation.json",
            "evidence_sha256": "a" * 64,
        }
        return dict(self.evaluation)

    def mark_evaluated(self, _evidence):
        self.phase = "EVALUATED"
        self.revision += 1
        return self.load_state()

    def load_evaluation(self, _candidate):
        return dict(self.evaluation)

    def promote(self, _candidate, _evaluation):
        self.phase = "READY"
        self.next_round = 1
        self.open_round_number = None
        self.revision += 1
        return {
            "schema": "fast_roborender.round_dagger.policy_promotion.v1",
            "macro_round": 0,
            "policy_version": 1,
        }


class RecordingRunner:
    def __init__(self, backend, *, fail_on=None):
        self.backend = backend
        self.fail_on = fail_on
        self.calls = []

    def run(self, name, _hook, *, environment, workdir):
        cohort = environment.get("ROUND_DAGGER_COHORT_INDEX")
        identity = (name, None if cohort is None else int(cohort))
        self.calls.append((identity, dict(environment), workdir))
        if identity == self.fail_on:
            raise OrchestrationError("injected hook failure")
        if name == "transfer":
            self.backend.publish_transfer(
                Path(environment["ROUND_DAGGER_SOURCE_COHORT"]),
                Path(environment["ROUND_DAGGER_DESTINATION_COHORT"]),
            )
        elif name == "train":
            result = Path(environment["ROUND_DAGGER_LEARNER_RESULT"])
            result.parent.mkdir(parents=True, exist_ok=True)
            result.write_text("{}", encoding="utf-8")
        elif name == "evaluate":
            result = Path(environment["ROUND_DAGGER_EVALUATOR_RESULT"])
            result.parent.mkdir(parents=True, exist_ok=True)
            result.write_text("{}", encoding="utf-8")


class ConfigAndTopologyTest(unittest.TestCase):
    def test_learner_result_requires_exact_colocated_policy_sidecars(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configuration = config(root)
            backend = ProductionBackend(configuration)
            output = backend.learner_output_dir()
            output.mkdir(parents=True)
            checkpoint = output / "candidate_round_000000.pt"
            checkpoint.write_bytes(b"candidate")
            spec, bundle_manifest = publish_policy_bundle(
                checkpoint, SHA["runtime"]
            )
            result_path = backend.learner_result_path()
            publish_immutable_json(
                result_path,
                {
                    "schema": "fast_roborender.round_dagger.learner_result.v2",
                    "campaign_id": "production-campaign",
                    "macro_round": 0,
                    "checkpoint_path": checkpoint.name,
                    "checkpoint_sha256": sha256_file(checkpoint),
                    "runtime_policy_sha256": SHA["runtime"],
                    "policy_spec_path": spec.name,
                    "policy_spec_sha256": sha256_file(spec),
                    "policy_bundle_manifest_path": bundle_manifest.name,
                    "policy_bundle_manifest_sha256": sha256_file(bundle_manifest),
                    "model_lineage_sha256": SHA["policy"],
                    "replay_snapshot_sha256": SHA["snapshot_semantic"],
                    "code_sha256": SHA["learner_code"],
                    "learner_config_sha256": SHA["learner_config"],
                    "optimizer_steps": 12,
                    "initial_loss": 2.0,
                    "final_loss": 1.0,
                },
            )
            loaded = backend.load_learner_result(
                campaign={"campaign_id": "production-campaign"},
                snapshot={"snapshot_sha256": SHA["snapshot_semantic"]},
            )
            self.assertEqual(loaded["runtime_policy_sha256"], SHA["runtime"])
            self.assertEqual(loaded["absolute_policy_spec_path"], str(spec))

            spec.chmod(0o600)
            spec.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(OrchestrationError, "sidecar SHA-256"):
                backend.load_learner_result(
                    campaign={"campaign_id": "production-campaign"},
                    snapshot={"snapshot_sha256": SHA["snapshot_semantic"]},
                )

    def test_evaluator_result_v2_requires_canonical_digest_bound_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configuration = config(root)
            backend = ProductionBackend(configuration)
            bundle = (
                root
                / "rounds"
                / "macro_round_000000"
                / "evaluation"
                / "results"
                / "evaluation_bundle.json"
            )
            bundle_sha = publish_immutable_json(
                bundle,
                {
                    "schema": "test.paired_evaluation_bundle.v1",
                    "candidate_manifest_sha256": SHA["candidate"],
                },
            )
            result_path = backend.evaluator_result_path()
            publish_immutable_json(
                result_path,
                {
                    "schema": EVALUATOR_RESULT_SCHEMA,
                    "campaign_id": "production-campaign",
                    "macro_round": 0,
                    "candidate_manifest_sha256": SHA["candidate"],
                    "evaluator_config_sha256": SHA["evaluator_config"],
                    "evidence_bundle_path": bundle.relative_to(root).as_posix(),
                    "evidence_bundle_sha256": bundle_sha,
                    "passed": True,
                    "metrics": {"paired_cases": 8.0},
                },
            )
            loaded = backend.load_evaluator_result(
                campaign={"campaign_id": "production-campaign"},
                candidate={"manifest_sha256": SHA["candidate"]},
            )
            self.assertEqual(loaded["evidence_bundle_sha256"], bundle_sha)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configuration = config(root)
            backend = ProductionBackend(configuration)
            noncanonical = root / "evaluation_bundle.json"
            bundle_sha = publish_immutable_json(noncanonical, {"schema": "test"})
            publish_immutable_json(
                backend.evaluator_result_path(),
                {
                    "schema": EVALUATOR_RESULT_SCHEMA,
                    "campaign_id": "production-campaign",
                    "macro_round": 0,
                    "candidate_manifest_sha256": SHA["candidate"],
                    "evaluator_config_sha256": SHA["evaluator_config"],
                    "evidence_bundle_path": noncanonical.relative_to(root).as_posix(),
                    "evidence_bundle_sha256": bundle_sha,
                    "passed": True,
                    "metrics": {"paired_cases": 8.0},
                },
            )
            with self.assertRaisesRegex(OrchestrationError, "non-canonical"):
                backend.load_evaluator_result(
                    campaign={"campaign_id": "production-campaign"},
                    candidate={"manifest_sha256": SHA["candidate"]},
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configuration = config(root)
            backend = ProductionBackend(configuration)
            bundle = (
                root
                / "rounds/macro_round_000000/evaluation/results/evaluation_bundle.json"
            )
            publish_immutable_json(bundle, {"schema": "test"})
            publish_immutable_json(
                backend.evaluator_result_path(),
                {
                    "schema": EVALUATOR_RESULT_SCHEMA,
                    "campaign_id": "production-campaign",
                    "macro_round": 0,
                    "candidate_manifest_sha256": SHA["candidate"],
                    "evaluator_config_sha256": SHA["evaluator_config"],
                    "evidence_bundle_path": bundle.relative_to(root).as_posix(),
                    "evidence_bundle_sha256": "0" * 64,
                    "passed": True,
                    "metrics": {"paired_cases": 8.0},
                },
            )
            with self.assertRaisesRegex(OrchestrationError, "SHA mismatch"):
                backend.load_evaluator_result(
                    campaign={"campaign_id": "production-campaign"},
                    candidate={"manifest_sha256": SHA["candidate"]},
                )

    def test_config_is_strict_and_rejects_credentials_in_argv(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hook_mapping = {
                "argv": [str(Path(sys.executable).resolve()), "-c", "pass"],
                "timeout_seconds": 30,
            }
            document = {
                "schema": ORCHESTRATION_CONFIG_SCHEMA,
                "campaign_root": str(root),
                "actor_root": str(root / "actor"),
                "macro_round": 0,
                "runtime_policy_sha256": SHA["runtime"],
                "learner_code_sha256": SHA["learner_code"],
                "learner_config_sha256": SHA["learner_config"],
                "learner_model_lineage_sha256": SHA["model_lineage"],
                "evaluator_config_sha256": SHA["evaluator_config"],
                "hooks": {
                    "collect": hook_mapping,
                    "transfer": None,
                    "ingest": None,
                    "train": hook_mapping,
                    "evaluate": hook_mapping,
                    "promote": None,
                },
            }
            path = root / "config.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            loaded = MacroRoundConfig.load(path)
            self.assertEqual(loaded.macro_round, 0)
            self.assertIsNone(loaded.hooks["transfer"])

            document["hooks"]["collect"] = {
                "argv": [
                    str(Path(sys.executable).resolve()),
                    "--api-key",
                    "do-not-log",
                ],
                "timeout_seconds": 30,
            }
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(OrchestrationError, "credential"):
                MacroRoundConfig.load(path)

    def test_topology_is_exact_and_not_tunable(self):
        validate_production_topology(campaign(), contract())
        wrong = campaign()
        wrong["target_environments"] = 255
        with self.assertRaisesRegex(OrchestrationError, "target_environments=256"):
            validate_production_topology(wrong)
        wrong_contract = contract()
        wrong_contract["cohorts"][3]["environment_count"] = 63
        with self.assertRaisesRegex(OrchestrationError, "0..255"):
            validate_production_topology(campaign(), wrong_contract)

    def test_public_wandb_projection_has_a_fixed_non_secret_allowlist(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configuration = config(root)
            document = _public_wandb_config(
                config=configuration,
                campaign=campaign(),
                contract=contract(),
                cohort_index=2,
            )
            path = root / "public.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            loaded = wandb_run_watcher.load_public_config(path)
            self.assertEqual(loaded["round_dagger/cohort_index"], 2)
            serialized = json.dumps(document).lower()
            self.assertNotIn("api_key", serialized)
            self.assertNotIn(str(root).lower(), serialized)
            self.assertNotIn("argv", serialized)

            document["config"]["wandb_api_key"] = "secret"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(
                wandb_run_watcher.ArtifactError, "unknown=.*wandb_api_key"
            ):
                wandb_run_watcher.load_public_config(path)

    def test_optional_sidecar_uses_public_config_and_never_passes_api_key(self):
        source = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "round_dagger"
            / "run_actor_cohort_with_wandb.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("WANDB_PUBLIC_CONFIG.json", source)
        self.assertIn("wandb_run_watcher.py", source)
        self.assertIn("--authorization-scope team", source)
        self.assertIn("RUN_DONE", source)
        self.assertNotIn("WANDB_API_KEY", source)
        self.assertNotIn("set -x", source)


class MacroRoundExecutionTest(unittest.TestCase):
    def test_collect_hook_receives_complete_production_topology_environment(self):
        """The production actor hook must not reconstruct global topology."""

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configuration = config(root)
            backend = FakeBackend(configuration)
            runner = RecordingRunner(backend)
            MacroRoundOrchestrator(
                configuration, backend=backend, hook_runner=runner
            ).run()

            collect_environments = [
                environment
                for (name, _), environment, _workdir in runner.calls
                if name == "collect"
            ]
            self.assertEqual(len(collect_environments), COHORT_COUNT)
            common_topology = {
                "ROUND_DAGGER_TARGET_ENVIRONMENTS": str(TARGET_ENVIRONMENTS),
                "ROUND_DAGGER_COHORT_COUNT": str(COHORT_COUNT),
                "ROUND_DAGGER_ENVIRONMENTS_PER_COHORT": str(
                    ENVIRONMENTS_PER_COHORT
                ),
                "ROUND_DAGGER_STEPS_PER_ENVIRONMENT": str(
                    STEPS_PER_ENVIRONMENT
                ),
                "ROUND_DAGGER_TRANSITIONS_PER_MACRO_ROUND": str(
                    TRANSITIONS_PER_MACRO_ROUND
                ),
            }
            for cohort_index, environment in enumerate(collect_environments):
                with self.subTest(cohort_index=cohort_index):
                    self.assertEqual(
                        {
                            key: environment.get(key)
                            for key in common_topology
                        },
                        common_topology,
                    )
                    self.assertEqual(
                        environment.get("ROUND_DAGGER_GLOBAL_ENV_OFFSET"),
                        str(cohort_index * ENVIRONMENTS_PER_COHORT),
                    )
                    self.assertEqual(
                        environment.get("ROUND_DAGGER_LOGICAL_ENV_COUNT"),
                        str(ENVIRONMENTS_PER_COHORT),
                    )
                    self.assertEqual(
                        environment.get("ROUND_DAGGER_EXPECTED_TRANSITIONS"),
                        str(TRANSITIONS_PER_COHORT),
                    )

    def test_hooks_are_sequential_and_promotion_follows_strict_aggregate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configuration = config(root)
            backend = FakeBackend(configuration)
            runner = RecordingRunner(backend)
            result = MacroRoundOrchestrator(
                configuration, backend=backend, hook_runner=runner
            ).run()

            self.assertEqual(result["policy_version"], 1)
            self.assertEqual(
                [identity for identity, _environment, _workdir in runner.calls],
                [
                    ("collect", 0),
                    ("transfer", 0),
                    ("collect", 1),
                    ("transfer", 1),
                    ("collect", 2),
                    ("transfer", 2),
                    ("collect", 3),
                    ("transfer", 3),
                    ("ingest", None),
                    ("train", None),
                    ("evaluate", None),
                    ("promote", None),
                ],
            )
            collect_environments = [
                environment
                for (name, _), environment, _workdir in runner.calls
                if name == "collect"
            ]
            self.assertEqual(
                [int(item["ROUND_DAGGER_GLOBAL_ENV_OFFSET"]) for item in collect_environments],
                [0, 64, 128, 192],
            )
            for environment in collect_environments:
                checkpoint = environment["ROUND_DAGGER_POLICY_CHECKPOINT"]
                self.assertEqual(
                    environment["ROUND_DAGGER_POLICY_SPEC"],
                    f"{checkpoint}.policy_spec.json",
                )
                self.assertEqual(
                    environment["ROUND_DAGGER_POLICY_MANIFEST"],
                    f"{checkpoint}.manifest.json",
                )
                self.assertEqual(
                    environment["ROUND_DAGGER_POLICY_SPEC_SHA256"],
                    contract()["policy_spec_sha256"],
                )
                self.assertEqual(
                    environment["ROUND_DAGGER_POLICY_MANIFEST_SHA256"],
                    contract()["policy_bundle_manifest_sha256"],
                )
            self.assertTrue(backend.collection_sealed)
            self.assertTrue(backend.ingest_started_after_collection)
            self.assertEqual(backend.phase, "READY")
            self.assertEqual(backend.next_round, 1)

            public_config = (
                root
                / "rounds"
                / "macro_round_000000"
                / "orchestration_run"
                / "telemetry"
                / "WANDB_PUBLIC_CONFIG.json"
            )
            self.assertEqual(
                json.loads(public_config.read_text(encoding="utf-8"))["schema"],
                WANDB_PUBLIC_CONFIG_SCHEMA,
            )
            scalar_files = sorted(
                public_config.parent.joinpath("scalars").glob("*.json")
            )
            self.assertEqual(len(scalar_files), 10)

    def test_runtime_identity_must_equal_frozen_round_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configuration = config(root)
            backend = FakeBackend(configuration)
            backend.contract["policy_runtime_sha256"] = "0" * 64
            with self.assertRaisesRegex(
                OrchestrationError, "runtime policy SHA-256"
            ):
                MacroRoundOrchestrator(
                    configuration,
                    backend=backend,
                    hook_runner=RecordingRunner(backend),
                ).run()

    def test_hook_failure_stops_before_aggregate_ingest_train_and_promotion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configuration = config(root)
            backend = FakeBackend(configuration)
            runner = RecordingRunner(backend, fail_on=("collect", 2))
            with self.assertRaisesRegex(OrchestrationError, "injected hook failure"):
                MacroRoundOrchestrator(
                    configuration, backend=backend, hook_runner=runner
                ).run()

            self.assertEqual(
                [identity for identity, _environment, _workdir in runner.calls],
                [
                    ("collect", 0),
                    ("transfer", 0),
                    ("collect", 1),
                    ("transfer", 1),
                    ("collect", 2),
                ],
            )
            self.assertEqual(backend.phase, "COLLECTING")
            self.assertFalse(backend.collection_sealed)
            self.assertIsNone(backend.snapshot)
            self.assertIsNone(backend.candidate)

    def test_existing_sealed_cohort_is_verified_and_skipped_on_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configuration = config(root, transfer=False, ingest=False, promote=False)
            backend = FakeBackend(configuration)
            backend.open_round()
            actor_round = root / "actor" / "production-campaign" / "macro_round_000000"
            source = actor_round / "cohort_00"
            destination = (
                root
                / "rounds"
                / "macro_round_000000"
                / "cohorts"
                / "cohort_00"
            )
            backend.cohorts[str(source)] = backend._report(0)
            backend.cohorts[str(destination)] = dict(backend._report(0))
            runner = RecordingRunner(backend)
            MacroRoundOrchestrator(
                configuration, backend=backend, hook_runner=runner
            ).run()
            identities = [identity for identity, _environment, _workdir in runner.calls]
            self.assertNotIn(("collect", 0), identities)
            self.assertEqual(
                [item for item in identities if item[0] == "collect"],
                [("collect", 1), ("collect", 2), ("collect", 3)],
            )

    def test_resume_from_collected_reuses_immutable_telemetry_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configuration = config(root)
            backend = FakeBackend(configuration)
            interrupted = RecordingRunner(backend, fail_on=("ingest", None))
            with self.assertRaisesRegex(OrchestrationError, "injected hook failure"):
                MacroRoundOrchestrator(
                    configuration, backend=backend, hook_runner=interrupted
                ).run()
            self.assertEqual(backend.phase, "COLLECTED")

            resumed = RecordingRunner(backend)
            result = MacroRoundOrchestrator(
                configuration, backend=backend, hook_runner=resumed
            ).run()
            self.assertEqual(result["policy_version"], 1)
            identities = [
                identity for identity, _environment, _workdir in resumed.calls
            ]
            self.assertEqual(
                identities,
                [
                    ("ingest", None),
                    ("train", None),
                    ("evaluate", None),
                    ("promote", None),
                ],
            )


if __name__ == "__main__":
    unittest.main()
