"""Focused tests for live held-out scenario capture attestation."""

from __future__ import annotations

import copy
import ast
import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from round_dagger.production_data_lineage import (
    ProductionDataLineageError,
    RUNTIME_CONTRACT_SCHEMA,
    build_cohort_assignment,
    data_lineage_sha256,
    derive_data_lineage,
    load_verified_runtime_contract,
    rank_assignment,
)
from round_dagger.scenario_capture import (
    ATTESTATION_SCHEMA,
    EXPECTED_SEEDS,
    PLAN_SCHEMA,
    PROVENANCE_KEYS,
    RUNTIME_SCHEMA,
    ScenarioCaptureError,
    build_campaign_attestation,
    build_pass_attestation,
    canonical_sha256,
    validate_pass_document,
)


def sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def canonical_file(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


class ScenarioCaptureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.seed = 17
        cases = []
        for env_index in range(2):
            cases.append(
                {
                    "seed": self.seed,
                    "env_index": env_index,
                    "global_case_index": env_index,
                    "planned_case_id": sha(f"planned-{env_index}"),
                    "motion_relative_path": f"motion_{env_index:03d}.npz",
                    "motion_sha256": sha(f"motion-{env_index}"),
                }
            )
        self.split_sha = sha("split")
        self.plan = {
            "schema": PLAN_SCHEMA,
            "source": {"nested_split_manifest_sha256": self.split_sha},
            "capture": {"cases": cases},
        }
        self.plan_path = self.root / "plan.json"
        canonical_file(self.plan_path, self.plan)
        self.plan_sha = hashlib.sha256(self.plan_path.read_bytes()).hexdigest()
        self.renderer_identity = {
            "release_id": "test-release",
            "inference_steps": 5,
            "views": ["head_color", "hand_left", "hand_right"],
        }
        self.renderer_sha = canonical_sha256(self.renderer_identity)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def runtime(self, *, pass_index: int = 1):
        cases = []
        for planned in self.plan["capture"]["cases"]:
            env_index = planned["env_index"]
            condition = sha(f"condition-{env_index}")
            cases.append(
                {
                    **planned,
                    "first_condition_sha256": condition,
                    "first_generated_sha256": sha(f"generated-{env_index}"),
                    "scenario_id": canonical_sha256(
                        {
                            "seed": self.seed,
                            "env_index": env_index,
                            "first_condition_sha256": condition,
                        }
                    ),
                }
            )
        provenance = {
            "fast_roborender_revision": "a" * 40,
            "gr00t_revision": "b" * 40,
            "actor_source_inventory_sha256": sha("actor-source"),
            "evaluation_source_inventory_sha256": sha("eval-source"),
            "gr00t_source_tree_sha256": sha("gr00t-tree"),
            "heldout_plan_sha256": self.plan_sha,
            "motion_split_manifest_sha256": self.split_sha,
            "evaluation_config_sha256": sha("config"),
            "renderer_runtime_sha256": self.renderer_sha,
            "camera_acceptance_sha256": sha("camera"),
            "anchor_inventory_sha256": sha("anchor"),
            "trainer_container_sha256": sha("trainer-container"),
            "renderer_cohort_sha256": sha("renderer-cohort"),
            "parent_checkpoint_sha256": sha("parent"),
            "parent_runtime_policy_sha256": sha("parent-runtime"),
        }
        return {
            "schema": RUNTIME_SCHEMA,
            "plan_sha256": self.plan_sha,
            "seed": self.seed,
            "pass_index": pass_index,
            "capture_run_id": f"smoke-seed17-pass{pass_index}",
            "capture_nonce": f"{pass_index:032x}",
            "capture_profile": "smoke2",
            "release_eligible": False,
            "expected_env_count": 2,
            "cases": cases,
            "execution": {
                "capture_mode": True,
                "paired_evaluation": False,
                "eval_relabel_enabled": True,
                "training_enabled": False,
                "policy_step_calls": 1,
                "env_step_calls": 0,
                "generated_policy_steps": 1,
                "generated_frames": 2,
                "renderer_requests_delta": 2,
                "renderer_errors_delta": 0,
                "renderer_replays_delta": 0,
            },
            "runtime": {
                "num_envs": 2,
                "generation_seed": self.seed,
                "motion_split_role": "eval",
                "motion_split_offset": 0,
                "motion_split_limit": 2,
                "policy_views": ["head_color", "hand_left", "hand_right"],
                "view_layout": "vertical",
                "reference_lag": 256,
                "raytraced_mix_prob": 0.0,
                "anchor_source": "simulator_rgb",
            },
            "renderer_streams": [
                {
                    "env_index": env_index,
                    "endpoint_file": f"renderer_{env_index}.txt",
                    "server_id": f"renderer-{env_index}",
                    "server_incarnation_id": f"incarnation-{env_index}",
                }
                for env_index in range(2)
            ],
            "provenance": provenance,
        }

    def renderer_snapshots(self):
        before = {"selected": []}
        after = {"selected": []}
        for index in range(2):
            row = {
                "file": f"renderer_{index}.txt",
                "response": {
                    "server_id": f"renderer-{index}",
                    "runtime_identity": self.renderer_identity,
                    "metrics": {"requests": 10, "errors": 0, "replays": 0},
                },
            }
            before["selected"].append(row)
            later = copy.deepcopy(row)
            later["response"]["metrics"]["requests"] = 11
            after["selected"].append(later)
        return before, after

    def build_smoke_pass(self):
        runtime_path = self.root / "runtime.json"
        before_path = self.root / "before.json"
        after_path = self.root / "after.json"
        canonical_file(runtime_path, self.runtime())
        before, after = self.renderer_snapshots()
        canonical_file(before_path, before)
        canonical_file(after_path, after)
        return build_pass_attestation(
            plan_path=self.plan_path,
            runtime_path=runtime_path,
            renderer_before_path=before_path,
            renderer_after_path=after_path,
            production=False,
        )

    def test_two_env_smoke_attests_real_calls_but_is_non_release(self) -> None:
        capture = self.build_smoke_pass()
        normalized = validate_pass_document(
            capture,
            plan=self.plan,
            plan_sha256=self.plan_sha,
            production=False,
        )
        self.assertEqual(normalized["served_requests"], 2)
        self.assertFalse(normalized["runtime"]["release_eligible"])
        self.assertEqual(normalized["runtime"]["execution"]["env_step_calls"], 0)
        with self.assertRaisesRegex(ScenarioCaptureError, "exact"):
            validate_pass_document(
                capture,
                plan=self.plan,
                plan_sha256=self.plan_sha,
                production=True,
            )
        with self.assertRaisesRegex(ScenarioCaptureError, "never produce"):
            build_campaign_attestation(
                plan_path=self.plan_path,
                pass_paths=[],
                production=False,
            )

    def test_nested_and_outer_provenance_schemas_are_exactly_aligned(self) -> None:
        nested = (
            Path(__file__).resolve().parents[1]
            / "GR00T-VisualSim2Real/gr00t/rl/trl/trainer/distill_trainer_videogen.py"
        )
        tree = ast.parse(nested.read_text(encoding="utf-8"))
        observed = None
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "provenance"
                and isinstance(node.value, ast.Dict)
            ):
                observed = {
                    key.value
                    for key in node.value.keys
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                }
        self.assertEqual(observed, PROVENANCE_KEYS)

    def test_smoke2_accepts_first_two_cases_of_checked_in_8x32_plan(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        plan_path = repository / "configs/round_dagger/production_heldout_motion_plan_8x32.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        planned = [
            case
            for case in plan["capture"]["cases"]
            if case["seed"] == EXPECTED_SEEDS[0]
        ][:2]
        runtime = self.runtime()
        runtime["plan_sha256"] = plan_sha
        runtime["provenance"]["heldout_plan_sha256"] = plan_sha
        runtime["provenance"]["motion_split_manifest_sha256"] = plan["source"][
            "nested_split_manifest_sha256"
        ]
        runtime["cases"] = []
        for case in planned:
            condition = sha(f"actual-plan-condition-{case['global_case_index']}")
            runtime["cases"].append(
                {
                    **case,
                    "first_condition_sha256": condition,
                    "first_generated_sha256": sha(
                        f"actual-plan-generated-{case['global_case_index']}"
                    ),
                    "scenario_id": canonical_sha256(
                        {
                            "seed": case["seed"],
                            "env_index": case["env_index"],
                            "first_condition_sha256": condition,
                        }
                    ),
                }
            )
        runtime["runtime"]["motion_split_offset"] = planned[0]["global_case_index"]
        runtime_path = self.root / "checked_in_plan_smoke_runtime.json"
        before_path = self.root / "checked_in_plan_smoke_before.json"
        after_path = self.root / "checked_in_plan_smoke_after.json"
        canonical_file(runtime_path, runtime)
        before, after = self.renderer_snapshots()
        canonical_file(before_path, before)
        canonical_file(after_path, after)
        capture = build_pass_attestation(
            plan_path=plan_path,
            runtime_path=runtime_path,
            renderer_before_path=before_path,
            renderer_after_path=after_path,
            production=False,
        )
        self.assertEqual(capture["served_requests"], 2)
        self.assertFalse(capture["runtime"]["release_eligible"])

    def test_smoke_fails_closed_on_env_step_replay_and_condition_drift(self) -> None:
        capture = self.build_smoke_pass()
        stepped = copy.deepcopy(capture)
        stepped["runtime"]["execution"]["env_step_calls"] = 1
        with self.assertRaisesRegex(ScenarioCaptureError, "non-stepping"):
            validate_pass_document(
                stepped,
                plan=self.plan,
                plan_sha256=self.plan_sha,
                production=False,
            )

        replay = copy.deepcopy(capture)
        replay["renderer_after"]["selected"][0]["response"]["metrics"]["replays"] = 1
        # Recompute the embedded-document hash to prove counter validation, not
        # merely a stale hash, rejects the replay.
        replay["renderer_after_sha256"] = hashlib.sha256(
            (
                json.dumps(
                    replay["renderer_after"], sort_keys=True, separators=(",", ":")
                )
                + "\n"
            ).encode("utf-8")
        ).hexdigest()
        with self.assertRaisesRegex(ScenarioCaptureError, "replays"):
            validate_pass_document(
                replay,
                plan=self.plan,
                plan_sha256=self.plan_sha,
                production=False,
            )

        drift = copy.deepcopy(capture)
        drift["runtime"]["cases"][0]["first_condition_sha256"] = sha("drift")
        drift["runtime_artifact_sha256"] = hashlib.sha256(
            (
                json.dumps(drift["runtime"], sort_keys=True, separators=(",", ":"))
                + "\n"
            ).encode("utf-8")
        ).hexdigest()
        with self.assertRaisesRegex(ScenarioCaptureError, "condition-derived"):
            validate_pass_document(
                drift,
                plan=self.plan,
                plan_sha256=self.plan_sha,
                production=False,
            )

    def test_full_8x32_two_pass_capture_emits_builder_evidence(self) -> None:
        cases = []
        for seed_index, seed in enumerate(EXPECTED_SEEDS):
            for env_index in range(32):
                global_index = seed_index * 32 + env_index
                cases.append(
                    {
                        "seed": seed,
                        "env_index": env_index,
                        "global_case_index": global_index,
                        "planned_case_id": sha(f"prod-planned-{global_index}"),
                        "motion_relative_path": f"prod_motion_{global_index:03d}.npz",
                        "motion_sha256": sha(f"prod-motion-{global_index}"),
                    }
                )
        plan = {
            "schema": PLAN_SCHEMA,
            "source": {"nested_split_manifest_sha256": self.split_sha},
            "capture": {"cases": cases},
        }
        plan_path = self.root / "production_plan.json"
        canonical_file(plan_path, plan)
        plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        pass_paths = []
        full_plan_smoke_inputs = None
        for seed_index, seed in enumerate(EXPECTED_SEEDS):
            seed_cases = [case for case in cases if case["seed"] == seed]
            for pass_index in (1, 2):
                runtime_cases = []
                for planned in seed_cases:
                    env_index = planned["env_index"]
                    condition = sha(f"prod-condition-{planned['global_case_index']}")
                    runtime_cases.append(
                        {
                            **planned,
                            "first_condition_sha256": condition,
                            "first_generated_sha256": sha(
                                f"prod-generated-{planned['global_case_index']}"
                            ),
                            "scenario_id": canonical_sha256(
                                {
                                    "seed": seed,
                                    "env_index": env_index,
                                    "first_condition_sha256": condition,
                                }
                            ),
                        }
                    )
                provenance = {
                    "fast_roborender_revision": "a" * 40,
                    "gr00t_revision": "b" * 40,
                    "actor_source_inventory_sha256": sha("prod-actor-source"),
                    "evaluation_source_inventory_sha256": sha("prod-eval-source"),
                    "gr00t_source_tree_sha256": sha("prod-gr00t-tree"),
                    "heldout_plan_sha256": plan_sha,
                    "motion_split_manifest_sha256": self.split_sha,
                    "evaluation_config_sha256": sha("prod-config"),
                    "renderer_runtime_sha256": self.renderer_sha,
                    "camera_acceptance_sha256": sha("prod-camera"),
                    "anchor_inventory_sha256": sha("prod-anchor"),
                    "trainer_container_sha256": sha("prod-trainer-container"),
                    "renderer_cohort_sha256": sha("prod-renderer-cohort"),
                    "parent_checkpoint_sha256": sha("prod-parent"),
                    "parent_runtime_policy_sha256": sha("prod-parent-runtime"),
                }
                runtime = {
                    "schema": RUNTIME_SCHEMA,
                    "plan_sha256": plan_sha,
                    "seed": seed,
                    "pass_index": pass_index,
                    "capture_run_id": f"capture-seed{seed}-pass{pass_index}",
                    "capture_nonce": f"{seed_index * 2 + pass_index:032x}",
                    "capture_profile": "production32",
                    "release_eligible": True,
                    "expected_env_count": 32,
                    "cases": runtime_cases,
                    "execution": {
                        "capture_mode": True,
                        "paired_evaluation": False,
                        "eval_relabel_enabled": True,
                        "training_enabled": False,
                        "policy_step_calls": 1,
                        "env_step_calls": 0,
                        "generated_policy_steps": 1,
                        "generated_frames": 32,
                        "renderer_requests_delta": 32,
                        "renderer_errors_delta": 0,
                        "renderer_replays_delta": 0,
                    },
                    "runtime": {
                        "num_envs": 32,
                        "generation_seed": seed,
                        "motion_split_role": "eval",
                        "motion_split_offset": seed_index * 32,
                        "motion_split_limit": 32,
                        "policy_views": ["head_color", "hand_left", "hand_right"],
                        "view_layout": "vertical",
                        "reference_lag": 256,
                        "raytraced_mix_prob": 0.0,
                        "anchor_source": "simulator_rgb",
                    },
                    "renderer_streams": [
                        {
                            "env_index": env_index,
                            "endpoint_file": "renderer_0.txt",
                            "server_id": "renderer-0",
                            "server_incarnation_id": "incarnation-0",
                        }
                        for env_index in range(32)
                    ],
                    "provenance": provenance,
                }
                runtime_path = self.root / f"runtime_{seed}_{pass_index}.json"
                before_path = self.root / f"before_{seed}_{pass_index}.json"
                after_path = self.root / f"after_{seed}_{pass_index}.json"
                pass_path = self.root / f"pass_{seed}_{pass_index}.json"
                canonical_file(runtime_path, runtime)
                before = {
                    "selected": [
                        {
                            "file": "renderer_0.txt",
                            "response": {
                                "server_id": "renderer-0",
                                "runtime_identity": self.renderer_identity,
                                "metrics": {"requests": 10, "errors": 0, "replays": 0},
                            },
                        }
                    ]
                }
                after = copy.deepcopy(before)
                after["selected"][0]["response"]["metrics"]["requests"] = 42
                if seed_index == 0 and pass_index == 1:
                    full_plan_smoke_inputs = (
                        copy.deepcopy(runtime),
                        copy.deepcopy(before),
                    )
                canonical_file(before_path, before)
                canonical_file(after_path, after)
                pass_document = build_pass_attestation(
                    plan_path=plan_path,
                    runtime_path=runtime_path,
                    renderer_before_path=before_path,
                    renderer_after_path=after_path,
                )
                canonical_file(pass_path, pass_document)
                pass_paths.append(pass_path)
        attestation, evidence, scenarios = build_campaign_attestation(
            plan_path=plan_path,
            pass_paths=pass_paths,
        )
        self.assertEqual(attestation["schema"], ATTESTATION_SCHEMA)
        self.assertTrue(attestation["release_eligible"])
        self.assertEqual(attestation["case_count"], 256)
        self.assertEqual(len(attestation["pass_artifacts"]), 16)
        self.assertEqual(evidence["schema"], "fast_roborender.round_dagger.motion_capture_evidence.v1")
        self.assertEqual(len(evidence["cases"]), 256)
        self.assertTrue(all("first_condition_sha256" in case for case in evidence["cases"]))
        self.assertEqual(len(scenarios), 256)

        # Materialize the independent post-seal contract exactly as production
        # does.  The contract SHA is supplied by the caller and covers the
        # complete JSON; there is intentionally no circular self-hash field.
        attestation_path = self.root / "capture_attestation.json"
        evidence_path = self.root / "capture_evidence.json"
        scenario_path = self.root / "paired_scenarios.json"
        camera_path = self.root / "camera_acceptance.json"
        endpoint_manifest_path = self.root / "endpoint_manifest.txt"
        endpoint_directory = self.root / "renderer_endpoints"
        canonical_file(attestation_path, attestation)
        canonical_file(evidence_path, evidence)
        canonical_file(scenario_path, scenarios)
        canonical_file(camera_path, {"reviewer": "fixture-reviewer"})
        endpoint_manifest_path.write_text("renderer_0.txt\n", encoding="utf-8")
        endpoint_directory.mkdir()
        split_path = self.root / "split.json"
        split_path.write_bytes(b"split")
        self.assertEqual(
            hashlib.sha256(split_path.read_bytes()).hexdigest(), self.split_sha
        )
        runtime_contract_document = {
            "schema": RUNTIME_CONTRACT_SCHEMA,
            "status": "release_eligible",
            "invariant_profile_sha256": sha("invariant-profile"),
            "camera_acceptance": {
                "path": str(camera_path),
                "sha256": hashlib.sha256(camera_path.read_bytes()).hexdigest(),
            },
            "renderer_fleet": {
                "endpoint_manifest_path": str(endpoint_manifest_path),
                "endpoint_manifest_sha256": hashlib.sha256(
                    endpoint_manifest_path.read_bytes()
                ).hexdigest(),
                "endpoint_directory": str(endpoint_directory),
                "renderer_count": 1,
                "cohort_identity_sha256": provenance[
                    "renderer_cohort_sha256"
                ],
                "runtime_identity_sha256": provenance[
                    "renderer_runtime_sha256"
                ],
            },
            "live_capture": {
                "split_manifest_path": str(split_path),
                "split_manifest_sha256": self.split_sha,
                "heldout_plan_path": str(plan_path),
                "heldout_plan_sha256": plan_sha,
                "pass_artifacts": [
                    {
                        "seed": seed,
                        "pass_index": pass_index,
                        "path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for path, (seed, pass_index) in zip(
                        pass_paths,
                        (
                            (seed, pass_index)
                            for seed in EXPECTED_SEEDS
                            for pass_index in (1, 2)
                        ),
                        strict=True,
                    )
                ],
                "attestation_path": str(attestation_path),
                "attestation_sha256": hashlib.sha256(
                    attestation_path.read_bytes()
                ).hexdigest(),
                "evidence_path": str(evidence_path),
                "evidence_sha256": hashlib.sha256(
                    evidence_path.read_bytes()
                ).hexdigest(),
                "scenario_path": str(scenario_path),
                "scenario_sha256": hashlib.sha256(
                    scenario_path.read_bytes()
                ).hexdigest(),
            },
        }
        runtime_contract_path = self.root / "production_runtime_contract.json"
        canonical_file(runtime_contract_path, runtime_contract_document)
        runtime_contract_sha = hashlib.sha256(
            runtime_contract_path.read_bytes()
        ).hexdigest()
        motion_split_contract = {
            "manifest_sha256": self.split_sha,
            "heldout_plan_sha256": plan_sha,
            "collection_role": "collection",
            "evaluation_role": "eval",
            "collection_motion_count": 39,
            "heldout_motion_count": 256,
        }
        verified_runtime = load_verified_runtime_contract(
            runtime_contract_path,
            expected_sha256=runtime_contract_sha,
            expected_invariant_profile_sha256=sha("invariant-profile"),
            motion_split_contract=motion_split_contract,
        )
        lineage = derive_data_lineage(
            verified_runtime,
            motion_split_contract=motion_split_contract,
            topology={
                "cohort_count": 4,
                "actor_ranks_per_cohort": 2,
                "environments_per_actor_rank": 32,
                "active_environments": 64,
            },
        )
        assignment = build_cohort_assignment(
            lineage, cohort_index=3, expected_ranks=(0, 1)
        )
        self.assertEqual(data_lineage_sha256(lineage), assignment["data_lineage_sha256"])
        self.assertEqual(rank_assignment(assignment, 0)["motion_assignment_offset"], 192)
        self.assertEqual(rank_assignment(assignment, 1)["motion_assignment_offset"], 224)

        # A caller cannot replace only the 256-case list and bless its new
        # bytes with a new top-level contract hash: the two-pass attestation
        # still binds the original scenario identities.
        swapped_scenarios = copy.deepcopy(scenarios)
        swapped_scenarios[0]["prompt"] = "swapped scenario"
        swapped_path = self.root / "swapped_paired_scenarios.json"
        canonical_file(swapped_path, swapped_scenarios)
        swapped_contract = copy.deepcopy(runtime_contract_document)
        swapped_contract["live_capture"]["scenario_path"] = str(swapped_path)
        swapped_contract["live_capture"]["scenario_sha256"] = hashlib.sha256(
            swapped_path.read_bytes()
        ).hexdigest()
        swapped_contract_path = self.root / "swapped_runtime_contract.json"
        canonical_file(swapped_contract_path, swapped_contract)
        with self.assertRaisesRegex(
            ProductionDataLineageError, "live capture bundle verification failed"
        ):
            load_verified_runtime_contract(
                swapped_contract_path,
                expected_sha256=hashlib.sha256(
                    swapped_contract_path.read_bytes()
                ).hexdigest(),
                expected_invariant_profile_sha256=sha("invariant-profile"),
                motion_split_contract=motion_split_contract,
            )

        # The safe smoke profile must select only the first two assignments
        # even when it is run against the real 8x32 production plan.
        self.assertIsNotNone(full_plan_smoke_inputs)
        smoke_runtime, smoke_before = full_plan_smoke_inputs
        smoke_runtime["capture_profile"] = "smoke2"
        smoke_runtime["release_eligible"] = False
        smoke_runtime["expected_env_count"] = 2
        smoke_runtime["cases"] = smoke_runtime["cases"][:2]
        smoke_runtime["renderer_streams"] = smoke_runtime["renderer_streams"][:2]
        smoke_runtime["execution"]["generated_frames"] = 2
        smoke_runtime["execution"]["renderer_requests_delta"] = 2
        smoke_runtime["runtime"]["num_envs"] = 2
        smoke_runtime["runtime"]["motion_split_limit"] = 2
        smoke_after = copy.deepcopy(smoke_before)
        smoke_after["selected"][0]["response"]["metrics"]["requests"] = 12
        smoke_runtime_path = self.root / "full_plan_smoke_runtime.json"
        smoke_before_path = self.root / "full_plan_smoke_before.json"
        smoke_after_path = self.root / "full_plan_smoke_after.json"
        canonical_file(smoke_runtime_path, smoke_runtime)
        canonical_file(smoke_before_path, smoke_before)
        canonical_file(smoke_after_path, smoke_after)
        smoke_pass = build_pass_attestation(
            plan_path=plan_path,
            runtime_path=smoke_runtime_path,
            renderer_before_path=smoke_before_path,
            renderer_after_path=smoke_after_path,
            production=False,
        )
        self.assertFalse(smoke_pass["runtime"]["release_eligible"])
        self.assertEqual(smoke_pass["served_requests"], 2)


class ScenarioCaptureLauncherContractTest(unittest.TestCase):
    """Keep the live capture CLI aligned with the evaluator's runtime needs."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.repository = Path(__file__).resolve().parents[1]
        cls.launcher = (
            cls.repository / "scripts/round_dagger/run_heldout_scenario_capture.sh"
        )
        cls.source = cls.launcher.read_text(encoding="utf-8")

    def test_cli_is_valid_and_documents_the_production_shape(self) -> None:
        syntax = subprocess.run(
            ["bash", "-n", str(self.launcher)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        help_result = subprocess.run(
            [str(self.launcher), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        for option in (
            "--plan",
            "--checkpoint-sha256",
            "--evaluation-config",
            "--motion-split-sha256",
            "--expected-renderers",
            "--smoke2",
        ):
            self.assertIn(option, help_result.stdout)

    def test_runtime_stages_exact_bundle_in_private_writable_namespace(self) -> None:
        source = self.source
        self.assertIn('INPUT_STAGE="$OUTPUT_DIR/evaluator_inputs"', source)
        self.assertIn('mkdir "$INPUT_STAGE"', source)
        self.assertIn('chmod 0700 "$INPUT_STAGE"', source)
        self.assertIn("stage_exact_file()", source)
        self.assertIn('install -m 0600 -- "$source" "$destination"', source)
        self.assertIn('source_before == "$source_after"', source)
        self.assertIn('destination_sha == "$expected_sha"', source)
        self.assertIn(
            'stage_exact_file "$CHECKPOINT" "$STAGED_CHECKPOINT" '
            '"$CHECKPOINT_SHA" checkpoint',
            source,
        )
        self.assertIn(
            'stage_exact_file "$EVALUATION_CONFIG" "$STAGED_CONFIG" '
            '"$CONFIG_SHA"',
            source,
        )
        self.assertIn('SOURCE_POLICY_SPEC="${CHECKPOINT}.policy_spec.json"', source)
        self.assertIn('SOURCE_POLICY_MANIFEST="${CHECKPOINT}.manifest.json"', source)
        self.assertIn("-m round_dagger.policy_bundle", source)
        self.assertIn("--expected-runtime-policy-sha256", source)
        self.assertIn('"$ROUND_DAGGER_RUNTIME_POLICY_SHA256"', source)
        self.assertIn('++checkpoint="$STAGED_CHECKPOINT"', source)
        self.assertNotIn('++checkpoint="$CHECKPOINT"', source)
        self.assertLess(
            source.index('stage_exact_file "$CHECKPOINT"'),
            source.index('++checkpoint="$STAGED_CHECKPOINT"'),
        )
        self.assertGreaterEqual(source.count("verify_staged_file"), 5)

    def test_runtime_preserves_imports_and_supplies_fail_closed_overrides(self) -> None:
        source = self.source
        self.assertIn(
            'export PYTHONPATH="$REPOSITORY_ROOT:$GR00T_ROOT'
            '${PYTHONPATH:+:$PYTHONPATH}"',
            source,
        )
        evaluator = source.index('"$PYTHON_BIN" "$EVALUATOR"')
        self.assertLess(source.rindex('cd "$GR00T_ROOT"', 0, evaluator), evaluator)
        self.assertIn('hydra.run.dir="$OUTPUT_DIR/hydra"', source)
        self.assertIn('++output_dir="$OUTPUT_DIR/output"', source)
        evaluator_source = (
            self.repository / "GR00T-VisualSim2Real/gr00t/rl/eval_agent_trl.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'config_path = checkpoint.parent / "config.yaml"', evaluator_source
        )
        self.assertIn('STAGED_CONFIG="$INPUT_STAGE/config.yaml"', source)
        approved_thresholds = {
            "contact_force_n": "10.0",
            "gravity_xy": "0.5",
            "min_base_height_m": "0.3",
            "dof_velocity_limit_fraction": "0.98",
            "torque_limit_fraction": "0.98",
            "min_object_height_m": "0.6",
            "right_arm_velocity_rad_s": "3.14",
        }
        for name, value in approved_thresholds.items():
            self.assertIn(
                f"++env.config.safety_thresholds.{name}={value}", source
            )


if __name__ == "__main__":
    unittest.main()
