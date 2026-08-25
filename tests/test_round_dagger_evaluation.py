"""Deterministic tests for the paired fixed-seed candidate gate."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from round_dagger.artifacts import ArtifactError, canonical_json_sha256, sha256_file
from round_dagger.evaluation import (
    POLICY_VIEWS,
    RUNTIME_SCHEMA,
    EvaluationError,
    EVALUATOR_RESULT_SCHEMA,
    attest_outcome,
    create_contract,
    evaluation_run_id,
    evaluator_source_bundle,
    evaluator_source_bundle_sha256,
    load_contract,
    load_outcome,
    publish_case_outcome,
    summarize_to_evaluator_result,
)
from round_dagger.orchestration import ProductionBackend
from round_dagger.promotion import PromotionError, promote_candidate
from round_dagger.state import load_state, transition

try:
    from test_round_dagger_state import _campaign, _sha, _stage_and_bind, _to_ingested
except ImportError:  # pragma: no cover - direct package-style unittest invocation
    from tests.test_round_dagger_state import (  # type: ignore[no-redef]
        _campaign,
        _sha,
        _stage_and_bind,
        _to_ingested,
    )


class PairedEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.tmp = Path(self.temporary.name)
        self.root = _campaign(self.tmp)
        _to_ingested(self.root)
        self.candidate = _stage_and_bind(self.root, self.tmp)
        self.repository_root = Path(__file__).parents[1]
        self.gr00t = self.tmp / "gr00t"
        fixture_sources = (
            "gr00t/rl/eval_agent_trl.py",
            "gr00t/rl/trl/trainer/distill_trainer_obj_pred_homie_api.py",
            "gr00t/rl/trl/trainer/distill_trainer_videogen.py",
            "gr00t/rl/utils/image3f_frame_client.py",
            "gr00t/rl/utils/round_dagger_paired_eval.py",
        )
        for relative in fixture_sources:
            source = self.gr00t / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(f"# deterministic fixture: {relative}\n")
        self.evaluator = self.gr00t / fixture_sources[0]
        self.evaluator_code_sha = evaluator_source_bundle_sha256(
            self.repository_root, self.gr00t
        )
        self.config = self.tmp / "evaluation_config.yaml"
        self.config.write_text("fixed_seed: true\n")
        self.renderer_identity = {
            "schema_version": 1,
            "release_id": "fixture",
            "adapter_sha256": _sha("adapter"),
            "generation": {
                "steps": 5,
                "reference_lag_max": 512,
                "views": list(POLICY_VIEWS),
            },
        }
        self.renderer_runtime_sha = canonical_json_sha256(self.renderer_identity)
        self.cases = [
            {"seed": seed, "env_index": env, "scenario_id": _sha(f"{seed}:{env}")}
            for seed in (17, 29)
            for env in range(4)
        ]
        self.contract = create_contract(
            self.root,
            candidate_manifest=self.candidate["manifest_path"],
            scenario_cases=self.cases,
            parent_runtime_policy_sha256=_sha("parent-runtime"),
            candidate_runtime_policy_sha256=_sha("candidate-runtime"),
            renderer_runtime_sha256=self.renderer_runtime_sha,
            evaluation_config_sha256=sha256_file(self.config),
            evaluator_code_sha256=self.evaluator_code_sha,
            expected_renderer_count=1,
            output="rounds/macro_round_000000/evaluation/contract.json",
        )
        self.results_relative = "rounds/macro_round_000000/evaluation/results"
        self.result_relative = "rounds/macro_round_000000/evaluation/result.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _raw_evidence(
        self,
        role: str,
        seed: int,
        *,
        successes: list[bool],
        safety: list[bool],
        action_mse: float,
        renderer_errors: int = 0,
        bad_scenario: bool = False,
        omit_view: bool = False,
        attempt: int = 1,
        condition_tag: str = "shared",
        generated_tag: str = "shared",
        episode_length: int = 300,
    ) -> tuple[Path, Path, Path]:
        raw = (
            self.root
            / "evaluation"
            / "raw"
            / role
            / f"seed_{seed}"
            / f"attempt_{attempt:06d}"
        )
        raw.mkdir(parents=True, exist_ok=True)
        policy = next(
            item for item in self.contract["policies"] if item["role"] == role
        )
        expected_cases = [item for item in self.cases if item["seed"] == seed]
        image3f_run_id = evaluation_run_id(
            self.contract, role=role, seed=seed, attempt=attempt
        )
        episodes = []
        fixed_episodes = []
        first_conditions = {}
        first_generated = {}
        for case, success, unsafe in zip(
            expected_cases, successes, safety, strict=True
        ):
            env_index = case["env_index"]
            views = {
                view: _sha(f"{role}:{seed}:{env_index}:{view}")
                for view in POLICY_VIEWS
            }
            if omit_view and env_index == 0:
                views.pop("hand_right")
            condition_sha = _sha(f"condition:{condition_tag}:{seed}:{env_index}")
            generated_sha = _sha(
                f"generated:{generated_tag}:{seed}:{env_index}"
            )
            first_conditions[str(env_index)] = condition_sha
            first_generated[str(env_index)] = generated_sha
            episodes.append(
                {
                    "env_index": env_index,
                    "seed": seed,
                    "scenario_id": _sha("wrong")
                    if bad_scenario and env_index == 0
                    else case["scenario_id"],
                    "success": success,
                    "safety_violation": unsafe,
                    "action_imitation_mse": action_mse,
                    "episode_length": episode_length,
                    "generated_view_sha256": views,
                }
            )
            fixed_episodes.append(
                {
                    "env_index": env_index,
                    "success": success,
                    "reward": float(env_index),
                    "episode_length": episode_length,
                    "object_class": "fixture",
                    "seed": seed,
                    "first_condition_sha256": condition_sha,
                    "first_generated_sha256": generated_sha,
                    "scenario_id": canonical_json_sha256(
                        {
                            "seed": seed,
                            "env_index": env_index,
                            "first_condition_sha256": condition_sha,
                        }
                    ),
                }
            )
        generated_policy_steps = episode_length
        generated_frames = generated_policy_steps * len(expected_cases)
        metrics = {
            "_roborender_eval_adapter": {
                "schema_version": 1,
                "image3f_run_id": image3f_run_id,
            },
            "_roborender_paired_evaluation": {
                "schema": RUNTIME_SCHEMA,
                "eval_relabel_enabled": True,
                "policy_views": list(POLICY_VIEWS),
                "view_layout": "vertical",
                "raytraced_mix_prob": 0.0,
                "reference_lag": 256,
                "training_enabled": False,
                "one_episode_per_environment": True,
                "generation_seed": seed,
                "runtime_policy_sha256": policy["runtime_policy_sha256"],
                "evaluation_config_sha256": self.contract["evaluation_config_sha256"],
                "evaluator_code_sha256": self.contract["evaluator_code_sha256"],
            },
            "generated_policy_steps": generated_policy_steps,
            "generated_frames": generated_frames,
            "_roborender_image3f_eval": {
                "schema": "fast_roborender.image3f_fixed_eval_runtime.v1",
                "eval_relabel_enabled": True,
                "raytraced_mix_prob": 0.0,
                "reference_lag": 256,
                "generation_seed": seed,
                "policy_views": list(POLICY_VIEWS),
                "generated_policy_steps": generated_policy_steps,
                "generated_frames": generated_frames,
                "first_condition_sha256_by_env": first_conditions,
                "first_generated_sha256_by_env": first_generated,
            },
            "_roborender_fixed_seed_episodes": fixed_episodes,
            "_roborender_paired_episodes": episodes,
        }
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
        after = json.loads(json.dumps(before))
        after["selected"][0]["response"]["metrics"] = {
            "requests": 10 + generated_frames,
            "errors": renderer_errors,
            "replays": 0,
        }
        paths = raw / "metrics.json", raw / "before.json", raw / "after.json"
        for path, value in zip(paths, (metrics, before, after), strict=True):
            path.write_text(json.dumps(value, sort_keys=True))
        return paths

    def _attest_pair(
        self,
        *,
        candidate_successes: list[bool],
        candidate_safety: list[bool],
        candidate_mse: float,
    ) -> None:
        for role in ("parent", "candidate"):
            for seed in self.contract["seeds"]:
                if role == "parent":
                    values = (
                        [True, False, False, False],
                        [False, False, True, False],
                        1.0,
                    )
                else:
                    values = (candidate_successes, candidate_safety, candidate_mse)
                metrics, before, after = self._raw_evidence(
                    role,
                    seed,
                    successes=values[0],
                    safety=values[1],
                    action_mse=values[2],
                )
                attest_outcome(
                    self.root,
                    contract_path=self.contract["contract_path"],
                    role=role,
                    seed=seed,
                    metrics_path=metrics,
                    renderer_before_path=before,
                    renderer_after_path=after,
                    image3f_run_id=evaluation_run_id(
                        self.contract, role=role, seed=seed, attempt=1
                    ),
                    output=(
                        f"{self.results_relative}/{role}/seed_{seed}/outcome.json"
                    ),
                )

    def _attested_attempt(self, role: str, seed: int, attempt: int) -> Path:
        metrics, before, after = self._raw_evidence(
            role,
            seed,
            successes=[True, False, False, False],
            safety=[False] * 4,
            action_mse=1.0,
            attempt=attempt,
        )
        outcome = (
            self.root
            / self.results_relative
            / "attempts"
            / role
            / f"seed_{seed}"
            / f"attempt_{attempt:06d}"
            / "outcome.json"
        )
        attest_outcome(
            self.root,
            contract_path=self.contract["contract_path"],
            role=role,
            seed=seed,
            metrics_path=metrics,
            renderer_before_path=before,
            renderer_after_path=after,
            image3f_run_id=evaluation_run_id(
                self.contract, role=role, seed=seed, attempt=attempt
            ),
            output=outcome,
        )
        return outcome

    def _run_launcher(self, results: Path) -> subprocess.CompletedProcess[str]:
        endpoints = self.tmp / "launcher-endpoints"
        endpoints.mkdir(exist_ok=True)
        endpoint = endpoints / "renderer_0.txt"
        if not endpoint.exists():
            endpoint.write_text("127.0.0.1 9999\n")
        launcher = (
            self.repository_root
            / "scripts"
            / "round_dagger"
            / "run_paired_evaluation.sh"
        )
        return subprocess.run(
            [
                "bash",
                str(launcher),
                "--root",
                str(self.root),
                "--contract",
                str(self.root / self.contract["contract_path"]),
                "--results-dir",
                str(results),
                "--gr00t-root",
                str(self.gr00t),
                "--evaluation-config",
                str(self.config),
                "--endpoint-dir",
                str(endpoints),
            ],
            text=True,
            capture_output=True,
            check=False,
            env={
                **os.environ,
                "ROUND_DAGGER_EVALUATOR_RESULT": str(
                    self.root / self.result_relative
                ),
                "ROUND_DAGGER_EVALUATOR_CONFIG_SHA256": self.contract[
                    "evaluation_config_sha256"
                ],
            },
        )

    def test_contract_binds_parent_candidate_cases_and_lagged_three_views(self) -> None:
        loaded = load_contract(self.root, self.contract["contract_path"])
        self.assertEqual(
            [item["role"] for item in loaded["policies"]], ["parent", "candidate"]
        )
        self.assertEqual(loaded["seeds"], [17, 29])
        self.assertEqual(loaded["runtime"]["reference_lag"], 256)
        self.assertEqual(loaded["runtime"]["raytraced_mix_prob"], 0.0)
        self.assertEqual(loaded["runtime"]["policy_views"], list(POLICY_VIEWS))
        self.assertFalse(loaded["runtime"]["training_enabled"])

        incomplete = self.cases[:-1]
        with self.assertRaisesRegex(EvaluationError, "same environment"):
            create_contract(
                self.root,
                candidate_manifest=self.candidate["manifest_path"],
                scenario_cases=incomplete,
                parent_runtime_policy_sha256=_sha("p"),
                candidate_runtime_policy_sha256=_sha("c"),
                renderer_runtime_sha256=self.renderer_runtime_sha,
                evaluation_config_sha256=sha256_file(self.config),
                evaluator_code_sha256=self.evaluator_code_sha,
                expected_renderer_count=1,
                output="evaluation/bad-contract.json",
            )

    def test_role_seed_and_retry_stream_namespaces_are_disjoint(self) -> None:
        run_ids = {
            evaluation_run_id(self.contract, role=role, seed=seed, attempt=1)
            for role in ("parent", "candidate")
            for seed in self.contract["seeds"]
        }
        self.assertEqual(len(run_ids), 4)
        # Image3F appends rank/environment to this namespace.  Consequently,
        # even the first request (episode 0, step 0) cannot collide across any
        # parent/candidate fixed-seed invocation.
        step_zero_streams = {f"{run_id}:rank0:env0" for run_id in run_ids}
        self.assertEqual(len(step_zero_streams), 4)
        first = evaluation_run_id(
            self.contract, role="parent", seed=self.contract["seeds"][0], attempt=1
        )
        retry = evaluation_run_id(
            self.contract, role="parent", seed=self.contract["seeds"][0], attempt=2
        )
        self.assertNotEqual(first, retry)

    def test_launcher_recovers_attested_attempt_after_publish_crash(self) -> None:
        role = "parent"
        seed = self.contract["seeds"][0]
        results = self.root / self.results_relative
        attempt = self._attested_attempt(role, seed, 1)
        canonical = results / role / f"seed_{seed}" / "outcome.json"
        self.assertFalse(canonical.exists())

        resumed = self._run_launcher(results)
        self.assertNotEqual(resumed.returncode, 0)
        self.assertIn(
            f"PAIRED_EVALUATION_CASE_RECOVERED role={role} seed={seed} attempt=1",
            resumed.stdout,
        )
        self.assertTrue(canonical.is_file())
        self.assertEqual(canonical.read_bytes(), attempt.read_bytes())
        self.assertFalse(
            (attempt.parent.parent / "attempt_000002").exists(),
            "recovery must publish the valid attempt without rerunning the case",
        )
        loaded = load_outcome(
            self.root,
            canonical,
            contract=self.contract,
            expected_role=role,
            expected_seed=seed,
        )
        self.assertEqual(
            loaded["image3f_run_id"],
            evaluation_run_id(self.contract, role=role, seed=seed, attempt=1),
        )

    def test_launcher_rejects_multiple_valid_unpublished_attempts(self) -> None:
        role = "parent"
        seed = self.contract["seeds"][0]
        results = self.root / self.results_relative
        first = self._attested_attempt(role, seed, 1)
        second = self._attested_attempt(role, seed, 2)

        resumed = self._run_launcher(results)
        self.assertNotEqual(resumed.returncode, 0)
        self.assertIn("multiple valid unpublished outcomes", resumed.stderr)
        self.assertTrue(first.is_file())
        self.assertTrue(second.is_file())
        self.assertFalse((results / role / f"seed_{seed}").exists())
        self.assertFalse((second.parent.parent / "attempt_000003").exists())

    def test_launcher_rejects_tampered_attested_attempt(self) -> None:
        role = "parent"
        seed = self.contract["seeds"][0]
        results = self.root / self.results_relative
        outcome = self._attested_attempt(role, seed, 1)
        document = json.loads(outcome.read_text(encoding="utf-8"))
        metrics = self.root / document["source_artifacts"]["metrics_path"]
        metrics.write_text("{}\n", encoding="utf-8")

        resumed = self._run_launcher(results)
        self.assertNotEqual(resumed.returncode, 0)
        self.assertIn("invalid or tampered attested attempt", resumed.stderr)
        self.assertTrue(outcome.is_file())
        self.assertFalse((results / role / f"seed_{seed}").exists())
        self.assertFalse((outcome.parent.parent / "attempt_000002").exists())

    def test_attempt_directory_number_must_match_attested_run_id(self) -> None:
        role = "parent"
        seed = self.contract["seeds"][0]
        results = self.root / self.results_relative
        outcome = self._attested_attempt(role, seed, 1)
        mismatched = outcome.parent.parent / "attempt_000002"
        outcome.parent.rename(mismatched)

        resumed = self._run_launcher(results)
        self.assertNotEqual(resumed.returncode, 0)
        self.assertIn("invalid or tampered attested attempt", resumed.stderr)
        self.assertFalse((results / role / f"seed_{seed}").exists())

    def test_attempt_outcome_is_validated_before_immutable_publication(self) -> None:
        role = "parent"
        seed = self.contract["seeds"][0]
        metrics, before, after = self._raw_evidence(
            role,
            seed,
            successes=[True, False, False, False],
            safety=[False] * 4,
            action_mse=1.0,
            attempt=1,
        )
        source = (
            self.root
            / self.results_relative
            / "attempts"
            / role
            / f"seed_{seed}"
            / "attempt_000001"
            / "outcome.json"
        )
        run_id = evaluation_run_id(
            self.contract, role=role, seed=seed, attempt=1
        )
        attest_outcome(
            self.root,
            contract_path=self.contract["contract_path"],
            role=role,
            seed=seed,
            metrics_path=metrics,
            renderer_before_path=before,
            renderer_after_path=after,
            image3f_run_id=run_id,
            output=source,
        )
        canonical = (
            self.root / self.results_relative / role / f"seed_{seed}" / "outcome.json"
        )
        published = publish_case_outcome(
            self.root,
            contract_path=self.contract["contract_path"],
            role=role,
            seed=seed,
            source=source,
            output=canonical,
        )
        self.assertEqual(published["image3f_run_id"], run_id)
        loaded = load_outcome(
            self.root,
            canonical,
            contract=self.contract,
            expected_role=role,
            expected_seed=seed,
        )
        self.assertEqual(loaded["image3f_run_id"], run_id)

        metrics2, before2, after2 = self._raw_evidence(
            role,
            seed,
            successes=[True, False, False, False],
            safety=[False] * 4,
            action_mse=1.0,
            attempt=2,
        )
        source2 = source.parents[0].parent / "attempt_000002" / "outcome.json"
        run_id2 = evaluation_run_id(
            self.contract, role=role, seed=seed, attempt=2
        )
        attest_outcome(
            self.root,
            contract_path=self.contract["contract_path"],
            role=role,
            seed=seed,
            metrics_path=metrics2,
            renderer_before_path=before2,
            renderer_after_path=after2,
            image3f_run_id=run_id2,
            output=source2,
        )
        with self.assertRaisesRegex(ArtifactError, "immutable artifact conflict"):
            publish_case_outcome(
                self.root,
                contract_path=self.contract["contract_path"],
                role=role,
                seed=seed,
                source=source2,
                output=canonical,
            )

        endpoints = self.tmp / "completed-case-endpoints"
        endpoints.mkdir()
        (endpoints / "renderer_0.txt").write_text("127.0.0.1 9999\n")
        launcher = (
            self.repository_root
            / "scripts"
            / "round_dagger"
            / "run_paired_evaluation.sh"
        )
        resumed = subprocess.run(
            [
                "bash",
                str(launcher),
                "--root",
                str(self.root),
                "--contract",
                str(self.root / self.contract["contract_path"]),
                "--results-dir",
                str(self.root / self.results_relative),
                "--gr00t-root",
                str(self.gr00t),
                "--evaluation-config",
                str(self.config),
                "--endpoint-dir",
                str(endpoints),
            ],
            text=True,
            capture_output=True,
            check=False,
            env={
                **os.environ,
                "ROUND_DAGGER_EVALUATOR_RESULT": str(
                    self.root / self.result_relative
                ),
                "ROUND_DAGGER_EVALUATOR_CONFIG_SHA256": self.contract[
                    "evaluation_config_sha256"
                ],
            },
        )
        self.assertNotEqual(resumed.returncode, 0)
        self.assertIn(
            f"PAIRED_EVALUATION_CASE_REUSED role={role} seed={seed}",
            resumed.stdout,
        )
        self.assertFalse(
            (source.parent.parent / "attempt_000003").exists(),
            "a fully attested case must not allocate another renderer stream",
        )

    def test_improving_candidate_emits_orchestrator_result_without_state_mutation(self) -> None:
        self._attest_pair(
            candidate_successes=[True, True, False, False],
            candidate_safety=[False, False, False, False],
            candidate_mse=0.8,
        )
        result = summarize_to_evaluator_result(
            self.root,
            contract_path=self.contract["contract_path"],
            results_dir=self.results_relative,
            output=self.result_relative,
        )
        report = result["report"]
        self.assertTrue(report["decision"]["passed"])
        self.assertAlmostEqual(report["metrics"]["paired_success_rate_delta"], 0.25)
        self.assertAlmostEqual(
            report["metrics"]["paired_safety_violation_rate_delta"], -0.25
        )
        self.assertAlmostEqual(
            report["metrics"]["paired_action_imitation_mse_relative_change"], -0.2
        )
        evaluator_result = result["evaluator_result"]
        self.assertEqual(evaluator_result["schema"], EVALUATOR_RESULT_SCHEMA)
        self.assertTrue(evaluator_result["passed"])
        self.assertEqual(
            evaluator_result["evaluator_config_sha256"],
            self.contract["evaluation_config_sha256"],
        )
        self.assertEqual(
            evaluator_result["evidence_bundle_sha256"], result["bundle"]["sha256"]
        )
        self.assertEqual(
            evaluator_result["evidence_bundle_path"], result["bundle"]["path"]
        )
        self.assertEqual(
            sha256_file(self.root / result["bundle"]["path"]),
            result["bundle"]["sha256"],
        )
        self.assertEqual(load_state(self.root)["phase"], "TRAINED")
        self.assertFalse(
            (self.root / "policies/policy_000001/evaluation.json").exists()
        )

        backend = ProductionBackend(
            SimpleNamespace(
                campaign_root=self.root,
                macro_round=0,
                evaluator_config_sha256=self.contract["evaluation_config_sha256"],
            )
        )
        loaded = backend.load_evaluator_result(
            campaign=backend.load_campaign(), candidate=self.candidate
        )
        self.assertEqual(loaded["evidence_bundle_sha256"], result["bundle"]["sha256"])
        evaluation = backend.publish_evaluation(self.candidate, loaded)
        self.assertEqual(
            evaluation["evidence_bundle_sha256"], result["bundle"]["sha256"]
        )

    def test_mismatch_renderer_errors_and_missing_view_fail_closed(self) -> None:
        cases = (
            {"bad_scenario": True},
            {"renderer_errors": 1},
            {"omit_view": True},
        )
        for index, override in enumerate(cases):
            with self.subTest(override=override):
                role = "parent"
                seed = self.contract["seeds"][0]
                metrics, before, after = self._raw_evidence(
                    role,
                    seed,
                    successes=[True, False, False, False],
                    safety=[False] * 4,
                    action_mse=1.0,
                    **override,
                )
                with self.assertRaises(EvaluationError):
                    attest_outcome(
                        self.root,
                        contract_path=self.contract["contract_path"],
                        role=role,
                        seed=seed,
                        metrics_path=metrics,
                        renderer_before_path=before,
                        renderer_after_path=after,
                        image3f_run_id=evaluation_run_id(
                            self.contract, role=role, seed=seed, attempt=1
                        ),
                        output=f"evaluation/bad/{index}/outcome.json",
                    )

    def test_actual_initial_conditions_must_match_across_policy_roles(self) -> None:
        for role in ("parent", "candidate"):
            for seed in self.contract["seeds"]:
                metrics, before, after = self._raw_evidence(
                    role,
                    seed,
                    successes=[True, False, False, False],
                    safety=[False] * 4,
                    action_mse=1.0,
                    condition_tag=role,
                )
                attest_outcome(
                    self.root,
                    contract_path=self.contract["contract_path"],
                    role=role,
                    seed=seed,
                    metrics_path=metrics,
                    renderer_before_path=before,
                    renderer_after_path=after,
                    image3f_run_id=evaluation_run_id(
                        self.contract, role=role, seed=seed, attempt=1
                    ),
                    output=f"{self.results_relative}/{role}/seed_{seed}/outcome.json",
                )
        with self.assertRaisesRegex(EvaluationError, "not case-paired"):
            summarize_to_evaluator_result(
                self.root,
                contract_path=self.contract["contract_path"],
                results_dir=self.results_relative,
                output=self.result_relative,
            )

    def test_actual_initial_generated_observations_must_match_across_roles(self) -> None:
        for role in ("parent", "candidate"):
            for seed in self.contract["seeds"]:
                metrics, before, after = self._raw_evidence(
                    role,
                    seed,
                    successes=[True, False, False, False],
                    safety=[False, False, False, False],
                    action_mse=1.0,
                    generated_tag="parent" if role == "parent" else "candidate",
                )
                attest_outcome(
                    self.root,
                    contract_path=self.contract["contract_path"],
                    role=role,
                    seed=seed,
                    metrics_path=metrics,
                    renderer_before_path=before,
                    renderer_after_path=after,
                    image3f_run_id=evaluation_run_id(
                        self.contract, role=role, seed=seed, attempt=1
                    ),
                    output=f"{self.results_relative}/{role}/seed_{seed}/outcome.json",
                )
        with self.assertRaisesRegex(EvaluationError, "not case-paired"):
            summarize_to_evaluator_result(
                self.root,
                contract_path=self.contract["contract_path"],
                results_dir=self.results_relative,
                output=self.result_relative,
            )

    def test_first_episode_must_cross_generated_lag_boundary(self) -> None:
        seed = self.contract["seeds"][0]
        metrics, before, after = self._raw_evidence(
            "parent",
            seed,
            successes=[True, False, False, False],
            safety=[False] * 4,
            action_mse=1.0,
            episode_length=256,
        )
        with self.assertRaisesRegex(EvaluationError, "crossed the lag-256"):
            attest_outcome(
                self.root,
                contract_path=self.contract["contract_path"],
                role="parent",
                seed=seed,
                metrics_path=metrics,
                renderer_before_path=before,
                renderer_after_path=after,
                image3f_run_id=evaluation_run_id(
                    self.contract, role="parent", seed=seed, attempt=1
                ),
                output="evaluation/short/outcome.json",
            )

    def test_source_corruption_after_attestation_is_detected(self) -> None:
        seed = self.contract["seeds"][0]
        metrics, before, after = self._raw_evidence(
            "parent",
            seed,
            successes=[True, False, False, False],
            safety=[False] * 4,
            action_mse=1.0,
        )
        outcome = attest_outcome(
            self.root,
            contract_path=self.contract["contract_path"],
            role="parent",
            seed=seed,
            metrics_path=metrics,
            renderer_before_path=before,
            renderer_after_path=after,
            image3f_run_id=evaluation_run_id(
                self.contract, role="parent", seed=seed, attempt=1
            ),
            output=f"{self.results_relative}/parent/seed_{seed}/outcome.json",
        )
        metrics.write_text("{}")
        with self.assertRaisesRegex(EvaluationError, "source artifact changed"):
            load_outcome(
                self.root,
                outcome["outcome_path"],
                contract=self.contract,
                expected_role="parent",
                expected_seed=seed,
            )

    def test_evaluator_identity_binds_all_custom_producer_sources(self) -> None:
        bundle = evaluator_source_bundle(self.repository_root, self.gr00t)
        self.assertEqual(len(bundle["sources"]), 9)
        self.assertIn(
            "gr00t/gr00t/rl/utils/image3f_frame_client.py",
            {item["source"] for item in bundle["sources"]},
        )
        self.assertEqual(canonical_json_sha256(bundle), self.evaluator_code_sha)
        helper = self.gr00t / "gr00t/rl/utils/round_dagger_paired_eval.py"
        helper.write_text("# changed paired evidence semantics\n")
        self.assertNotEqual(
            evaluator_source_bundle_sha256(self.repository_root, self.gr00t),
            self.evaluator_code_sha,
        )

    def test_regressing_candidate_emits_failed_gate_and_cannot_promote(self) -> None:
        self._attest_pair(
            candidate_successes=[False, False, False, False],
            candidate_safety=[True, True, False, False],
            candidate_mse=1.5,
        )
        result = summarize_to_evaluator_result(
            self.root,
            contract_path=self.contract["contract_path"],
            results_dir=self.results_relative,
            output=self.result_relative,
        )
        evaluator_result = result["evaluator_result"]
        self.assertFalse(evaluator_result["passed"])
        self.assertIn(
            "SUCCESS_REGRESSION", result["report"]["decision"]["reason_codes"]
        )
        backend = ProductionBackend(
            SimpleNamespace(
                campaign_root=self.root,
                macro_round=0,
                evaluator_config_sha256=self.contract["evaluation_config_sha256"],
            )
        )
        evaluation = backend.publish_evaluation(
            self.candidate,
            backend.load_evaluator_result(
                campaign=backend.load_campaign(), candidate=self.candidate
            ),
        )
        transition(
            self.root,
            expected_phase="TRAINED",
            next_phase="EVALUATED",
            macro_round=0,
            evidence_path=evaluation["manifest_path"],
            evidence_sha256=evaluation["evidence_sha256"],
        )
        with self.assertRaisesRegex(PromotionError, "failed"):
            promote_candidate(
                self.root,
                macro_round=0,
                candidate_manifest=self.candidate["manifest_path"],
                evaluation_manifest=evaluation["manifest_path"],
            )

    def test_shell_launcher_dry_run_is_read_only_and_explicit(self) -> None:
        endpoints = self.tmp / "endpoints"
        endpoints.mkdir()
        (endpoints / "renderer_0.txt").write_text("127.0.0.1 9999\n")
        results = self.root / self.results_relative
        orchestrator_result = self.root / self.result_relative
        script = (
            Path(__file__).parents[1]
            / "scripts"
            / "round_dagger"
            / "run_paired_evaluation.sh"
        )
        completed = subprocess.run(
            [
                "bash",
                str(script),
                "--dry-run",
                "--root",
                str(self.root),
                "--contract",
                str(self.root / self.contract["contract_path"]),
                "--results-dir",
                str(results),
                "--gr00t-root",
                str(self.gr00t),
                "--evaluation-config",
                str(self.config),
                "--endpoint-dir",
                str(endpoints),
            ],
            text=True,
            capture_output=True,
            check=False,
            env={
                **os.environ,
                "ROUND_DAGGER_EVALUATOR_RESULT": str(orchestrator_result),
                "ROUND_DAGGER_EVALUATOR_CONFIG_SHA256": self.contract[
                    "evaluation_config_sha256"
                ],
            },
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("PAIRED_EVALUATION_DRY_RUN_OK", completed.stdout)
        self.assertIn("reference_lag=256", completed.stdout)
        self.assertIn("IMAGE_DAGGER_EVAL_RELABEL=1", completed.stdout)
        self.assertIn("algo.config.videogen.view_names", completed.stdout)
        self.assertIn("--output", completed.stdout)
        self.assertIn(str(orchestrator_result), completed.stdout)
        for view in POLICY_VIEWS:
            self.assertIn(view, completed.stdout)
        run_ids = [
            token.split("=", 1)[1]
            for token in completed.stdout.split()
            if token.startswith("IMAGE3F_RUN_ID=")
        ]
        self.assertEqual(len(run_ids), 4)
        self.assertEqual(len(set(run_ids)), 4)
        self.assertEqual(
            len({f"{run_id}:rank0:env0" for run_id in run_ids}), 4
        )
        self.assertFalse(results.exists())

    def test_shell_launcher_preserves_partial_case_and_allocates_fresh_retry(self) -> None:
        endpoints = self.tmp / "retry-endpoints"
        endpoints.mkdir()
        (endpoints / "renderer_0.txt").write_text("127.0.0.1 9999\n")
        results = self.root / self.results_relative
        partial = results / "parent" / "seed_17"
        partial.mkdir(parents=True)
        (partial / "interrupted.marker").write_text("preserve me\n")
        script = (
            Path(__file__).parents[1]
            / "scripts"
            / "round_dagger"
            / "run_paired_evaluation.sh"
        )
        command = [
            "bash",
            str(script),
            "--root",
            str(self.root),
            "--contract",
            str(self.root / self.contract["contract_path"]),
            "--results-dir",
            str(results),
            "--gr00t-root",
            str(self.gr00t),
            "--evaluation-config",
            str(self.config),
            "--endpoint-dir",
            str(endpoints),
        ]
        environment = {
            **os.environ,
            "ROUND_DAGGER_EVALUATOR_RESULT": str(self.root / self.result_relative),
            "ROUND_DAGGER_EVALUATOR_CONFIG_SHA256": self.contract[
                "evaluation_config_sha256"
            ],
        }
        first = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            env=environment,
        )
        self.assertNotEqual(first.returncode, 0)
        attempt_base = results / "attempts" / "parent" / "seed_17"
        self.assertEqual(
            (attempt_base / "attempt_000001/interrupted.marker").read_text(),
            "preserve me\n",
        )
        self.assertTrue((attempt_base / "attempt_000002").is_dir())
        self.assertFalse(partial.exists())

        second = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            env=environment,
        )
        self.assertNotEqual(second.returncode, 0)
        self.assertTrue((attempt_base / "attempt_000003").is_dir())
        self.assertEqual(
            evaluation_run_id(
                self.contract, role="parent", seed=17, attempt=2
            ).rsplit("-a", 1)[0],
            evaluation_run_id(
                self.contract, role="parent", seed=17, attempt=3
            ).rsplit("-a", 1)[0],
        )
        self.assertNotEqual(
            evaluation_run_id(self.contract, role="parent", seed=17, attempt=2),
            evaluation_run_id(self.contract, role="parent", seed=17, attempt=3),
        )


if __name__ == "__main__":
    unittest.main()
