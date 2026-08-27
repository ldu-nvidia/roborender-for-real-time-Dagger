"""Deterministic tests for the paired fixed-seed candidate gate."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from round_dagger.artifacts import ArtifactError, canonical_json_sha256, sha256_file
from round_dagger.evaluation import (
    POLICY_VIEWS,
    RUNTIME_SCHEMA,
    _score,
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
from scripts.round_dagger.verify_anchor_pool import AnchorRecord, inventory_sha256

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
            "gr00t/rl/trl/trainer/anchor_io.py",
            "gr00t/rl/trl/trainer/checkpoint_io.py",
            "gr00t/rl/trl/trainer/distill_trainer.py",
            "gr00t/rl/trl/trainer/distill_trainer_obj_pred_homie_api.py",
            "gr00t/rl/trl/trainer/distill_trainer_videogen.py",
            "gr00t/rl/trl/trainer/ppo_trainer.py",
            "gr00t/rl/utils/image3f_frame_client.py",
            "gr00t/rl/utils/round_dagger_paired_eval.py",
        )
        for relative in fixture_sources:
            source = self.gr00t / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(f"# deterministic fixture: {relative}\n")
        self.evaluator = self.gr00t / fixture_sources[0]
        self.evaluation_inventory_sha = _sha("evaluation-source-inventory")
        self.evaluator_code_sha = evaluator_source_bundle_sha256(
            self.repository_root,
            self.gr00t,
            self.evaluation_inventory_sha,
        )
        self.anchor_pool = self.tmp / "anchors"
        self.anchor_pool.mkdir()
        anchor = self.anchor_pool / "anchor_000000.npz"
        anchor.write_bytes(b"deterministic anchor fixture")
        self.anchor_inventory_sha = inventory_sha256(
            [
                AnchorRecord(
                    relative_path=anchor.name,
                    size=anchor.stat().st_size,
                    sha256=sha256_file(anchor),
                )
            ]
        )
        self.teacher_checkpoint = self.tmp / "teacher.pt"
        # _campaign pins sha256(b"teacher") as the expert identity.
        self.teacher_checkpoint.write_bytes(b"teacher")
        self.teacher_checkpoint.chmod(0o444)
        self.sealed_teacher = self.tmp / "sealed-teacher.pt"
        self.sealed_teacher.write_bytes(b"teacher")
        self.sealed_teacher.chmod(0o444)
        self.teacher_sha = sha256_file(self.teacher_checkpoint)
        self.config = self.tmp / "evaluation_config.yaml"
        self.config.write_text(
            f"teacher_actor_path: {self.teacher_checkpoint}\n"
            "algo:\n"
            "  config:\n"
            "    videogen:\n"
            f"      anchor_pool_dir: {self.anchor_pool}\n"
            "      anchor_index: 0\n"
            "fixed_seed: true\n"
        )
        self.scratch = self.tmp / "evaluation-scratch"
        self.scratch.mkdir()
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
            anchor_pool_path=self.anchor_pool,
            anchor_count=1,
            anchor_index=0,
            anchor_inventory_sha256=self.anchor_inventory_sha,
            output="rounds/macro_round_000000/evaluation/contract.json",
        )
        self.results_relative = "rounds/macro_round_000000/evaluation/results"
        self.result_relative = "rounds/macro_round_000000/evaluation/result.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_production_thresholds_reject_an_unchanged_candidate(self) -> None:
        thresholds = json.loads(
            (
                self.repository_root
                / "configs/round_dagger/production_paired_thresholds.json"
            ).read_text(encoding="utf-8")
        )
        rows = [
            {
                "seed": 17 + index // 8,
                "env_index": index % 8,
                "scenario_id": _sha(f"scenario:{index}"),
                "first_condition_sha256": _sha(f"condition:{index}"),
                "first_generated_sha256": _sha(f"generated:{index}"),
                "success": index % 2 == 0,
                "safety_violation": False,
                "action_imitation_mse": 1.0,
                "episode_length": 300,
            }
            for index in range(16)
        ]
        _, decision = _score(rows, rows, thresholds)
        self.assertTrue(decision["bounded_nonregression_all"])
        self.assertFalse(decision["meaningful_improvement_any"])
        self.assertFalse(decision["passed"])

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

    def _run_launcher(
        self,
        results: Path,
        *,
        teacher_sha: str | None = None,
        require_teacher_readonly_mount: bool = False,
        dry_run: bool = False,
        extra_environment: dict[str, str] | None = None,
        anchor_pool: Path | None = None,
        anchor_inventory_sha: str | None = None,
        python_bin: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
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
        command = [
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
            "--teacher-checkpoint",
            str(self.teacher_checkpoint),
            "--teacher-sha256",
            teacher_sha or self.teacher_sha,
            "--sealed-teacher-checkpoint",
            str(self.sealed_teacher),
            "--evaluation-source-inventory-sha256",
            self.evaluation_inventory_sha,
            "--anchor-pool-dir",
            str(anchor_pool or self.anchor_pool),
            "--anchor-count",
            "1",
            "--anchor-index",
            "0",
            "--anchor-inventory-sha256",
            anchor_inventory_sha or self.anchor_inventory_sha,
            "--scratch-root",
            str(self.scratch),
        ]
        if require_teacher_readonly_mount:
            command.append("--require-teacher-readonly-mount")
        if python_bin is not None:
            command.extend(("--python", str(python_bin)))
        if dry_run:
            command.append("--dry-run")
        return subprocess.run(
            command,
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
                **(extra_environment or {}),
            },
        )

    def test_alternate_valid_anchor_pool_is_rejected_by_contract(self) -> None:
        alternate = self.tmp / "alternate-anchors"
        alternate.mkdir()
        anchor = alternate / "anchor_000000.npz"
        anchor.write_bytes(b"different but internally valid anchor")
        alternate_sha = inventory_sha256(
            [
                AnchorRecord(
                    relative_path=anchor.name,
                    size=anchor.stat().st_size,
                    sha256=sha256_file(anchor),
                )
            ]
        )
        completed = self._run_launcher(
            self.root / self.results_relative,
            dry_run=True,
            anchor_pool=alternate,
            anchor_inventory_sha=alternate_sha,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "anchor inputs differ from the immutable evaluation contract",
            completed.stderr,
        )

    def test_launcher_rejects_teacher_digest_and_mount_mismatches(self) -> None:
        results = self.root / self.results_relative
        bad_digest = self._run_launcher(
            results, teacher_sha=_sha("wrong-teacher")
        )
        self.assertNotEqual(bad_digest.returncode, 0)
        self.assertIn(
            "teacher checkpoint SHA differs from campaign", bad_digest.stderr
        )
        self.assertFalse(results.exists())

        self.teacher_checkpoint.chmod(0o644)
        self.teacher_checkpoint.write_bytes(b"tampered teacher")
        self.teacher_checkpoint.chmod(0o444)
        try:
            bad_bytes = self._run_launcher(results)
        finally:
            self.teacher_checkpoint.chmod(0o644)
            self.teacher_checkpoint.write_bytes(b"teacher")
            self.teacher_checkpoint.chmod(0o444)
        self.assertNotEqual(bad_bytes.returncode, 0)
        self.assertIn("teacher checkpoint SHA mismatch", bad_bytes.stderr)
        self.assertFalse(results.exists())

        ordinary_file = self._run_launcher(
            results, require_teacher_readonly_mount=True
        )
        self.assertNotEqual(ordinary_file.returncode, 0)
        self.assertIn(
            "teacher target and sealed source mounts differ",
            ordinary_file.stderr,
        )
        self.assertFalse(results.exists())

    def test_launcher_rechecks_teacher_before_every_role_seed_case(self) -> None:
        fake_bin = self.tmp / "fake-bin"
        fake_bin.mkdir()
        teacher_log = self.tmp / "teacher-sha.log"
        sha_wrapper = fake_bin / "sha256sum"
        sha_wrapper.write_text(
            "#!/usr/bin/env bash\n"
            "set -Eeuo pipefail\n"
            "if [[ ${1:-} == \"$TEACHER_LOG_TARGET\" ]]; then\n"
            "  printf '%s\\n' \"$1\" >>\"$TEACHER_SHA_LOG\"\n"
            "fi\n"
            "exec /usr/bin/sha256sum \"$@\"\n",
            encoding="utf-8",
        )
        sha_wrapper.chmod(0o755)
        completed = self._run_launcher(
            self.root / self.results_relative,
            dry_run=True,
            extra_environment={
                "PATH": f"{fake_bin}:{os.environ['PATH']}",
                "TEACHER_LOG_TARGET": str(self.teacher_checkpoint),
                "TEACHER_SHA_LOG": str(teacher_log),
            },
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        # One startup check plus parent/candidate x two fixed seeds.
        self.assertEqual(
            teacher_log.read_text(encoding="utf-8").splitlines(),
            [str(self.teacher_checkpoint)] * 5,
        )
        self.assertFalse((self.root / self.results_relative).exists())

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
        self.assertEqual(
            loaded["anchor_pool"],
            {
                "path": str(self.anchor_pool),
                "anchor_count": 1,
                "anchor_index": 0,
                "inventory_sha256": self.anchor_inventory_sha,
            },
        )

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
                anchor_pool_path=self.anchor_pool,
                anchor_count=1,
                anchor_index=0,
                anchor_inventory_sha256=self.anchor_inventory_sha,
                output="evaluation/bad-contract.json",
            )
        with self.assertRaisesRegex(
            EvaluationError, "anchor_index must be smaller than anchor_count"
        ):
            create_contract(
                self.root,
                candidate_manifest=self.candidate["manifest_path"],
                scenario_cases=self.cases,
                parent_runtime_policy_sha256=_sha("p"),
                candidate_runtime_policy_sha256=_sha("c"),
                renderer_runtime_sha256=self.renderer_runtime_sha,
                evaluation_config_sha256=sha256_file(self.config),
                evaluator_code_sha256=self.evaluator_code_sha,
                expected_renderer_count=1,
                anchor_pool_path=self.anchor_pool,
                anchor_count=1,
                anchor_index=1,
                anchor_inventory_sha256=self.anchor_inventory_sha,
                output="evaluation/bad-anchor-index-contract.json",
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
                "--teacher-checkpoint",
                str(self.teacher_checkpoint),
                "--teacher-sha256",
                self.teacher_sha,
                "--sealed-teacher-checkpoint",
                str(self.sealed_teacher),
                "--evaluation-source-inventory-sha256",
                self.evaluation_inventory_sha,
                "--anchor-pool-dir",
                str(self.anchor_pool),
                "--anchor-count",
                "1",
                "--anchor-index",
                "0",
                "--anchor-inventory-sha256",
                self.anchor_inventory_sha,
                "--scratch-root",
                str(self.scratch),
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
        bundle = evaluator_source_bundle(
            self.repository_root,
            self.gr00t,
            self.evaluation_inventory_sha,
        )
        self.assertEqual(len(bundle["sources"]), 24)
        self.assertIn(
            "gr00t/gr00t/rl/trl/trainer/anchor_io.py",
            {item["source"] for item in bundle["sources"]},
        )
        self.assertIn(
            "gr00t/gr00t/rl/utils/image3f_frame_client.py",
            {item["source"] for item in bundle["sources"]},
        )
        self.assertEqual(canonical_json_sha256(bundle), self.evaluator_code_sha)
        helper = self.gr00t / "gr00t/rl/utils/round_dagger_paired_eval.py"
        helper.write_text("# changed paired evidence semantics\n")
        self.assertNotEqual(
            evaluator_source_bundle_sha256(
                self.repository_root,
                self.gr00t,
                self.evaluation_inventory_sha,
            ),
            self.evaluator_code_sha,
        )

    def test_evaluator_identity_binds_full_evaluation_inventory(self) -> None:
        self.assertNotEqual(
            evaluator_source_bundle_sha256(
                self.repository_root,
                self.gr00t,
                _sha("different-evaluation-source-inventory"),
            ),
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
                "--teacher-checkpoint",
                str(self.teacher_checkpoint),
                "--teacher-sha256",
                self.teacher_sha,
                "--sealed-teacher-checkpoint",
                str(self.sealed_teacher),
                "--evaluation-source-inventory-sha256",
                self.evaluation_inventory_sha,
                "--anchor-pool-dir",
                str(self.anchor_pool),
                "--anchor-count",
                "1",
                "--anchor-index",
                "0",
                "--anchor-inventory-sha256",
                self.anchor_inventory_sha,
                "--scratch-root",
                str(self.scratch),
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
        self.assertIn("EXPECTED_TRAIN_CHECKPOINT_SHA256=", completed.stdout)
        self.assertIn(
            "++algo.config.init_at_random_ep_len=false", completed.stdout
        )
        rendered_command = completed.stdout.replace("\\", "")
        for override in (
            "raytraced_mix_prob=0.0",
            "image3f_endpoint_files=",
            "view_names=[head_color,hand_left,hand_right]",
            "reference_lag=256",
            "primary_view=head_color",
            "anchor_pool_dir=",
            "anchor_index=0",
        ):
            self.assertIn(f"++algo.config.videogen.{override}", rendered_command)
        self.assertIn(
            "++simulator.config.cameras.policy_views="
            "[head_color,hand_left,hand_right]",
            rendered_command,
        )
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

    @unittest.skipUnless(
        importlib.util.find_spec("hydra") is not None,
        "Hydra is unavailable in the control interpreter",
    )
    def test_production_overrides_compose_against_real_base_eval(self) -> None:
        production_gr00t = self.repository_root / "GR00T-VisualSim2Real"
        evaluator = production_gr00t / "gr00t/rl/eval_agent_trl.py"
        overrides = [
            "++checkpoint=/tmp/checkpoint.pt",
            "++num_envs=1",
            "++headless=true",
            "++seed=17",
            "++eval_output_dir=/tmp/eval",
            "++algo.config.eval.eval_num_envs_episodes=true",
            "++algo.config.eval.num_eval_episodes=1",
            "++algo.config.eval.save_videos=false",
            "++algo.config.eval.save_trajectories=false",
            "++algo.config.init_at_random_ep_len=false",
            "++algo.config.videogen.raytraced_mix_prob=0.0",
            "++algo.config.videogen.image3f_endpoint_files=[/tmp/renderer_0.txt]",
            "++algo.config.videogen.view_names=[head_color,hand_left,hand_right]",
            "++algo.config.videogen.reference_lag=256",
            "++algo.config.videogen.primary_view=head_color",
            "++algo.config.videogen.seed=17",
            "++algo.config.videogen.anchor_pool_dir=/tmp/anchors",
            "++algo.config.videogen.anchor_index=0",
            "++simulator.config.cameras.policy_views="
            "[head_color,hand_left,hand_right]",
            "++env.config.image_dagger_policy_view_count=3",
            "++algo.config.eval.training_enabled=false",
            "++simulator.config.render_results=false",
            "++use_wandb=false",
            "hydra.run.dir=/tmp/eval/hydra",
            "++output_dir=/tmp/eval/output",
        ]
        completed = subprocess.run(
            [sys.executable, str(evaluator), "--cfg", "job", *overrides],
            cwd=production_gr00t,
            env={
                **os.environ,
                "PYTHONPATH": f"{production_gr00t}:{self.repository_root}",
            },
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("init_at_random_ep_len: false", completed.stdout)
        self.assertIn("videogen:", completed.stdout)
        self.assertIn("reference_lag: 256", completed.stdout)
        self.assertIn("policy_views:", completed.stdout)
        self.assertIn("image_dagger_policy_view_count: 3", completed.stdout)

    def test_sequential_sixteen_invocations_bound_and_clean_scratch(self) -> None:
        """One staged checkpoint namespace is retained at a time.

        The fake evaluator creates both a staged checkpoint and the loader's
        sibling checkpoint for 16 sequential role/seed invocations.  It fails
        immediately if any prior invocation remains in scratch, records the
        high-water mark, and verifies the loader receives the SHA returned by
        the final staged-input verification.
        """

        fake_python = self.tmp / "fake-evaluation-python"
        fake_python.write_text(
            r'''#!/usr/bin/env python3
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

args = sys.argv[1:]
scratch = Path(os.environ["FAKE_EVAL_SCRATCH"])
high_water_path = Path(os.environ["FAKE_EVAL_HIGH_WATER"])
invocations_path = Path(os.environ["FAKE_EVAL_INVOCATIONS"])

def scratch_bytes():
    return sum(path.stat().st_size for path in scratch.rglob("*") if path.is_file())

def record_high_water():
    observed = scratch_bytes()
    previous = int(high_water_path.read_text()) if high_water_path.exists() else 0
    high_water_path.write_text(str(max(previous, observed)))
    if observed > 2 * 1024 * 1024 + 4096:
        raise SystemExit("scratch high-water exceeded one checkpoint pair")

def option(name):
    return args[args.index(name) + 1]

if args and args[0] == "-c":
    document = json.load(sys.stdin)
    print(document["staged_checkpoint_sha256"])
    raise SystemExit(0)

if args and args[0] == "-":
    if len(args) == 4:
        print(os.environ["FAKE_EVALUATOR_SHA"])
    elif len(args) == 3:
        for value in (
            os.environ["FAKE_CONFIG_SHA"],
            os.environ["FAKE_EVALUATOR_SHA"],
            "1", "4", os.environ["FAKE_TEACHER_SHA"],
            os.environ["FAKE_ANCHOR_POOL"], "1", "0",
            os.environ["FAKE_ANCHOR_SHA"],
        ):
            print(value)
    elif len(args) == 2:
        print(os.environ["FAKE_TEACHER"])
        print(os.environ["FAKE_ANCHOR_POOL"])
        print("0")
    else:
        raise SystemExit("unexpected inline Python invocation")
    raise SystemExit(0)

if len(args) >= 2 and args[:2] == ["-m", "round_dagger.evaluation_inputs"]:
    staged = Path(option("--staged-checkpoint"))
    digest = hashlib.sha256(staged.read_bytes()).hexdigest()
    print(json.dumps({
        "status": "PASS",
        "staged_checkpoint_sha256": digest,
    }))
    raise SystemExit(0)

if len(args) >= 2 and args[:2] == ["-m", "round_dagger.evaluation"]:
    command = args[2]
    if command == "verify-contract" or command == "verify-outcome":
        raise SystemExit(0)
    if command == "list-cases":
        print(Path(os.environ["FAKE_EVAL_CASES"]).read_text(), end="")
        raise SystemExit(0)
    if command == "run-id":
        role = option("--role")
        seed = option("--seed")
        attempt = int(option("--attempt"))
        print(f"rde-0123456789abcdef-{role}-s{seed}-a{attempt:06d}")
        raise SystemExit(0)
    if command == "attest":
        output = Path(option("--output"))
        output.write_text("{}\n")
        raise SystemExit(0)
    if command == "publish-outcome":
        source = Path(option("--source"))
        output = Path(option("--output"))
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, output)
        raise SystemExit(0)
    if command == "summarize":
        output = Path(option("--output"))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("{}\n")
        raise SystemExit(0)
    raise SystemExit(f"unexpected evaluation command: {command}")

program = args[0] if args else ""
if program.endswith("verify_anchor_pool.py"):
    raise SystemExit(0)
if program.endswith("fixed_seed_gate.py"):
    if "sanitize-checkpoint" not in args:
        raise SystemExit("unexpected fixed-seed command")
    if scratch_bytes() != 0:
        raise SystemExit("previous staged invocation was not cleaned")
    output = Path(option("--output"))
    provenance = Path(option("--provenance"))
    if not output.parent.is_dir():
        raise SystemExit("launcher did not allocate isolated stage directory")
    output.write_bytes(b"S" * (1024 * 1024))
    provenance.write_text("{}\n")
    output.chmod(0o444)
    provenance.chmod(0o444)
    record_high_water()
    raise SystemExit(0)
if program.endswith("image3f_probe.py"):
    output = Path(option("--write-report"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("{}\n")
    raise SystemExit(0)
if program.endswith("eval_agent_metrics_adapter.py"):
    checkpoint_arg = next(value for value in args if value.startswith("++checkpoint="))
    checkpoint = Path(checkpoint_arg.split("=", 1)[1])
    expected = os.environ.get("EXPECTED_TRAIN_CHECKPOINT_SHA256")
    actual = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if expected != actual:
        raise SystemExit("actual loader did not receive verified staged SHA")
    checkpoint.with_name("model_step_000000.pt").write_bytes(b"L" * (1024 * 1024))
    record_high_water()
    metrics = Path(option("--metrics-output"))
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text("{}\n")
    count = int(invocations_path.read_text()) if invocations_path.exists() else 0
    next_count = count + 1
    invocations_path.write_text(str(next_count))
    if int(os.environ.get("FAKE_EVAL_FAIL_AT", "0")) == next_count:
        raise SystemExit("injected evaluator failure after scratch allocation")
    raise SystemExit(0)
raise SystemExit(f"unexpected fake Python invocation: {args!r}")
''',
            encoding="utf-8",
        )
        fake_python.chmod(0o755)

        rows = []
        for policy in self.contract["policies"]:
            for seed in range(100, 108):
                rows.append(
                    "\t".join(
                        (
                            policy["role"],
                            str(seed),
                            policy["checkpoint_path"],
                            policy["checkpoint_sha256"],
                            policy["runtime_policy_sha256"],
                        )
                    )
                )
        cases_path = self.tmp / "fake-sixteen-cases.tsv"
        cases_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        high_water = self.tmp / "fake-scratch-high-water.txt"
        invocations = self.tmp / "fake-evaluator-invocations.txt"
        fake_environment = {
            "FAKE_EVAL_SCRATCH": str(self.scratch),
            "FAKE_EVAL_HIGH_WATER": str(high_water),
            "FAKE_EVAL_INVOCATIONS": str(invocations),
            "FAKE_EVAL_CASES": str(cases_path),
            "FAKE_EVALUATOR_SHA": self.evaluator_code_sha,
            "FAKE_CONFIG_SHA": self.contract["evaluation_config_sha256"],
            "FAKE_TEACHER_SHA": self.teacher_sha,
            "FAKE_TEACHER": str(self.teacher_checkpoint),
            "FAKE_ANCHOR_POOL": str(self.anchor_pool),
            "FAKE_ANCHOR_SHA": self.anchor_inventory_sha,
        }
        completed = self._run_launcher(
            self.root / self.results_relative,
            python_bin=fake_python,
            extra_environment=fake_environment,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(int(invocations.read_text()), 16)
        self.assertLessEqual(int(high_water.read_text()), 2 * 1024 * 1024 + 4096)
        self.assertEqual(
            [path for path in self.scratch.rglob("*") if path.is_file()], []
        )

        # A failure after both checkpoint-sized files exist must take the same
        # bounded cleanup path while preserving the durable failed attempt.
        failed_cases = self.tmp / "fake-failed-case.tsv"
        failed_cases.write_text(rows[0].replace("\t100\t", "\t900\t") + "\n")
        failed_invocations = self.tmp / "fake-failed-invocations.txt"
        failed = self._run_launcher(
            self.root / "rounds/macro_round_000000/evaluation/failure-results",
            python_bin=fake_python,
            extra_environment={
                **fake_environment,
                "FAKE_EVAL_CASES": str(failed_cases),
                "FAKE_EVAL_INVOCATIONS": str(failed_invocations),
                "FAKE_EVAL_FAIL_AT": "1",
            },
        )
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("injected evaluator failure", failed.stderr)
        self.assertEqual(int(failed_invocations.read_text()), 1)
        self.assertEqual(
            [path for path in self.scratch.rglob("*") if path.is_file()], []
        )

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
            "--teacher-checkpoint",
            str(self.teacher_checkpoint),
            "--teacher-sha256",
            self.teacher_sha,
            "--sealed-teacher-checkpoint",
            str(self.sealed_teacher),
            "--evaluation-source-inventory-sha256",
            self.evaluation_inventory_sha,
            "--anchor-pool-dir",
            str(self.anchor_pool),
            "--anchor-count",
            "1",
            "--anchor-index",
            "0",
            "--anchor-inventory-sha256",
            self.anchor_inventory_sha,
            "--scratch-root",
            str(self.scratch),
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
