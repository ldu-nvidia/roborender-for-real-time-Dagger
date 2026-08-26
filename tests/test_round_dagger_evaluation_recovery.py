"""End-to-end tests for bounded hardened evaluation recovery."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from round_dagger import evaluation, evaluation_recovery, promotion, state
from round_dagger.artifacts import (
    canonical_json_sha256,
    publish_immutable_json,
    sha256_file,
)
from round_dagger.state import load_campaign, load_state
from scripts.round_dagger.verify_anchor_pool import AnchorRecord, inventory_sha256

try:
    from test_round_dagger_state import _campaign, _sha, _stage_and_bind, _to_ingested
except ImportError:  # pragma: no cover
    from tests.test_round_dagger_state import (  # type: ignore[no-redef]
        _campaign,
        _sha,
        _stage_and_bind,
        _to_ingested,
    )


class HardenedEvaluationRecoveryTest(unittest.TestCase):
    def setUp(self) -> None:
        scratch = Path(
            "/mnt/data2/ldu/roborender_round_dagger_validation/unit_tests"
        )
        scratch.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=scratch)
        self.tmp = Path(self.temporary.name)
        self.live_repository_root = Path(__file__).parents[1]
        self.repository_root = self.tmp / "evaluation-source"
        repository_sources = set(evaluation._REPOSITORY_EVALUATOR_SOURCES)
        repository_sources.update(
            evaluation_recovery._RECOVERY_REPOSITORY_SOURCES
        )
        for relative in sorted(repository_sources):
            source = self.live_repository_root / relative
            destination = self.repository_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        self.evaluation_source_inventory_sha = _sha(
            "evaluation-source-inventory"
        )
        self.root = _campaign(self.tmp)
        _to_ingested(self.root)
        self.candidate = _stage_and_bind(self.root, self.tmp)
        self.state = load_state(self.root)
        self.campaign = load_campaign(self.root)

        self.gr00t = self.repository_root / "GR00T-VisualSim2Real"
        for relative in (
            "gr00t/rl/eval_agent_trl.py",
            "gr00t/rl/trl/trainer/anchor_io.py",
            "gr00t/rl/trl/trainer/checkpoint_io.py",
            "gr00t/rl/trl/trainer/distill_trainer.py",
            "gr00t/rl/trl/trainer/distill_trainer_obj_pred_homie_api.py",
            "gr00t/rl/trl/trainer/distill_trainer_videogen.py",
            "gr00t/rl/trl/trainer/ppo_trainer.py",
            "gr00t/rl/utils/image3f_frame_client.py",
            "gr00t/rl/utils/round_dagger_paired_eval.py",
        ):
            source = self.gr00t / relative
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_text(f"# recovery fixture: {relative}\n", encoding="utf-8")
        self.anchor_pool = self.tmp / "anchors"
        self.anchor_pool.mkdir()
        anchor = self.anchor_pool / "anchor_000000.npz"
        anchor.write_bytes(b"recovery anchor fixture")
        self.anchor_inventory_sha = inventory_sha256(
            [
                AnchorRecord(
                    relative_path=anchor.name,
                    size=anchor.stat().st_size,
                    sha256=sha256_file(anchor),
                )
            ]
        )
        self.config = self.tmp / "evaluation.yaml"
        self.config.write_text("fixed_seed: true\n", encoding="utf-8")
        self.scenarios = self.tmp / "scenarios.json"
        self.cases = [
            {"seed": seed, "env_index": env, "scenario_id": _sha(f"{seed}:{env}")}
            for seed in (17, 29)
            for env in range(4)
        ]
        self.scenarios.write_text(json.dumps(self.cases), encoding="utf-8")
        self.thresholds = self.tmp / "thresholds.json"
        self.thresholds.write_text(
            json.dumps(evaluation.DEFAULT_THRESHOLDS), encoding="utf-8"
        )
        self.renderer_identity = {
            "schema_version": 1,
            "release_id": "recovery-fixture",
            "adapter_sha256": _sha("adapter"),
            "generation": {"steps": 5, "views": list(evaluation.POLICY_VIEWS)},
        }
        self.renderer_runtime_sha = canonical_json_sha256(self.renderer_identity)
        self.receipt = self._publish_learner_receipt()
        policy_dir = self.root / "policies/policy_000001"
        os.chmod(policy_dir, 0o750)
        self.hold = evaluation_recovery.attest_promotion_hold(
            self.root, legacy_orchestrator_pid=999_999_937
        )

    def tearDown(self) -> None:
        policy_dir = self.root / "policies/policy_000001"
        if policy_dir.exists():
            os.chmod(policy_dir, 0o750)
        self.temporary.cleanup()

    def _publish_learner_receipt(self) -> dict:
        current = load_state(self.root)
        learner = self.root / "rounds/macro_round_000000/learner"
        learner.mkdir(parents=True, exist_ok=True)
        result = learner / "learner_result_round_000000.json"
        publish_immutable_json(result, {"fixture": "recovered learner"})
        snapshot = self.root / "replay/snapshots/through_macro_round_000000.json"
        publish_immutable_json(snapshot, {"fixture": "replay snapshot"})
        receipt = {
            "schema": evaluation_recovery.LEARNER_RECOVERY_RECEIPT_SCHEMA,
            "status": "TRAINED",
            "campaign_id": self.campaign["campaign_id"],
            "campaign_sha256": current["campaign_sha256"],
            "macro_round": 0,
            "ingested_revision": current["revision"] - 1,
            "trained_revision": current["revision"],
            "replay_snapshot_file_sha256": sha256_file(snapshot),
            "replay_snapshot_sha256": current["replay_snapshot_sha256"],
            "sealed_source_inventory_sha256": _sha("sealed-source"),
            "learner_config_sha256": _sha("learner-config"),
            "learner_result_path": result.relative_to(self.root).as_posix(),
            "learner_result_sha256": sha256_file(result),
            "candidate_manifest_path": self.candidate["manifest_path"],
            "candidate_manifest_sha256": self.candidate["manifest_sha256"],
            "candidate_checkpoint_sha256": self.candidate["checkpoint_sha256"],
            "plan_sha256": _sha("learner-recovery-plan"),
        }
        path = (
            self.root
            / "rounds/macro_round_000000/learner_recovery/trained_receipt.json"
        )
        digest = publish_immutable_json(path, receipt)
        return {
            **receipt,
            "path": path.relative_to(self.root).as_posix(),
            "sha256": digest,
        }

    def _prepare(self) -> dict:
        return evaluation_recovery.prepare_recovery(
            self.root,
            repository_root=self.repository_root,
            gr00t_root=self.gr00t,
            hold_attestation_path=self.hold["attestation_path"],
            expected_hold_attestation_sha256=self.hold["attestation_sha256"],
            scenario_cases_path=self.scenarios,
            thresholds_path=self.thresholds,
            evaluation_config_path=self.config,
            renderer_runtime_sha256=self.renderer_runtime_sha,
            expected_renderer_count=1,
            learner_recovery_receipt_path=self.receipt["path"],
            expected_learner_recovery_receipt_sha256=self.receipt["sha256"],
            evaluation_source_inventory_sha256=(
                self.evaluation_source_inventory_sha
            ),
            anchor_pool_path=self.anchor_pool,
            anchor_count=1,
            anchor_index=0,
            anchor_inventory_sha256=self.anchor_inventory_sha,
        )

    def _trusted_run(self, *, environment: dict[str, str] | None = None) -> dict:
        return evaluation_recovery.run_trusted_recovery(
            self.root,
            repository_root=self.repository_root,
            gr00t_root=self.gr00t,
            hold_attestation_path=self.hold["attestation_path"],
            expected_hold_attestation_sha256=self.hold["attestation_sha256"],
            scenario_cases_path=self.scenarios,
            thresholds_path=self.thresholds,
            evaluation_config_path=self.config,
            renderer_runtime_sha256=self.renderer_runtime_sha,
            expected_renderer_count=1,
            learner_recovery_receipt_path=self.receipt["path"],
            expected_learner_recovery_receipt_sha256=self.receipt["sha256"],
            evaluation_source_inventory_sha256=(
                self.evaluation_source_inventory_sha
            ),
            anchor_pool_path=self.anchor_pool,
            anchor_count=1,
            anchor_index=0,
            anchor_inventory_sha256=self.anchor_inventory_sha,
            environment={} if environment is None else environment,
            hook_timeout_seconds=60,
        )

    def _snapshot(self, descriptors: list[int]) -> tuple[int, list[str]]:
        descriptor = evaluation_recovery.orchestration._create_sealable_memfd(
            "recovery-test-hook"
        )
        os.write(descriptor, b"#!/usr/bin/env bash\nexit 0\n")
        os.lseek(descriptor, 0, os.SEEK_SET)
        descriptors.append(descriptor)
        return descriptor, ["/bin/bash", f"/proc/self/fd/{descriptor}"]

    def _prepared_plan(self) -> dict:
        paths = list(
            self.root.glob(
                "rounds/macro_round_000000/evaluation_recovery/"
                "hardened-*/recovery_plan.json"
            )
        )
        self.assertEqual(len(paths), 1)
        path = paths[0]
        return {
            **json.loads(path.read_text(encoding="utf-8")),
            "plan_path": path.relative_to(self.root).as_posix(),
            "plan_sha256": sha256_file(path),
        }

    def _produce_result(self, plan: dict, *, passing: bool) -> dict:
        contract = evaluation.load_contract(
            self.root, self.root / plan["evaluator_contract_path"]
        )
        recovery_root = self.root / plan["recovery_root"]
        for role in ("parent", "candidate"):
            for seed in contract["seeds"]:
                raw = recovery_root / "raw" / role / f"seed_{seed}"
                raw.mkdir(parents=True, exist_ok=True)
                policy = next(
                    item for item in contract["policies"] if item["role"] == role
                )
                run_id = evaluation.evaluation_run_id(
                    contract, role=role, seed=seed, attempt=1
                )
                conditions = {
                    str(env): _sha(f"condition:{seed}:{env}") for env in range(4)
                }
                generated = {
                    str(env): _sha(f"generated:{seed}:{env}") for env in range(4)
                }
                successes = (
                    [True, True, True, True]
                    if role == "parent" or passing
                    else [False, False, False, False]
                )
                paired = []
                fixed = []
                for case, success in zip(
                    [item for item in self.cases if item["seed"] == seed],
                    successes,
                    strict=True,
                ):
                    env = case["env_index"]
                    paired.append(
                        {
                            "env_index": env,
                            "seed": seed,
                            "scenario_id": case["scenario_id"],
                            "success": success,
                            "safety_violation": False,
                            "action_imitation_mse": 1.0,
                            "episode_length": 300,
                            "generated_view_sha256": {
                                view: _sha(f"{role}:{seed}:{env}:{view}")
                                for view in evaluation.POLICY_VIEWS
                            },
                        }
                    )
                    fixed.append(
                        {
                            "env_index": env,
                            "success": success,
                            "reward": float(env),
                            "episode_length": 300,
                            "object_class": "fixture",
                            "seed": seed,
                            "first_condition_sha256": conditions[str(env)],
                            "first_generated_sha256": generated[str(env)],
                            "scenario_id": canonical_json_sha256(
                                {
                                    "seed": seed,
                                    "env_index": env,
                                    "first_condition_sha256": conditions[str(env)],
                                }
                            ),
                        }
                    )
                frames = 300 * 4
                metrics = {
                    "_roborender_eval_adapter": {
                        "schema_version": 1,
                        "image3f_run_id": run_id,
                    },
                    "_roborender_paired_evaluation": {
                        "schema": evaluation.RUNTIME_SCHEMA,
                        "eval_relabel_enabled": True,
                        "policy_views": list(evaluation.POLICY_VIEWS),
                        "view_layout": "vertical",
                        "raytraced_mix_prob": 0.0,
                        "reference_lag": 256,
                        "training_enabled": False,
                        "one_episode_per_environment": True,
                        "generation_seed": seed,
                        "runtime_policy_sha256": policy["runtime_policy_sha256"],
                        "evaluation_config_sha256": contract[
                            "evaluation_config_sha256"
                        ],
                        "evaluator_code_sha256": contract["evaluator_code_sha256"],
                    },
                    "generated_policy_steps": 300,
                    "generated_frames": frames,
                    "_roborender_image3f_eval": {
                        "schema": "fast_roborender.image3f_fixed_eval_runtime.v1",
                        "eval_relabel_enabled": True,
                        "raytraced_mix_prob": 0.0,
                        "reference_lag": 256,
                        "generation_seed": seed,
                        "policy_views": list(evaluation.POLICY_VIEWS),
                        "generated_policy_steps": 300,
                        "generated_frames": frames,
                        "first_condition_sha256_by_env": conditions,
                        "first_generated_sha256_by_env": generated,
                    },
                    "_roborender_fixed_seed_episodes": fixed,
                    "_roborender_paired_episodes": paired,
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
                after["selected"][0]["response"]["metrics"]["requests"] += frames
                metrics_path = raw / "metrics.json"
                before_path = raw / "before.json"
                after_path = raw / "after.json"
                for path, payload in (
                    (metrics_path, metrics),
                    (before_path, before),
                    (after_path, after),
                ):
                    path.write_text(json.dumps(payload), encoding="utf-8")
                evaluation.attest_outcome(
                    self.root,
                    contract_path=contract["contract_path"],
                    role=role,
                    seed=seed,
                    metrics_path=metrics_path,
                    renderer_before_path=before_path,
                    renderer_after_path=after_path,
                    image3f_run_id=run_id,
                    output=(
                        recovery_root
                        / "results"
                        / role
                        / f"seed_{seed}"
                        / "outcome.json"
                    ),
                )
        return evaluation.summarize_to_evaluator_result(
            self.root,
            contract_path=contract["contract_path"],
            results_dir=recovery_root / "results",
            output=recovery_root / "result.json",
        )

    def _finalize(self, plan: dict) -> dict:
        return evaluation_recovery.finalize_recovery(
            self.root,
            plan_path=plan["plan_path"],
            expected_plan_sha256=plan["plan_sha256"],
            repository_root=self.repository_root,
            gr00t_root=self.gr00t,
        )

    def _publish_evaluation_before_crash(self, plan: dict) -> dict:
        verified = evaluation.verify_evaluator_result_evidence(
            self.root,
            contract_path=self.root / plan["evaluator_contract_path"],
            results_dir=self.root / plan["recovery_root"] / "results",
            result_path=self.root / plan["evaluator_result_path"],
        )
        policy_dir = self.root / "policies/policy_000001"
        os.chmod(policy_dir, 0o750)
        return promotion.write_evaluation(
            self.root,
            candidate_manifest=self.candidate["manifest_path"],
            passed=True,
            metrics=verified["metrics"],
            evaluator_config_sha256=verified["evaluator_config_sha256"],
            evidence_bundle_path=verified["evidence_bundle_path"],
            evidence_bundle_sha256=verified["evidence_bundle_sha256"],
        )

    def test_absent_legacy_evaluation_is_attested_and_promotion_is_atomic(self) -> None:
        legacy = self.root / "rounds/macro_round_000000/evaluation"
        self.assertFalse(legacy.exists())
        plan = self._prepare()
        inventory = json.loads(
            (self.root / plan["legacy_inventory_path"]).read_text(encoding="utf-8")
        )
        self.assertEqual(inventory["presence"], "absent")
        self.assertEqual(inventory["files"], [])
        self.assertIsNone(plan["legacy_contract_sha256"])
        self.assertIsNone(plan["legacy_result_sha256"])
        self.assertEqual(
            plan["learner_source_inventory_sha256"],
            self.receipt["sealed_source_inventory_sha256"],
        )
        self.assertEqual(
            plan["evaluation_source_inventory_sha256"],
            self.evaluation_source_inventory_sha,
        )
        self.assertNotEqual(
            plan["learner_source_inventory_sha256"],
            plan["evaluation_source_inventory_sha256"],
        )
        self.assertEqual(plan["anchor_pool_path"], str(self.anchor_pool))
        self.assertEqual(plan["anchor_count"], 1)
        self.assertEqual(plan["anchor_index"], 0)
        self.assertEqual(
            plan["anchor_inventory_sha256"], self.anchor_inventory_sha
        )
        self.assertFalse(legacy.exists())
        self._produce_result(plan, passing=True)
        completed = self._finalize(plan)
        current = load_state(self.root)
        self.assertEqual(current["phase"], "READY")
        self.assertEqual(current["active_policy"]["version"], 1)
        self.assertEqual(
            (self.root / "policies/policy_000001").stat().st_mode & 0o777,
            0o750,
        )
        self.assertFalse(legacy.exists())
        self.assertEqual(self._finalize(plan), completed)

    def test_trusted_runner_verifies_before_any_recovery_publication(self) -> None:
        before = load_state(self.root)
        prepare = mock.patch.object(
            evaluation_recovery,
            "prepare_recovery",
            wraps=evaluation_recovery.prepare_recovery,
        )
        with (
            prepare as prepare_call,
            mock.patch.object(evaluation_recovery.subprocess, "run") as execute,
            mock.patch.object(
                evaluation_recovery, "finalize_recovery"
            ) as finalize,
        ):
            with self.assertRaisesRegex(
                evaluation_recovery.EvaluationRecoveryError,
                "trusted coordinator verification",
            ):
                self._trusted_run()
        self.assertNotEqual(self.repository_root.stat().st_mode & 0o777, 0o555)
        prepare_call.assert_not_called()
        execute.assert_not_called()
        finalize.assert_not_called()
        self.assertEqual(load_state(self.root), before)
        self.assertFalse(
            (
                self.root
                / "rounds/macro_round_000000/evaluation_recovery"
            ).exists()
        )
        self.assertFalse(
            (self.root / "policies/policy_000001/evaluation.json").exists()
        )
        self.assertEqual(
            (self.root / "policies/policy_000001").stat().st_mode & 0o777,
            0o555,
        )

    def test_recovery_shell_is_a_logic_free_python_entry(self) -> None:
        script = (
            self.repository_root
            / "scripts/round_dagger/run_hardened_evaluation_recovery.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("-m round_dagger.evaluation_recovery run", script)
        self.assertNotIn("evaluation_recovery prepare", script)
        self.assertNotIn("evaluation_recovery finalize", script)
        self.assertNotIn('bash "$HOOK"', script)
        self.assertNotIn("ROUND_DAGGER_TRUSTED_EVALUATION_DISPATCH", script)

    def test_trusted_runner_hook_failure_is_held_and_retry_is_idempotent(self) -> None:
        descriptors: list[int] = []
        calls = 0

        def snapshot(*_args, **_kwargs):
            return self._snapshot(descriptors)

        def execute(argv, **kwargs):
            nonlocal calls
            calls += 1
            descriptor = kwargs["pass_fds"][0]
            self.assertEqual(
                argv, ["/bin/bash", f"/proc/self/fd/{descriptor}"]
            )
            self.assertEqual(
                kwargs["env"]["ROUND_DAGGER_TRUSTED_EVALUATION_DISPATCH"],
                "1",
            )
            if calls == 1:
                return subprocess.CompletedProcess(argv, 19)
            self._produce_result(self._prepared_plan(), passing=True)
            return subprocess.CompletedProcess(argv, 0)

        with (
            mock.patch.object(
                evaluation_recovery.orchestration,
                "_verified_evaluation_hook",
                side_effect=snapshot,
            ),
            mock.patch.object(
                evaluation_recovery.subprocess, "run", side_effect=execute
            ),
        ):
            with self.assertRaisesRegex(
                evaluation_recovery.EvaluationRecoveryError,
                "exit status 19",
            ):
                self._trusted_run()
            first_plan = self._prepared_plan()
            self.assertEqual(load_state(self.root)["phase"], "TRAINED")
            self.assertEqual(
                (self.root / "policies/policy_000001").stat().st_mode & 0o777,
                0o555,
            )
            self.assertFalse(
                (self.root / "policies/policy_000001/evaluation.json").exists()
            )
            completed = self._trusted_run()
            second_plan = self._prepared_plan()

        self.assertEqual(first_plan["plan_sha256"], second_plan["plan_sha256"])
        self.assertEqual(calls, 2)
        self.assertEqual(load_state(self.root)["phase"], "READY")
        self.assertEqual(completed["active_policy_version"], 1)
        for descriptor in descriptors:
            with self.assertRaises(OSError):
                os.fstat(descriptor)

    def test_trusted_runner_uses_latched_hook_and_finalizes_after_success(self) -> None:
        descriptors: list[int] = []
        events: list[str] = []
        observed_environment: dict[str, str] = {}
        real_prepare = evaluation_recovery.prepare_recovery
        real_finalize = evaluation_recovery.finalize_recovery

        def snapshot(*_args, **_kwargs):
            events.append("snapshot")
            return self._snapshot(descriptors)

        def prepare(*args, **kwargs):
            events.append("prepare")
            return real_prepare(*args, **kwargs)

        def execute(argv, **kwargs):
            events.append("execute")
            observed_environment.update(kwargs["env"])
            descriptor = kwargs["pass_fds"][0]
            self.assertEqual(
                argv, ["/bin/bash", f"/proc/self/fd/{descriptor}"]
            )
            self._produce_result(self._prepared_plan(), passing=True)
            return subprocess.CompletedProcess(argv, 0)

        def finalize(*args, **kwargs):
            events.append("finalize")
            return real_finalize(*args, **kwargs)

        inherited = {
            "ROUND_DAGGER_TRUSTED_EVALUATION_DISPATCH": "spoofed",
            "ROUND_DAGGER_EVALUATION_RECOVERY_ID": "wrong",
            "ROUND_DAGGER_EVALUATOR_RESULT": "/tmp/wrong-result",
        }
        with (
            mock.patch.object(
                evaluation_recovery.orchestration,
                "_verified_evaluation_hook",
                side_effect=snapshot,
            ),
            mock.patch.object(
                evaluation_recovery, "prepare_recovery", side_effect=prepare
            ),
            mock.patch.object(
                evaluation_recovery.subprocess, "run", side_effect=execute
            ),
            mock.patch.object(
                evaluation_recovery, "finalize_recovery", side_effect=finalize
            ),
        ):
            completed = self._trusted_run(environment=inherited)

        self.assertEqual(events, ["snapshot", "prepare", "execute", "finalize"])
        plan = self._prepared_plan()
        self.assertEqual(
            observed_environment["ROUND_DAGGER_EVALUATION_RECOVERY_ID"],
            plan["recovery_id"],
        )
        self.assertEqual(
            observed_environment["ROUND_DAGGER_EVALUATOR_RESULT"],
            str(self.root / plan["evaluator_result_path"]),
        )
        self.assertEqual(
            observed_environment["ROUND_DAGGER_TRUSTED_EVALUATION_DISPATCH"],
            "1",
        )
        self.assertEqual(completed["active_policy_version"], 1)
        self.assertEqual(load_state(self.root)["phase"], "READY")
        for descriptor in descriptors:
            with self.assertRaises(OSError):
                os.fstat(descriptor)

    def test_failed_hardened_gate_keeps_candidate_held_and_unpublished(self) -> None:
        plan = self._prepare()
        self._produce_result(plan, passing=False)
        with self.assertRaisesRegex(
            evaluation_recovery.EvaluationRecoveryError, "failed; candidate"
        ):
            self._finalize(plan)
        self.assertEqual(load_state(self.root)["phase"], "TRAINED")
        self.assertEqual(
            (self.root / "policies/policy_000001").stat().st_mode & 0o777,
            0o555,
        )
        self.assertFalse((self.root / "policies/policy_000001/evaluation.json").exists())

    def test_restart_reholds_train_phase_writable_window(self) -> None:
        plan = self._prepare()
        self._produce_result(plan, passing=True)
        # Simulate SIGKILL after evaluation.json publication but before the
        # normal code can reapply 0555 and execute TRAINED -> EVALUATED.
        self._publish_evaluation_before_crash(plan)
        policy_dir = self.root / "policies/policy_000001"
        self.assertEqual(policy_dir.stat().st_mode & 0o777, 0o750)
        self.assertEqual(load_state(self.root)["phase"], "TRAINED")
        self._finalize(plan)
        self.assertEqual(load_state(self.root)["phase"], "READY")
        self.assertEqual(policy_dir.stat().st_mode & 0o777, 0o750)

    def test_restart_reholds_evaluated_phase_writable_window(self) -> None:
        plan = self._prepare()
        self._produce_result(plan, passing=True)
        evidence = self._publish_evaluation_before_crash(plan)
        policy_dir = self.root / "policies/policy_000001"
        os.chmod(policy_dir, 0o555)
        current = load_state(self.root)
        state.transition(
            self.root,
            expected_phase="TRAINED",
            next_phase="EVALUATED",
            macro_round=0,
            evidence_path=evidence["manifest_path"],
            evidence_sha256=evidence["evidence_sha256"],
            expected_revision=current["revision"],
        )
        # Simulate SIGKILL after restoring the policy directory for
        # promotion.json publication but before the EVALUATED -> READY CAS.
        os.chmod(policy_dir, 0o750)
        self.assertEqual(load_state(self.root)["phase"], "EVALUATED")
        self._finalize(plan)
        self.assertEqual(load_state(self.root)["phase"], "READY")
        self.assertEqual(policy_dir.stat().st_mode & 0o777, 0o750)

    def test_restart_reholds_before_rejecting_changed_legacy_tree(self) -> None:
        plan = self._prepare()
        self._produce_result(plan, passing=True)
        self._publish_evaluation_before_crash(plan)
        policy_dir = self.root / "policies/policy_000001"
        legacy = self.root / "rounds/macro_round_000000/evaluation"
        legacy.mkdir(parents=True)
        (legacy / "appeared_after_crash.txt").write_text(
            "must not become promotion evidence\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(
            evaluation_recovery.EvaluationRecoveryError,
            "legacy evaluation changed",
        ):
            self._finalize(plan)
        self.assertEqual(load_state(self.root)["phase"], "TRAINED")
        self.assertEqual(policy_dir.stat().st_mode & 0o777, 0o555)

    def test_legacy_diagnostic_tree_is_inventoried_and_never_overwritten(self) -> None:
        # Recreate setup before the hold so the attestation binds a present tree.
        policy_dir = self.root / "policies/policy_000001"
        os.chmod(policy_dir, 0o750)
        hold_path = self.root / self.hold["attestation_path"]
        hold_path.chmod(0o600)
        hold_path.unlink()
        legacy = self.root / "rounds/macro_round_000000/evaluation"
        legacy.mkdir(parents=True)
        (legacy / "diagnostic.txt").write_text("legacy-only\n", encoding="utf-8")
        old_digest = sha256_file(legacy / "diagnostic.txt")
        self.hold = evaluation_recovery.attest_promotion_hold(
            self.root, legacy_orchestrator_pid=999_999_937
        )
        plan = self._prepare()
        inventory = json.loads(
            (self.root / plan["legacy_inventory_path"]).read_text(encoding="utf-8")
        )
        self.assertEqual(inventory["presence"], "present")
        self.assertEqual(inventory["files"][0]["sha256"], old_digest)
        self.assertEqual(sha256_file(legacy / "diagnostic.txt"), old_digest)


if __name__ == "__main__":
    unittest.main()
