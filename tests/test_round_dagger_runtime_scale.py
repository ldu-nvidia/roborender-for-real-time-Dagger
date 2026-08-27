"""Contracts for the exact-PASS-bound metadata-only shadow scale ladder."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from round_dagger.artifacts import publish_immutable_json, publish_rollout_shard
from round_dagger.runtime import (
    RUNTIME_ADAPTER_INPUT_PROFILE_ENV,
    RUNTIME_ADAPTER_MODE_ENV,
    RUNTIME_ADAPTER_SHADOW_ACK_ENV,
    RUNTIME_ADAPTER_VALIDATION_ACK_ENV,
)
from round_dagger.runtime_ab import compare_runtime_ab
from round_dagger.runtime_scale import (
    EVIDENCE_SCHEMA,
    RUNGS,
    RUNTIME_SCALE_EVIDENCE_ENV,
    RUNTIME_SCALE_RUNG_CONTRACT_SHA256_ENV,
    RUNTIME_SCALE_VALIDATION_ACK,
    RuntimeScaleError,
    authorize_runtime_scale,
    authorize_runtime_scale_process,
    prepare_runtime_scale,
    runtime_scale_environment,
    seal_runtime_scale,
    validate_runtime_scale,
    verify_runtime_scale,
)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class RuntimeScaleTest(unittest.TestCase):
    def setUp(self) -> None:
        # Reuse the exact A/B fixture without inheriting its test methods.
        from tests.test_round_dagger_runtime_ab import RuntimeAdapterABTest

        self.ab = RuntimeAdapterABTest(methodName="test_cli_exposes_all_validation_operations")
        self.ab.setUp()
        self.addCleanup(self.ab.tearDown)
        self.ab.complete()
        self.ab_result = compare_runtime_ab(self.ab.plan_path)["result"]
        self.scale_root = self.ab.base / "runtime-scale"
        self.report = prepare_runtime_scale(
            self.scale_root,
            run_id="runtime-scale-test",
            runtime_ab_result=self.ab_result,
        )
        self.plan = self.scale_root / "plan.json"

    def _documents(self, environments: int):
        plan = json.loads(self.plan.read_text(encoding="utf-8"))
        reference = next(
            item
            for item in plan["rung_contracts"]
            if item["environments_per_rank"] == environments
        )
        rung = json.loads(Path(reference["path"]).read_text(encoding="utf-8"))
        ab_plan = json.loads(self.ab.plan_path.read_text(encoding="utf-8"))
        return plan, rung, ab_plan

    @staticmethod
    def _rollout_arrays(environments: int):
        shape = (2, environments)
        expert = np.zeros((*shape, 26), dtype=np.float32)
        student = np.ones((*shape, 26), dtype=np.float32)
        expert_mask = np.zeros(shape, dtype=np.bool_)
        global_ids = np.broadcast_to(
            np.arange(environments, dtype=np.int64), shape
        ).copy()
        episode_steps = np.broadcast_to(
            np.arange(2, dtype=np.int64)[:, None], shape
        ).copy()
        policy_rgb = np.zeros(
            (2, environments, 324, 192, 3), dtype=np.uint8
        )
        for step in range(2):
            for environment in range(environments):
                value = 1 + step * 64 + environment
                policy_rgb[step, environment].fill(value)
                policy_rgb[step, environment, 0, 0, 0] = value + 1
        return {
            "policy_rgb": policy_rgb,
            "expert_actions": expert,
            "student_actions": student,
            "behavior_actions": student.copy(),
            "expert_mask": expert_mask,
            "dones": np.zeros(shape, dtype=np.bool_),
            "observation_is_raytraced": np.ones(shape, dtype=np.bool_),
            "global_env_ids": global_ids,
            "episode_ids": np.zeros(shape, dtype=np.int64),
            "episode_steps": episode_steps,
            "obs__actor_obs": np.zeros((2, environments, 3), dtype=np.float32),
        }

    @staticmethod
    def _telemetry(step: int, environments: int):
        return {
            "step": step,
            "metrics": {
                "renderer/client_wall_ms": 9.0 + step,
                "renderer/inference_ms": 7.0 + step,
                "renderer/ms": 8.0 + step,
                "renderer/parallel_batch_size": environments,
                "renderer/parallel_batch_wall_ms": 10.0 + step,
                "renderer/queue_ms": 0.25 + step,
                "renderer/reference_provenance/configured_lag": 256,
                "renderer/reference_provenance/request_step": step,
                "renderer/reference_provenance/source_step": 0,
                "renderer/reference_provenance_valid": 1,
                "renderer/reference_source_anchor": 1,
                "renderer/reference_source_client": 0,
                "renderer/reference_source_generated_lag": 0,
                "renderer/reference_source_idempotent_replay": 0,
                "renderer/reference_source_unknown": 0,
                "renderer/stream_step": step,
                "rollout_frame_step": step,
                "telemetry/view_count": 3,
            },
        }

    @staticmethod
    def _evidence(environments: int):
        return {
            "schema": EVIDENCE_SCHEMA,
            "status": "informational_only",
            "promotion_gate": False,
            "validation_ack": RUNTIME_SCALE_VALIDATION_ACK,
            "adapter_arm": "shadow",
            "input_profile": "live-isaac-v1",
            "rank": 0,
            "world_size": 1,
            "dimensions": {"local_num_envs": environments, "rollout_steps": 2},
            "cuda_determinism": {
                "cublas_workspace_config": None,
                "cudnn_allow_tf32": True,
                "cudnn_benchmark": True,
                "cudnn_deterministic": False,
                "deterministic_algorithms": False,
                "deterministic_warn_only": False,
                "enabled": False,
                "matmul_allow_tf32": True,
                "matmul_precision": "highest",
            },
            "policy_inputs": {
                "key": "vision_obs",
                "records": [
                    {
                        "dtype": "torch.float32",
                        "sha256": digest(f"scale-input-{environments}-{step}"),
                        "shape": [environments, 324, 192, 3],
                        "step": step,
                    }
                    for step in range(2)
                ],
            },
            "rollout": {
                "wall_seconds": 0.5,
                "synchronized": True,
                "cuda": {
                    "enabled": True,
                    "device": "cuda:0",
                    "baseline_allocated_bytes": 100,
                    "baseline_reserved_bytes": 120,
                    "peak_allocated_bytes": 140,
                    "peak_reserved_bytes": 180,
                    "peak_delta_allocated_bytes": 40,
                    "peak_delta_reserved_bytes": 60,
                },
            },
            "audit": {
                "enabled": True,
                "num_envs": environments,
                "expected_steps": 2,
                "decision_count": 2,
                "transition_count": 2,
                "callback_count": 6,
                "callback_order_valid": True,
                "sequential_steps_valid": True,
                "batch_dimensions_valid": True,
            },
            "replay_export": {"successful": True, "published_shard_parts": 1},
        }

    def _complete(
        self,
        environments: int,
        *,
        evidence: bool = True,
        telemetry: bool = True,
        degenerate_rgb: bool = False,
        duplicate_rgb: bool = False,
    ) -> None:
        plan, rung, ab_plan = self._documents(environments)
        arrays = self._rollout_arrays(environments)
        if degenerate_rgb:
            arrays["policy_rgb"].fill(0)
        if duplicate_rgb:
            arrays["policy_rgb"][:, 1:] = arrays["policy_rgb"][:, :1]
        publish_rollout_shard(
            rung["cohort_dir"],
            rank=0,
            rollout_id=0,
            part_index=0,
            arrays=arrays,
            campaign_id=plan["campaign_id"],
            macro_round=plan["macro_round"],
            cohort_index=rung["cohort_index"],
            purpose=plan["purpose"],
            policy_sha256=ab_plan["policy_sha256"],
            runtime_policy_sha256=ab_plan["runtime_policy_sha256"],
            teacher_sha256=ab_plan["teacher_sha256"],
            renderer_sha256=ab_plan["renderer_sha256"],
            beta_sha256=ab_plan["beta_sha256"],
            config_sha256=plan["config_sha256"],
        )
        if telemetry:
            scalar_dir = Path(rung["run_dir"]) / "telemetry" / "scalars"
            for step in range(2):
                publish_immutable_json(
                    scalar_dir / f"step_{step:09d}.json",
                    self._telemetry(step, environments),
                )
        if evidence:
            publish_immutable_json(rung["evidence_path"], self._evidence(environments))

    def test_prepare_is_exact_five_rung_empty_plan_and_idempotent(self) -> None:
        self.assertEqual(tuple(map(int, self.report["rungs"])), RUNGS)
        self.assertTrue(
            all(item["state"] == "empty" for item in self.report["rungs"].values())
        )
        self.assertEqual(self.report["prerequisite"]["status"], "PASS")
        repeated = prepare_runtime_scale(
            self.scale_root,
            run_id="runtime-scale-test",
            runtime_ab_result=self.ab_result,
        )
        self.assertEqual(repeated["plan_sha256"], self.report["plan_sha256"])

    def test_environment_is_shadow_only_rank_specific_and_exactly_authorized(self) -> None:
        for environments in RUNGS:
            environment = runtime_scale_environment(self.plan, environments)
            self.assertEqual(environment["ROUND_DAGGER_VALIDATION_RANKS"], "1")
            self.assertEqual(
                environment["ROUND_DAGGER_VALIDATION_ENVS_PER_RANK"], str(environments)
            )
            self.assertEqual(environment["ROUND_DAGGER_VALIDATION_STEPS"], "2")
            self.assertEqual(environment["ROUND_DAGGER_VALIDATION_PURPOSE"], "train")
            self.assertEqual(environment[RUNTIME_ADAPTER_MODE_ENV], "shadow")
            self.assertEqual(
                environment[RUNTIME_ADAPTER_SHADOW_ACK_ENV], "metadata-only-v1"
            )
            self.assertEqual(
                environment[RUNTIME_ADAPTER_VALIDATION_ACK_ENV],
                RUNTIME_SCALE_VALIDATION_ACK,
            )
            self.assertEqual(
                environment[RUNTIME_SCALE_EVIDENCE_ENV],
                environment["ROUND_DAGGER_RUNTIME_ADAPTER_EVIDENCE"],
            )
            self.assertIn("/ranks/rank_0000/", environment[RUNTIME_SCALE_EVIDENCE_ENV])
            authorized = authorize_runtime_scale(
                self.plan,
                self.report["plan_sha256"],
                expected_rung_contract_sha256=environment[
                    RUNTIME_SCALE_RUNG_CONTRACT_SHA256_ENV
                ],
                environment=environment,
            )
            self.assertEqual(authorized["environments_per_rank"], environments)
            self.assertEqual(authorized["expected_transitions"], environments * 2)
            self._complete(environments)
            sealed = seal_runtime_scale(self.plan, environments)
            self.assertEqual(sealed["scope"]["classification"], "bounded-shadow-execution-capacity-only")

    def test_authorization_rejects_plan_rung_and_environment_tampering(self) -> None:
        self._complete(2)
        seal_runtime_scale(self.plan, 2)
        environment = runtime_scale_environment(self.plan, 4)
        with self.assertRaisesRegex(RuntimeScaleError, "plan SHA-256"):
            authorize_runtime_scale(
                self.plan, digest("wrong-plan"), environment=environment
            )
        with self.assertRaisesRegex(RuntimeScaleError, "rung SHA-256"):
            authorize_runtime_scale(
                self.plan,
                self.report["plan_sha256"],
                expected_rung_contract_sha256=digest("wrong-rung"),
                environment=environment,
            )
        tampered = dict(environment)
        tampered[RUNTIME_ADAPTER_INPUT_PROFILE_ENV] = "live-but-unbound"
        with self.assertRaisesRegex(RuntimeScaleError, "environment differs"):
            authorize_runtime_scale(
                self.plan, self.report["plan_sha256"], environment=tampered
            )

    def test_process_authorization_accepts_only_actor_shell_empty_directories(self) -> None:
        environment = runtime_scale_environment(self.plan, 2)
        _, rung, _ = self._documents(2)
        run_dir = Path(rung["run_dir"])
        for relative in (
            "ranks/rank_0000/hydra",
            "ranks/rank_0000/output",
        ):
            (run_dir / relative).mkdir(parents=True, exist_ok=True)
        with self.assertRaisesRegex(RuntimeScaleError, "partial"):
            authorize_runtime_scale(
                self.plan,
                self.report["plan_sha256"],
                environment=environment,
            )
        authorized = authorize_runtime_scale_process(
            self.plan,
            self.report["plan_sha256"],
            expected_rung_contract_sha256=environment[
                RUNTIME_SCALE_RUNG_CONTRACT_SHA256_ENV
            ],
            environment=environment,
        )
        self.assertEqual(authorized["status"], "authorized")
        (run_dir / "ranks/rank_0000/hydra/config.yaml").write_text(
            "already-started: true\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(RuntimeScaleError, "contains files"):
            authorize_runtime_scale_process(
                self.plan,
                self.report["plan_sha256"],
                environment=environment,
            )

    def test_complete_rung_requires_evidence_then_seals_and_verifies(self) -> None:
        self._complete(2, evidence=False)
        self.assertEqual(validate_runtime_scale(self.plan)["rungs"]["2"]["state"], "partial")
        with self.assertRaisesRegex(RuntimeScaleError, "partial"):
            seal_runtime_scale(self.plan, 2)
        _, rung, _ = self._documents(2)
        publish_immutable_json(rung["evidence_path"], self._evidence(2))
        self.assertEqual(validate_runtime_scale(self.plan)["rungs"]["2"]["state"], "complete")
        sealed = seal_runtime_scale(self.plan, 2)
        self.assertEqual(sealed["status"], "PASS")
        self.assertEqual(sealed["transition_count"], 4)
        self.assertEqual(
            sealed["evidence"]["telemetry"]["records"][0]["parallel_batch_size"],
            2,
        )
        self.assertEqual(
            sealed["evidence"]["policy_rgb_content"]["steps"][0][
                "unique_environment_count"
            ],
            2,
        )
        self.assertEqual(verify_runtime_scale(self.plan, 2)["status"], "PASS")
        with self.assertRaisesRegex(RuntimeScaleError, "sealed"):
            runtime_scale_environment(self.plan, 2)

    def test_sealed_rung_survives_fleet_cleanup_but_next_launch_fails_closed(self) -> None:
        self._complete(2)
        sealed = seal_runtime_scale(self.plan, 2)
        self.ab.descriptors[1].unlink()

        verified = verify_runtime_scale(self.plan, 2)
        self.assertEqual(verified["status"], "PASS")
        self.assertEqual(verified["transition_count"], 4)
        with self.assertRaisesRegex(
            RuntimeScaleError, "live renderer endpoint validation failed"
        ):
            runtime_scale_environment(self.plan, 4)

    def test_wrong_scale_audit_or_mutable_evidence_fails_closed(self) -> None:
        self._complete(4, evidence=False)
        _, rung, _ = self._documents(4)
        evidence = self._evidence(4)
        evidence["audit"]["num_envs"] = 2
        publish_immutable_json(rung["evidence_path"], evidence)
        with self.assertRaisesRegex(RuntimeScaleError, "audit"):
            validate_runtime_scale(self.plan)

    def test_rungs_require_every_predecessor_to_be_sealed(self) -> None:
        with self.assertRaisesRegex(RuntimeScaleError, "sealed predecessors: 2"):
            runtime_scale_environment(self.plan, 4)
        self._complete(4)
        with self.assertRaisesRegex(RuntimeScaleError, "sealed predecessors: 2"):
            seal_runtime_scale(self.plan, 4)

    def test_missing_telemetry_or_degenerate_rgb_fails_closed(self) -> None:
        self._complete(2, telemetry=False)
        with self.assertRaisesRegex(RuntimeScaleError, "scalar telemetry"):
            validate_runtime_scale(self.plan)

    def test_duplicate_or_zero_policy_rgb_fails_closed(self) -> None:
        self._complete(2, degenerate_rgb=True)
        with self.assertRaisesRegex(RuntimeScaleError, "degenerate range"):
            validate_runtime_scale(self.plan)

    def test_duplicate_policy_rgb_environments_fail_closed(self) -> None:
        self._complete(2, duplicate_rgb=True)
        with self.assertRaisesRegex(RuntimeScaleError, "not unique per environment"):
            validate_runtime_scale(self.plan)

    def test_cli_exposes_prepare_validate_environment_authorize_seal_verify(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "round_dagger.runtime_scale", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            capture_output=True,
            check=True,
        )
        for operation in ("prepare", "validate", "environment", "authorize", "seal", "verify"):
            self.assertIn(operation, completed.stdout)


if __name__ == "__main__":
    unittest.main()
