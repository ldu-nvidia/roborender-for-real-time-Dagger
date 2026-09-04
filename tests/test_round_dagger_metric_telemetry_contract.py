"""Cross-boundary contract for DAgger quality telemetry.

The evaluator and collection artifact schemas are immutable production
interfaces.  This test therefore does not add fields to either schema.  It
binds the existing local evidence fields to their existing W&B projections and
fails if any of the operator-facing model-quality signals disappears at either
boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import time
import unittest
from pathlib import Path

from round_dagger import artifacts, wandb_delivery
from round_dagger.actor_pair_terminal import SCALAR_SCHEMA
from round_dagger.evaluation import (
    DEFAULT_THRESHOLDS,
    _EPISODE_KEYS,
    _score,
)
from round_dagger.quality import publish_student_quality
from scripts.round_dagger import evaluation_wandb_bridge as eval_bridge
from scripts.scaling import wandb_run_watcher as watcher

try:
    from test_round_dagger_quality import (
        BETA,
        DIGESTS,
        IDENTITY,
        _arrays,
        _expected_lineage,
    )
except ImportError:  # pragma: no cover - package-style unittest invocation
    from tests.test_round_dagger_quality import (  # type: ignore[no-redef]
        BETA,
        DIGESTS,
        IDENTITY,
        _arrays,
        _expected_lineage,
    )


# Semantic signal -> raw per-episode evidence, immutable aggregate evidence,
# W&B projection.  Each paired aggregate deliberately includes parent,
# candidate, and candidate-minus-parent delta values.
PAIRED_EVALUATION_CONTRACT = {
    "task_success": (
        "success",
        (
            "parent_success_rate",
            "candidate_success_rate",
            "paired_success_rate_delta",
        ),
        (
            "eval/parent/success_rate",
            "eval/candidate/success_rate",
            "eval/delta/success_rate",
        ),
    ),
    "safety": (
        "safety_violation",
        (
            "parent_safety_violation_rate",
            "candidate_safety_violation_rate",
            "paired_safety_violation_rate_delta",
        ),
        (
            "eval/parent/safety_violation_rate",
            "eval/candidate/safety_violation_rate",
            "eval/delta/safety_violation_rate",
        ),
    ),
    "episode_return": (
        "episode_return",
        (
            "parent_episode_return_mean",
            "candidate_episode_return_mean",
            "paired_episode_return_mean_delta",
        ),
        (
            "eval/parent/episode_return_mean",
            "eval/candidate/episode_return_mean",
            "eval/delta/episode_return_mean",
        ),
    ),
    "episode_length": (
        "episode_length",
        (
            "parent_episode_length_mean",
            "candidate_episode_length_mean",
            "paired_episode_length_mean_delta",
        ),
        (
            "eval/parent/episode_length_mean",
            "eval/candidate/episode_length_mean",
            "eval/delta/episode_length_mean",
        ),
    ),
    "action_imitation": (
        "action_imitation_mse",
        (
            "parent_action_imitation_mse",
            "candidate_action_imitation_mse",
            "paired_action_imitation_mse_delta",
        ),
        (
            "eval/parent/action_imitation_mse",
            "eval/candidate/action_imitation_mse",
            "eval/delta/action_imitation_mse",
        ),
    ),
}

COLLECTION_INTERVENTION_METRIC = "student_quality/expert_intervention_rate"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _paired_outcomes() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    parent: list[dict[str, object]] = []
    candidate: list[dict[str, object]] = []
    for index in range(8):
        identity = {
            "seed": 17,
            "env_index": index,
            "scenario_id": _sha(f"scenario:{index}"),
            "first_condition_sha256": _sha(f"condition:{index}"),
            "first_generated_sha256": _sha(f"generated:{index}"),
        }
        parent.append(
            {
                **identity,
                "success": index < 4,
                "safety_violation": index < 2,
                "action_imitation_mse": 0.8,
                "episode_return": 10.0,
                "episode_length": 320,
            }
        )
        candidate.append(
            {
                **identity,
                "success": index < 6,
                "safety_violation": index < 1,
                "action_imitation_mse": 0.4,
                "episode_return": 12.0,
                "episode_length": 300,
            }
        )
    return parent, candidate


class DaggerMetricTelemetryContractTest(unittest.TestCase):
    def test_paired_eval_metrics_are_local_evidence_and_wandb_payload(self) -> None:
        parent, candidate = _paired_outcomes()
        metrics, decision = _score(parent, candidate, DEFAULT_THRESHOLDS)
        self.assertTrue(decision["passed"])

        # The raw immutable episode contract must retain the source observation
        # for every aggregate below; otherwise an aggregate could not be
        # independently audited from evaluator evidence.
        for raw_field, _, _ in PAIRED_EVALUATION_CONTRACT.values():
            self.assertIn(raw_field, _EPISODE_KEYS)

        with tempfile.TemporaryDirectory() as temporary:
            local_path = Path(temporary) / "paired_report.json"
            artifacts.publish_immutable_json(local_path, {"metrics": metrics})
            self.assertTrue(local_path.is_file())
            self.assertFalse(local_path.is_symlink())
            self.assertEqual(stat.S_IMODE(os.lstat(local_path).st_mode), 0o444)
            local_metrics = artifacts.load_strict_json(local_path)["metrics"]

            record = eval_bridge.EvaluationRecord(
                macro_round=1,
                result_path=local_path,
                result_sha256=artifacts.sha256_file(local_path),
                evidence_bundle_sha256=_sha("evaluation-bundle"),
                passed=True,
                metrics=local_metrics,
                parent_policy_version=0,
                candidate_policy_version=1,
                candidate_optimizer_steps=1200,
            )
            wandb_payload = eval_bridge.wandb_metrics(record)

        for semantic_name, (_, local_keys, wandb_keys) in (
            PAIRED_EVALUATION_CONTRACT.items()
        ):
            with self.subTest(signal=semantic_name):
                for local_key, wandb_key in zip(
                    local_keys, wandb_keys, strict=True
                ):
                    self.assertIn(local_key, local_metrics)
                    self.assertIn(wandb_key, wandb_payload)
                    self.assertEqual(
                        float(wandb_payload[wandb_key]),
                        float(local_metrics[local_key]),
                    )

    def test_collection_intervention_is_immutable_and_receipt_covered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            cohort = root / "cohort"
            run = root / "run"
            artifacts.create_cohort(
                cohort,
                expected_ranks=[0],
                expected_global_env_ids=[4, 5],
                expected_transitions=6,
                view_contract={
                    "dtype": "uint8",
                    "layout": "vertical",
                    "shape": [6, 4, 3],
                    "views": ["head_color", "hand_left", "hand_right"],
                },
                action_contract={"dtype": "float32", "shape": [13]},
                **IDENTITY,
                **DIGESTS,
            )
            artifacts.publish_rollout_shard(
                cohort,
                rank=0,
                rollout_id=0,
                part_index=0,
                arrays=_arrays(),
                **IDENTITY,
                **DIGESTS,
            )
            artifacts.seal_cohort(cohort)
            publication = publish_student_quality(
                cohort,
                run,
                expected_lineage=_expected_lineage(),
                expected_beta=BETA,
            )

            artifact_path = Path(publication["artifact_path"])
            scalar_path = Path(publication["scalar_path"])
            for path in (artifact_path, scalar_path):
                self.assertTrue(path.is_file())
                self.assertFalse(path.is_symlink())
                self.assertEqual(stat.S_IMODE(os.lstat(path).st_mode), 0o444)

            local_quality = artifacts.load_strict_json(artifact_path)["metrics"]
            step, wandb_projection = watcher.load_scalar_file(scalar_path)
            self.assertEqual(step, publication["wandb_step"])
            self.assertIn(COLLECTION_INTERVENTION_METRIC, local_quality)
            self.assertEqual(
                wandb_projection[COLLECTION_INTERVENTION_METRIC],
                local_quality[COLLECTION_INTERVENTION_METRIC],
            )

            # Add the terminal scalar required by the actor delivery gate, then
            # emulate a successfully drained sidecar cursor and receipt.  The
            # delivery verifier must count the quality scalar in that exact
            # acknowledged artifact set.
            terminal = (
                run
                / "telemetry"
                / "scalars"
                / "step_000002_actor_pair_terminal_contract.json"
            )
            terminal.write_text(
                json.dumps(
                    {
                        "schema": SCALAR_SCHEMA,
                        "step": 2,
                        "metrics": {"collection/terminal_pending": 1},
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            terminal.chmod(0o444)
            run_id = "metric_contract_test"
            run_id_path = run / "telemetry" / "WANDB_SIDECAR_RUN_ID"
            run_id_path.write_text(run_id + "\n", encoding="utf-8")

            errors: list[str] = []
            discovered = watcher.discover_scalars(
                run,
                min_age_s=0.0,
                now=time.time() + 1.0,
                errors=errors,
            ) + watcher.discover_artifacts(
                run,
                min_age_s=0.0,
                now=time.time() + 1.0,
                errors=errors,
            )
            self.assertEqual(errors, [])
            delivered = {
                item.relative_path: item.fingerprint for item in discovered
            }
            quality_relative = scalar_path.relative_to(run).as_posix()
            self.assertIn(quality_relative, delivered)

            state_path = run / "telemetry" / ".wandb_upload_state.json"
            watcher.save_state(state_path, delivered, run_id=run_id)
            receipt_path = run / "telemetry" / "WANDB_DELIVERY_RECEIPT.json"
            receipt_path.write_text(
                json.dumps(
                    {
                        "schema": watcher.DELIVERY_RECEIPT_SCHEMA,
                        "wandb_run_id": run_id,
                        "upload_state_sha256": hashlib.sha256(
                            state_path.read_bytes()
                        ).hexdigest(),
                        "outcome": "SUCCEEDED",
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            receipt_path.chmod(0o444)

            delivery = wandb_delivery.verify_delivery(run)
            self.assertIsNotNone(delivery)
            assert delivery is not None
            self.assertEqual(delivery["status"], "DELIVERED")
            self.assertEqual(
                delivery["artifact_count"],
                len(discovered),
            )


if __name__ == "__main__":
    unittest.main()
