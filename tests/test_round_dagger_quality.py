"""Focused gates for sealed student-versus-expert quality telemetry."""

from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from round_dagger import artifacts
from round_dagger.quality import (
    QUALITY_SCHEMA,
    StudentQualityError,
    publish_student_quality,
    summarize_sealed_cohort,
)
from scripts.scaling.wandb_run_watcher import load_scalar_file


BETA = 0.5
DIGESTS = {
    "policy_sha256": "1" * 64,
    "runtime_policy_sha256": "2" * 64,
    "teacher_sha256": "3" * 64,
    "renderer_sha256": "4" * 64,
    "beta_sha256": artifacts.canonical_json_sha256({"beta": BETA}),
    "config_sha256": "6" * 64,
}
IDENTITY = {
    "campaign_id": "quality-test",
    "macro_round": 2,
    "cohort_index": 3,
    "purpose": "train",
}


def _expected_lineage() -> dict[str, object]:
    return {
        key: value
        for key, value in {**IDENTITY, **DIGESTS}.items()
        if key != "purpose"
    }


def _arrays() -> dict[str, np.ndarray]:
    steps, environments, actions = 3, 2, 13
    expert = np.ones((steps, environments, actions), dtype=np.float32)
    student = expert.copy()
    error = np.arange(steps * environments, dtype=np.float32).reshape(
        steps, environments
    )
    student[..., :11] += error[..., None]
    # The unlearned tail must have no influence on any quality metric.
    student[..., 11:] += np.float32(1000.0)
    mask = np.asarray(
        [[True, False], [True, False], [False, True]], dtype=np.bool_
    )
    behavior = np.where(mask[..., None], expert, student).astype(np.float32)
    return {
        "policy_rgb": np.zeros((steps, environments, 6, 4, 3), dtype=np.uint8),
        "expert_actions": expert,
        "student_actions": student,
        "behavior_actions": behavior,
        "expert_mask": mask,
        "dones": np.asarray(
            [[False, False], [True, False], [False, True]], dtype=np.bool_
        ),
        "observation_is_raytraced": np.zeros(
            (steps, environments), dtype=np.bool_
        ),
        "global_env_ids": np.asarray([4, 5], dtype=np.int64),
        "episode_ids": np.asarray([[0, 10], [0, 10], [1, 10]], dtype=np.int64),
        "episode_steps": np.asarray([[0, 0], [1, 1], [0, 2]], dtype=np.int64),
        "obs__actor_obs": np.zeros((steps, environments, 7), dtype=np.float32),
    }


class StudentQualityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.cohort = self.root / "cohort"
        artifacts.create_cohort(
            self.cohort,
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

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _publish_and_seal(self) -> None:
        artifacts.publish_rollout_shard(
            self.cohort,
            rank=0,
            rollout_id=0,
            part_index=0,
            arrays=_arrays(),
            **IDENTITY,
            **DIGESTS,
        )
        artifacts.seal_cohort(self.cohort)

    def test_metrics_use_only_first_eleven_actions_and_bind_counts(self) -> None:
        self._publish_and_seal()
        result = summarize_sealed_cohort(
            self.cohort,
            expected_lineage=_expected_lineage(),
            expected_beta=BETA,
        )
        self.assertEqual(result["schema"], QUALITY_SCHEMA)
        self.assertEqual(
            result["action_slice"], {"start_inclusive": 0, "stop_exclusive": 11}
        )
        self.assertEqual(
            result["counts"],
            {
                "transitions": 6,
                "episodes": 3,
                "dones": 2,
                "steps_per_environment": 3,
            },
        )
        metrics = result["metrics"]
        self.assertAlmostEqual(metrics["student_quality/action_mse"], 55.0 / 6.0)
        self.assertAlmostEqual(metrics["student_quality/action_mae"], 2.5)
        self.assertAlmostEqual(
            metrics["student_quality/action_rmse"], math.sqrt(55.0 / 6.0)
        )
        self.assertAlmostEqual(
            metrics["student_quality/action_cosine_similarity_mean"], 1.0
        )
        self.assertAlmostEqual(
            metrics["student_quality/action_l2_p50"], 2.5 * math.sqrt(11.0)
        )
        self.assertAlmostEqual(
            metrics["student_quality/action_l2_p90"], 4.5 * math.sqrt(11.0)
        )
        self.assertAlmostEqual(
            metrics["student_quality/action_l2_p99"], 4.95 * math.sqrt(11.0)
        )
        self.assertAlmostEqual(
            metrics["student_quality/behavior_expert_match_rate"], 0.5
        )
        self.assertAlmostEqual(
            metrics["student_quality/expert_intervention_rate"], 0.5
        )
        self.assertEqual(metrics["student_quality/scheduled_beta"], BETA)
        self.assertEqual(metrics["student_quality/expert_queries"], 6)
        self.assertEqual(metrics["student_quality/expert_interventions"], 3)
        self.assertEqual(metrics["student_quality/environments"], 2)
        self.assertEqual(metrics["student_quality/steps_per_environment"], 3)
        self.assertEqual(metrics["student_quality/episodes_with_done"], 2)
        self.assertAlmostEqual(
            metrics["student_quality/episode_done_fraction"], 2.0 / 3.0
        )
        self.assertEqual(metrics["student_quality/right_censored_episodes"], 1)
        self.assertAlmostEqual(
            metrics["student_quality/observed_episode_length_mean"], 2.0
        )
        self.assertAlmostEqual(
            metrics["student_quality/done_episode_length_mean"], 2.5
        )
        for action_index in (0, 10):
            prefix = f"student_quality/action_{action_index:02d}"
            self.assertAlmostEqual(metrics[f"{prefix}/mse"], 55.0 / 6.0)
            self.assertAlmostEqual(metrics[f"{prefix}/mae"], 2.5)
            self.assertAlmostEqual(metrics[f"{prefix}/mean_error"], 2.5)
            self.assertAlmostEqual(
                metrics[f"{prefix}/absolute_error_p90"], 4.5
            )
        self.assertEqual(metrics["student_quality/transitions"], 6)
        self.assertEqual(metrics["student_quality/episodes"], 3)
        self.assertEqual(metrics["student_quality/dones"], 2)
        self.assertAlmostEqual(metrics["student_quality/done_rate"], 1.0 / 3.0)
        self.assertNotIn("reward", json.dumps(result).lower())
        self.assertNotIn("success", json.dumps(result).lower())

    def test_publication_is_immutable_and_wandb_watcher_consumable(self) -> None:
        self._publish_and_seal()
        run_dir = self.root / "run"
        first = publish_student_quality(
            self.cohort,
            run_dir,
            expected_lineage=_expected_lineage(),
            expected_beta=BETA,
        )
        second = publish_student_quality(
            self.cohort,
            run_dir,
            expected_lineage=_expected_lineage(),
            expected_beta=BETA,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["wandb_step"], 2)
        step, metrics = load_scalar_file(Path(first["scalar_path"]))
        self.assertEqual(step, 2)
        self.assertEqual(metrics, first["metrics"])
        quality = artifacts.load_strict_json(first["artifact_path"])
        self.assertEqual(
            quality["lineage"]["cohort_sha256"],
            artifacts.sha256_file(self.cohort / "seal.json"),
        )

    def test_unsealed_or_wrong_lineage_fails_closed(self) -> None:
        with self.assertRaisesRegex(StudentQualityError, "sealed cohort verification"):
            summarize_sealed_cohort(
                self.cohort,
                expected_lineage=_expected_lineage(),
                expected_beta=BETA,
            )
        self._publish_and_seal()
        wrong = _expected_lineage()
        wrong["policy_sha256"] = "a" * 64
        with self.assertRaisesRegex(StudentQualityError, "lineage mismatch"):
            summarize_sealed_cohort(
                self.cohort, expected_lineage=wrong, expected_beta=BETA
            )

    def test_changed_seal_schema_fails_closed(self) -> None:
        self._publish_and_seal()
        seal_path = self.cohort / "seal.json"
        document = artifacts.load_strict_json(seal_path)
        document["schema"] = "invalid.schema"
        os.chmod(seal_path, 0o644)
        seal_path.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(StudentQualityError, "sealed cohort verification"):
            summarize_sealed_cohort(
                self.cohort,
                expected_lineage=_expected_lineage(),
                expected_beta=BETA,
            )

    def test_actor_pair_invokes_post_seal_quality_gate(self) -> None:
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts/round_dagger/run_actor_cohort_pair.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("python3 -m round_dagger.quality", script)
        self.assertGreaterEqual(script.count("publish_student_quality"), 4)
        self.assertLess(
            script.rfind("python3 -m round_dagger.actor_pair seal"),
            script.rfind("publish_student_quality"),
        )
        function_start = script.index("finalize_completed_cohort()")
        function_end = script.index("\n}\n", function_start)
        function = script[function_start:function_end]
        self.assertLess(
            function.index("python3 -m round_dagger.actor_pair seal"),
            function.index("publish_student_quality"),
        )
        owner_wait = script.index('OWNER_ACTOR_STATUS=$(tr -d')
        pre_drain = script.index("if finalize_completed_cohort; then", owner_wait)
        terminal_and_finish = script.index(
            "if ! publish_attempt_terminal; then", pre_drain
        )
        watcher_reaped = script.index('wait "$OWNER_PID"', terminal_and_finish)
        self.assertLess(pre_drain, terminal_and_finish)
        self.assertLess(terminal_and_finish, watcher_reaped)
        self.assertIn('command+=(--finish-marker "$WANDB_PAIR_FINISH")', script)


if __name__ == "__main__":
    unittest.main()
