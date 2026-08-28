"""Read-only student-quality bridge metric tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from round_dagger import artifacts
from scripts.round_dagger.student_quality_wandb_bridge import summarize_cohort
from tests.test_round_dagger_quality import DIGESTS, IDENTITY, _arrays


class StudentQualityWandbBridgeTest(unittest.TestCase):
    def test_deep_verified_metrics_match_the_sealed_quality_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cohort = Path(temporary).resolve() / "cohort"
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
            seal = artifacts.seal_cohort(cohort)

            record = summarize_cohort(
                cohort,
                macro_round=IDENTITY["macro_round"],
                cohort_index=IDENTITY["cohort_index"],
            )
            self.assertEqual(record["seal_sha256"], seal["cohort_sha256"])
            self.assertEqual(record["event"], 203)
            self.assertAlmostEqual(
                record["metrics"]["rollout_quality/action_mse"], 55.0 / 6.0
            )
            self.assertAlmostEqual(
                record["metrics"]["rollout_quality/expert_intervention_rate"],
                0.5,
            )
            self.assertEqual(
                record["metrics"]["rollout_quality/transitions"], 6
            )


if __name__ == "__main__":
    unittest.main()
