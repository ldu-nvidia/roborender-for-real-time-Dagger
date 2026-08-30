"""Focused tests for the Lepton H200 distributed learner smoke."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch  # noqa: F401
except ImportError:  # pragma: no cover - lightweight control-plane interpreter
    torch = None

if torch is not None:
    from round_dagger import replay
    from round_dagger.learner import LearnerBundleConfig, load_replay_partition
    from scripts.round_dagger import lepton_h200_learner_smoke as smoke


@unittest.skipIf(torch is None, "Lepton learner smoke tests require PyTorch")
class LeptonH200LearnerSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_fixture_is_8_rank_compatible_and_holds_exactly_256_eval_cases(self):
        output = self.root / "fixture-smoke"
        output.mkdir()
        fixture = smoke.prepare_fixture(
            output,
            world_size=8,
            heldout_cases=256,
            max_steps=4,
            save_every=1,
            heldout_eval_interval=1,
        )
        training = replay.load_snapshot(
            fixture["training_snapshot"],
            expected_file_sha256=fixture["training_snapshot_file_sha256"],
        )
        heldout = replay.load_heldout_snapshot(
            fixture["heldout_snapshot"],
            expected_file_sha256=fixture["heldout_snapshot_file_sha256"],
        )
        self.assertEqual(len(training["resolved_shards"]), 8)
        self.assertEqual(len(heldout["resolved_shards"]), 8)
        self.assertEqual(heldout["transition_count"], 256)
        config = LearnerBundleConfig.load(fixture["bundle"])
        rank_counts = []
        for rank in range(8):
            partition = load_replay_partition(
                fixture["training_snapshot"],
                config,
                rank=rank,
                world_size=8,
                expected_snapshot_file_sha256=fixture[
                    "training_snapshot_file_sha256"
                ],
            )
            rank_counts.append(partition.transition_count)
        self.assertEqual(rank_counts, [16] * 8)

    def test_invalid_wandb_identity_fails_before_artifact_creation(self):
        output = self.root / "must-not-exist"
        with self.assertRaisesRegex(smoke.SmokeError, "safe canonical"):
            smoke.run(
                output,
                world_size=2,
                device="cpu",
                heldout_cases=256,
                max_steps=2,
                split_step=1,
                wandb_project="smoke-project",
                wandb_run_id="not/a/canonical/id",
            )
        self.assertFalse(output.exists())

    def test_two_rank_cpu_smoke_proves_restart_and_quality_equivalence(self):
        output = self.root / "run"
        report = smoke.run(
            output,
            world_size=2,
            device="cpu",
            heldout_cases=256,
            max_steps=4,
            split_step=2,
            save_every=1,
            heldout_eval_interval=1,
        )
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["restart_semantic_equivalent"])
        self.assertTrue(report["restart_results_equivalent"])
        quality = report["student_quality"]
        self.assertLess(
            quality["heldout_action_mse_last"],
            quality["heldout_action_mse_first"],
        )
        durable = json.loads((output / "smoke_report.json").read_text())
        self.assertEqual(durable["report_sha256"], report["report_sha256"])
        heldout_rows = [
            json.loads(line)
            for line in (output / "resumed/heldout_telemetry.jsonl")
            .read_text()
            .splitlines()
        ]
        self.assertEqual(
            [row["optimizer_step"] for row in heldout_rows], [0, 1, 2, 3, 4]
        )
        self.assertTrue(
            all(row["heldout_transitions_global"] == 256 for row in heldout_rows)
        )


if __name__ == "__main__":
    unittest.main()
