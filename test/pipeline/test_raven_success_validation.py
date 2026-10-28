from __future__ import annotations
from test.deployment.source_paths import PIPELINE_ROOT

import json
import unittest
from pathlib import Path

from pipeline.dagger.raven_success import (
    EpisodeGoalLatch,
    expand_validation_panel,
    success_summary,
)


ROOT = PIPELINE_ROOT


class RavenSuccessTests(unittest.TestCase):
    def test_goal_is_latched_until_episode_completion(self):
        latch = EpisodeGoalLatch()
        for value in (False, False, True, False, False):
            latch.update(value)
        self.assertTrue(latch.finish())
        self.assertFalse(latch.finish())

    def test_success_rate_denominator_is_all_completed_episodes(self):
        result = success_summary([True, False, True, False])
        self.assertEqual(result["successes"], 2)
        self.assertEqual(result["completed_episodes"], 4)
        self.assertEqual(result["goal_reached_rate"], 0.5)
        self.assertEqual(len(result["wilson_95_interval"]), 2)
        self.assertNotIn("mse", result)

    def test_fixed_panel_expands_to_256_unique_episodes(self):
        panel = json.loads((
            ROOT / "dagger/config/fixed_validation_panel.v1.json"
        ).read_text())
        episodes = expand_validation_panel(panel)
        self.assertEqual(len(episodes), 256)
        self.assertEqual(len({item["scenario_id"] for item in episodes}), 256)
        self.assertEqual({item["seed"] for item in episodes}, {1103, 2207, 3301, 4409})

if __name__ == "__main__":
    unittest.main()
