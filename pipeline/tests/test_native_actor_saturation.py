"""Essential tests for the fixed eight-H200 BC collection topology."""

import json
import unittest
from pathlib import Path

from round_dagger.actor import _native_generation_shard_ids_from_environment
from round_dagger.native_actor_saturation import (
    BC_COLLECTION_RELATIVE_ROOT,
    CANDIDATES,
    GPU_COUNT,
    NOMINAL_TRANSITIONS,
    PRODUCTION_ADMITTED_PHYSICAL_GPU_INDICES,
    VIRTUAL_ACTOR_WORLD_SIZE,
    WAVE_STEPS,
    expected_shard_names,
    group_rotation,
    virtual_rank_cohorts,
    wave_shape,
)


ROOT = Path(__file__).resolve().parents[1]


class NativeActorTopologyTest(unittest.TestCase):
    def test_production_shape_is_one_wave_of_8x32(self) -> None:
        self.assertEqual(GPU_COUNT, 8)
        self.assertEqual(VIRTUAL_ACTOR_WORLD_SIZE, 8)
        self.assertEqual(PRODUCTION_ADMITTED_PHYSICAL_GPU_INDICES, tuple(range(8)))
        self.assertEqual(CANDIDATES, (32,))
        self.assertEqual(BC_COLLECTION_RELATIVE_ROOT, Path("BC/isaac/collection"))
        shape = wave_shape(32)
        self.assertEqual(
            shape,
            {
                "environments_per_gpu": 32,
                "total_environments": 256,
                "transitions_per_wave": NOMINAL_TRANSITIONS,
                "planned_waves": 1,
                "planned_transitions": NOMINAL_TRANSITIONS,
                "planned_overshoot": 0,
            },
        )
        self.assertEqual(NOMINAL_TRANSITIONS, 256 * WAVE_STEPS)

    def test_each_rank_is_bound_to_its_h200_in_one_concurrent_cohort(self) -> None:
        self.assertEqual(
            virtual_rank_cohorts(tuple(range(8)), 8),
            (tuple((index, index) for index in range(8)),),
        )

    def test_shard_inventory_covers_all_ranks_without_collision(self) -> None:
        names = expected_shard_names(
            wave_index=0, envs_per_gpu=32, environments_per_shard=4
        )
        self.assertEqual(len(names), 8 * 8 * 2)
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(
            names[0], "gen-000-bc__wave-000__gpu-00__shard-000000.npz"
        )
        self.assertEqual(
            names[-1], "gen-000-bc__wave-000__gpu-07__shard-000063.json"
        )

    def test_actor_export_derives_only_its_rank_shards(self) -> None:
        campaign = Path("/workspace/project/roborender/data")
        generation = campaign / BC_COLLECTION_RELATIVE_ROOT
        names = expected_shard_names(
            wave_index=0, envs_per_gpu=32, environments_per_shard=4
        )
        environment = {
            "ROUND_DAGGER_CAMPAIGN_ROOT": str(campaign),
            "ROUND_DAGGER_GENERATION_ROOT": str(generation),
            "ROUND_DAGGER_COHORT_DIR": str(
                generation / "conditioning/waves/wave-000"
            ),
            "ROUND_DAGGER_GENERATION_ID": "gen-000-bc",
            "ROUND_DAGGER_MACRO_ROUND": "0",
            "ROUND_DAGGER_ACTOR_ROLE": "bc_collect",
            "ROUND_DAGGER_BETA": "1.0",
            "ROUND_DAGGER_SATURATION_PHASE": "collect",
            "ROUND_DAGGER_SATURATION_WAVE_INDEX": "0",
            "ROUND_DAGGER_EXPECTED_SHARD_NAMES_JSON": json.dumps(names),
            "ROUND_DAGGER_OBSERVATION_SOURCE": "native",
            "ROUND_DAGGER_PURPOSE": "train",
            "ROUND_DAGGER_ACTOR_RANK": "3",
            "ROUND_DAGGER_ACTOR_WORLD_SIZE": "8",
        }
        shard_ids = _native_generation_shard_ids_from_environment(
            environment, rank=3, batch_size=32, environments_per_shard=4
        )
        self.assertEqual(len(shard_ids), 8)
        self.assertTrue(all("__gpu-03__" in item for item in shard_ids))

    def test_group_rotation_is_balanced_across_256_streams(self) -> None:
        groups = [f"motion-{index:02d}.npz" for index in range(23)]
        rotation = group_rotation(groups, 32, 0)
        self.assertEqual(sum(rotation["counts"].values()), 256)
        self.assertLessEqual(
            max(rotation["counts"].values()) - min(rotation["counts"].values()), 1
        )

    def test_launcher_has_only_canonical_paths_and_all_h200s(self) -> None:
        launcher = (ROOT / "scripts/round_dagger/run_native_h200_bc_collection.sh").read_text()
        self.assertIn("$CAMPAIGN/BC/isaac/collection", launcher)
        self.assertIn("--gpu-indices 0,1,2,3,4,5,6,7", launcher)
        self.assertIn("checkpoints/teachers/robot_policy/", launcher)
        self.assertIn("checkpoints/students/bootstrap/", launcher)
        self.assertNotIn("/users/", launcher)
        self.assertNotIn("/workspace/projects/", launcher)


if __name__ == "__main__":
    unittest.main()
