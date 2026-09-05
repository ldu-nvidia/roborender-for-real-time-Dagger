from __future__ import annotations

import unittest

from scripts.round_dagger import finalize_native_h200_bc_collection as finalizer


class FinalizeNativeH200BcCollectionTest(unittest.TestCase):
    def test_completion_records_host_inventory_and_actual_actor_topology(self):
        reports = {
            split: {
                "image_count": count,
                "episode_count": count,
                "verified_plan_sha256": "a" * 64,
                "seal_sha256": (str(index + 1) * 64)[:64],
                "manifest_sha256": (str(index + 4) * 64)[:64],
            }
            for index, (split, count) in enumerate(
                (("train", 178), ("validation", 39), ("test", 39))
            )
        }
        document = finalizer._completion_document(
            source_evidence={
                "image_count": 256,
                "episode_count": 256,
                "saturation_manifest_sha256": "b" * 64,
                "source_inventory_sha256": "c" * 64,
            },
            reports=reports,
            rendering_audit={"rendering_started": False},
            postprocess_provenance={"schema": "test"},
        )

        collection = document["collection"]
        self.assertEqual(
            collection["hardware"],
            {
                "host_resource": "8xNVIDIA H200 SM90",
                "host_physical_gpu_count": 8,
                "admitted_physical_gpu_indices": [0],
                "physical_execution_concurrency": 1,
            },
        )
        self.assertEqual(collection["actor_process_count"], 1)
        self.assertEqual(collection["gpus_per_actor_process"], 1)
        self.assertEqual(collection["virtual_actor_world_size"], 1)
        self.assertEqual(collection["environments_per_actor_process"], 128)
        self.assertEqual(collection["collection_wave_count"], 2)
        self.assertEqual(collection["total_environment_streams"], 256)
        self.assertEqual(collection["steps_per_environment"], 512)
        self.assertEqual(collection["transitions"], 131072)


if __name__ == "__main__":
    unittest.main()
