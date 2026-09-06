from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from round_dagger import artifacts
from round_dagger.actor import (
    ActorExportError,
    _observation_provenance_from_environment,
)


VIEW_CONTRACT = {
    "dtype": "uint8",
    "layout": "vertical",
    "shape": [6, 4, 3],
    "views": ["head_color", "hand_left", "hand_right"],
}
ACTION_CONTRACT = {"dtype": "float32", "shape": [2]}
OBSERVATION_RUNTIME_SCHEMA = (
    "fast_roborender.round_dagger.native_h200_observation_runtime.v1"
)


def _arrays() -> dict[str, np.ndarray]:
    return {
        "policy_rgb": np.zeros((2, 1, 6, 4, 3), dtype=np.uint8),
        "expert_actions": np.zeros((2, 1, 2), dtype=np.float32),
        "student_actions": np.zeros((2, 1, 2), dtype=np.float32),
        "behavior_actions": np.zeros((2, 1, 2), dtype=np.float32),
        "expert_mask": np.ones((2, 1), dtype=np.bool_),
        "dones": np.zeros((2, 1), dtype=np.bool_),
        "global_env_ids": np.asarray([0], dtype=np.int64),
        "episode_ids": np.zeros((2, 1), dtype=np.int64),
        "episode_steps": np.asarray([[0], [1]], dtype=np.int64),
        "obs__actor_obs": np.zeros((2, 1, 113), dtype=np.float32),
    }


class NativeArtifactProvenanceTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="native_artifacts_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        physical_inventory = [
            {
                "rank": index,
                "physical_index": index,
                "uuid": f"GPU-00000000-0000-0000-0000-{index:012x}",
                "pci_bus_id": f"00000000:{index:02X}:00.0",
                "name": "NVIDIA H200",
                "compute_capability": "9.0",
                "total_memory_mib": 143771,
                "mig_mode": "Disabled",
            }
            for index in range(8)
        ]
        physical_inventory_sha256 = artifacts.canonical_json_sha256(
            physical_inventory
        )
        runtime = {
            "schema": OBSERVATION_RUNTIME_SCHEMA,
            "observation_source": "native",
            "isaac_runtime_sha256": "8" * 64,
            "actor_python_freeze_sha256": "9" * 64,
            "config_sha256": "6" * 64,
            "actor_source_inventory_sha256": "4" * 64,
            "gr00t_revision": "6" * 40,
            "gr00t_source_tree_sha256": "7" * 64,
            "gpu_name": "NVIDIA H200",
            "compute_capability": "9.0",
            "gpu_count": 8,
            "physical_gpu_inventory": physical_inventory,
            "physical_gpu_inventory_sha256": physical_inventory_sha256,
        }
        self.provenance = {
            "observation_source": "native",
            "observation_runtime_sha256": artifacts.canonical_json_sha256(runtime),
            **{
                key: runtime[key]
                for key in (
                    "isaac_runtime_sha256",
                    "actor_python_freeze_sha256",
                    "actor_source_inventory_sha256",
                    "gr00t_revision",
                    "gr00t_source_tree_sha256",
                    "gpu_name",
                    "compute_capability",
                    "gpu_count",
                    "physical_gpu_inventory",
                    "physical_gpu_inventory_sha256",
                )
            },
        }
        self.identity = {
            "campaign_id": "native-artifact-test",
            "macro_round": 0,
            "cohort_index": 0,
            "purpose": "train",
        }
        self.digests = {
            "policy_sha256": "1" * 64,
            "runtime_policy_sha256": "2" * 64,
            "teacher_sha256": "3" * 64,
            "beta_sha256": "5" * 64,
            "config_sha256": "6" * 64,
        }

    def _create(self, **provenance):
        return artifacts.create_cohort(
            self.root / "cohort",
            expected_ranks=[0],
            expected_global_env_ids=[0],
            expected_transitions=2,
            view_contract=VIEW_CONTRACT,
            action_contract=ACTION_CONTRACT,
            **self.identity,
            **self.digests,
            **provenance,
        )

    def _publish(self, **provenance):
        return artifacts.publish_rollout_shard(
            self.root / "cohort",
            rank=0,
            rollout_id=0,
            part_index=0,
            arrays=_arrays(),
            **self.identity,
            **self.digests,
            **provenance,
        )

    def test_native_identity_is_bound_to_contract_shard_and_seal(self) -> None:
        contract = self._create(**self.provenance)
        report = self._publish(**self.provenance)
        manifest = artifacts.load_strict_json(
            self.root / "cohort" / report["manifest_path"]
        )
        seal = artifacts.seal_cohort(self.root / "cohort")
        for field, value in self.provenance.items():
            self.assertEqual(contract[field], value)
            self.assertEqual(manifest[field], value)
            self.assertEqual(seal[field], value)

        verified = artifacts.verify_cohort(self.root / "cohort")
        self.assertEqual(verified["observation_source"], "native")

    def test_native_bundle_and_attestation_consistency_fail_closed(self) -> None:
        incomplete = dict(self.provenance)
        incomplete.pop("gr00t_source_tree_sha256")
        with self.assertRaisesRegex(artifacts.ArtifactError, "complete direct-runtime"):
            self._create(**incomplete)

        mismatch = dict(self.provenance)
        mismatch["actor_source_inventory_sha256"] = "c" * 64
        with self.assertRaisesRegex(
            artifacts.ArtifactError,
            "native observation runtime SHA-256 does not match its bundle",
        ):
            self._create(**mismatch)

    def test_shard_cannot_substitute_native_provenance(self) -> None:
        self._create(**self.provenance)
        mismatch = dict(self.provenance)
        mismatch["gr00t_revision"] = "d" * 40
        with self.assertRaisesRegex(artifacts.ArtifactError, "gr00t_revision"):
            self._publish(**mismatch)

    def test_actor_environment_boundary_requires_native_bundle(self) -> None:
        environment = {
            "ROUND_DAGGER_OBSERVATION_SOURCE": "native",
            "ROUND_DAGGER_OBSERVATION_RUNTIME_SHA256": self.provenance[
                "observation_runtime_sha256"
            ],
            "ROUND_DAGGER_ISAAC_RUNTIME_SHA256": self.provenance[
                "isaac_runtime_sha256"
            ],
            "ROUND_DAGGER_ACTOR_PYTHON_FREEZE_SHA256": self.provenance[
                "actor_python_freeze_sha256"
            ],
            "ROUND_DAGGER_ACTOR_SOURCE_INVENTORY_SHA256": self.provenance[
                "actor_source_inventory_sha256"
            ],
            "ROUND_DAGGER_GR00T_REVISION": self.provenance["gr00t_revision"],
            "ROUND_DAGGER_GR00T_SOURCE_TREE_SHA256": self.provenance[
                "gr00t_source_tree_sha256"
            ],
            "ROUND_DAGGER_GPU_NAME": self.provenance["gpu_name"],
            "ROUND_DAGGER_COMPUTE_CAPABILITY": self.provenance[
                "compute_capability"
            ],
            "ROUND_DAGGER_GPU_COUNT": str(self.provenance["gpu_count"]),
            "ROUND_DAGGER_PHYSICAL_GPU_INVENTORY_JSON": json.dumps(
                self.provenance["physical_gpu_inventory"]
            ),
            "ROUND_DAGGER_PHYSICAL_GPU_INVENTORY_SHA256": self.provenance[
                "physical_gpu_inventory_sha256"
            ],
        }
        self.assertEqual(
            _observation_provenance_from_environment(environment), self.provenance
        )
        incomplete = dict(environment)
        incomplete.pop("ROUND_DAGGER_GR00T_REVISION")
        with self.assertRaisesRegex(ActorExportError, "incomplete"):
            _observation_provenance_from_environment(incomplete)


if __name__ == "__main__":
    unittest.main()
