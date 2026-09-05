from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

from round_dagger.conditioning_artifacts import (
    ARTIFACT_KIND,
    CONTROL_HEIGHT,
    CONTROL_WIDTH,
    VIEWS,
    derive_sparse_image_schedule,
    load_raw_conditioning_artifact,
    publish_conditioning_shard,
)
from round_dagger.artifacts import (
    canonical_json_sha256,
    publish_immutable_json,
    sha256_file,
)
from round_dagger.conditioning_split import (
    ConditioningSplitError,
    EXPECTED_BC_GROUP_COUNTS,
    EXPECTED_BC_NOMINAL_TRANSITIONS,
    EXPECTED_BC_PROTOCOL_SHA256,
    EXPECTED_BC_TOTAL_ENVIRONMENT_STREAMS,
    EXPECTED_BC_TRANSITIONS_PER_GLOBAL_ENV,
    EXPECTED_BC_TRANSITIONS_PER_WAVE,
    EXPECTED_BC_WAVE_COUNT,
    FROZEN_BC_SOURCE_RELATIVE_ROOT,
    SOURCE_RELATIVE_ROOT,
    _assign,
    _exact_wave_shard_names,
    _generation_lock,
    _inventory,
    _materialize,
    _sealed_source_manifests,
    _target_counts,
    _validate_exact_bc_saturation,
    _validate_exact_bc_prepublication,
    audit_pre_render_replay_root,
    split_collection,
    verify_split,
    verify_training_projection_source,
)

FINALIZER = (
    Path(__file__).resolve().parents[1]
    / "scripts/round_dagger/finalize_native_h200_bc_collection.py"
)
PROTOCOL = (
    Path(__file__).resolve().parents[1]
    / "configs/round_dagger/native_h200_experiment_protocol.json"
)


def load_finalizer_module():
    spec = importlib.util.spec_from_file_location("bc_collection_finalizer", FINALIZER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load BC collection finalizer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def arrays_for(global_ids: list[int]) -> dict[str, np.ndarray]:
    steps, envs = 11, len(global_ids)
    episode_steps = np.broadcast_to(
        np.arange(steps, dtype=np.int64)[:, None], (steps, envs)
    ).copy()
    episode_ids = np.broadcast_to(
        np.asarray([global_id * 10 for global_id in global_ids], dtype=np.int64)[
            None, :
        ],
        (steps, envs),
    ).copy()
    global_env_ids = np.broadcast_to(
        np.asarray(global_ids, dtype=np.int64)[None, :], (steps, envs)
    ).copy()
    schedule = derive_sparse_image_schedule(global_env_ids, episode_ids, episode_steps)
    expert = np.arange(steps * envs * 2, dtype=np.float32).reshape(steps, envs, 2)
    anchors = sorted((global_id, global_id * 10) for global_id in global_ids)
    result: dict[str, np.ndarray] = {
        "expert_actions": expert,
        "student_actions": expert + np.float32(1),
        "behavior_actions": expert.copy(),
        "expert_mask": np.ones((steps, envs), dtype=np.bool_),
        "dones": np.zeros((steps, envs), dtype=np.bool_),
        "global_env_ids": global_env_ids,
        "episode_ids": episode_ids,
        "episode_steps": episode_steps,
        "anchor_valid": episode_steps == 0,
        **schedule,
        "anchor_global_env_ids": np.asarray(
            [row[0] for row in anchors], dtype=np.int64
        ),
        "anchor_episode_ids": np.asarray([row[1] for row in anchors], dtype=np.int64),
        "obs__actor_obs": np.arange(steps * envs * 3, dtype=np.float32).reshape(
            steps, envs, 3
        ),
    }
    image_count = int(schedule["image_global_env_ids"].size)
    for index, view in enumerate(VIEWS):
        depth = np.empty((image_count, CONTROL_HEIGHT, CONTROL_WIDTH), dtype=np.uint8)
        robot = np.zeros(
            (image_count, CONTROL_HEIGHT, CONTROL_WIDTH, 3), dtype=np.uint8
        )
        for image_index in range(image_count):
            depth[image_index] = index + image_index + 1
            depth[image_index, :, CONTROL_WIDTH // 2 :] += 1
            robot[image_index, 40:100, 80:180] = index + image_index + 4
        anchor_rgb = np.empty(
            (len(anchors), CONTROL_HEIGHT, CONTROL_WIDTH, 3), dtype=np.uint8
        )
        for anchor_index in range(len(anchors)):
            anchor_rgb[anchor_index] = index + anchor_index + 7
            anchor_rgb[anchor_index, :, CONTROL_WIDTH // 2 :] += 1
        result[f"depth_ctrl__{view}"] = depth
        result[f"rgb_robot__{view}"] = robot
        result[f"rgb_anchor__{view}"] = anchor_rgb
    return result


def physical_gpu_inventory() -> list[dict[str, object]]:
    return [
        {
            "rank": rank,
            "physical_index": rank,
            "uuid": f"GPU-{rank:08x}-0000-0000-0000-{rank:012x}",
            "pci_bus_id": f"00000000:{rank + 1:02X}:00.0",
            "name": "NVIDIA H200",
            "compute_capability": "9.0",
            "total_memory_mib": 143_771,
            "mig_mode": "Disabled",
        }
        for rank in range(8)
    ]


class ConditioningSplitTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="bc_conditioning_split_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "experiment-data"
        self.source = (
            self.root / "BC/collection/conditioning/waves/wave-000/shards"
        )
        self.source.mkdir(parents=True)
        (self.root / "contracts").mkdir()
        self.protocol = Path(temporary.name) / "protocol.json"
        self.protocol.write_text(
            json.dumps(
                {
                    "group_split": {
                        "train_count": 23,
                        "validation_count": 5,
                        "test_count": 5,
                    }
                }
            ),
            encoding="utf-8",
        )

    def publish_source(self, name: str, global_ids: list[int]) -> None:
        rank = int(name.rsplit("-", 1)[1])
        group_ids = ["motion-near", "motion-far"]
        motion_ids = [
            "motion-near" if value % 2 else "motion-far" for value in global_ids
        ]
        publish_conditioning_shard(
            self.source / f"{name}.npz",
            arrays_for(global_ids),
            manifest_path=self.source / f"{name}.json",
            purpose="train",
            prompt="bimanual robot workspace",
            seed=42,
            reference_lag=256,
            render_metadata={
                "campaign_id": "dagger-redo-v001",
                "scenario_ids": [
                    "near" if value % 2 else "far" for value in global_ids
                ],
                "task_ids": ["bottle_to_tray"] * len(global_ids),
                "rank": rank,
                "part_index": 0,
                "split_manifest_sha256": "1" * 64,
                "split_sha256": "2" * 64,
                "group_ids": group_ids,
                "motion_ids": motion_ids,
            },
        )

    def seal_sources(self, seal_mutation=None) -> None:
        cohort = self.source.parent
        manifests = sorted(self.source.glob("*.json"))
        reports = [load_raw_conditioning_artifact(path)[0] for path in manifests]
        expected_names = sorted(
            [path.name for path in manifests]
            + [Path(report["payload_path"]).name for report in reports]
        )
        global_ids = sorted(
            global_id
            for path in manifests
            for global_id in set(
                int(value)
                for value in load_raw_conditioning_artifact(path)[1][
                    "global_env_ids"
                ].ravel()
            )
        )
        transitions = sum(report["transition_count"] for report in reports)
        episodes = sum(report["episode_count"] for report in reports)
        images = sum(report["image_count"] for report in reports)
        expected_ranks = [report["render"]["metadata"]["rank"] for report in reports]
        environments_per_shard = reports[0]["num_envs"]
        train_group_ids = list(reports[0]["render"]["metadata"]["group_ids"])
        rank_group_offsets = [
            train_group_ids.index(report["render"]["metadata"]["motion_ids"][0])
            for report in reports
        ]
        motion_counts = Counter(
            motion_id
            for report in reports
            for motion_id in report["render"]["metadata"]["motion_ids"]
        )
        group_rotation = {
            "start_group_offset": rank_group_offsets[0],
            "rank_group_offsets": rank_group_offsets,
            "counts": {
                group_id: motion_counts[group_id] for group_id in train_group_ids
            },
        }
        contract = {
            "schema": "fast_roborender.round_dagger.isaac_conditioning_cohort.v1",
            "artifact_kind": ARTIFACT_KIND,
            "status": "prepared",
            "learner_replay_ready": False,
            "campaign_id": "dagger-redo-v001",
            "generation_id": "gen-000-bc",
            "macro_round": 0,
            "cohort_index": 0,
            "purpose": "train",
            "expected_ranks": expected_ranks,
            "expected_global_env_ids": global_ids,
            "expected_transitions": transitions,
            "expected_shard_names": expected_names,
            "environments_per_shard": environments_per_shard,
            "split_manifest_sha256": "1" * 64,
            "split_sha256": "2" * 64,
            "train_group_ids": train_group_ids,
            "group_rotation": group_rotation,
        }
        contract_sha = publish_immutable_json(cohort / "contract.json", contract)
        shard_rows = []
        flat_inventory = []
        for path, report in zip(manifests, reports):
            stem = path.stem
            payload = Path(report["payload_path"])
            ids = sorted(
                set(
                    int(value)
                    for value in load_raw_conditioning_artifact(path)[1][
                        "global_env_ids"
                    ].ravel()
                )
            )
            manifest_descriptor = {
                "path": f"shards/{path.name}",
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            payload_descriptor = {
                "path": f"shards/{payload.name}",
                "sha256": sha256_file(payload),
                "size_bytes": payload.stat().st_size,
            }
            shard_rows.append(
                {
                    "shard_id": stem,
                    "rank": report["render"]["metadata"]["rank"],
                    "part_index": report["render"]["metadata"]["part_index"],
                    "global_env_ids": ids,
                    "motion_ids": report["render"]["metadata"]["motion_ids"],
                    "transition_count": report["transition_count"],
                    "episode_count": report["episode_count"],
                    "image_count": report["image_count"],
                    "manifest": manifest_descriptor,
                    "payload": payload_descriptor,
                }
            )
            flat_inventory.extend(
                [
                    {
                        "name": path.name,
                        "size_bytes": manifest_descriptor["size_bytes"],
                        "sha256": manifest_descriptor["sha256"],
                    },
                    {
                        "name": payload.name,
                        "size_bytes": payload_descriptor["size_bytes"],
                        "sha256": payload_descriptor["sha256"],
                    },
                ]
            )
        flat_inventory.sort(key=lambda row: row["name"])
        seal = {
            "schema": "fast_roborender.round_dagger.isaac_conditioning_cohort_seal.v1",
            "artifact_kind": ARTIFACT_KIND,
            "sealed": True,
            "learner_replay_ready": False,
            "campaign_id": "dagger-redo-v001",
            "generation_id": "gen-000-bc",
            "macro_round": 0,
            "cohort_index": 0,
            "purpose": "train",
            "contract_sha256": contract_sha,
            "split_manifest_sha256": contract["split_manifest_sha256"],
            "split_sha256": contract["split_sha256"],
            "train_group_ids": contract["train_group_ids"],
            "group_rotation": contract["group_rotation"],
            "transition_count": transitions,
            "episode_count": episodes,
            "image_count": images,
            "global_env_ids": global_ids,
            "shard_count": len(shard_rows),
            "shards": shard_rows,
        }
        if seal_mutation is not None:
            seal_mutation(seal)
        seal_path = cohort / "seal.json"
        seal_sha = publish_immutable_json(seal_path, seal)
        generation = self.root / "BC/collection"
        saturation = {
            "schema": "fast_roborender.round_dagger.native_actor_saturation.v1",
            "generation_id": "gen-000-bc",
            "status": "complete",
            "stop_reason": "nominal_target_met",
            "nominal_transitions": transitions,
            "completed_transitions": transitions,
            "completed_image_count": images,
            "completed_episode_count": episodes,
            "waves": [
                {
                    "wave_index": 0,
                    "cohort_dir": str(cohort),
                    "transition_count": transitions,
                    "group_rotation": group_rotation,
                    "shard_names": expected_names,
                    "image_count": images,
                    "episode_count": episodes,
                    "seal": {
                        "path": str(seal_path),
                        "sha256": seal_sha,
                        "shards": flat_inventory,
                        "image_count": images,
                        "episode_count": episodes,
                    },
                }
            ],
            "shard_inventory": flat_inventory,
            "train_conditioning_cohorts": [
                {
                    "path": str(cohort),
                    "seal_path": str(seal_path),
                    "seal_sha256": seal_sha,
                    "transition_count": transitions,
                    "image_count": images,
                    "purpose": "train",
                }
            ],
        }
        publish_immutable_json(generation / "saturation_manifest.json", saturation)

    def rewrite_saturation_cohort_paths(self, prefix: Path) -> None:
        path = self.root / "BC/collection/saturation_manifest.json"
        document = json.loads(path.read_text(encoding="utf-8"))
        for reference in document["train_conditioning_cohorts"]:
            cohort = prefix / Path(reference["path"]).name
            reference["path"] = str(cohort)
            reference["seal_path"] = str(cohort / "seal.json")
        for wave in document["waves"]:
            cohort = prefix / Path(wave["cohort_dir"]).name
            wave["cohort_dir"] = str(cohort)
            wave["seal"]["path"] = str(cohort / "seal.json")
        path.chmod(0o644)
        path.write_text(json.dumps(document), encoding="utf-8")
        path.chmod(0o444)

    def test_protocol_ratio_whole_environment_coverage_and_training_isolation(
        self,
    ) -> None:
        self.publish_source("rank-0000", [10, 11, 12])
        self.publish_source("rank-0001", [13, 14, 15])
        self.seal_sources()
        result = split_collection(
            self.root,
            protocol_path=self.protocol,
            expected_transitions_per_group=11,
        )

        self.assertEqual(result["weights"], {"train": 23, "validation": 5, "test": 5})
        self.assertEqual(
            {
                split: result["splits"][split]["group_count"]
                for split in result["splits"]
            },
            {"train": 4, "validation": 1, "test": 1},
        )
        all_ids = [
            value
            for split in ("train", "validation", "test")
            for value in result["splits"][split]["global_env_ids"]
        ]
        self.assertEqual(sorted(all_ids), [10, 11, 12, 13, 14, 15])
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(result["source_transition_count"], 66)
        self.assertEqual(result["source_episode_count"], 6)
        self.assertEqual(result["source_image_count"], 12)
        self.assertEqual(
            sum(split["image_count"] for split in result["splits"].values()), 12
        )
        plan = json.loads(
            (self.root / "BC/contracts/bc_collection_split.v1.json").read_text()
        )
        self.assertEqual(
            plan["weight_semantics"],
            "post_collection_global_environment_stream_quota_not_motion_holdout",
        )
        self.assertTrue(all("motion" in row["strata"] for row in plan["units"]))
        self.assertEqual(
            _target_counts(256, {"train": 23, "validation": 5, "test": 5}),
            {"train": 178, "validation": 39, "test": 39},
        )

        train_root = self.root / "BC/train"
        verify_split(
            train_root,
            consumer="training",
            expected_transitions_per_group=11,
        )
        verify_split(train_root, consumer="renderer")
        with patch(
            "round_dagger.conditioning_split." "EXPECTED_BC_TRANSITIONS_PER_GLOBAL_ENV",
            11,
        ):
            verify_training_projection_source(train_root, campaign_root=self.root)
        with self.assertRaisesRegex(ConditioningSplitError, "global plan"):
            verify_split(
                train_root,
                consumer="training",
                expected_transitions_per_group=512,
            )
        with self.assertRaisesRegex(ConditioningSplitError, "training cannot consume"):
            verify_split(
                self.root / "validation/BC",
                consumer="training",
            )
        with self.assertRaisesRegex(ConditioningSplitError, "renderer cannot consume"):
            verify_split(
                self.root / "validation/BC",
                consumer="renderer",
            )
        with self.assertRaisesRegex(ConditioningSplitError, "canonical sealed train"):
            verify_training_projection_source(
                self.source.parent,
                campaign_root=self.root,
            )

        # Every materialized group contains all 11 control transitions and its
        # two sparse image rows; no transition- or image-level split is possible.
        split_roots = {
            "train": self.root / "BC/train",
            "validation": self.root / "validation/BC",
            "test": self.root / "test/BC",
        }
        for split, split_root in split_roots.items():
            manifest = json.loads((split_root / "manifest.json").read_text())
            for row in manifest["artifacts"]:
                report, arrays = load_raw_conditioning_artifact(
                    split_root / row["manifest_path"]
                )
                self.assertEqual(report["num_steps"], 11)
                self.assertEqual(report["num_envs"], 1)
                self.assertEqual(report["episode_count"], 1)
                self.assertEqual(report["image_count"], 2)
                np.testing.assert_array_equal(
                    arrays["transition_image_indices"][:, 0],
                    np.asarray([0] * 10 + [1], dtype=np.int64),
                )
                np.testing.assert_array_equal(
                    arrays["image_episode_steps"],
                    np.asarray([0, 10], dtype=np.int64),
                )
                np.testing.assert_array_equal(
                    arrays["image_reference_source_steps"],
                    np.asarray([-1, 0], dtype=np.int64),
                )
                np.testing.assert_array_equal(
                    arrays["image_reference_indices"],
                    np.asarray([-1, 0], dtype=np.int64),
                )
                self.assertEqual(
                    set(arrays["image_global_env_ids"].tolist()),
                    set(arrays["global_env_ids"].ravel().tolist()),
                )
                self.assertEqual(
                    sorted(set(arrays["episode_ids"].ravel().tolist())),
                    row["episode_ids"],
                )

        # Create-only publication makes an identical retry safe and deterministic.
        retry = split_collection(
            self.root,
            protocol_path=self.protocol,
            expected_transitions_per_group=11,
        )
        self.assertEqual(retry["plan_sha256"], result["plan_sha256"])
        self.assertEqual(
            retry["splits"]["train"]["manifest_sha256"],
            result["splits"]["train"]["manifest_sha256"],
        )

    def test_duplicate_environment_across_source_shards_is_rejected(self) -> None:
        self.publish_source("rank-0000", [10, 11, 12])
        self.publish_source("rank-0001", [12, 13, 14])
        self.seal_sources()
        with self.assertRaisesRegex(ConditioningSplitError, "coverage differs"):
            split_collection(self.root, protocol_path=self.protocol)

    def test_fixed_seed_and_minimum_collection_are_enforced(self) -> None:
        self.publish_source("rank-0000", [10, 11])
        with self.assertRaisesRegex(ConditioningSplitError, "exactly 42"):
            split_collection(self.root, protocol_path=self.protocol, seed=41)
        self.seal_sources()
        with self.assertRaisesRegex(ConditioningSplitError, "at least three"):
            split_collection(self.root, protocol_path=self.protocol)

    def test_wrong_per_environment_length_fails_before_split_publication(self) -> None:
        self.publish_source("rank-0000", [10, 11, 12])
        self.seal_sources()
        with self.assertRaisesRegex(
            ConditioningSplitError, "transition count differs from exact target"
        ):
            split_collection(
                self.root,
                protocol_path=self.protocol,
                expected_transitions_per_group=512,
            )
        self.assertFalse((self.root / "BC/contracts/bc_collection_split.v1.json").exists())
        self.assertFalse((self.root / "train").exists())

    def test_exact_production_contract_is_checked_before_publication(self) -> None:
        units = [
            {
                "global_env_id": global_env_id,
                "episode_count": 1,
                "transition_count": EXPECTED_BC_TRANSITIONS_PER_GLOBAL_ENV,
                "image_count": 52,
            }
            for global_env_id in range(EXPECTED_BC_TOTAL_ENVIRONMENT_STREAMS)
        ]
        completion = {
            "completed_episode_count": EXPECTED_BC_TOTAL_ENVIRONMENT_STREAMS,
            "completed_transitions": EXPECTED_BC_NOMINAL_TRANSITIONS,
            "completed_image_count": 52 * EXPECTED_BC_TOTAL_ENVIRONMENT_STREAMS,
            "cohorts": [{}, {}],
            "physical_gpu_inventory": [{} for _ in range(8)],
        }
        self.assertEqual(
            _validate_exact_bc_prepublication(
                units=units,
                sources=[{}],
                targets=EXPECTED_BC_GROUP_COUNTS,
                weights={"train": 23, "validation": 5, "test": 5},
                completion=completion,
            )["transition_count"],
            EXPECTED_BC_NOMINAL_TRANSITIONS,
        )

        units[0] = {**units[0], "global_env_id": 256}
        with self.assertRaisesRegex(ConditioningSplitError, "exact 8xH200"):
            _validate_exact_bc_prepublication(
                units=units,
                sources=[{}],
                targets=EXPECTED_BC_GROUP_COUNTS,
                weights={"train": 23, "validation": 5, "test": 5},
                completion=completion,
            )

        self.assertEqual(sha256_file(PROTOCOL), EXPECTED_BC_PROTOCOL_SHA256)

    def test_exact_saturation_requires_two_rank0_waves_on_admitted_gpu0(self) -> None:
        inventory = physical_gpu_inventory()
        waves = []
        flattened = []
        references = []
        for wave_index in range(EXPECTED_BC_WAVE_COUNT):
            names = _exact_wave_shard_names(wave_index)
            file_inventory = [
                {"name": name, "size_bytes": 1, "sha256": "a" * 64}
                for name in sorted(names)
            ]
            flattened.extend(file_inventory)
            waves.append(
                {
                    "wave_index": wave_index,
                    "transition_count": EXPECTED_BC_TRANSITIONS_PER_WAVE,
                    "shard_names": names,
                    "seal": {"shards": file_inventory},
                }
            )
            references.append({"wave": wave_index})
        saturation = {
            "schema": "fast_roborender.round_dagger.native_actor_saturation.v1",
            "generation_id": "gen-000-bc",
            "status": "complete",
            "stop_reason": "nominal_target_met",
            "nominal_transitions": EXPECTED_BC_NOMINAL_TRANSITIONS,
            "completed_transitions": EXPECTED_BC_NOMINAL_TRANSITIONS,
            "actual_overshoot": 0,
            "physical_gpu_inventory": inventory,
            "physical_gpu_inventory_sha256": canonical_json_sha256(inventory),
            "virtual_actor_world_size": 1,
            "admitted_physical_gpu_indices": [0],
            "physical_execution_concurrency": 1,
            "virtual_rank_cohorts": [
                [{"virtual_rank": 0, "physical_gpu_index": 0}]
            ],
            "selection": {
                "environments_per_gpu": 128,
                "total_environments": 128,
                "transitions_per_wave": EXPECTED_BC_TRANSITIONS_PER_WAVE,
                "planned_waves": EXPECTED_BC_WAVE_COUNT,
                "planned_transitions": EXPECTED_BC_NOMINAL_TRANSITIONS,
                "planned_overshoot": 0,
            },
            "waves": waves,
            "train_conditioning_cohorts": references,
            "shard_inventory": flattened,
        }
        self.assertEqual(_validate_exact_bc_saturation(saturation), inventory)

        for field, wrong in (
            ("virtual_actor_world_size", 8),
            ("admitted_physical_gpu_indices", [1]),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(
                ConditioningSplitError, "one-actor"
            ):
                _validate_exact_bc_saturation({**saturation, field: wrong})

        wrong_wave = dict(waves[1])
        wrong_wave["transition_count"] -= 1
        with self.assertRaisesRegex(ConditioningSplitError, "wave 1"):
            _validate_exact_bc_saturation(
                {**saturation, "waves": [waves[0], wrong_wave]}
            )

    def test_cohort_seal_partition_binding_cannot_drift_from_contract(self) -> None:
        self.publish_source("rank-0000", [10, 11, 12])
        self.seal_sources(
            lambda seal: seal.__setitem__("split_sha256", "0" * 64)
        )
        with self.assertRaisesRegex(
            ConditioningSplitError, "contract differs from its seal"
        ):
            split_collection(self.root, protocol_path=self.protocol)

    def test_cohort_seal_part_index_is_checked_against_artifact_metadata(self) -> None:
        self.publish_source("rank-0000", [10, 11, 12])
        self.seal_sources(
            lambda seal: seal["shards"][0].__setitem__("part_index", 1)
        )
        with self.assertRaisesRegex(
            ConditioningSplitError, "sealed conditioning shard evidence differs"
        ):
            split_collection(self.root, protocol_path=self.protocol)

    def test_cohort_seal_motion_ids_are_checked_against_artifact_metadata(self) -> None:
        self.publish_source("rank-0000", [10, 11, 12])

        def mutate(seal):
            row = seal["shards"][0]
            row["motion_ids"][0] = "motion-near"

        self.seal_sources(mutate)
        with self.assertRaisesRegex(
            ConditioningSplitError, "sealed conditioning shard evidence differs"
        ):
            split_collection(self.root, protocol_path=self.protocol)

    def test_generation_lock_fails_closed_and_releases(self) -> None:
        generation = self.root / "BC/collection"
        generation.mkdir(parents=True, exist_ok=True)
        with _generation_lock(generation):
            with self.assertRaisesRegex(ConditioningSplitError, "controller is active"):
                with _generation_lock(generation):
                    self.fail("overlapping generation lock unexpectedly succeeded")
        with _generation_lock(generation):
            pass

    def test_deep_verifier_rejects_modes_seal_digests_and_extra_entries(self) -> None:
        self.publish_source("rank-0000", [10, 11, 12])
        self.publish_source("rank-0001", [13, 14, 15])
        self.seal_sources()
        split_collection(
            self.root,
            protocol_path=self.protocol,
            expected_transitions_per_group=11,
        )
        train_root = self.root / "BC/train"
        seal_path = train_root / "seal.json"

        seal_path.chmod(0o644)
        with self.assertRaisesRegex(ConditioningSplitError, "mode-0444"):
            verify_split(train_root, expected_transitions_per_group=11)
        seal_path.chmod(0o444)

        unexpected = train_root / "unexpected"
        unexpected.mkdir()
        with self.assertRaisesRegex(ConditioningSplitError, "inventory differs"):
            verify_split(train_root, expected_transitions_per_group=11)
        unexpected.rmdir()

        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        seal["artifact_payload_sha256"][0] = "0" * 64
        seal_path.chmod(0o644)
        seal_path.write_text(
            json.dumps(seal, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        seal_path.chmod(0o444)
        with self.assertRaisesRegex(ConditioningSplitError, "digest inventory differs"):
            verify_split(train_root, expected_transitions_per_group=11)

    def test_available_scenario_metadata_is_balanced_deterministically(self) -> None:
        units = [
            {
                "group_id": f"global-env-{index:012d}",
                "global_env_id": index,
                "strata": {"scenario": "near" if index % 2 else "far"},
            }
            for index in range(12)
        ]
        targets = {"train": 6, "validation": 3, "test": 3}
        first = _assign(units, targets, 42)
        second = _assign(units, targets, 42)
        self.assertEqual(first, second)
        for split in ("train", "validation", "test"):
            represented = {
                "near" if global_id % 2 else "far" for global_id in first[split]
            }
            self.assertEqual(represented, {"near", "far"})

    def test_missing_saturation_manifest_is_rejected_before_discovery(self) -> None:
        self.publish_source("rank-0000", [10, 11, 12])
        with self.assertRaisesRegex(
            ConditioningSplitError, "generation saturation manifest"
        ):
            split_collection(self.root, protocol_path=self.protocol)

    def test_frozen_cohort_paths_resolve_into_canonical_collection(self) -> None:
        self.publish_source("rank-0000", [10, 11, 12])
        self.seal_sources()
        frozen = Path("/retired/campaign") / FROZEN_BC_SOURCE_RELATIVE_ROOT
        self.rewrite_saturation_cohort_paths(frozen)

        with patch(
            "round_dagger.conditioning_split.CANONICAL_CAMPAIGN_ROOT", self.root
        ):
            manifests, completion, identity_root = _sealed_source_manifests(
                self.root,
                self.root / SOURCE_RELATIVE_ROOT,
            )

        self.assertEqual(identity_root, FROZEN_BC_SOURCE_RELATIVE_ROOT)
        self.assertEqual(len(manifests), 1)
        self.assertEqual(
            completion["cohorts"][0]["path"],
            (
                FROZEN_BC_SOURCE_RELATIVE_ROOT
                / "wave-000"
            ).as_posix(),
        )

    def test_frozen_cohort_path_requires_exact_generation_suffix(self) -> None:
        self.publish_source("rank-0000", [10, 11, 12])
        self.seal_sources()
        self.rewrite_saturation_cohort_paths(
            Path("/retired/campaign/not-a-generation/conditioning/waves")
        )

        with (
            patch(
                "round_dagger.conditioning_split.CANONICAL_CAMPAIGN_ROOT",
                self.root,
            ),
            self.assertRaisesRegex(
                ConditioningSplitError, "escapes conditioning waves"
            ),
        ):
            _sealed_source_manifests(
                self.root,
                self.root / SOURCE_RELATIVE_ROOT,
            )

    def test_frozen_source_identity_materializes_from_canonical_files(self) -> None:
        self.publish_source("rank-0000", [10, 11, 12])
        self.seal_sources()
        frozen = Path("/retired/campaign") / FROZEN_BC_SOURCE_RELATIVE_ROOT
        self.rewrite_saturation_cohort_paths(frozen)
        source_root = self.root / SOURCE_RELATIVE_ROOT

        with patch(
            "round_dagger.conditioning_split.CANONICAL_CAMPAIGN_ROOT", self.root
        ):
            manifests, _, identity_root = _sealed_source_manifests(
                self.root,
                source_root,
            )
        units, _ = _inventory(source_root, identity_root, manifests)
        entries = _materialize(
            self.root,
            source_root,
            identity_root,
            units,
            {"train": [10], "validation": [11], "test": [12]},
            "a" * 64,
        )

        self.assertEqual(
            {split: len(rows) for split, rows in entries.items()},
            {"train": 1, "validation": 1, "test": 1},
        )
        self.assertTrue(
            (self.root / "BC/train/artifacts/global-env-000000000010.npz").is_file()
        )

    def test_unexpected_unsealed_shard_is_rejected(self) -> None:
        self.publish_source("rank-0000", [10, 11, 12])
        self.seal_sources()
        (self.source / "stray.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ConditioningSplitError, "shard inventory differs"):
            split_collection(self.root, protocol_path=self.protocol)

    def test_noncomplete_saturation_manifest_is_rejected(self) -> None:
        self.publish_source("rank-0000", [10, 11, 12])
        self.seal_sources()
        path = self.root / "BC/collection/saturation_manifest.json"
        document = json.loads(path.read_text())
        document["status"] = "budget_capped"
        path.chmod(0o644)
        path.write_text(json.dumps(document), encoding="utf-8")
        path.chmod(0o444)
        with self.assertRaisesRegex(ConditioningSplitError, "not complete"):
            split_collection(self.root, protocol_path=self.protocol)

    def test_referenced_cohort_without_seal_is_rejected(self) -> None:
        self.publish_source("rank-0000", [10, 11, 12])
        self.seal_sources()
        seal = self.source.parent / "seal.json"
        seal.unlink()
        with self.assertRaisesRegex(ConditioningSplitError, "cohort seal"):
            split_collection(self.root, protocol_path=self.protocol)

    def test_saturation_cohort_seal_hash_mismatch_is_rejected(self) -> None:
        self.publish_source("rank-0000", [10, 11, 12])
        self.seal_sources()
        path = self.root / "BC/collection/saturation_manifest.json"
        document = json.loads(path.read_text())
        document["train_conditioning_cohorts"][0]["seal_sha256"] = "0" * 64
        path.chmod(0o644)
        path.write_text(json.dumps(document), encoding="utf-8")
        path.chmod(0o444)
        with self.assertRaisesRegex(ConditioningSplitError, "seal SHA-256 differs"):
            split_collection(self.root, protocol_path=self.protocol)

    def test_pre_render_audit_requires_exact_empty_replay_layout(self) -> None:
        generation = self.root / "BC/collection"
        (generation / "replay/manifests").mkdir(parents=True)
        (generation / "replay/shards").mkdir()
        report = audit_pre_render_replay_root(generation)
        self.assertIs(report["rendering_started"], False)
        self.assertEqual(report["observed_entry_count"], 0)
        self.assertEqual(report["observed_inventory"], [])

        (generation / "replay/shards/rendered-000.npz").write_bytes(b"rendered")
        with self.assertRaisesRegex(ConditioningSplitError, "output exists"):
            audit_pre_render_replay_root(generation)

    def test_finalizer_is_idempotent_after_rendering_starts(self) -> None:
        finalizer = load_finalizer_module()
        campaign = self.root
        source_root = self.protocol.parent / "source"
        source_root.mkdir()
        generation = campaign / "BC/collection"
        (generation / "replay/manifests").mkdir(parents=True, exist_ok=True)
        (generation / "replay/shards").mkdir(exist_ok=True)
        provenance = {"schema": "historical", "sources": [], "protocol": {}}
        current_provenance = {
            "schema": "current",
            "sources": [],
            "protocol": {},
        }
        source_evidence = {
            "saturation_manifest_sha256": "a" * 64,
            "source_inventory_sha256": "b" * 64,
            "source_group_count": EXPECTED_BC_TOTAL_ENVIRONMENT_STREAMS,
            "episode_count": EXPECTED_BC_TOTAL_ENVIRONMENT_STREAMS,
            "transition_count": EXPECTED_BC_NOMINAL_TRANSITIONS,
            "image_count": 52 * EXPECTED_BC_TOTAL_ENVIRONMENT_STREAMS,
            "target_counts": EXPECTED_BC_GROUP_COUNTS,
            "weights": {"train": 23, "validation": 5, "test": 5},
            "postprocess_provenance": provenance,
        }
        current_source_evidence = {
            **source_evidence,
            "postprocess_provenance": current_provenance,
        }
        boundaries = {
            "train": range(0, 178),
            "validation": range(178, 217),
            "test": range(217, 256),
        }

        observed_expected_provenance = []

        def report_for(path, **kwargs):
            observed_expected_provenance.append(
                kwargs["expected_postprocess_provenance"]
            )
            split = {
                campaign / "BC/train": "train",
                campaign / "validation/BC": "validation",
                campaign / "test/BC": "test",
            }[Path(path)]
            ids = list(boundaries[split])
            return {
                "group_count": len(ids),
                "transition_count": len(ids) * EXPECTED_BC_TRANSITIONS_PER_GLOBAL_ENV,
                "episode_count": len(ids),
                "image_count": 52 * len(ids),
                "global_env_ids": ids,
                "manifest_sha256": f"{len(ids):064x}",
                "seal_sha256": f"{len(ids) + 1:064x}",
                "verified_plan_sha256": "c" * 64,
            }

        with (
            patch.object(finalizer, "CANONICAL_CAMPAIGN_ROOT", campaign),
            patch.object(finalizer, "ROOT", source_root),
            patch.object(
                finalizer,
                "build_postprocess_provenance",
                side_effect=[provenance, current_provenance],
            ),
            patch.object(
                finalizer,
                "verify_exact_bc_collection_source",
                side_effect=[source_evidence, current_source_evidence],
            ),
            patch.object(finalizer, "verify_split", side_effect=report_for),
            patch.object(
                finalizer,
                "audit_pre_render_replay_root",
                wraps=audit_pre_render_replay_root,
            ) as replay_audit,
        ):
            first = finalizer.finalize_collection(campaign, source_root)
            (generation / "replay/shards/rendered-000.npz").write_bytes(b"rendered")
            second = finalizer.finalize_collection(campaign, source_root)

        self.assertEqual(first, second)
        self.assertEqual(replay_audit.call_count, 1)
        marker = json.loads(
            (generation / "BC_COLLECTION_COMPLETE.json").read_text(encoding="utf-8")
        )
        self.assertIs(marker["rendering_started"], False)
        self.assertIs(marker["rendering_audit"]["rendering_started"], False)
        self.assertEqual(marker["postprocess_provenance"], provenance)
        self.assertEqual(
            observed_expected_provenance,
            [provenance] * (2 * len(("train", "validation", "test"))),
        )


if __name__ == "__main__":
    unittest.main()
