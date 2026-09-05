from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from round_dagger.conditioning_artifacts import (
    ARTIFACT_KIND,
    CONFIG_SCHEMA,
    CONTROL_HZ,
    CONTROL_HEIGHT,
    CONTROL_WIDTH,
    IMAGE_STRIDE,
    REFERENCE_LAG_UNIT,
    REPLAY_ADMISSION,
    RENDER_HZ,
    SCHEMA,
    VIEWS,
    ConditioningArtifactError,
    h5_v4_control,
    h5_v4_recipe_document,
    load_raw_conditioning_artifact,
    publish_conditioning_shard,
    resize_uint8_batch,
    validate_conditioning_arrays,
    validate_conditioning_contract,
    verify_raw_conditioning_artifact,
)


def _arrays() -> dict[str, np.ndarray]:
    steps, envs, action_dim, state_dim = 19, 2, 4, 5
    expert = np.arange(steps * envs * action_dim, dtype=np.float32).reshape(
        steps, envs, action_dim
    )
    student = expert + np.float32(100)
    expert_mask = (
        np.indices((steps, envs), dtype=np.int64).sum(axis=0) % 2 == 0
    )
    episode_steps = np.column_stack(
        (
            np.asarray(list(range(7)) + list(range(12)), dtype=np.int64),
            np.asarray(list(range(10)) + list(range(9)), dtype=np.int64),
        )
    )
    episode_ids = np.column_stack(
        (
            np.asarray([100] * 7 + [101] * 12, dtype=np.int64),
            np.asarray([200] * 10 + [201] * 9, dtype=np.int64),
        )
    )
    dones = np.zeros((steps, envs), dtype=np.bool_)
    dones[6, 0] = True
    dones[9, 1] = True
    transition_image_indices = np.column_stack(
        (
            np.asarray([0] * 7 + [2] * 10 + [4] * 2, dtype=np.int64),
            np.asarray([1] * 10 + [3] * 9, dtype=np.int64),
        )
    )
    result: dict[str, np.ndarray] = {
        "expert_actions": expert,
        "student_actions": student,
        "behavior_actions": np.where(expert_mask[..., None], expert, student),
        "expert_mask": expert_mask,
        "dones": dones,
        "global_env_ids": np.broadcast_to(
            np.asarray([10, 20], dtype=np.int64)[None, :], (steps, envs)
        ).copy(),
        "episode_ids": episode_ids,
        "episode_steps": episode_steps,
        "anchor_valid": episode_steps == 0,
        "transition_image_indices": transition_image_indices,
        "image_global_env_ids": np.asarray([10, 20, 10, 20, 10], dtype=np.int64),
        "image_episode_ids": np.asarray([100, 200, 101, 201, 101], dtype=np.int64),
        "image_episode_steps": np.asarray([0, 0, 0, 0, 10], dtype=np.int64),
        "image_reference_source_steps": np.asarray(
            [-1, -1, -1, -1, 0], dtype=np.int64
        ),
        "image_reference_indices": np.asarray(
            [-1, -1, -1, -1, 2], dtype=np.int64
        ),
        "anchor_global_env_ids": np.asarray([10, 10, 20, 20], dtype=np.int64),
        "anchor_episode_ids": np.asarray([100, 101, 200, 201], dtype=np.int64),
        "obs__actor_obs": np.arange(steps * envs * state_dim, dtype=np.float32).reshape(
            steps, envs, state_dim
        ),
    }
    image_count = 5
    episode_count = 4
    for view_index, view in enumerate(VIEWS):
        depth = np.empty(
            (image_count, CONTROL_HEIGHT, CONTROL_WIDTH), dtype=np.uint8
        )
        robot = np.zeros(
            (image_count, CONTROL_HEIGHT, CONTROL_WIDTH, 3), dtype=np.uint8
        )
        for image_index in range(image_count):
            depth[image_index] = 1 + view_index + image_index
            depth[image_index, :, CONTROL_WIDTH // 2 :] += 1
            robot[
                image_index,
                40:100,
                80 + 10 * view_index : 180 + 10 * view_index,
            ] = 10 + view_index + image_index
        anchors = np.empty(
            (episode_count, CONTROL_HEIGHT, CONTROL_WIDTH, 3), dtype=np.uint8
        )
        for episode_index in range(episode_count):
            anchors[episode_index] = 20 + view_index + episode_index
            anchors[episode_index, :, CONTROL_WIDTH // 2 :] += 1
        result[f"depth_ctrl__{view}"] = depth
        result[f"rgb_robot__{view}"] = robot
        result[f"rgb_anchor__{view}"] = anchors
    return result


class ConditioningContractTest(unittest.TestCase):
    def test_exact_yaml_projection_is_accepted_and_drift_is_rejected(self) -> None:
        projection = {
            "views": list(VIEWS),
            "control_resolution": [CONTROL_HEIGHT, CONTROL_WIDTH],
            "depth_mode": "h5_v4",
            "depth_h5v4_per_view": h5_v4_recipe_document(),
            "raw_metric_depth_retained": False,
            "high_resolution_rasters_retained": False,
            "control_hz": 50,
            "render_hz": 5,
            "image_stride": 10,
            "reference_lag_unit": "control_steps",
        }
        canonical = validate_conditioning_contract(projection)
        self.assertEqual(canonical["resolution"], [240, 416])
        self.assertFalse(canonical["raw_metric_depth_retained"])
        self.assertFalse(canonical["high_resolution_rasters_retained"])
        self.assertEqual(canonical["control_hz"], CONTROL_HZ)
        self.assertEqual(canonical["render_hz"], RENDER_HZ)
        self.assertEqual(canonical["image_stride"], IMAGE_STRIDE)
        self.assertEqual(canonical["reference_lag_unit"], REFERENCE_LAG_UNIT)

        projection["views"] = ["hand_left", "head_color", "hand_right"]
        with self.assertRaisesRegex(ConditioningArtifactError, "views differs"):
            validate_conditioning_contract(projection)

    def test_full_native_yaml_contract_is_accepted_without_projection(self) -> None:
        config = {
            "schema": CONFIG_SCHEMA,
            "view_names": list(VIEWS),
            "sensor_resolution": [480, 832],
            "control_resolution": [240, 416],
            "depth_mode": "h5_v4",
            "depth_h5v4_per_view": h5_v4_recipe_document(),
            "prompt": "robot workspace",
            "seed": 42,
            "reference_policy": "lagged_generated_with_episode_anchor",
            "reference_lag": 256,
            "reference_lag_unit": "control_steps",
            "control_hz": 50,
            "render_hz": 5,
            "image_stride": 10,
        }
        canonical = validate_conditioning_contract(config)
        self.assertEqual(canonical["view_names"], list(VIEWS))
        self.assertEqual(canonical["sensor_resolution"], [480, 832])
        self.assertEqual(canonical["control_resolution"], [240, 416])

        config["sensor_resolution"] = [240, 416]
        with self.assertRaisesRegex(
            ConditioningArtifactError, "sensor_resolution differs"
        ):
            validate_conditioning_contract(config)

    def test_cadence_and_reference_units_are_frozen(self) -> None:
        for key, changed in (
            ("control_hz", 25),
            ("render_hz", 10),
            ("image_stride", 5),
            ("reference_lag_unit", "render_frames"),
        ):
            with self.subTest(key=key):
                contract = validate_conditioning_contract()
                contract[key] = changed
                with self.assertRaises(ConditioningArtifactError):
                    validate_conditioning_contract(contract)

    def test_h5_v4_is_pinned_to_byte_exact_production_fixture(self) -> None:
        random = np.random.RandomState(17)
        depth = random.uniform(0, 8, (4, 2, 7, 9)).astype(np.float32)
        depth[depth < 0.4] = 0
        depth[0, 0, :2, :3] = np.nan
        expected = {
            "head_color": "0ba96884830e6e901a42be13fb4933468bf9c8393eba9e035603f427d04914c5",
            "hand_left": "761b726aee3d95735dfe1ef95199dd90f34fb03d9dd75c03d9ae5c0ab42546a7",
            "hand_right": "761b726aee3d95735dfe1ef95199dd90f34fb03d9dd75c03d9ae5c0ab42546a7",
        }
        for view in VIEWS:
            control = h5_v4_control(depth, view)
            self.assertEqual(control.dtype, np.uint8)
            self.assertEqual(control.shape, depth.shape)
            self.assertEqual(
                hashlib.sha256(control.tobytes()).hexdigest(), expected[view]
            )

        invalid = np.full((2, 1, 3, 4), np.nan, dtype=np.float32)
        self.assertFalse(h5_v4_control(invalid, "head_color").any())

    def test_exact_2x_area_resize_handles_depth_and_rgb(self) -> None:
        depth_tile = np.asarray([[0, 1], [2, 3]], dtype=np.uint8)
        depth = np.tile(depth_tile, (CONTROL_HEIGHT, CONTROL_WIDTH))
        resized_depth = resize_uint8_batch(depth)
        self.assertEqual(resized_depth.shape, (CONTROL_HEIGHT, CONTROL_WIDTH))
        self.assertTrue(np.all(resized_depth == 2))

        rgb = np.stack((depth, depth + 4, depth + 8), axis=-1)
        resized_rgb = resize_uint8_batch(rgb)
        self.assertEqual(resized_rgb.shape, (CONTROL_HEIGHT, CONTROL_WIDTH, 3))
        self.assertTrue(np.all(resized_rgb == np.asarray([2, 6, 10], dtype=np.uint8)))
        self.assertTrue(np.array_equal(resize_uint8_batch(resized_rgb), resized_rgb))
        with self.assertRaisesRegex(ConditioningArtifactError, "480x832"):
            resize_uint8_batch(np.zeros((10, 10), dtype=np.uint8))


class ConditioningArraysTest(unittest.TestCase):
    def test_sparse_anchor_and_transition_contract_is_valid(self) -> None:
        validated = validate_conditioning_arrays(_arrays(), purpose="validation")
        self.assertEqual(validated["num_steps"], 19)
        self.assertEqual(validated["num_envs"], 2)
        self.assertEqual(validated["transition_count"], 38)
        self.assertEqual(validated["episode_count"], 4)
        self.assertEqual(validated["image_count"], 5)
        self.assertEqual(validated["action_dim"], 4)
        self.assertEqual(validated["state_dim"], 5)
        self.assertEqual(validated["purpose"], "validation")

    def test_learner_rgb_and_noncompact_arrays_are_fail_closed(self) -> None:
        for forbidden in ("policy_rgb", "depth_raw__head_color", "rgb_sim__head_color"):
            arrays = _arrays()
            arrays[forbidden] = np.zeros((1,), dtype=np.uint8)
            with self.assertRaisesRegex(ConditioningArtifactError, "unknown"):
                validate_conditioning_arrays(arrays)

    def test_sparse_anchor_table_must_be_complete_unique_and_ordered(self) -> None:
        mutations = {
            "missing": (
                np.asarray([10, 10, 20], dtype=np.int64),
                np.asarray([100, 101, 200], dtype=np.int64),
            ),
            "duplicate": (
                np.asarray([10, 10, 20, 20], dtype=np.int64),
                np.asarray([100, 100, 200, 201], dtype=np.int64),
            ),
            "out of order": (
                np.asarray([20, 10, 10, 20], dtype=np.int64),
                np.asarray([200, 100, 101, 201], dtype=np.int64),
            ),
        }
        for label, (global_ids, episode_ids) in mutations.items():
            with self.subTest(label=label):
                arrays = _arrays()
                arrays["anchor_global_env_ids"] = global_ids
                arrays["anchor_episode_ids"] = episode_ids
                with self.assertRaises(ConditioningArtifactError):
                    validate_conditioning_arrays(arrays)

    def test_anchor_valid_and_episode_continuity_are_exact(self) -> None:
        arrays = _arrays()
        arrays["anchor_valid"] = arrays["anchor_valid"].copy()
        arrays["anchor_valid"][7, 0] = False
        with self.assertRaisesRegex(ConditioningArtifactError, "anchor_valid"):
            validate_conditioning_arrays(arrays)

        arrays = _arrays()
        arrays["episode_steps"] = arrays["episode_steps"].copy()
        arrays["episode_steps"][1, 1] = 7
        arrays["anchor_valid"] = arrays["episode_steps"] == 0
        with self.assertRaisesRegex(ConditioningArtifactError, "contiguous"):
            validate_conditioning_arrays(arrays)

    def test_behavior_actions_and_exact_dtypes_are_enforced(self) -> None:
        arrays = _arrays()
        arrays["behavior_actions"] = arrays["behavior_actions"].copy()
        arrays["behavior_actions"][0, 0, 0] += 1
        with self.assertRaisesRegex(ConditioningArtifactError, "exactly select"):
            validate_conditioning_arrays(arrays)

        arrays = _arrays()
        arrays["global_env_ids"] = arrays["global_env_ids"].astype(np.int32)
        with self.assertRaisesRegex(ConditioningArtifactError, "dtype int64"):
            validate_conditioning_arrays(arrays)

    def test_sparse_rows_are_exact_time_major_episode_cadence(self) -> None:
        mutations = {
            "missing reset": ("image_episode_ids", 2, 100),
            "wrong cadence": ("image_episode_steps", 4, 9),
            "wrong time-major order": ("image_global_env_ids", 0, 20),
        }
        for label, (key, index, value) in mutations.items():
            with self.subTest(label=label):
                arrays = _arrays()
                arrays[key] = arrays[key].copy()
                arrays[key][index] = value
                with self.assertRaisesRegex(
                    ConditioningArtifactError, "exactly follow.*cadence"
                ):
                    validate_conditioning_arrays(arrays)

    def test_transition_image_map_rejects_stale_cross_episode_and_bad_range(self) -> None:
        mutations = {
            "stale sampled row": (17, 0, 2),
            "cross episode": (7, 0, 0),
            "cross environment": (1, 0, 1),
            "negative": (1, 0, -1),
            "out of range": (1, 0, 5),
        }
        for label, (time_index, env_index, value) in mutations.items():
            with self.subTest(label=label):
                arrays = _arrays()
                arrays["transition_image_indices"] = arrays[
                    "transition_image_indices"
                ].copy()
                arrays["transition_image_indices"][time_index, env_index] = value
                with self.assertRaisesRegex(
                    ConditioningArtifactError, "latest same-episode image row"
                ):
                    validate_conditioning_arrays(arrays)

    def test_reference_arrays_resolve_control_step_lag_through_hold_map(self) -> None:
        arrays = _arrays()
        arrays["image_reference_source_steps"] = arrays[
            "image_reference_source_steps"
        ].copy()
        arrays["image_reference_source_steps"][4] = 1
        with self.assertRaisesRegex(ConditioningArtifactError, "256-control-step"):
            validate_conditioning_arrays(arrays)

        arrays = _arrays()
        arrays["image_reference_indices"] = arrays["image_reference_indices"].copy()
        arrays["image_reference_indices"][4] = 0
        with self.assertRaisesRegex(ConditioningArtifactError, "hold mapping"):
            validate_conditioning_arrays(arrays)

    def test_sparse_identity_mapping_and_rasters_have_exact_dtypes_and_shapes(self) -> None:
        mutations = []
        arrays = _arrays()
        arrays["image_global_env_ids"] = arrays["image_global_env_ids"].astype(
            np.int32
        )
        mutations.append(("image ID dtype", arrays))
        arrays = _arrays()
        arrays["transition_image_indices"] = arrays[
            "transition_image_indices"
        ].astype(np.int32)
        mutations.append(("transition map dtype", arrays))
        arrays = _arrays()
        arrays["depth_ctrl__head_color"] = np.zeros(
            (19, 2, CONTROL_HEIGHT, CONTROL_WIDTH), dtype=np.uint8
        )
        mutations.append(("legacy dense depth shape", arrays))
        arrays = _arrays()
        arrays["rgb_robot__hand_left"] = arrays["rgb_robot__hand_left"].astype(
            np.float32
        )
        mutations.append(("robot dtype", arrays))
        for label, arrays in mutations:
            with self.subTest(label=label):
                with self.assertRaises(ConditioningArtifactError):
                    validate_conditioning_arrays(arrays)

    def test_depth_and_anchor_rows_must_have_spatial_content(self) -> None:
        for key in (
            "depth_ctrl__head_color",
            "rgb_anchor__hand_right",
        ):
            with self.subTest(key=key):
                arrays = _arrays()
                arrays[key] = arrays[key].copy()
                arrays[key][0] = 0
                with self.assertRaisesRegex(
                    ConditioningArtifactError, "empty or spatially constant"
                ):
                    validate_conditioning_arrays(arrays)

    def test_robot_mask_allows_one_empty_view_but_requires_cross_view_content(self) -> None:
        arrays = _arrays()
        arrays["rgb_robot__head_color"] = arrays["rgb_robot__head_color"].copy()
        arrays["rgb_robot__head_color"][0] = 0
        validate_conditioning_arrays(arrays)

        for view in VIEWS:
            key = f"rgb_robot__{view}"
            arrays[key] = arrays[key].copy()
            arrays[key][1] = 0
        with self.assertRaisesRegex(
            ConditioningArtifactError, "empty across all synchronized views"
        ):
            validate_conditioning_arrays(arrays)

    def test_nonempty_robot_mask_requires_a_black_background(self) -> None:
        arrays = _arrays()
        arrays["rgb_robot__hand_left"] = arrays["rgb_robot__hand_left"].copy()
        arrays["rgb_robot__hand_left"][0] = 7
        with self.assertRaisesRegex(
            ConditioningArtifactError, "required black background"
        ):
            validate_conditioning_arrays(arrays)


class ConditioningPublicationTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="conditioning_artifact_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def _publish(self, arrays: dict[str, np.ndarray] | None = None) -> dict:
        return publish_conditioning_shard(
            self.root / "gen-000_rank-0000_raw.npz",
            _arrays() if arrays is None else arrays,
            purpose="train",
            prompt="a bimanual robot manipulation workspace",
            seed=7000,
            reference_lag=256,
            render_metadata={
                "campaign_id": "dagger-redo-v001",
                "generation": 0,
                "renderer_checkpoint_sha256": "a" * 64,
            },
        )

    def test_publish_load_and_identical_retry_are_immutable(self) -> None:
        published = self._publish()
        manifest_path = Path(published["manifest_path"])
        payload_path = Path(published["payload_path"])
        self.assertEqual(stat_mode(manifest_path), 0o444)
        self.assertEqual(stat_mode(payload_path), 0o444)

        on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk["artifact_kind"], ARTIFACT_KIND)
        self.assertFalse(on_disk["learner_replay_ready"])
        self.assertEqual(on_disk["replay_admission"], REPLAY_ADMISSION)
        self.assertNotIn("prompt", on_disk["arrays"])
        self.assertNotIn("seed", on_disk["arrays"])
        self.assertEqual(on_disk["render"]["reference_lag"], 256)
        self.assertEqual(on_disk["schema"], SCHEMA)
        self.assertEqual(on_disk["image_count"], 5)
        self.assertEqual(on_disk["render"]["control_hz"], 50)
        self.assertEqual(on_disk["render"]["render_hz"], 5)
        self.assertEqual(on_disk["render"]["image_stride"], 10)
        self.assertEqual(on_disk["render"]["reference_lag_unit"], "control_steps")

        report, arrays = load_raw_conditioning_artifact(manifest_path)
        self.assertEqual(report["transition_count"], 38)
        self.assertEqual(report["image_count"], 5)
        self.assertNotIn("policy_rgb", arrays)
        self.assertEqual(set(arrays), set(_arrays()))
        self.assertEqual(
            self._publish()["manifest_sha256"], published["manifest_sha256"]
        )
        self.assertEqual(
            verify_raw_conditioning_artifact(manifest_path)["payload"]["sha256"],
            published["payload"]["sha256"],
        )

    def test_conflicting_retry_cannot_replace_payload_or_seal(self) -> None:
        original = self._publish()
        payload_path = Path(original["payload_path"])
        manifest_path = Path(original["manifest_path"])
        before_payload = payload_path.read_bytes()
        before_manifest = manifest_path.read_bytes()

        changed = _arrays()
        changed["depth_ctrl__head_color"] = changed["depth_ctrl__head_color"].copy()
        changed["depth_ctrl__head_color"][0, 0, 0] = 99
        with self.assertRaisesRegex(ConditioningArtifactError, "conflict"):
            self._publish(changed)
        self.assertEqual(payload_path.read_bytes(), before_payload)
        self.assertEqual(manifest_path.read_bytes(), before_manifest)

    def test_manifest_or_payload_must_remain_read_only(self) -> None:
        published = self._publish()
        manifest_path = Path(published["manifest_path"])
        manifest_path.chmod(0o644)
        with self.assertRaisesRegex(ConditioningArtifactError, "read-only"):
            load_raw_conditioning_artifact(manifest_path)

    def test_render_metadata_is_json_only_and_never_an_object_array(self) -> None:
        with self.assertRaisesRegex(ConditioningArtifactError, "canonical JSON"):
            publish_conditioning_shard(
                self.root / "bad-metadata.npz",
                _arrays(),
                purpose="train",
                prompt="robot workspace",
                seed=1,
                reference_lag=256,
                render_metadata={"bad": object()},
            )
        self.assertFalse((self.root / "bad-metadata.manifest.json").exists())


def stat_mode(path: Path) -> int:
    return os.stat(path, follow_symlinks=False).st_mode & 0o777


if __name__ == "__main__":
    unittest.main()
