from __future__ import annotations

import unittest

import numpy as np

from round_dagger.conditioning_artifacts import (
    VIEWS,
    derive_sparse_image_schedule,
    validate_conditioning_contract,
)
from round_dagger.conditioning_capture import (
    ConditioningCaptureBuffer,
    ConditioningCaptureError,
)


def _views(frame: int, envs: int = 2) -> dict[str, dict[str, np.ndarray]]:
    height, width = 480, 832
    ramp = np.linspace(0.1, 3.0, width, dtype=np.float32)
    result: dict[str, dict[str, np.ndarray]] = {}
    for view_index, view in enumerate(VIEWS):
        depth = np.broadcast_to(
            ramp[None, None, :, None], (envs, height, width, 1)
        ).copy()
        depth += np.float32(view_index * 0.1 + frame * 0.01)

        robot = np.zeros((envs, height, width, 3), dtype=np.uint8)
        for env_index in range(envs):
            robot[
                env_index,
                100:220,
                200 + 20 * view_index : 400 + 20 * view_index,
            ] = 50 + view_index + frame + env_index

        rgb = np.empty((envs, height, width, 3), dtype=np.uint8)
        for env_index in range(envs):
            value = 20 + 10 * view_index + frame + env_index
            rgb[env_index] = value
            rgb[env_index, :, width // 2 :] = value + 1
        result[view] = {"depth": depth, "robot_rgb": robot, "rgb": rgb}
    return result


def _async_reset_identities() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """T=19/B=2: asynchronous resets restart each episode's image cadence."""

    global_ids = np.broadcast_to(
        np.asarray([10, 20], dtype=np.int64)[None, :], (19, 2)
    ).copy()
    episode_ids = np.column_stack(
        (
            np.asarray([100] * 7 + [101] * 12, dtype=np.int64),
            np.asarray([200] * 10 + [201] * 9, dtype=np.int64),
        )
    )
    episode_steps = np.column_stack(
        (
            np.asarray(list(range(7)) + list(range(12)), dtype=np.int64),
            np.asarray(list(range(10)) + list(range(9)), dtype=np.int64),
        )
    )
    return global_ids, episode_ids, episode_steps


def _capture_async_reset_fixture() -> tuple[
    ConditioningCaptureBuffer, np.ndarray, np.ndarray, np.ndarray
]:
    global_ids, episode_ids, episode_steps = _async_reset_identities()
    capture = ConditioningCaptureBuffer(
        validate_conditioning_contract(), num_steps=19, num_envs=2
    )
    for time_index in range(19):
        capture.append(
            _views(time_index),
            global_env_ids=global_ids[time_index],
            episode_ids=episode_ids[time_index],
            episode_steps=episode_steps[time_index],
        )
    capture.finalize()
    return capture, global_ids, episode_ids, episode_steps


class ConditioningCaptureTest(unittest.TestCase):
    def test_capture_compacts_controls_and_keeps_one_anchor_per_episode(self):
        capture = ConditioningCaptureBuffer(
            validate_conditioning_contract(), num_steps=2, num_envs=2
        )
        capture.append(
            _views(0),
            global_env_ids=np.asarray([10, 11], dtype=np.int64),
            episode_ids=np.asarray([0, 0], dtype=np.int64),
            episode_steps=np.asarray([0, 0], dtype=np.int64),
        )
        capture.append(
            _views(1),
            global_env_ids=np.asarray([10, 11], dtype=np.int64),
            episode_ids=np.asarray([0, 0], dtype=np.int64),
            episode_steps=np.asarray([1, 1], dtype=np.int64),
        )
        capture.finalize()
        self.assertEqual(capture.images_captured, 2)
        arrays = capture.arrays_for_slice(
            env_start=0,
            env_stop=1,
            global_env_ids=np.asarray([[10], [10]], dtype=np.int64),
            episode_ids=np.asarray([[0], [0]], dtype=np.int64),
            episode_steps=np.asarray([[0], [1]], dtype=np.int64),
        )
        self.assertNotIn("policy_rgb", arrays)
        np.testing.assert_array_equal(arrays["anchor_valid"], [[True], [False]])
        np.testing.assert_array_equal(arrays["anchor_global_env_ids"], [10])
        np.testing.assert_array_equal(arrays["anchor_episode_ids"], [0])
        np.testing.assert_array_equal(arrays["transition_image_indices"], [[0], [0]])
        np.testing.assert_array_equal(arrays["image_global_env_ids"], [10])
        np.testing.assert_array_equal(arrays["image_episode_ids"], [0])
        np.testing.assert_array_equal(arrays["image_episode_steps"], [0])
        np.testing.assert_array_equal(arrays["image_reference_source_steps"], [-1])
        np.testing.assert_array_equal(arrays["image_reference_indices"], [-1])
        for view in VIEWS:
            self.assertEqual(arrays[f"depth_ctrl__{view}"].shape, (1, 240, 416))
            self.assertEqual(
                arrays[f"rgb_robot__{view}"].shape, (1, 240, 416, 3)
            )
            self.assertEqual(arrays[f"rgb_anchor__{view}"].shape, (1, 240, 416, 3))
            self.assertEqual(arrays[f"depth_ctrl__{view}"].dtype, np.uint8)
            self.assertTrue(bool(arrays[f"rgb_robot__{view}"].any()))
        capture.release()

    def test_sparse_cadence_is_episode_local_and_async_resets_force_one_image(self):
        capture, global_ids, episode_ids, episode_steps = _capture_async_reset_fixture()
        self.addCleanup(capture.release)
        arrays = capture.arrays_for_slice(
            env_start=0,
            env_stop=2,
            global_env_ids=global_ids,
            episode_ids=episode_ids,
            episode_steps=episode_steps,
        )

        # t=7 is an asynchronous reset. Its next sample is episode step 10 at
        # absolute t=17; the other environment resets independently at t=10.
        observed_rows = list(
            zip(
                arrays["image_global_env_ids"].tolist(),
                arrays["image_episode_ids"].tolist(),
                arrays["image_episode_steps"].tolist(),
            )
        )
        self.assertEqual(
            observed_rows,
            [
                (10, 100, 0),
                (20, 200, 0),
                (10, 101, 0),
                (20, 201, 0),
                (10, 101, 10),
            ],
        )
        self.assertEqual(capture.images_captured, 5)
        expected_mapping = np.column_stack(
            (
                np.asarray([0] * 7 + [2] * 10 + [4] * 2, dtype=np.int64),
                np.asarray([1] * 10 + [3] * 9, dtype=np.int64),
            )
        )
        np.testing.assert_array_equal(
            arrays["transition_image_indices"], expected_mapping
        )
        np.testing.assert_array_equal(
            arrays["image_reference_source_steps"], [-1, -1, -1, -1, 0]
        )
        np.testing.assert_array_equal(
            arrays["image_reference_indices"], [-1, -1, -1, -1, 2]
        )
        for env_index in range(2):
            self.assertTrue(
                bool(np.all(np.diff(arrays["transition_image_indices"][:, env_index]) >= 0))
            )
            for time_index in range(19):
                image_index = int(arrays["transition_image_indices"][time_index, env_index])
                self.assertEqual(
                    int(arrays["image_global_env_ids"][image_index]),
                    int(global_ids[time_index, env_index]),
                )
                self.assertEqual(
                    int(arrays["image_episode_ids"][image_index]),
                    int(episode_ids[time_index, env_index]),
                )
                self.assertLessEqual(
                    int(arrays["image_episode_steps"][image_index]),
                    int(episode_steps[time_index, env_index]),
                )

        for view_index, view in enumerate(VIEWS):
            self.assertEqual(arrays[f"depth_ctrl__{view}"].shape, (5, 240, 416))
            self.assertEqual(
                arrays[f"rgb_robot__{view}"].shape, (5, 240, 416, 3)
            )
            self.assertEqual(arrays[f"depth_ctrl__{view}"].dtype, np.uint8)
            self.assertEqual(arrays[f"rgb_robot__{view}"].dtype, np.uint8)
            np.testing.assert_array_equal(
                arrays[f"rgb_robot__{view}"][:, 60, 150, 0],
                np.asarray([50, 51, 57, 61, 67], dtype=np.uint8) + view_index,
            )

    def test_sparse_slice_rebases_transition_and_reference_indices(self):
        capture, global_ids, episode_ids, episode_steps = _capture_async_reset_fixture()
        self.addCleanup(capture.release)
        arrays = capture.arrays_for_slice(
            env_start=0,
            env_stop=1,
            global_env_ids=global_ids[:, :1],
            episode_ids=episode_ids[:, :1],
            episode_steps=episode_steps[:, :1],
        )
        np.testing.assert_array_equal(arrays["image_global_env_ids"], [10, 10, 10])
        np.testing.assert_array_equal(arrays["image_episode_ids"], [100, 101, 101])
        np.testing.assert_array_equal(arrays["image_episode_steps"], [0, 0, 10])
        np.testing.assert_array_equal(
            arrays["transition_image_indices"][:, 0], [0] * 7 + [1] * 10 + [2] * 2
        )
        np.testing.assert_array_equal(arrays["image_reference_indices"], [-1, -1, 1])
        np.testing.assert_array_equal(
            arrays["anchor_global_env_ids"], np.asarray([10, 10], dtype=np.int64)
        )
        np.testing.assert_array_equal(
            arrays["anchor_episode_ids"], np.asarray([100, 101], dtype=np.int64)
        )
        self.assertEqual(int(arrays["transition_image_indices"].max()), 2)

    def test_reference_lag_uses_control_steps_then_resolves_the_held_image(self):
        steps = 278
        global_ids = np.full((steps, 1), 10, dtype=np.int64)
        episode_ids = np.concatenate(
            (
                np.full(7, 100, dtype=np.int64),
                np.full(steps - 7, 101, dtype=np.int64),
            )
        )[:, None]
        episode_steps = np.concatenate(
            (
                np.arange(7, dtype=np.int64),
                np.arange(steps - 7, dtype=np.int64),
            )
        )[:, None]

        schedule = derive_sparse_image_schedule(
            global_ids, episode_ids, episode_steps, reference_lag=256
        )
        target = int(
            np.flatnonzero(
                (schedule["image_episode_ids"] == 101)
                & (schedule["image_episode_steps"] == 270)
            )[0]
        )
        self.assertEqual(int(schedule["image_reference_source_steps"][target]), 14)
        referenced = int(schedule["image_reference_indices"][target])
        self.assertEqual(int(schedule["image_episode_ids"][referenced]), 101)
        self.assertEqual(int(schedule["image_episode_steps"][referenced]), 10)

    def test_sampled_source_content_must_be_valid_and_spatially_varying(self):
        cases = []
        empty_depth = _views(0)
        empty_depth["head_color"]["depth"][:] = 0
        cases.append(("depth", empty_depth))
        flat_anchor = _views(0)
        flat_anchor["head_color"]["rgb"][:] = 7
        cases.append(("anchor", flat_anchor))
        for label, views in cases:
            with self.subTest(label=label):
                capture = ConditioningCaptureBuffer(
                    validate_conditioning_contract(), num_steps=1, num_envs=2
                )
                self.addCleanup(capture.release)
                with self.assertRaisesRegex(
                    ConditioningCaptureError, "empty or spatially constant"
                ):
                    capture.append(
                        views,
                        global_env_ids=np.asarray([10, 20], dtype=np.int64),
                        episode_ids=np.asarray([0, 0], dtype=np.int64),
                        episode_steps=np.asarray([0, 0], dtype=np.int64),
                    )

    def test_robot_mask_allows_one_empty_view_but_requires_cross_view_content(self):
        views = _views(0)
        views["head_color"]["robot_rgb"][:] = 0
        capture = ConditioningCaptureBuffer(
            validate_conditioning_contract(), num_steps=1, num_envs=2
        )
        self.addCleanup(capture.release)
        capture.append(
            views,
            global_env_ids=np.asarray([10, 20], dtype=np.int64),
            episode_ids=np.asarray([0, 0], dtype=np.int64),
            episode_steps=np.asarray([0, 0], dtype=np.int64),
        )
        capture.finalize()
        arrays = capture.arrays_for_slice(
            env_start=0,
            env_stop=2,
            global_env_ids=np.asarray([[10, 20]], dtype=np.int64),
            episode_ids=np.asarray([[0, 0]], dtype=np.int64),
            episode_steps=np.asarray([[0, 0]], dtype=np.int64),
        )
        self.assertFalse(bool(arrays["rgb_robot__head_color"].any()))
        self.assertTrue(bool(arrays["rgb_robot__hand_left"].any()))

        all_empty = _views(0)
        for view in VIEWS:
            all_empty[view]["robot_rgb"][1] = 0
        rejected = ConditioningCaptureBuffer(
            validate_conditioning_contract(), num_steps=1, num_envs=2
        )
        self.addCleanup(rejected.release)
        with self.assertRaisesRegex(
            ConditioningCaptureError, "empty across all synchronized views"
        ):
            rejected.append(
                all_empty,
                global_env_ids=np.asarray([10, 20], dtype=np.int64),
                episode_ids=np.asarray([0, 0], dtype=np.int64),
                episode_steps=np.asarray([0, 0], dtype=np.int64),
            )

    def test_nonempty_robot_mask_requires_a_black_background(self):
        views = _views(0)
        views["head_color"]["robot_rgb"][0] = 7
        capture = ConditioningCaptureBuffer(
            validate_conditioning_contract(), num_steps=1, num_envs=2
        )
        self.addCleanup(capture.release)
        with self.assertRaisesRegex(
            ConditioningCaptureError, "required black background"
        ):
            capture.append(
                views,
                global_env_ids=np.asarray([10, 20], dtype=np.int64),
                episode_ids=np.asarray([0, 0], dtype=np.int64),
                episode_steps=np.asarray([0, 0], dtype=np.int64),
            )

    def test_constructor_has_no_dense_transition_raster_allocation(self):
        capture = ConditioningCaptureBuffer(
            validate_conditioning_contract(), num_steps=512, num_envs=32
        )
        self.addCleanup(capture.release)
        self.assertTrue(all(not chunks for chunks in capture._depth.values()))
        self.assertTrue(all(not chunks for chunks in capture._robot_rgb.values()))
        self.assertEqual(capture._transition_image_indices.nbytes, 512 * 32 * 8)

        nominal_images = ((512 - 1) // 10 + 1) * 32
        staging_bytes_per_image = 3 * 480 * 832 * 4 + 3 * 240 * 416 * 3
        artifact_bytes_per_image = 3 * 240 * 416 + 3 * 240 * 416 * 3
        self.assertEqual(nominal_images, 1664)
        self.assertEqual(nominal_images * staging_bytes_per_image, 9_469_624_320)
        self.assertEqual(nominal_images * artifact_bytes_per_image, 1_993_605_120)
        dense_staging = 512 * 32 * staging_bytes_per_image
        self.assertLess(nominal_images * staging_bytes_per_image, dense_staging / 9)

    def test_capture_fails_on_missing_view_and_duplicate_anchor(self):
        capture = ConditioningCaptureBuffer(
            validate_conditioning_contract(), num_steps=2, num_envs=2
        )
        malformed = _views(0)
        malformed.pop("hand_right")
        with self.assertRaisesRegex(ConditioningCaptureError, "views must be exactly"):
            capture.append(
                malformed,
                global_env_ids=np.asarray([0, 1], dtype=np.int64),
                episode_ids=np.asarray([0, 0], dtype=np.int64),
                episode_steps=np.asarray([0, 0], dtype=np.int64),
            )

        capture.append(
            _views(0),
            global_env_ids=np.asarray([0, 1], dtype=np.int64),
            episode_ids=np.asarray([0, 0], dtype=np.int64),
            episode_steps=np.asarray([0, 0], dtype=np.int64),
        )
        with self.assertRaisesRegex(ConditioningCaptureError, "duplicate episode"):
            capture.append(
                _views(1),
                global_env_ids=np.asarray([0, 1], dtype=np.int64),
                episode_ids=np.asarray([0, 0], dtype=np.int64),
                episode_steps=np.asarray([0, 0], dtype=np.int64),
            )
        capture.release()


if __name__ == "__main__":
    unittest.main()
