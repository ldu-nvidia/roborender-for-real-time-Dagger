"""Bounded host capture for synchronized Isaac-to-RoboRender inputs.

Robot control, state, and expert labels remain at 50 Hz. The expensive
three-view raster stream is sampled at 5 Hz (every ten episode steps), with
step zero forcing a fresh sample at every asynchronous reset. A sealed
transition-to-image map makes the 5-Hz observation hold explicit without
discarding any 50-Hz supervision.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


class ConditioningCaptureError(RuntimeError):
    """Isaac did not produce a complete, aligned RoboRender input stream."""


@dataclass(frozen=True)
class _Anchor:
    global_env_id: int
    episode_id: int
    rgb: dict[str, np.ndarray]


def _numpy(value: Any) -> np.ndarray:
    """Detach a tensor-like value without retaining CUDA storage."""

    candidate = value
    for method in ("detach", "cpu"):
        function = getattr(candidate, method, None)
        if callable(function):
            candidate = function()
    function = getattr(candidate, "numpy", None)
    if callable(function):
        candidate = function()
    return np.asarray(candidate)


def _resize_rgb_batch(value: Any, output_hw: tuple[int, int]) -> np.ndarray:
    """Area-resize ``[B,H,W,3]`` uint8 RGB with Raven's exact 2x rule."""

    from .conditioning_artifacts import resize_uint8_batch

    array = _numpy(value)
    if array.dtype != np.uint8 or array.ndim != 4 or array.shape[-1] < 3:
        raise ConditioningCaptureError("conditioning RGB must be uint8 [B,H,W,3]")
    return resize_uint8_batch(
        np.ascontiguousarray(array[..., :3]), output_hw=output_hw
    )


def _require_source_content(
    value: np.ndarray, label: str, *, black_background: bool
) -> None:
    """Reject malformed sampled rasters before they enter host staging.

    A textured-robot condition may legitimately be all black when that camera
    does not see the robot.  Nonempty robot conditions must still contain a
    black background.  The caller separately requires at least one nonempty
    robot condition across the three synchronized views for every image row.
    """

    flattened = value.reshape(value.shape[0], -1)
    minimum = flattened.min(axis=1)
    maximum = flattened.max(axis=1)
    if black_background:
        invalid = (maximum > 0) & (minimum != 0)
    else:
        invalid = maximum <= minimum
    if bool(np.any(invalid)):
        rows = np.flatnonzero(invalid)[:8].tolist()
        problem = (
            "has nonempty rows without the required black background"
            if black_background
            else "has empty or spatially constant sampled rows"
        )
        raise ConditioningCaptureError(
            f"{label} {problem}: {rows}"
        )


class ConditioningCaptureBuffer:
    """Sparse CPU staging for one rank-local, fixed-length 50-Hz rollout."""

    def __init__(
        self,
        contract: Mapping[str, Any],
        *,
        num_steps: int,
        num_envs: int,
    ) -> None:
        from .conditioning_artifacts import validate_conditioning_contract

        self.contract = validate_conditioning_contract(contract)
        if type(num_steps) is not int or num_steps <= 0:
            raise ConditioningCaptureError("capture num_steps must be positive")
        if type(num_envs) is not int or num_envs <= 0:
            raise ConditioningCaptureError("capture num_envs must be positive")
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.views = tuple(self.contract["view_names"])
        self.sensor_hw = tuple(self.contract["sensor_resolution"])
        self.control_hw = tuple(self.contract["control_resolution"])
        self.image_stride = int(self.contract["image_stride"])

        # Each list item contains only the environments sampled at one rollout
        # transition. This avoids allocating the former full-rank 512x32
        # 480x832 float32 tensor while retaining byte-exact metric depth until
        # H5-v4 is computed over each episode's sampled video rows.
        self._depth: dict[str, list[np.ndarray]] = {view: [] for view in self.views}
        self._robot_rgb: dict[str, list[np.ndarray]] = {
            view: [] for view in self.views
        }
        self._image_global_env_ids: list[np.ndarray] = []
        self._image_episode_ids: list[np.ndarray] = []
        self._image_episode_steps: list[np.ndarray] = []
        self._image_count = 0
        self._transition_image_indices = np.full(
            (num_steps, num_envs), -1, dtype=np.int64
        )
        self._latest_image_indices = np.full(num_envs, -1, dtype=np.int64)
        self._latest_image_episode_ids = np.full(num_envs, -1, dtype=np.int64)
        self._anchors: dict[tuple[int, int], _Anchor] = {}
        self._cursor = 0

    @property
    def steps_captured(self) -> int:
        return self._cursor

    @property
    def images_captured(self) -> int:
        return self._image_count

    def append(
        self,
        conditioning_views: Mapping[str, Mapping[str, Any]],
        *,
        global_env_ids: Sequence[int] | np.ndarray,
        episode_ids: Sequence[int] | np.ndarray,
        episode_steps: Sequence[int] | np.ndarray,
    ) -> None:
        """Capture raster rows for one pre-action state when cadence requires it."""

        if self._cursor >= self.num_steps:
            raise ConditioningCaptureError("conditioning capture exceeds rollout length")
        if not isinstance(conditioning_views, Mapping) or tuple(
            conditioning_views
        ) != self.views:
            raise ConditioningCaptureError(
                f"conditioning views must be exactly {self.views!r} in order"
            )
        ids = np.asarray(global_env_ids)
        episodes = np.asarray(episode_ids)
        steps = np.asarray(episode_steps)
        for name, value in (
            ("global_env_ids", ids),
            ("episode_ids", episodes),
            ("episode_steps", steps),
        ):
            if value.shape != (self.num_envs,) or value.dtype.kind not in "iu":
                raise ConditioningCaptureError(
                    f"{name} must be an integer vector of length {self.num_envs}"
                )
        if np.any(ids < 0) or np.any(episodes < 0) or np.any(steps < 0):
            raise ConditioningCaptureError("conditioning transition IDs must be non-negative")

        sample_mask = steps % self.image_stride == 0
        sample_indices = np.flatnonzero(sample_mask)

        held_indices = np.flatnonzero(~sample_mask)
        if held_indices.size and (
            np.any(self._latest_image_indices[held_indices] < 0)
            or np.any(
                self._latest_image_episode_ids[held_indices]
                != episodes[held_indices]
            )
        ):
            raise ConditioningCaptureError("a held image would cross an episode boundary")
        if sample_indices.size == 0:
            self._transition_image_indices[self._cursor] = self._latest_image_indices
            self._cursor += 1
            return

        image_indices = np.arange(
            self._image_count,
            self._image_count + sample_indices.size,
            dtype=np.int64,
        )
        anchor_indices = np.flatnonzero(steps == 0)
        anchor_rgb: dict[str, np.ndarray] = {}
        sampled_depth_by_view: dict[str, np.ndarray] = {}
        sampled_robot_by_view: dict[str, np.ndarray] = {}
        robot_visible = np.zeros(sample_indices.size, dtype=np.bool_)
        robot_nonzero_counts: dict[str, np.ndarray] = {}
        expected_depth = (self.num_envs, *self.sensor_hw, 1)
        expected_rgb = (self.num_envs, *self.sensor_hw, 3)
        for view in self.views:
            payload = conditioning_views[view]
            if not isinstance(payload, Mapping) or not {
                "depth", "robot_rgb", "rgb"
            }.issubset(payload):
                raise ConditioningCaptureError(
                    f"conditioning view {view!r} lacks depth/robot_rgb/rgb"
                )
            depth = _numpy(payload["depth"])
            if depth.shape != expected_depth or depth.dtype.kind != "f":
                raise ConditioningCaptureError(
                    f"conditioning depth[{view}] must be float {expected_depth}"
                )
            sampled_depth = np.ascontiguousarray(
                depth[sample_indices, ..., 0], dtype=np.float32
            )
            if not np.isfinite(sampled_depth).all() or np.any(sampled_depth < 0):
                raise ConditioningCaptureError(
                    f"conditioning depth[{view}] must be finite and non-negative"
                )
            _require_source_content(
                sampled_depth, f"conditioning depth[{view}]", black_background=False
            )
            sampled_depth_by_view[view] = sampled_depth

            robot = _numpy(payload["robot_rgb"])
            if robot.shape != expected_rgb or robot.dtype != np.uint8:
                raise ConditioningCaptureError(
                    f"conditioning robot_rgb[{view}] must be uint8 {expected_rgb}"
                )
            sampled_robot = _resize_rgb_batch(robot[sample_indices], self.control_hw)
            _require_source_content(
                sampled_robot,
                f"conditioning robot_rgb[{view}]",
                black_background=True,
            )
            sampled_robot_by_view[view] = sampled_robot
            counts = np.count_nonzero(
                sampled_robot.reshape(sampled_robot.shape[0], -1), axis=1
            )
            robot_nonzero_counts[view] = counts
            robot_visible |= counts > 0

            if anchor_indices.size:
                rgb = _numpy(payload["rgb"])
                if rgb.shape != expected_rgb or rgb.dtype != np.uint8:
                    raise ConditioningCaptureError(
                        f"conditioning rgb[{view}] must be uint8 {expected_rgb}"
                    )
                anchor_rgb[view] = _resize_rgb_batch(
                    rgb[anchor_indices], self.control_hw
                )
                _require_source_content(
                    anchor_rgb[view],
                    f"conditioning anchor rgb[{view}]",
                    black_background=False,
                )

        if bool(np.any(~robot_visible)):
            rows = np.flatnonzero(~robot_visible)[:8].tolist()
            details = [
                {
                    "sampled_row": row,
                    "global_env_id": int(ids[sample_indices[row]]),
                    "episode_id": int(episodes[sample_indices[row]]),
                    "episode_step": int(steps[sample_indices[row]]),
                    "nonzero_values_by_view": {
                        view: int(robot_nonzero_counts[view][row])
                        for view in self.views
                    },
                }
                for row in rows
            ]
            raise ConditioningCaptureError(
                "conditioning robot_rgb is empty across all synchronized views "
                f"for sampled rows: {details}"
            )

        anchor_keys = [
            (int(ids[env_index]), int(episodes[env_index]))
            for env_index in anchor_indices.tolist()
        ]
        if len(anchor_keys) != len(set(anchor_keys)):
            raise ConditioningCaptureError("duplicate episode bootstrap anchor in one state")
        duplicate = next((key for key in anchor_keys if key in self._anchors), None)
        if duplicate is not None:
            raise ConditioningCaptureError(
                f"duplicate episode bootstrap anchor for {duplicate!r}"
            )

        # Commit only after all three views and every new anchor have passed.
        # A rejected append therefore leaves the buffer safe for a corrected
        # retry at the same transition.
        self._latest_image_indices[sample_indices] = image_indices
        self._latest_image_episode_ids[sample_indices] = episodes[sample_indices]
        self._transition_image_indices[self._cursor] = self._latest_image_indices
        self._image_global_env_ids.append(
            np.ascontiguousarray(ids[sample_indices], dtype=np.int64)
        )
        self._image_episode_ids.append(
            np.ascontiguousarray(episodes[sample_indices], dtype=np.int64)
        )
        self._image_episode_steps.append(
            np.ascontiguousarray(steps[sample_indices], dtype=np.int64)
        )
        for view in self.views:
            self._depth[view].append(sampled_depth_by_view[view])
            self._robot_rgb[view].append(sampled_robot_by_view[view])
        for offset, (env_index, key) in enumerate(
            zip(anchor_indices.tolist(), anchor_keys)
        ):
            self._anchors[key] = _Anchor(
                global_env_id=key[0],
                episode_id=key[1],
                rgb={view: anchor_rgb[view][offset].copy() for view in self.views},
            )
        self._image_count += int(sample_indices.size)
        self._cursor += 1

    def finalize(self) -> None:
        """Fail closed before any part can be published."""

        if self._cursor != self.num_steps:
            raise ConditioningCaptureError(
                f"captured {self._cursor} conditioning steps, expected {self.num_steps}"
            )
        if self._image_count <= 0 or np.any(self._transition_image_indices < 0):
            raise ConditioningCaptureError("conditioning image mapping is incomplete")
        if any(
            len(self._depth[view]) != len(self._image_global_env_ids)
            for view in self.views
        ):
            raise ConditioningCaptureError("conditioning depth chunk inventory differs")
        if any(
            len(self._robot_rgb[view]) != len(self._image_global_env_ids)
            for view in self.views
        ):
            raise ConditioningCaptureError("conditioning robot chunk inventory differs")

    def _slice_sparse_rows(
        self, selected_global_ids: np.ndarray
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Return selected old row IDs and payloads in first-use order."""

        selected_set = set(int(value) for value in selected_global_ids.tolist())
        old_rows: list[np.ndarray] = []
        metadata: dict[str, list[np.ndarray]] = {
            "image_global_env_ids": [],
            "image_episode_ids": [],
            "image_episode_steps": [],
        }
        per_view: dict[str, list[np.ndarray]] = {
            f"depth_ctrl__{view}": [] for view in self.views
        }
        per_view.update({f"rgb_robot__{view}": [] for view in self.views})
        offset = 0
        for chunk_index, chunk_global_ids in enumerate(self._image_global_env_ids):
            mask = np.asarray(
                [int(value) in selected_set for value in chunk_global_ids],
                dtype=np.bool_,
            )
            count = int(chunk_global_ids.size)
            if bool(np.any(mask)):
                old_rows.append(np.arange(offset, offset + count, dtype=np.int64)[mask])
                metadata["image_global_env_ids"].append(chunk_global_ids[mask])
                metadata["image_episode_ids"].append(
                    self._image_episode_ids[chunk_index][mask]
                )
                metadata["image_episode_steps"].append(
                    self._image_episode_steps[chunk_index][mask]
                )
                for view in self.views:
                    per_view[f"depth_ctrl__{view}"].append(
                        self._depth[view][chunk_index][mask]
                    )
                    per_view[f"rgb_robot__{view}"].append(
                        self._robot_rgb[view][chunk_index][mask]
                    )
            offset += count
        if offset != self._image_count or not old_rows:
            raise ConditioningCaptureError("conditioning sparse image inventory is incomplete")

        selected_old_rows = np.concatenate(old_rows).astype(np.int64, copy=False)
        rows: dict[str, np.ndarray] = {
            key: np.ascontiguousarray(np.concatenate(parts))
            for key, parts in metadata.items()
        }
        rows.update(
            {
                key: np.ascontiguousarray(np.concatenate(parts))
                for key, parts in per_view.items()
            }
        )
        return selected_old_rows, rows

    def arrays_for_slice(
        self,
        *,
        env_start: int,
        env_stop: int,
        global_env_ids: np.ndarray,
        episode_ids: np.ndarray,
        episode_steps: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Compact one environment slice without materializing the full rank."""

        from .conditioning_artifacts import (
            ConditioningArtifactError,
            derive_sparse_image_schedule,
            h5_v4_control,
            resize_uint8_batch,
        )

        if not (0 <= env_start < env_stop <= self.num_envs):
            raise ConditioningCaptureError("conditioning environment slice is invalid")
        width = env_stop - env_start
        expected = (self.num_steps, width)
        identities = {
            "global_env_ids": np.asarray(global_env_ids),
            "episode_ids": np.asarray(episode_ids),
            "episode_steps": np.asarray(episode_steps),
        }
        for name, value in identities.items():
            if value.shape != expected or value.dtype.kind not in "iu":
                raise ConditioningCaptureError(
                    f"conditioning {name} slice must have shape {expected}"
                )
        if not np.all(
            identities["global_env_ids"] == identities["global_env_ids"][0:1]
        ):
            raise ConditioningCaptureError("global environment IDs changed within rollout")
        expected_slice_ids = identities["global_env_ids"][0]
        selected_old_rows, image_rows = self._slice_sparse_rows(expected_slice_ids)
        if set(image_rows["image_global_env_ids"].tolist()) != set(
            expected_slice_ids.tolist()
        ):
            raise ConditioningCaptureError("conditioning image rows cover wrong environments")

        remap = np.full(self._image_count, -1, dtype=np.int64)
        remap[selected_old_rows] = np.arange(selected_old_rows.size, dtype=np.int64)
        old_mapping = self._transition_image_indices[:, env_start:env_stop]
        transition_image_indices = remap[old_mapping]
        if np.any(transition_image_indices < 0):
            raise ConditioningCaptureError("conditioning image remap lost a transition")

        arrays: dict[str, np.ndarray] = {
            "anchor_valid": np.ascontiguousarray(
                identities["episode_steps"] == 0, dtype=np.bool_
            ),
            "transition_image_indices": np.ascontiguousarray(
                transition_image_indices, dtype=np.int64
            ),
            **{
                key: np.ascontiguousarray(value, dtype=np.int64)
                for key, value in image_rows.items()
                if key.startswith("image_")
            },
        }
        try:
            schedule = derive_sparse_image_schedule(
                identities["global_env_ids"],
                identities["episode_ids"],
                identities["episode_steps"],
                reference_lag=int(self.contract["reference_lag"]),
            )
        except ConditioningArtifactError as exc:
            raise ConditioningCaptureError(f"invalid conditioning cadence: {exc}") from exc
        for key in (
            "transition_image_indices",
            "image_global_env_ids",
            "image_episode_ids",
            "image_episode_steps",
        ):
            if not np.array_equal(arrays[key], schedule[key]):
                raise ConditioningCaptureError(
                    "captured sparse images differ from the canonical cadence"
                )
        arrays["image_reference_source_steps"] = schedule[
            "image_reference_source_steps"
        ]
        arrays["image_reference_indices"] = schedule["image_reference_indices"]

        image_keys = list(
            zip(
                arrays["image_global_env_ids"].tolist(),
                arrays["image_episode_ids"].tolist(),
            )
        )
        for view in self.views:
            metric_depth = image_rows[f"depth_ctrl__{view}"]
            output = np.empty(
                (selected_old_rows.size, *self.control_hw), dtype=np.uint8
            )
            for episode_key in dict.fromkeys(image_keys):
                row_indices = np.asarray(
                    [index for index, key in enumerate(image_keys) if key == episode_key],
                    dtype=np.int64,
                )
                control = h5_v4_control(
                    metric_depth[row_indices, None, ...], view, self.contract
                )[:, 0]
                output[row_indices] = resize_uint8_batch(
                    control, output_hw=self.control_hw
                )
            arrays[f"depth_ctrl__{view}"] = output
            arrays[f"rgb_robot__{view}"] = np.ascontiguousarray(
                image_rows[f"rgb_robot__{view}"]
            )

        needed = sorted(
            {
                (
                    int(identities["global_env_ids"][t, b]),
                    int(identities["episode_ids"][t, b]),
                )
                for t in range(self.num_steps)
                for b in range(width)
            }
        )
        missing = [key for key in needed if key not in self._anchors]
        if missing:
            raise ConditioningCaptureError(
                f"conditioning slice lacks episode anchors: {missing[:3]!r}"
            )
        arrays["anchor_global_env_ids"] = np.asarray(
            [key[0] for key in needed], dtype=np.int64
        )
        arrays["anchor_episode_ids"] = np.asarray(
            [key[1] for key in needed], dtype=np.int64
        )
        for view in self.views:
            arrays[f"rgb_anchor__{view}"] = np.stack(
                [self._anchors[key].rgb[view] for key in needed], axis=0
            ).astype(np.uint8, copy=False)
        return arrays

    def release(self) -> None:
        """Drop all staged image rows as soon as publication finishes."""

        for chunks in self._depth.values():
            chunks.clear()
        for chunks in self._robot_rgb.values():
            chunks.clear()
        self._depth.clear()
        self._robot_rgb.clear()
        self._image_global_env_ids.clear()
        self._image_episode_ids.clear()
        self._image_episode_steps.clear()
        self._anchors.clear()


__all__ = [
    "ConditioningCaptureBuffer",
    "ConditioningCaptureError",
]
