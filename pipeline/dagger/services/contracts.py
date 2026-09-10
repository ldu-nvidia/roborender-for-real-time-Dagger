"""Strict wire contracts shared by the H200 services and future RTX clients."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


IDENTIFIER = re.compile(r"^(?!\.{1,2}$)[A-Za-z0-9._-]{1,128}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
VIEWS = ("head_color", "hand_left", "hand_right")
ROLLOUT_STEPS = 81
CANONICAL_BETA = {1: 0.75, 2: 0.50, 3: 0.25, 4: 0.0}


class ContractError(ValueError):
    """A request does not satisfy the versioned DAgger service contract."""


def identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or IDENTIFIER.fullmatch(value) is None:
        raise ContractError(f"{field} must match {IDENTIFIER.pattern}")
    return value


def sha256_digest(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise ContractError(f"{field} must be a lowercase SHA-256 digest")
    return value


def exact_keys(payload: dict[str, Any], required: set[str], optional: set[str] = set()) -> None:
    if not isinstance(payload, dict):
        raise ContractError("JSON body must be an object")
    missing = required - payload.keys()
    unknown = payload.keys() - required - optional
    if missing:
        raise ContractError(f"missing fields: {sorted(missing)}")
    if unknown:
        raise ContractError(f"unknown fields: {sorted(unknown)}")


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ChunkReservation:
    chunk_id: str
    sha256: str
    size_bytes: int
    run_id: str
    round_id: int
    actor_id: str
    environment_id: int
    rollout_window_id: str
    first_step_id: int
    last_step_id: int
    checkpoint_sha256: str
    teacher_checkpoint_sha256: str
    sealed_window_sha256: str

    @classmethod
    def parse(cls, payload: dict[str, Any]) -> "ChunkReservation":
        fields = {
            "chunk_id", "sha256", "size_bytes", "run_id", "round_id",
            "actor_id", "environment_id", "rollout_window_id", "first_step_id",
            "last_step_id", "checkpoint_sha256", "teacher_checkpoint_sha256",
            "sealed_window_sha256",
        }
        exact_keys(payload, fields)
        round_id = payload["round_id"]
        if not isinstance(round_id, int) or round_id not in CANONICAL_BETA:
            raise ContractError("round_id must be 1..4")
        integer_fields = (
            "size_bytes", "environment_id", "first_step_id", "last_step_id"
        )
        for field in integer_fields:
            if not isinstance(payload[field], int) or payload[field] < 0:
                raise ContractError(f"{field} must be a non-negative integer")
        if payload["size_bytes"] == 0:
            raise ContractError("size_bytes must be positive")
        if payload["last_step_id"] - payload["first_step_id"] + 1 != ROLLOUT_STEPS:
            raise ContractError(f"a chunk must cover exactly {ROLLOUT_STEPS} steps")
        return cls(
            chunk_id=identifier(payload["chunk_id"], "chunk_id"),
            sha256=sha256_digest(payload["sha256"], "sha256"),
            size_bytes=payload["size_bytes"],
            run_id=identifier(payload["run_id"], "run_id"),
            round_id=round_id,
            actor_id=identifier(payload["actor_id"], "actor_id"),
            environment_id=payload["environment_id"],
            rollout_window_id=identifier(payload["rollout_window_id"], "rollout_window_id"),
            first_step_id=payload["first_step_id"],
            last_step_id=payload["last_step_id"],
            checkpoint_sha256=sha256_digest(payload["checkpoint_sha256"], "checkpoint_sha256"),
            teacher_checkpoint_sha256=sha256_digest(
                payload["teacher_checkpoint_sha256"], "teacher_checkpoint_sha256"
            ),
            sealed_window_sha256=sha256_digest(
                payload["sealed_window_sha256"], "sealed_window_sha256"
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _shape(array: np.ndarray, expected: tuple[int, ...], name: str) -> None:
    if array.shape != expected:
        raise ContractError(f"{name} has shape {array.shape}, expected {expected}")


def validate_dagger_chunk(path: Path) -> dict[str, Any]:
    """Validate one complete synchronized Raven DAgger rollout window."""
    try:
        with np.load(path, allow_pickle=False) as chunk:
            if "learner_chunk_format" in chunk.files:
                marker = str(chunk["learner_chunk_format"].item())
                if marker != "roborender.dagger.learner.v1":
                    raise ContractError(f"unknown learner chunk format: {marker}")
                required = {
                    "learner_chunk_format", "gt_actions", "obs_actor_obs",
                    *(f"rgb_gen__{view}" for view in VIEWS),
                }
                if set(chunk.files) != required:
                    raise ContractError(
                        "lean learner chunk arrays differ; "
                        f"missing={sorted(required - set(chunk.files))}, "
                        f"unknown={sorted(set(chunk.files) - required)}"
                    )
                _shape(chunk["gt_actions"], (ROLLOUT_STEPS, 26), "gt_actions")
                _shape(chunk["obs_actor_obs"], (ROLLOUT_STEPS, 113), "obs_actor_obs")
                for name in ("gt_actions", "obs_actor_obs"):
                    array = chunk[name]
                    if array.dtype != np.float32 or not np.isfinite(array).all():
                        raise ContractError(f"{name} must contain finite float32 values")
                for view in VIEWS:
                    name = f"rgb_gen__{view}"
                    rgb = chunk[name]
                    _shape(rgb, (ROLLOUT_STEPS, 108, 192, 3), name)
                    if rgb.dtype != np.float32 or not np.isfinite(rgb).all():
                        raise ContractError(f"{name} must contain finite float32 values")
                    if np.any(rgb < 0.0) or np.any(rgb > 1.0):
                        raise ContractError(f"{name} must be RGB in [0,1]")
                return {
                    "format": marker,
                    "steps": ROLLOUT_STEPS,
                    "views": list(VIEWS),
                    "teacher_fraction": None,
                    "beta_min": None,
                    "beta_max": None,
                }
            required = {
                "gt_actions", "student_actions", "behavior_actions",
                "behavior_is_teacher", "beta", "dones", "obs_actor_obs",
                "obs_teacher_obs", "homie_actions", "timeouts", "episode_ids",
                "episode_step_indices", "rollout_step_indices", "env_id_per_step",
                "seq_per_step", "observation_source",
            }
            required.update(f"rgb_gen__{view}" for view in VIEWS)
            required.update(f"depth_ctrl__{view}" for view in VIEWS)
            required.update(f"rgb_robot__{view}" for view in VIEWS)
            missing = required - set(chunk.files)
            if missing:
                raise ContractError(f"chunk is missing arrays: {sorted(missing)}")

            for view in VIEWS:
                _shape(chunk[f"rgb_gen__{view}"], (ROLLOUT_STEPS, 108, 192, 3), f"rgb_gen__{view}")
                _shape(chunk[f"depth_ctrl__{view}"], (ROLLOUT_STEPS, 240, 416), f"depth_ctrl__{view}")
                _shape(chunk[f"rgb_robot__{view}"], (ROLLOUT_STEPS, 240, 416, 3), f"rgb_robot__{view}")
            _shape(chunk["gt_actions"], (ROLLOUT_STEPS, 26), "gt_actions")
            _shape(chunk["student_actions"], (ROLLOUT_STEPS, 26), "student_actions")
            _shape(chunk["behavior_actions"], (ROLLOUT_STEPS, 26), "behavior_actions")
            _shape(chunk["behavior_is_teacher"], (ROLLOUT_STEPS,), "behavior_is_teacher")
            _shape(chunk["beta"], (ROLLOUT_STEPS,), "beta")
            _shape(chunk["dones"], (ROLLOUT_STEPS,), "dones")
            _shape(chunk["obs_actor_obs"], (ROLLOUT_STEPS, 113), "obs_actor_obs")
            _shape(chunk["obs_teacher_obs"], (ROLLOUT_STEPS, 226), "obs_teacher_obs")
            _shape(chunk["homie_actions"], (ROLLOUT_STEPS, 15), "homie_actions")
            _shape(chunk["timeouts"], (ROLLOUT_STEPS,), "timeouts")
            _shape(chunk["episode_ids"], (ROLLOUT_STEPS,), "episode_ids")
            _shape(chunk["episode_step_indices"], (ROLLOUT_STEPS,), "episode_step_indices")
            _shape(chunk["rollout_step_indices"], (ROLLOUT_STEPS,), "rollout_step_indices")
            _shape(chunk["env_id_per_step"], (ROLLOUT_STEPS,), "env_id_per_step")
            _shape(chunk["seq_per_step"], (ROLLOUT_STEPS,), "seq_per_step")
            _shape(chunk["observation_source"], (ROLLOUT_STEPS,), "observation_source")

            if chunk["behavior_is_teacher"].dtype != np.bool_:
                raise ContractError("behavior_is_teacher must be bool")
            if chunk["dones"].dtype != np.bool_ or chunk["timeouts"].dtype != np.bool_:
                raise ContractError("dones and timeouts must be bool")
            for name in ("gt_actions", "student_actions", "behavior_actions", "beta",
                         "obs_actor_obs", "obs_teacher_obs", "homie_actions"):
                array = chunk[name]
                if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
                    raise ContractError(f"{name} must contain finite floating-point values")
            if np.any((chunk["beta"] < 0.0) | (chunk["beta"] > 1.0)):
                raise ContractError("beta must be within [0,1]")
            selected = np.where(
                chunk["behavior_is_teacher"][:, None],
                chunk["gt_actions"],
                chunk["student_actions"],
            )
            if not np.array_equal(chunk["behavior_actions"], selected):
                raise ContractError(
                    "behavior_actions do not match the recorded teacher/student selector"
                )
            if not np.array_equal(chunk["gt_actions"][:, 11:], chunk["homie_actions"]):
                raise ContractError("teacher action tail differs from Homie action")
            if not np.array_equal(chunk["student_actions"][:, 11:], chunk["homie_actions"]):
                raise ContractError("student action tail differs from Homie action")
            if not np.array_equal(
                chunk["rollout_step_indices"], np.arange(ROLLOUT_STEPS, dtype=np.int64)
            ):
                raise ContractError("rollout_step_indices must be exactly 0..80")
            if np.any(chunk["episode_step_indices"] < 0):
                raise ContractError("episode_step_indices must be non-negative")
            sources = set(str(item) for item in chunk["observation_source"].tolist())
            if not sources <= {"raytraced", "generated_controls"}:
                raise ContractError("observation_source contains an unknown value")
            for view in VIEWS:
                rgb = chunk[f"rgb_gen__{view}"]
                if rgb.dtype != np.float32 or not np.isfinite(rgb).all():
                    raise ContractError(f"rgb_gen__{view} must be finite float32")
                if np.any(rgb < 0.0) or np.any(rgb > 1.0):
                    raise ContractError(f"rgb_gen__{view} must be RGB in [0,1]")
                if chunk[f"depth_ctrl__{view}"].dtype != np.uint8:
                    raise ContractError(f"depth_ctrl__{view} must be uint8")
                if chunk[f"rgb_robot__{view}"].dtype != np.uint8:
                    raise ContractError(f"rgb_robot__{view} must be uint8")
            return {
                "steps": ROLLOUT_STEPS,
                "views": list(VIEWS),
                "teacher_fraction": float(chunk["behavior_is_teacher"].mean()),
                "beta_min": float(chunk["beta"].min()),
                "beta_max": float(chunk["beta"].max()),
            }
    except (OSError, ValueError) as exc:
        if isinstance(exc, ContractError):
            raise
        raise ContractError(f"invalid NPZ chunk: {exc}") from exc
