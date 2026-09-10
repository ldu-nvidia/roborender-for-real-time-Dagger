from __future__ import annotations

import base64
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

CONTRACT_VERSION = "roborender.dagger.rtx.v1"
# Capture dictionaries remain name-addressed. Only this explicit positional wire
# order may be used for renderer/student messages.
RAVEN_CAPTURE_VIEWS = ("hand_left", "hand_right", "head_color")
RAVEN_VIEW_ORDER = ("head_color", "hand_left", "hand_right")
MODEL_VIEW_MAP = {"ext1": "head_color", "ext2": "hand_left", "wrist": "hand_right"}
ROUND_BETAS = {1: 0.75, 2: 0.50, 3: 0.25, 4: 0.00}
ACTION_DIM = 26
PINNED_RAVEN_ROOT = "/home/ldu/BC_render/rtx_BC_transfer"
PINNED_RAVEN_COMMIT = "94d79851d9c471b1210be81c75116d9bea835a4d"
CAPTURE_METADATA_KEYS = frozenset(
    {
        "view_order",
        "model_view_map",
        "observation_source",
        "prompt",
        "env_id",
        "seq",
        "depth_units",
        "depth_control_encoding",
        "robot_control_encoding",
    }
)
STUDENT_ACTION_DIM = 11
HOMIE_ACTION_DIM = 15


class ContractError(ValueError):
    """A request does not satisfy the versioned wire contract."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require_identifier(value: Any, name: str, max_len: int = 128) -> str:
    if not isinstance(value, str) or not value or len(value) > max_len:
        raise ContractError(f"{name} must be a non-empty string no longer than {max_len}")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    if any(c not in allowed for c in value):
        raise ContractError(f"{name} contains a disallowed character")
    if value in {".", ".."}:
        raise ContractError(f"{name} may not be a dot path segment")
    return value


def require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ContractError(f"{name} must be a lowercase SHA-256 hex string")
    return value


@dataclass(frozen=True)
class StepKey:
    run_id: str
    round_id: int
    actor_id: str
    rollout_window_id: str
    rollout_step_index: int
    episode_id: str
    episode_step_index: int

    @classmethod
    def from_mapping(cls, obj: Mapping[str, Any]) -> "StepKey":
        if not isinstance(obj, Mapping):
            raise ContractError("step_key must be an object")
        expected = {
            "run_id", "round_id", "actor_id", "rollout_window_id",
            "rollout_step_index", "episode_id", "episode_step_index",
        }
        if set(obj) != expected:
            raise ContractError(f"step_key keys must be exactly {sorted(expected)}")
        round_id = obj.get("round_id")
        if isinstance(round_id, bool) or not isinstance(round_id, int) or round_id not in ROUND_BETAS:
            raise ContractError("round_id must be an integer in [1, 4]")
        rollout_step = obj.get("rollout_step_index")
        if isinstance(rollout_step, bool) or not isinstance(rollout_step, int) or rollout_step < 0 or rollout_step >= 81:
            raise ContractError("rollout_step_index must be an integer in [0, 80]")
        episode_step = obj.get("episode_step_index")
        if isinstance(episode_step, bool) or not isinstance(episode_step, int) or episode_step < 0:
            raise ContractError("episode_step_index must be a non-negative integer")
        return cls(
            require_identifier(obj.get("run_id"), "run_id"),
            round_id,
            require_identifier(obj.get("actor_id"), "actor_id"),
            require_identifier(obj.get("rollout_window_id"), "rollout_window_id"),
            rollout_step,
            require_identifier(obj.get("episode_id"), "episode_id"),
            episode_step,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "round_id": self.round_id,
            "actor_id": self.actor_id,
            "rollout_window_id": self.rollout_window_id,
            "rollout_step_index": self.rollout_step_index,
            "episode_id": self.episode_id,
            "episode_step_index": self.episode_step_index,
        }

    @property
    def token(self) -> str:
        return (
            f"{self.run_id}__D{self.round_id}__{self.actor_id}__{self.rollout_window_id}"
            f"__{self.rollout_step_index:03d}__{self.episode_id}__{self.episode_step_index:05d}"
        )


@dataclass(frozen=True)
class ArrayRule:
    shape: tuple[int, ...]
    dtype: str
    finite: bool = False


@dataclass(frozen=True)
class RavenShapeProfile:
    actor_obs: int = 113
    teacher_obs: int = 226
    critic_obs: int = 226
    homie_obs: int = 516
    depth_h: int = 480
    depth_w: int = 832
    control_h: int = 240
    control_w: int = 416
    policy_h: int = 108
    policy_w: int = 192

    def capture_rules(self) -> dict[str, ArrayRule]:
        rules = {
            "actor_obs": ArrayRule((self.actor_obs,), "float32", True),
            "teacher_obs": ArrayRule((self.teacher_obs,), "float32", True),
            "critic_obs": ArrayRule((self.critic_obs,), "float32", True),
            "homie_obs": ArrayRule((self.homie_obs,), "float32", True),
            "gt_obj_pos": ArrayRule((3,), "float32", True),
            "teacher_action": ArrayRule((ACTION_DIM,), "float32", True),
            "homie_action": ArrayRule((HOMIE_ACTION_DIM,), "float32", True),
        }
        for view in RAVEN_CAPTURE_VIEWS:
            rules[f"depth_metric__{view}"] = ArrayRule((self.depth_h, self.depth_w), "float32", True)
            rules[f"depth_rawq__{view}"] = ArrayRule((self.depth_h, self.depth_w), "uint16")
            rules[f"depth_raw__{view}"] = ArrayRule((self.control_h, self.control_w), "float16", True)
            rules[f"depth_ctrl__{view}"] = ArrayRule((self.control_h, self.control_w), "uint8")
            rules[f"rgb_robot__{view}"] = ArrayRule((self.control_h, self.control_w, 3), "uint8")
            rules[f"anchor_rgb__{view}"] = ArrayRule((self.control_h, self.control_w, 3), "uint8")
        return rules


ALLOWED_WIRE_DTYPES = frozenset({"float16", "float32", "uint8", "uint16", "int64", "bool"})


def canonical_wire_array(value: np.ndarray) -> np.ndarray:
    """Return the only byte representation allowed on the wire: LE, C-order."""
    dtype_name = value.dtype.name
    if dtype_name not in ALLOWED_WIRE_DTYPES:
        raise ContractError(f"wire dtype {dtype_name} is not allowed")
    target = np.dtype(dtype_name)
    if target.itemsize > 1:
        target = target.newbyteorder("<")
    return np.ascontiguousarray(value, dtype=target)


def validate_array(name: str, value: Any, rule: ArrayRule) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ContractError(f"{name} must be a numpy array")
    if tuple(value.shape) != rule.shape:
        raise ContractError(f"{name} shape {tuple(value.shape)} != {rule.shape}")
    if value.dtype.name != rule.dtype:
        raise ContractError(f"{name} dtype {value.dtype.name} != {rule.dtype}")
    if value.dtype.itemsize > 1 and (
        value.dtype.byteorder == ">" or (value.dtype.byteorder == "=" and sys.byteorder != "little")
    ):
        raise ContractError(f"{name} must use little-endian element bytes")
    if rule.finite and not bool(np.isfinite(value).all()):
        raise ContractError(f"{name} contains NaN or infinity")
    return np.ascontiguousarray(value)


def encode_array(value: np.ndarray, max_array_bytes: int = 16 * 1024 * 1024) -> dict[str, Any]:
    if not isinstance(value, np.ndarray):
        raise ContractError("wire value must be a numpy array")
    arr = canonical_wire_array(value)
    raw = arr.tobytes(order="C")
    if len(raw) > max_array_bytes:
        raise ContractError(f"array body exceeds {max_array_bytes} bytes")
    return {
        "dtype": arr.dtype.name,
        "shape": list(arr.shape),
        "sha256": sha256_bytes(raw),
        "data_b64": base64.b64encode(raw).decode("ascii"),
    }


def decode_array(obj: Mapping[str, Any], *, max_array_bytes: int = 16 * 1024 * 1024) -> np.ndarray:
    if not isinstance(obj, Mapping) or set(obj) != {"dtype", "shape", "sha256", "data_b64"}:
        raise ContractError("invalid array envelope keys")
    dtype_name = obj["dtype"]
    if dtype_name not in ALLOWED_WIRE_DTYPES:
        raise ContractError("array dtype is not allowed")
    shape = obj["shape"]
    if not isinstance(shape, list) or len(shape) > 5 or any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in shape):
        raise ContractError("invalid array shape")
    try:
        raw = base64.b64decode(obj["data_b64"], validate=True)
    except Exception as exc:
        raise ContractError("invalid base64 array body") from exc
    if len(raw) > max_array_bytes:
        raise ContractError("decoded array exceeds body limit")
    dtype = np.dtype(dtype_name)
    if dtype.itemsize > 1:
        dtype = dtype.newbyteorder("<")
    expected = math.prod(shape) * dtype.itemsize
    if expected != len(raw):
        raise ContractError(f"array byte length {len(raw)} != expected {expected}")
    require_sha256(obj["sha256"], "array sha256")
    if sha256_bytes(raw) != obj["sha256"]:
        raise ContractError("array sha256 mismatch")
    return np.frombuffer(raw, dtype=dtype).reshape(tuple(shape)).copy()


def encode_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {name: encode_array(arrays[name]) for name in sorted(arrays)}


def decode_arrays(arrays: Mapping[str, Any], *, max_total_bytes: int = 64 * 1024 * 1024) -> dict[str, np.ndarray]:
    if not isinstance(arrays, Mapping):
        raise ContractError("arrays must be an object")
    result: dict[str, np.ndarray] = {}
    total = 0
    for name in sorted(arrays):
        require_identifier(name, "array name", 96)
        result[name] = decode_array(arrays[name])
        total += result[name].nbytes
        if total > max_total_bytes:
            raise ContractError("decoded arrays exceed total body limit")
    return result


def payload_digest(metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    h.update(canonical_json(metadata))
    for name in sorted(arrays):
        arr = canonical_wire_array(arrays[name])
        h.update(canonical_json({"name": name, "dtype": arr.dtype.name, "shape": list(arr.shape)}))
        h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def anchor_rgb_digest(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash the actual fixed Image3F reference triplet in online view order."""
    h = hashlib.sha256()
    for view in RAVEN_VIEW_ORDER:
        name = f"anchor_rgb__{view}"
        if name not in arrays:
            raise ContractError(f"missing fixed anchor array {name}")
        arr = canonical_wire_array(arrays[name])
        if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[-1] != 3:
            raise ContractError(f"{name} must be HxWx3 uint8")
        h.update(canonical_json({"view": view, "dtype": "uint8", "shape": list(arr.shape)}))
        h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def validate_beta(round_id: int, beta: Any) -> float:
    if round_id not in ROUND_BETAS:
        raise ContractError(f"round_id must be one of {sorted(ROUND_BETAS)}")
    if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not math.isfinite(float(beta)):
        raise ContractError("beta must be finite")
    expected = ROUND_BETAS[round_id]
    if float(beta) != expected:
        raise ContractError(f"beta {beta} does not match canonical {round_id} beta {expected}")
    return expected


def validate_action(value: np.ndarray) -> np.ndarray:
    return validate_array("selected_action", value, ArrayRule((ACTION_DIM,), "float32", True))
