"""Strict JSON/ndarray contract for the RTX-to-H200 live DAgger step boundary."""

from __future__ import annotations

import base64
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .contracts import CANONICAL_BETA, ContractError, VIEWS, exact_keys, identifier, sha256_digest


STEP_CONTRACT = "roborender.dagger.rtx.v1"
RAVEN_COMMIT = "94d79851d9c471b1210be81c75116d9bea835a4d"
MODEL_VIEW_MAP = {"ext1": "head_color", "ext2": "hand_left", "wrist": "hand_right"}
MAX_DECODED_ARRAY_BYTES = 64 * 1024 * 1024
ARRAY_ENVELOPE_KEYS = {"dtype", "shape", "sha256", "data_b64"}
ALLOWED_DTYPES = {"float16", "float32", "uint8", "uint16", "int64", "bool"}


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractError(f"value is not canonical JSON: {exc}") from exc


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def encode_array(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    if array.dtype.name not in ALLOWED_DTYPES:
        raise ContractError(f"unsupported wire dtype {array.dtype.name}")
    raw = array.tobytes(order="C")
    return {
        "dtype": array.dtype.name,
        "shape": list(array.shape),
        "sha256": bytes_sha256(raw),
        "data_b64": base64.b64encode(raw).decode("ascii"),
    }


def decode_array(value: Any, *, max_bytes: int = 16 * 1024 * 1024) -> np.ndarray:
    if not isinstance(value, Mapping) or set(value) != ARRAY_ENVELOPE_KEYS:
        raise ContractError("array envelope keys must be dtype,shape,sha256,data_b64")
    dtype_name = value["dtype"]
    if dtype_name not in ALLOWED_DTYPES:
        raise ContractError(f"unsupported wire dtype {dtype_name!r}")
    shape = value["shape"]
    if (
        not isinstance(shape, list)
        or len(shape) > 5
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in shape)
    ):
        raise ContractError("invalid array shape")
    try:
        raw = base64.b64decode(value["data_b64"], validate=True)
    except Exception as exc:
        raise ContractError("invalid base64 array body") from exc
    if len(raw) > max_bytes:
        raise ContractError(f"decoded array exceeds {max_bytes} bytes")
    dtype = np.dtype(dtype_name)
    if math.prod(shape) * dtype.itemsize != len(raw):
        raise ContractError("array byte count does not match dtype and shape")
    if bytes_sha256(raw) != sha256_digest(value["sha256"], "array sha256"):
        raise ContractError("array SHA-256 mismatch")
    return np.frombuffer(raw, dtype=dtype).reshape(tuple(shape)).copy()


def decode_arrays(values: Any) -> dict[str, np.ndarray]:
    if not isinstance(values, Mapping):
        raise ContractError("arrays must be an object")
    result: dict[str, np.ndarray] = {}
    total = 0
    for name in sorted(values):
        identifier(name, "array name")
        result[name] = decode_array(values[name])
        total += result[name].nbytes
        if total > MAX_DECODED_ARRAY_BYTES:
            raise ContractError("decoded arrays exceed 64 MiB")
    return result


def arrays_digest(metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update(canonical_bytes(metadata))
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(canonical_bytes({
            "name": name, "dtype": array.dtype.name, "shape": list(array.shape)
        }))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def anchor_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for view in VIEWS:
        array = np.ascontiguousarray(arrays[f"anchor_rgb__{view}"])
        digest.update(canonical_bytes({
            "view": view, "dtype": "uint8", "shape": [240, 416, 3]
        }))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def observation_input_digest(
    metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
) -> str:
    """Bind the exact Raven observation inputs used for this policy decision."""
    source = metadata["observation_source"]
    names: list[str] = []
    for view in VIEWS:
        if source == "raytraced":
            names.append(f"raytraced_rgb__{view}")
        else:
            names.extend((
                f"anchor_rgb__{view}",
                f"depth_ctrl__{view}",
                f"rgb_robot__{view}",
            ))
    inputs = []
    for name in names:
        array = np.ascontiguousarray(arrays[name])
        inputs.append({
            "name": name,
            "dtype": array.dtype.name,
            "shape": list(array.shape),
            "sha256": bytes_sha256(array.tobytes(order="C")),
        })
    preimage = {
        "contract_version": metadata.get("contract_version", STEP_CONTRACT),
        "step_key": metadata["step_key"],
        "observation_source": source,
        "prompt": metadata["prompt"],
        "view_order": metadata["view_order"],
        "model_view_map": metadata["model_view_map"],
        "inputs": inputs,
    }
    return bytes_sha256(canonical_bytes(preimage))


def _array(array: np.ndarray, name: str, shape: tuple[int, ...], dtype: str) -> None:
    if array.shape != shape or array.dtype.name != dtype:
        raise ContractError(
            f"{name} must be {dtype}{list(shape)}, got {array.dtype.name}{list(array.shape)}"
        )
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ContractError(f"{name} contains NaN or infinity")


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
    def parse(cls, payload: Any) -> "StepKey":
        required = {
            "run_id", "round_id", "actor_id", "rollout_window_id",
            "rollout_step_index", "episode_id", "episode_step_index",
        }
        exact_keys(payload, required)
        round_id = payload["round_id"]
        if isinstance(round_id, bool) or not isinstance(round_id, int) or round_id not in CANONICAL_BETA:
            raise ContractError("round_id must be an integer from 1 through 4")
        rollout_index = payload["rollout_step_index"]
        episode_index = payload["episode_step_index"]
        if isinstance(rollout_index, bool) or not isinstance(rollout_index, int) or not 0 <= rollout_index < 81:
            raise ContractError("rollout_step_index must be an integer in [0,80]")
        if isinstance(episode_index, bool) or not isinstance(episode_index, int) or episode_index < 0:
            raise ContractError("episode_step_index must be a non-negative integer")
        return cls(
            run_id=identifier(payload["run_id"], "run_id"),
            round_id=round_id,
            actor_id=identifier(payload["actor_id"], "actor_id"),
            rollout_window_id=identifier(payload["rollout_window_id"], "rollout_window_id"),
            rollout_step_index=rollout_index,
            episode_id=identifier(payload["episode_id"], "episode_id"),
            episode_step_index=episode_index,
        )

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def validate_metadata(
    value: Any,
    key: StepKey,
    beta: float,
    *,
    expected_contract: str = STEP_CONTRACT,
) -> dict[str, Any]:
    required = {
        "contract_version", "step_key", "beta", "view_order", "model_view_map",
        "observation_source", "prompt", "env_id", "seq", "depth_units",
        "depth_control_encoding", "robot_control_encoding", "episode_assignment",
    }
    exact_keys(value, required)
    if value["contract_version"] != expected_contract:
        raise ContractError("metadata contract_version mismatch")
    if StepKey.parse(value["step_key"]) != key:
        raise ContractError("metadata step_key differs from top-level step_key")
    if float(value["beta"]) != beta:
        raise ContractError("metadata beta differs from top-level beta")
    if value["view_order"] != list(VIEWS) or value["model_view_map"] != MODEL_VIEW_MAP:
        raise ContractError("Raven three-view order or model mapping mismatch")
    source = value["observation_source"]
    if source not in ("raytraced", "generated_controls"):
        raise ContractError("observation_source must be raytraced or generated_controls")
    if not isinstance(value["prompt"], str) or not 1 <= len(value["prompt"]) <= 4096:
        raise ContractError("prompt must contain 1..4096 characters")
    for field in ("env_id", "seq"):
        if isinstance(value[field], bool) or not isinstance(value[field], int):
            raise ContractError(f"{field} must be an integer")
    constants = {
        "depth_units": "meter",
        "depth_control_encoding": "raven_h5_v4",
        "robot_control_encoding": "textured_robot_rgb_on_black",
    }
    for field, expected in constants.items():
        if value[field] != expected:
            raise ContractError(f"{field} must equal {expected}")
    assignment = value["episode_assignment"]
    exact_keys(assignment, {
        "use_raytraced", "observation_source", "anchor_sha256", "assignment_source",
        "raven_source", "raven_commit",
    })
    if not isinstance(assignment["use_raytraced"], bool):
        raise ContractError("use_raytraced must be bool")
    if assignment["observation_source"] != source:
        raise ContractError("episode assignment observation source mismatch")
    if assignment["use_raytraced"] != (source == "raytraced"):
        raise ContractError("use_raytraced disagrees with observation_source")
    sha256_digest(assignment["anchor_sha256"], "anchor_sha256")
    if assignment["assignment_source"] != "pinned_raven_id_sample_anchor":
        raise ContractError("episode assignment source mismatch")
    if assignment["raven_source"] != "/home/ldu/BC_render/rtx_BC_transfer":
        raise ContractError("raven_source must identify the pinned RTX Raven tree")
    if assignment["raven_commit"] != RAVEN_COMMIT:
        raise ContractError("Raven commit mismatch")
    return dict(value)


def validate_capture_arrays(arrays: dict[str, np.ndarray], source: str) -> None:
    required = {
        "actor_obs", "teacher_obs", "critic_obs", "homie_obs", "gt_obj_pos",
        "teacher_action", "homie_action",
    }
    for view in VIEWS:
        required.update({
            f"depth_metric__{view}", f"depth_rawq__{view}", f"depth_raw__{view}",
            f"depth_ctrl__{view}", f"rgb_robot__{view}", f"anchor_rgb__{view}",
        })
        if source == "raytraced":
            required.add(f"raytraced_rgb__{view}")
    if set(arrays) != required:
        raise ContractError(
            f"capture arrays differ; missing={sorted(required - set(arrays))}, "
            f"unknown={sorted(set(arrays) - required)}"
        )
    fixed = {
        "actor_obs": ((113,), "float32"),
        "teacher_obs": ((226,), "float32"),
        "critic_obs": ((226,), "float32"),
        "homie_obs": ((516,), "float32"),
        "gt_obj_pos": ((3,), "float32"),
        "teacher_action": ((26,), "float32"),
        "homie_action": ((15,), "float32"),
    }
    for name, (shape, dtype) in fixed.items():
        _array(arrays[name], name, shape, dtype)
    if not np.array_equal(arrays["teacher_action"][11:], arrays["homie_action"]):
        raise ContractError("teacher_action[11:] must exactly equal homie_action")
    for view in VIEWS:
        _array(arrays[f"depth_metric__{view}"], f"depth_metric__{view}", (480, 832), "float32")
        _array(arrays[f"depth_rawq__{view}"], f"depth_rawq__{view}", (480, 832), "uint16")
        _array(arrays[f"depth_raw__{view}"], f"depth_raw__{view}", (240, 416), "float16")
        _array(arrays[f"depth_ctrl__{view}"], f"depth_ctrl__{view}", (240, 416), "uint8")
        _array(arrays[f"rgb_robot__{view}"], f"rgb_robot__{view}", (240, 416, 3), "uint8")
        _array(arrays[f"anchor_rgb__{view}"], f"anchor_rgb__{view}", (240, 416, 3), "uint8")
        if source == "raytraced":
            name = f"raytraced_rgb__{view}"
            _array(arrays[name], name, (108, 192, 3), "float32")
            if np.any(arrays[name] < 0.0) or np.any(arrays[name] > 1.0):
                raise ContractError(f"{name} must be finite RGB in [0,1]")


def parse_prepare(payload: dict[str, Any]) -> tuple[StepKey, float, dict[str, Any], dict[str, np.ndarray]]:
    exact_keys(payload, {
        "contract_version", "idempotency_key", "step_key", "prepare_receipt",
        "capture_digest", "beta", "metadata", "arrays",
    })
    if payload["contract_version"] != STEP_CONTRACT:
        raise ContractError("contract_version mismatch")
    identifier(payload["idempotency_key"], "idempotency_key")
    sha256_digest(payload["prepare_receipt"], "prepare_receipt")
    capture_digest = sha256_digest(payload["capture_digest"], "capture_digest")
    key = StepKey.parse(payload["step_key"])
    beta = payload["beta"]
    if isinstance(beta, bool) or not isinstance(beta, (int, float)) or not math.isfinite(float(beta)):
        raise ContractError("beta must be finite")
    beta = float(beta)
    if beta != CANONICAL_BETA[key.round_id]:
        raise ContractError("beta differs from canonical round schedule")
    metadata = validate_metadata(payload["metadata"], key, beta)
    arrays = decode_arrays(payload["arrays"])
    validate_capture_arrays(arrays, metadata["observation_source"])
    if anchor_digest(arrays) != metadata["episode_assignment"]["anchor_sha256"]:
        raise ContractError("anchor RGB bytes do not match episode anchor_sha256")
    if arrays_digest(metadata, arrays) != capture_digest:
        raise ContractError("capture_digest mismatch")
    return key, beta, metadata, arrays


def validate_transition(value: Any) -> dict[str, Any]:
    exact_keys(value, {"reward", "done", "timeout", "metadata"})
    reward = value["reward"]
    if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(float(reward)):
        raise ContractError("transition reward must be finite")
    if not isinstance(value["done"], bool) or not isinstance(value["timeout"], bool):
        raise ContractError("transition done and timeout must be bool")
    if not isinstance(value["metadata"], dict):
        raise ContractError("transition metadata must be an object")
    canonical_bytes(value["metadata"])
    return dict(value)


def validate_sealed_window(
    value: Any,
    key: StepKey,
    *,
    expected_contract: str = STEP_CONTRACT,
) -> dict[str, Any]:
    required = {
        "schema", "contract_version", "run_id", "round_id", "actor_id",
        "rollout_window_id", "row_count", "artifact_role", "learner_complete",
        "contains_policy_rgb", "authoritative_learner_chunk_owner",
        "episode_ids_are_per_step",
        "actual_dones_preserved", "no_previous_round_replay", "view_order",
        "per_step_identity_fields", "episode_ids", "episode_step_indices",
        "env_ids", "seqs", "round_ids", "observation_sources", "dones", "timeouts",
        "rewards", "behavior_sources", "prepare_receipts", "capture_digests",
        "selection_ids", "commit_digests", "commit_receipts",
        "path", "bytes", "sha256", "provenance",
    }
    exact_keys(value, required)
    constants = {
        "schema": "roborender.raven.dagger-window.v1",
        "contract_version": expected_contract,
        "run_id": key.run_id,
        "round_id": key.round_id,
        "actor_id": key.actor_id,
        "rollout_window_id": key.rollout_window_id,
        "row_count": 81,
        "artifact_role": "rtx_identity_outcome_audit",
        "learner_complete": False,
        "contains_policy_rgb": False,
        "authoritative_learner_chunk_owner": "h200_render_student_ledger",
        "episode_ids_are_per_step": True,
        "actual_dones_preserved": True,
        "no_previous_round_replay": True,
        "view_order": list(VIEWS),
        "per_step_identity_fields": [
            "episode_ids", "episode_step_indices", "rollout_step_indices", "round_ids",
            "env_id_per_step", "seq_per_step", "observation_source", "dones", "timeouts",
            "rewards", "behavior_source", "prepare_receipts", "capture_digests",
            "selection_ids", "commit_digests", "commit_receipts",
        ],
        "path": "sealed.npz",
    }
    for field, expected in constants.items():
        if value[field] != expected:
            raise ContractError(f"sealed_window {field} differs from the frozen contract")
    vector_rules = {
        "episode_ids": lambda item: isinstance(item, str),
        "episode_step_indices": lambda item: isinstance(item, int) and not isinstance(item, bool) and item >= 0,
        "env_ids": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "seqs": lambda item: isinstance(item, int) and not isinstance(item, bool),
        "round_ids": lambda item: isinstance(item, int) and not isinstance(item, bool) and item == key.round_id,
        "observation_sources": lambda item: item in ("raytraced", "generated_controls"),
        "dones": lambda item: isinstance(item, bool),
        "timeouts": lambda item: isinstance(item, bool),
        "rewards": lambda item: (
            isinstance(item, (int, float)) and not isinstance(item, bool)
            and math.isfinite(float(item))
        ),
        "behavior_sources": lambda item: item in ("teacher", "student"),
        "prepare_receipts": lambda item: isinstance(item, str) and len(item) == 64,
        "capture_digests": lambda item: isinstance(item, str) and len(item) == 64,
        "selection_ids": lambda item: isinstance(item, str),
        "commit_digests": lambda item: isinstance(item, str) and len(item) == 64,
        "commit_receipts": lambda item: isinstance(item, str) and len(item) == 64,
    }
    for field, predicate in vector_rules.items():
        vector = value[field]
        if not isinstance(vector, list) or len(vector) != 81 or not all(
            predicate(item) for item in vector
        ):
            raise ContractError(f"sealed_window {field} must contain 81 valid entries")
    for episode_id in value["episode_ids"]:
        identifier(episode_id, "sealed_window episode_id")
    for selection_id in value["selection_ids"]:
        identifier(selection_id, "sealed_window selection_id")
    for field in ("prepare_receipts", "capture_digests", "commit_digests", "commit_receipts"):
        for digest in value[field]:
            sha256_digest(digest, f"sealed_window {field}")
    if isinstance(value["bytes"], bool) or not isinstance(value["bytes"], int) or value["bytes"] <= 0:
        raise ContractError("sealed_window bytes must be positive")
    sha256_digest(value["sha256"], "sealed_window sha256")
    for index in range(1, 81):
        same_episode = value["episode_ids"][index] == value["episode_ids"][index - 1]
        if same_episode:
            if value["episode_step_indices"][index] != value["episode_step_indices"][index - 1] + 1:
                raise ContractError("sealed_window episode steps are not contiguous")
            if value["dones"][index - 1] or value["timeouts"][index - 1]:
                raise ContractError("sealed_window continued an episode after done/timeout")
            if value["observation_sources"][index] != value["observation_sources"][index - 1]:
                raise ContractError("sealed_window observation source changed within an episode")
        else:
            if not (value["dones"][index - 1] or value["timeouts"][index - 1]):
                raise ContractError("sealed_window changed episode without done/timeout")
            if value["episode_step_indices"][index] != 0:
                raise ContractError("sealed_window new episode did not reset episode_step_index")
    if (
        value["episode_ids"][80] != key.episode_id
        or value["episode_step_indices"][80] != key.episode_step_index
    ):
        raise ContractError("sealed_window final episode identity differs from commit step")
    provenance = value["provenance"]
    exact_keys(provenance, {
        "raven_root", "raven_commit", "raven_videogen_trainer_sha256",
        "raven_teacher_homie_trainer_sha256", "isaac_capture_sha256",
        "image_dagger_config_sha256", "teacher_checkpoint_sha256",
        "homie_walk_sha256", "homie_stand_sha256", "episode_bindings",
    })
    if provenance["raven_root"] != "/home/ldu/BC_render/rtx_BC_transfer":
        raise ContractError("sealed_window Raven root mismatch")
    if provenance["raven_commit"] != RAVEN_COMMIT:
        raise ContractError("sealed_window Raven commit mismatch")
    for field in (
        "raven_videogen_trainer_sha256", "raven_teacher_homie_trainer_sha256",
        "isaac_capture_sha256", "image_dagger_config_sha256",
        "teacher_checkpoint_sha256", "homie_walk_sha256", "homie_stand_sha256",
    ):
        sha256_digest(provenance[field], field)
    bindings = provenance["episode_bindings"]
    if not isinstance(bindings, list) or not bindings:
        raise ContractError("episode_bindings must be a non-empty array")
    seen: set[str] = set()
    previous_index = -1
    for binding in bindings:
        exact_keys(binding, {
            "episode_id", "first_rollout_step_index", "use_raytraced",
            "observation_source", "anchor_sha256",
        })
        episode_id = identifier(binding["episode_id"], "episode binding episode_id")
        if episode_id in seen:
            raise ContractError("episode_bindings contains a duplicate episode_id")
        seen.add(episode_id)
        index = binding["first_rollout_step_index"]
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 81:
            raise ContractError("invalid episode binding first_rollout_step_index")
        if index <= previous_index:
            raise ContractError("episode bindings must be in rollout order")
        previous_index = index
        if not isinstance(binding["use_raytraced"], bool):
            raise ContractError("episode binding use_raytraced must be bool")
        source = binding["observation_source"]
        if source not in ("raytraced", "generated_controls"):
            raise ContractError("invalid episode binding observation_source")
        if binding["use_raytraced"] != (source == "raytraced"):
            raise ContractError("episode binding source mismatch")
        sha256_digest(binding["anchor_sha256"], "episode binding anchor_sha256")
    if bindings[0]["first_rollout_step_index"] != 0:
        raise ContractError("first episode binding must start at rollout step zero")
    expected_indices = [0] + [
        index for index in range(1, 81)
        if value["episode_ids"][index] != value["episode_ids"][index - 1]
    ]
    if len(bindings) != len(expected_indices):
        raise ContractError("episode bindings do not cover every episode transition")
    for binding, index in zip(bindings, expected_indices):
        source = value["observation_sources"][index]
        if (
            binding["episode_id"] != value["episode_ids"][index]
            or binding["first_rollout_step_index"] != index
            or binding["observation_source"] != source
            or binding["use_raytraced"] != (source == "raytraced")
        ):
            raise ContractError("episode binding differs from per-step identity")
    return dict(value)


def response_receipt(response_without_receipt: Mapping[str, Any]) -> str:
    return bytes_sha256(canonical_bytes(response_without_receipt))
