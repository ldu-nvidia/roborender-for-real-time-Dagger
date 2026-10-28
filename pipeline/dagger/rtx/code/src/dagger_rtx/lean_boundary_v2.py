"""Frozen RTX builders and validators for the lean batched DAgger v2 wire."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import (
    ACTION_DIM,
    CAPTURE_METADATA_KEYS,
    CONTRACT_VERSION,
    HOMIE_ACTION_DIM,
    MODEL_VIEW_MAP,
    RAVEN_VIEW_ORDER,
    ROUND_BETAS,
    STUDENT_ACTION_DIM,
    ArrayRule,
    ContractError,
    RavenShapeProfile,
    StepKey,
    anchor_rgb_digest,
    canonical_json,
    decode_array,
    payload_digest,
    require_identifier,
    require_sha256,
    sha256_bytes,
    validate_array,
    validate_beta,
)
from .ndarray_batch_v2 import pack_ndarray_batch


CONTRACT_VERSION_V2 = "roborender.dagger.rtx.v2"
H200_ANCHOR_BATCH_PATH_V2 = "/v2/anchors/register-batch"
H200_PREPARE_BATCH_PATH_V2 = "/v2/steps/prepare-batch"
H200_COMMIT_BATCH_PATH_V2 = "/v2/steps/commit-batch"

STATE_SUPERVISION_ARRAYS = (
    "actor_obs",
    "teacher_obs",
    "critic_obs",
    "homie_obs",
    "gt_obj_pos",
    "teacher_action",
    "homie_action",
)
FORBIDDEN_PREPARE_ARRAY_PREFIXES = ("depth_metric__", "depth_rawq__", "depth_raw__")
_PREPARE_TOP_KEYS = frozenset({"contract_version", "kind", "batch_id", "lanes"})
_ANCHOR_TOP_KEYS = frozenset({"contract_version", "kind", "batch_id", "anchors"})
_PREPARE_LANE_KEYS = frozenset(
    {
        "contract_version",
        "idempotency_key",
        "step_key",
        "prepare_receipt",
        "capture_digest",
        "wire_digest",
        "beta",
        "metadata",
        "arrays",
    }
)
_SELECTION_LANE_KEYS = frozenset(
    {
        "contract_version",
        "idempotency_key",
        "step_key",
        "prepare_receipt",
        "capture_digest",
        "wire_digest",
        "beta",
        "selection_id",
        "selection_draw",
        "selected_source",
        "student_action_11",
        "selected_action",
        "observation_digest",
        "selection_receipt",
    }
)
_COMMIT_LANE_KEYS = frozenset(
    {
        "contract_version",
        "idempotency_key",
        "step_key",
        "prepare_receipt",
        "selection_id",
        "selection_receipt",
        "commit_receipt",
        "transition",
        "sealed_window",
    }
)
_COMMIT_ACK_LANE_KEYS = frozenset(
    {
        "contract_version",
        "idempotency_key",
        "step_key",
        "commit_receipt",
        "remote_commit_receipt",
        "status",
    }
)
_LOCAL_METADATA_KEYS = CAPTURE_METADATA_KEYS | {
    "contract_version",
    "step_key",
    "beta",
    "episode_assignment",
}


def deterministic_batch_id(kind: str, identities: Sequence[Any]) -> str:
    require_identifier(kind, "batch kind", 32)
    digest = sha256_bytes(canonical_json({"kind": kind, "identities": list(identities)}))
    return f"{kind}-{digest[:48]}"


def _raw_local_arrays(local_prepare: Mapping[str, Any]) -> Mapping[str, np.ndarray]:
    arrays = local_prepare.get("arrays")
    if not isinstance(arrays, Mapping) or not arrays:
        raise ContractError("lean local prepare must carry raw numpy arrays")
    if any(not isinstance(value, np.ndarray) for value in arrays.values()):
        raise ContractError("lean local prepare arrays must not be JSON/base64 envelopes")
    return arrays


def _v2_metadata(local_prepare: Mapping[str, Any]) -> dict[str, Any]:
    metadata = local_prepare.get("metadata")
    if not isinstance(metadata, Mapping) or set(metadata) != _LOCAL_METADATA_KEYS:
        raise ContractError(f"local metadata keys must be exactly {sorted(_LOCAL_METADATA_KEYS)}")
    if metadata.get("contract_version") != CONTRACT_VERSION:
        raise ContractError("source capture metadata must use the frozen v1 local contract")
    value = dict(metadata)
    value["contract_version"] = CONTRACT_VERSION_V2
    if tuple(value.get("view_order", ())) != RAVEN_VIEW_ORDER:
        raise ContractError("v2 metadata view_order mismatch")
    if value.get("model_view_map") != MODEL_VIEW_MAP:
        raise ContractError("v2 metadata model_view_map mismatch")
    assignment = value.get("episode_assignment")
    if not isinstance(assignment, Mapping):
        raise ContractError("v2 metadata requires episode_assignment")
    source = value.get("observation_source")
    if assignment.get("observation_source") != source:
        raise ContractError("v2 assignment observation_source mismatch")
    if assignment.get("use_raytraced") is not (source == "raytraced"):
        raise ContractError("v2 assignment use_raytraced mismatch")
    require_sha256(assignment.get("anchor_sha256"), "anchor_sha256")
    return value


def project_lean_arrays(
    local_prepare: Mapping[str, Any],
    *,
    shape_profile: RavenShapeProfile | None = None,
) -> dict[str, np.ndarray]:
    """Select only H200-required arrays and reject source-dependent mistakes."""

    profile = shape_profile or RavenShapeProfile()
    source = local_prepare.get("metadata", {}).get("observation_source")
    if source not in {"generated_controls", "raytraced"}:
        raise ContractError("observation_source must be generated_controls or raytraced")
    arrays = _raw_local_arrays(local_prepare)
    rules = profile.capture_rules()
    names = list(STATE_SUPERVISION_ARRAYS)
    for view in RAVEN_VIEW_ORDER:
        names.extend((f"depth_ctrl__{view}", f"rgb_robot__{view}"))
        if source == "raytraced":
            names.append(f"raytraced_rgb__{view}")
    missing = sorted(set(names) - set(arrays))
    if missing:
        raise ContractError(f"lean prepare source is missing arrays: {missing}")
    projected: dict[str, np.ndarray] = {}
    for name in names:
        value = arrays[name]
        if name.startswith("raytraced_rgb__"):
            validate_array(
                name,
                value,
                ArrayRule((profile.policy_h, profile.policy_w, 3), "float32", True),
            )
            if bool((value < 0).any()) or bool((value > 1).any()):
                raise ContractError(f"{name} must be in [0,1]")
        else:
            validate_array(name, value, rules[name])
        projected[name] = np.ascontiguousarray(value)
    if source == "generated_controls" and any(
        name.startswith("raytraced_rgb__") for name in projected
    ):
        raise ContractError("generated-controls lane may not send raytraced RGB")
    if any(
        name.startswith(FORBIDDEN_PREPARE_ARRAY_PREFIXES) or name.startswith("anchor_rgb__")
        for name in projected
    ):
        raise ContractError("lean projection contains a forbidden raw-depth or anchor array")
    return projected


def anchor_records_from_prepares(
    local_prepares: Sequence[Mapping[str, Any]],
    *,
    shape_profile: RavenShapeProfile | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, np.ndarray]]]:
    """Return first-occurrence ordered unique fixed anchors."""

    profile = shape_profile or RavenShapeProfile()
    records: list[dict[str, Any]] = []
    groups: list[dict[str, np.ndarray]] = []
    seen: dict[str, str] = {}
    for local in local_prepares:
        arrays = _raw_local_arrays(local)
        assignment = local.get("metadata", {}).get("episode_assignment")
        if not isinstance(assignment, Mapping):
            raise ContractError("local prepare is missing episode_assignment")
        expected = require_sha256(assignment.get("anchor_sha256"), "anchor_sha256")
        anchors = {
            f"anchor_rgb__{view}": arrays[f"anchor_rgb__{view}"]
            for view in RAVEN_VIEW_ORDER
        }
        for name, value in anchors.items():
            validate_array(
                name,
                value,
                ArrayRule((profile.control_h, profile.control_w, 3), "uint8"),
            )
        actual = anchor_rgb_digest(anchors)
        if actual != expected:
            raise ContractError("fixed anchor bytes do not match episode anchor_sha256")
        byte_identity = sha256_bytes(
            b"".join(anchors[name].tobytes(order="C") for name in sorted(anchors))
        )
        prior = seen.get(expected)
        if prior is not None:
            if prior != byte_identity:
                raise ContractError("same anchor_sha256 was associated with different bytes")
            continue
        seen[expected] = byte_identity
        records.append({"anchor_sha256": expected})
        groups.append(anchors)
    return records, groups


def build_anchor_batch(
    anchors: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, np.ndarray]],
    *,
    batch_id: str | None = None,
) -> tuple[str, bytes]:
    identities = [require_sha256(item.get("anchor_sha256"), "anchor_sha256") for item in anchors]
    actual_batch_id = batch_id or deterministic_batch_id("anchors", identities)
    require_identifier(actual_batch_id, "batch_id")
    header = {
        "contract_version": CONTRACT_VERSION_V2,
        "kind": "anchor_batch",
        "batch_id": actual_batch_id,
        "anchors": [dict(item) for item in anchors],
    }
    return actual_batch_id, pack_ndarray_batch(header, groups)


def validate_anchor_batch_response(
    response: Mapping[str, Any],
    *,
    batch_id: str,
    expected: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(response, Mapping) or set(response) != _ANCHOR_TOP_KEYS:
        raise ContractError(f"anchor response keys must be exactly {sorted(_ANCHOR_TOP_KEYS)}")
    if response["contract_version"] != CONTRACT_VERSION_V2:
        raise ContractError("anchor response contract version mismatch")
    if response["kind"] != "anchor_batch_response" or response["batch_id"] != batch_id:
        raise ContractError("anchor response kind or batch_id mismatch")
    rows = response["anchors"]
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ContractError("anchor response length mismatch")
    result: list[dict[str, Any]] = []
    for row, source in zip(rows, expected, strict=True):
        keys = {"anchor_sha256", "status", "bytes"}
        if not isinstance(row, Mapping) or set(row) != keys:
            raise ContractError(f"anchor response row keys must be exactly {sorted(keys)}")
        if row["anchor_sha256"] != source["anchor_sha256"]:
            raise ContractError("anchor response order or SHA-256 mismatch")
        if row["status"] not in {"registered", "cached"}:
            raise ContractError("anchor response status must be registered or cached")
        if isinstance(row["bytes"], bool) or not isinstance(row["bytes"], int) or row["bytes"] <= 0:
            raise ContractError("anchor response bytes must be positive")
        result.append(dict(row))
    return result


def build_prepare_batch(
    local_prepares: Sequence[Mapping[str, Any]],
    idempotency_keys: Sequence[str],
    *,
    shape_profile: RavenShapeProfile | None = None,
    batch_id: str | None = None,
) -> tuple[str, bytes, list[dict[str, Any]], list[dict[str, np.ndarray]]]:
    if not local_prepares or len(local_prepares) != len(idempotency_keys):
        raise ContractError("prepare batch requires equally sized non-empty lane inputs")
    records: list[dict[str, Any]] = []
    groups: list[dict[str, np.ndarray]] = []
    identities: list[dict[str, Any]] = []
    for local, idempotency_key in zip(local_prepares, idempotency_keys, strict=True):
        if local.get("contract_version") != CONTRACT_VERSION:
            raise ContractError("local prepare contract version mismatch")
        key = StepKey.from_mapping(local.get("step_key"))
        require_identifier(idempotency_key, "idempotency_key")
        require_sha256(local.get("prepare_receipt"), "prepare_receipt")
        require_sha256(local.get("capture_digest"), "capture_digest")
        metadata = _v2_metadata(local)
        if metadata.get("step_key") != key.as_dict():
            raise ContractError("prepare metadata step_key mismatch")
        beta = validate_beta(key.round_id, metadata.get("beta"))
        lean = project_lean_arrays(local, shape_profile=shape_profile)
        if not np.array_equal(
            lean["teacher_action"][STUDENT_ACTION_DIM:], lean["homie_action"]
        ):
            raise ContractError("teacher_action[11:] must equal homie_action before send")
        wire_digest = payload_digest(metadata, lean)
        record = {
            "contract_version": CONTRACT_VERSION_V2,
            "idempotency_key": idempotency_key,
            "step_key": key.as_dict(),
            "prepare_receipt": local["prepare_receipt"],
            "capture_digest": local["capture_digest"],
            "wire_digest": wire_digest,
            "beta": beta,
            "metadata": metadata,
        }
        records.append(record)
        groups.append(lean)
        identities.append(
            {
                "idempotency_key": idempotency_key,
                "step_key": key.as_dict(),
                "wire_digest": wire_digest,
            }
        )
    if len(set(idempotency_keys)) != len(idempotency_keys):
        raise ContractError("prepare batch idempotency keys must be unique per lane")
    step_tokens = [StepKey.from_mapping(record["step_key"]).token for record in records]
    if len(set(step_tokens)) != len(step_tokens):
        raise ContractError("prepare batch logical step identities must be unique")
    shared = {
        (
            record["step_key"]["run_id"],
            record["step_key"]["round_id"],
            record["step_key"]["rollout_step_index"],
        )
        for record in records
    }
    if len(shared) != 1:
        raise ContractError("prepare lanes must share run, round, and rollout index")
    actual_batch_id = batch_id or deterministic_batch_id("prepare", identities)
    require_identifier(actual_batch_id, "batch_id")
    header = {
        "contract_version": CONTRACT_VERSION_V2,
        "kind": "prepare_batch",
        "batch_id": actual_batch_id,
        "lanes": records,
    }
    body = pack_ndarray_batch(header, groups)
    return actual_batch_id, body, records, groups


def validate_prepare_batch_response(
    response: Mapping[str, Any],
    *,
    batch_id: str,
    request_records: Sequence[Mapping[str, Any]],
    local_prepares: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(response, Mapping) or set(response) != _PREPARE_TOP_KEYS:
        raise ContractError(f"prepare response keys must be exactly {sorted(_PREPARE_TOP_KEYS)}")
    if response["contract_version"] != CONTRACT_VERSION_V2:
        raise ContractError("prepare response contract version mismatch")
    if response["kind"] != "prepare_batch_response" or response["batch_id"] != batch_id:
        raise ContractError("prepare response kind or batch_id mismatch")
    lanes = response["lanes"]
    if not isinstance(lanes, list) or len(lanes) != len(request_records):
        raise ContractError("prepare response lane count mismatch")
    selections: list[dict[str, Any]] = []
    for raw, request, local in zip(lanes, request_records, local_prepares, strict=True):
        if not isinstance(raw, Mapping) or set(raw) != _SELECTION_LANE_KEYS:
            raise ContractError(
                f"v2 selection lane keys must be exactly {sorted(_SELECTION_LANE_KEYS)}"
            )
        if raw["contract_version"] != CONTRACT_VERSION_V2:
            raise ContractError("v2 selection contract version mismatch")
        for name in (
            "idempotency_key",
            "step_key",
            "prepare_receipt",
            "capture_digest",
            "wire_digest",
            "beta",
        ):
            if raw[name] != request[name]:
                raise ContractError(f"v2 selection {name} mismatch")
        key = StepKey.from_mapping(raw["step_key"])
        beta = validate_beta(key.round_id, raw["beta"])
        draw = raw["selection_draw"]
        if (
            isinstance(draw, bool)
            or not isinstance(draw, (int, float))
            or not math.isfinite(float(draw))
            or not 0 <= float(draw) < 1
        ):
            raise ContractError("selection_draw must be finite in [0,1)")
        expected_source = "teacher" if float(draw) < beta else "student"
        if raw["selected_source"] != expected_source:
            raise ContractError("selected_source does not match draw < beta")
        require_sha256(raw["selection_id"], "selection_id")
        student = decode_array(raw["student_action_11"])
        validate_array(
            "student_action_11",
            student,
            ArrayRule((STUDENT_ACTION_DIM,), "float32", True),
        )
        selected = decode_array(raw["selected_action"])
        validate_array("selected_action", selected, ArrayRule((ACTION_DIM,), "float32", True))
        local_arrays = _raw_local_arrays(local)
        teacher = local_arrays["teacher_action"]
        homie = local_arrays["homie_action"]
        validate_array("teacher_action", teacher, ArrayRule((ACTION_DIM,), "float32", True))
        validate_array(
            "homie_action", homie, ArrayRule((HOMIE_ACTION_DIM,), "float32", True)
        )
        if not np.array_equal(teacher[STUDENT_ACTION_DIM:], homie):
            raise ContractError("teacher_action tail does not equal RTX Homie action")
        expected_action = (
            teacher
            if expected_source == "teacher"
            else np.concatenate((student, homie)).astype(np.float32)
        )
        if not np.array_equal(selected, expected_action):
            raise ContractError("selected action is not teacher or student11+Homie15")
        require_sha256(raw["observation_digest"], "observation_digest")
        receipt = require_sha256(raw["selection_receipt"], "selection_receipt")
        receipt_body = {
            name: raw[name] for name in sorted(raw) if name != "selection_receipt"
        }
        if receipt != sha256_bytes(canonical_json(receipt_body)):
            raise ContractError("v2 selection receipt mismatch")
        selections.append({**dict(raw), "_selected_action_array": selected})
    return selections


def build_commit_batch(
    selections: Sequence[Mapping[str, Any]],
    local_commits: Sequence[Mapping[str, Any]],
    idempotency_keys: Sequence[str],
    *,
    batch_id: str | None = None,
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    if not selections or not (
        len(selections) == len(local_commits) == len(idempotency_keys)
    ):
        raise ContractError("commit batch lane inputs must be equally sized and non-empty")
    lanes: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for selection, local, idempotency_key in zip(
        selections, local_commits, idempotency_keys, strict=True
    ):
        require_identifier(idempotency_key, "idempotency_key")
        key = StepKey.from_mapping(selection.get("step_key"))
        if local.get("step_key") != key.as_dict():
            raise ContractError("local commit step_key does not match selection")
        for name in ("prepare_receipt", "selection_receipt"):
            require_sha256(selection.get(name), name)
        require_sha256(local.get("commit_receipt"), "commit_receipt")
        should_seal = key.rollout_step_index == 80
        if bool(local.get("sealed")) != should_seal:
            raise ContractError("local sealed flag must be true exactly at index 80")
        seal = local.get("seal") if should_seal else None
        if should_seal:
            if not isinstance(seal, Mapping):
                raise ContractError("final rollout step is missing sealed_window")
            seal = dict(seal)
            seal["contract_version"] = CONTRACT_VERSION_V2
            for name in ("run_id", "round_id", "actor_id", "rollout_window_id"):
                if seal.get(name) != key.as_dict()[name]:
                    raise ContractError(f"sealed_window {name} mismatch")
        lane = {
            "contract_version": CONTRACT_VERSION_V2,
            "idempotency_key": idempotency_key,
            "step_key": key.as_dict(),
            "prepare_receipt": selection["prepare_receipt"],
            "selection_id": selection["selection_id"],
            "selection_receipt": selection["selection_receipt"],
            "commit_receipt": local["commit_receipt"],
            "transition": local["result"],
            "sealed_window": seal,
        }
        if set(lane) != _COMMIT_LANE_KEYS:
            raise AssertionError("internal v2 commit lane key drift")
        lanes.append(lane)
        identities.append(
            {
                "idempotency_key": idempotency_key,
                "step_key": key.as_dict(),
                "commit_receipt": local["commit_receipt"],
            }
        )
    if len(set(idempotency_keys)) != len(idempotency_keys):
        raise ContractError("commit batch idempotency keys must be unique per lane")
    tokens = [StepKey.from_mapping(lane["step_key"]).token for lane in lanes]
    if len(set(tokens)) != len(tokens):
        raise ContractError("commit batch logical step identities must be unique")
    shared = {
        (
            lane["step_key"]["run_id"],
            lane["step_key"]["round_id"],
            lane["step_key"]["rollout_step_index"],
        )
        for lane in lanes
    }
    if len(shared) != 1:
        raise ContractError("commit lanes must share run, round, and rollout index")
    actual_batch_id = batch_id or deterministic_batch_id("commit", identities)
    require_identifier(actual_batch_id, "batch_id")
    payload = {
        "contract_version": CONTRACT_VERSION_V2,
        "kind": "commit_batch",
        "batch_id": actual_batch_id,
        "lanes": lanes,
    }
    return actual_batch_id, payload, lanes


def validate_commit_batch_response(
    response: Mapping[str, Any],
    *,
    batch_id: str,
    request_lanes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(response, Mapping) or set(response) != _PREPARE_TOP_KEYS:
        raise ContractError("commit response top-level keys mismatch")
    if response["contract_version"] != CONTRACT_VERSION_V2:
        raise ContractError("commit response contract version mismatch")
    if response["kind"] != "commit_batch_response" or response["batch_id"] != batch_id:
        raise ContractError("commit response kind or batch_id mismatch")
    rows = response["lanes"]
    if not isinstance(rows, list) or len(rows) != len(request_lanes):
        raise ContractError("commit response lane count mismatch")
    acknowledgements: list[dict[str, Any]] = []
    for row, request in zip(rows, request_lanes, strict=True):
        if not isinstance(row, Mapping) or set(row) != _COMMIT_ACK_LANE_KEYS:
            raise ContractError(
                f"commit ack lane keys must be exactly {sorted(_COMMIT_ACK_LANE_KEYS)}"
            )
        if row["contract_version"] != CONTRACT_VERSION_V2 or row["status"] != "committed":
            raise ContractError("commit acknowledgement is not committed v2")
        for name in ("idempotency_key", "step_key", "commit_receipt"):
            if row[name] != request[name]:
                raise ContractError(f"commit acknowledgement {name} mismatch")
        receipt = require_sha256(row["remote_commit_receipt"], "remote_commit_receipt")
        receipt_body = {
            name: row[name] for name in sorted(row) if name != "remote_commit_receipt"
        }
        if receipt != sha256_bytes(canonical_json(receipt_body)):
            raise ContractError("v2 remote_commit_receipt mismatch")
        acknowledgements.append(dict(row))
    return acknowledgements
