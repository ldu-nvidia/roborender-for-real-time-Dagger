"""Strict RTX builders and validators for pipelined DAgger v4.

The scientific payload is unchanged from v3.  V4 changes only the transport
schedule: after the initial prepare, each exchange atomically acknowledges the
previous transition and prepares the current action.  A final commit-only flush
closes step 80.
"""

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
    STUDENT_ACTION_DIM,
    ArrayRule,
    ContractError,
    RavenShapeProfile,
    StepKey,
    canonical_json,
    decode_array,
    payload_digest,
    require_identifier,
    require_sha256,
    sha256_bytes,
    validate_array,
    validate_beta,
)
from .lean_boundary_v3 import project_minimal_arrays_v3
from .ndarray_batch_v4 import pack_ndarray_batch_v4


CONTRACT_VERSION_V4 = "roborender.dagger.rtx.v4"
H200_PREPARE_BATCH_PATH_V4 = "/v4/steps/prepare-batch"
H200_EXCHANGE_BATCH_PATH_V4 = "/v4/steps/exchange-batch"
H200_FINALIZE_BATCH_PATH_V4 = "/v4/steps/finalize-batch"

_LOCAL_METADATA_KEYS = CAPTURE_METADATA_KEYS | {
    "contract_version",
    "step_key",
    "beta",
    "episode_assignment",
}
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
_PREPARE_RESPONSE_KEYS = frozenset(
    {"contract_version", "kind", "batch_id", "gateway_instance_id", "lanes"}
)
_EXCHANGE_RESPONSE_KEYS = frozenset(
    {
        "contract_version",
        "kind",
        "batch_id",
        "gateway_instance_id",
        "previous_commit_acks",
        "lanes",
    }
)
_FINALIZE_RESPONSE_KEYS = _PREPARE_RESPONSE_KEYS


def lightweight_capture_digest_v4(
    step_key: Mapping[str, Any] | StepKey,
    wire_digest: str,
) -> str:
    """Bind the v4 step identity to exactly the tensors consumed by H200.

    V4 intentionally does not hash RTX-only metric/raw-depth tensors.  Those
    tensors never cross the v4 boundary and cannot affect Image3F, student
    inference, beta selection, or learner data.  ``wire_digest`` already binds
    canonical v4 metadata and every source-specific lean ndarray byte.
    """

    key = step_key if isinstance(step_key, StepKey) else StepKey.from_mapping(step_key)
    return sha256_bytes(
        canonical_json(
            {
                "contract_version": CONTRACT_VERSION_V4,
                "step_key": key.as_dict(),
                "wire_digest": require_sha256(wire_digest, "wire_digest"),
            }
        )
    )


def deterministic_batch_id_v4(kind: str, identities: Sequence[Any]) -> str:
    require_identifier(kind, "batch kind", 32)
    digest = sha256_bytes(canonical_json({"kind": kind, "identities": list(identities)}))
    return f"{kind}-{digest[:48]}"


def _gateway_instance(value: Any) -> str:
    return require_identifier(value, "gateway_instance_id", 160)


def _raw_arrays(local_prepare: Mapping[str, Any]) -> Mapping[str, np.ndarray]:
    arrays = local_prepare.get("arrays")
    if not isinstance(arrays, Mapping) or not arrays:
        raise ContractError("v4 local prepare must carry raw numpy arrays")
    if any(not isinstance(value, np.ndarray) for value in arrays.values()):
        raise ContractError("v4 local arrays must not be JSON/base64 envelopes")
    return arrays


def _metadata_v4(local_prepare: Mapping[str, Any]) -> dict[str, Any]:
    metadata = local_prepare.get("metadata")
    if not isinstance(metadata, Mapping) or set(metadata) != _LOCAL_METADATA_KEYS:
        raise ContractError(f"local metadata keys must be exactly {sorted(_LOCAL_METADATA_KEYS)}")
    if metadata.get("contract_version") != CONTRACT_VERSION:
        raise ContractError("source capture metadata must use the frozen local v1 contract")
    value = dict(metadata)
    value["contract_version"] = CONTRACT_VERSION_V4
    if tuple(value.get("view_order", ())) != RAVEN_VIEW_ORDER:
        raise ContractError("v4 metadata view_order mismatch")
    if value.get("model_view_map") != MODEL_VIEW_MAP:
        raise ContractError("v4 metadata model_view_map mismatch")
    assignment = value.get("episode_assignment")
    if not isinstance(assignment, Mapping):
        raise ContractError("v4 metadata requires episode_assignment")
    source = value.get("observation_source")
    if source not in {"generated_controls", "raytraced"}:
        raise ContractError("v4 observation_source is invalid")
    if assignment.get("observation_source") != source:
        raise ContractError("v4 assignment observation_source mismatch")
    if assignment.get("use_raytraced") is not (source == "raytraced"):
        raise ContractError("v4 assignment use_raytraced mismatch")
    require_sha256(assignment.get("anchor_sha256"), "anchor_sha256")
    return value


def _prepare_records_v4(
    local_prepares: Sequence[Mapping[str, Any]],
    idempotency_keys: Sequence[str],
    *,
    shape_profile: RavenShapeProfile | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, np.ndarray]]]:
    if not local_prepares or len(local_prepares) != len(idempotency_keys):
        raise ContractError("v4 prepare inputs must be equally sized and non-empty")
    records: list[dict[str, Any]] = []
    groups: list[dict[str, np.ndarray]] = []
    for local, idempotency_key in zip(local_prepares, idempotency_keys, strict=True):
        if local.get("contract_version") != CONTRACT_VERSION:
            raise ContractError("local prepare contract version mismatch")
        key = StepKey.from_mapping(local.get("step_key"))
        require_identifier(idempotency_key, "idempotency_key")
        prepare_receipt = require_sha256(local.get("prepare_receipt"), "prepare_receipt")
        capture_digest = require_sha256(local.get("capture_digest"), "capture_digest")
        metadata = _metadata_v4(local)
        if metadata.get("step_key") != key.as_dict():
            raise ContractError("prepare metadata step_key mismatch")
        beta = validate_beta(key.round_id, metadata.get("beta"))
        lean = project_minimal_arrays_v3(local, shape_profile=shape_profile)
        local_arrays = _raw_arrays(local)
        homie = local_arrays.get("homie_action")
        validate_array(
            "homie_action", homie, ArrayRule((HOMIE_ACTION_DIM,), "float32", True)
        )
        if not np.array_equal(lean["teacher_action"][STUDENT_ACTION_DIM:], homie):
            raise ContractError("teacher_action[11:] must equal local homie_action")
        # The native v4 owner computes this once, before the full capture is
        # released.  RDB4 separately attests every descriptor and the complete
        # raw blob, while H200 recomputes this semantic digest.  Re-hashing the
        # same ~27 MB lean B=28 payload here would add no integrity coverage.
        wire_digest = require_sha256(local.get("wire_digest"), "local wire_digest")
        if capture_digest != lightweight_capture_digest_v4(key, wire_digest):
            raise ContractError("v4 lightweight capture_digest preimage mismatch")
        records.append(
            {
                "contract_version": CONTRACT_VERSION_V4,
                "idempotency_key": idempotency_key,
                "step_key": key.as_dict(),
                "prepare_receipt": prepare_receipt,
                "capture_digest": capture_digest,
                "wire_digest": wire_digest,
                "beta": beta,
                "metadata": metadata,
            }
        )
        groups.append(lean)
    if len(set(idempotency_keys)) != len(idempotency_keys):
        raise ContractError("v4 prepare idempotency keys must be unique")
    _validate_shared_step([record["step_key"] for record in records], "prepare")
    return records, groups


def _validate_shared_step(step_keys: Sequence[Mapping[str, Any]], label: str) -> None:
    keys = [StepKey.from_mapping(value) for value in step_keys]
    if len({key.token for key in keys}) != len(keys):
        raise ContractError(f"v4 {label} logical step identities must be unique")
    shared = {(key.run_id, key.round_id, key.rollout_step_index) for key in keys}
    if len(shared) != 1:
        raise ContractError(f"v4 {label} lanes must share run, round, and rollout index")


def build_prepare_batch_v4(
    local_prepares: Sequence[Mapping[str, Any]],
    idempotency_keys: Sequence[str],
    *,
    gateway_instance_id: str,
    shape_profile: RavenShapeProfile | None = None,
    batch_id: str | None = None,
    scratch: bytearray | None = None,
) -> tuple[str, bytes, list[dict[str, Any]]]:
    instance = _gateway_instance(gateway_instance_id)
    records, groups = _prepare_records_v4(
        local_prepares, idempotency_keys, shape_profile=shape_profile
    )
    identities = [
        {
            "idempotency_key": row["idempotency_key"],
            "step_key": row["step_key"],
            "wire_digest": row["wire_digest"],
        }
        for row in records
    ]
    actual_id = batch_id or deterministic_batch_id_v4("prepare", identities)
    header = {
        "contract_version": CONTRACT_VERSION_V4,
        "kind": "prepare_batch",
        "batch_id": require_identifier(actual_id, "batch_id"),
        "gateway_instance_id": instance,
        "lanes": records,
    }
    return actual_id, pack_ndarray_batch_v4(header, groups, scratch=scratch), records


def _validate_selection_rows_v4(
    rows: Any,
    *,
    request_records: Sequence[Mapping[str, Any]],
    local_prepares: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != len(request_records):
        raise ContractError("v4 selection lane count mismatch")
    selections: list[dict[str, Any]] = []
    for raw, request, local in zip(rows, request_records, local_prepares, strict=True):
        if not isinstance(raw, Mapping) or set(raw) != _SELECTION_LANE_KEYS:
            raise ContractError(
                f"v4 selection lane keys must be exactly {sorted(_SELECTION_LANE_KEYS)}"
            )
        if raw["contract_version"] != CONTRACT_VERSION_V4:
            raise ContractError("v4 selection contract version mismatch")
        for name in (
            "idempotency_key",
            "step_key",
            "prepare_receipt",
            "capture_digest",
            "wire_digest",
            "beta",
        ):
            if raw[name] != request[name]:
                raise ContractError(f"v4 selection {name} mismatch")
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
        validate_array(
            "selected_action", selected, ArrayRule((ACTION_DIM,), "float32", True)
        )
        arrays = _raw_arrays(local)
        teacher = arrays["teacher_action"]
        homie = arrays["homie_action"]
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
            raise ContractError("v4 selection_receipt mismatch")
        selections.append({**dict(raw), "_selected_action_array": selected})
    return selections


def validate_prepare_batch_response_v4(
    response: Mapping[str, Any],
    *,
    batch_id: str,
    gateway_instance_id: str,
    request_records: Sequence[Mapping[str, Any]],
    local_prepares: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(response, Mapping) or set(response) != _PREPARE_RESPONSE_KEYS:
        raise ContractError("v4 prepare response top-level keys mismatch")
    if (
        response["contract_version"] != CONTRACT_VERSION_V4
        or response["kind"] != "prepare_batch_response"
        or response["batch_id"] != batch_id
        or response["gateway_instance_id"] != _gateway_instance(gateway_instance_id)
    ):
        raise ContractError("v4 prepare response identity mismatch")
    return _validate_selection_rows_v4(
        response["lanes"],
        request_records=request_records,
        local_prepares=local_prepares,
    )


def _commit_lanes_v4(
    selections: Sequence[Mapping[str, Any]],
    local_commits: Sequence[Mapping[str, Any]],
    idempotency_keys: Sequence[str],
    *,
    final: bool,
) -> list[dict[str, Any]]:
    if not selections or not (
        len(selections) == len(local_commits) == len(idempotency_keys)
    ):
        raise ContractError("v4 commit inputs must be equally sized and non-empty")
    lanes: list[dict[str, Any]] = []
    for selection, local, idem in zip(
        selections, local_commits, idempotency_keys, strict=True
    ):
        require_identifier(idem, "idempotency_key")
        key = StepKey.from_mapping(selection.get("step_key"))
        if local.get("step_key") != key.as_dict():
            raise ContractError("local commit step_key does not match selection")
        for name in ("prepare_receipt", "selection_receipt"):
            require_sha256(selection.get(name), name)
        commit_receipt = require_sha256(local.get("commit_receipt"), "commit_receipt")
        if final != (key.rollout_step_index == 80):
            raise ContractError("v4 final commit must be exactly rollout index 80")
        if bool(local.get("sealed")) != final:
            raise ContractError("local sealed flag does not match v4 commit phase")
        seal = local.get("seal") if final else None
        if final:
            if not isinstance(seal, Mapping):
                raise ContractError("final v4 commit is missing sealed_window")
            seal = dict(seal)
            seal["contract_version"] = CONTRACT_VERSION_V4
            for name in ("run_id", "round_id", "actor_id", "rollout_window_id"):
                if seal.get(name) != key.as_dict()[name]:
                    raise ContractError(f"sealed_window {name} mismatch")
        lane = {
            "contract_version": CONTRACT_VERSION_V4,
            "idempotency_key": idem,
            "step_key": key.as_dict(),
            "prepare_receipt": selection["prepare_receipt"],
            "selection_id": selection["selection_id"],
            "selection_receipt": selection["selection_receipt"],
            "commit_receipt": commit_receipt,
            "transition": local["result"],
            "sealed_window": seal,
        }
        if set(lane) != _COMMIT_LANE_KEYS:
            raise AssertionError("internal v4 commit lane key drift")
        lanes.append(lane)
    if len(set(idempotency_keys)) != len(idempotency_keys):
        raise ContractError("v4 commit idempotency keys must be unique")
    _validate_shared_step([lane["step_key"] for lane in lanes], "commit")
    return lanes


def _validate_exchange_continuity(
    commits: Sequence[Mapping[str, Any]], prepares: Sequence[Mapping[str, Any]]
) -> None:
    if len(commits) != len(prepares):
        raise ContractError("v4 exchange commit/prepare cardinality mismatch")
    for lane, (commit, prepare) in enumerate(zip(commits, prepares, strict=True)):
        previous = StepKey.from_mapping(commit["step_key"])
        current = StepKey.from_mapping(prepare["step_key"])
        if (
            previous.run_id,
            previous.round_id,
            previous.actor_id,
            previous.rollout_window_id,
        ) != (
            current.run_id,
            current.round_id,
            current.actor_id,
            current.rollout_window_id,
        ) or current.rollout_step_index != previous.rollout_step_index + 1:
            raise ContractError(f"v4 exchange lane {lane} route/step continuity mismatch")
        transition = commit.get("transition")
        if not isinstance(transition, Mapping):
            raise ContractError("v4 exchange commit transition must be an object")
        ended = bool(transition.get("done")) or bool(transition.get("timeout"))
        if ended:
            if current.episode_id == previous.episode_id or current.episode_step_index != 0:
                raise ContractError(f"v4 exchange lane {lane} did not reset after done/timeout")
        elif (
            current.episode_id != previous.episode_id
            or current.episode_step_index != previous.episode_step_index + 1
        ):
            raise ContractError(f"v4 exchange lane {lane} episode continuity mismatch")


def build_exchange_batch_v4(
    previous_selections: Sequence[Mapping[str, Any]],
    previous_local_commits: Sequence[Mapping[str, Any]],
    previous_commit_idempotency_keys: Sequence[str],
    current_local_prepares: Sequence[Mapping[str, Any]],
    current_prepare_idempotency_keys: Sequence[str],
    *,
    gateway_instance_id: str,
    shape_profile: RavenShapeProfile | None = None,
    batch_id: str | None = None,
    scratch: bytearray | None = None,
) -> tuple[str, bytes, list[dict[str, Any]], list[dict[str, Any]]]:
    commits = _commit_lanes_v4(
        previous_selections,
        previous_local_commits,
        previous_commit_idempotency_keys,
        final=False,
    )
    prepares, groups = _prepare_records_v4(
        current_local_prepares,
        current_prepare_idempotency_keys,
        shape_profile=shape_profile,
    )
    _validate_exchange_continuity(commits, prepares)
    identities = [
        {
            "previous_commit_idempotency_key": commit["idempotency_key"],
            "previous_commit_receipt": commit["commit_receipt"],
            "current_prepare_idempotency_key": prepare["idempotency_key"],
            "current_wire_digest": prepare["wire_digest"],
        }
        for commit, prepare in zip(commits, prepares, strict=True)
    ]
    actual_id = batch_id or deterministic_batch_id_v4("exchange", identities)
    header = {
        "contract_version": CONTRACT_VERSION_V4,
        "kind": "exchange_batch",
        "batch_id": require_identifier(actual_id, "batch_id"),
        "gateway_instance_id": _gateway_instance(gateway_instance_id),
        "previous_commits": commits,
        "lanes": prepares,
    }
    body = pack_ndarray_batch_v4(header, groups, scratch=scratch)
    return actual_id, body, commits, prepares


def _validate_commit_acks_v4(
    rows: Any, request_lanes: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or len(rows) != len(request_lanes):
        raise ContractError("v4 commit acknowledgement lane count mismatch")
    acknowledgements: list[dict[str, Any]] = []
    for row, request in zip(rows, request_lanes, strict=True):
        if not isinstance(row, Mapping) or set(row) != _COMMIT_ACK_LANE_KEYS:
            raise ContractError(
                f"v4 commit ack keys must be exactly {sorted(_COMMIT_ACK_LANE_KEYS)}"
            )
        if row["contract_version"] != CONTRACT_VERSION_V4 or row["status"] != "committed":
            raise ContractError("commit acknowledgement is not committed v4")
        for name in ("idempotency_key", "step_key", "commit_receipt"):
            if row[name] != request[name]:
                raise ContractError(f"v4 commit acknowledgement {name} mismatch")
        receipt = require_sha256(row["remote_commit_receipt"], "remote_commit_receipt")
        body = {name: row[name] for name in sorted(row) if name != "remote_commit_receipt"}
        if receipt != sha256_bytes(canonical_json(body)):
            raise ContractError("v4 remote_commit_receipt mismatch")
        acknowledgements.append(dict(row))
    return acknowledgements


def validate_exchange_batch_response_v4(
    response: Mapping[str, Any],
    *,
    batch_id: str,
    gateway_instance_id: str,
    previous_commit_lanes: Sequence[Mapping[str, Any]],
    current_prepare_records: Sequence[Mapping[str, Any]],
    current_local_prepares: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(response, Mapping) or set(response) != _EXCHANGE_RESPONSE_KEYS:
        raise ContractError("v4 exchange response top-level keys mismatch")
    if (
        response["contract_version"] != CONTRACT_VERSION_V4
        or response["kind"] != "exchange_batch_response"
        or response["batch_id"] != batch_id
        or response["gateway_instance_id"] != _gateway_instance(gateway_instance_id)
    ):
        raise ContractError("v4 exchange response identity mismatch")
    acknowledgements = _validate_commit_acks_v4(
        response["previous_commit_acks"], previous_commit_lanes
    )
    selections = _validate_selection_rows_v4(
        response["lanes"],
        request_records=current_prepare_records,
        local_prepares=current_local_prepares,
    )
    return acknowledgements, selections


def build_finalize_batch_v4(
    selections: Sequence[Mapping[str, Any]],
    local_commits: Sequence[Mapping[str, Any]],
    idempotency_keys: Sequence[str],
    *,
    gateway_instance_id: str,
    batch_id: str | None = None,
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    lanes = _commit_lanes_v4(selections, local_commits, idempotency_keys, final=True)
    identities = [
        {
            "idempotency_key": row["idempotency_key"],
            "step_key": row["step_key"],
            "commit_receipt": row["commit_receipt"],
        }
        for row in lanes
    ]
    actual_id = batch_id or deterministic_batch_id_v4("finalize", identities)
    payload = {
        "contract_version": CONTRACT_VERSION_V4,
        "kind": "finalize_batch",
        "batch_id": require_identifier(actual_id, "batch_id"),
        "gateway_instance_id": _gateway_instance(gateway_instance_id),
        "lanes": lanes,
    }
    return actual_id, payload, lanes


def validate_finalize_batch_response_v4(
    response: Mapping[str, Any],
    *,
    batch_id: str,
    gateway_instance_id: str,
    request_lanes: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(response, Mapping) or set(response) != _FINALIZE_RESPONSE_KEYS:
        raise ContractError("v4 finalize response top-level keys mismatch")
    if (
        response["contract_version"] != CONTRACT_VERSION_V4
        or response["kind"] != "finalize_batch_response"
        or response["batch_id"] != batch_id
        or response["gateway_instance_id"] != _gateway_instance(gateway_instance_id)
    ):
        raise ContractError("v4 finalize response identity mismatch")
    return _validate_commit_acks_v4(response["lanes"], request_lanes)
