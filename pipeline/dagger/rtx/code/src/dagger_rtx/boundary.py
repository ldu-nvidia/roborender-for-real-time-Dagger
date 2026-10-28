from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .contracts import (
    ACTION_DIM,
    CONTRACT_VERSION,
    HOMIE_ACTION_DIM,
    STUDENT_ACTION_DIM,
    ArrayRule,
    ContractError,
    StepKey,
    canonical_json,
    decode_array,
    encode_array,
    require_identifier,
    require_sha256,
    sha256_bytes,
    validate_array,
    validate_beta,
)

H200_BOUNDARY_PORT = 18620
H200_PREPARE_PATH = "/v1/steps/prepare"
H200_COMMIT_PATH = "/v1/steps/commit"


def build_h200_prepare(local_prepare: Mapping[str, Any], idempotency_key: str) -> dict[str, Any]:
    key = StepKey.from_mapping(local_prepare.get("step_key"))
    require_identifier(idempotency_key, "idempotency_key")
    if local_prepare.get("contract_version") != CONTRACT_VERSION:
        raise ContractError("local prepare contract version mismatch")
    metadata = local_prepare.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ContractError("local prepare metadata must be an object")
    if metadata.get("step_key") != key.as_dict():
        raise ContractError("local prepare metadata step_key mismatch")
    validate_beta(key.round_id, metadata.get("beta"))
    assignment = metadata.get("episode_assignment")
    if not isinstance(assignment, Mapping):
        raise ContractError("local prepare episode_assignment must be an object")
    if assignment.get("observation_source") != metadata.get("observation_source"):
        raise ContractError("metadata and episode assignment observation source mismatch")
    expected_raytraced = metadata.get("observation_source") == "raytraced"
    if assignment.get("use_raytraced") is not expected_raytraced:
        raise ContractError("use_raytraced does not match observation_source")
    require_sha256(local_prepare.get("prepare_receipt"), "prepare_receipt")
    require_sha256(local_prepare.get("capture_digest"), "capture_digest")
    return {
        "contract_version": CONTRACT_VERSION,
        "idempotency_key": idempotency_key,
        "step_key": key.as_dict(),
        "prepare_receipt": local_prepare["prepare_receipt"],
        "capture_digest": local_prepare["capture_digest"],
        "beta": metadata["beta"],
        "metadata": metadata,
        "arrays": local_prepare["arrays"],
    }


def selection_receipt_body(response: Mapping[str, Any]) -> dict[str, Any]:
    return {name: response[name] for name in sorted(response) if name != "selection_receipt"}


def validate_h200_selection(
    response: Mapping[str, Any],
    local_prepare: Mapping[str, Any],
    expected_idempotency_key: str | None = None,
) -> dict[str, Any]:
    expected_keys = {
        "contract_version", "idempotency_key", "step_key", "prepare_receipt", "capture_digest",
        "beta", "selection_id", "selection_draw", "selected_source", "student_action_11",
        "selected_action", "observation_digest", "selection_receipt",
    }
    if not isinstance(response, Mapping) or set(response) != expected_keys:
        raise ContractError(f"H200 selection keys must be exactly {sorted(expected_keys)}")
    if response["contract_version"] != CONTRACT_VERSION:
        raise ContractError("H200 selection contract version mismatch")
    key = StepKey.from_mapping(response["step_key"])
    if key.as_dict() != local_prepare["step_key"]:
        raise ContractError("H200 selection step key mismatch")
    for name in ("prepare_receipt", "capture_digest"):
        if response[name] != local_prepare[name]:
            raise ContractError(f"H200 selection {name} mismatch")
    require_identifier(response["idempotency_key"], "idempotency_key")
    if expected_idempotency_key is not None and response["idempotency_key"] != expected_idempotency_key:
        raise ContractError("H200 selection idempotency_key mismatch")
    require_identifier(response["selection_id"], "selection_id")
    beta = validate_beta(key.round_id, response["beta"])
    draw = response["selection_draw"]
    if isinstance(draw, bool) or not isinstance(draw, (int, float)) or not math.isfinite(float(draw)) or not 0 <= float(draw) < 1:
        raise ContractError("selection_draw must be finite in [0,1)")
    expected_source = "teacher" if float(draw) < beta else "student"
    if response["selected_source"] != expected_source:
        raise ContractError("selected_source does not match Raven draw < beta semantics")
    student11 = decode_array(response["student_action_11"])
    validate_array("student_action_11", student11, ArrayRule((STUDENT_ACTION_DIM,), "float32", True))
    selected = decode_array(response["selected_action"])
    validate_array("selected_action", selected, ArrayRule((ACTION_DIM,), "float32", True))
    teacher = decode_array(local_prepare["arrays"]["teacher_action"])
    homie = decode_array(local_prepare["arrays"]["homie_action"])
    validate_array("homie_action", homie, ArrayRule((HOMIE_ACTION_DIM,), "float32", True))
    if not np.array_equal(teacher[STUDENT_ACTION_DIM:], homie):
        raise ContractError("RTX teacher_action tail does not equal homie_action")
    expected_action = teacher if expected_source == "teacher" else np.concatenate((student11, homie)).astype(np.float32)
    if not np.array_equal(selected, expected_action):
        raise ContractError("H200 selected_action is not the exact teacher or student11+RTX-Homie action")
    require_sha256(response["observation_digest"], "observation_digest")
    require_sha256(response["selection_receipt"], "selection_receipt")
    expected_receipt = sha256_bytes(canonical_json(selection_receipt_body(response)))
    if response["selection_receipt"] != expected_receipt:
        raise ContractError("H200 selection receipt mismatch")
    return {**dict(response), "_selected_action_array": selected}


def build_owner_commit(selection: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "step_key": selection["step_key"],
        "prepare_receipt": selection["prepare_receipt"],
        "selection_id": selection["selection_id"],
        "selected_source": selection["selected_source"],
        "beta": selection["beta"],
        "selected_action": selection["selected_action"],
    }


def build_h200_commit(
    selection: Mapping[str, Any], local_commit: Mapping[str, Any], idempotency_key: str
) -> dict[str, Any]:
    require_identifier(idempotency_key, "idempotency_key")
    key = StepKey.from_mapping(selection.get("step_key"))
    if local_commit.get("step_key") != key.as_dict():
        raise ContractError("local commit step_key does not match H200 selection")
    require_sha256(selection.get("prepare_receipt"), "prepare_receipt")
    require_sha256(selection.get("selection_receipt"), "selection_receipt")
    require_sha256(local_commit.get("commit_receipt"), "commit_receipt")
    should_seal = key.rollout_step_index == 80
    if bool(local_commit.get("sealed")) != should_seal:
        raise ContractError("local sealed flag must be true exactly at rollout_step_index 80")
    seal = local_commit.get("seal") if should_seal else None
    if should_seal:
        if not isinstance(seal, Mapping):
            raise ContractError("final rollout step is missing sealed_window")
        for name in ("run_id", "round_id", "actor_id", "rollout_window_id"):
            expected = key.as_dict()[name]
            if seal.get(name) != expected:
                raise ContractError(f"sealed_window {name} does not match final step")
    return {
        "contract_version": CONTRACT_VERSION,
        "idempotency_key": idempotency_key,
        "step_key": selection["step_key"],
        "prepare_receipt": selection["prepare_receipt"],
        "selection_id": selection["selection_id"],
        "selection_receipt": selection["selection_receipt"],
        "commit_receipt": local_commit["commit_receipt"],
        "transition": local_commit["result"],
        "sealed_window": seal,
    }


def validate_h200_commit_ack(ack: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "contract_version", "idempotency_key", "step_key", "commit_receipt", "remote_commit_receipt", "status"
    }
    if not isinstance(ack, Mapping) or set(ack) != expected:
        raise ContractError(f"H200 commit ack keys must be exactly {sorted(expected)}")
    if ack["contract_version"] != CONTRACT_VERSION or ack["status"] != "committed":
        raise ContractError("H200 commit acknowledgement is not committed")
    for name in ("idempotency_key", "step_key", "commit_receipt"):
        if ack[name] != request[name]:
            raise ContractError(f"H200 commit acknowledgement {name} mismatch")
    receipt = require_sha256(ack["remote_commit_receipt"], "remote_commit_receipt")
    body = {name: ack[name] for name in sorted(ack) if name != "remote_commit_receipt"}
    if receipt != sha256_bytes(canonical_json(body)):
        raise ContractError("remote_commit_receipt digest mismatch")
    return dict(ack)
