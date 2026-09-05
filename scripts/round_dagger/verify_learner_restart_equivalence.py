#!/usr/bin/env python3
"""Fail closed unless interrupted and uninterrupted learner states are identical."""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from round_dagger.artifacts import (  # noqa: E402
    load_strict_json,
    publish_immutable_json,
    sha256_file,
)
from round_dagger.checkpoint_compat import (  # noqa: E402
    CheckpointCompatibilityError,
    load_restricted_restart_document,
)
from round_dagger.learner import (  # noqa: E402
    CHECKPOINT_SCHEMA,
    LEGACY_RESULT_SCHEMA,
    ORDERED_ALLGATHER_GRADIENT_REDUCTION,
    RESULT_SCHEMA,
    SINGLE_PROCESS_GRADIENT_REDUCTION,
)
from round_dagger.policy_bundle import (  # noqa: E402
    PolicyBundleError,
    validate_policy_bundle_identity,
)
from round_dagger.policy_identity import (  # noqa: E402
    PolicyIdentityError,
    validate_homie_controller_sha256,
)


REPORT_SCHEMA = "fast_roborender.round_dagger.restart_equivalence.v1"
_CHECKPOINT_FIELDS = {
    "schema",
    "checkpoint_schema",
    "policy_state_dict",
    "policy_parameter_names",
    "homie_controller_sha256",
    "campaign_id",
    "macro_round",
    "model_lineage_sha256",
    "code_sha256",
    "learner_config_sha256",
    "learner_identity_sha256",
    "runtime_container_sha256",
    "learner_contract_sha256",
    "replay_snapshot_sha256",
    "replay_snapshot_file_sha256",
    "completed_steps",
    "initial_loss",
    "world_size",
    "gradient_reduction",
    "model_state_dict",
    "optimizer_state_dict",
    "lr_scheduler_state_dict",
    "value_state_dict",
    "rng_states",
}
_LEGACY_RESULT_FIELDS = {
    "schema",
    "campaign_id",
    "macro_round",
    "checkpoint_path",
    "checkpoint_sha256",
    "runtime_policy_sha256",
    "policy_spec_path",
    "policy_spec_sha256",
    "policy_bundle_manifest_path",
    "policy_bundle_manifest_sha256",
    "model_lineage_sha256",
    "replay_snapshot_sha256",
    "code_sha256",
    "learner_config_sha256",
    "optimizer_steps",
    "initial_loss",
    "final_loss",
}
_RESULT_FIELDS = _LEGACY_RESULT_FIELDS | {
    "learner_identity_sha256",
    "runtime_container_sha256",
    "learner_contract_sha256",
}
_LEGACY_RESULT_EQUIVALENCE_FIELDS = {
    "campaign_id",
    "macro_round",
    "model_lineage_sha256",
    "replay_snapshot_sha256",
    "code_sha256",
    "learner_config_sha256",
    "optimizer_steps",
    "initial_loss",
    "final_loss",
    "runtime_policy_sha256",
    "policy_spec_sha256",
    "policy_bundle_manifest_sha256",
}
_RESULT_EQUIVALENCE_FIELDS = _LEGACY_RESULT_EQUIVALENCE_FIELDS | {
    "learner_identity_sha256",
    "runtime_container_sha256",
    "learner_contract_sha256",
}


class RestartEquivalenceError(RuntimeError):
    """A learner checkpoint or result violated restart equivalence."""


def _absolute_regular_file(value: Path, label: str) -> Path:
    path = Path(os.path.abspath(value.expanduser()))
    if path.is_symlink() or not path.is_file():
        raise RestartEquivalenceError(f"{label} must be a regular non-symlink file")
    return path.resolve(strict=True)


def _load_checkpoint(path: Path, label: str) -> tuple[Path, Mapping[str, Any]]:
    path = _absolute_regular_file(path, label)
    try:
        document = load_restricted_restart_document(path, map_location="cpu")
    except CheckpointCompatibilityError as exc:
        raise RestartEquivalenceError(f"cannot load {label}: {exc}") from exc
    if not isinstance(document, Mapping) or set(document) != _CHECKPOINT_FIELDS:
        raise RestartEquivalenceError(f"{label} has unexpected or missing fields")
    if document.get("schema") != CHECKPOINT_SCHEMA:
        raise RestartEquivalenceError(f"{label} schema mismatch")
    if document.get("checkpoint_schema") != "gr00t.trl.full_state.v2":
        raise RestartEquivalenceError(f"{label} actor checkpoint schema mismatch")
    try:
        validate_homie_controller_sha256(
            document.get("homie_controller_sha256")
        )
    except PolicyIdentityError as exc:
        raise RestartEquivalenceError(
            f"{label} Homie controller identity is malformed: {exc}"
        ) from exc
    initial_loss = document.get("initial_loss")
    if (
        isinstance(initial_loss, bool)
        or not isinstance(initial_loss, (int, float))
        or not math.isfinite(float(initial_loss))
        or float(initial_loss) < 0.0
    ):
        raise RestartEquivalenceError(f"{label} initial_loss is invalid")
    world_size = document.get("world_size")
    if (
        isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size < 1
    ):
        raise RestartEquivalenceError(f"{label} world_size is invalid")
    expected_reduction = (
        SINGLE_PROCESS_GRADIENT_REDUCTION
        if world_size == 1
        else ORDERED_ALLGATHER_GRADIENT_REDUCTION
    )
    if document.get("gradient_reduction") != expected_reduction:
        raise RestartEquivalenceError(
            f"{label} gradient_reduction is invalid for world_size"
        )
    return path, document


def _assert_equal(left: Any, right: Any, location: str) -> None:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise RestartEquivalenceError(f"{location} tensor type differs")
        if (
            left.dtype != right.dtype
            or tuple(left.shape) != tuple(right.shape)
            or not torch.equal(left, right)
        ):
            raise RestartEquivalenceError(f"{location} tensor bytes differ")
        return
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            raise RestartEquivalenceError(f"{location} array type differs")
        if (
            left.dtype != right.dtype
            or left.shape != right.shape
            or not np.array_equal(left, right, equal_nan=True)
        ):
            raise RestartEquivalenceError(f"{location} array bytes differ")
        return
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            raise RestartEquivalenceError(f"{location} mapping type differs")
        if set(left) != set(right):
            raise RestartEquivalenceError(f"{location} mapping keys differ")
        for key in sorted(left, key=lambda item: (type(item).__name__, repr(item))):
            _assert_equal(left[key], right[key], f"{location}[{key!r}]")
        return
    if (
        isinstance(left, Sequence)
        and not isinstance(left, (str, bytes, bytearray))
    ) or (
        isinstance(right, Sequence)
        and not isinstance(right, (str, bytes, bytearray))
    ):
        if (
            not isinstance(left, Sequence)
            or isinstance(left, (str, bytes, bytearray))
            or not isinstance(right, Sequence)
            or isinstance(right, (str, bytes, bytearray))
            or len(left) != len(right)
        ):
            raise RestartEquivalenceError(f"{location} sequence differs")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            _assert_equal(left_item, right_item, f"{location}[{index}]")
        return
    if type(left) is not type(right) or left != right:
        raise RestartEquivalenceError(f"{location} differs")


def _validate_internal_policy_aliases(document: Mapping[str, Any], label: str) -> None:
    parameter_names = document["policy_parameter_names"]
    if (
        not isinstance(parameter_names, list)
        or any(not isinstance(item, str) for item in parameter_names)
        or parameter_names != sorted(set(parameter_names))
    ):
        raise RestartEquivalenceError(f"{label} parameter names are malformed")
    policy_state = document["policy_state_dict"]
    model_state = document["model_state_dict"]
    if not isinstance(policy_state, Mapping) or not isinstance(model_state, Mapping):
        raise RestartEquivalenceError(f"{label} policy state is malformed")
    _assert_equal(policy_state, model_state, f"{label}.policy/model_state")


def _load_result(path: Path, label: str) -> tuple[Path, Mapping[str, Any]]:
    path = _absolute_regular_file(path, label)
    document = load_strict_json(path)
    if not isinstance(document, Mapping):
        raise RestartEquivalenceError(f"{label} has unexpected or missing fields")
    expected_fields = (
        _RESULT_FIELDS
        if document.get("schema") == RESULT_SCHEMA
        else _LEGACY_RESULT_FIELDS
        if document.get("schema") == LEGACY_RESULT_SCHEMA
        else None
    )
    if expected_fields is None:
        raise RestartEquivalenceError(f"{label} schema mismatch")
    if set(document) != expected_fields:
        raise RestartEquivalenceError(f"{label} has unexpected or missing fields")
    checkpoint_name = document["checkpoint_path"]
    if (
        not isinstance(checkpoint_name, str)
        or not checkpoint_name
        or Path(checkpoint_name).name != checkpoint_name
    ):
        raise RestartEquivalenceError(f"{label} checkpoint path is unsafe")
    checkpoint = _absolute_regular_file(path.parent / checkpoint_name, f"{label} candidate")
    if sha256_file(checkpoint) != document["checkpoint_sha256"]:
        raise RestartEquivalenceError(f"{label} candidate digest mismatch")
    if document["schema"] == RESULT_SCHEMA:
        _, candidate_document = _load_checkpoint(
            checkpoint, f"{label} candidate"
        )
        for field in (
            "learner_identity_sha256",
            "runtime_container_sha256",
            "learner_contract_sha256",
        ):
            if document[field] != candidate_document[field]:
                raise RestartEquivalenceError(
                    f"{label} candidate/result {field} differs"
                )
    spec = _absolute_regular_file(
        path.parent / document["policy_spec_path"], f"{label} policy spec"
    )
    bundle_manifest = _absolute_regular_file(
        path.parent / document["policy_bundle_manifest_path"],
        f"{label} policy bundle manifest",
    )
    if spec != Path(f"{checkpoint}.policy_spec.json") or bundle_manifest != Path(
        f"{checkpoint}.manifest.json"
    ):
        raise RestartEquivalenceError(f"{label} policy sidecar path is non-canonical")
    if (
        sha256_file(spec) != document["policy_spec_sha256"]
        or sha256_file(bundle_manifest)
        != document["policy_bundle_manifest_sha256"]
    ):
        raise RestartEquivalenceError(f"{label} policy sidecar digest mismatch")
    try:
        validate_policy_bundle_identity(
            checkpoint=checkpoint,
            policy_spec=spec,
            manifest=bundle_manifest,
            expected_checkpoint_sha256=document["checkpoint_sha256"],
            expected_runtime_policy_sha256=document["runtime_policy_sha256"],
        )
    except PolicyBundleError as exc:
        raise RestartEquivalenceError(
            f"{label} policy bundle identity mismatch: {exc}"
        ) from exc
    return path, document


def verify(
    *,
    uninterrupted_checkpoint: Path,
    resumed_checkpoint: Path,
    expected_completed_steps: int,
    uninterrupted_result: Path | None = None,
    resumed_result: Path | None = None,
) -> dict[str, Any]:
    if isinstance(expected_completed_steps, bool) or expected_completed_steps < 1:
        raise RestartEquivalenceError("expected completed steps must be positive")
    full_path, full = _load_checkpoint(uninterrupted_checkpoint, "uninterrupted checkpoint")
    resumed_path, resumed = _load_checkpoint(resumed_checkpoint, "resumed checkpoint")
    for label, document in (("uninterrupted", full), ("resumed", resumed)):
        if document["completed_steps"] != expected_completed_steps:
            raise RestartEquivalenceError(f"{label} completed_steps mismatch")
        _validate_internal_policy_aliases(document, f"{label} checkpoint")
    _assert_equal(full, resumed, "learner restart state")

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "semantic_equivalent": True,
        "expected_completed_steps": expected_completed_steps,
        "uninterrupted_checkpoint": str(full_path),
        "uninterrupted_checkpoint_sha256": sha256_file(full_path),
        "resumed_checkpoint": str(resumed_path),
        "resumed_checkpoint_sha256": sha256_file(resumed_path),
        "results_equivalent": None,
    }
    if (uninterrupted_result is None) != (resumed_result is None):
        raise RestartEquivalenceError("both result manifests must be supplied together")
    if uninterrupted_result is not None and resumed_result is not None:
        full_result_path, full_result = _load_result(
            uninterrupted_result, "uninterrupted result"
        )
        resumed_result_path, resumed_result_document = _load_result(
            resumed_result, "resumed result"
        )
        if full_result["schema"] != resumed_result_document["schema"]:
            raise RestartEquivalenceError("learner result schemas differ")
        equivalence_fields = (
            _RESULT_EQUIVALENCE_FIELDS
            if full_result["schema"] == RESULT_SCHEMA
            else _LEGACY_RESULT_EQUIVALENCE_FIELDS
        )
        for field in sorted(equivalence_fields):
            _assert_equal(
                full_result[field],
                resumed_result_document[field],
                f"learner results[{field!r}]",
            )
        if full_result["optimizer_steps"] != expected_completed_steps:
            raise RestartEquivalenceError("result optimizer_steps mismatch")
        report.update(
            {
                "results_equivalent": True,
                "uninterrupted_result": str(full_result_path),
                "uninterrupted_result_sha256": sha256_file(full_result_path),
                "resumed_result": str(resumed_result_path),
                "resumed_result_sha256": sha256_file(resumed_result_path),
            }
        )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uninterrupted-checkpoint", required=True, type=Path)
    parser.add_argument("--resumed-checkpoint", required=True, type=Path)
    parser.add_argument("--expected-completed-steps", required=True, type=int)
    parser.add_argument("--uninterrupted-result", type=Path)
    parser.add_argument("--resumed-result", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = verify(
        uninterrupted_checkpoint=arguments.uninterrupted_checkpoint,
        resumed_checkpoint=arguments.resumed_checkpoint,
        expected_completed_steps=arguments.expected_completed_steps,
        uninterrupted_result=arguments.uninterrupted_result,
        resumed_result=arguments.resumed_result,
    )
    output = Path(os.path.abspath(arguments.output.expanduser()))
    if output.is_symlink():
        raise RestartEquivalenceError("output must not be a symbolic link")
    digest = publish_immutable_json(output, report)
    print(f"LEARNER_RESTART_EQUIVALENT report={output} sha256={digest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RestartEquivalenceError as exc:
        print(f"learner restart equivalence failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
