#!/usr/bin/env python3
"""Convert immutable expert captures into ordinary sealed eval replay.

The simulator capture deliberately retains ``teacher_obs`` as privileged audit
evidence.  Learners must never receive it.  This adapter recursively verifies
the fixed 8x32 capture aggregate, removes exactly that one array, and publishes
one ordinary v1 ``purpose=eval`` cohort per seed.  A create-only attestation
binds every source capture/lineage digest to the resulting shard and payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from round_dagger import artifacts
from round_dagger.heldout_admission import (
    build_fixed_heldout_bundle,
    verify_fixed_heldout_bundle,
)
from round_dagger.heldout_capture import (
    ACTION_DIM,
    POLICY_VIEWS,
    _load_capture,
    verify_capture_aggregate,
    verify_capture_lineage,
)
from round_dagger.scenario_capture import EXPECTED_ENVS_PER_SEED, EXPECTED_SEEDS


ATTESTATION_SCHEMA = "fast_roborender.round_dagger.heldout_eval_conversion.v1"
PRIVILEGED_ARRAY = "teacher_obs"
IDENTITY_FIELDS = (
    "policy_sha256",
    "runtime_policy_sha256",
    "teacher_sha256",
    "renderer_sha256",
    "beta_sha256",
    "config_sha256",
)
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ATTESTATION_KEYS = {
    "schema",
    "campaign_id",
    "purpose",
    "case_count",
    "seed_count",
    "seeds",
    "expected_beta",
    "identity",
    "view_contract",
    "action_contract",
    "inputs",
    "aggregate_sha256",
    "conversions",
    "conversion_inventory_sha256",
    "fixed_bundle",
    "attestation_sha256",
}
_CONVERSION_KEYS = {
    "seed",
    "cohort_index",
    "global_env_ids",
    "case_inventory_sha256",
    "stripped_array",
    "source",
    "output",
    "conversion_sha256",
}
_SOURCE_KEYS = {
    "lineage_path",
    "lineage_file_sha256",
    "lineage_sha256",
    "capture_path",
    "capture_sha256",
    "array_sha256",
}
_OUTPUT_KEYS = {
    "cohort_path",
    "cohort_manifest_path",
    "cohort_manifest_sha256",
    "logical_id",
    "shard_manifest_path",
    "shard_manifest_sha256",
    "payload_path",
    "payload_sha256",
    "array_sha256",
}
_FIXED_KEYS = {
    "snapshot_path",
    "snapshot_file_sha256",
    "snapshot_sha256",
    "case_manifest_path",
    "case_manifest_file_sha256",
    "case_manifest_sha256",
    "verified_snapshot_sha256",
}


class HeldoutConversionError(RuntimeError):
    """A held-out conversion is incomplete, ambiguous, or not reproducible."""


def _canonical_sha(value: Any) -> str:
    return artifacts.canonical_json_sha256(value)


def _array_sha(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(item) for item in value.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _regular(path: str | os.PathLike[str], label: str) -> Path:
    target = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    try:
        info = os.lstat(target)
    except OSError as exc:
        raise HeldoutConversionError(f"cannot inspect {label}: {exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise HeldoutConversionError(f"{label} must be a regular non-symlink file")
    return target


def _directory(path: str | os.PathLike[str], label: str) -> Path:
    target = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    if target.exists() or target.is_symlink():
        try:
            info = os.lstat(target)
        except OSError as exc:  # pragma: no cover - race defensive
            raise HeldoutConversionError(f"cannot inspect {label}: {exc}") from exc
        if not stat.S_ISDIR(info.st_mode):
            raise HeldoutConversionError(f"{label} must be a non-symlink directory")
    return target


def _identity(value: Mapping[str, Any], *, expected_beta: float) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(IDENTITY_FIELDS):
        raise HeldoutConversionError(
            "conversion identity must contain exactly the six replay digest fields"
        )
    result = dict(value)
    for key in IDENTITY_FIELDS:
        if not isinstance(result[key], str) or _SHA.fullmatch(result[key]) is None:
            raise HeldoutConversionError(f"{key} must be an exact lowercase SHA-256")
    if (
        isinstance(expected_beta, bool)
        or not isinstance(expected_beta, (int, float))
        or not math.isfinite(float(expected_beta))
        or not 0.0 <= float(expected_beta) <= 1.0
    ):
        raise HeldoutConversionError("expected beta must be finite and in [0,1]")
    if artifacts.canonical_json_sha256({"beta": float(expected_beta)}) != result[
        "beta_sha256"
    ]:
        raise HeldoutConversionError("beta value does not match beta_sha256")
    return result


def _verify_lineage_identity(
    lineage: Mapping[str, Any], identity: Mapping[str, str]
) -> None:
    provenance = lineage.get("provenance")
    if not isinstance(provenance, Mapping):
        raise HeldoutConversionError("capture lineage lacks provenance")
    expected = {
        "parent_checkpoint_sha256": identity["policy_sha256"],
        "parent_runtime_policy_sha256": identity["runtime_policy_sha256"],
        "renderer_cohort_sha256": identity["renderer_sha256"],
        "evaluation_config_sha256": identity["config_sha256"],
    }
    mismatches = [key for key, value in expected.items() if provenance.get(key) != value]
    if mismatches:
        raise HeldoutConversionError(
            "capture provenance differs from replay identity: " + ", ".join(mismatches)
        )


def _source_arrays(lineage: Mapping[str, Any]) -> dict[str, np.ndarray]:
    capture = lineage.get("capture")
    if not isinstance(capture, Mapping):
        raise HeldoutConversionError("capture lineage lacks capture evidence")
    capture_path = _regular(capture.get("path", ""), "source capture")
    expected_file_sha = capture.get("sha256")
    if artifacts.sha256_file(capture_path) != expected_file_sha:
        raise HeldoutConversionError("source capture file digest mismatch")
    arrays = _load_capture(capture_path)
    # Close the verification/use race by hashing the regular file again after
    # NumPy has consumed it.  Immutable publication makes an identical retry OK.
    if artifacts.sha256_file(capture_path) != expected_file_sha:
        raise HeldoutConversionError("source capture changed while being read")
    declared = capture.get("array_sha256")
    observed = {key: _array_sha(value) for key, value in sorted(arrays.items())}
    if not isinstance(declared, Mapping) or dict(declared) != observed:
        raise HeldoutConversionError("source capture array inventory mismatch")
    return arrays


def _load_npz(path: Path, expected_sha: str) -> dict[str, np.ndarray]:
    path = _regular(path, "converted rollout payload")
    if artifacts.sha256_file(path) != expected_sha:
        raise HeldoutConversionError("converted rollout payload digest mismatch")
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {
                key: np.ascontiguousarray(archive[key]) for key in archive.files
            }
    except Exception as exc:
        raise HeldoutConversionError(f"cannot load converted rollout payload: {exc}") from exc
    if artifacts.sha256_file(path) != expected_sha:
        raise HeldoutConversionError("converted rollout payload changed while being read")
    return arrays


def _kept_arrays(source: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if PRIVILEGED_ARRAY not in source:
        raise HeldoutConversionError("source capture lacks privileged teacher_obs evidence")
    kept = {key: np.ascontiguousarray(value) for key, value in source.items() if key != PRIVILEGED_ARRAY}
    if set(source) - set(kept) != {PRIVILEGED_ARRAY}:
        raise HeldoutConversionError("conversion must strip exactly teacher_obs")
    return kept


def _cohort_record(
    *,
    lineage: Mapping[str, Any],
    source_arrays: Mapping[str, np.ndarray],
    cohort_path: Path,
    cohort_report: Mapping[str, Any],
) -> dict[str, Any]:
    shards = cohort_report.get("shards")
    if not isinstance(shards, list) or len(shards) != 1:
        raise HeldoutConversionError("each held-out seed cohort must contain one shard")
    shard = shards[0]
    manifest_path = cohort_path / shard["manifest_path"]
    payload_path = cohort_path / shard["payload_path"]
    output_arrays = _load_npz(payload_path, shard["payload_sha256"])
    expected_arrays = _kept_arrays(source_arrays)
    if set(output_arrays) != set(expected_arrays):
        raise HeldoutConversionError("converted payload did not strip exactly teacher_obs")
    for key in sorted(expected_arrays):
        if (
            expected_arrays[key].dtype != output_arrays[key].dtype
            or expected_arrays[key].shape != output_arrays[key].shape
            or not np.array_equal(expected_arrays[key], output_arrays[key])
        ):
            raise HeldoutConversionError(f"converted payload changed replay array {key}")
    ids = output_arrays["global_env_ids"][0].tolist()
    cases = lineage.get("cases")
    if not isinstance(cases, list) or [case.get("global_case_index") for case in cases] != ids:
        raise HeldoutConversionError("converted payload IDs differ from bound lineage cases")
    source_inventory = {
        key: _array_sha(value) for key, value in sorted(source_arrays.items())
    }
    output_inventory = {
        key: _array_sha(value) for key, value in sorted(output_arrays.items())
    }
    source = {
        "lineage_path": lineage["path"],
        "lineage_file_sha256": lineage["file_sha256"],
        "lineage_sha256": lineage["lineage_sha256"],
        "capture_path": lineage["capture"]["path"],
        "capture_sha256": lineage["capture"]["sha256"],
        "array_sha256": source_inventory,
    }
    output = {
        "cohort_path": str(cohort_path),
        "cohort_manifest_path": cohort_report["manifest_path"],
        "cohort_manifest_sha256": cohort_report["cohort_sha256"],
        "logical_id": cohort_report["logical_id"],
        "shard_manifest_path": str(manifest_path),
        "shard_manifest_sha256": shard["manifest_sha256"],
        "payload_path": str(payload_path),
        "payload_sha256": shard["payload_sha256"],
        "array_sha256": output_inventory,
    }
    record = {
        "seed": lineage["seed"],
        "cohort_index": EXPECTED_SEEDS.index(lineage["seed"]),
        "global_env_ids": ids,
        "case_inventory_sha256": _canonical_sha(cases),
        "stripped_array": {
            "name": PRIVILEGED_ARRAY,
            "sha256": source_inventory[PRIVILEGED_ARRAY],
        },
        "source": source,
        "output": output,
    }
    record["conversion_sha256"] = _canonical_sha(record)
    return record


def _input_record(path: str | os.PathLike[str]) -> dict[str, str]:
    value = _regular(path, "fixed held-out input")
    return {"path": str(value), "sha256": artifacts.sha256_file(value)}


def _verify_replay_contracts(
    view_contract: Any, action_contract: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(view_contract, Mapping) or set(view_contract) != {
        "dtype",
        "layout",
        "shape",
        "views",
    }:
        raise HeldoutConversionError("converted view contract is malformed")
    view = dict(view_contract)
    shape = view.get("shape")
    if (
        view.get("dtype") != "uint8"
        or view.get("layout") != "vertical"
        or view.get("views") != list(POLICY_VIEWS)
        or not isinstance(shape, list)
        or len(shape) != 3
        or any(type(item) is not int or item <= 0 for item in shape)
        or shape[0] % len(POLICY_VIEWS)
        or shape[2] != 3
    ):
        raise HeldoutConversionError("converted view contract is not exact three-view RGB")
    expected_action = {"dtype": "float32", "shape": [ACTION_DIM]}
    if not isinstance(action_contract, Mapping) or dict(action_contract) != expected_action:
        raise HeldoutConversionError("converted action contract is not exact 26D float32")
    return view, expected_action


def _verify_eval_cohort_identity(
    report: Mapping[str, Any],
    *,
    campaign_id: str,
    cohort_index: int,
    identity: Mapping[str, str],
    view_contract: Mapping[str, Any],
    action_contract: Mapping[str, Any],
    global_env_ids: Sequence[int],
) -> None:
    expected = {
        "campaign_id": campaign_id,
        "macro_round": 0,
        "cohort_index": cohort_index,
        "purpose": "eval",
        "transition_count": EXPECTED_ENVS_PER_SEED,
        "expected_ranks": [0],
        "expected_global_env_ids": list(global_env_ids),
        "view_contract": dict(view_contract),
        "action_contract": dict(action_contract),
        **identity,
    }
    mismatches = [key for key, value in expected.items() if report.get(key) != value]
    if mismatches:
        raise HeldoutConversionError(
            "sealed eval cohort identity/contract mismatch: " + ", ".join(mismatches)
        )


def convert_capture_aggregate(
    aggregate_path: str | os.PathLike[str],
    *,
    campaign_root: str | os.PathLike[str],
    campaign_id: str,
    identity: Mapping[str, Any],
    expected_beta: float,
    plan: str | os.PathLike[str],
    motion_split: str | os.PathLike[str],
    scenarios: str | os.PathLike[str],
) -> dict[str, Any]:
    """Publish and recursively verify the fixed 8x32 eval replay bundle."""

    root = _directory(campaign_root, "campaign root")
    if not isinstance(campaign_id, str) or re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", campaign_id
    ) is None:
        raise HeldoutConversionError("campaign_id is not a safe artifact token")
    replay_identity = _identity(identity, expected_beta=expected_beta)
    aggregate = verify_capture_aggregate(aggregate_path)
    lineages = [
        verify_capture_lineage(row["path"]) for row in aggregate["lineages"]
    ]
    lineages.sort(key=lambda item: item["seed"])
    if [item["seed"] for item in lineages] != list(EXPECTED_SEEDS):
        raise HeldoutConversionError("capture aggregate does not contain exact fixed seeds")

    prepared: list[tuple[dict[str, Any], dict[str, np.ndarray]]] = []
    image_shape: tuple[int, int, int] | None = None
    for lineage in lineages:
        _verify_lineage_identity(lineage, replay_identity)
        arrays = _source_arrays(lineage)
        env_ids = arrays["global_env_ids"][0].tolist()
        seed_index = EXPECTED_SEEDS.index(lineage["seed"])
        expected_ids = list(
            range(
                seed_index * EXPECTED_ENVS_PER_SEED,
                (seed_index + 1) * EXPECTED_ENVS_PER_SEED,
            )
        )
        if env_ids != expected_ids:
            raise HeldoutConversionError("capture IDs are not canonical contiguous 8x32 IDs")
        current_shape = tuple(int(item) for item in arrays["policy_rgb"].shape[2:])
        if image_shape is None:
            image_shape = current_shape
        elif image_shape != current_shape:
            raise HeldoutConversionError("capture view shapes differ across seeds")
        prepared.append((lineage, arrays))

    # Validate every dependency before publishing the first cohort.
    inputs = {
        "aggregate": _input_record(aggregate_path),
        "plan": _input_record(plan),
        "motion_split": _input_record(motion_split),
        "scenarios": _input_record(scenarios),
    }
    if inputs["aggregate"]["sha256"] != aggregate["file_sha256"]:
        raise HeldoutConversionError("aggregate path differs from verified aggregate bytes")
    if inputs["plan"]["sha256"] != aggregate["plan_sha256"]:
        raise HeldoutConversionError("fixed plan differs from capture aggregate")

    view_contract = {
        "dtype": "uint8",
        "layout": "vertical",
        "shape": list(image_shape or ()),
        "views": list(POLICY_VIEWS),
    }
    action_contract = {"dtype": "float32", "shape": [ACTION_DIM]}
    _verify_replay_contracts(view_contract, action_contract)
    cohort_paths: list[Path] = []
    records: list[dict[str, Any]] = []
    for cohort_index, (lineage, arrays) in enumerate(prepared):
        cohort_path = root / f"eval_seed_{lineage['seed']:03d}"
        env_ids = arrays["global_env_ids"][0].tolist()
        artifacts.create_cohort(
            cohort_path,
            campaign_id=campaign_id,
            macro_round=0,
            cohort_index=cohort_index,
            purpose="eval",
            expected_ranks=[0],
            expected_global_env_ids=env_ids,
            expected_transitions=EXPECTED_ENVS_PER_SEED,
            view_contract=view_contract,
            action_contract=action_contract,
            **replay_identity,
        )
        artifacts.publish_rollout_shard(
            cohort_path,
            rank=0,
            rollout_id=0,
            part_index=0,
            arrays=_kept_arrays(arrays),
            campaign_id=campaign_id,
            macro_round=0,
            cohort_index=cohort_index,
            purpose="eval",
            **replay_identity,
        )
        report = artifacts.seal_cohort(cohort_path)
        report = artifacts.verify_cohort(cohort_path)
        _verify_eval_cohort_identity(
            report,
            campaign_id=campaign_id,
            cohort_index=cohort_index,
            identity=replay_identity,
            view_contract=view_contract,
            action_contract=action_contract,
            global_env_ids=env_ids,
        )
        records.append(
            _cohort_record(
                lineage=lineage,
                source_arrays=arrays,
                cohort_path=cohort_path,
                cohort_report=report,
            )
        )
        cohort_paths.append(cohort_path)

    fixed = build_fixed_heldout_bundle(
        cohort_paths,
        plan=inputs["plan"]["path"],
        motion_split=inputs["motion_split"]["path"],
        scenarios=inputs["scenarios"]["path"],
        snapshot_output=root / "heldout_snapshot.json",
        manifest_output=root / "heldout_case_manifest.json",
    )
    verified_fixed = verify_fixed_heldout_bundle(
        fixed["snapshot"]["path"],
        fixed["case_manifest"]["path"],
        plan=inputs["plan"]["path"],
        motion_split=inputs["motion_split"]["path"],
        scenarios=inputs["scenarios"]["path"],
    )
    document = {
        "schema": ATTESTATION_SCHEMA,
        "campaign_id": campaign_id,
        "purpose": "eval",
        "case_count": 256,
        "seed_count": 8,
        "seeds": list(EXPECTED_SEEDS),
        "expected_beta": float(expected_beta),
        "identity": replay_identity,
        "view_contract": view_contract,
        "action_contract": action_contract,
        "inputs": inputs,
        "aggregate_sha256": aggregate["aggregate_sha256"],
        "conversions": records,
        "conversion_inventory_sha256": _canonical_sha(records),
        "fixed_bundle": {
            "snapshot_path": fixed["snapshot"]["path"],
            "snapshot_file_sha256": fixed["snapshot"]["file_sha256"],
            "snapshot_sha256": fixed["snapshot"]["snapshot_sha256"],
            "case_manifest_path": fixed["case_manifest"]["path"],
            "case_manifest_file_sha256": fixed["case_manifest"]["file_sha256"],
            "case_manifest_sha256": fixed["case_manifest"]["manifest_sha256"],
            "verified_snapshot_sha256": verified_fixed["snapshot_sha256"],
        },
    }
    document["attestation_sha256"] = _canonical_sha(document)
    attestation_path = root / "heldout_conversion_attestation.json"
    file_sha = artifacts.publish_immutable_json(attestation_path, document)
    return verify_conversion_bundle(attestation_path) | {"file_sha256": file_sha}


def verify_conversion_bundle(
    attestation_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Recursively verify source captures, eval cohorts, and fixed snapshot."""

    path = _regular(attestation_path, "held-out conversion attestation")
    document = artifacts.load_strict_json(path)
    if (
        not isinstance(document, Mapping)
        or set(document) != _ATTESTATION_KEYS
        or document.get("schema") != ATTESTATION_SCHEMA
    ):
        raise HeldoutConversionError("held-out conversion attestation schema mismatch")
    semantic = dict(document)
    declared = semantic.pop("attestation_sha256", None)
    if not isinstance(declared, str) or _SHA.fullmatch(declared) is None or _canonical_sha(semantic) != declared:
        raise HeldoutConversionError("held-out conversion attestation digest mismatch")
    if (
        document.get("purpose") != "eval"
        or document.get("case_count") != 256
        or document.get("seed_count") != 8
        or document.get("seeds") != list(EXPECTED_SEEDS)
    ):
        raise HeldoutConversionError("held-out conversion is not exact fixed 8x32 eval")
    identity = _identity(
        document.get("identity", {}), expected_beta=document.get("expected_beta")
    )
    view_contract, action_contract = _verify_replay_contracts(
        document.get("view_contract"), document.get("action_contract")
    )
    campaign_id = document.get("campaign_id")
    if not isinstance(campaign_id, str) or re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", campaign_id
    ) is None:
        raise HeldoutConversionError("attested campaign_id is not a safe artifact token")
    inputs = document.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "aggregate",
        "plan",
        "motion_split",
        "scenarios",
    }:
        raise HeldoutConversionError("held-out conversion input inventory mismatch")
    for label, item in inputs.items():
        if (
            not isinstance(item, Mapping)
            or set(item) != {"path", "sha256"}
            or artifacts.sha256_file(_regular(item["path"], label)) != item["sha256"]
        ):
            raise HeldoutConversionError(f"held-out conversion input changed: {label}")
    aggregate = verify_capture_aggregate(inputs["aggregate"]["path"])
    if (
        aggregate["aggregate_sha256"] != document.get("aggregate_sha256")
        or aggregate["plan_sha256"] != inputs["plan"]["sha256"]
    ):
        raise HeldoutConversionError("verified aggregate differs from conversion attestation")

    raw_records = document.get("conversions")
    if not isinstance(raw_records, list) or len(raw_records) != len(EXPECTED_SEEDS):
        raise HeldoutConversionError("conversion record inventory mismatch")
    if _canonical_sha(raw_records) != document.get("conversion_inventory_sha256"):
        raise HeldoutConversionError("conversion record inventory digest mismatch")
    aggregate_by_seed = {row["seed"]: row for row in aggregate["lineages"]}
    rebuilt: list[dict[str, Any]] = []
    for cohort_index, record in enumerate(raw_records):
        seed = EXPECTED_SEEDS[cohort_index]
        if (
            not isinstance(record, Mapping)
            or set(record) != _CONVERSION_KEYS
            or record.get("seed") != seed
            or record.get("cohort_index") != cohort_index
            or not isinstance(record.get("source"), Mapping)
            or set(record["source"]) != _SOURCE_KEYS
            or not isinstance(record.get("output"), Mapping)
            or set(record["output"]) != _OUTPUT_KEYS
            or record.get("stripped_array")
            != {
                "name": PRIVILEGED_ARRAY,
                "sha256": record.get("source", {})
                .get("array_sha256", {})
                .get(PRIVILEGED_ARRAY),
            }
        ):
            raise HeldoutConversionError("conversion records are not in canonical seed order")
        if seed not in aggregate_by_seed:
            raise HeldoutConversionError("conversion seed is absent from aggregate")
        lineage = verify_capture_lineage(aggregate_by_seed[seed]["path"])
        _verify_lineage_identity(lineage, identity)
        arrays = _source_arrays(lineage)
        cohort_path = _directory(record["output"]["cohort_path"], "eval cohort")
        if cohort_path.parent != path.parent or cohort_path.name != f"eval_seed_{seed:03d}":
            raise HeldoutConversionError("eval cohort is outside the canonical campaign root")
        report = artifacts.verify_cohort(cohort_path)
        _verify_eval_cohort_identity(
            report,
            campaign_id=campaign_id,
            cohort_index=cohort_index,
            identity=identity,
            view_contract=view_contract,
            action_contract=action_contract,
            global_env_ids=list(range(cohort_index * 32, (cohort_index + 1) * 32)),
        )
        current = _cohort_record(
            lineage=lineage,
            source_arrays=arrays,
            cohort_path=cohort_path,
            cohort_report=report,
        )
        if current != record:
            raise HeldoutConversionError("conversion record differs from source/output bytes")
        rebuilt.append(current)
    if _canonical_sha(rebuilt) != document["conversion_inventory_sha256"]:
        raise HeldoutConversionError("rebuilt conversion inventory digest mismatch")

    fixed = document.get("fixed_bundle")
    if not isinstance(fixed, Mapping) or set(fixed) != _FIXED_KEYS:
        raise HeldoutConversionError("fixed held-out bundle evidence is missing")
    if (
        Path(fixed["snapshot_path"]).parent != path.parent
        or Path(fixed["case_manifest_path"]).parent != path.parent
    ):
        raise HeldoutConversionError("fixed held-out bundle is outside campaign root")
    verified = verify_fixed_heldout_bundle(
        fixed["snapshot_path"],
        fixed["case_manifest_path"],
        plan=inputs["plan"]["path"],
        motion_split=inputs["motion_split"]["path"],
        scenarios=inputs["scenarios"]["path"],
    )
    expected_fixed = {
        "snapshot_file_sha256": artifacts.sha256_file(fixed["snapshot_path"]),
        "snapshot_sha256": verified["snapshot_sha256"],
        "case_manifest_file_sha256": artifacts.sha256_file(
            fixed["case_manifest_path"]
        ),
        "case_manifest_sha256": verified["manifest_sha256"],
        "verified_snapshot_sha256": verified["snapshot_sha256"],
    }
    mismatches = [key for key, value in expected_fixed.items() if fixed.get(key) != value]
    if mismatches:
        raise HeldoutConversionError(
            "fixed bundle differs from conversion attestation: " + ", ".join(mismatches)
        )
    return {
        "status": "verified",
        "path": str(path),
        "file_sha256": artifacts.sha256_file(path),
        "attestation_sha256": declared,
        "campaign_id": document["campaign_id"],
        "case_count": 256,
        "cohort_count": 8,
        "snapshot_path": fixed["snapshot_path"],
        "snapshot_sha256": verified["snapshot_sha256"],
        "case_manifest_path": fixed["case_manifest_path"],
    }


def _cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    convert = subparsers.add_parser("convert")
    convert.add_argument("--aggregate", required=True)
    convert.add_argument("--campaign-root", required=True)
    convert.add_argument("--campaign-id", required=True)
    convert.add_argument("--expected-beta", required=True, type=float)
    for field in IDENTITY_FIELDS:
        convert.add_argument("--" + field.replace("_", "-"), required=True)
    convert.add_argument("--plan", required=True)
    convert.add_argument("--motion-split", required=True)
    convert.add_argument("--scenarios", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--attestation", required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "convert":
        result = convert_capture_aggregate(
            arguments.aggregate,
            campaign_root=arguments.campaign_root,
            campaign_id=arguments.campaign_id,
            identity={field: getattr(arguments, field) for field in IDENTITY_FIELDS},
            expected_beta=arguments.expected_beta,
            plan=arguments.plan,
            motion_split=arguments.motion_split,
            scenarios=arguments.scenarios,
        )
    else:
        result = verify_conversion_bundle(arguments.attestation)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())


__all__ = [
    "HeldoutConversionError",
    "convert_capture_aggregate",
    "verify_conversion_bundle",
]
