#!/usr/bin/env python3
"""Fail-closed verification of a completed production smoke profile.

The trainer writes ``profile_validation_manifest.txt`` only after its artifact
checks pass.  This utility independently binds that manifest to the requested
repository revisions and re-verifies every content-addressed artifact.  It is
pure standard library so it can run on a login node without the training
environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any


LEGACY_EXPECTED_FIELDS = frozenset(
    {
        "run_profile",
        "fast_roborender_revision",
        "gr00t_revision",
        "global_envs",
        "rollout_steps",
        "trainer_nodes",
        "policy_views",
        "policy_view_count",
        "policy_view_layout",
        "policy_image_shape",
        "initial_policy_checkpoint",
        "initial_policy_sha256",
        "final_checkpoint",
        "final_checkpoint_sha256",
        "finite_bc_loss_marker",
        "finite_bc_loss_marker_sha256",
        "ddp_sync_marker",
        "ddp_sync_marker_sha256",
        "camera_calibration_manifest",
        "camera_calibration_manifest_sha256",
        "rollout_iterations",
        "status",
    }
)
HSG_EXPECTED_FIELDS = frozenset(
    {
        "manifest_schema",
        "run_profile",
        "fast_roborender_revision",
        "fast_roborender_source_tree_sha256",
        "gr00t_revision",
        "gr00t_source_tree_sha256",
        "source_snapshot_attestation_sha256",
        "source_snapshot_inventory_sha256",
        "immutable_asset_contract_sha256",
        "global_envs",
        "rollout_steps",
        "trainer_nodes",
        "policy_views",
        "policy_view_count",
        "policy_view_layout",
        "policy_image_shape",
        "initial_policy_checkpoint",
        "initial_checkpoint_sha256",
        "initial_policy_state_sha256",
        "final_checkpoint",
        "final_checkpoint_sha256",
        "final_policy_state_sha256",
        "policy_change_marker",
        "policy_change_marker_sha256",
        "finite_bc_loss_marker",
        "finite_bc_loss_marker_sha256",
        "ddp_sync_marker",
        "ddp_sync_marker_sha256",
        "renderer_placement",
        "renderer_placement_sha256",
        "renderer_clean_stop_evidence",
        "renderer_clean_stop_evidence_sha256",
        "renderer_stop_ack_count",
        "renderer_stop_ack_set_sha256",
        "wandb_media_drain",
        "wandb_media_drain_sha256",
        "wandb_remote_readback",
        "wandb_remote_readback_sha256",
        "wandb_trainer_run_id",
        "wandb_sidecar_run_id",
        "camera_calibration_manifest",
        "camera_calibration_manifest_sha256",
        "rollout_iterations",
        "status",
    }
)
PROFILE_CONTRACTS = {
    "smoke_32x300": {
        "global_envs": 32,
        "rollout_steps": 300,
        "trainer_nodes": 1,
        "ddp_world_size": None,
        "renderer_count": 7,
        "renderer_host_count": 2,
    },
    "smoke_256x300": {
        "global_envs": 256,
        "rollout_steps": 300,
        "trainer_nodes": 8,
        "ddp_world_size": 8,
        "renderer_count": 52,
        "renderer_host_count": 15,
    },
}
EXPECTED_POLICY = {
    "policy_views": "head_color,hand_left,hand_right",
    "policy_view_count": "3",
    "policy_view_layout": "vertical",
    "policy_image_shape": "324x192x3",
}
FINITE_SCHEMA = "fast_roborender.dagger_bc_loss.v1"
DDP_SCHEMA = "fast_roborender.ddp_policy_sync.v1"
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_KEY = re.compile(r"[a-z][a-z0-9_]*\Z")
_RUN_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")


class GateError(RuntimeError):
    """Raised when a profile or one of its attestations is invalid."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise GateError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_revision(value: str, label: str) -> str:
    if _REVISION.fullmatch(value) is None:
        raise GateError(f"{label} must be an exact lowercase 40-hex Git revision")
    return value


def _require_sha256(value: str, label: str) -> str:
    if _SHA256.fullmatch(value) is None:
        raise GateError(f"{label} must be a lowercase 64-hex SHA-256")
    return value


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise GateError(f"duplicate JSON key {key!r}")
        document[key] = value
    return document


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle, object_pairs_hook=_reject_duplicate_json_keys)
    except GateError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GateError(f"cannot read JSON marker {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise GateError(f"JSON marker {path} must contain exactly one object")
    return document


def load_manifest(path: Path) -> tuple[Path, dict[str, str]]:
    if path.is_symlink():
        raise GateError(f"manifest must not be a symbolic link: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise GateError(f"cannot resolve manifest {path}: {exc}") from exc
    if not resolved.is_file():
        raise GateError(f"manifest is not a regular file: {resolved}")
    try:
        text = resolved.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise GateError(f"cannot read manifest {resolved}: {exc}") from exc

    fields: dict[str, str] = {}
    lines = text.splitlines()
    if not lines:
        raise GateError("manifest is empty")
    for line_number, line in enumerate(lines, 1):
        if not line or "=" not in line:
            raise GateError(
                f"manifest line {line_number} must be one non-empty key=value record"
            )
        key, value = line.split("=", 1)
        if _KEY.fullmatch(key) is None:
            raise GateError(f"manifest line {line_number} has invalid key {key!r}")
        if key in fields:
            raise GateError(f"duplicate manifest key {key!r}")
        fields[key] = value

    field_set = frozenset(fields)
    if field_set not in {LEGACY_EXPECTED_FIELDS, HSG_EXPECTED_FIELDS}:
        expected = HSG_EXPECTED_FIELDS if "manifest_schema" in fields else LEGACY_EXPECTED_FIELDS
        missing = sorted(expected - fields.keys())
        unexpected = sorted(fields.keys() - expected)
        raise GateError(
            f"manifest field set mismatch: missing={missing!r}, "
            f"unexpected={unexpected!r}"
        )
    return resolved, fields


def _require_exact(fields: dict[str, str], key: str, expected: str) -> None:
    if fields[key] != expected:
        raise GateError(
            f"manifest {key} mismatch: found {fields[key]!r}, expected {expected!r}"
        )


def _require_int(fields: dict[str, str], key: str, expected: int) -> None:
    value = fields[key]
    if value != str(expected):
        raise GateError(
            f"manifest {key} mismatch: found {value!r}, expected {expected}"
        )


def _artifact(
    fields: dict[str, str],
    manifest_path: Path,
    path_key: str,
    hash_key: str,
    relative_path: Path,
) -> tuple[Path, str]:
    value = fields[path_key]
    if not value:
        raise GateError(f"manifest {path_key} must not be empty")
    declared = Path(value)
    if not declared.is_absolute():
        raise GateError(f"manifest {path_key} must be an absolute path")
    if declared.is_symlink():
        raise GateError(f"manifest {path_key} must not be a symbolic link")
    expected = (manifest_path.parent / relative_path).resolve()
    try:
        actual = declared.resolve(strict=True)
    except OSError as exc:
        raise GateError(f"cannot resolve {path_key} artifact {declared}: {exc}") from exc
    if actual != expected:
        raise GateError(
            f"manifest {path_key} must resolve to {expected}, found {actual}"
        )
    if not actual.is_file():
        raise GateError(f"{path_key} artifact is not a regular file: {actual}")
    if actual.stat().st_size <= 0:
        raise GateError(f"{path_key} artifact is empty: {actual}")
    declared_hash = _require_sha256(fields[hash_key], hash_key)
    actual_hash = sha256_file(actual)
    if actual_hash != declared_hash:
        raise GateError(
            f"{path_key} SHA-256 mismatch: found {actual_hash}, "
            f"manifest declares {declared_hash}"
        )
    return actual, actual_hash


def _require_json_keys(
    document: dict[str, Any], expected: frozenset[str], label: str
) -> None:
    missing = sorted(expected - document.keys())
    unexpected = sorted(document.keys() - expected)
    if missing or unexpected:
        raise GateError(
            f"{label} field set mismatch: missing={missing!r}, "
            f"unexpected={unexpected!r}"
        )


def _validate_finite_marker(path: Path, expected_world_size: int) -> float:
    marker = load_json(path)
    _require_json_keys(
        marker,
        frozenset(
            {
                "schema",
                "update",
                "world_size",
                "loss",
                "finite_verified",
                "nonnegative_verified",
            }
        ),
        "finite BC-loss marker",
    )
    expected = {
        "schema": FINITE_SCHEMA,
        "update": 1,
        "world_size": expected_world_size,
        "finite_verified": True,
        "nonnegative_verified": True,
    }
    for key, value in expected.items():
        if marker[key] != value or type(marker[key]) is not type(value):
            raise GateError(
                f"finite BC-loss marker {key} mismatch: found {marker[key]!r}, "
                f"expected {value!r}"
            )
    loss = marker["loss"]
    if isinstance(loss, bool) or not isinstance(loss, (int, float)):
        raise GateError("finite BC-loss marker loss must be numeric")
    result = float(loss)
    if not math.isfinite(result) or result < 0.0:
        raise GateError(
            f"finite BC-loss marker loss must be finite and nonnegative, found {loss!r}"
        )
    return result


def _validate_ddp_marker(path: Path, expected_world_size: int) -> str:
    marker = load_json(path)
    _require_json_keys(
        marker,
        frozenset(
            {
                "schema",
                "verified",
                "optimizer_update",
                "world_size",
                "policy_parameter_count",
                "policy_parameters_sha256",
            }
        ),
        "DDP synchronization marker",
    )
    expected = {
        "schema": DDP_SCHEMA,
        "verified": True,
        "optimizer_update": 1,
        "world_size": expected_world_size,
    }
    for key, value in expected.items():
        if marker[key] != value or type(marker[key]) is not type(value):
            raise GateError(
                f"DDP synchronization marker {key} mismatch: "
                f"found {marker[key]!r}, expected {value!r}"
            )
    parameter_count = marker["policy_parameter_count"]
    if isinstance(parameter_count, bool) or not isinstance(parameter_count, int):
        raise GateError("DDP synchronization marker parameter count must be an integer")
    if parameter_count <= 0:
        raise GateError("DDP synchronization marker parameter count must be positive")
    return _require_sha256(
        marker["policy_parameters_sha256"],
        "DDP synchronization marker policy_parameters_sha256",
    )


def _validate_renderer_clean_stop(
    path: Path,
    expected_profile: str,
    expected_renderers: int,
    expected_ack_set_sha256: str,
) -> tuple[int, str]:
    evidence = load_json(path)
    expected = {
        "schema": "fast_roborender.renderer_clean_stop.v1",
        "status": "PASS",
        "run_profile": expected_profile,
        "expected_renderers": expected_renderers,
        "srun_was_live_before_stop": True,
        "renderer_srun_exit_code": 0,
        "forced_termination_used": False,
        "endpoint_entry_count": 0,
        "ready_entry_count": 0,
        "stop_ack_count": expected_renderers,
    }
    for key, value in expected.items():
        if evidence.get(key) != value or type(evidence.get(key)) is not type(value):
            raise GateError(f"renderer clean-stop evidence {key} mismatch")
    run_id = evidence.get("run_id")
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise GateError("renderer clean-stop run ID is malformed")
    attempt_id = evidence.get("attempt_id")
    if not isinstance(attempt_id, str) or not attempt_id:
        raise GateError("renderer clean-stop attempt ID is malformed")
    job_id = evidence.get("job_id")
    if not isinstance(job_id, str) or re.fullmatch(r"[0-9]+", job_id) is None:
        raise GateError("renderer clean-stop job ID is malformed")
    run_root = path.parent
    fleet_dir = run_root / "fleet"
    if fleet_dir.is_symlink() or not fleet_dir.is_dir():
        raise GateError("renderer fleet directory is missing or unsafe")
    if evidence.get("fleet_dir") != str(fleet_dir.resolve(strict=True)):
        raise GateError("renderer clean-stop fleet path is not run-scoped")
    stop_path = fleet_dir / "STOP"
    if stop_path.is_symlink() or not stop_path.is_file():
        raise GateError("renderer STOP sentinel is missing or unsafe")
    stop_path = stop_path.resolve(strict=True)
    stop_hash = sha256_file(stop_path)
    if (
        evidence.get("stop_sentinel_path") != str(stop_path)
        or evidence.get("stop_sentinel_sha256") != stop_hash
    ):
        raise GateError("renderer STOP sentinel evidence changed")
    stop_request = load_json(stop_path)
    stop_expected = {
        "schema": "fast_roborender.renderer_stop_request.v1",
        "run_id": run_id,
        "attempt_id": attempt_id,
        "job_id": job_id,
    }
    for key, value in stop_expected.items():
        if stop_request.get(key) != value:
            raise GateError(f"renderer STOP request {key} mismatch")
    endpoint_dir = fleet_dir / "endpoints"
    if endpoint_dir.is_symlink() or not endpoint_dir.is_dir() or any(
        endpoint_dir.iterdir()
    ):
        raise GateError("renderer endpoint directory is not clean")
    queue_dir = fleet_dir / "queues"
    if queue_dir.is_symlink() or not queue_dir.is_dir():
        raise GateError("renderer queue directory is missing or unsafe")
    for directory, directory_names, file_names in os.walk(
        queue_dir, followlinks=False
    ):
        base = Path(directory)
        for name in directory_names + file_names:
            entry = base / name
            if entry.is_symlink():
                raise GateError(f"renderer queue contains an unsafe symlink: {entry}")
            if name == "READY":
                raise GateError("renderer queue contains a residual READY marker")
    records = evidence.get("stop_ack_records")
    if not isinstance(records, list) or len(records) != expected_renderers:
        raise GateError("renderer STOP ACK records are incomplete")
    expected_names = [
        f"renderer_{index:03d}.json" for index in range(expected_renderers)
    ]
    if [record.get("name") for record in records if isinstance(record, dict)] != expected_names:
        raise GateError("renderer STOP ACK records are not ordered and complete")
    ack_dir = fleet_dir / "stop_acks"
    if (
        ack_dir.is_symlink()
        or not ack_dir.is_dir()
        or {path.name for path in ack_dir.iterdir()} != set(expected_names)
    ):
        raise GateError("renderer STOP ACK directory contents changed")
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {"name", "sha256"}:
            raise GateError("renderer STOP ACK record schema changed")
        ack_path = ack_dir / record["name"]
        if ack_path.is_symlink() or not ack_path.is_file():
            raise GateError(f"renderer {index} STOP ACK is missing or unsafe")
        if sha256_file(ack_path) != _require_sha256(
            record["sha256"], f"renderer {index} STOP ACK SHA-256"
        ):
            raise GateError(f"renderer {index} STOP ACK changed")
        ack = load_json(ack_path)
        if (
            ack.get("schema") != "fast_roborender.image3f_stop_ack.v1"
            or ack.get("status") != "STOP_observed"
            or ack.get("server_id") != f"{run_id}:{index:03d}"
            or ack.get("stop_sentinel_path") != str(stop_path)
            or ack.get("stop_sentinel_sha256") != stop_hash
            or type(ack.get("pid")) is not int
            or ack["pid"] <= 0
        ):
            raise GateError(f"renderer {index} STOP ACK is inconsistent")
    ack_set_sha256 = canonical_sha256(records)
    declared_set_hash = _require_sha256(
        expected_ack_set_sha256, "renderer STOP ACK-set SHA-256"
    )
    if (
        evidence.get("stop_ack_set_sha256") != ack_set_sha256
        or ack_set_sha256 != declared_set_hash
    ):
        raise GateError("renderer STOP ACK-set SHA-256 mismatch")
    return expected_renderers, ack_set_sha256


def validate_profile(
    manifest: Path,
    expected_profile: str,
    expected_fast_revision: str,
    expected_gr00t_revision: str,
    expected_manifest_sha256: str | None = None,
    expected_fast_source_tree_sha256: str | None = None,
    expected_gr00t_source_tree_sha256: str | None = None,
    expected_source_snapshot_attestation_sha256: str | None = None,
    expected_source_snapshot_inventory_sha256: str | None = None,
    expected_asset_contract_sha256: str | None = None,
    expected_wandb_email: str | None = None,
    expected_wandb_entity: str | None = None,
) -> dict[str, Any]:
    if expected_profile not in PROFILE_CONTRACTS:
        raise GateError(
            "expected profile must be exactly smoke_32x300 or smoke_256x300"
        )
    fast_revision = _require_revision(
        expected_fast_revision, "expected fast_roborender revision"
    )
    gr00t_revision = _require_revision(expected_gr00t_revision, "expected GR00T revision")
    manifest_path, fields = load_manifest(manifest)
    manifest_hash = sha256_file(manifest_path)
    if expected_manifest_sha256 is not None:
        expected_hash = _require_sha256(
            expected_manifest_sha256, "expected manifest SHA-256"
        )
        if manifest_hash != expected_hash:
            raise GateError(
                f"manifest SHA-256 mismatch: found {manifest_hash}, expected {expected_hash}"
            )

    _require_exact(fields, "status", "validated")
    _require_exact(fields, "run_profile", expected_profile)
    _require_exact(fields, "fast_roborender_revision", fast_revision)
    _require_exact(fields, "gr00t_revision", gr00t_revision)
    is_hsg = fields.get("manifest_schema") == "fast_roborender.hsg_profile_validation.v3"
    requires_hsg = any(
        value is not None
        for value in (
            expected_fast_source_tree_sha256,
            expected_gr00t_source_tree_sha256,
            expected_source_snapshot_attestation_sha256,
            expected_source_snapshot_inventory_sha256,
            expected_asset_contract_sha256,
            expected_wandb_email,
            expected_wandb_entity,
        )
    )
    if requires_hsg and not is_hsg:
        raise GateError("legacy predecessor manifest cannot authorize an HSG run")
    if is_hsg:
        expected_hsg_values = {
            "fast_roborender_source_tree_sha256": expected_fast_source_tree_sha256,
            "gr00t_source_tree_sha256": expected_gr00t_source_tree_sha256,
            "source_snapshot_attestation_sha256": (
                expected_source_snapshot_attestation_sha256
            ),
            "source_snapshot_inventory_sha256": (
                expected_source_snapshot_inventory_sha256
            ),
            "immutable_asset_contract_sha256": expected_asset_contract_sha256,
        }
        for key, expected in expected_hsg_values.items():
            if expected is None:
                raise GateError(f"expected {key} is required for an HSG manifest")
            _require_exact(fields, key, _require_sha256(expected, key))
        if not expected_wandb_email or not expected_wandb_entity:
            raise GateError("expected authorized-team W&B identity is required for an HSG manifest")
    contract = PROFILE_CONTRACTS[expected_profile]
    for key in ("global_envs", "rollout_steps", "trainer_nodes"):
        _require_int(fields, key, contract[key])
    _require_int(fields, "rollout_iterations", 1)
    for key, expected in EXPECTED_POLICY.items():
        _require_exact(fields, key, expected)
    _require_exact(fields, "camera_calibration_manifest", "")
    _require_exact(fields, "camera_calibration_manifest_sha256", "")

    initial_path, initial_hash = _artifact(
        fields,
        manifest_path,
        "initial_policy_checkpoint",
        "initial_checkpoint_sha256" if is_hsg else "initial_policy_sha256",
        Path("initial_policy.pt"),
    )
    final_path, final_hash = _artifact(
        fields,
        manifest_path,
        "final_checkpoint",
        "final_checkpoint_sha256",
        Path("model_step_000001.pt"),
    )
    finite_path, finite_hash = _artifact(
        fields,
        manifest_path,
        "finite_bc_loss_marker",
        "finite_bc_loss_marker_sha256",
        Path("validation/dagger_bc_loss_step_000000001.json"),
    )
    loss = _validate_finite_marker(finite_path, contract["trainer_nodes"])

    initial_policy_state_hash = ""
    final_policy_state_hash = ""
    placement_hash = ""
    clean_stop_hash = ""
    clean_stop_ack_count = 0
    clean_stop_ack_set_hash = ""
    wandb_drain_hash = ""
    wandb_readback_hash = ""
    if is_hsg:
        initial_policy_state_hash = _require_sha256(
            fields["initial_policy_state_sha256"], "initial policy-state SHA-256"
        )
        final_policy_state_hash = _require_sha256(
            fields["final_policy_state_sha256"], "final policy-state SHA-256"
        )
        if initial_policy_state_hash == final_policy_state_hash:
            raise GateError("HSG predecessor policy tensors did not change")
        policy_path, _ = _artifact(
            fields,
            manifest_path,
            "policy_change_marker",
            "policy_change_marker_sha256",
            Path("validation/checkpoint_policy_change.json"),
        )
        policy_marker = load_json(policy_path)
        if (
            policy_marker.get("schema")
            != "fast_roborender.checkpoint_policy_change.v1"
            or policy_marker.get("status") != "PASS"
            or policy_marker.get("initial", {}).get("checkpoint_sha256") != initial_hash
            or policy_marker.get("final", {}).get("checkpoint_sha256") != final_hash
            or policy_marker.get("initial", {}).get("policy_sha256")
            != initial_policy_state_hash
            or policy_marker.get("final", {}).get("policy_sha256")
            != final_policy_state_hash
        ):
            raise GateError("HSG predecessor policy-state marker is inconsistent")
        placement_path, placement_hash = _artifact(
            fields,
            manifest_path,
            "renderer_placement",
            "renderer_placement_sha256",
            Path("validation/renderer_placement.json"),
        )
        placement = load_json(placement_path)
        renderer_count = contract["renderer_count"]
        expected_counts = placement.get("expected_host_counts")
        actual_counts = placement.get("actual_host_counts")
        if (
            placement.get("schema")
            != "fast_roborender.renderer_placement_attestation.v1"
            or placement.get("status") != "PASS"
            or placement.get("renderer_count") != renderer_count
            or placement.get("unique_endpoint_count") != renderer_count
            or not isinstance(expected_counts, dict)
            or actual_counts != expected_counts
            or len(expected_counts) != contract["renderer_host_count"]
            or any(
                not isinstance(host, str)
                or not host
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 1
                for host, count in expected_counts.items()
            )
            or sum(expected_counts.values()) != renderer_count
        ):
            raise GateError("HSG predecessor renderer placement is invalid")
        clean_stop_path, clean_stop_hash = _artifact(
            fields,
            manifest_path,
            "renderer_clean_stop_evidence",
            "renderer_clean_stop_evidence_sha256",
            Path("../renderer_clean_stop.json"),
        )
        _require_int(
            fields, "renderer_stop_ack_count", contract["renderer_count"]
        )
        clean_stop_ack_count, clean_stop_ack_set_hash = (
            _validate_renderer_clean_stop(
                clean_stop_path,
                expected_profile,
                contract["renderer_count"],
                fields["renderer_stop_ack_set_sha256"],
            )
        )
        drain_path, wandb_drain_hash = _artifact(
            fields,
            manifest_path,
            "wandb_media_drain",
            "wandb_media_drain_sha256",
            Path("validation/wandb_media_drain.json"),
        )
        readback_path, wandb_readback_hash = _artifact(
            fields,
            manifest_path,
            "wandb_remote_readback",
            "wandb_remote_readback_sha256",
            Path("validation/wandb_training_readback.json"),
        )
        drain = load_json(drain_path)
        readback = load_json(readback_path)
        trainer_run_id = fields["wandb_trainer_run_id"]
        sidecar_run_id = fields["wandb_sidecar_run_id"]
        if _RUN_ID.fullmatch(trainer_run_id) is None or _RUN_ID.fullmatch(sidecar_run_id) is None:
            raise GateError("HSG predecessor W&B run ID is malformed")
        kinds = drain.get("artifact_kinds")
        local_counts = readback.get("local_artifact_counts")
        required_trainer = readback.get("required_trainer_keys")
        observed_trainer = readback.get("observed_trainer_keys")
        required_sidecar = readback.get("required_sidecar_keys")
        observed_sidecar = readback.get("observed_sidecar_keys")
        if (
            drain.get("schema") != "fast_roborender.wandb_media_drain.v1"
            or drain.get("status") != "PASS"
            or readback.get("schema")
            != "fast_roborender.wandb_training_readback.v2"
            or readback.get("status") != "PASS"
            or readback.get("account_email") != expected_wandb_email
            or readback.get("entity") != expected_wandb_entity
            or readback.get("authorization_scope") != "team"
            or readback.get("team_member") is not True
            or readback.get("entity_is_team") is not True
            or readback.get("personal") is not False
            or readback.get("trainer_run_id") != trainer_run_id
            or readback.get("sidecar_run_id") != sidecar_run_id
            or drain.get("wandb_run_id") != sidecar_run_id
            or not isinstance(kinds, dict)
            or not isinstance(local_counts, dict)
            or any(
                isinstance(kinds.get(kind), bool)
                or not isinstance(kinds.get(kind), int)
                or kinds[kind] < 1
                or local_counts.get(kind) != kinds[kind]
                for kind in ("scalar", "image", "video")
            )
            or not isinstance(required_trainer, list)
            or not isinstance(observed_trainer, list)
            or any(not isinstance(key, str) for key in required_trainer)
            or any(not isinstance(key, str) for key in observed_trainer)
            or "production/dagger_bc_loss" not in required_trainer
            or not set(required_trainer).issubset(observed_trainer)
            or not isinstance(required_sidecar, list)
            or not required_sidecar
            or not isinstance(observed_sidecar, list)
            or any(not isinstance(key, str) for key in required_sidecar)
            or any(not isinstance(key, str) for key in observed_sidecar)
            or not set(required_sidecar).issubset(observed_sidecar)
        ):
            raise GateError("HSG predecessor W&B evidence is inconsistent")

    ddp_path: Path | None = None
    ddp_hash = ""
    parameter_hash = ""
    if contract["ddp_world_size"] is None:
        _require_exact(fields, "ddp_sync_marker", "")
        _require_exact(fields, "ddp_sync_marker_sha256", "")
    else:
        ddp_path, ddp_hash = _artifact(
            fields,
            manifest_path,
            "ddp_sync_marker",
            "ddp_sync_marker_sha256",
            Path("validation/ddp_policy_sync_step_000000001.json"),
        )
        parameter_hash = _validate_ddp_marker(ddp_path, contract["ddp_world_size"])
        if is_hsg and parameter_hash != final_policy_state_hash:
            raise GateError(
                "DDP synchronized policy digest does not match the persisted final policy"
            )

    return {
        "schema": "fast_roborender.profile_validation_verification.v1",
        "status": "verified",
        "run_profile": expected_profile,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "fast_roborender_revision": fast_revision,
        "fast_roborender_source_tree_sha256": (
            fields.get("fast_roborender_source_tree_sha256", "")
        ),
        "gr00t_revision": gr00t_revision,
        "gr00t_source_tree_sha256": fields.get("gr00t_source_tree_sha256", ""),
        "source_snapshot_attestation_sha256": fields.get(
            "source_snapshot_attestation_sha256", ""
        ),
        "source_snapshot_inventory_sha256": fields.get(
            "source_snapshot_inventory_sha256", ""
        ),
        "immutable_asset_contract_sha256": fields.get(
            "immutable_asset_contract_sha256", ""
        ),
        "global_envs": contract["global_envs"],
        "rollout_steps": contract["rollout_steps"],
        "trainer_nodes": contract["trainer_nodes"],
        "policy_views": EXPECTED_POLICY["policy_views"].split(","),
        "initial_policy_checkpoint": str(initial_path),
        "initial_policy_sha256": initial_hash,
        "initial_policy_state_sha256": initial_policy_state_hash,
        "final_checkpoint": str(final_path),
        "final_checkpoint_sha256": final_hash,
        "final_policy_state_sha256": final_policy_state_hash,
        "finite_bc_loss_marker": str(finite_path),
        "finite_bc_loss_marker_sha256": finite_hash,
        "finite_bc_loss": loss,
        "ddp_sync_marker": "" if ddp_path is None else str(ddp_path),
        "ddp_sync_marker_sha256": ddp_hash,
        "ddp_policy_parameters_sha256": parameter_hash,
        "renderer_placement_sha256": placement_hash,
        "renderer_clean_stop_evidence_sha256": clean_stop_hash,
        "renderer_stop_ack_count": clean_stop_ack_count,
        "renderer_stop_ack_set_sha256": clean_stop_ack_set_hash,
        "wandb_media_drain_sha256": wandb_drain_hash,
        "wandb_remote_readback_sha256": wandb_readback_hash,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--expected-profile",
        required=True,
        choices=sorted(PROFILE_CONTRACTS),
    )
    parser.add_argument("--expected-fast-revision", required=True)
    parser.add_argument("--expected-gr00t-revision", required=True)
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--expected-fast-source-tree-sha256")
    parser.add_argument("--expected-gr00t-source-tree-sha256")
    parser.add_argument("--expected-source-snapshot-attestation-sha256")
    parser.add_argument("--expected-source-snapshot-inventory-sha256")
    parser.add_argument("--expected-asset-contract-sha256")
    parser.add_argument("--expected-wandb-email")
    parser.add_argument("--expected-wandb-entity")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = validate_profile(
            args.manifest,
            args.expected_profile,
            args.expected_fast_revision,
            args.expected_gr00t_revision,
            args.expected_manifest_sha256,
            args.expected_fast_source_tree_sha256,
            args.expected_gr00t_source_tree_sha256,
            args.expected_source_snapshot_attestation_sha256,
            args.expected_source_snapshot_inventory_sha256,
            args.expected_asset_contract_sha256,
            args.expected_wandb_email,
            args.expected_wandb_entity,
        )
    except GateError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
