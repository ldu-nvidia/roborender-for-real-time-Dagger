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
import re
import sys
from pathlib import Path
from typing import Any


EXPECTED_FIELDS = frozenset(
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
PROFILE_CONTRACTS = {
    "smoke_32x300": {
        "global_envs": 32,
        "rollout_steps": 300,
        "trainer_nodes": 1,
        "ddp_world_size": None,
    },
    "smoke_256x300": {
        "global_envs": 256,
        "rollout_steps": 300,
        "trainer_nodes": 8,
        "ddp_world_size": 8,
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

    missing = sorted(EXPECTED_FIELDS - fields.keys())
    unexpected = sorted(fields.keys() - EXPECTED_FIELDS)
    if missing or unexpected:
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


def validate_profile(
    manifest: Path,
    expected_profile: str,
    expected_fast_revision: str,
    expected_gr00t_revision: str,
    expected_manifest_sha256: str | None = None,
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
        "initial_policy_sha256",
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

    return {
        "schema": "fast_roborender.profile_validation_verification.v1",
        "status": "verified",
        "run_profile": expected_profile,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "fast_roborender_revision": fast_revision,
        "gr00t_revision": gr00t_revision,
        "global_envs": contract["global_envs"],
        "rollout_steps": contract["rollout_steps"],
        "trainer_nodes": contract["trainer_nodes"],
        "policy_views": EXPECTED_POLICY["policy_views"].split(","),
        "initial_policy_checkpoint": str(initial_path),
        "initial_policy_sha256": initial_hash,
        "final_checkpoint": str(final_path),
        "final_checkpoint_sha256": final_hash,
        "finite_bc_loss_marker": str(finite_path),
        "finite_bc_loss_marker_sha256": finite_hash,
        "finite_bc_loss": loss,
        "ddp_sync_marker": "" if ddp_path is None else str(ddp_path),
        "ddp_sync_marker_sha256": ddp_hash,
        "ddp_policy_parameters_sha256": parameter_hash,
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
        )
    except GateError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
