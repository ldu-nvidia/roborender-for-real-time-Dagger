#!/usr/bin/env python3
"""Upgrade a trusted legacy bootstrap to the complete policy identity contract.

Legacy fresh-student checkpoints predated the requirement that the executed
Homie walk and stand controllers be part of the runtime policy identity.  This
tool performs a one-way, content-addressed migration into a new immutable
bundle.  It never edits the source checkpoint or its sidecars.

The source checkpoint is deserialized only after its caller-supplied SHA-256
has been verified.  The migrated checkpoint records the two controller
digests, and the portable policy specification is upgraded to the current
schema with a serialization-independent policy-state digest and the complete
runtime identity.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from round_dagger.actor import runtime_state_dict_sha256
from round_dagger.artifacts import sha256_file
from round_dagger.gr00t_policy import POLICY_SPEC_SCHEMA, validate_policy_spec
from round_dagger.policy_bundle import (
    BOOTSTRAP_MANIFEST_SCHEMA,
    validate_policy_bundle_identity,
)
from round_dagger.policy_identity import runtime_policy_identity_sha256


LEGACY_POLICY_SPEC_SCHEMA = "fast_roborender.round_dagger.gr00t_policy_spec.v2"


class BootstrapUpgradeError(RuntimeError):
    """The legacy bundle or migration destination is unsafe or inconsistent."""


def _sha256(value: str, label: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise BootstrapUpgradeError(f"{label} must be a lowercase SHA-256")
    return normalized


def _trusted_regular(path: Path, expected_sha256: str, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise BootstrapUpgradeError(f"{label} must be an absolute regular non-symlink file")
    expected = _sha256(expected_sha256, f"{label} expected SHA-256")
    actual = sha256_file(path)
    if actual != expected:
        raise BootstrapUpgradeError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return path.resolve(strict=True)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    def strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise BootstrapUpgradeError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=strict_object,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    BootstrapUpgradeError(f"{label} contains non-finite {token}")
                ),
            )
    except BootstrapUpgradeError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BootstrapUpgradeError(f"cannot read {label}: {exc}") from None
    if not isinstance(value, dict):
        raise BootstrapUpgradeError(f"{label} must contain a JSON object")
    return value


def _atomic_json(path: Path, document: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(
                document,
                handle,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _policy_state(document: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("policy_state_dict", "actor_model_state_dict", "model_state_dict"):
        value = document.get(key)
        if isinstance(value, Mapping) and value:
            return value
    raise BootstrapUpgradeError("legacy checkpoint has no supported policy state_dict")


def upgrade_bundle(
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    policy_spec: Path,
    policy_spec_sha256: str,
    walk_controller: Path,
    walk_controller_sha256: str,
    stand_controller: Path,
    stand_controller_sha256: str,
    output_dir: Path,
) -> dict[str, Any]:
    source_checkpoint = _trusted_regular(
        checkpoint, checkpoint_sha256, "legacy checkpoint"
    )
    source_spec_path = _trusted_regular(
        policy_spec, policy_spec_sha256, "legacy policy spec"
    )
    walk_path = _trusted_regular(
        walk_controller, walk_controller_sha256, "Homie walk controller"
    )
    stand_path = _trusted_regular(
        stand_controller, stand_controller_sha256, "Homie stand controller"
    )
    if output_dir.exists() or output_dir.is_symlink():
        raise BootstrapUpgradeError("output directory already exists")
    if not output_dir.is_absolute():
        raise BootstrapUpgradeError("output directory must be absolute")

    source_spec = _load_json(source_spec_path, "legacy policy spec")
    if source_spec.get("schema") != LEGACY_POLICY_SPEC_SCHEMA:
        raise BootstrapUpgradeError("legacy policy spec schema mismatch")
    if source_spec.get("checkpoint_sha256") != sha256_file(source_checkpoint):
        raise BootstrapUpgradeError("legacy policy spec/checkpoint mismatch")

    import torch

    # Pickle is allowed only after the exact source bytes were verified above.
    checkpoint_document = torch.load(
        source_checkpoint, map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint_document, Mapping):
        raise BootstrapUpgradeError("legacy checkpoint root must be a mapping")
    state = _policy_state(checkpoint_document)
    policy_state_sha256 = runtime_state_dict_sha256(state)
    if source_spec.get("runtime_policy_sha256") != policy_state_sha256:
        raise BootstrapUpgradeError(
            "legacy policy spec runtime identity differs from checkpoint state"
        )
    controller_sha256 = {
        "walk": sha256_file(walk_path),
        "stand": sha256_file(stand_path),
    }
    runtime_policy_sha256 = runtime_policy_identity_sha256(
        policy_state_sha256, controller_sha256
    )

    migrated_checkpoint = dict(checkpoint_document)
    existing_controllers = migrated_checkpoint.get("homie_controller_sha256")
    if existing_controllers is not None and existing_controllers != controller_sha256:
        raise BootstrapUpgradeError(
            "legacy checkpoint already binds different Homie controllers"
        )
    migrated_checkpoint["homie_controller_sha256"] = controller_sha256

    output_parent = output_dir.parent
    if (
        not output_parent.exists()
        or output_parent.is_symlink()
        or not output_parent.is_dir()
    ):
        raise BootstrapUpgradeError(
            "output directory parent must be an existing non-symlink directory"
        )
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging.", dir=output_parent)
    )
    staging_dir.chmod(0o700)
    output_checkpoint = staging_dir / "student-v0.pt"
    temporary_checkpoint = staging_dir / f".student-v0.pt.tmp.{os.getpid()}"
    try:
        torch.save(migrated_checkpoint, temporary_checkpoint)
        with temporary_checkpoint.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_checkpoint, output_checkpoint)
    finally:
        temporary_checkpoint.unlink(missing_ok=True)
    output_checkpoint_sha256 = sha256_file(output_checkpoint)

    migrated_spec = dict(source_spec)
    migrated_spec.update(
        {
            "schema": POLICY_SPEC_SCHEMA,
            "checkpoint_sha256": output_checkpoint_sha256,
            "policy_state_sha256": policy_state_sha256,
            "homie_controller_sha256": controller_sha256,
            "runtime_policy_sha256": runtime_policy_sha256,
        }
    )
    migrated_spec = validate_policy_spec(migrated_spec)
    try:
        output_spec = Path(f"{output_checkpoint}.policy_spec.json")
        _atomic_json(output_spec, migrated_spec)
        output_spec_sha256 = sha256_file(output_spec)
        manifest = {
            "schema": BOOTSTRAP_MANIFEST_SCHEMA,
            "checkpoint_path": output_checkpoint.name,
            "checkpoint_sha256": output_checkpoint_sha256,
            "runtime_policy_sha256": runtime_policy_sha256,
            "policy_spec_path": output_spec.name,
            "policy_spec_sha256": output_spec_sha256,
        }
        output_manifest = Path(f"{output_checkpoint}.manifest.json")
        _atomic_json(output_manifest, manifest)

        for path in (output_checkpoint, output_spec, output_manifest):
            path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        report = validate_policy_bundle_identity(
            checkpoint=output_checkpoint,
            policy_spec=output_spec,
            manifest=output_manifest,
            expected_checkpoint_sha256=output_checkpoint_sha256,
            expected_runtime_policy_sha256=runtime_policy_sha256,
        )
        os.replace(staging_dir, output_dir)
    except BaseException:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    output_checkpoint = output_dir / output_checkpoint.name
    output_spec = output_dir / output_spec.name
    output_manifest = output_dir / output_manifest.name
    return {
        "schema": "fast_roborender.round_dagger.bootstrap_identity_upgrade.v1",
        "source_checkpoint_sha256": sha256_file(source_checkpoint),
        "source_policy_spec_sha256": sha256_file(source_spec_path),
        "checkpoint": str(output_checkpoint),
        "checkpoint_sha256": output_checkpoint_sha256,
        "policy_spec": str(output_spec),
        "policy_spec_sha256": output_spec_sha256,
        "manifest": str(output_manifest),
        "manifest_sha256": sha256_file(output_manifest),
        "policy_state_sha256": policy_state_sha256,
        "homie_controller_sha256": controller_sha256,
        "runtime_policy_sha256": runtime_policy_sha256,
        "preflight": report,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--policy-spec", type=Path, required=True)
    parser.add_argument("--policy-spec-sha256", required=True)
    parser.add_argument("--walk-controller", type=Path, required=True)
    parser.add_argument("--walk-controller-sha256", required=True)
    parser.add_argument("--stand-controller", type=Path, required=True)
    parser.add_argument("--stand-controller-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    report = upgrade_bundle(
        checkpoint=arguments.checkpoint,
        checkpoint_sha256=arguments.checkpoint_sha256,
        policy_spec=arguments.policy_spec,
        policy_spec_sha256=arguments.policy_spec_sha256,
        walk_controller=arguments.walk_controller,
        walk_controller_sha256=arguments.walk_controller_sha256,
        stand_controller=arguments.stand_controller,
        stand_controller_sha256=arguments.stand_controller_sha256,
        output_dir=arguments.output_dir,
    )
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
