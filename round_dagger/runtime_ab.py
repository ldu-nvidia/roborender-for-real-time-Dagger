"""Immutable same-seed legacy/shadow validation for the round actor runtime.

This module deliberately does not launch Docker, Isaac Sim, or a renderer.  It
prepares two bounded train-validation cohorts, emits the exact environment for
each arm, authorizes that environment inside the actor container, and compares
the resulting immutable rollout shards byte-for-byte.  The ordinary production
campaign is not a supported input to this gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from scripts.round_dagger.verify_sealed_source_snapshot import (
    VerificationError as SourceVerificationError,
)
from scripts.round_dagger.verify_sealed_source_snapshot import (
    verify as verify_sealed_source_snapshot,
)

from .actor_pair import ActorPairError, inspect_actor_rank
from .artifacts import (
    ArtifactError,
    create_cohort,
    load_strict_json,
    publish_immutable_json,
    sha256_file,
    verify_cohort,
)
from .runtime import (
    RUNTIME_ADAPTER_EVIDENCE_ENV,
    RUNTIME_ADAPTER_INPUT_PROFILE_CANONICAL_ZERO_MASK,
    RUNTIME_ADAPTER_INPUT_PROFILE_ENV,
    RUNTIME_ADAPTER_INPUT_PROFILE_LIVE,
    RUNTIME_ADAPTER_MODE_ENV,
    RUNTIME_ADAPTER_SHADOW_ACK,
    RUNTIME_ADAPTER_SHADOW_ACK_ENV,
    RUNTIME_ADAPTER_VALIDATION_ACK,
    RUNTIME_ADAPTER_VALIDATION_ACK_ENV,
)


PLAN_SCHEMA = "fast_roborender.round_dagger.runtime_adapter_ab_plan.v2"
CONFIG_SCHEMA = "fast_roborender.round_dagger.runtime_adapter_ab_config.v2"
ARM_SCHEMA = "fast_roborender.round_dagger.runtime_adapter_ab_arm.v1"
EVIDENCE_SCHEMA = "fast_roborender.round_dagger.runtime_adapter_ab_evidence.v2"
RESULT_SCHEMA = "fast_roborender.round_dagger.runtime_adapter_ab_result.v2"
RUNTIME_EVIDENCE_SCHEMA = "fast_roborender.round_runtime_validation.v2"
TELEMETRY_SCHEMA = "fast_roborender.round_dagger.runtime_adapter_ab_telemetry.v1"
WARMUP_SCHEMA = "fast_roborender.round_dagger.image3f_warmup.v1"
WARMUP_SPEC_SCHEMA = "fast_roborender.round_dagger.runtime_adapter_ab_warmup.v1"

CAMPAIGN_ID = "rtx-runtime-adapter-ab-v1"
MACRO_ROUND = 0
PURPOSE = "train"
ARMS = ("legacy", "shadow")
VALIDATION_ACK = "bounded-train-replay"
AB_ACK = "bounded-runtime-adapter-ab-v1"
RANKS = 1
ENVIRONMENTS_PER_RANK = 1
STEPS_PER_ENVIRONMENT = 2
GLOBAL_ENV_ID = 0
REFERENCE_LAG = 256
SIMULATOR_SEED = 0
IMAGE3F_SEED = 42

VIEW_CONTRACT = {
    "dtype": "uint8",
    "layout": "vertical",
    "shape": [324, 192, 3],
    "views": ["head_color", "hand_left", "hand_right"],
}
ACTION_CONTRACT = {"dtype": "float32", "shape": [26]}

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_PLAN_KEYS = {
    "accelerate",
    "acknowledgement",
    "actor_gpu_id",
    "actor_image",
    "actor_image_id_sha256",
    "anchor_count",
    "anchor_inventory_sha256",
    "anchors",
    "arms",
    "beta_path",
    "beta_sha256",
    "campaign_id",
    "config_path",
    "config_sha256",
    "gr00t_python",
    "gr00t_root",
    "gr00t_revision",
    "gr00t_source_tree_sha256",
    "fast_roborender_revision",
    "fast_roborender_source_tree_sha256",
    "input_profile",
    "macro_round",
    "policy_checkpoint",
    "policy_manifest",
    "policy_sha256",
    "policy_spec",
    "purpose",
    "renderer_sha256",
    "renderer_slot",
    "repository_root",
    "run_id",
    "runtime_policy_sha256",
    "schema",
    "selected_endpoint_descriptor",
    "selected_endpoint_descriptor_sha256",
    "selected_endpoint_manifest",
    "selected_endpoint_manifest_sha256",
    "source_endpoint_manifest",
    "source_endpoint_manifest_sha256",
    "source_attestation_path",
    "source_attestation_sha256",
    "source_file_count",
    "source_inventory_path",
    "source_inventory_sha256",
    "source_manifest_path",
    "source_manifest_sha256",
    "source_snapshot_name",
    "source_total_file_bytes",
    "teacher_checkpoint",
    "teacher_sha256",
    "warmup_anchor",
    "warmup_anchor_sha256",
    "warmup_evidence_path",
    "warmup_stream_id",
}
_ARM_REFERENCE_KEYS = {"arm", "contract_path", "contract_sha256"}
_ARM_KEYS = {
    "actor_gpu_id",
    "arm",
    "campaign_id",
    "cohort_contract_sha256",
    "cohort_dir",
    "cohort_index",
    "config_sha256",
    "endpoint_manifest",
    "endpoint_manifest_sha256",
    "comparison_evidence_path",
    "global_env_id",
    "image3f_run_id",
    "macro_round",
    "purpose",
    "renderer_slot",
    "run_dir",
    "runtime_evidence_path",
    "runtime_adapter_mode",
    "schema",
    "shadow_acknowledgement",
}
_INPUT_PROFILES = frozenset(
    {
        RUNTIME_ADAPTER_INPUT_PROFILE_LIVE,
        RUNTIME_ADAPTER_INPUT_PROFILE_CANONICAL_ZERO_MASK,
    }
)


class RuntimeABError(RuntimeError):
    """The bounded legacy/shadow A/B contract is unsafe or incomplete."""


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RuntimeABError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _token(value: Any, label: str) -> str:
    if not isinstance(value, str) or _TOKEN_RE.fullmatch(value) is None:
        raise RuntimeABError(f"{label} must be a safe non-empty token")
    return value


def _input_profile(value: Any) -> str:
    if not isinstance(value, str) or value not in _INPUT_PROFILES:
        allowed = ", ".join(sorted(_INPUT_PROFILES))
        raise RuntimeABError(f"runtime A/B input profile must be one of {allowed}")
    return value


def _absolute_file(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() or candidate.is_symlink() or not candidate.is_file():
        raise RuntimeABError(f"{label} must be an absolute regular non-symlink file")
    try:
        sha256_file(candidate)
    except ArtifactError as exc:
        raise RuntimeABError(f"unsafe {label}: {exc}") from exc
    return candidate


def _absolute_directory(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() or candidate.is_symlink() or not candidate.is_dir():
        raise RuntimeABError(
            f"{label} must be an absolute existing non-symlink directory"
        )
    return candidate


def _sealed_source_report(
    repository_root: str | Path, expected_inventory_sha256: str
) -> dict[str, Any]:
    repository = _absolute_directory(repository_root, "sealed repository root")
    if repository.resolve(strict=True) != repository:
        raise RuntimeABError("sealed repository root must be its canonical real path")
    if stat.S_IMODE(os.stat(repository, follow_symlinks=False).st_mode) != 0o555:
        raise RuntimeABError("sealed repository root mode must be exactly 0555")
    expected = _sha(expected_inventory_sha256, "source inventory")
    try:
        report = verify_sealed_source_snapshot(repository, expected)
    except SourceVerificationError as exc:
        raise RuntimeABError(f"sealed repository verification failed: {exc}") from exc
    keys = {
        "attestation_path",
        "attestation_sha256",
        "fast_roborender_revision",
        "fast_roborender_source_tree_sha256",
        "file_count",
        "gr00t_revision",
        "gr00t_source_tree_sha256",
        "inventory_path",
        "inventory_sha256",
        "manifest_sha256",
        "root",
        "schema",
        "snapshot_name",
        "status",
        "total_file_bytes",
    }
    report = _exact_mapping(report, keys, "sealed repository verification report")
    if (
        report["schema"] != "fast_roborender.sealed_source_verification.v1"
        or report["status"] != "PASS"
        or report["root"] != str(repository)
        or report["inventory_sha256"] != expected
    ):
        raise RuntimeABError("sealed repository verification report identity mismatch")
    return report


def _source_plan_binding(report: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(report["root"])
    return {
        "fast_roborender_revision": report["fast_roborender_revision"],
        "fast_roborender_source_tree_sha256": report[
            "fast_roborender_source_tree_sha256"
        ],
        "gr00t_revision": report["gr00t_revision"],
        "gr00t_source_tree_sha256": report["gr00t_source_tree_sha256"],
        "source_attestation_path": report["attestation_path"],
        "source_attestation_sha256": report["attestation_sha256"],
        "source_file_count": report["file_count"],
        "source_inventory_path": report["inventory_path"],
        "source_inventory_sha256": report["inventory_sha256"],
        "source_manifest_path": str(root / ".source_snapshot_manifest.json"),
        "source_manifest_sha256": report["manifest_sha256"],
        "source_snapshot_name": report["snapshot_name"],
        "source_total_file_bytes": report["total_file_bytes"],
    }


def _validate_source_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    report = _sealed_source_report(
        plan["repository_root"], plan["source_inventory_sha256"]
    )
    expected = _source_plan_binding(report)
    mismatches = sorted(key for key, value in expected.items() if plan.get(key) != value)
    if mismatches:
        raise RuntimeABError(
            "sealed repository evidence differs from immutable plan: "
            + ", ".join(mismatches)
        )
    gr00t = _absolute_directory(plan["gr00t_root"], "sealed GR00T root")
    required_gr00t = Path(report["root"]) / "GR00T-VisualSim2Real"
    if gr00t != required_gr00t or gr00t.resolve(strict=True) != required_gr00t:
        raise RuntimeABError(
            "GR00T root must equal sealed repository_root/GR00T-VisualSim2Real"
        )
    return report


def _exact_mapping(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        actual = set(value) if isinstance(value, Mapping) else set()
        raise RuntimeABError(
            f"{label} key-set mismatch: missing={sorted(keys - actual)} "
            f"unknown={sorted(actual - keys)}"
        )
    return dict(value)


def _publish_immutable_text(path: Path, text: str) -> str:
    payload = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise RuntimeABError(f"symbolic links are forbidden: {path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o444)
    except FileExistsError:
        if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
            raise RuntimeABError(f"immutable artifact conflict at {path}")
    else:
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fchmod(handle.fileno(), 0o444)
                os.fsync(handle.fileno())
        except Exception:
            path.unlink(missing_ok=True)
            raise
    return sha256_file(path)


def _file_identity(path: Path) -> tuple[int, ...]:
    info = os.stat(path, follow_symlinks=False)
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def anchor_inventory(path: str | Path) -> tuple[int, str]:
    """Return the production anchor inventory count and aggregate SHA-256."""

    root = _absolute_directory(path, "anchor pool")
    entries = sorted(root.iterdir(), key=lambda item: item.name)
    if any(item.is_symlink() for item in entries):
        raise RuntimeABError("anchor pool contains a symbolic link")
    anchors = tuple(item for item in entries if item.name.endswith(".npz"))
    if not anchors or any(not item.is_file() for item in anchors):
        raise RuntimeABError("anchor pool must contain regular top-level NPZ files")
    before = {item.name: _file_identity(item) for item in entries}
    aggregate = hashlib.sha256()
    for anchor in anchors:
        digest = sha256_file(anchor)
        size = anchor.stat(follow_symlinks=False).st_size
        relative = anchor.name.encode("utf-8")
        aggregate.update(len(relative).to_bytes(8, "big"))
        aggregate.update(relative)
        aggregate.update(size.to_bytes(8, "big"))
        aggregate.update(bytes.fromhex(digest))
    after_entries = sorted(root.iterdir(), key=lambda item: item.name)
    after = {item.name: _file_identity(item) for item in after_entries}
    if before != after:
        raise RuntimeABError("anchor pool changed while its inventory was computed")
    return len(anchors), aggregate.hexdigest()


def _select_warmup_anchor(path: Path) -> Path:
    anchors = tuple(sorted(path.glob("*.npz"), key=lambda item: item.name))
    if not anchors:
        raise RuntimeABError("anchor pool has no warm-up candidate")
    anchor = _absolute_file(anchors[0], "warm-up anchor")
    try:
        with np.load(anchor, allow_pickle=False) as archive:
            required = {
                "prompt",
                "gen__head_color",
                "gen__hand_left",
                "gen__hand_right",
            }
            if not required.issubset(archive.files):
                raise RuntimeABError(
                    "warm-up anchor lacks prompt or required three generated views"
                )
            prompt = archive["prompt"]
            if prompt.size != 1:
                raise RuntimeABError("warm-up anchor prompt must be scalar")
            for view in VIEW_CONTRACT["views"]:
                value = archive[f"gen__{view}"]
                if (
                    value.dtype != np.uint8
                    or value.ndim != 4
                    or value.shape[0] < 1
                    or value.shape[1:] != (240, 416, 3)
                ):
                    raise RuntimeABError(
                        f"warm-up anchor {view} must be uint8 [N>=1,240,416,3]"
                    )
    except RuntimeABError:
        raise
    except Exception as exc:
        raise RuntimeABError(f"cannot validate warm-up anchor: {exc}") from exc
    return anchor


def _bootstrap_identity(checkpoint: Path) -> tuple[Path, Path, str]:
    spec = _absolute_file(Path(f"{checkpoint}.policy_spec.json"), "policy spec")
    manifest_path = _absolute_file(
        Path(f"{checkpoint}.manifest.json"), "policy manifest"
    )
    manifest = load_strict_json(manifest_path)
    keys = {
        "schema",
        "checkpoint_path",
        "checkpoint_sha256",
        "runtime_policy_sha256",
        "policy_spec_path",
        "policy_spec_sha256",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != keys:
        raise RuntimeABError("bootstrap policy manifest key-set mismatch")
    if manifest.get("schema") != "fast_roborender.round_dagger.bootstrap_policy.v1":
        raise RuntimeABError("bootstrap policy manifest schema mismatch")
    if manifest.get("checkpoint_path") != checkpoint.name:
        raise RuntimeABError("bootstrap manifest/checkpoint path mismatch")
    if manifest.get("policy_spec_path") != spec.name:
        raise RuntimeABError("bootstrap manifest/spec path mismatch")
    if manifest.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise RuntimeABError("bootstrap manifest/checkpoint digest mismatch")
    if manifest.get("policy_spec_sha256") != sha256_file(spec):
        raise RuntimeABError("bootstrap manifest/spec digest mismatch")
    return spec, manifest_path, _sha(
        manifest.get("runtime_policy_sha256"), "runtime policy identity"
    )


def _endpoint_descriptors(path: Path) -> tuple[Path, ...]:
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    if not lines or raw != "\n".join(lines) + "\n":
        raise RuntimeABError("source endpoint manifest must be canonical and non-empty")
    if len(lines) != len(set(lines)) or any(not line or line.strip() != line for line in lines):
        raise RuntimeABError("source endpoint manifest repeats or mangles an endpoint")
    result = []
    for index, line in enumerate(lines):
        descriptor = _absolute_file(line, f"renderer {index} endpoint descriptor")
        fields = descriptor.read_text(encoding="utf-8").split()
        if len(fields) != 2 or fields[0] != "127.0.0.1":
            raise RuntimeABError("runtime A/B requires local 127.0.0.1 renderers")
        try:
            port = int(fields[1])
        except ValueError as exc:
            raise RuntimeABError("renderer endpoint port is not an integer") from exc
        if not 1 <= port <= 65535:
            raise RuntimeABError("renderer endpoint port is outside 1..65535")
        result.append(descriptor)
    return tuple(result)


def _expected_config(
    beta_sha256: str,
    source_inventory_sha256: str,
    input_profile: str,
) -> dict[str, Any]:
    return {
        "acknowledgement": VALIDATION_ACK,
        "action_contract": {
            "environment": 31,
            "homie": 15,
            "policy": 11,
            "replay": 26,
        },
        "beta_sha256": beta_sha256,
        "environments_per_rank": ENVIRONMENTS_PER_RANK,
        "image3f_seed": IMAGE3F_SEED,
        "input_profile": _input_profile(input_profile),
        "purpose": PURPOSE,
        "ranks": RANKS,
        "reference_lag": REFERENCE_LAG,
        "rollout_action_source": "beta",
        "schema": CONFIG_SCHEMA,
        "simulator_seed": SIMULATOR_SEED,
        "source_inventory_sha256": source_inventory_sha256,
        "steps_per_environment": STEPS_PER_ENVIRONMENT,
        "validation_only": True,
        "views": VIEW_CONTRACT["views"],
    }


def _expected_arm(
    *,
    root: Path,
    run_id: str,
    arm: str,
    cohort_index: int,
    config_sha256: str,
    endpoint_manifest: Path,
    endpoint_manifest_sha256: str,
    actor_gpu_id: int,
    renderer_slot: int,
    cohort_contract_sha256: str,
) -> dict[str, Any]:
    arm_root = root / arm
    mode = arm
    return {
        "actor_gpu_id": actor_gpu_id,
        "arm": arm,
        "campaign_id": CAMPAIGN_ID,
        "cohort_contract_sha256": cohort_contract_sha256,
        "cohort_dir": str(arm_root / "cohort"),
        "cohort_index": cohort_index,
        "config_sha256": config_sha256,
        "endpoint_manifest": str(endpoint_manifest),
        "endpoint_manifest_sha256": endpoint_manifest_sha256,
        "comparison_evidence_path": str(arm_root / "comparison_evidence.json"),
        "global_env_id": GLOBAL_ENV_ID,
        "image3f_run_id": f"{run_id}-{arm}",
        "macro_round": MACRO_ROUND,
        "purpose": PURPOSE,
        "renderer_slot": renderer_slot,
        "run_dir": str(arm_root / "run"),
        "runtime_evidence_path": str(
            arm_root / "run" / "runtime_adapter_evidence.json"
        ),
        "runtime_adapter_mode": mode,
        "schema": ARM_SCHEMA,
        "shadow_acknowledgement": (
            RUNTIME_ADAPTER_SHADOW_ACK if arm == "shadow" else None
        ),
    }


def prepare_runtime_ab(
    root: str | Path,
    *,
    run_id: str,
    endpoint_manifest: str | Path,
    renderer_slot: int,
    actor_gpu_id: int,
    policy_checkpoint: str | Path,
    teacher_checkpoint: str | Path,
    renderer_sha256: str,
    anchors: str | Path,
    anchor_count: int,
    anchor_inventory_sha256: str,
    repository_root: str | Path,
    source_inventory_sha256: str,
    gr00t_root: str | Path,
    actor_image: str,
    actor_image_id_sha256: str,
    input_profile: str = RUNTIME_ADAPTER_INPUT_PROFILE_LIVE,
    gr00t_python: str = "/opt/round-dagger-rtx/bin/python",
    accelerate: str = "/opt/round-dagger-rtx/bin/accelerate",
) -> dict[str, Any]:
    """Publish an idempotent immutable two-arm A/B plan."""

    target = Path(root).expanduser()
    if (
        not target.is_absolute()
        or target.is_symlink()
        or os.path.normpath(str(target)) != str(target)
    ):
        raise RuntimeABError("runtime A/B root must be an absolute non-symlink path")
    target.mkdir(parents=True, exist_ok=True)
    try:
        resolved_target = target.resolve(strict=True)
    except OSError as exc:
        raise RuntimeABError("runtime A/B root cannot be resolved") from exc
    if not resolved_target.is_dir() or resolved_target != target:
        raise RuntimeABError("runtime A/B root must be its canonical real directory")
    target = resolved_target
    run_token = _token(run_id, "runtime A/B run ID")
    if type(renderer_slot) is not int or renderer_slot < 0:
        raise RuntimeABError("renderer_slot must be a non-negative integer")
    if type(actor_gpu_id) is not int or actor_gpu_id < 0:
        raise RuntimeABError("actor_gpu_id must be a non-negative integer")
    source_manifest = _absolute_file(endpoint_manifest, "source endpoint manifest")
    descriptors = _endpoint_descriptors(source_manifest)
    if renderer_slot >= len(descriptors):
        raise RuntimeABError("renderer_slot is outside the source endpoint manifest")
    descriptor = descriptors[renderer_slot]
    checkpoint = _absolute_file(policy_checkpoint, "policy checkpoint")
    teacher = _absolute_file(teacher_checkpoint, "teacher checkpoint")
    anchor_root = _absolute_directory(anchors, "anchor pool")
    repository = _absolute_directory(repository_root, "repository root")
    gr00t = _absolute_directory(gr00t_root, "GR00T root")
    source_sha = _sha(source_inventory_sha256, "source inventory")
    source_report = _sealed_source_report(repository, source_sha)
    required_gr00t = repository / "GR00T-VisualSim2Real"
    if gr00t != required_gr00t or gr00t.resolve(strict=True) != required_gr00t:
        raise RuntimeABError(
            "GR00T root must equal sealed repository_root/GR00T-VisualSim2Real"
        )
    renderer_sha = _sha(renderer_sha256, "renderer identity")
    image_sha = _sha(actor_image_id_sha256, "actor image identity")
    profile = _input_profile(input_profile)
    if not isinstance(actor_image, str) or not actor_image or any(
        character.isspace() for character in actor_image
    ):
        raise RuntimeABError("actor image must be a non-empty token")
    for executable, label in (
        (gr00t_python, "GR00T Python"),
        (accelerate, "accelerate"),
    ):
        if not Path(executable).is_absolute() or any(
            character in executable for character in "\r\n\0"
        ):
            raise RuntimeABError(f"{label} must be an absolute container path")

    actual_anchor_count, actual_anchor_sha = anchor_inventory(anchor_root)
    if type(anchor_count) is not int or anchor_count <= 0:
        raise RuntimeABError("anchor_count must be a positive integer")
    if actual_anchor_count != anchor_count:
        raise RuntimeABError("anchor pool count differs from the requested contract")
    if actual_anchor_sha != _sha(anchor_inventory_sha256, "anchor inventory"):
        raise RuntimeABError("anchor pool inventory differs from the requested contract")
    warmup_anchor = _select_warmup_anchor(anchor_root)
    spec, manifest, runtime_policy_sha = _bootstrap_identity(checkpoint)

    beta_path = target / "beta.json"
    beta_sha = publish_immutable_json(
        beta_path,
        {
            "schedule": "constant",
            "schema": "fast_roborender.round_dagger.beta.v1",
            "value": 0.5,
        },
    )
    config_path = target / "config.json"
    config_sha = publish_immutable_json(
        config_path, _expected_config(beta_sha, source_sha, profile)
    )
    selected_manifest = target / "renderer_endpoint_manifest.txt"
    selected_manifest_sha = _publish_immutable_text(
        selected_manifest, f"{descriptor}\n"
    )

    arm_references = []
    for cohort_index, arm in enumerate(ARMS):
        arm_root = target / arm
        cohort_dir = arm_root / "cohort"
        create_cohort(
            cohort_dir,
            campaign_id=CAMPAIGN_ID,
            macro_round=MACRO_ROUND,
            cohort_index=cohort_index,
            purpose=PURPOSE,
            expected_ranks=[0],
            expected_global_env_ids=[GLOBAL_ENV_ID],
            expected_transitions=STEPS_PER_ENVIRONMENT,
            view_contract=VIEW_CONTRACT,
            action_contract=ACTION_CONTRACT,
            policy_sha256=sha256_file(checkpoint),
            runtime_policy_sha256=runtime_policy_sha,
            teacher_sha256=sha256_file(teacher),
            renderer_sha256=renderer_sha,
            beta_sha256=beta_sha,
            config_sha256=config_sha,
        )
        arm_document = _expected_arm(
            root=target,
            run_id=run_token,
            arm=arm,
            cohort_index=cohort_index,
            config_sha256=config_sha,
            endpoint_manifest=selected_manifest,
            endpoint_manifest_sha256=selected_manifest_sha,
            actor_gpu_id=actor_gpu_id,
            renderer_slot=renderer_slot,
            cohort_contract_sha256=sha256_file(cohort_dir / "contract.json"),
        )
        arm_path = arm_root / "arm.json"
        arm_sha = publish_immutable_json(arm_path, arm_document)
        arm_references.append(
            {"arm": arm, "contract_path": str(arm_path), "contract_sha256": arm_sha}
        )

    plan = {
        "accelerate": accelerate,
        "acknowledgement": AB_ACK,
        "actor_gpu_id": actor_gpu_id,
        "actor_image": actor_image,
        "actor_image_id_sha256": image_sha,
        "anchor_count": anchor_count,
        "anchor_inventory_sha256": actual_anchor_sha,
        "anchors": str(anchor_root),
        "arms": arm_references,
        "beta_path": str(beta_path),
        "beta_sha256": beta_sha,
        "campaign_id": CAMPAIGN_ID,
        "config_path": str(config_path),
        "config_sha256": config_sha,
        "gr00t_python": gr00t_python,
        "gr00t_root": str(gr00t),
        "input_profile": profile,
        "macro_round": MACRO_ROUND,
        "policy_checkpoint": str(checkpoint),
        "policy_manifest": str(manifest),
        "policy_sha256": sha256_file(checkpoint),
        "policy_spec": str(spec),
        "purpose": PURPOSE,
        "renderer_sha256": renderer_sha,
        "renderer_slot": renderer_slot,
        "repository_root": str(repository),
        "run_id": run_token,
        "runtime_policy_sha256": runtime_policy_sha,
        "schema": PLAN_SCHEMA,
        "selected_endpoint_descriptor": str(descriptor),
        "selected_endpoint_descriptor_sha256": sha256_file(descriptor),
        "selected_endpoint_manifest": str(selected_manifest),
        "selected_endpoint_manifest_sha256": selected_manifest_sha,
        "source_endpoint_manifest": str(source_manifest),
        "source_endpoint_manifest_sha256": sha256_file(source_manifest),
        "teacher_checkpoint": str(teacher),
        "teacher_sha256": sha256_file(teacher),
        "warmup_anchor": str(warmup_anchor),
        "warmup_anchor_sha256": sha256_file(warmup_anchor),
        "warmup_evidence_path": str(target / "warmup_evidence.json"),
        "warmup_stream_id": f"{run_token}-warmup",
        **_source_plan_binding(source_report),
    }
    plan_path = target / "plan.json"
    publish_immutable_json(plan_path, plan)
    return validate_runtime_ab(plan_path)


def _load_plan(path: str | Path) -> tuple[Path, dict[str, Any]]:
    plan_path = _absolute_file(path, "runtime A/B plan")
    plan = _exact_mapping(load_strict_json(plan_path), _PLAN_KEYS, "runtime A/B plan")
    if plan.get("schema") != PLAN_SCHEMA:
        raise RuntimeABError("runtime A/B plan schema mismatch")
    if plan.get("acknowledgement") != AB_ACK:
        raise RuntimeABError("runtime A/B plan acknowledgement mismatch")
    if (
        plan.get("campaign_id") != CAMPAIGN_ID
        or plan.get("macro_round") != MACRO_ROUND
        or plan.get("purpose") != PURPOSE
    ):
        raise RuntimeABError("runtime A/B plan identity mismatch")
    _token(plan.get("run_id"), "runtime A/B run ID")
    _input_profile(plan.get("input_profile"))
    return plan_path, plan


def _load_arms(plan: Mapping[str, Any]) -> dict[str, tuple[Path, dict[str, Any]]]:
    references = plan.get("arms")
    if not isinstance(references, list) or len(references) != len(ARMS):
        raise RuntimeABError("runtime A/B plan must contain exactly two arms")
    result = {}
    for index, expected_arm in enumerate(ARMS):
        reference = _exact_mapping(
            references[index], _ARM_REFERENCE_KEYS, f"{expected_arm} arm reference"
        )
        if reference.get("arm") != expected_arm:
            raise RuntimeABError("runtime A/B arm order must be legacy then shadow")
        path = _absolute_file(reference["contract_path"], f"{expected_arm} arm contract")
        if sha256_file(path) != reference.get("contract_sha256"):
            raise RuntimeABError(f"{expected_arm} arm contract digest mismatch")
        document = _exact_mapping(
            load_strict_json(path), _ARM_KEYS, f"{expected_arm} arm contract"
        )
        result[expected_arm] = path, document
    return result


def _warmup_report(
    plan_path: Path, plan: Mapping[str, Any], anchor_root: Path
) -> dict[str, Any]:
    expected_anchor = _select_warmup_anchor(anchor_root)
    anchor = _absolute_file(plan["warmup_anchor"], "bound warm-up anchor")
    if anchor != expected_anchor or sha256_file(anchor) != plan["warmup_anchor_sha256"]:
        raise RuntimeABError("bound warm-up anchor changed")
    endpoint = _absolute_file(
        plan["selected_endpoint_descriptor"], "bound warm-up endpoint"
    )
    if sha256_file(endpoint) != plan["selected_endpoint_descriptor_sha256"]:
        raise RuntimeABError("bound warm-up endpoint changed")
    stream_id = _token(plan["warmup_stream_id"], "warm-up stream ID")
    if stream_id != f"{plan['run_id']}-warmup":
        raise RuntimeABError("warm-up stream ID is not bound to the A/B run ID")
    evidence_path = Path(plan["warmup_evidence_path"])
    expected_evidence_path = plan_path.parent / "warmup_evidence.json"
    if (
        evidence_path != expected_evidence_path
        or not evidence_path.is_absolute()
        or os.path.normpath(str(evidence_path)) != str(evidence_path)
    ):
        raise RuntimeABError("warm-up evidence path is not canonical under plan root")
    report = {
        "schema": WARMUP_SPEC_SCHEMA,
        "state": "pending",
        "endpoint": str(endpoint),
        "endpoint_sha256": plan["selected_endpoint_descriptor_sha256"],
        "anchor": str(anchor),
        "anchor_sha256": plan["warmup_anchor_sha256"],
        "output": str(evidence_path),
        "renderer_sha256": plan["renderer_sha256"],
        "stream_id": stream_id,
        "seed": IMAGE3F_SEED,
        "reference_lag": REFERENCE_LAG,
        "actor_image": plan["actor_image"],
        "actor_image_id_sha256": plan["actor_image_id_sha256"],
        "gr00t_python": plan["gr00t_python"],
        "evidence_sha256": None,
        "inference_ms": None,
        "client_wall_ms": None,
    }
    if not evidence_path.exists() and not evidence_path.is_symlink():
        return report
    evidence = _absolute_file(evidence_path, "warm-up evidence")
    if stat.S_IMODE(os.stat(evidence, follow_symlinks=False).st_mode) != 0o444:
        raise RuntimeABError("warm-up evidence mode must be exactly 0444")
    digest = sha256_file(evidence)
    document = _exact_mapping(
        load_strict_json(evidence),
        {
            "anchor",
            "anchor_sha256",
            "client_wall_ms",
            "endpoint",
            "endpoint_sha256",
            "generated_sha256",
            "inference_ms",
            "reference_lag",
            "reference_source",
            "renderer_sha256",
            "replayed",
            "schema",
            "seed",
            "stream_id",
        },
        "warm-up evidence",
    )
    if sha256_file(evidence) != digest:
        raise RuntimeABError("warm-up evidence changed while it was loaded")
    expected = {
        "schema": WARMUP_SCHEMA,
        "endpoint": str(endpoint),
        "endpoint_sha256": plan["selected_endpoint_descriptor_sha256"],
        "anchor": str(anchor),
        "anchor_sha256": plan["warmup_anchor_sha256"],
        "renderer_sha256": plan["renderer_sha256"],
        "stream_id": stream_id,
        "seed": IMAGE3F_SEED,
        "reference_lag": REFERENCE_LAG,
        "replayed": False,
        "reference_source": "anchor",
    }
    mismatches = sorted(
        key
        for key, value in expected.items()
        if document.get(key) != value or type(document.get(key)) is not type(value)
    )
    if mismatches:
        raise RuntimeABError(
            "warm-up evidence differs from immutable plan: " + ", ".join(mismatches)
        )
    inference_ms = _finite_number(
        document["inference_ms"], "warm-up inference_ms", positive=True
    )
    client_wall_ms = _finite_number(
        document["client_wall_ms"], "warm-up client_wall_ms", positive=True
    )
    generated = _exact_mapping(
        document["generated_sha256"], set(VIEW_CONTRACT["views"]), "warm-up outputs"
    )
    for view, value in generated.items():
        _sha(value, f"warm-up {view} output")
    return {
        **report,
        "state": "validated",
        "evidence_sha256": digest,
        "inference_ms": inference_ms,
        "client_wall_ms": client_wall_ms,
    }


def validate_runtime_ab(path: str | Path) -> dict[str, Any]:
    """Revalidate every input, contract, and current arm state."""

    plan_path, plan = _load_plan(path)
    if type(plan.get("actor_gpu_id")) is not int or plan["actor_gpu_id"] < 0:
        raise RuntimeABError("actor GPU ID is malformed")
    if type(plan.get("renderer_slot")) is not int or plan["renderer_slot"] < 0:
        raise RuntimeABError("renderer slot is malformed")
    profile = _input_profile(plan.get("input_profile"))
    _sha(plan.get("source_inventory_sha256"), "source inventory")
    _sha(plan.get("renderer_sha256"), "renderer identity")
    _sha(plan.get("actor_image_id_sha256"), "actor image identity")

    checkpoint = _absolute_file(plan["policy_checkpoint"], "policy checkpoint")
    teacher = _absolute_file(plan["teacher_checkpoint"], "teacher checkpoint")
    if sha256_file(checkpoint) != plan.get("policy_sha256"):
        raise RuntimeABError("policy checkpoint changed after plan publication")
    if sha256_file(teacher) != plan.get("teacher_sha256"):
        raise RuntimeABError("teacher checkpoint changed after plan publication")
    spec, manifest, runtime_sha = _bootstrap_identity(checkpoint)
    if str(spec) != plan.get("policy_spec") or str(manifest) != plan.get("policy_manifest"):
        raise RuntimeABError("bootstrap policy bundle paths changed")
    if runtime_sha != plan.get("runtime_policy_sha256"):
        raise RuntimeABError("runtime policy identity changed")
    source_report = _validate_source_plan(plan)

    source_manifest = _absolute_file(
        plan["source_endpoint_manifest"], "source endpoint manifest"
    )
    if sha256_file(source_manifest) != plan.get("source_endpoint_manifest_sha256"):
        raise RuntimeABError("source endpoint manifest changed")
    descriptors = _endpoint_descriptors(source_manifest)
    slot = plan["renderer_slot"]
    if slot >= len(descriptors):
        raise RuntimeABError("renderer slot is outside endpoint manifest")
    descriptor = _absolute_file(
        plan["selected_endpoint_descriptor"], "selected endpoint descriptor"
    )
    if descriptor != descriptors[slot] or sha256_file(descriptor) != plan.get(
        "selected_endpoint_descriptor_sha256"
    ):
        raise RuntimeABError("selected endpoint descriptor changed")
    selected_manifest = _absolute_file(
        plan["selected_endpoint_manifest"], "selected endpoint manifest"
    )
    if selected_manifest.read_text(encoding="utf-8") != f"{descriptor}\n":
        raise RuntimeABError("selected endpoint manifest mapping changed")
    if sha256_file(selected_manifest) != plan.get("selected_endpoint_manifest_sha256"):
        raise RuntimeABError("selected endpoint manifest digest changed")

    anchors = _absolute_directory(plan["anchors"], "anchor pool")
    count, inventory = anchor_inventory(anchors)
    if count != plan.get("anchor_count") or inventory != plan.get(
        "anchor_inventory_sha256"
    ):
        raise RuntimeABError("anchor pool count or inventory changed")
    warmup = _warmup_report(plan_path, plan, anchors)
    beta_path = _absolute_file(plan["beta_path"], "beta document")
    if sha256_file(beta_path) != plan.get("beta_sha256"):
        raise RuntimeABError("beta document digest changed")
    if load_strict_json(beta_path) != {
        "schedule": "constant",
        "schema": "fast_roborender.round_dagger.beta.v1",
        "value": 0.5,
    }:
        raise RuntimeABError("beta document content mismatch")
    config_path = _absolute_file(plan["config_path"], "runtime A/B config")
    if sha256_file(config_path) != plan.get("config_sha256"):
        raise RuntimeABError("runtime A/B config digest changed")
    if load_strict_json(config_path) != _expected_config(
        plan["beta_sha256"], plan["source_inventory_sha256"], profile
    ):
        raise RuntimeABError("runtime A/B shared config content mismatch")

    arms = _load_arms(plan)
    arm_reports = {}
    states = []
    for cohort_index, arm in enumerate(ARMS):
        arm_path, document = arms[arm]
        cohort_dir = _absolute_directory(document["cohort_dir"], f"{arm} cohort")
        expected = _expected_arm(
            root=plan_path.parent,
            run_id=plan["run_id"],
            arm=arm,
            cohort_index=cohort_index,
            config_sha256=plan["config_sha256"],
            endpoint_manifest=selected_manifest,
            endpoint_manifest_sha256=plan["selected_endpoint_manifest_sha256"],
            actor_gpu_id=plan["actor_gpu_id"],
            renderer_slot=plan["renderer_slot"],
            cohort_contract_sha256=sha256_file(cohort_dir / "contract.json"),
        )
        if document != expected:
            raise RuntimeABError(f"{arm} arm contract content mismatch")
        contract = load_strict_json(cohort_dir / "contract.json")
        expected_contract = {
            "campaign_id": CAMPAIGN_ID,
            "macro_round": MACRO_ROUND,
            "cohort_index": cohort_index,
            "purpose": PURPOSE,
            "expected_ranks": [0],
            "expected_global_env_ids": [GLOBAL_ENV_ID],
            "expected_transitions": STEPS_PER_ENVIRONMENT,
            "expected_steps_per_env": STEPS_PER_ENVIRONMENT,
            "view_contract": VIEW_CONTRACT,
            "action_contract": ACTION_CONTRACT,
            "policy_sha256": plan["policy_sha256"],
            "runtime_policy_sha256": plan["runtime_policy_sha256"],
            "teacher_sha256": plan["teacher_sha256"],
            "renderer_sha256": plan["renderer_sha256"],
            "beta_sha256": plan["beta_sha256"],
            "config_sha256": plan["config_sha256"],
        }
        mismatches = [
            key for key, value in expected_contract.items() if contract.get(key) != value
        ]
        if mismatches:
            raise RuntimeABError(f"{arm} cohort contract mismatch: {mismatches}")
        if (cohort_dir / "seal.json").exists():
            sealed = verify_cohort(cohort_dir)
            if (
                sealed.get("purpose") != PURPOSE
                or sealed.get("ranks") != [0]
                or sealed.get("global_env_ids") != [GLOBAL_ENV_ID]
                or sealed.get("transition_count") != STEPS_PER_ENVIRONMENT
            ):
                raise RuntimeABError(f"{arm} cohort seal coverage mismatch")
            state = "sealed"
        else:
            state = inspect_actor_rank(cohort_dir, 0)["state"]
        states.append(state)
        arm_reports[arm] = {
            "arm_contract": str(arm_path),
            "arm_contract_sha256": sha256_file(arm_path),
            "cohort_dir": str(cohort_dir),
            "comparison_evidence_path": document["comparison_evidence_path"],
            "image3f_run_id": document["image3f_run_id"],
            "runtime_evidence_path": document["runtime_evidence_path"],
            "runtime_adapter_mode": document["runtime_adapter_mode"],
            "state": state,
        }
    if arm_reports["legacy"]["image3f_run_id"] == arm_reports["shadow"]["image3f_run_id"]:
        raise RuntimeABError("legacy and shadow IMAGE3F run IDs must be distinct")
    return {
        "schema": "fast_roborender.round_dagger.runtime_adapter_ab_validation.v2",
        "plan": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "campaign_id": CAMPAIGN_ID,
        "shared_config_sha256": plan["config_sha256"],
        "actor_gpu_id": plan["actor_gpu_id"],
        "renderer_slot": plan["renderer_slot"],
        "global_env_id": GLOBAL_ENV_ID,
        "input_profile": profile,
        "source": {
            **_source_plan_binding(source_report),
            "repository_root": source_report["root"],
        },
        "warmup": warmup,
        "arms": arm_reports,
        "states": states,
    }


def runtime_ab_environment(path: str | Path, arm: str) -> dict[str, str]:
    """Return the exact process environment for one still-empty A/B arm."""

    if arm not in ARMS:
        raise RuntimeABError("runtime A/B arm must be legacy or shadow")
    report = validate_runtime_ab(path)
    if report["warmup"]["state"] != "validated":
        raise RuntimeABError(
            "renderer warm-up must be validated before either runtime A/B arm"
        )
    arm_index = ARMS.index(arm)
    if report["states"][arm_index] != "empty":
        raise RuntimeABError(
            f"runtime A/B arm {arm} is {report['states'][arm_index]}; refusing overwrite"
        )
    plan_path, plan = _load_plan(path)
    arm_path, document = _load_arms(plan)[arm]
    return {
        "ACCELERATE": plan["accelerate"],
        "ANCHORS": plan["anchors"],
        "DRY_RUN": "0",
        "GR00T_PYTHON": plan["gr00t_python"],
        "GR00T_ROOT": plan["gr00t_root"],
        "IMAGE3F_ENDPOINT_MANIFEST": document["endpoint_manifest"],
        "IMAGE3F_RUN_ID": document["image3f_run_id"],
        RUNTIME_ADAPTER_INPUT_PROFILE_ENV: plan["input_profile"],
        "ROBORENDER_TELEMETRY_DIR": f"{document['run_dir']}/telemetry",
        "ROLLOUT_STEPS": str(STEPS_PER_ENVIRONMENT),
        "ROUND_DAGGER_ACTOR_DOCKER_GPU_ID": str(document["actor_gpu_id"]),
        "ROUND_DAGGER_BETA": "0.5",
        "ROUND_DAGGER_BETA_SHA256": plan["beta_sha256"],
        "ROUND_DAGGER_CAMPAIGN_ID": CAMPAIGN_ID,
        "ROUND_DAGGER_COHORT_DIR": document["cohort_dir"],
        "ROUND_DAGGER_COHORT_INDEX": str(document["cohort_index"]),
        "ROUND_DAGGER_CONFIG_SHA256": plan["config_sha256"],
        "ROUND_DAGGER_ENVS_PER_SHARD": "1",
        "ROUND_DAGGER_EXPECTED_RENDERERS": "1",
        "ROUND_DAGGER_GLOBAL_ENV_OFFSET": str(GLOBAL_ENV_ID),
        "ROUND_DAGGER_IMAGE3F_SEED": str(IMAGE3F_SEED),
        "ROUND_DAGGER_MACRO_ROUND": str(MACRO_ROUND),
        "ROUND_DAGGER_MIN_ANCHORS": str(plan["anchor_count"]),
        "ROUND_DAGGER_ANCHOR_INVENTORY_SHA256": plan[
            "anchor_inventory_sha256"
        ],
        "ROUND_DAGGER_POLICY_CHECKPOINT": plan["policy_checkpoint"],
        "ROUND_DAGGER_POLICY_MANIFEST": plan["policy_manifest"],
        "ROUND_DAGGER_POLICY_SHA256": plan["policy_sha256"],
        "ROUND_DAGGER_POLICY_SPEC": plan["policy_spec"],
        "ROUND_DAGGER_RENDERER_SHA256": plan["renderer_sha256"],
        "ROUND_DAGGER_REPOSITORY_ROOT": plan["repository_root"],
        "ROUND_DAGGER_ROLLOUT_ID": "0",
        "ROUND_DAGGER_RTX_IMAGE": plan["actor_image"],
        "ROUND_DAGGER_RTX_IMAGE_ID_SHA256": plan["actor_image_id_sha256"],
        "ROUND_DAGGER_RUNTIME_AB_ARM": arm,
        "ROUND_DAGGER_RUNTIME_AB_ARM_CONTRACT": str(arm_path),
        "ROUND_DAGGER_RUNTIME_AB_ARM_CONTRACT_SHA256": sha256_file(arm_path),
        "ROUND_DAGGER_RUNTIME_AB_EVIDENCE": document[
            "comparison_evidence_path"
        ],
        "ROUND_DAGGER_RUNTIME_AB_PLAN": str(plan_path),
        "ROUND_DAGGER_RUNTIME_AB_PLAN_SHA256": report["plan_sha256"],
        "ROUND_DAGGER_RUNTIME_AB_WARMUP_EVIDENCE": report["warmup"]["output"],
        "ROUND_DAGGER_RUNTIME_AB_WARMUP_EVIDENCE_SHA256": report["warmup"][
            "evidence_sha256"
        ],
        RUNTIME_ADAPTER_EVIDENCE_ENV: document["runtime_evidence_path"],
        RUNTIME_ADAPTER_MODE_ENV: document["runtime_adapter_mode"],
        RUNTIME_ADAPTER_SHADOW_ACK_ENV: document["shadow_acknowledgement"] or "",
        RUNTIME_ADAPTER_VALIDATION_ACK_ENV: RUNTIME_ADAPTER_VALIDATION_ACK,
        "ROUND_DAGGER_RUNTIME_POLICY_SHA256": plan["runtime_policy_sha256"],
        "ROUND_DAGGER_SIMULATOR_SEED": str(SIMULATOR_SEED),
        "ROUND_DAGGER_SOURCE_INVENTORY_SHA256": plan["source_inventory_sha256"],
        "ROUND_DAGGER_TEACHER_CHECKPOINT": plan["teacher_checkpoint"],
        "ROUND_DAGGER_TEACHER_SHA256": plan["teacher_sha256"],
        "ROUND_DAGGER_VALIDATION_ENVS_PER_RANK": str(ENVIRONMENTS_PER_RANK),
        "ROUND_DAGGER_VALIDATION_PURPOSE": PURPOSE,
        "ROUND_DAGGER_VALIDATION_RANKS": str(RANKS),
        "ROUND_DAGGER_VALIDATION_STEPS": str(STEPS_PER_ENVIRONMENT),
        "ROUND_DAGGER_VALIDATION_TRAIN_ACK": VALIDATION_ACK,
        "TRAIN_RUN_DIR": document["run_dir"],
    }


def runtime_ab_warmup(path: str | Path) -> dict[str, Any]:
    """Return the immutable renderer warm-up invocation and current state."""

    return validate_runtime_ab(path)["warmup"]


def authorize_runtime_ab(
    path: str | Path,
    arm: str,
    expected_plan_sha256: str,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Fail closed unless an actor process has the plan's exact environment."""

    expected_sha = _sha(expected_plan_sha256, "expected runtime A/B plan")
    expected_environment = runtime_ab_environment(path, arm)
    if expected_environment["ROUND_DAGGER_RUNTIME_AB_PLAN_SHA256"] != expected_sha:
        raise RuntimeABError("runtime A/B plan SHA-256 differs from authorization")
    observed = os.environ if environment is None else environment
    mismatches = [
        name
        for name, value in expected_environment.items()
        if observed.get(name) != value
    ]
    if mismatches:
        raise RuntimeABError(
            "runtime A/B process environment differs from immutable plan: "
            + ", ".join(sorted(mismatches))
        )
    return {
        "schema": "fast_roborender.round_dagger.runtime_adapter_ab_authorization.v2",
        "status": "authorized",
        "plan": expected_environment["ROUND_DAGGER_RUNTIME_AB_PLAN"],
        "plan_sha256": expected_sha,
        "arm": arm,
        "runtime_adapter_mode": expected_environment[RUNTIME_ADAPTER_MODE_ENV],
        "input_profile": expected_environment[RUNTIME_ADAPTER_INPUT_PROFILE_ENV],
        "image3f_run_id": expected_environment["IMAGE3F_RUN_ID"],
        "expected_transitions": STEPS_PER_ENVIRONMENT,
    }


def _array_sha256(name: str, array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    encoded_name = name.encode("utf-8")
    encoded_dtype = value.dtype.str.encode("ascii")
    digest.update(len(encoded_name).to_bytes(8, "big"))
    digest.update(encoded_name)
    digest.update(len(encoded_dtype).to_bytes(8, "big"))
    digest.update(encoded_dtype)
    digest.update(value.ndim.to_bytes(8, "big"))
    for dimension in value.shape:
        digest.update(int(dimension).to_bytes(8, "big"))
    payload = value.tobytes(order="C")
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
    return digest.hexdigest()


def _rollout_sha256(arrays: Mapping[str, np.ndarray]) -> tuple[str, dict[str, str]]:
    per_array = {name: _array_sha256(name, arrays[name]) for name in sorted(arrays)}
    digest = hashlib.sha256()
    for name, value in per_array.items():
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(value))
    return digest.hexdigest(), per_array


def _require_bool(value: Any, expected: bool, label: str) -> None:
    if type(value) is not bool or value is not expected:
        raise RuntimeABError(f"{label} must be exactly {expected}")


def _require_int(value: Any, expected: int, label: str) -> None:
    if type(value) is not int or value != expected:
        raise RuntimeABError(f"{label} must be exactly {expected}")


def _finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise RuntimeABError(f"{label} must be a finite numeric scalar")
    number = float(value)
    if not np.isfinite(number) or (positive and number <= 0.0):
        qualifier = "positive finite" if positive else "finite"
        raise RuntimeABError(f"{label} must be {qualifier}")
    return number


def _runtime_evidence(
    arm: str,
    document: Mapping[str, Any],
    input_profile: str,
) -> dict[str, Any]:
    """Load and semantically validate trainer-published per-arm evidence."""

    path = _absolute_file(document["runtime_evidence_path"], f"{arm} runtime evidence")
    if stat.S_IMODE(os.stat(path, follow_symlinks=False).st_mode) != 0o444:
        raise RuntimeABError(f"{arm} runtime evidence mode must be exactly 0444")
    expected_parent = Path(document["run_dir"])
    if path.parent != expected_parent or path.resolve(strict=True) != path:
        raise RuntimeABError(f"{arm} runtime evidence is outside its canonical run dir")
    digest = sha256_file(path)
    raw = _exact_mapping(
        load_strict_json(path),
        {
            "adapter_arm",
            "audit",
            "dimensions",
            "input_profile",
            "promotion_gate",
            "rank",
            "replay_export",
            "rollout",
            "schema",
            "status",
            "validation_ack",
            "world_size",
        },
        f"{arm} runtime evidence",
    )
    if sha256_file(path) != digest:
        raise RuntimeABError(f"{arm} runtime evidence changed while it was loaded")
    if raw["schema"] != RUNTIME_EVIDENCE_SCHEMA:
        raise RuntimeABError(f"{arm} runtime evidence schema mismatch")
    if raw["status"] != "informational_only":
        raise RuntimeABError(f"{arm} runtime evidence status mismatch")
    _require_bool(raw["promotion_gate"], False, f"{arm} runtime promotion gate")
    if raw["validation_ack"] != RUNTIME_ADAPTER_VALIDATION_ACK:
        raise RuntimeABError(f"{arm} runtime validation acknowledgement mismatch")
    if raw["adapter_arm"] != arm:
        raise RuntimeABError(f"{arm} runtime evidence arm mismatch")
    if raw["input_profile"] != _input_profile(input_profile):
        raise RuntimeABError(f"{arm} runtime evidence input profile mismatch")
    _require_int(raw["rank"], 0, f"{arm} runtime rank")
    _require_int(raw["world_size"], RANKS, f"{arm} runtime world size")

    dimensions = _exact_mapping(
        raw["dimensions"],
        {"local_num_envs", "rollout_steps"},
        f"{arm} runtime dimensions",
    )
    _require_int(
        dimensions["local_num_envs"],
        ENVIRONMENTS_PER_RANK,
        f"{arm} runtime local environments",
    )
    _require_int(
        dimensions["rollout_steps"],
        STEPS_PER_ENVIRONMENT,
        f"{arm} runtime rollout steps",
    )

    rollout = _exact_mapping(
        raw["rollout"],
        {"cuda", "synchronized", "wall_seconds"},
        f"{arm} runtime rollout",
    )
    wall_seconds = _finite_number(
        rollout["wall_seconds"], f"{arm} runtime rollout wall_seconds", positive=True
    )
    _require_bool(
        rollout["synchronized"], True, f"{arm} runtime rollout synchronization"
    )
    cuda = _exact_mapping(
        rollout["cuda"],
        {
            "baseline_allocated_bytes",
            "baseline_reserved_bytes",
            "device",
            "enabled",
            "peak_allocated_bytes",
            "peak_delta_allocated_bytes",
            "peak_delta_reserved_bytes",
            "peak_reserved_bytes",
        },
        f"{arm} runtime CUDA metrics",
    )
    _require_bool(cuda["enabled"], True, f"{arm} runtime CUDA enabled")
    if not isinstance(cuda["device"], str) or not cuda["device"].startswith("cuda"):
        raise RuntimeABError(f"{arm} runtime CUDA device must be a CUDA device")
    for key in (
        "baseline_allocated_bytes",
        "baseline_reserved_bytes",
        "peak_allocated_bytes",
        "peak_delta_allocated_bytes",
        "peak_delta_reserved_bytes",
        "peak_reserved_bytes",
    ):
        if type(cuda[key]) is not int or cuda[key] < 0:
            raise RuntimeABError(f"{arm} runtime CUDA {key} must be a non-negative int")
    if cuda["peak_allocated_bytes"] < cuda["baseline_allocated_bytes"]:
        raise RuntimeABError(f"{arm} runtime allocated CUDA peak precedes baseline")
    if cuda["peak_reserved_bytes"] < cuda["baseline_reserved_bytes"]:
        raise RuntimeABError(f"{arm} runtime reserved CUDA peak precedes baseline")
    if cuda["peak_delta_allocated_bytes"] != (
        cuda["peak_allocated_bytes"] - cuda["baseline_allocated_bytes"]
    ):
        raise RuntimeABError(f"{arm} runtime allocated CUDA delta mismatch")
    if cuda["peak_delta_reserved_bytes"] != (
        cuda["peak_reserved_bytes"] - cuda["baseline_reserved_bytes"]
    ):
        raise RuntimeABError(f"{arm} runtime reserved CUDA delta mismatch")

    replay = _exact_mapping(
        raw["replay_export"],
        {"published_shard_parts", "successful"},
        f"{arm} runtime replay export",
    )
    _require_bool(replay["successful"], True, f"{arm} runtime replay export")
    _require_int(
        replay["published_shard_parts"], 1, f"{arm} runtime published shard parts"
    )

    if arm == "legacy":
        audit = _exact_mapping(raw["audit"], {"enabled"}, "legacy runtime audit")
        _require_bool(audit["enabled"], False, "legacy runtime audit enabled")
    else:
        audit = _exact_mapping(
            raw["audit"],
            {
                "batch_dimensions_valid",
                "callback_count",
                "callback_order_valid",
                "decision_count",
                "enabled",
                "expected_steps",
                "num_envs",
                "sequential_steps_valid",
                "transition_count",
            },
            "shadow runtime audit",
        )
        _require_bool(audit["enabled"], True, "shadow runtime audit enabled")
        _require_int(audit["num_envs"], 1, "shadow runtime audit environments")
        _require_int(audit["expected_steps"], 2, "shadow runtime audit expected steps")
        _require_int(audit["decision_count"], 2, "shadow runtime audit decisions")
        _require_int(audit["transition_count"], 2, "shadow runtime audit transitions")
        _require_int(audit["callback_count"], 6, "shadow runtime audit callbacks")
        for key in (
            "batch_dimensions_valid",
            "callback_order_valid",
            "sequential_steps_valid",
        ):
            _require_bool(audit[key], True, f"shadow runtime audit {key}")

    return {
        "path": str(path),
        "sha256": digest,
        "schema": raw["schema"],
        "input_profile": raw["input_profile"],
        "rollout": {
            **rollout,
            "wall_seconds": wall_seconds,
            "cuda": cuda,
        },
        "audit": audit,
        "replay_export": replay,
    }


def _require_metric(
    metrics: Mapping[str, Any], name: str, expected: int, *, arm: str, step: int
) -> None:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise RuntimeABError(f"{arm} telemetry step {step} metric {name} is not numeric")
    if float(value) != float(expected):
        raise RuntimeABError(
            f"{arm} telemetry step {step} metric {name} must equal {expected}"
        )


def _telemetry_evidence(arm: str, document: Mapping[str, Any]) -> dict[str, Any]:
    """Require the exact two scalar renderer records for the bounded rollout."""

    scalar_dir = Path(document["run_dir"]) / "telemetry" / "scalars"
    scalar_dir = _absolute_directory(scalar_dir, f"{arm} scalar telemetry directory")
    if scalar_dir.resolve(strict=True) != scalar_dir:
        raise RuntimeABError(f"{arm} scalar telemetry directory is not canonical")
    expected_paths = tuple(
        scalar_dir / f"step_{step:09d}.json" for step in range(STEPS_PER_ENVIRONMENT)
    )
    actual_paths = tuple(sorted(scalar_dir.glob("step_*.json")))
    if actual_paths != expected_paths:
        raise RuntimeABError(
            f"{arm} scalar telemetry must contain exactly steps 0 and 1"
        )
    records = []
    inference_values = []
    for step, candidate in enumerate(expected_paths):
        path = _absolute_file(candidate, f"{arm} scalar telemetry step {step}")
        digest = sha256_file(path)
        scalar = _exact_mapping(
            load_strict_json(path), {"metrics", "step"}, f"{arm} scalar step {step}"
        )
        if sha256_file(path) != digest:
            raise RuntimeABError(f"{arm} scalar telemetry changed while it was loaded")
        _require_int(scalar["step"], step, f"{arm} scalar document step")
        if not isinstance(scalar["metrics"], Mapping):
            raise RuntimeABError(f"{arm} scalar telemetry metrics must be a mapping")
        metrics = dict(scalar["metrics"])
        inference_ms = _finite_number(
            metrics.get("renderer/inference_ms"),
            f"{arm} telemetry step {step} renderer/inference_ms",
            positive=True,
        )
        for name, expected in (
            ("renderer/reference_source_idempotent_replay", 0),
            ("renderer/reference_provenance_valid", 1),
            ("renderer/stream_step", step),
            ("renderer/parallel_batch_size", 1),
        ):
            _require_metric(metrics, name, expected, arm=arm, step=step)
        inference_values.append(inference_ms)
        records.append(
            {
                "path": str(path),
                "sha256": digest,
                "step": step,
                "renderer_inference_ms": inference_ms,
                "reference_source_idempotent_replay": 0,
                "reference_provenance_valid": 1,
                "stream_step": step,
                "parallel_batch_size": 1,
            }
        )
    return {
        "schema": TELEMETRY_SCHEMA,
        "records": records,
        "renderer_inference_ms": {
            "count": len(inference_values),
            "mean": float(sum(inference_values) / len(inference_values)),
            "values": inference_values,
        },
        "idempotent_replay_count": 0,
        "provenance_valid_count": STEPS_PER_ENVIRONMENT,
    }


def _sealed_arm_arrays(
    plan_sha256: str,
    arm: str,
    document: Mapping[str, Any],
    arm_contract_path: Path,
    input_profile: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cohort_dir = Path(document["cohort_dir"])
    seal = verify_cohort(cohort_dir)
    shards = seal.get("shards")
    if not isinstance(shards, list) or len(shards) != 1:
        raise RuntimeABError(f"{arm} A/B cohort must contain exactly one shard")
    shard = shards[0]
    payload_path = cohort_dir / shard["payload_path"]
    payload = _absolute_file(payload_path, f"{arm} rollout payload")
    expected_payload_sha = shard["payload_sha256"]
    if sha256_file(payload) != expected_payload_sha:
        raise RuntimeABError(f"{arm} rollout payload changed after cohort verification")
    try:
        with np.load(payload, allow_pickle=False) as archive:
            arrays = {
                name: np.ascontiguousarray(archive[name]) for name in archive.files
            }
    except Exception as exc:
        raise RuntimeABError(f"cannot load {arm} rollout payload: {exc}") from exc
    if sha256_file(payload) != expected_payload_sha:
        raise RuntimeABError(f"{arm} rollout payload changed while it was loaded")
    rollout_sha, per_array = _rollout_sha256(arrays)
    evidence = {
        "actor_gpu_id": document["actor_gpu_id"],
        "arm": arm,
        "arm_contract": str(arm_contract_path),
        "arm_contract_sha256": sha256_file(arm_contract_path),
        "array_sha256": per_array,
        "canonical_rollout_sha256": rollout_sha,
        "cohort_seal": seal["manifest_path"],
        "cohort_seal_sha256": seal["cohort_sha256"],
        "config_sha256": document["config_sha256"],
        "global_env_ids": seal["global_env_ids"],
        "image3f_run_id": document["image3f_run_id"],
        "input_profile": _input_profile(input_profile),
        "payload": str(payload),
        "payload_sha256": expected_payload_sha,
        "plan_sha256": plan_sha256,
        "renderer_slot": document["renderer_slot"],
        "runtime_adapter_mode": document["runtime_adapter_mode"],
        "runtime_evidence": _runtime_evidence(arm, document, input_profile),
        "schema": EVIDENCE_SCHEMA,
        "shadow_acknowledgement": document["shadow_acknowledgement"],
        "telemetry": _telemetry_evidence(arm, document),
        "transition_count": seal["transition_count"],
    }
    return arrays, evidence


def _canonical_json_payload(document: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(document),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RuntimeABError(f"comparison document is not canonical JSON: {exc}") from exc


def _preflight_immutable_json(path: Path, document: Mapping[str, Any]) -> str:
    """Prove one target absent or byte-identical before multi-file publication."""

    if not path.is_absolute() or os.path.normpath(str(path)) != str(path):
        raise RuntimeABError("comparison publication path must be canonical and absolute")
    payload = _canonical_json_payload(document)
    digest = hashlib.sha256(payload).hexdigest()
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise RuntimeABError(f"immutable comparison artifact conflict at {path}")
    return digest


def compare_runtime_ab(
    path: str | Path, *, output: str | Path | None = None
) -> dict[str, Any]:
    """Require exact legacy/shadow rollout equality and publish evidence."""

    report = validate_runtime_ab(path)
    if report["states"] != ["sealed", "sealed"]:
        raise RuntimeABError("runtime A/B comparison requires both sealed arms")
    plan_path, plan = _load_plan(path)
    arms = _load_arms(plan)
    material = {}
    evidence_documents = {}
    for arm in ARMS:
        arm_path, document = arms[arm]
        material[arm], evidence_documents[arm] = _sealed_arm_arrays(
            report["plan_sha256"],
            arm,
            document,
            arm_path,
            plan["input_profile"],
        )
    legacy_keys = set(material["legacy"])
    shadow_keys = set(material["shadow"])
    if legacy_keys != shadow_keys:
        raise RuntimeABError(
            "legacy/shadow rollout array key mismatch: "
            f"legacy_only={sorted(legacy_keys - shadow_keys)} "
            f"shadow_only={sorted(shadow_keys - legacy_keys)}"
        )
    mismatches = []
    for name in sorted(legacy_keys):
        legacy = material["legacy"][name]
        shadow = material["shadow"][name]
        if (
            legacy.dtype != shadow.dtype
            or legacy.shape != shadow.shape
            or legacy.tobytes(order="C") != shadow.tobytes(order="C")
        ):
            mismatches.append(name)
    if mismatches:
        raise RuntimeABError(
            "legacy/shadow rollout arrays differ exactly: " + ", ".join(mismatches)
        )
    rollout_sha = evidence_documents["legacy"]["canonical_rollout_sha256"]
    if evidence_documents["shadow"]["canonical_rollout_sha256"] != rollout_sha:
        raise RuntimeABError("legacy/shadow canonical rollout digest mismatch")
    payload_sha = evidence_documents["legacy"]["payload_sha256"]
    if evidence_documents["shadow"]["payload_sha256"] != payload_sha:
        raise RuntimeABError("legacy/shadow deterministic rollout payloads differ")

    legacy_wall_ms = (
        evidence_documents["legacy"]["runtime_evidence"]["rollout"]["wall_seconds"]
        * 1000.0
    )
    shadow_wall_ms = (
        evidence_documents["shadow"]["runtime_evidence"]["rollout"]["wall_seconds"]
        * 1000.0
    )
    performance_ratio = _finite_number(
        shadow_wall_ms / legacy_wall_ms,
        "shadow-over-legacy runtime performance ratio",
        positive=True,
    )
    performance = {
        "legacy_rollout_wall_ms": legacy_wall_ms,
        "shadow_rollout_wall_ms": shadow_wall_ms,
        "shadow_over_legacy_ratio": performance_ratio,
        "peak_delta_allocated_bytes": {
            arm: evidence_documents[arm]["runtime_evidence"]["rollout"]["cuda"][
                "peak_delta_allocated_bytes"
            ]
            for arm in ARMS
        },
        "peak_delta_reserved_bytes": {
            arm: evidence_documents[arm]["runtime_evidence"]["rollout"]["cuda"][
                "peak_delta_reserved_bytes"
            ]
            for arm in ARMS
        },
        "promotion_gate": False,
        "note": "informational exact 1x1x2 train-validation diagnostic only",
    }

    evidence_paths = {
        arm: Path(arms[arm][1]["comparison_evidence_path"]) for arm in ARMS
    }
    result_path = Path(output) if output is not None else plan_path.parent / "result.json"
    if len({*evidence_paths.values(), result_path}) != 3:
        raise RuntimeABError("runtime A/B publication targets must be distinct")
    evidence_receipts = {}
    for arm in ARMS:
        evidence_path = evidence_paths[arm]
        evidence_sha = _preflight_immutable_json(
            evidence_path, evidence_documents[arm]
        )
        evidence_receipts[arm] = {
            "comparison_evidence_path": str(evidence_path),
            "comparison_evidence_sha256": evidence_sha,
            "runtime_evidence_path": evidence_documents[arm]["runtime_evidence"][
                "path"
            ],
            "runtime_evidence_sha256": evidence_documents[arm]["runtime_evidence"][
                "sha256"
            ],
        }
    result = {
        "arms": evidence_receipts,
        "canonical_rollout_sha256": rollout_sha,
        "exact_array_match": True,
        "exact_payload_match": True,
        "global_env_id": GLOBAL_ENV_ID,
        "input_profile": plan["input_profile"],
        "performance": performance,
        "plan": str(plan_path),
        "plan_sha256": report["plan_sha256"],
        "schema": RESULT_SCHEMA,
        "shared_config_sha256": report["shared_config_sha256"],
        "source": report["source"],
        "status": "PASS",
        "transition_count_per_arm": STEPS_PER_ENVIRONMENT,
        "warmup": report["warmup"],
    }
    _preflight_immutable_json(result_path, result)
    for arm in ARMS:
        evidence_sha = publish_immutable_json(
            evidence_paths[arm], evidence_documents[arm]
        )
        if evidence_sha != evidence_receipts[arm]["comparison_evidence_sha256"]:
            raise RuntimeABError(f"{arm} comparison evidence digest changed at publication")
    result_sha = publish_immutable_json(result_path, result)
    return {**result, "result": str(result_path), "result_sha256": result_sha}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--root", type=Path, required=True)
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--endpoint-manifest", type=Path, required=True)
    prepare.add_argument("--renderer-slot", type=int, required=True)
    prepare.add_argument("--actor-gpu-id", type=int, required=True)
    prepare.add_argument("--policy-checkpoint", type=Path, required=True)
    prepare.add_argument("--teacher-checkpoint", type=Path, required=True)
    prepare.add_argument("--renderer-sha256", required=True)
    prepare.add_argument("--anchors", type=Path, required=True)
    prepare.add_argument("--anchor-count", type=int, required=True)
    prepare.add_argument("--anchor-inventory-sha256", required=True)
    prepare.add_argument("--repository-root", type=Path, required=True)
    prepare.add_argument("--source-inventory-sha256", required=True)
    prepare.add_argument("--gr00t-root", type=Path, required=True)
    prepare.add_argument("--actor-image", required=True)
    prepare.add_argument("--actor-image-id-sha256", required=True)
    prepare.add_argument(
        "--input-profile",
        choices=sorted(_INPUT_PROFILES),
        default=RUNTIME_ADAPTER_INPUT_PROFILE_LIVE,
    )
    prepare.add_argument("--gr00t-python", default="/opt/round-dagger-rtx/bin/python")
    prepare.add_argument("--accelerate", default="/opt/round-dagger-rtx/bin/accelerate")
    validate = subparsers.add_parser("validate")
    validate.add_argument("--plan", type=Path, required=True)
    environment = subparsers.add_parser("environment")
    environment.add_argument("--plan", type=Path, required=True)
    environment.add_argument("--arm", choices=ARMS, required=True)
    environment.add_argument("--null", action="store_true")
    warmup = subparsers.add_parser("warmup")
    warmup.add_argument("--plan", type=Path, required=True)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--plan", type=Path, required=True)
    compare.add_argument("--output", type=Path)
    authorize = subparsers.add_parser("authorize")
    authorize.add_argument("--plan", type=Path, required=True)
    authorize.add_argument("--arm", choices=ARMS, required=True)
    authorize.add_argument("--expected-plan-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.operation == "prepare":
        result = prepare_runtime_ab(
            arguments.root,
            run_id=arguments.run_id,
            endpoint_manifest=arguments.endpoint_manifest,
            renderer_slot=arguments.renderer_slot,
            actor_gpu_id=arguments.actor_gpu_id,
            policy_checkpoint=arguments.policy_checkpoint,
            teacher_checkpoint=arguments.teacher_checkpoint,
            renderer_sha256=arguments.renderer_sha256,
            anchors=arguments.anchors,
            anchor_count=arguments.anchor_count,
            anchor_inventory_sha256=arguments.anchor_inventory_sha256,
            repository_root=arguments.repository_root,
            source_inventory_sha256=arguments.source_inventory_sha256,
            gr00t_root=arguments.gr00t_root,
            actor_image=arguments.actor_image,
            actor_image_id_sha256=arguments.actor_image_id_sha256,
            input_profile=arguments.input_profile,
            gr00t_python=arguments.gr00t_python,
            accelerate=arguments.accelerate,
        )
    elif arguments.operation == "validate":
        result = validate_runtime_ab(arguments.plan)
    elif arguments.operation == "environment":
        result = runtime_ab_environment(arguments.plan, arguments.arm)
        if arguments.null:
            for key in sorted(result):
                sys.stdout.buffer.write(key.encode("ascii") + b"\0")
                sys.stdout.buffer.write(result[key].encode("utf-8") + b"\0")
            return 0
    elif arguments.operation == "warmup":
        result = runtime_ab_warmup(arguments.plan)
    elif arguments.operation == "compare":
        result = compare_runtime_ab(arguments.plan, output=arguments.output)
    else:
        result = authorize_runtime_ab(
            arguments.plan,
            arguments.arm,
            arguments.expected_plan_sha256,
        )
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeABError, ActorPairError, ArtifactError, KeyError) as error:
        print(f"runtime A/B gate failed: {error}", file=sys.stderr)
        raise SystemExit(2)


__all__ = [
    "AB_ACK",
    "ARMS",
    "EVIDENCE_SCHEMA",
    "PLAN_SCHEMA",
    "RESULT_SCHEMA",
    "RuntimeABError",
    "anchor_inventory",
    "authorize_runtime_ab",
    "compare_runtime_ab",
    "prepare_runtime_ab",
    "runtime_ab_environment",
    "runtime_ab_warmup",
    "validate_runtime_ab",
]
