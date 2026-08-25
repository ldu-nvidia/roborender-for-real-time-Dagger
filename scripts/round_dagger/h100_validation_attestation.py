#!/usr/bin/env python3
"""Publish fail-closed success evidence for the OCI-NRT H100 live gate."""

from __future__ import annotations

import argparse
import math
import os
import re
import stat
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from round_dagger.artifacts import (  # noqa: E402
    ArtifactError,
    load_strict_json,
    publish_immutable_json,
    sha256_file,
)
from round_dagger.learner import (  # noqa: E402
    CHECKPOINT_SCHEMA,
    RESULT_SCHEMA,
    LearnerBundleConfig,
    LearnerError,
)
from round_dagger.learner_identity import (  # noqa: E402
    LearnerIdentityError,
    learner_wandb_run_id,
)
from round_dagger.policy_bundle import (  # noqa: E402
    PolicyBundleError,
    validate_policy_bundle_identity,
)
from scripts.round_dagger.h100_validation_wandb_readback import (  # noqa: E402
    H100ValidationReadbackError,
    RESUME_EVIDENCE_SCHEMA,
    SCHEMA as WANDB_READBACK_SCHEMA,
    _load_checkpoint as _readback_checkpoint,
    _result as _readback_result,
    _resume_evidence as _readback_resume_evidence,
    _semantic_identity as _readback_semantic_identity,
    _telemetry as _readback_telemetry,
)
from scripts.round_dagger.verify_learner_restart_equivalence import (  # noqa: E402
    REPORT_SCHEMA as RESTART_REPORT_SCHEMA,
)


SCHEMA = "fast_roborender.round_dagger.oci_nrt_h100_validation.v2"
VALIDATION_CONTRACT_SCHEMA = (
    "fast_roborender.round_dagger.validation_learner_contract.v1"
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_CONTRACT_CONTAINER_PATH = (
    "/lustre/fs1/portfolios/general/projects/general_cs_infra__ngcsre/"
    "users/ldu/roborender_h100_build/containers/"
    "roborender-isaac-gr00t-x86_64-sm90-isaac5.1.sqsh"
)
_PRODUCTION_CONTRACT_SHA256 = (
    "b044ba7d5bd5e3c1645e79616f19729438d00f66e2a40b19b4e955ae8d630d90"
)
_CHECKPOINT_FIELDS = {
    "schema",
    "checkpoint_schema",
    "policy_state_dict",
    "policy_parameter_names",
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
    "model_state_dict",
    "optimizer_state_dict",
    "lr_scheduler_state_dict",
    "value_state_dict",
    "rng_states",
}
_RESULT_FIELDS = {
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
_CONTRACT_FIELDS = {
    "schema",
    "purpose",
    "allocation",
    "optimization",
    "restart_validation",
    "cluster",
    "container",
    "wandb",
    "production_target_contract_sha256",
}


class H100ValidationAttestationError(RuntimeError):
    """The staged validation evidence is incomplete or inconsistent."""


def _digest(value: str, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise H100ValidationAttestationError(f"{label} must be a lowercase SHA-256")
    return value


def _regular(value: Path, label: str) -> Path:
    path = Path(os.path.abspath(value.expanduser()))
    if path.is_symlink() or not path.is_file():
        raise H100ValidationAttestationError(
            f"{label} must be a regular non-symlink file"
        )
    return path.resolve(strict=True)


def _directory(value: Path, label: str) -> Path:
    path = Path(os.path.abspath(value.expanduser()))
    if path.is_symlink() or not path.is_dir():
        raise H100ValidationAttestationError(
            f"{label} must be a real directory"
        )
    return path.resolve(strict=True)


def _strict_object(path: Path, label: str) -> tuple[Path, dict[str, Any]]:
    path = _regular(path, label)
    value = load_strict_json(path)
    if not isinstance(value, Mapping):
        raise H100ValidationAttestationError(f"{label} must contain a JSON object")
    return path, dict(value)


def _finite_nonnegative(value: object, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise H100ValidationAttestationError(f"{label} is invalid")
    return float(value)


def _inside(path: Path, root: Path, label: str) -> None:
    try:
        common = Path(os.path.commonpath((str(path), str(root))))
    except ValueError as exc:
        raise H100ValidationAttestationError(
            f"{label} is outside its provenance root"
        ) from exc
    if common != root:
        raise H100ValidationAttestationError(
            f"{label} is outside its provenance root"
        )


def _validation_contract(
    *,
    source_root: Path,
    path: Path,
    expected_sha256: str,
    container: Path,
    container_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    canonical = (
        source_root
        / "configs"
        / "round_dagger"
        / "validation_learner_1x8h100_2step.json"
    ).resolve(strict=True)
    path, document = _strict_object(path, "validation execution contract")
    if path != canonical:
        raise H100ValidationAttestationError(
            "validation execution contract path is non-canonical"
        )
    if path.stat().st_mode & 0o222:
        raise H100ValidationAttestationError(
            "validation execution contract must be read-only"
        )
    if sha256_file(path) != expected_sha256:
        raise H100ValidationAttestationError(
            "validation execution contract digest mismatch"
        )
    if set(document) != _CONTRACT_FIELDS:
        raise H100ValidationAttestationError(
            "validation execution contract field set mismatch"
        )
    exact = {
        "schema": VALIDATION_CONTRACT_SCHEMA,
        "purpose": "live_h100_restart_and_telemetry_gate",
        "allocation": {
            "nodes": 1,
            "gpus_per_node": 8,
            "gpu_architecture": "H100-SM90",
            "backend": "nccl",
            "batch_size_per_rank": 1,
            "global_batch_size": 8,
        },
        "optimization": {
            "optimizer_steps": 2,
            "save_every_steps": 1,
            "intentional_stop_step": 1,
            "learning_rate": 2.5e-5,
            "weight_decay": 0.01,
            "seed": 17,
            "partition_seed": 23,
        },
        "restart_validation": {
            "uninterrupted_comparator": "local_only",
            "wandb_lineage": "single_canonical_run",
            "first_invocation_resume": "allow",
            "second_invocation_resume": "must",
        },
        "cluster": {
            "host": "oci-nrt",
            "account": "general_cs_infra__ngcsre",
            "partition": "batch_singlenode",
            "qos": "normal",
        },
        "container": {
            # The contract records the host-side Lustre image.  Inside the
            # Enroot job those bytes are mounted at a short read-only alias,
            # which is the path received by this attestation process.
            "path": _CONTRACT_CONTAINER_PATH,
            "sha256": container_sha256,
            "python": "/opt/viral-h100/bin/python",
        },
        "wandb": {
            "mode": "online",
            "expected_email": "ldu@nvidia.com",
            "entity": "nvidia",
            "authorization_scope": "team",
            "project": "roborender-round-dagger-validation",
        },
    }
    if any(document.get(key) != value for key, value in exact.items()):
        raise H100ValidationAttestationError(
            "validation execution contract semantics mismatch"
        )
    if (
        document["production_target_contract_sha256"]
        != _PRODUCTION_CONTRACT_SHA256
    ):
        raise H100ValidationAttestationError(
            "production target contract identity mismatch"
        )
    return path, document


def _semantic_identity(
    *,
    campaign_id: str,
    macro_round: int,
    learner_code_sha256: str,
    learner_config_sha256: str,
    model_lineage_sha256: str,
    replay_snapshot_file_sha256: str,
    replay_snapshot_sha256: str,
    parent_runtime_policy_sha256: str,
    learner_identity_sha256: str,
    runtime_container_sha256: str,
    validation_contract_sha256: str,
) -> tuple[str, dict[str, Any]]:
    try:
        run_id = learner_wandb_run_id(
            campaign_id=campaign_id,
            macro_round=macro_round,
            learner_identity_sha256=learner_identity_sha256,
        )
        semantic = _readback_semantic_identity(
            run_id=run_id,
            campaign_id=campaign_id,
            macro_round=macro_round,
            learner_code_sha256=learner_code_sha256,
            learner_config_sha256=learner_config_sha256,
            model_lineage_sha256=model_lineage_sha256,
            replay_snapshot_file_sha256=replay_snapshot_file_sha256,
            replay_snapshot_sha256=replay_snapshot_sha256,
            parent_runtime_policy_sha256=parent_runtime_policy_sha256,
            learner_identity_sha256=learner_identity_sha256,
            runtime_container_sha256=runtime_container_sha256,
            validation_contract_sha256=validation_contract_sha256,
        )
    except (LearnerIdentityError, H100ValidationReadbackError) as exc:
        raise H100ValidationAttestationError(
            f"validation semantic identity mismatch: {exc}"
        ) from exc
    return run_id, semantic


def _checkpoint(
    path: Path,
    *,
    label: str,
    completed_steps: int,
    semantic: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], str]:
    try:
        result = _readback_checkpoint(
            path,
            label=label,
            completed_steps=completed_steps,
            semantic=semantic,
        )
    except H100ValidationReadbackError as exc:
        raise H100ValidationAttestationError(str(exc)) from exc
    checkpoint_path, document, digest = result
    if set(document) != _CHECKPOINT_FIELDS:
        raise H100ValidationAttestationError(
            f"{label} has unexpected or missing v2 fields"
        )
    names = document.get("policy_parameter_names")
    if (
        not isinstance(names, list)
        or any(not isinstance(item, str) for item in names)
        or names != sorted(set(names))
        or not isinstance(document.get("policy_state_dict"), Mapping)
        or not isinstance(document.get("model_state_dict"), Mapping)
        or set(document["policy_state_dict"]) != set(document["model_state_dict"])
    ):
        raise H100ValidationAttestationError(
            f"{label} policy frontier is malformed"
        )
    return checkpoint_path, document, digest


def _result(
    path: Path,
    *,
    label: str,
    semantic: Mapping[str, Any],
    expected_checkpoint_sha256: str,
) -> tuple[Path, dict[str, Any], Path]:
    try:
        result = _readback_result(
            path,
            label=label,
            semantic=semantic,
            expected_checkpoint_sha256=expected_checkpoint_sha256,
        )
    except H100ValidationReadbackError as exc:
        raise H100ValidationAttestationError(str(exc)) from exc
    result_path, document, candidate_path = result
    if set(document) != _RESULT_FIELDS:
        raise H100ValidationAttestationError(
            f"{label} has unexpected or missing v2 fields"
        )
    for field, suffix in (
        ("policy_spec_path", ".policy_spec.json"),
        ("policy_bundle_manifest_path", ".manifest.json"),
    ):
        name = document.get(field)
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or name != f"{document['checkpoint_path']}{suffix}"
        ):
            raise H100ValidationAttestationError(
                f"{label} has a non-canonical policy sidecar path"
            )
        sidecar = _regular(result_path.parent / name, f"{label} {field}")
        expected_digest = document[field.replace("_path", "_sha256")]
        if sha256_file(sidecar) != expected_digest:
            raise H100ValidationAttestationError(
                f"{label} policy sidecar digest mismatch"
            )
    try:
        validate_policy_bundle_identity(
            checkpoint=candidate_path,
            policy_spec=result_path.parent / document["policy_spec_path"],
            manifest=(
                result_path.parent
                / document["policy_bundle_manifest_path"]
            ),
            expected_checkpoint_sha256=document["checkpoint_sha256"],
            expected_runtime_policy_sha256=document["runtime_policy_sha256"],
        )
    except PolicyBundleError as exc:
        raise H100ValidationAttestationError(
            f"{label} candidate policy bundle mismatch: {exc}"
        ) from exc
    return result_path, document, candidate_path


def _telemetry(
    path: Path,
    *,
    campaign_id: str,
    macro_round: int,
) -> tuple[Path, list[dict[str, Any]]]:
    try:
        telemetry_path, rows = _readback_telemetry(
            path, campaign_id=campaign_id, macro_round=macro_round
        )
    except H100ValidationReadbackError as exc:
        raise H100ValidationAttestationError(str(exc)) from exc
    expected_counts = [9600] * 8
    for row in rows:
        if (
            row["replay_transitions_rank"] != 9600
            or row["replay_transitions_per_rank"] != expected_counts
            or row["replay_transitions_global"] != 76800
        ):
            raise H100ValidationAttestationError(
                "validation telemetry does not cover the exact 76,800-transition replay"
            )
    return telemetry_path, rows


def build_attestation(
    *,
    source_root: Path,
    source_inventory_sha256: str,
    container: Path,
    container_sha256: str,
    validation_contract: Path,
    validation_contract_sha256: str,
    learner_identity_sha256: str,
    parent_runtime_policy_sha256: str,
    campaign_root: Path,
    snapshot: Path,
    snapshot_file_sha256: str,
    snapshot_sha256: str,
    replay_shards: int,
    policy_checkpoint: Path,
    policy_checkpoint_sha256: str,
    policy_spec: Path,
    policy_spec_sha256: str,
    policy_manifest: Path,
    policy_manifest_sha256: str,
    bundle: Path,
    config_sha256: str,
    import_report: Path,
    wandb_report: Path,
    wandb_readback: Path,
    uninterrupted_telemetry: Path,
    resumed_telemetry: Path,
    resume_evidence: Path,
    uninterrupted_checkpoint: Path,
    uninterrupted_result: Path,
    interrupted_checkpoint: Path,
    resumed_checkpoint: Path,
    resumed_result: Path,
    restart_report: Path,
    campaign_id: str,
    macro_round: int,
) -> dict[str, Any]:
    source_inventory_sha256 = _digest(
        source_inventory_sha256, "source inventory SHA-256"
    )
    container_sha256 = _digest(container_sha256, "container SHA-256")
    validation_contract_sha256 = _digest(
        validation_contract_sha256, "validation contract SHA-256"
    )
    learner_identity_sha256 = _digest(
        learner_identity_sha256, "learner identity SHA-256"
    )
    parent_runtime_policy_sha256 = _digest(
        parent_runtime_policy_sha256, "parent runtime policy SHA-256"
    )
    snapshot_file_sha256 = _digest(snapshot_file_sha256, "snapshot file SHA-256")
    snapshot_sha256 = _digest(snapshot_sha256, "snapshot semantic SHA-256")
    policy_checkpoint_sha256 = _digest(
        policy_checkpoint_sha256, "policy checkpoint SHA-256"
    )
    policy_spec_sha256 = _digest(policy_spec_sha256, "policy spec SHA-256")
    policy_manifest_sha256 = _digest(
        policy_manifest_sha256, "policy manifest SHA-256"
    )
    config_sha256 = _digest(config_sha256, "learner config SHA-256")
    if type(replay_shards) is not int or replay_shards != 64:
        raise H100ValidationAttestationError(
            "validation replay requires exactly 64 shards"
        )
    if type(macro_round) is not int or macro_round < 0:
        raise H100ValidationAttestationError("macro round is invalid")

    source_root = _directory(source_root, "sealed source root")
    campaign_root = _directory(campaign_root, "campaign root")
    source_manifest_path, source_manifest = _strict_object(
        source_root / ".source_snapshot_manifest.json", "sealed source manifest"
    )
    fast_revision = str(source_manifest.get("fast_roborender_revision"))
    gr00t_revision = str(source_manifest.get("gr00t_revision"))
    expected_snapshot_name = (
        f"fast_roborender_{fast_revision[:7]}_{gr00t_revision[:7]}_sealed"
    )
    if (
        source_manifest.get("schema")
        != "fast_roborender.hsg_source_snapshot.v2"
        or source_manifest.get("status")
        != "sealed_tracked_hydrated_sources"
        or _REVISION.fullmatch(fast_revision) is None
        or _REVISION.fullmatch(gr00t_revision) is None
        or source_manifest.get("snapshot_name") != expected_snapshot_name
        or source_manifest.get("tracked_only") is not True
        or source_manifest.get("symlinks_rejected") is not True
        or source_manifest.get("bytecode_rejected") is not True
        or source_manifest.get("lfs_pointer_oids_verified") is not True
    ):
        raise H100ValidationAttestationError(
            "sealed source manifest provenance mismatch"
        )
    container = _regular(container, "trainer container")
    if sha256_file(container) != container_sha256:
        raise H100ValidationAttestationError("container digest mismatch")
    contract_path, contract = _validation_contract(
        source_root=source_root,
        path=validation_contract,
        expected_sha256=validation_contract_sha256,
        container=container,
        container_sha256=container_sha256,
    )
    project = contract["wandb"]["project"]

    snapshot = _regular(snapshot, "replay snapshot")
    policy_checkpoint = _regular(policy_checkpoint, "parent policy checkpoint")
    policy_spec = _regular(policy_spec, "parent policy spec")
    policy_manifest = _regular(policy_manifest, "parent policy manifest")
    for path, label in (
        (snapshot, "replay snapshot"),
        (policy_checkpoint, "parent policy checkpoint"),
        (policy_spec, "parent policy spec"),
        (policy_manifest, "parent policy manifest"),
    ):
        _inside(path, campaign_root, label)
    if policy_spec != Path(f"{policy_checkpoint}.policy_spec.json"):
        raise H100ValidationAttestationError("parent policy spec path is non-canonical")
    if policy_manifest != Path(f"{policy_checkpoint}.manifest.json"):
        raise H100ValidationAttestationError(
            "parent policy manifest path is non-canonical"
        )
    exact_files = (
        (snapshot, snapshot_file_sha256, "snapshot"),
        (policy_checkpoint, policy_checkpoint_sha256, "parent checkpoint"),
        (policy_spec, policy_spec_sha256, "parent spec"),
        (policy_manifest, policy_manifest_sha256, "parent manifest"),
    )
    for path, expected, label in exact_files:
        if sha256_file(path) != expected:
            raise H100ValidationAttestationError(f"{label} digest mismatch")
    snapshot_document = load_strict_json(snapshot)
    if (
        not isinstance(snapshot_document, Mapping)
        or snapshot_document.get("snapshot_sha256") != snapshot_sha256
        or snapshot_document.get("campaign_id") != campaign_id
        or snapshot_document.get("through_macro_round") != macro_round
        or snapshot_document.get("expected_train_cohorts") != 4
    ):
        raise H100ValidationAttestationError("replay snapshot lineage mismatch")

    bundle = _regular(bundle, "validation learner bundle")
    if sha256_file(bundle) != config_sha256:
        raise H100ValidationAttestationError("learner bundle digest mismatch")
    try:
        bundle_config = LearnerBundleConfig.load(bundle)
    except LearnerError as exc:
        raise H100ValidationAttestationError(
            f"validation learner bundle is invalid: {exc}"
        ) from exc
    model_config = bundle_config.to_mapping()["model_config"]
    expected_bundle = {
        "campaign_id": campaign_id,
        "macro_round": macro_round,
        "code_sha256": source_inventory_sha256,
        "model_lineage_sha256": policy_checkpoint_sha256,
        "initial_checkpoint_sha256": policy_checkpoint_sha256,
        "batch_size": 1,
        "max_steps": 2,
        "save_every": 1,
        "learning_rate": 2.5e-5,
        "weight_decay": 0.01,
        "seed": 17,
        "partition_seed": 23,
    }
    if (
        bundle_config.sha256 != config_sha256
        or any(
            getattr(bundle_config, key) != value
            for key, value in expected_bundle.items()
        )
        or model_config.get("runtime_policy_sha256")
        != parent_runtime_policy_sha256
    ):
        raise H100ValidationAttestationError(
            "validation learner bundle semantics mismatch"
        )

    canonical_run_id, semantic = _semantic_identity(
        campaign_id=campaign_id,
        macro_round=macro_round,
        learner_code_sha256=source_inventory_sha256,
        learner_config_sha256=config_sha256,
        model_lineage_sha256=policy_checkpoint_sha256,
        replay_snapshot_file_sha256=snapshot_file_sha256,
        replay_snapshot_sha256=snapshot_sha256,
        parent_runtime_policy_sha256=parent_runtime_policy_sha256,
        learner_identity_sha256=learner_identity_sha256,
        runtime_container_sha256=container_sha256,
        validation_contract_sha256=validation_contract_sha256,
    )

    import_report_path, import_report_document = _strict_object(
        import_report, "CUDA/import report"
    )
    if (
        import_report_document.get("status") != "PASS"
        or import_report_document.get("visible_cuda_devices") != 1
        or import_report_document.get("compute_capability") != [9, 0]
        or import_report_document.get("nccl_available") is not True
        or import_report_document.get("campaign_mount") != "ro"
        or import_report_document.get("learner_output_mount") != "rw"
    ):
        raise H100ValidationAttestationError("CUDA/import mount gate did not pass")
    wandb_report_path, wandb_report_document = _strict_object(
        wandb_report, "W&B preflight report"
    )
    if (
        wandb_report_document.get("email") != "ldu@nvidia.com"
        or wandb_report_document.get("entity") != "nvidia"
        or wandb_report_document.get("authorization_scope") != "team"
        or wandb_report_document.get("team_member") is not True
        or wandb_report_document.get("entity_is_team") is not True
        or wandb_report_document.get("personal") is not False
        or wandb_report_document.get("mode") != "online"
    ):
        raise H100ValidationAttestationError("W&B preflight identity mismatch")

    uninterrupted_telemetry_path, uninterrupted_rows = _telemetry(
        uninterrupted_telemetry,
        campaign_id=campaign_id,
        macro_round=macro_round,
    )
    resumed_telemetry_path, resumed_rows = _telemetry(
        resumed_telemetry,
        campaign_id=campaign_id,
        macro_round=macro_round,
    )
    if uninterrupted_rows != resumed_rows:
        raise H100ValidationAttestationError(
            "uninterrupted and resumed telemetry histories differ"
        )

    full_checkpoint_path, full_checkpoint, full_checkpoint_sha256 = _checkpoint(
        uninterrupted_checkpoint,
        label="uninterrupted checkpoint",
        completed_steps=2,
        semantic=semantic,
    )
    interrupted_path, interrupted_document, interrupted_sha256 = _checkpoint(
        interrupted_checkpoint,
        label="intentional-stop checkpoint",
        completed_steps=1,
        semantic=semantic,
    )
    if interrupted_path.stat().st_mode & 0o222:
        raise H100ValidationAttestationError(
            "intentional-stop checkpoint must be read-only"
        )
    resumed_checkpoint_path, resumed_checkpoint_document, resumed_checkpoint_sha256 = (
        _checkpoint(
            resumed_checkpoint,
            label="resumed checkpoint",
            completed_steps=2,
            semantic=semantic,
        )
    )
    full_result_path, full_result, full_candidate_path = _result(
        uninterrupted_result,
        label="uninterrupted result",
        semantic=semantic,
        expected_checkpoint_sha256=full_checkpoint_sha256,
    )
    resumed_result_path, resumed_result_document, resumed_candidate_path = _result(
        resumed_result,
        label="resumed result",
        semantic=semantic,
        expected_checkpoint_sha256=resumed_checkpoint_sha256,
    )
    initial_loss = _finite_nonnegative(full_result["initial_loss"], "initial loss")
    final_loss = _finite_nonnegative(full_result["final_loss"], "final loss")
    if (
        full_result["initial_loss"] != resumed_result_document["initial_loss"]
        or full_result["final_loss"] != resumed_result_document["final_loss"]
        or full_checkpoint["initial_loss"] != full_result["initial_loss"]
        or interrupted_document["initial_loss"]
        != resumed_result_document["initial_loss"]
        or resumed_checkpoint_document["initial_loss"]
        != resumed_result_document["initial_loss"]
        or full_result["initial_loss"] != uninterrupted_rows[0]["loss"]
        or resumed_result_document["initial_loss"] != resumed_rows[0]["loss"]
    ):
        raise H100ValidationAttestationError(
            "checkpoint, result, and telemetry loss lineage mismatch"
        )

    try:
        resume_evidence_path, resume_evidence_document = _readback_resume_evidence(
            resume_evidence,
            run_id=canonical_run_id,
            semantic=semantic,
            stopped_checkpoint_sha256=interrupted_sha256,
            resumed_checkpoint_sha256=resumed_checkpoint_sha256,
        )
    except H100ValidationReadbackError as exc:
        raise H100ValidationAttestationError(str(exc)) from exc

    restart_report_path, restart = _strict_object(
        restart_report, "restart equivalence report"
    )
    if (
        restart.get("schema") != RESTART_REPORT_SCHEMA
        or restart.get("semantic_equivalent") is not True
        or restart.get("results_equivalent") is not True
        or restart.get("expected_completed_steps") != 2
        or restart.get("uninterrupted_checkpoint") != str(full_checkpoint_path)
        or restart.get("resumed_checkpoint") != str(resumed_checkpoint_path)
        or restart.get("uninterrupted_result") != str(full_result_path)
        or restart.get("resumed_result") != str(resumed_result_path)
        or restart.get("uninterrupted_checkpoint_sha256")
        != full_checkpoint_sha256
        or restart.get("resumed_checkpoint_sha256")
        != resumed_checkpoint_sha256
        or restart.get("uninterrupted_result_sha256")
        != sha256_file(full_result_path)
        or restart.get("resumed_result_sha256")
        != sha256_file(resumed_result_path)
    ):
        raise H100ValidationAttestationError(
            "restart equivalence evidence mismatch"
        )

    readback_path, readback = _strict_object(
        wandb_readback, "W&B learner readback report"
    )
    if readback_path.stat().st_mode & 0o222:
        raise H100ValidationAttestationError(
            "W&B learner readback report must be read-only"
        )
    semantic_expected = {
        **semantic,
        "validation_contract_sha256": validation_contract_sha256,
    }
    if any(readback.get(key) != value for key, value in semantic_expected.items()):
        raise H100ValidationAttestationError(
            "W&B readback semantic identity mismatch"
        )
    canonical_run = readback.get("canonical_run")
    comparator = readback.get("uninterrupted_comparator")
    resumed_execution = readback.get("resumed_execution")
    expected_hashes = {
        "uninterrupted_telemetry_sha256": sha256_file(
            uninterrupted_telemetry_path
        ),
        "resumed_telemetry_sha256": sha256_file(resumed_telemetry_path),
        "uninterrupted_result_sha256": sha256_file(full_result_path),
        "resumed_result_sha256": sha256_file(resumed_result_path),
        "uninterrupted_checkpoint_sha256": full_checkpoint_sha256,
        "stopped_checkpoint_sha256": interrupted_sha256,
        "resumed_checkpoint_sha256": resumed_checkpoint_sha256,
        "strict_resume_evidence_sha256": sha256_file(resume_evidence_path),
    }
    if (
        readback.get("schema") != WANDB_READBACK_SCHEMA
        or readback.get("status") != "PASS"
        or readback.get("account_email") != "ldu@nvidia.com"
        or readback.get("entity") != "nvidia"
        or readback.get("authorization_scope") != "team"
        or readback.get("team_member") is not True
        or readback.get("entity_is_team") is not True
        or readback.get("personal") is not False
        or readback.get("project") != project
        or readback.get("wandb_run_id") != canonical_run_id
        or readback.get("single_canonical_wandb_run") is not True
        or readback.get("histories_identical") is not True
        or readback.get("resumed_history_contiguous") is not True
        or readback.get("strict_resume_mode") != "must"
        or readback.get("strict_resume_verified") is not True
        or readback.get("initial_loss") != full_result["initial_loss"]
        or readback.get("final_loss") != full_result["final_loss"]
        or any(readback.get(key) != value for key, value in expected_hashes.items())
        or not isinstance(canonical_run, Mapping)
        or canonical_run.get("run_id") != canonical_run_id
        or canonical_run.get("history") != resumed_rows
        or canonical_run.get("history_steps") != [1, 2]
        or canonical_run.get("history_contiguous") is not True
        or canonical_run.get("remote_state") != "finished"
        or canonical_run.get("config_verified") is not True
        or not isinstance(comparator, Mapping)
        or comparator
        != {
            "wandb_enabled": False,
            "history_steps": [1, 2],
            "telemetry_sha256": expected_hashes[
                "uninterrupted_telemetry_sha256"
            ],
            "result_sha256": expected_hashes["uninterrupted_result_sha256"],
            "checkpoint_sha256": full_checkpoint_sha256,
            "candidate_checkpoint_sha256": sha256_file(full_candidate_path),
        }
        or not isinstance(resumed_execution, Mapping)
        or resumed_execution
        != {
            "wandb_enabled": True,
            "history_steps": [1, 2],
            "telemetry_sha256": expected_hashes["resumed_telemetry_sha256"],
            "result_sha256": expected_hashes["resumed_result_sha256"],
            "stopped_checkpoint_sha256": interrupted_sha256,
            "checkpoint_sha256": resumed_checkpoint_sha256,
            "candidate_checkpoint_sha256": sha256_file(
                resumed_candidate_path
            ),
        }
    ):
        raise H100ValidationAttestationError(
            "W&B learner readback evidence mismatch"
        )

    return {
        "schema": SCHEMA,
        "status": "PASS",
        "campaign_id": campaign_id,
        "macro_round": macro_round,
        "source_root": str(source_root),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_inventory_sha256": source_inventory_sha256,
        "container_path": str(container),
        "container_sha256": container_sha256,
        "validation_contract_path": str(contract_path),
        "validation_contract_sha256": validation_contract_sha256,
        "production_target_contract_sha256": contract[
            "production_target_contract_sha256"
        ],
        "campaign_root": str(campaign_root),
        "snapshot_path": str(snapshot),
        "snapshot_file_sha256": snapshot_file_sha256,
        "snapshot_sha256": snapshot_sha256,
        "replay_shards": replay_shards,
        "parent_policy_checkpoint_sha256": policy_checkpoint_sha256,
        "parent_policy_spec_sha256": policy_spec_sha256,
        "parent_policy_manifest_sha256": policy_manifest_sha256,
        "parent_runtime_policy_sha256": parent_runtime_policy_sha256,
        "learner_config_sha256": config_sha256,
        "learner_identity_sha256": learner_identity_sha256,
        "canonical_wandb_run_id": canonical_run_id,
        "learner_execution_contract": {
            "purpose": "validation_only",
            "backend": "nccl",
            "nodes": 1,
            "world_size": 8,
            "batch_size_per_rank": 1,
            "global_batch_size": 8,
            "optimizer_steps": 2,
            "save_every": 1,
            "intentional_stop_step": 1,
            "uninterrupted_comparator_wandb": False,
            "single_canonical_wandb_run": True,
            "strict_resume_mode": "must",
        },
        "cuda_import_report_path": str(import_report_path),
        "cuda_import_report_sha256": sha256_file(import_report_path),
        "wandb_preflight_report_path": str(wandb_report_path),
        "wandb_preflight_report_sha256": sha256_file(wandb_report_path),
        "wandb_learner_readback_path": str(readback_path),
        "wandb_learner_readback_sha256": sha256_file(readback_path),
        "uninterrupted_telemetry_path": str(uninterrupted_telemetry_path),
        "uninterrupted_telemetry_sha256": expected_hashes[
            "uninterrupted_telemetry_sha256"
        ],
        "resumed_telemetry_path": str(resumed_telemetry_path),
        "resumed_telemetry_sha256": expected_hashes[
            "resumed_telemetry_sha256"
        ],
        "strict_resume_evidence_path": str(resume_evidence_path),
        "strict_resume_evidence_sha256": expected_hashes[
            "strict_resume_evidence_sha256"
        ],
        "intentional_stop_checkpoint_path": str(interrupted_path),
        "intentional_stop_checkpoint_sha256": interrupted_sha256,
        "uninterrupted_checkpoint_path": str(full_checkpoint_path),
        "uninterrupted_checkpoint_sha256": full_checkpoint_sha256,
        "uninterrupted_result_path": str(full_result_path),
        "uninterrupted_result_sha256": expected_hashes[
            "uninterrupted_result_sha256"
        ],
        "uninterrupted_candidate_sha256": full_result["checkpoint_sha256"],
        "resumed_checkpoint_path": str(resumed_checkpoint_path),
        "resumed_checkpoint_sha256": resumed_checkpoint_sha256,
        "resumed_result_path": str(resumed_result_path),
        "resumed_result_sha256": expected_hashes["resumed_result_sha256"],
        "resumed_candidate_sha256": resumed_result_document[
            "checkpoint_sha256"
        ],
        "restart_equivalence_report_path": str(restart_report_path),
        "restart_equivalence_report_sha256": sha256_file(restart_report_path),
        "restart_semantic_equivalent": True,
        "restart_results_equivalent": True,
        "wandb_resumed_history_contiguous": True,
        "wandb_strict_resume_verified": True,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "resume_evidence_schema": RESUME_EVIDENCE_SCHEMA,
        "resume_evidence_run_id": resume_evidence_document["run_id"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    path_or_value = (
        "source-root",
        "source-inventory-sha256",
        "container",
        "container-sha256",
        "validation-contract",
        "validation-contract-sha256",
        "learner-identity-sha256",
        "parent-runtime-policy-sha256",
        "campaign-root",
        "snapshot",
        "snapshot-file-sha256",
        "snapshot-sha256",
        "policy-checkpoint",
        "policy-checkpoint-sha256",
        "policy-spec",
        "policy-spec-sha256",
        "policy-manifest",
        "policy-manifest-sha256",
        "bundle",
        "config-sha256",
        "import-report",
        "wandb-report",
        "wandb-readback",
        "uninterrupted-telemetry",
        "resumed-telemetry",
        "resume-evidence",
        "uninterrupted-checkpoint",
        "uninterrupted-result",
        "interrupted-checkpoint",
        "resumed-checkpoint",
        "resumed-result",
        "restart-report",
        "campaign-id",
        "output",
    )
    for name in path_or_value:
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--replay-shards", required=True, type=int)
    parser.add_argument("--macro-round", required=True, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    document = build_attestation(
        source_root=Path(arguments.source_root),
        source_inventory_sha256=arguments.source_inventory_sha256,
        container=Path(arguments.container),
        container_sha256=arguments.container_sha256,
        validation_contract=Path(arguments.validation_contract),
        validation_contract_sha256=arguments.validation_contract_sha256,
        learner_identity_sha256=arguments.learner_identity_sha256,
        parent_runtime_policy_sha256=arguments.parent_runtime_policy_sha256,
        campaign_root=Path(arguments.campaign_root),
        snapshot=Path(arguments.snapshot),
        snapshot_file_sha256=arguments.snapshot_file_sha256,
        snapshot_sha256=arguments.snapshot_sha256,
        replay_shards=arguments.replay_shards,
        policy_checkpoint=Path(arguments.policy_checkpoint),
        policy_checkpoint_sha256=arguments.policy_checkpoint_sha256,
        policy_spec=Path(arguments.policy_spec),
        policy_spec_sha256=arguments.policy_spec_sha256,
        policy_manifest=Path(arguments.policy_manifest),
        policy_manifest_sha256=arguments.policy_manifest_sha256,
        bundle=Path(arguments.bundle),
        config_sha256=arguments.config_sha256,
        import_report=Path(arguments.import_report),
        wandb_report=Path(arguments.wandb_report),
        wandb_readback=Path(arguments.wandb_readback),
        uninterrupted_telemetry=Path(arguments.uninterrupted_telemetry),
        resumed_telemetry=Path(arguments.resumed_telemetry),
        resume_evidence=Path(arguments.resume_evidence),
        uninterrupted_checkpoint=Path(arguments.uninterrupted_checkpoint),
        uninterrupted_result=Path(arguments.uninterrupted_result),
        interrupted_checkpoint=Path(arguments.interrupted_checkpoint),
        resumed_checkpoint=Path(arguments.resumed_checkpoint),
        resumed_result=Path(arguments.resumed_result),
        restart_report=Path(arguments.restart_report),
        campaign_id=arguments.campaign_id,
        macro_round=arguments.macro_round,
    )
    output = Path(os.path.abspath(Path(arguments.output).expanduser()))
    if output.is_symlink():
        raise H100ValidationAttestationError("attestation output must not be a symlink")
    digest = publish_immutable_json(output, document)
    os.chmod(output, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print(f"OCI_NRT_H100_VALIDATION_PASS attestation={output} sha256={digest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ArtifactError, H100ValidationAttestationError) as exc:
        print(f"OCI-NRT H100 validation attestation failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
