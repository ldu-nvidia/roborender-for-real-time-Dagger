#!/usr/bin/env python3
"""Publish the immutable success evidence for the OCI-NRT H100 live gate."""

from __future__ import annotations

import argparse
import json
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
from round_dagger.learner import CHECKPOINT_SCHEMA, RESULT_SCHEMA  # noqa: E402
from scripts.round_dagger.verify_learner_restart_equivalence import (  # noqa: E402
    REPORT_SCHEMA as RESTART_REPORT_SCHEMA,
)
from scripts.round_dagger.h100_validation_wandb_readback import (  # noqa: E402
    SCHEMA as WANDB_READBACK_SCHEMA,
)


SCHEMA = "fast_roborender.round_dagger.oci_nrt_h100_validation.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


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


def _strict_object(path: Path, label: str) -> tuple[Path, dict[str, Any]]:
    path = _regular(path, label)
    value = load_strict_json(path)
    if not isinstance(value, Mapping):
        raise H100ValidationAttestationError(f"{label} must contain a JSON object")
    return path, dict(value)


def _validate_result(
    path: Path,
    *,
    campaign_id: str,
    macro_round: int,
    source_inventory_sha256: str,
    config_sha256: str,
    snapshot_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    path, result = _strict_object(path, "learner result")
    expected = {
        "schema": RESULT_SCHEMA,
        "campaign_id": campaign_id,
        "macro_round": macro_round,
        "code_sha256": source_inventory_sha256,
        "learner_config_sha256": config_sha256,
        "replay_snapshot_sha256": snapshot_sha256,
        "optimizer_steps": 2,
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise H100ValidationAttestationError("learner result lineage mismatch")
    checkpoint_name = result.get("checkpoint_path")
    checkpoint_sha = result.get("checkpoint_sha256")
    if (
        not isinstance(checkpoint_name, str)
        or Path(checkpoint_name).name != checkpoint_name
        or _SHA256.fullmatch(str(checkpoint_sha)) is None
    ):
        raise H100ValidationAttestationError("learner candidate identity is malformed")
    checkpoint = _regular(path.parent / checkpoint_name, "learner candidate")
    if sha256_file(checkpoint) != checkpoint_sha:
        raise H100ValidationAttestationError("learner candidate digest mismatch")
    return path, result


def _load_checkpoint(path: Path, *, expected_steps: int, label: str) -> tuple[Path, dict]:
    path = _regular(path, label)
    try:
        import torch

        value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise H100ValidationAttestationError(f"cannot load {label}: {exc}") from exc
    if not isinstance(value, Mapping) or value.get("schema") != CHECKPOINT_SCHEMA:
        raise H100ValidationAttestationError(f"{label} schema mismatch")
    if value.get("completed_steps") != expected_steps:
        raise H100ValidationAttestationError(f"{label} completed step mismatch")
    if value.get("world_size") != 8:
        raise H100ValidationAttestationError(f"{label} did not use eight ranks")
    states = value.get("rng_states")
    if not isinstance(states, list) or len(states) != 8:
        raise H100ValidationAttestationError(f"{label} rank frontier mismatch")
    return path, dict(value)


def build_attestation(
    *,
    source_root: Path,
    source_inventory_sha256: str,
    container: Path,
    container_sha256: str,
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
    if isinstance(replay_shards, bool) or replay_shards < 8:
        raise H100ValidationAttestationError("validation replay requires at least 8 shards")
    if isinstance(macro_round, bool) or macro_round < 0:
        raise H100ValidationAttestationError("macro round is invalid")

    source_manifest = _regular(
        source_root / ".source_snapshot_manifest.json", "sealed source manifest"
    )
    container = _regular(container, "trainer container")
    snapshot = _regular(snapshot, "replay snapshot")
    policy_checkpoint = _regular(policy_checkpoint, "parent policy checkpoint")
    policy_spec = _regular(policy_spec, "parent policy spec")
    policy_manifest = _regular(policy_manifest, "parent policy manifest")
    bundle = _regular(bundle, "validation learner bundle")
    exact_files = (
        (container, container_sha256, "container"),
        (snapshot, snapshot_file_sha256, "snapshot"),
        (policy_checkpoint, policy_checkpoint_sha256, "parent checkpoint"),
        (policy_spec, policy_spec_sha256, "parent spec"),
        (policy_manifest, policy_manifest_sha256, "parent manifest"),
        (bundle, config_sha256, "learner bundle"),
    )
    for path, expected, label in exact_files:
        if sha256_file(path) != expected:
            raise H100ValidationAttestationError(f"{label} digest mismatch")

    import_report_path, import_report_doc = _strict_object(
        import_report, "CUDA/import report"
    )
    if (
        import_report_doc.get("status") != "PASS"
        or import_report_doc.get("visible_cuda_devices") != 1
        or import_report_doc.get("compute_capability") != [9, 0]
        or import_report_doc.get("nccl_available") is not True
        or import_report_doc.get("campaign_mount") != "ro"
        or import_report_doc.get("learner_output_mount") != "rw"
    ):
        raise H100ValidationAttestationError("CUDA/import mount gate did not pass")
    wandb_report_path, wandb_report_doc = _strict_object(
        wandb_report, "W&B preflight report"
    )
    if (
        wandb_report_doc.get("email") != "ldu@nvidia.com"
        or wandb_report_doc.get("entity") != "nvidia"
        or wandb_report_doc.get("authorization_scope") != "team"
        or wandb_report_doc.get("team_member") is not True
        or wandb_report_doc.get("entity_is_team") is not True
        or wandb_report_doc.get("personal") is not False
        or wandb_report_doc.get("mode") != "online"
    ):
        raise H100ValidationAttestationError("W&B preflight identity mismatch")

    interrupted_path, interrupted = _load_checkpoint(
        interrupted_checkpoint, expected_steps=1, label="intentional-stop checkpoint"
    )
    full_checkpoint_path, _ = _load_checkpoint(
        uninterrupted_checkpoint, expected_steps=2, label="uninterrupted checkpoint"
    )
    resumed_checkpoint_path, _ = _load_checkpoint(
        resumed_checkpoint, expected_steps=2, label="resumed checkpoint"
    )
    full_result_path, full_result = _validate_result(
        uninterrupted_result,
        campaign_id=campaign_id,
        macro_round=macro_round,
        source_inventory_sha256=source_inventory_sha256,
        config_sha256=config_sha256,
        snapshot_sha256=snapshot_sha256,
    )
    resumed_result_path, resumed_result_doc = _validate_result(
        resumed_result,
        campaign_id=campaign_id,
        macro_round=macro_round,
        source_inventory_sha256=source_inventory_sha256,
        config_sha256=config_sha256,
        snapshot_sha256=snapshot_sha256,
    )
    restart_report_path, restart = _strict_object(
        restart_report, "restart equivalence report"
    )
    if (
        restart.get("schema") != RESTART_REPORT_SCHEMA
        or restart.get("semantic_equivalent") is not True
        or restart.get("results_equivalent") is not True
        or restart.get("expected_completed_steps") != 2
        or restart.get("uninterrupted_checkpoint_sha256")
        != sha256_file(full_checkpoint_path)
        or restart.get("resumed_checkpoint_sha256")
        != sha256_file(resumed_checkpoint_path)
        or restart.get("uninterrupted_result_sha256") != sha256_file(full_result_path)
        or restart.get("resumed_result_sha256") != sha256_file(resumed_result_path)
    ):
        raise H100ValidationAttestationError("restart equivalence evidence mismatch")

    readback_path, readback = _strict_object(
        wandb_readback, "W&B learner readback report"
    )
    if readback_path.stat().st_mode & 0o222:
        raise H100ValidationAttestationError(
            "W&B learner readback report must be read-only"
        )
    full_run = readback.get("full_run")
    resumed_run = readback.get("resumed_run")
    if (
        readback.get("schema") != WANDB_READBACK_SCHEMA
        or readback.get("status") != "PASS"
        or readback.get("account_email") != "ldu@nvidia.com"
        or readback.get("entity") != "nvidia"
        or readback.get("authorization_scope") != "team"
        or readback.get("team_member") is not True
        or readback.get("entity_is_team") is not True
        or readback.get("personal") is not False
        or readback.get("campaign_id") != campaign_id
        or readback.get("macro_round") != macro_round
        or readback.get("learner_config_sha256") != config_sha256
        or readback.get("model_lineage_sha256") != policy_checkpoint_sha256
        or readback.get("full_result_sha256") != sha256_file(full_result_path)
        or readback.get("resumed_result_sha256") != sha256_file(resumed_result_path)
        or readback.get("initial_loss") != full_result["initial_loss"]
        or readback.get("final_loss") != full_result["final_loss"]
        or readback.get("histories_identical") is not True
        or readback.get("resumed_history_contiguous") is not True
        or readback.get("strict_resume_mode") != "must"
        or readback.get("strict_resume_verified") is not True
        or not isinstance(full_run, Mapping)
        or not isinstance(resumed_run, Mapping)
        or full_run.get("history_steps") != [1, 2]
        or resumed_run.get("history_steps") != [1, 2]
        or full_run.get("run_id") == resumed_run.get("run_id")
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
        "source_manifest_sha256": sha256_file(source_manifest),
        "source_inventory_sha256": source_inventory_sha256,
        "container_path": str(container),
        "container_sha256": container_sha256,
        "campaign_root": str(campaign_root),
        "snapshot_path": str(snapshot),
        "snapshot_file_sha256": snapshot_file_sha256,
        "snapshot_sha256": snapshot_sha256,
        "replay_shards": replay_shards,
        "parent_policy_checkpoint_sha256": policy_checkpoint_sha256,
        "parent_policy_spec_sha256": policy_spec_sha256,
        "parent_policy_manifest_sha256": policy_manifest_sha256,
        "learner_config_sha256": config_sha256,
        "learner_contract": {
            "backend": "nccl",
            "world_size": 8,
            "batch_size_per_rank": 1,
            "optimizer_steps": 2,
            "save_every": 1,
            "intentional_stop_step": 1,
        },
        "cuda_import_report_sha256": sha256_file(import_report_path),
        "wandb_preflight_report_sha256": sha256_file(wandb_report_path),
        "wandb_learner_readback_sha256": sha256_file(readback_path),
        "wandb_full_run_id": full_run["run_id"],
        "wandb_resumed_run_id": resumed_run["run_id"],
        "wandb_resumed_history_contiguous": True,
        "wandb_strict_resume_verified": True,
        "intentional_stop_checkpoint_sha256": sha256_file(interrupted_path),
        "uninterrupted_result_sha256": sha256_file(full_result_path),
        "uninterrupted_candidate_sha256": full_result["checkpoint_sha256"],
        "resumed_result_sha256": sha256_file(resumed_result_path),
        "resumed_candidate_sha256": resumed_result_doc["checkpoint_sha256"],
        "restart_equivalence_report_sha256": sha256_file(restart_report_path),
        "restart_semantic_equivalent": True,
        "restart_results_equivalent": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source-root",
        "source-inventory-sha256",
        "container",
        "container-sha256",
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
        "uninterrupted-checkpoint",
        "uninterrupted-result",
        "interrupted-checkpoint",
        "resumed-checkpoint",
        "resumed-result",
        "restart-report",
        "campaign-id",
        "output",
    ):
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
