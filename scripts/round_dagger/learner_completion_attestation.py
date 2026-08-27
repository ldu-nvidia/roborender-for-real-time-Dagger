#!/usr/bin/env python3
"""Seal and verify a fully completed production round-DAgger learner.

The H100 trainer publishes this attestation only after W&B has been finished and
read back.  The RTX bridge may then recover the narrow crash window where the
learner result and this attestation exist but the allocation success marker does
not.  Verification is intentionally usable with the login-node Python: tensor
contents are inspected while publishing inside the trainer container and are
subsequently bound by immutable SHA-256 identities.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPOSITORY_ROOT))

from round_dagger.artifacts import (  # noqa: E402
    load_strict_json,
    publish_immutable_json,
    sha256_file,
)
from round_dagger.learner_identity import (  # noqa: E402
    learner_identity_sha256 as compute_learner_identity_sha256,
    learner_wandb_run_id,
)
from round_dagger.policy_bundle import validate_policy_bundle_identity  # noqa: E402
from round_dagger.replay import load_snapshot  # noqa: E402
from scripts.round_dagger import remote_learner_success  # noqa: E402


SCHEMA = "fast_roborender.round_dagger.learner_completion.v2"
READBACK_SCHEMA = "fast_roborender.round_dagger.learner_wandb_readback.v3"
BUNDLE_SCHEMA = "fast_roborender.round_dagger.learner_bundle.v1"
CHECKPOINT_SCHEMA = "fast_roborender.round_dagger.learner_checkpoint.v3"
RESULT_SCHEMA = "fast_roborender.round_dagger.learner_result.v2"
TELEMETRY_SCHEMA = "fast_roborender.round_dagger.learner_telemetry.v2"
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
ARTIFACT_NAMES = (
    "bundle",
    "result",
    "restart",
    "candidate",
    "policy_spec",
    "policy_manifest",
    "telemetry",
    "wandb_readback",
)
ATTESTATION_KEYS = {
    "schema",
    "status",
    "campaign_id",
    "macro_round",
    "code_sha256",
    "learner_config_sha256",
    "learner_contract_sha256",
    "learner_identity_sha256",
    "model_lineage_sha256",
    "parent_runtime_policy_sha256",
    "runtime_container_sha256",
    "runtime_policy_sha256",
    "replay_snapshot_file_sha256",
    "replay_snapshot_sha256",
    "world_size",
    "batch_size_per_rank",
    "optimizer_steps",
    "save_every",
    "artifacts",
    "wandb",
}


class CompletionAttestationError(RuntimeError):
    pass


def _sha(value: str, label: str) -> str:
    if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
        raise CompletionAttestationError(f"{label} must be a lowercase SHA-256")
    return value


def _token(value: str, label: str) -> str:
    if not isinstance(value, str) or TOKEN_RE.fullmatch(value) is None:
        raise CompletionAttestationError(f"{label} is malformed")
    return value


def _regular(path: Path, label: str) -> Path:
    result = Path(os.path.abspath(path.expanduser()))
    if result.is_symlink() or not result.is_file():
        raise CompletionAttestationError(f"{label} is missing or unsafe")
    return result


def _output_file(output: Path, name: str, label: str) -> Path:
    if not isinstance(name, str) or Path(name).name != name or name in {"", ".", ".."}:
        raise CompletionAttestationError(f"{label} basename is unsafe")
    path = _regular(output / name, label)
    if path.parent != output:
        raise CompletionAttestationError(f"{label} escaped learner output")
    return path


def _artifact(path: Path) -> dict[str, str]:
    return {"path": path.name, "sha256": sha256_file(path)}


def _telemetry_records(
    path: Path,
    *,
    campaign_id: str,
    macro_round: int,
    optimizer_steps: int,
    replay_transitions_global: int,
) -> dict[int, dict[str, Any]]:
    expected_keys = {
        "schema",
        "campaign_id",
        "macro_round",
        "optimizer_step",
        "loss",
        "replay_transitions_rank",
        "replay_transitions_per_rank",
        "replay_transitions_global",
        "world_size",
    }
    by_step: dict[int, dict[str, Any]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise CompletionAttestationError("cannot read learner telemetry") from exc
    if not lines:
        raise CompletionAttestationError("learner telemetry is empty")
    expected_counts: tuple[int, ...] | None = None
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CompletionAttestationError("learner telemetry is malformed") from exc
        if not isinstance(row, Mapping) or set(row) != expected_keys:
            raise CompletionAttestationError("learner telemetry fields mismatch")
        step = row.get("optimizer_step")
        loss = row.get("loss")
        transition_counts = row.get("replay_transitions_per_rank")
        global_transitions = row.get("replay_transitions_global")
        if (
            row.get("schema") != TELEMETRY_SCHEMA
            or row.get("campaign_id") != campaign_id
            or row.get("macro_round") != macro_round
            or type(step) is not int
            or not 1 <= step <= optimizer_steps
            or isinstance(loss, bool)
            or not isinstance(loss, (int, float))
            or not math.isfinite(float(loss))
            or row.get("world_size") != 8
            or type(row.get("replay_transitions_rank")) is not int
            or row["replay_transitions_rank"] < 1
            or not isinstance(transition_counts, list)
            or len(transition_counts) != 8
            or any(type(value) is not int or value < 1 for value in transition_counts)
            or row["replay_transitions_rank"] != transition_counts[0]
            or type(global_transitions) is not int
            or global_transitions != sum(transition_counts)
            or global_transitions != replay_transitions_global
        ):
            raise CompletionAttestationError("learner telemetry contract mismatch")
        observed_counts = tuple(transition_counts)
        if expected_counts is None:
            expected_counts = observed_counts
        elif observed_counts != expected_counts:
            raise CompletionAttestationError(
                "learner telemetry replay partition changed between steps"
            )
        normalized = dict(row)
        previous = by_step.get(step)
        # Scheduler retries can replay the at-most-four updates after the last
        # save-every=5 frontier.  Determinism requires those duplicate records
        # to be byte-semantically identical.
        if previous is not None and previous != normalized:
            raise CompletionAttestationError("learner telemetry retry diverged")
        by_step[step] = normalized
    if set(by_step) != set(range(1, optimizer_steps + 1)):
        raise CompletionAttestationError("learner telemetry step coverage is incomplete")
    return by_step


def _wandb_readback(
    *,
    telemetry: dict[int, dict[str, Any]],
    campaign_id: str,
    macro_round: int,
    code_sha256: str,
    config_sha256: str,
    learner_identity_sha256: str,
    parent_sha256: str,
    parent_runtime_policy_sha256: str,
    runtime_container_sha256: str,
    learner_contract_sha256: str,
    snapshot_file_sha256: str,
    snapshot_sha256: str,
    entity: str,
    project: str,
    run_id: str,
    expected_email: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    try:
        from scripts.scaling.wandb_preflight import _load_wandb, query_identity

        wandb = _load_wandb()
        identity = query_identity(
            wandb,
            expected_email=expected_email,
            requested_entity=entity,
            authorization_scope="team",
        )
    except Exception as exc:
        raise CompletionAttestationError(
            f"W&B identity readback failed ({type(exc).__name__})"
        ) from exc
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    observed: set[int] = set()
    state = ""
    while time.monotonic() < deadline:
        try:
            run = wandb.Api(timeout=30).run(f"{identity.entity}/{project}/{run_id}")
            state = str(getattr(run, "state", ""))
            config = dict(getattr(run, "config", {}) or {})
            if config.get("campaign_id") != campaign_id:
                raise CompletionAttestationError("W&B campaign identity mismatch")
            if config.get("macro_round") != macro_round:
                raise CompletionAttestationError("W&B macro-round identity mismatch")
            if config.get("learner_code_sha256") != code_sha256:
                raise CompletionAttestationError("W&B learner code identity mismatch")
            if config.get("learner_config_sha256") != config_sha256:
                raise CompletionAttestationError("W&B learner config identity mismatch")
            if config.get("learner_identity_sha256") != learner_identity_sha256:
                raise CompletionAttestationError("W&B learner identity mismatch")
            if config.get("model_lineage_sha256") != parent_sha256:
                raise CompletionAttestationError("W&B model lineage mismatch")
            if (
                config.get("parent_runtime_policy_sha256")
                != parent_runtime_policy_sha256
            ):
                raise CompletionAttestationError("W&B parent runtime identity mismatch")
            if config.get("runtime_container_sha256") != runtime_container_sha256:
                raise CompletionAttestationError("W&B runtime container identity mismatch")
            if config.get("learner_contract_sha256") != learner_contract_sha256:
                raise CompletionAttestationError("W&B learner contract identity mismatch")
            if config.get("replay_snapshot_file_sha256") != snapshot_file_sha256:
                raise CompletionAttestationError("W&B replay snapshot file identity mismatch")
            if config.get("replay_snapshot_sha256") != snapshot_sha256:
                raise CompletionAttestationError("W&B replay snapshot identity mismatch")
            observed = set()
            for row in run.scan_history(page_size=500):
                if not isinstance(row, Mapping):
                    continue
                step = row.get("optimizer_step")
                local = telemetry.get(step) if type(step) is int else None
                if local is None:
                    continue
                loss = row.get("loss")
                if (
                    not isinstance(loss, bool)
                    and isinstance(loss, (int, float))
                    and math.isfinite(float(loss))
                    and row.get("schema") == TELEMETRY_SCHEMA
                    and math.isclose(
                        float(loss), float(local["loss"]), rel_tol=1e-12, abs_tol=1e-12
                    )
                    and row.get("campaign_id") == campaign_id
                    and row.get("macro_round") == macro_round
                    and row.get("world_size") == 8
                    and row.get("replay_transitions_rank")
                    == local["replay_transitions_rank"]
                    and row.get("replay_transitions_per_rank")
                    == local["replay_transitions_per_rank"]
                    and row.get("replay_transitions_global")
                    == local["replay_transitions_global"]
                ):
                    observed.add(step)
            if state == "finished" and observed == set(telemetry):
                break
            last_error = "remote history incomplete"
        except CompletionAttestationError:
            raise
        except Exception as exc:
            last_error = type(exc).__name__
        time.sleep(5)
    if state != "finished" or observed != set(telemetry):
        raise CompletionAttestationError(
            f"W&B completion readback failed ({last_error or 'timeout'})"
        )
    return {
        "schema": READBACK_SCHEMA,
        "status": "PASS",
        "account_email": identity.email,
        "entity": identity.entity,
        "authorization_scope": identity.authorization_scope,
        "team_member": identity.team_member,
        "entity_is_team": identity.entity_is_team,
        "personal": identity.personal,
        "project": project,
        "run_id": run_id,
        "run_state": state,
        "campaign_id": campaign_id,
        "macro_round": macro_round,
        "learner_code_sha256": code_sha256,
        "learner_config_sha256": config_sha256,
        "learner_identity_sha256": learner_identity_sha256,
        "model_lineage_sha256": parent_sha256,
        "parent_runtime_policy_sha256": parent_runtime_policy_sha256,
        "runtime_container_sha256": runtime_container_sha256,
        "learner_contract_sha256": learner_contract_sha256,
        "replay_snapshot_file_sha256": snapshot_file_sha256,
        "replay_snapshot_sha256": snapshot_sha256,
        "verified_optimizer_steps": len(observed),
        "first_optimizer_step": min(observed),
        "last_optimizer_step": max(observed),
    }


def _validate_readback(
    document: Mapping[str, Any],
    *,
    campaign_id: str,
    macro_round: int,
    code_sha256: str,
    config_sha256: str,
    learner_identity_sha256: str,
    parent_sha256: str,
    parent_runtime_policy_sha256: str,
    runtime_container_sha256: str,
    learner_contract_sha256: str,
    snapshot_file_sha256: str,
    snapshot_sha256: str,
    entity: str,
    project: str,
    run_id: str,
) -> None:
    expected = {
        "schema": READBACK_SCHEMA,
        "status": "PASS",
        "account_email": "ldu@nvidia.com",
        "entity": entity,
        "authorization_scope": "team",
        "team_member": True,
        "entity_is_team": True,
        "personal": False,
        "project": project,
        "run_id": run_id,
        "run_state": "finished",
        "campaign_id": campaign_id,
        "macro_round": macro_round,
        "learner_code_sha256": code_sha256,
        "learner_config_sha256": config_sha256,
        "learner_identity_sha256": learner_identity_sha256,
        "model_lineage_sha256": parent_sha256,
        "parent_runtime_policy_sha256": parent_runtime_policy_sha256,
        "runtime_container_sha256": runtime_container_sha256,
        "learner_contract_sha256": learner_contract_sha256,
        "replay_snapshot_file_sha256": snapshot_file_sha256,
        "replay_snapshot_sha256": snapshot_sha256,
        "verified_optimizer_steps": 1200,
        "first_optimizer_step": 1,
        "last_optimizer_step": 1200,
    }
    if dict(document) != expected:
        raise CompletionAttestationError("W&B completion readback mismatch")


def publish_attestation(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from round_dagger.learner import LearnerBundleConfig

    output = Path(os.path.abspath(Path(args.output_dir).expanduser()))
    if output.is_symlink() or not output.is_dir():
        raise CompletionAttestationError("learner output directory is unsafe")
    campaign_id = _token(args.campaign_id, "campaign ID")
    code_sha = _sha(args.code_sha256, "code SHA-256")
    config_sha = _sha(args.config_sha256, "learner config SHA-256")
    contract_sha = _sha(args.learner_contract_sha256, "learner contract SHA-256")
    container_sha = _sha(
        args.runtime_container_sha256, "runtime container SHA-256"
    )
    provided_identity_sha = _sha(
        args.learner_identity_sha256, "learner identity SHA-256"
    )
    parent_sha = _sha(args.parent_sha256, "parent SHA-256")
    snapshot_file_sha = _sha(args.snapshot_file_sha256, "snapshot file SHA-256")
    snapshot_sha = _sha(args.snapshot_sha256, "snapshot semantic SHA-256")
    parent = _regular(Path(args.parent_checkpoint), "parent checkpoint")
    snapshot_path = _regular(Path(args.snapshot), "replay snapshot")
    if sha256_file(parent) != parent_sha:
        raise CompletionAttestationError("parent checkpoint digest mismatch")
    snapshot = load_snapshot(snapshot_path, expected_file_sha256=snapshot_file_sha)
    if (
        snapshot["snapshot_sha256"] != snapshot_sha
        or snapshot["campaign_id"] != campaign_id
        or snapshot["through_macro_round"] != args.macro_round
    ):
        raise CompletionAttestationError("replay snapshot lineage mismatch")
    bundle_path = _output_file(output, args.bundle, "learner bundle")
    bundle = LearnerBundleConfig.load(bundle_path)
    if bundle.sha256 != config_sha or sha256_file(bundle_path) != config_sha:
        raise CompletionAttestationError("learner bundle digest mismatch")
    if (
        bundle.campaign_id != campaign_id
        or bundle.macro_round != args.macro_round
        or bundle.code_sha256 != code_sha
        or bundle.model_lineage_sha256 != parent_sha
        or bundle.initial_checkpoint_sha256 != parent_sha
        or bundle.batch_size != 8
        or bundle.max_steps != 1200
        or bundle.save_every != 5
        or bundle.learning_rate != 2.5e-5
        or bundle.weight_decay != 0.01
        or bundle.seed != 17
        or bundle.partition_seed != 23
    ):
        raise CompletionAttestationError("production learner bundle contract mismatch")
    parent_runtime_sha = _sha(
        bundle.to_mapping()["model_config"]["runtime_policy_sha256"],
        "parent runtime policy SHA-256",
    )
    identity_sha = compute_learner_identity_sha256(
        campaign_id=campaign_id,
        macro_round=args.macro_round,
        learner_code_sha256=code_sha,
        learner_config_sha256=config_sha,
        replay_snapshot_file_sha256=snapshot_file_sha,
        replay_snapshot_sha256=snapshot_sha,
        parent_checkpoint_sha256=parent_sha,
        parent_runtime_policy_sha256=parent_runtime_sha,
        runtime_container_sha256=container_sha,
        learner_contract_sha256=contract_sha,
    )
    if identity_sha != provided_identity_sha:
        raise CompletionAttestationError(
            "learner identity SHA-256 does not bind the exact semantic run"
        )
    expected_run_id = learner_wandb_run_id(
        campaign_id=campaign_id,
        macro_round=args.macro_round,
        learner_identity_sha256=identity_sha,
    )
    if args.wandb_run_id != expected_run_id:
        raise CompletionAttestationError(
            "W&B run ID is not bound to the exact learner identity"
        )

    result_path = _output_file(output, args.result, "learner result")
    result = load_strict_json(result_path)
    expected_result = {
        "schema": RESULT_SCHEMA,
        "campaign_id": campaign_id,
        "macro_round": args.macro_round,
        "model_lineage_sha256": parent_sha,
        "replay_snapshot_sha256": snapshot_sha,
        "code_sha256": code_sha,
        "learner_config_sha256": config_sha,
        "optimizer_steps": 1200,
    }
    if any(result.get(key) != value for key, value in expected_result.items()):
        raise CompletionAttestationError("learner result lineage mismatch")
    runtime_sha = _sha(
        result.get("runtime_policy_sha256"), "candidate runtime policy SHA-256"
    )
    for key in ("initial_loss", "final_loss"):
        value = result.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise CompletionAttestationError("learner result loss is non-finite")

    restart = _output_file(output, args.restart, "restart checkpoint")
    candidate = _output_file(output, result["checkpoint_path"], "candidate checkpoint")
    spec = _output_file(output, result["policy_spec_path"], "candidate policy spec")
    manifest = _output_file(
        output, result["policy_bundle_manifest_path"], "candidate policy manifest"
    )
    if (
        sha256_file(candidate) != result["checkpoint_sha256"]
        or sha256_file(spec) != result["policy_spec_sha256"]
        or sha256_file(manifest) != result["policy_bundle_manifest_sha256"]
        or sha256_file(candidate) != sha256_file(restart)
    ):
        raise CompletionAttestationError("candidate policy artifact digest mismatch")
    policy_evidence = validate_policy_bundle_identity(
        checkpoint=candidate,
        policy_spec=spec,
        manifest=manifest,
        expected_checkpoint_sha256=result["checkpoint_sha256"],
        expected_runtime_policy_sha256=result["runtime_policy_sha256"],
    )
    checkpoint = torch.load(restart, map_location="cpu", weights_only=False)
    restart_expected = {
        "schema": CHECKPOINT_SCHEMA,
        "campaign_id": campaign_id,
        "macro_round": args.macro_round,
        "model_lineage_sha256": parent_sha,
        "code_sha256": code_sha,
        "learner_config_sha256": config_sha,
        "learner_identity_sha256": identity_sha,
        "runtime_container_sha256": container_sha,
        "learner_contract_sha256": contract_sha,
        "replay_snapshot_sha256": snapshot_sha,
        "replay_snapshot_file_sha256": snapshot_file_sha,
        "completed_steps": 1200,
        "world_size": 8,
        "homie_controller_sha256": policy_evidence[
            "homie_controller_sha256"
        ],
    }
    if not isinstance(checkpoint, Mapping) or any(
        checkpoint.get(key) != value for key, value in restart_expected.items()
    ):
        raise CompletionAttestationError("restart checkpoint lineage/frontier mismatch")
    if (
        not isinstance(checkpoint.get("model_state_dict"), Mapping)
        or not isinstance(checkpoint.get("optimizer_state_dict"), Mapping)
        or not isinstance(checkpoint.get("rng_states"), list)
        or len(checkpoint["rng_states"]) != 8
    ):
        raise CompletionAttestationError("restart checkpoint state is incomplete")

    telemetry_path = _output_file(output, args.telemetry, "learner telemetry")
    telemetry = _telemetry_records(
        telemetry_path,
        campaign_id=campaign_id,
        macro_round=args.macro_round,
        optimizer_steps=1200,
        replay_transitions_global=int(snapshot["transition_count"]),
    )
    readback_document = _wandb_readback(
        telemetry=telemetry,
        campaign_id=campaign_id,
        macro_round=args.macro_round,
        code_sha256=code_sha,
        config_sha256=config_sha,
        learner_identity_sha256=identity_sha,
        parent_sha256=parent_sha,
        parent_runtime_policy_sha256=parent_runtime_sha,
        runtime_container_sha256=container_sha,
        learner_contract_sha256=contract_sha,
        snapshot_file_sha256=snapshot_file_sha,
        snapshot_sha256=snapshot_sha,
        entity=args.wandb_entity,
        project=args.wandb_project,
        run_id=args.wandb_run_id,
        expected_email=args.wandb_expected_email,
        timeout_seconds=args.wandb_timeout_seconds,
    )
    readback_path = Path(os.path.abspath(Path(args.wandb_readback).expanduser()))
    if readback_path.parent != output or readback_path.is_symlink():
        raise CompletionAttestationError("W&B readback path is non-canonical")
    publish_immutable_json(readback_path, readback_document)
    _validate_readback(
        load_strict_json(readback_path),
        campaign_id=campaign_id,
        macro_round=args.macro_round,
        code_sha256=code_sha,
        config_sha256=config_sha,
        learner_identity_sha256=identity_sha,
        parent_sha256=parent_sha,
        parent_runtime_policy_sha256=parent_runtime_sha,
        runtime_container_sha256=container_sha,
        learner_contract_sha256=contract_sha,
        snapshot_file_sha256=snapshot_file_sha,
        snapshot_sha256=snapshot_sha,
        entity=args.wandb_entity,
        project=args.wandb_project,
        run_id=args.wandb_run_id,
    )
    artifacts = {
        "bundle": _artifact(bundle_path),
        "result": _artifact(result_path),
        "restart": _artifact(restart),
        "candidate": _artifact(candidate),
        "policy_spec": _artifact(spec),
        "policy_manifest": _artifact(manifest),
        "telemetry": _artifact(telemetry_path),
        "wandb_readback": _artifact(readback_path),
    }
    document = {
        "schema": SCHEMA,
        "status": "PASS",
        "campaign_id": campaign_id,
        "macro_round": args.macro_round,
        "code_sha256": code_sha,
        "learner_config_sha256": config_sha,
        "learner_contract_sha256": contract_sha,
        "learner_identity_sha256": identity_sha,
        "model_lineage_sha256": parent_sha,
        "parent_runtime_policy_sha256": parent_runtime_sha,
        "runtime_container_sha256": container_sha,
        "runtime_policy_sha256": runtime_sha,
        "replay_snapshot_file_sha256": snapshot_file_sha,
        "replay_snapshot_sha256": snapshot_sha,
        "world_size": 8,
        "batch_size_per_rank": 8,
        "optimizer_steps": 1200,
        "save_every": 5,
        "artifacts": artifacts,
        "wandb": {
            "entity": args.wandb_entity,
            "project": args.wandb_project,
            "run_id": args.wandb_run_id,
            "run_state": "finished",
            "verified_optimizer_steps": 1200,
        },
    }
    attestation_path = Path(os.path.abspath(Path(args.attestation).expanduser()))
    if attestation_path.parent != output or attestation_path.is_symlink():
        raise CompletionAttestationError("completion attestation path is non-canonical")
    publish_immutable_json(attestation_path, document)
    os.chmod(attestation_path, 0o444)
    return document


def verify_attestation(
    attestation: Path,
    *,
    result: Path,
    campaign_id: str,
    macro_round: int,
    code_sha256: str,
    config_sha256: str,
    parent_sha256: str,
    snapshot_file_sha256: str,
    snapshot_sha256: str,
    runtime_container_sha256: str,
    learner_contract_sha256: str,
    learner_identity_sha256: str,
) -> dict[str, Any]:
    import torch

    path = _regular(attestation, "learner completion attestation")
    if path.stat().st_mode & 0o222:
        raise CompletionAttestationError("completion attestation must be read-only")
    output = path.parent
    document = load_strict_json(path)
    if not isinstance(document, Mapping) or set(document) != ATTESTATION_KEYS:
        raise CompletionAttestationError("completion attestation fields mismatch")
    expected = {
        "schema": SCHEMA,
        "status": "PASS",
        "campaign_id": _token(campaign_id, "campaign ID"),
        "macro_round": macro_round,
        "code_sha256": _sha(code_sha256, "code SHA-256"),
        "learner_config_sha256": _sha(config_sha256, "config SHA-256"),
        "learner_contract_sha256": _sha(
            learner_contract_sha256, "learner contract SHA-256"
        ),
        "learner_identity_sha256": _sha(
            learner_identity_sha256, "learner identity SHA-256"
        ),
        "model_lineage_sha256": _sha(parent_sha256, "parent SHA-256"),
        "runtime_container_sha256": _sha(
            runtime_container_sha256, "runtime container SHA-256"
        ),
        "replay_snapshot_file_sha256": _sha(
            snapshot_file_sha256, "snapshot file SHA-256"
        ),
        "replay_snapshot_sha256": _sha(snapshot_sha256, "snapshot SHA-256"),
        "world_size": 8,
        "batch_size_per_rank": 8,
        "optimizer_steps": 1200,
        "save_every": 5,
    }
    if any(document.get(key) != value for key, value in expected.items()):
        raise CompletionAttestationError("completion attestation lineage mismatch")
    _sha(document.get("runtime_policy_sha256"), "runtime policy SHA-256")
    parent_runtime_sha256 = _sha(
        document.get("parent_runtime_policy_sha256"),
        "parent runtime policy SHA-256",
    )
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(ARTIFACT_NAMES):
        raise CompletionAttestationError("completion artifact inventory mismatch")
    resolved: dict[str, Path] = {}
    for name in ARTIFACT_NAMES:
        item = artifacts[name]
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256"}:
            raise CompletionAttestationError("completion artifact entry mismatch")
        artifact_path = _output_file(output, item["path"], name)
        if sha256_file(artifact_path) != _sha(item["sha256"], f"{name} SHA-256"):
            raise CompletionAttestationError(f"{name} changed after completion")
        resolved[name] = artifact_path
    expected_result = _regular(result, "learner result")
    if resolved["result"] != expected_result:
        raise CompletionAttestationError("completion attestation binds another result")
    if sha256_file(resolved["candidate"]) != sha256_file(resolved["restart"]):
        raise CompletionAttestationError("candidate and restart checkpoint differ")
    result_doc = load_strict_json(expected_result)
    result_expected = {
        "schema": RESULT_SCHEMA,
        "campaign_id": campaign_id,
        "macro_round": macro_round,
        "code_sha256": code_sha256,
        "learner_config_sha256": config_sha256,
        "model_lineage_sha256": parent_sha256,
        "replay_snapshot_sha256": snapshot_sha256,
        "optimizer_steps": 1200,
        "checkpoint_path": resolved["candidate"].name,
        "checkpoint_sha256": sha256_file(resolved["candidate"]),
        "runtime_policy_sha256": document["runtime_policy_sha256"],
        "policy_spec_path": resolved["policy_spec"].name,
        "policy_spec_sha256": sha256_file(resolved["policy_spec"]),
        "policy_bundle_manifest_path": resolved["policy_manifest"].name,
        "policy_bundle_manifest_sha256": sha256_file(resolved["policy_manifest"]),
    }
    if any(result_doc.get(key) != value for key, value in result_expected.items()):
        raise CompletionAttestationError("attested learner result mismatch")
    policy_evidence = validate_policy_bundle_identity(
        checkpoint=resolved["candidate"],
        policy_spec=resolved["policy_spec"],
        manifest=resolved["policy_manifest"],
        expected_checkpoint_sha256=result_doc["checkpoint_sha256"],
        expected_runtime_policy_sha256=result_doc["runtime_policy_sha256"],
    )
    bundle = load_strict_json(resolved["bundle"])
    if sha256_file(resolved["bundle"]) != config_sha256:
        raise CompletionAttestationError("attested learner bundle mismatch")
    if (
        bundle.get("schema") != BUNDLE_SCHEMA
        or bundle.get("campaign_id") != campaign_id
        or bundle.get("macro_round") != macro_round
        or bundle.get("code_sha256") != code_sha256
        or bundle.get("model_lineage_sha256") != parent_sha256
        or bundle.get("initial_checkpoint_sha256") != parent_sha256
        or (bundle.get("batch_size"), bundle.get("max_steps"), bundle.get("save_every"))
        != (8, 1200, 5)
        or bundle.get("learning_rate") != 2.5e-5
        or bundle.get("weight_decay") != 0.01
        or bundle.get("seed") != 17
        or bundle.get("partition_seed") != 23
    ):
        raise CompletionAttestationError("attested production learner pins mismatch")
    model_config = bundle.get("model_config")
    if not isinstance(model_config, Mapping):
        raise CompletionAttestationError("attested learner model config is malformed")
    bundle_parent_runtime_sha256 = _sha(
        model_config.get("runtime_policy_sha256"),
        "bundle parent runtime policy SHA-256",
    )
    if bundle_parent_runtime_sha256 != parent_runtime_sha256:
        raise CompletionAttestationError("attested parent runtime identity mismatch")
    provided_identity_sha256 = learner_identity_sha256
    computed_identity_sha256 = compute_learner_identity_sha256(
        campaign_id=campaign_id,
        macro_round=macro_round,
        learner_code_sha256=code_sha256,
        learner_config_sha256=config_sha256,
        replay_snapshot_file_sha256=snapshot_file_sha256,
        replay_snapshot_sha256=snapshot_sha256,
        parent_checkpoint_sha256=parent_sha256,
        parent_runtime_policy_sha256=parent_runtime_sha256,
        runtime_container_sha256=runtime_container_sha256,
        learner_contract_sha256=learner_contract_sha256,
    )
    if computed_identity_sha256 != provided_identity_sha256:
        raise CompletionAttestationError(
            "attested learner identity does not bind the exact semantic run"
        )
    restart_document = torch.load(
        resolved["restart"], map_location="cpu", weights_only=False
    )
    restart_expected = {
        "schema": CHECKPOINT_SCHEMA,
        "campaign_id": campaign_id,
        "macro_round": macro_round,
        "model_lineage_sha256": parent_sha256,
        "code_sha256": code_sha256,
        "learner_config_sha256": config_sha256,
        "learner_identity_sha256": provided_identity_sha256,
        "runtime_container_sha256": runtime_container_sha256,
        "learner_contract_sha256": learner_contract_sha256,
        "replay_snapshot_sha256": snapshot_sha256,
        "replay_snapshot_file_sha256": snapshot_file_sha256,
        "completed_steps": 1200,
        "world_size": 8,
        "homie_controller_sha256": policy_evidence[
            "homie_controller_sha256"
        ],
    }
    if not isinstance(restart_document, Mapping) or any(
        restart_document.get(key) != value
        for key, value in restart_expected.items()
    ):
        raise CompletionAttestationError("attested restart semantic identity mismatch")
    wandb = document.get("wandb")
    if not isinstance(wandb, Mapping) or set(wandb) != {
        "entity", "project", "run_id", "run_state", "verified_optimizer_steps"
    }:
        raise CompletionAttestationError("W&B completion summary fields mismatch")
    if (
        wandb.get("entity") != "nvidia"
        or wandb.get("run_state") != "finished"
        or wandb.get("verified_optimizer_steps") != 1200
        or TOKEN_RE.fullmatch(str(wandb.get("project", ""))) is None
        or TOKEN_RE.fullmatch(str(wandb.get("run_id", ""))) is None
        or wandb.get("run_id")
        != learner_wandb_run_id(
            campaign_id=campaign_id,
            macro_round=macro_round,
            learner_identity_sha256=provided_identity_sha256,
        )
    ):
        raise CompletionAttestationError("W&B completion summary mismatch")
    _validate_readback(
        load_strict_json(resolved["wandb_readback"]),
        campaign_id=campaign_id,
        macro_round=macro_round,
        code_sha256=code_sha256,
        config_sha256=config_sha256,
        learner_identity_sha256=provided_identity_sha256,
        parent_sha256=parent_sha256,
        parent_runtime_policy_sha256=parent_runtime_sha256,
        runtime_container_sha256=runtime_container_sha256,
        learner_contract_sha256=learner_contract_sha256,
        snapshot_file_sha256=snapshot_file_sha256,
        snapshot_sha256=snapshot_sha256,
        entity=str(wandb["entity"]),
        project=str(wandb["project"]),
        run_id=str(wandb["run_id"]),
    )
    return dict(document)


def _common(command: argparse.ArgumentParser) -> None:
    command.add_argument("--attestation", required=True, type=Path)
    command.add_argument("--result", required=True, type=Path)
    command.add_argument("--campaign-id", required=True)
    command.add_argument("--macro-round", required=True, type=int)
    command.add_argument("--code-sha256", required=True)
    command.add_argument("--config-sha256", required=True)
    command.add_argument("--parent-sha256", required=True)
    command.add_argument("--snapshot-file-sha256", required=True)
    command.add_argument("--snapshot-sha256", required=True)
    command.add_argument("--runtime-container-sha256", required=True)
    command.add_argument("--learner-contract-sha256", required=True)
    command.add_argument("--learner-identity-sha256", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    publish = sub.add_parser("publish")
    _common(publish)
    publish.add_argument("--output-dir", required=True, type=Path)
    publish.add_argument("--bundle", required=True)
    publish.add_argument("--restart", default="restart.pt")
    publish.add_argument("--telemetry", default="telemetry.jsonl")
    publish.add_argument("--parent-checkpoint", required=True, type=Path)
    publish.add_argument("--snapshot", required=True, type=Path)
    publish.add_argument("--wandb-readback", required=True, type=Path)
    publish.add_argument("--wandb-entity", required=True)
    publish.add_argument("--wandb-project", required=True)
    publish.add_argument("--wandb-run-id", required=True)
    publish.add_argument("--wandb-expected-email", required=True)
    publish.add_argument("--wandb-timeout-seconds", type=int, default=180)
    verify = sub.add_parser("verify")
    _common(verify)
    adopt = sub.add_parser("adopt")
    _common(adopt)
    adopt.add_argument("--success-marker", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "publish":
        document = publish_attestation(args)
    else:
        document = verify_attestation(
            args.attestation,
            result=args.result,
            campaign_id=args.campaign_id,
            macro_round=args.macro_round,
            code_sha256=args.code_sha256,
            config_sha256=args.config_sha256,
            parent_sha256=args.parent_sha256,
            snapshot_file_sha256=args.snapshot_file_sha256,
            snapshot_sha256=args.snapshot_sha256,
            runtime_container_sha256=args.runtime_container_sha256,
            learner_contract_sha256=args.learner_contract_sha256,
            learner_identity_sha256=args.learner_identity_sha256,
        )
        if args.command == "adopt":
            remote_learner_success.publish(
                args.success_marker,
                args.result,
                completion_attestation=args.attestation,
                campaign_id=args.campaign_id,
                macro_round=args.macro_round,
                learner_code_sha256=args.code_sha256,
                learner_config_sha256=args.config_sha256,
                learner_identity_sha256=args.learner_identity_sha256,
            )
    print(json.dumps(document, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        print(
            f"learner completion attestation failed: {type(exc).__name__}: {exc}",
            file=__import__("sys").stderr,
        )
        raise SystemExit(2)
