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
    canonical_json_sha256,
    load_strict_json,
    publish_immutable_json,
    sha256_file,
)
from round_dagger.checkpoint_compat import (  # noqa: E402
    CheckpointCompatibilityError,
    load_restricted_restart_document,
)
from round_dagger.learner_identity import (  # noqa: E402
    learner_identity_sha256 as compute_learner_identity_sha256,
    learner_wandb_run_id,
)
from round_dagger.policy_bundle import validate_policy_bundle_identity  # noqa: E402
from round_dagger.replay import load_heldout_snapshot, load_snapshot  # noqa: E402
from scripts.round_dagger import remote_learner_success  # noqa: E402


SCHEMA = "fast_roborender.round_dagger.learner_completion.v5"
READBACK_SCHEMA = "fast_roborender.round_dagger.learner_wandb_readback.v5"
BUNDLE_SCHEMA = "fast_roborender.round_dagger.learner_bundle.v1"
CHECKPOINT_SCHEMA = "fast_roborender.round_dagger.learner_checkpoint.v3"
RESULT_SCHEMA = "fast_roborender.round_dagger.learner_result.v2"
TELEMETRY_SCHEMA = "fast_roborender.round_dagger.learner_telemetry.v2"
HELDOUT_TELEMETRY_SCHEMA = (
    "fast_roborender.round_dagger.learner_heldout_telemetry.v1"
)
TRAINING_QUALITY_TELEMETRY_SCHEMA = (
    "fast_roborender.round_dagger.learner_training_quality.v2"
)
TRAINING_CONTEXT_TELEMETRY_SCHEMA = (
    "fast_roborender.round_dagger.learner_training_context.v1"
)
TRAINING_QUALITY_PROBE_POLICY = (
    "lineage_stratified_current_historical_replay.v1"
)
TRAINING_QUALITY_PROBE_BUDGET = 256
TRAINING_QUALITY_PROBE_DOMAIN = (
    "fast_roborender.round_dagger.training_quality_probe.v1"
)
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
    "training_quality_telemetry",
    "training_context",
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
    "training_quality_evaluation",
    "training_context",
    "wandb",
}

TRAINING_CONTEXT_METRIC_KEYS = {
    "parent_rollout_action_mse",
    "parent_rollout_action_mae",
    "parent_rollout_action_rmse",
    "parent_rollout_action_cosine",
    "parent_rollout_action_l2_p90",
    "expert_intervention_rate",
    "behavior_expert_match_rate",
    "expert_label_coverage",
    "replay_transitions",
    "replay_environments",
    "done_rate",
    "scheduled_beta",
}


class CompletionAttestationError(RuntimeError):
    pass


def _restart_document(path: Path, label: str) -> Mapping[str, Any]:
    try:
        return load_restricted_restart_document(path, map_location="cpu")
    except CheckpointCompatibilityError as exc:
        raise CompletionAttestationError(f"{label} is unsafe: {exc}") from exc


def _evaluation_steps(optimizer_steps: int, eval_interval: int) -> list[int]:
    if type(optimizer_steps) is not int or optimizer_steps < 1:
        raise CompletionAttestationError("optimizer steps must be positive")
    if type(eval_interval) is not int or eval_interval < 1:
        raise CompletionAttestationError("evaluation interval must be positive")
    return sorted(
        {0, *range(eval_interval, optimizer_steps + 1, eval_interval), optimizer_steps}
    )


def _sha(value: str, label: str) -> str:
    if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
        raise CompletionAttestationError(f"{label} must be a lowercase SHA-256")
    return value


_TRAINING_QUALITY_PROBE_SUMMARY_KEYS = {
    "policy",
    "budget",
    "campaign_id",
    "macro_round",
    "snapshot_file_sha256",
    "snapshot_sha256",
    "partition_seed",
    "world_size",
    "replay_transitions_per_rank",
    "replay_transitions_global",
    "transitions_per_rank",
    "transitions_global",
    "current_round_transitions_per_rank",
    "current_round_transitions_global",
    "historical_transitions_per_rank",
    "historical_transitions_global",
    "rank_selection_sha256s",
    "selection_sha256",
}


def _training_quality_probe_summary(
    summary: Mapping[str, Any],
    *,
    campaign_id: str,
    macro_round: int,
    snapshot_file_sha256: str,
    snapshot_sha256: str,
    partition_seed: int,
    world_size: int,
) -> dict[str, Any]:
    """Validate the bounded probe identity without importing the GPU learner."""

    if not isinstance(summary, Mapping) or set(summary) != (
        _TRAINING_QUALITY_PROBE_SUMMARY_KEYS
    ):
        raise CompletionAttestationError(
            "training-quality probe summary fields mismatch"
        )
    clean = dict(summary)
    if (
        clean.get("policy") != TRAINING_QUALITY_PROBE_POLICY
        or clean.get("budget") != TRAINING_QUALITY_PROBE_BUDGET
        or clean.get("campaign_id") != campaign_id
        or clean.get("macro_round") != macro_round
        or clean.get("snapshot_file_sha256") != snapshot_file_sha256
        or clean.get("snapshot_sha256") != snapshot_sha256
        or clean.get("partition_seed") != partition_seed
        or clean.get("world_size") != world_size
    ):
        raise CompletionAttestationError(
            "training-quality probe lineage/configuration mismatch"
        )
    list_fields = (
        "replay_transitions_per_rank",
        "transitions_per_rank",
        "current_round_transitions_per_rank",
        "historical_transitions_per_rank",
    )
    values_by_field: dict[str, list[int]] = {}
    for field in list_fields:
        values = clean.get(field)
        minimum = 1 if field in {
            "replay_transitions_per_rank",
            "transitions_per_rank",
        } else 0
        if (
            not isinstance(values, list)
            or len(values) != world_size
            or any(
                type(value) is not int or value < minimum for value in values
            )
        ):
            raise CompletionAttestationError(
                f"training-quality probe {field} is malformed"
            )
        values_by_field[field] = list(values)
    replay = values_by_field["replay_transitions_per_rank"]
    selected = values_by_field["transitions_per_rank"]
    current = values_by_field["current_round_transitions_per_rank"]
    historical = values_by_field["historical_transitions_per_rank"]
    if any(
        chosen > population or chosen != current_count + historical_count
        for chosen, population, current_count, historical_count in zip(
            selected, replay, current, historical
        )
    ):
        raise CompletionAttestationError(
            "training-quality probe per-rank coverage mismatch"
        )
    expected_totals = {
        "replay_transitions_global": sum(replay),
        "transitions_global": sum(selected),
        "current_round_transitions_global": sum(current),
        "historical_transitions_global": sum(historical),
    }
    if any(clean.get(key) != value for key, value in expected_totals.items()):
        raise CompletionAttestationError(
            "training-quality probe global coverage mismatch"
        )
    if clean["transitions_global"] != min(
        TRAINING_QUALITY_PROBE_BUDGET,
        clean["replay_transitions_global"],
    ):
        raise CompletionAttestationError(
            "training-quality probe does not exactly cover its bounded budget"
        )
    if (
        clean["current_round_transitions_global"]
        + clean["historical_transitions_global"]
        != clean["transitions_global"]
        or (macro_round == 0 and clean["historical_transitions_global"] != 0)
        or (
            macro_round > 0
            and clean["transitions_global"] == TRAINING_QUALITY_PROBE_BUDGET
            and (
                clean["current_round_transitions_global"]
                != TRAINING_QUALITY_PROBE_BUDGET // 2
                or clean["historical_transitions_global"]
                != TRAINING_QUALITY_PROBE_BUDGET // 2
            )
        )
    ):
        raise CompletionAttestationError(
            "training-quality probe current/historical split mismatch"
        )
    rank_digests = clean.get("rank_selection_sha256s")
    if (
        not isinstance(rank_digests, list)
        or len(rank_digests) != world_size
        or len(set(rank_digests)) != world_size
    ):
        raise CompletionAttestationError(
            "training-quality probe rank digest coverage mismatch"
        )
    for digest in rank_digests:
        _sha(digest, "training-quality probe rank selection SHA-256")
    declared = _sha(
        clean.get("selection_sha256"),
        "training-quality probe selection SHA-256",
    )
    digest_payload = {
        "domain": TRAINING_QUALITY_PROBE_DOMAIN,
        **{
            key: clean[key]
            for key in sorted(
                _TRAINING_QUALITY_PROBE_SUMMARY_KEYS - {"selection_sha256"}
            )
        },
    }
    if declared != canonical_json_sha256(digest_payload):
        raise CompletionAttestationError(
            "training-quality probe selection digest mismatch"
        )
    return clean


def _require_updated_runtime_policy(
    candidate_runtime_policy_sha256: str,
    parent_runtime_policy_sha256: str,
) -> str:
    candidate = _sha(
        candidate_runtime_policy_sha256, "candidate runtime policy SHA-256"
    )
    parent = _sha(parent_runtime_policy_sha256, "parent runtime policy SHA-256")
    if candidate == parent:
        raise CompletionAttestationError(
            "candidate runtime policy is bit-identical to its parent"
        )
    return candidate


def _token(value: str, label: str) -> str:
    if not isinstance(value, str) or TOKEN_RE.fullmatch(value) is None:
        raise CompletionAttestationError(f"{label} is malformed")
    return value


def _regular(path: Path, label: str) -> Path:
    result = Path(os.path.abspath(path.expanduser()))
    if result.is_symlink() or not result.is_file():
        raise CompletionAttestationError(f"{label} is missing or unsafe")
    return result


def _output_file(output: Path, name: str | os.PathLike[str], label: str) -> Path:
    try:
        raw_name = os.fspath(name)
    except TypeError as exc:
        raise CompletionAttestationError(f"{label} basename is unsafe") from exc
    if (
        not isinstance(raw_name, str)
        or Path(raw_name).name != raw_name
        or raw_name in {"", ".", ".."}
    ):
        raise CompletionAttestationError(f"{label} basename is unsafe")
    path = _regular(output / raw_name, label)
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
    world_size: int,
    batch_size_per_rank: int,
) -> dict[int, dict[str, Any]]:
    expected_keys = {
        "schema",
        "campaign_id",
        "macro_round",
        "optimizer_step",
        "loss",
        "gradient_l2_norm",
        "learning_rate",
        "global_samples_seen",
        "replay_transitions_rank",
        "replay_transitions_per_rank",
        "replay_transitions_global",
        "world_size",
        "gradient_reduction",
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
        gradient_l2_norm = row.get("gradient_l2_norm")
        learning_rate = row.get("learning_rate")
        global_samples_seen = row.get("global_samples_seen")
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
            or isinstance(gradient_l2_norm, bool)
            or not isinstance(gradient_l2_norm, (int, float))
            or not math.isfinite(float(gradient_l2_norm))
            or float(gradient_l2_norm) < 0.0
            or isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float))
            or not math.isfinite(float(learning_rate))
            or float(learning_rate) <= 0.0
            or type(global_samples_seen) is not int
            or global_samples_seen
            != step * batch_size_per_rank * world_size
            or row.get("world_size") != world_size
            or row.get("gradient_reduction")
            != "ordered_allgather_rank_sum_v1"
            or type(row.get("replay_transitions_rank")) is not int
            or row["replay_transitions_rank"] < 1
            or not isinstance(transition_counts, list)
            or len(transition_counts) != world_size
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


def _heldout_telemetry_records(
    path: Path,
    *,
    campaign_id: str,
    macro_round: int,
    snapshot_file_sha256: str,
    snapshot_sha256: str,
    optimizer_steps: int,
    eval_interval: int,
    learned_action_indices: list[int],
    world_size: int = 8,
) -> dict[int, dict[str, Any]]:
    """Validate every required held-out quality frontier, including step zero."""

    if eval_interval != 100:
        raise CompletionAttestationError(
            "production held-out evaluation interval must be exactly 100"
        )
    expected_steps = set(_evaluation_steps(optimizer_steps, eval_interval))
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise CompletionAttestationError(
            "cannot read held-out learner telemetry"
        ) from exc
    if not lines:
        raise CompletionAttestationError("held-out learner telemetry is empty")
    by_step: dict[int, dict[str, Any]] = {}
    expected_counts: tuple[int, ...] | None = None
    expected_per_action = {str(index) for index in learned_action_indices}
    expected_row_keys = {
        "schema",
        "campaign_id",
        "macro_round",
        "optimizer_step",
        "eval_frontier",
        "heldout_snapshot_file_sha256",
        "heldout_snapshot_sha256",
        "world_size",
        "heldout_transitions_rank",
        "heldout_transitions_per_rank",
        "heldout_transitions_global",
        "learned_action_indices",
        "action_mse",
        "action_mae",
        "action_rmse",
        "action_cosine",
        "action_absolute_error_p90",
        "action_l2_p90",
        "action_mse_delta_from_step0",
        "action_mse_relative_improvement_from_step0",
        "per_action",
    }
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CompletionAttestationError(
                "held-out learner telemetry is malformed"
            ) from exc
        if not isinstance(row, Mapping):
            raise CompletionAttestationError(
                "held-out learner telemetry row is not a mapping"
            )
        step = row.get("optimizer_step")
        counts = row.get("heldout_transitions_per_rank")
        per_action = row.get("per_action")
        if (
            set(row) != expected_row_keys
            or
            row.get("schema") != HELDOUT_TELEMETRY_SCHEMA
            or row.get("campaign_id") != campaign_id
            or row.get("macro_round") != macro_round
            or type(step) is not int
            or step not in expected_steps
            or row.get("eval_frontier") != step
            or row.get("heldout_snapshot_file_sha256") != snapshot_file_sha256
            or row.get("heldout_snapshot_sha256") != snapshot_sha256
            or row.get("world_size") != world_size
            or not isinstance(counts, list)
            or len(counts) != world_size
            or any(type(value) is not int or value < 1 for value in counts)
            or row.get("heldout_transitions_rank") != counts[0]
            or row.get("heldout_transitions_global") != sum(counts)
            or row.get("learned_action_indices") != learned_action_indices
            or not isinstance(per_action, Mapping)
            or set(per_action) != expected_per_action
        ):
            raise CompletionAttestationError(
                "held-out learner telemetry contract mismatch"
            )
        for metric in (
            "action_mse",
            "action_mae",
            "action_rmse",
            "action_cosine",
            "action_absolute_error_p90",
            "action_l2_p90",
            "action_mse_delta_from_step0",
            "action_mse_relative_improvement_from_step0",
        ):
            value = row.get(metric)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise CompletionAttestationError(
                    "held-out learner telemetry metric is non-finite"
                )
        if any(
            float(row[key]) < 0.0
            for key in (
                "action_mse",
                "action_mae",
                "action_rmse",
                "action_absolute_error_p90",
                "action_l2_p90",
            )
        ):
            raise CompletionAttestationError(
                "held-out learner telemetry error metric is negative"
            )
        if not -1.000001 <= float(row["action_cosine"]) <= 1.000001:
            raise CompletionAttestationError(
                "held-out learner telemetry cosine is out of range"
            )
        for metrics in per_action.values():
            if not isinstance(metrics, Mapping) or set(metrics) != {
                "mse", "mae", "rmse", "cosine"
            }:
                raise CompletionAttestationError(
                    "held-out per-action telemetry fields mismatch"
                )
            for metric, value in metrics.items():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or (metric != "cosine" and float(value) < 0.0)
                    or (
                        metric == "cosine"
                        and not -1.000001 <= float(value) <= 1.000001
                    )
                ):
                    raise CompletionAttestationError(
                        "held-out per-action telemetry metric is invalid"
                    )
        observed_counts = tuple(counts)
        if expected_counts is None:
            expected_counts = observed_counts
        elif observed_counts != expected_counts:
            raise CompletionAttestationError(
                "held-out replay partition changed between evaluations"
            )
        normalized = dict(row)
        previous = by_step.get(step)
        if previous is not None and previous != normalized:
            raise CompletionAttestationError(
                "held-out telemetry retry diverged"
            )
        by_step[step] = normalized
    if set(by_step) != expected_steps:
        raise CompletionAttestationError(
            "held-out telemetry evaluation frontier coverage is incomplete"
        )
    baseline_mse = float(by_step[0]["action_mse"])
    signatures: set[str] = set()
    for step in sorted(by_step):
        row = by_step[step]
        mse = float(row["action_mse"])
        expected_delta = mse - baseline_mse
        if baseline_mse == 0.0:
            if mse != 0.0:
                raise CompletionAttestationError(
                    "held-out relative improvement is undefined from zero baseline"
                )
            expected_relative = 0.0
        else:
            expected_relative = (baseline_mse - mse) / baseline_mse
        if not math.isclose(
            float(row["action_mse_delta_from_step0"]),
            expected_delta,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ) or not math.isclose(
            float(row["action_mse_relative_improvement_from_step0"]),
            expected_relative,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise CompletionAttestationError(
                "held-out baseline-relative quality metric mismatch"
            )
        quality_signature = {
            "action_mse": row["action_mse"],
            "action_mae": row["action_mae"],
            "action_rmse": row["action_rmse"],
            "action_cosine": row["action_cosine"],
            "action_absolute_error_p90": row["action_absolute_error_p90"],
            "action_l2_p90": row["action_l2_p90"],
            "per_action": row["per_action"],
        }
        signatures.add(
            json.dumps(
                quality_signature,
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            )
        )
    if len(signatures) < 2:
        raise CompletionAttestationError(
            "held-out quality is bit-identical across all evaluation frontiers"
        )
    return by_step


def _training_quality_telemetry_records(
    path: Path,
    *,
    campaign_id: str,
    macro_round: int,
    snapshot_file_sha256: str,
    snapshot_sha256: str,
    optimizer_steps: int,
    eval_interval: int,
    learned_action_indices: list[int],
    world_size: int,
    partition_seed: int,
    probe_summary: Mapping[str, Any],
) -> dict[int, dict[str, Any]]:
    """Validate every fixed held-in action-quality frontier."""

    if eval_interval != 100:
        raise CompletionAttestationError(
            "production training-quality interval must be exactly 100"
        )
    probe = _training_quality_probe_summary(
        probe_summary,
        campaign_id=campaign_id,
        macro_round=macro_round,
        snapshot_file_sha256=snapshot_file_sha256,
        snapshot_sha256=snapshot_sha256,
        partition_seed=partition_seed,
        world_size=world_size,
    )
    expected_steps = set(_evaluation_steps(optimizer_steps, eval_interval))
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise CompletionAttestationError(
            "cannot read training-quality telemetry"
        ) from exc
    if not lines:
        raise CompletionAttestationError("training-quality telemetry is empty")
    expected_identity = {
        "campaign_id": campaign_id,
        "macro_round": macro_round,
        "training_snapshot_file_sha256": snapshot_file_sha256,
        "training_snapshot_sha256": snapshot_sha256,
        "training_probe_policy": probe["policy"],
        "training_probe_budget": probe["budget"],
        "training_probe_partition_seed": probe["partition_seed"],
        "training_probe_selection_sha256": probe["selection_sha256"],
        "training_probe_rank_selection_sha256s": probe[
            "rank_selection_sha256s"
        ],
        "training_probe_record_rank": 0,
        "training_probe_rank_selection_sha256": probe[
            "rank_selection_sha256s"
        ][0],
        "training_probe_transitions_rank": probe["transitions_per_rank"][0],
        "training_probe_transitions_per_rank": probe["transitions_per_rank"],
        "training_probe_transitions_global": probe["transitions_global"],
        "training_probe_current_round_transitions_rank": probe[
            "current_round_transitions_per_rank"
        ][0],
        "training_probe_current_round_transitions_per_rank": probe[
            "current_round_transitions_per_rank"
        ],
        "training_probe_current_round_transitions_global": probe[
            "current_round_transitions_global"
        ],
        "training_probe_historical_transitions_rank": probe[
            "historical_transitions_per_rank"
        ][0],
        "training_probe_historical_transitions_per_rank": probe[
            "historical_transitions_per_rank"
        ],
        "training_probe_historical_transitions_global": probe[
            "historical_transitions_global"
        ],
        "training_replay_transitions_rank": probe[
            "replay_transitions_per_rank"
        ][0],
        "training_replay_transitions_per_rank": probe[
            "replay_transitions_per_rank"
        ],
        "training_replay_transitions_global": probe[
            "replay_transitions_global"
        ],
        "world_size": world_size,
        "learned_action_indices": learned_action_indices,
    }
    expected_keys = set(expected_identity) | {
        "schema",
        "optimizer_step",
        "action_mse",
        "action_mae",
        "action_rmse",
        "action_cosine",
        "action_absolute_error_p90",
        "action_l2_p90",
        "per_action",
        "per_action_absolute_error_p90",
    }
    expected_per_action = {str(index) for index in learned_action_indices}
    by_step: dict[int, dict[str, Any]] = {}
    signatures: set[str] = set()
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CompletionAttestationError(
                "training-quality telemetry is malformed"
            ) from exc
        if not isinstance(row, Mapping):
            raise CompletionAttestationError(
                "training-quality telemetry row is not a mapping"
            )
        step = row.get("optimizer_step")
        per_action = row.get("per_action")
        per_action_p90 = row.get("per_action_absolute_error_p90")
        if (
            set(row) != expected_keys
            or row.get("schema") != TRAINING_QUALITY_TELEMETRY_SCHEMA
            or any(row.get(key) != value for key, value in expected_identity.items())
            or type(step) is not int
            or step not in expected_steps
            or not isinstance(per_action, Mapping)
            or set(per_action) != expected_per_action
            or not isinstance(per_action_p90, Mapping)
            or set(per_action_p90) != expected_per_action
        ):
            raise CompletionAttestationError(
                "training-quality telemetry contract mismatch"
            )
        for metric in (
            "action_mse",
            "action_mae",
            "action_rmse",
            "action_cosine",
            "action_absolute_error_p90",
            "action_l2_p90",
        ):
            value = row.get(metric)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or (metric != "action_cosine" and float(value) < 0.0)
                or (
                    metric == "action_cosine"
                    and not -1.000001 <= float(value) <= 1.000001
                )
            ):
                raise CompletionAttestationError(
                    "training-quality telemetry metric is invalid"
                )
        for metrics in per_action.values():
            if not isinstance(metrics, Mapping) or set(metrics) != {
                "mse",
                "mae",
                "rmse",
                "cosine",
            }:
                raise CompletionAttestationError(
                    "training-quality per-action fields mismatch"
                )
            for metric, value in metrics.items():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or (metric != "cosine" and float(value) < 0.0)
                    or (
                        metric == "cosine"
                        and not -1.000001 <= float(value) <= 1.000001
                    )
                ):
                    raise CompletionAttestationError(
                        "training-quality per-action metric is invalid"
                    )
        for value in per_action_p90.values():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise CompletionAttestationError(
                    "training-quality per-action p90 is invalid"
                )
        normalized = dict(row)
        previous = by_step.get(step)
        if previous is not None and previous != normalized:
            raise CompletionAttestationError(
                "training-quality telemetry retry diverged"
            )
        by_step[step] = normalized
        signatures.add(
            json.dumps(
                {
                    key: row[key]
                    for key in (
                        "action_mse",
                        "action_mae",
                        "action_rmse",
                        "action_cosine",
                        "action_absolute_error_p90",
                        "action_l2_p90",
                        "per_action",
                        "per_action_absolute_error_p90",
                    )
                },
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            )
        )
    if set(by_step) != expected_steps:
        raise CompletionAttestationError(
            "training-quality evaluation frontier coverage is incomplete"
        )
    if len(signatures) < 2:
        raise CompletionAttestationError(
            "training quality is bit-identical across all evaluation frontiers"
        )
    return by_step


def _training_context_record(
    path: Path,
    *,
    campaign_id: str,
    macro_round: int,
    config_sha256: str,
    parent_sha256: str,
    snapshot_file_sha256: str,
    snapshot_sha256: str,
    probe_selection_sha256: str,
    replay_transitions_global: int,
) -> dict[str, Any]:
    """Validate the immutable DAgger collection context used by the learner."""

    path = _regular(path, "training-context telemetry")
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n") or len(raw.splitlines()) != 1:
        raise CompletionAttestationError(
            "training-context telemetry must be one complete JSON record"
        )
    try:
        record = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CompletionAttestationError(
            "training-context telemetry is malformed"
        ) from exc
    expected_keys = {
        "schema",
        "campaign_id",
        "macro_round",
        "learner_config_sha256",
        "model_lineage_sha256",
        "replay_snapshot_file_sha256",
        "replay_snapshot_sha256",
        "training_probe_selection_sha256",
        "metrics",
    }
    if (
        not isinstance(record, Mapping)
        or set(record) != expected_keys
        or record.get("schema") != TRAINING_CONTEXT_TELEMETRY_SCHEMA
        or record.get("campaign_id") != campaign_id
        or record.get("macro_round") != macro_round
        or record.get("learner_config_sha256") != config_sha256
        or record.get("model_lineage_sha256") != parent_sha256
        or record.get("replay_snapshot_file_sha256") != snapshot_file_sha256
        or record.get("replay_snapshot_sha256") != snapshot_sha256
        or record.get("training_probe_selection_sha256")
        != probe_selection_sha256
    ):
        raise CompletionAttestationError(
            "training-context telemetry identity mismatch"
        )
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != TRAINING_CONTEXT_METRIC_KEYS:
        raise CompletionAttestationError(
            "training-context telemetry metric coverage mismatch"
        )
    numbers: dict[str, float | int] = {}
    for key, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CompletionAttestationError(
                f"training-context metric {key} is not numeric"
            )
        number = float(value)
        if not math.isfinite(number):
            raise CompletionAttestationError(
                f"training-context metric {key} is non-finite"
            )
        numbers[key] = value
    for key in ("replay_transitions", "replay_environments"):
        if type(metrics[key]) is not int or metrics[key] < 1:
            raise CompletionAttestationError(
                f"training-context metric {key} must be a positive integer"
            )
    if (
        metrics["replay_transitions"] != replay_transitions_global
        or metrics["replay_environments"] > replay_transitions_global
    ):
        raise CompletionAttestationError(
            "training-context replay population differs from the sealed replay"
        )
    for key in (
        "expert_intervention_rate",
        "behavior_expert_match_rate",
        "expert_label_coverage",
        "done_rate",
        "scheduled_beta",
    ):
        if not 0.0 <= float(metrics[key]) <= 1.0:
            raise CompletionAttestationError(
                f"training-context metric {key} must be in [0,1]"
            )
    if float(metrics["expert_label_coverage"]) != 1.0:
        raise CompletionAttestationError(
            "training-context expert label coverage must be complete"
        )
    if float(metrics["behavior_expert_match_rate"]) < float(
        metrics["expert_intervention_rate"]
    ):
        raise CompletionAttestationError(
            "training-context behavior/expert match rate is inconsistent"
        )
    for key in (
        "parent_rollout_action_mse",
        "parent_rollout_action_mae",
        "parent_rollout_action_rmse",
        "parent_rollout_action_l2_p90",
    ):
        if float(metrics[key]) < 0.0:
            raise CompletionAttestationError(
                f"training-context metric {key} is negative"
            )
    if not -1.000001 <= float(metrics["parent_rollout_action_cosine"]) <= 1.000001:
        raise CompletionAttestationError(
            "training-context rollout action cosine is outside [-1,1]"
        )
    if not math.isclose(
        float(metrics["parent_rollout_action_rmse"]) ** 2,
        float(metrics["parent_rollout_action_mse"]),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise CompletionAttestationError(
            "training-context rollout RMSE does not match MSE"
        )
    return dict(record)


def _training_context_wandb_fields(record: Mapping[str, Any]) -> dict[str, Any]:
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != TRAINING_CONTEXT_METRIC_KEYS:
        raise CompletionAttestationError(
            "training-context telemetry metric coverage mismatch"
        )
    return {
        f"2_training_quality/{key}": metrics[key]
        for key in sorted(TRAINING_CONTEXT_METRIC_KEYS)
    }


def _training_quality_wandb_fields(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Project one sealed local quality row onto its required W&B fields."""

    expected: dict[str, Any] = {
        "2_training_quality/probe_policy": record["training_probe_policy"],
        "2_training_quality/probe_selection_sha256": record[
            "training_probe_selection_sha256"
        ],
        "2_training_quality/probe_snapshot_sha256": record[
            "training_snapshot_sha256"
        ],
        "2_training_quality/probe_budget": record["training_probe_budget"],
        "2_training_quality/action_mse": record["action_mse"],
        "2_training_quality/action_mae": record["action_mae"],
        "2_training_quality/action_rmse": record["action_rmse"],
        "2_training_quality/action_cosine": record["action_cosine"],
        "2_training_quality/action_absolute_error_p90": record[
            "action_absolute_error_p90"
        ],
        "2_training_quality/action_l2_p90": record["action_l2_p90"],
        "2_training_quality/probe_transitions": record[
            "training_probe_transitions_global"
        ],
        "2_training_quality/probe_current_round_transitions": record[
            "training_probe_current_round_transitions_global"
        ],
        "2_training_quality/probe_historical_transitions": record[
            "training_probe_historical_transitions_global"
        ],
        "2_training_quality/replay_population": record[
            "training_replay_transitions_global"
        ],
    }
    per_action = record["per_action"]
    per_action_p90 = record["per_action_absolute_error_p90"]
    for action_index, metrics in per_action.items():
        for metric in ("mse", "mae", "rmse", "cosine"):
            expected[f"2_training_quality/action_{action_index}/{metric}"] = (
                metrics[metric]
            )
        expected[
            f"2_training_quality/action_{action_index}/absolute_error_p90"
        ] = per_action_p90[action_index]
    return expected


def _training_optimizer_wandb_fields(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Project one durable optimizer row onto its visible dashboard aliases."""

    return {
        "2_training_quality/optimization_loss": record["loss"],
        "2_training_quality/gradient_l2_norm": record["gradient_l2_norm"],
        "2_training_quality/learning_rate": record["learning_rate"],
        "2_training_quality/global_samples_seen": record["global_samples_seen"],
        "2_training_quality/replay_transitions": record[
            "replay_transitions_global"
        ],
    }


def _validation_ood_wandb_fields(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Project one durable held-out row onto its visible dashboard aliases."""

    expected = {
        "3_validation_ood/eval_frontier": record["eval_frontier"],
        "3_validation_ood/action_mse": record["action_mse"],
        "3_validation_ood/action_mae": record["action_mae"],
        "3_validation_ood/action_rmse": record["action_rmse"],
        "3_validation_ood/action_cosine": record["action_cosine"],
        "3_validation_ood/action_absolute_error_p90": record[
            "action_absolute_error_p90"
        ],
        "3_validation_ood/action_l2_p90": record["action_l2_p90"],
        "3_validation_ood/action_mse_delta_from_step0": record[
            "action_mse_delta_from_step0"
        ],
        "3_validation_ood/action_mse_relative_improvement_from_step0": record[
            "action_mse_relative_improvement_from_step0"
        ],
        "3_validation_ood/transitions": record["heldout_transitions_global"],
    }
    per_action = record["per_action"]
    for action_index, metrics in per_action.items():
        for metric in ("mse", "mae", "rmse", "cosine"):
            expected[
                f"3_validation_ood/action_{action_index}/{metric}"
            ] = metrics[metric]
    return expected


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
    world_size: int = 8,
    training_quality_telemetry: Mapping[int, Mapping[str, Any]] | None = None,
    training_context: Mapping[str, Any] | None = None,
    training_context_file_sha256: str | None = None,
    heldout_telemetry: Mapping[int, Mapping[str, Any]] | None = None,
    heldout_snapshot_file_sha256: str | None = None,
    heldout_snapshot_sha256: str | None = None,
    heldout_eval_interval: int | None = None,
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
    training_quality_observed: set[int] = set()
    heldout_observed: set[int] = set()
    training_context_observed = False
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
            if training_context is not None:
                metrics = training_context["metrics"]
                if config.get("scheduled_dagger_beta") != metrics["scheduled_beta"]:
                    raise CompletionAttestationError(
                        "W&B scheduled DAgger beta differs from training context"
                    )
            if heldout_telemetry is not None:
                if (
                    config.get("heldout_snapshot_file_sha256")
                    != heldout_snapshot_file_sha256
                    or config.get("heldout_snapshot_sha256")
                    != heldout_snapshot_sha256
                    or config.get("heldout_eval_interval")
                    != heldout_eval_interval
                ):
                    raise CompletionAttestationError(
                        "W&B held-out evaluation identity mismatch"
                    )
            observed = set()
            training_quality_observed = set()
            heldout_observed = set()
            training_context_observed = False
            loss_signatures: dict[int, tuple[Any, ...]] = {}
            training_quality_signatures: dict[int, str] = {}
            quality_signatures: dict[int, tuple[tuple[str, float], ...]] = {}
            training_context_signature: tuple[tuple[str, float], ...] | None = None
            for row in run.scan_history(page_size=500):
                if not isinstance(row, Mapping):
                    continue
                step = row.get("optimizer_step")
                local = telemetry.get(step) if type(step) is int else None
                if local is not None:
                    loss = row.get("loss")
                    has_loss = (
                        not isinstance(loss, bool)
                        and isinstance(loss, (int, float))
                        and math.isfinite(float(loss))
                    )
                    if has_loss:
                        optimizer_alias_signature: tuple[tuple[str, float], ...] = ()
                        if training_quality_telemetry is not None:
                            expected_optimizer_aliases = (
                                _training_optimizer_wandb_fields(local)
                            )
                            present_optimizer_aliases: dict[str, float] = {}
                            for key, expected_value in expected_optimizer_aliases.items():
                                actual_value = row.get(key)
                                if (
                                    isinstance(actual_value, bool)
                                    or not isinstance(actual_value, (int, float))
                                    or not math.isfinite(float(actual_value))
                                    or not math.isclose(
                                        float(actual_value),
                                        float(expected_value),
                                        rel_tol=1e-12,
                                        abs_tol=1e-12,
                                    )
                                ):
                                    raise CompletionAttestationError(
                                        "W&B visible training-quality optimizer aliases "
                                        f"differ at optimizer step {step}"
                                    )
                                present_optimizer_aliases[key] = float(actual_value)
                            optimizer_alias_signature = tuple(
                                sorted(present_optimizer_aliases.items())
                            )
                        transition_counts = row.get(
                            "replay_transitions_per_rank"
                        )
                        transition_signature: Any = (
                            tuple(transition_counts)
                            if isinstance(transition_counts, list)
                            else repr(transition_counts)
                        )
                        signature = (
                            float(loss),
                            row.get("gradient_l2_norm"),
                            row.get("learning_rate"),
                            row.get("global_samples_seen"),
                            row.get("schema"),
                            row.get("campaign_id"),
                            row.get("macro_round"),
                            row.get("world_size"),
                            row.get("gradient_reduction"),
                            row.get("replay_transitions_rank"),
                            transition_signature,
                            row.get("replay_transitions_global"),
                            optimizer_alias_signature,
                        )
                        previous = loss_signatures.get(step)
                        if previous is not None and previous != signature:
                            raise CompletionAttestationError(
                                f"W&B training history has conflicting duplicate step {step}"
                            )
                        loss_signatures[step] = signature
                        if not (
                            row.get("schema") == TELEMETRY_SCHEMA
                            and math.isclose(
                                float(loss),
                                float(local["loss"]),
                                rel_tol=1e-12,
                                abs_tol=1e-12,
                            )
                            and not isinstance(row.get("gradient_l2_norm"), bool)
                            and isinstance(row.get("gradient_l2_norm"), (int, float))
                            and math.isclose(
                                float(row["gradient_l2_norm"]),
                                float(local["gradient_l2_norm"]),
                                rel_tol=1e-12,
                                abs_tol=1e-12,
                            )
                            and not isinstance(row.get("learning_rate"), bool)
                            and isinstance(row.get("learning_rate"), (int, float))
                            and math.isclose(
                                float(row["learning_rate"]),
                                float(local["learning_rate"]),
                                rel_tol=1e-12,
                                abs_tol=1e-12,
                            )
                            and row.get("global_samples_seen")
                            == local["global_samples_seen"]
                            and row.get("campaign_id") == campaign_id
                            and row.get("macro_round") == macro_round
                            and row.get("world_size") == world_size
                            and row.get("gradient_reduction")
                            == "ordered_allgather_rank_sum_v1"
                            and row.get("replay_transitions_rank")
                            == local["replay_transitions_rank"]
                            and row.get("replay_transitions_per_rank")
                            == local["replay_transitions_per_rank"]
                            and row.get("replay_transitions_global")
                            == local["replay_transitions_global"]
                        ):
                            raise CompletionAttestationError(
                                f"W&B training history differs at optimizer step {step}"
                            )
                        observed.add(step)
                quality_step = row.get("optimizer_step")
                if type(quality_step) is not int:
                    quality_step = row.get("_step")
                if training_context is not None:
                    expected_context = _training_context_wandb_fields(
                        training_context
                    )
                    if quality_step == 0:
                        present_context = {
                            key: row.get(key)
                            for key in expected_context
                            if not isinstance(row.get(key), bool)
                            and isinstance(row.get(key), (int, float))
                            and math.isfinite(float(row[key]))
                        }
                        if present_context:
                            if set(present_context) != set(expected_context) or not all(
                                math.isclose(
                                    float(present_context[key]),
                                    float(value),
                                    rel_tol=1e-12,
                                    abs_tol=1e-12,
                                )
                                for key, value in expected_context.items()
                            ):
                                raise CompletionAttestationError(
                                    "W&B training-context history differs at optimizer "
                                    "step 0"
                                )
                            signature = tuple(
                                sorted(
                                    (key, float(value))
                                    for key, value in present_context.items()
                                )
                            )
                            if (
                                training_context_signature is not None
                                and training_context_signature != signature
                            ):
                                raise CompletionAttestationError(
                                    "W&B training-context history has conflicting "
                                    "duplicates"
                                )
                            training_context_signature = signature
                            training_context_observed = True
                training_quality_local = (
                    training_quality_telemetry.get(quality_step)
                    if training_quality_telemetry is not None
                    and type(quality_step) is int
                    else None
                )
                if training_quality_local is not None:
                    expected_training_quality = _training_quality_wandb_fields(
                        training_quality_local
                    )
                    present_training_quality: dict[str, Any] = {}
                    for key, expected_value in expected_training_quality.items():
                        value = row.get(key)
                        if isinstance(expected_value, str):
                            if isinstance(value, str) and value:
                                present_training_quality[key] = value
                        elif (
                            not isinstance(value, bool)
                            and isinstance(value, (int, float))
                            and math.isfinite(float(value))
                        ):
                            present_training_quality[key] = value
                    if present_training_quality:
                        if set(present_training_quality) != set(
                            expected_training_quality
                        ):
                            raise CompletionAttestationError(
                                "W&B training-quality history is partial at optimizer "
                                f"step {quality_step}"
                            )
                        for key, expected_value in expected_training_quality.items():
                            actual_value = present_training_quality[key]
                            if isinstance(expected_value, str):
                                matches = actual_value == expected_value
                            else:
                                matches = math.isclose(
                                    float(actual_value),
                                    float(expected_value),
                                    rel_tol=1e-12,
                                    abs_tol=1e-12,
                                )
                            if not matches:
                                raise CompletionAttestationError(
                                    "W&B training-quality history differs at optimizer "
                                    f"step {quality_step}"
                                )
                        signature = json.dumps(
                            present_training_quality,
                            sort_keys=True,
                            allow_nan=False,
                            separators=(",", ":"),
                        )
                        previous = training_quality_signatures.get(quality_step)
                        if previous is not None and previous != signature:
                            raise CompletionAttestationError(
                                "W&B training-quality history has conflicting duplicate "
                                f"step {quality_step}"
                            )
                        training_quality_signatures[quality_step] = signature
                        training_quality_observed.add(quality_step)
                heldout_local = (
                    heldout_telemetry.get(quality_step)
                    if heldout_telemetry is not None
                    and type(quality_step) is int
                    else None
                )
                if heldout_local is not None:
                    expected_quality = {
                        "quality/eval_frontier": heldout_local["eval_frontier"],
                        "quality/heldout_action_mse": heldout_local["action_mse"],
                        "quality/heldout_action_mae": heldout_local["action_mae"],
                        "quality/heldout_action_rmse": heldout_local["action_rmse"],
                        "quality/heldout_action_cosine": heldout_local[
                            "action_cosine"
                        ],
                        "quality/heldout_action_absolute_error_p90": heldout_local[
                            "action_absolute_error_p90"
                        ],
                        "quality/heldout_action_l2_p90": heldout_local[
                            "action_l2_p90"
                        ],
                        "quality/heldout_action_mse_delta_from_step0": heldout_local[
                            "action_mse_delta_from_step0"
                        ],
                        "quality/heldout_action_mse_relative_improvement_from_step0": heldout_local[
                            "action_mse_relative_improvement_from_step0"
                        ],
                        "quality/heldout_transitions": heldout_local[
                            "heldout_transitions_global"
                        ],
                    }
                    for action_index, metrics in heldout_local["per_action"].items():
                        for metric in ("mse", "mae", "rmse", "cosine"):
                            expected_quality[
                                f"quality/action_{action_index}/{metric}"
                            ] = metrics[metric]
                    expected_quality.update(
                        _validation_ood_wandb_fields(heldout_local)
                    )
                    present_quality = {
                        key: row.get(key)
                        for key in expected_quality
                        if not isinstance(row.get(key), bool)
                        and isinstance(row.get(key), (int, float))
                        and math.isfinite(float(row[key]))
                    }
                    if present_quality:
                        if set(present_quality) != set(expected_quality) or not all(
                            math.isclose(
                                float(present_quality[key]),
                                float(value),
                                rel_tol=1e-12,
                                abs_tol=1e-12,
                            )
                            for key, value in expected_quality.items()
                        ):
                            raise CompletionAttestationError(
                                "W&B held-out history differs at optimizer step "
                                f"{quality_step}"
                            )
                        signature = tuple(
                            sorted(
                                (key, float(value))
                                for key, value in present_quality.items()
                            )
                        )
                        previous = quality_signatures.get(quality_step)
                        if previous is not None and previous != signature:
                            raise CompletionAttestationError(
                                "W&B held-out history has conflicting duplicate step "
                                f"{quality_step}"
                            )
                        quality_signatures[quality_step] = signature
                        heldout_observed.add(quality_step)
            heldout_complete = (
                heldout_telemetry is None
                or heldout_observed == set(heldout_telemetry)
            )
            training_quality_complete = (
                training_quality_telemetry is None
                or training_quality_observed == set(training_quality_telemetry)
            )
            training_context_complete = (
                training_context is None or training_context_observed
            )
            if (
                state == "finished"
                and observed == set(telemetry)
                and training_quality_complete
                and training_context_complete
                and heldout_complete
            ):
                break
            last_error = "remote history incomplete"
        except CompletionAttestationError:
            raise
        except Exception as exc:
            last_error = type(exc).__name__
        time.sleep(5)
    if (
        state != "finished"
        or observed != set(telemetry)
        or (
            training_quality_telemetry is not None
            and training_quality_observed != set(training_quality_telemetry)
        )
        or (
            heldout_telemetry is not None
            and heldout_observed != set(heldout_telemetry)
        )
        or (training_context is not None and not training_context_observed)
    ):
        raise CompletionAttestationError(
            f"W&B completion readback failed ({last_error or 'timeout'})"
        )
    result = {
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
    if training_quality_telemetry is not None:
        first_quality = next(iter(training_quality_telemetry.values()))
        result["training_quality_evaluation"] = {
            "snapshot_file_sha256": first_quality[
                "training_snapshot_file_sha256"
            ],
            "snapshot_sha256": first_quality["training_snapshot_sha256"],
            "probe_selection_sha256": first_quality[
                "training_probe_selection_sha256"
            ],
            "probe_transitions": first_quality[
                "training_probe_transitions_global"
            ],
            "verified_steps": sorted(training_quality_observed),
        }
    if training_context is not None:
        if training_context_file_sha256 is None:
            raise CompletionAttestationError(
                "training-context W&B readback lacks its local artifact digest"
            )
        result["training_context"] = {
            "file_sha256": _sha(
                training_context_file_sha256,
                "training-context telemetry SHA-256",
            ),
            "metric_keys": sorted(TRAINING_CONTEXT_METRIC_KEYS),
            "verified_optimizer_step": 0,
        }
    if heldout_telemetry is not None:
        result["heldout_evaluation"] = {
            "snapshot_file_sha256": heldout_snapshot_file_sha256,
            "snapshot_sha256": heldout_snapshot_sha256,
            "eval_interval": heldout_eval_interval,
            "verified_steps": sorted(heldout_observed),
        }
    return result


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
    optimizer_steps: int = 500,
    training_quality_probe_summary: Mapping[str, Any] | None = None,
    training_quality_verified_steps: list[int] | None = None,
    training_context_file_sha256: str | None = None,
    training_context_metric_keys: list[str] | None = None,
    heldout_snapshot_file_sha256: str | None = None,
    heldout_snapshot_sha256: str | None = None,
    heldout_eval_interval: int | None = None,
    heldout_verified_steps: list[int] | None = None,
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
        "verified_optimizer_steps": optimizer_steps,
        "first_optimizer_step": 1,
        "last_optimizer_step": optimizer_steps,
    }
    training_quality_values = (
        training_quality_probe_summary,
        training_quality_verified_steps,
    )
    if any(value is not None for value in training_quality_values):
        if any(value is None for value in training_quality_values):
            raise CompletionAttestationError(
                "training-quality W&B readback identity is incomplete"
            )
        assert training_quality_probe_summary is not None
        expected["training_quality_evaluation"] = {
            "snapshot_file_sha256": training_quality_probe_summary[
                "snapshot_file_sha256"
            ],
            "snapshot_sha256": training_quality_probe_summary[
                "snapshot_sha256"
            ],
            "probe_selection_sha256": training_quality_probe_summary[
                "selection_sha256"
            ],
            "probe_transitions": training_quality_probe_summary[
                "transitions_global"
            ],
            "verified_steps": training_quality_verified_steps,
        }
    training_context_values = (
        training_context_file_sha256,
        training_context_metric_keys,
    )
    if any(value is not None for value in training_context_values):
        if any(value is None for value in training_context_values):
            raise CompletionAttestationError(
                "training-context W&B readback identity is incomplete"
            )
        expected["training_context"] = {
            "file_sha256": training_context_file_sha256,
            "metric_keys": training_context_metric_keys,
            "verified_optimizer_step": 0,
        }
    heldout_values = (
        heldout_snapshot_file_sha256,
        heldout_snapshot_sha256,
        heldout_eval_interval,
        heldout_verified_steps,
    )
    if any(value is not None for value in heldout_values):
        if any(value is None for value in heldout_values):
            raise CompletionAttestationError(
                "held-out W&B readback identity is incomplete"
            )
        expected["heldout_evaluation"] = {
            "snapshot_file_sha256": heldout_snapshot_file_sha256,
            "snapshot_sha256": heldout_snapshot_sha256,
            "eval_interval": heldout_eval_interval,
            "verified_steps": heldout_verified_steps,
        }
    if dict(document) != expected:
        raise CompletionAttestationError("W&B completion readback mismatch")


def publish_attestation(args: argparse.Namespace) -> dict[str, Any]:
    from round_dagger.learner import (
        LearnerBundleConfig,
        LearnerError,
        build_training_quality_probe,
    )

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
    if type(args.world_size) is not int or args.world_size < 1:
        raise CompletionAttestationError("learner world size must be positive")
    artifact_root = Path(
        os.path.abspath(Path(args.artifact_root).expanduser())
    )
    if artifact_root.is_symlink() or not artifact_root.is_dir():
        raise CompletionAttestationError("campaign artifact root is unsafe")
    parent = _regular(Path(args.parent_checkpoint), "parent checkpoint")
    snapshot_path = _regular(Path(args.snapshot), "replay snapshot")
    if sha256_file(parent) != parent_sha:
        raise CompletionAttestationError("parent checkpoint digest mismatch")
    snapshot = load_snapshot(
        snapshot_path,
        expected_file_sha256=snapshot_file_sha,
        artifact_root=artifact_root,
    )
    if (
        snapshot["snapshot_sha256"] != snapshot_sha
        or snapshot["campaign_id"] != campaign_id
        or snapshot["through_macro_round"] != args.macro_round
    ):
        raise CompletionAttestationError("replay snapshot lineage mismatch")
    bundle_path = _output_file(output, args.bundle, "learner bundle")
    bundle = LearnerBundleConfig.load(bundle_path)
    optimizer_steps = bundle.max_steps
    if optimizer_steps not in {500, 1200}:
        raise CompletionAttestationError(
            "production learner optimizer steps must be 1200 or legacy 500"
        )
    if bundle.sha256 != config_sha or sha256_file(bundle_path) != config_sha:
        raise CompletionAttestationError("learner bundle digest mismatch")
    if (
        bundle.campaign_id != campaign_id
        or bundle.macro_round != args.macro_round
        or bundle.code_sha256 != code_sha
        or bundle.model_lineage_sha256 != parent_sha
        or bundle.initial_checkpoint_sha256 != parent_sha
        or bundle.batch_size != 8
        or bundle.save_every != 5
        or bundle.learning_rate != 2.5e-5
        or bundle.weight_decay != 0.01
        or bundle.seed != 17
        or bundle.partition_seed != 23
    ):
        raise CompletionAttestationError("production learner bundle contract mismatch")
    try:
        probe_summary = build_training_quality_probe(
            snapshot,
            world_size=args.world_size,
            partition_seed=bundle.partition_seed,
        ).summary()
    except LearnerError as exc:
        raise CompletionAttestationError(
            f"cannot reconstruct training-quality probe: {exc}"
        ) from exc
    heldout_arguments = (
        args.heldout_snapshot,
        args.heldout_snapshot_file_sha256,
        args.heldout_snapshot_sha256,
        args.heldout_eval_interval,
    )
    if any(value is not None for value in heldout_arguments) and any(
        value is None for value in heldout_arguments
    ):
        raise CompletionAttestationError(
            "held-out completion identity must be supplied all-or-none"
        )
    heldout_enabled = all(value is not None for value in heldout_arguments)
    if bundle.heldout_evaluation_enabled != heldout_enabled:
        raise CompletionAttestationError(
            "learner bundle held-out mode differs from completion inputs"
        )
    heldout_snapshot = None
    if heldout_enabled:
        if args.heldout_eval_interval != 100:
            raise CompletionAttestationError(
                "production held-out evaluation interval must be exactly 100"
            )
        heldout_file_sha = _sha(
            args.heldout_snapshot_file_sha256,
            "held-out snapshot file SHA-256",
        )
        heldout_semantic_sha = _sha(
            args.heldout_snapshot_sha256,
            "held-out snapshot semantic SHA-256",
        )
        heldout_snapshot = load_heldout_snapshot(
            _regular(Path(args.heldout_snapshot), "held-out replay snapshot"),
            expected_file_sha256=heldout_file_sha,
            artifact_root=artifact_root,
        )
        if (
            heldout_snapshot["snapshot_sha256"] != heldout_semantic_sha
            or heldout_snapshot["campaign_id"] != campaign_id
            or bundle.heldout_snapshot_file_sha256 != heldout_file_sha
            or bundle.heldout_snapshot_sha256 != heldout_semantic_sha
            or bundle.heldout_eval_interval != 100
        ):
            raise CompletionAttestationError(
                "held-out replay snapshot/bundle identity mismatch"
            )
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
        "optimizer_steps": optimizer_steps,
    }
    if any(result.get(key) != value for key, value in expected_result.items()):
        raise CompletionAttestationError("learner result lineage mismatch")
    runtime_sha = _require_updated_runtime_policy(
        result.get("runtime_policy_sha256"), parent_runtime_sha
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
    checkpoint = _restart_document(restart, "restart checkpoint")
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
        "completed_steps": optimizer_steps,
        "world_size": args.world_size,
        "gradient_reduction": "ordered_allgather_rank_sum_v1",
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
        or len(checkpoint["rng_states"]) != args.world_size
    ):
        raise CompletionAttestationError("restart checkpoint state is incomplete")

    telemetry_path = _output_file(output, args.telemetry, "learner telemetry")
    telemetry = _telemetry_records(
        telemetry_path,
        campaign_id=campaign_id,
        macro_round=args.macro_round,
        optimizer_steps=optimizer_steps,
        replay_transitions_global=int(snapshot["transition_count"]),
        world_size=args.world_size,
        batch_size_per_rank=bundle.batch_size,
    )
    training_quality_path = _output_file(
        output,
        args.training_quality_telemetry,
        "training-quality telemetry",
    )
    training_quality = _training_quality_telemetry_records(
        training_quality_path,
        campaign_id=campaign_id,
        macro_round=args.macro_round,
        snapshot_file_sha256=snapshot_file_sha,
        snapshot_sha256=snapshot_sha,
        optimizer_steps=optimizer_steps,
        eval_interval=100,
        learned_action_indices=list(range(*bundle.train_action_slice)),
        world_size=args.world_size,
        partition_seed=bundle.partition_seed,
        probe_summary=probe_summary,
    )
    training_context_path = _output_file(
        output,
        args.training_context,
        "training-context telemetry",
    )
    training_context = _training_context_record(
        training_context_path,
        campaign_id=campaign_id,
        macro_round=args.macro_round,
        config_sha256=config_sha,
        parent_sha256=parent_sha,
        snapshot_file_sha256=snapshot_file_sha,
        snapshot_sha256=snapshot_sha,
        probe_selection_sha256=str(probe_summary["selection_sha256"]),
        replay_transitions_global=int(snapshot["transition_count"]),
    )
    heldout_telemetry = None
    heldout_telemetry_path = None
    if heldout_snapshot is not None:
        heldout_telemetry_path = _output_file(
            output, args.heldout_telemetry, "held-out learner telemetry"
        )
        heldout_telemetry = _heldout_telemetry_records(
            heldout_telemetry_path,
            campaign_id=campaign_id,
            macro_round=args.macro_round,
            snapshot_file_sha256=heldout_snapshot["file_sha256"],
            snapshot_sha256=heldout_snapshot["snapshot_sha256"],
            optimizer_steps=optimizer_steps,
            eval_interval=100,
            learned_action_indices=list(range(*bundle.train_action_slice)),
            world_size=args.world_size,
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
        world_size=args.world_size,
        training_quality_telemetry=training_quality,
        training_context=training_context,
        training_context_file_sha256=sha256_file(training_context_path),
        heldout_telemetry=heldout_telemetry,
        heldout_snapshot_file_sha256=(
            None if heldout_snapshot is None else heldout_snapshot["file_sha256"]
        ),
        heldout_snapshot_sha256=(
            None
            if heldout_snapshot is None
            else heldout_snapshot["snapshot_sha256"]
        ),
        heldout_eval_interval=(None if heldout_snapshot is None else 100),
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
        optimizer_steps=optimizer_steps,
        training_quality_probe_summary=probe_summary,
        training_quality_verified_steps=sorted(training_quality),
        training_context_file_sha256=sha256_file(training_context_path),
        training_context_metric_keys=sorted(TRAINING_CONTEXT_METRIC_KEYS),
        heldout_snapshot_file_sha256=(
            None if heldout_snapshot is None else heldout_snapshot["file_sha256"]
        ),
        heldout_snapshot_sha256=(
            None
            if heldout_snapshot is None
            else heldout_snapshot["snapshot_sha256"]
        ),
        heldout_eval_interval=(None if heldout_snapshot is None else 100),
        heldout_verified_steps=(
            None if heldout_telemetry is None else sorted(heldout_telemetry)
        ),
    )
    artifacts = {
        "bundle": _artifact(bundle_path),
        "result": _artifact(result_path),
        "restart": _artifact(restart),
        "candidate": _artifact(candidate),
        "policy_spec": _artifact(spec),
        "policy_manifest": _artifact(manifest),
        "telemetry": _artifact(telemetry_path),
        "training_quality_telemetry": _artifact(training_quality_path),
        "training_context": _artifact(training_context_path),
        "wandb_readback": _artifact(readback_path),
    }
    if heldout_telemetry_path is not None:
        artifacts["heldout_telemetry"] = _artifact(heldout_telemetry_path)
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
        "world_size": args.world_size,
        "batch_size_per_rank": bundle.batch_size,
        "optimizer_steps": optimizer_steps,
        "save_every": 5,
        "artifacts": artifacts,
        "training_quality_evaluation": {
            "snapshot_file_sha256": snapshot_file_sha,
            "snapshot_sha256": snapshot_sha,
            "eval_interval": 100,
            "verified_steps": sorted(training_quality),
            "probe": probe_summary,
        },
        "training_context": {
            "schema": TRAINING_CONTEXT_TELEMETRY_SCHEMA,
            "training_probe_selection_sha256": probe_summary[
                "selection_sha256"
            ],
            "metrics": dict(training_context["metrics"]),
            "verified_wandb_step": 0,
        },
        "wandb": {
            "entity": args.wandb_entity,
            "project": args.wandb_project,
            "run_id": args.wandb_run_id,
            "run_state": "finished",
            "verified_optimizer_steps": optimizer_steps,
            "verified_training_quality_steps": sorted(training_quality),
            "training_quality_probe_selection_sha256": probe_summary[
                "selection_sha256"
            ],
            "training_context_verified_step": 0,
            "training_context_metric_keys": sorted(
                TRAINING_CONTEXT_METRIC_KEYS
            ),
        },
    }
    if heldout_snapshot is not None and heldout_telemetry is not None:
        document["heldout_evaluation"] = {
            "snapshot_file_sha256": heldout_snapshot["file_sha256"],
            "snapshot_sha256": heldout_snapshot["snapshot_sha256"],
            "eval_interval": 100,
            "verified_steps": sorted(heldout_telemetry),
            "transition_count": heldout_snapshot["transition_count"],
        }
        document["wandb"]["verified_heldout_steps"] = sorted(
            heldout_telemetry
        )
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
    path = _regular(attestation, "learner completion attestation")
    if path.stat().st_mode & 0o222:
        raise CompletionAttestationError("completion attestation must be read-only")
    output = path.parent
    document = load_strict_json(path)
    heldout_declared = isinstance(document, Mapping) and (
        "heldout_evaluation" in document
    )
    expected_attestation_keys = set(ATTESTATION_KEYS)
    if heldout_declared:
        expected_attestation_keys.add("heldout_evaluation")
    if (
        not isinstance(document, Mapping)
        or set(document) != expected_attestation_keys
    ):
        raise CompletionAttestationError("completion attestation fields mismatch")
    optimizer_steps = document.get("optimizer_steps")
    if type(optimizer_steps) is not int or optimizer_steps < 1:
        raise CompletionAttestationError(
            "completion optimizer steps must be positive"
        )
    world_size = document.get("world_size")
    batch_size_per_rank = document.get("batch_size_per_rank")
    if (
        type(world_size) is not int
        or world_size < 1
        or type(batch_size_per_rank) is not int
        or batch_size_per_rank < 1
    ):
        raise CompletionAttestationError(
            "completion learner topology is malformed"
        )
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
        "world_size": world_size,
        "batch_size_per_rank": batch_size_per_rank,
        "optimizer_steps": optimizer_steps,
        "save_every": 5,
    }
    if any(document.get(key) != value for key, value in expected.items()):
        raise CompletionAttestationError("completion attestation lineage mismatch")
    candidate_runtime_sha256 = _sha(
        document.get("runtime_policy_sha256"), "runtime policy SHA-256"
    )
    parent_runtime_sha256 = _sha(
        document.get("parent_runtime_policy_sha256"),
        "parent runtime policy SHA-256",
    )
    _require_updated_runtime_policy(
        candidate_runtime_sha256, parent_runtime_sha256
    )
    artifacts = document.get("artifacts")
    expected_artifact_names = set(ARTIFACT_NAMES)
    if heldout_declared:
        expected_artifact_names.add("heldout_telemetry")
    if (
        not isinstance(artifacts, Mapping)
        or set(artifacts) != expected_artifact_names
    ):
        raise CompletionAttestationError("completion artifact inventory mismatch")
    resolved: dict[str, Path] = {}
    for name in sorted(expected_artifact_names):
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
        "optimizer_steps": optimizer_steps,
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
        != (batch_size_per_rank, optimizer_steps, 5)
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
    train_action_slice = bundle.get("train_action_slice")
    if (
        not isinstance(train_action_slice, list)
        or len(train_action_slice) != 2
        or any(type(value) is not int for value in train_action_slice)
    ):
        raise CompletionAttestationError(
            "attested learner action slice is malformed"
        )
    training_quality_summary = document.get("training_quality_evaluation")
    if not isinstance(training_quality_summary, Mapping) or set(
        training_quality_summary
    ) != {
        "snapshot_file_sha256",
        "snapshot_sha256",
        "eval_interval",
        "verified_steps",
        "probe",
    }:
        raise CompletionAttestationError(
            "training-quality completion summary fields mismatch"
        )
    expected_training_quality_steps = _evaluation_steps(optimizer_steps, 100)
    if (
        training_quality_summary.get("snapshot_file_sha256")
        != snapshot_file_sha256
        or training_quality_summary.get("snapshot_sha256") != snapshot_sha256
        or training_quality_summary.get("eval_interval") != 100
        or training_quality_summary.get("verified_steps")
        != expected_training_quality_steps
    ):
        raise CompletionAttestationError(
            "training-quality completion summary mismatch"
        )
    probe_summary = _training_quality_probe_summary(
        training_quality_summary.get("probe"),
        campaign_id=campaign_id,
        macro_round=macro_round,
        snapshot_file_sha256=snapshot_file_sha256,
        snapshot_sha256=snapshot_sha256,
        partition_seed=bundle["partition_seed"],
        world_size=world_size,
    )
    training_quality_records = _training_quality_telemetry_records(
        resolved["training_quality_telemetry"],
        campaign_id=campaign_id,
        macro_round=macro_round,
        snapshot_file_sha256=snapshot_file_sha256,
        snapshot_sha256=snapshot_sha256,
        optimizer_steps=optimizer_steps,
        eval_interval=100,
        learned_action_indices=list(range(*train_action_slice)),
        world_size=world_size,
        partition_seed=bundle["partition_seed"],
        probe_summary=probe_summary,
    )
    if (
        sorted(training_quality_records) != expected_training_quality_steps
        or next(iter(training_quality_records.values()))[
            "training_probe_transitions_global"
        ]
        != probe_summary["transitions_global"]
    ):
        raise CompletionAttestationError(
            "training-quality telemetry differs from completion summary"
        )
    training_context_record = _training_context_record(
        resolved["training_context"],
        campaign_id=campaign_id,
        macro_round=macro_round,
        config_sha256=config_sha256,
        parent_sha256=parent_sha256,
        snapshot_file_sha256=snapshot_file_sha256,
        snapshot_sha256=snapshot_sha256,
        probe_selection_sha256=str(probe_summary["selection_sha256"]),
        replay_transitions_global=int(
            probe_summary["replay_transitions_global"]
        ),
    )
    training_context_summary = document.get("training_context")
    expected_training_context_summary = {
        "schema": TRAINING_CONTEXT_TELEMETRY_SCHEMA,
        "training_probe_selection_sha256": probe_summary["selection_sha256"],
        "metrics": dict(training_context_record["metrics"]),
        "verified_wandb_step": 0,
    }
    if training_context_summary != expected_training_context_summary:
        raise CompletionAttestationError(
            "training-context completion summary mismatch"
        )
    heldout_bundle_values = (
        bundle.get("heldout_snapshot_file_sha256"),
        bundle.get("heldout_snapshot_sha256"),
        bundle.get("heldout_eval_interval"),
    )
    heldout_bundle_enabled = all(value is not None for value in heldout_bundle_values)
    if any(value is not None for value in heldout_bundle_values) and not (
        heldout_bundle_enabled
    ):
        raise CompletionAttestationError(
            "attested learner bundle held-out identity is incomplete"
        )
    if heldout_bundle_enabled != heldout_declared:
        raise CompletionAttestationError(
            "held-out attestation mode differs from learner bundle"
        )
    heldout_records = None
    heldout_summary = None
    if heldout_declared:
        heldout_summary = document.get("heldout_evaluation")
        if not isinstance(heldout_summary, Mapping) or set(heldout_summary) != {
            "snapshot_file_sha256",
            "snapshot_sha256",
            "eval_interval",
            "verified_steps",
            "transition_count",
        }:
            raise CompletionAttestationError(
                "held-out completion summary fields mismatch"
            )
        expected_heldout_steps = _evaluation_steps(optimizer_steps, 100)
        if (
            heldout_summary.get("snapshot_file_sha256")
            != bundle["heldout_snapshot_file_sha256"]
            or heldout_summary.get("snapshot_sha256")
            != bundle["heldout_snapshot_sha256"]
            or heldout_summary.get("eval_interval") != 100
            or bundle["heldout_eval_interval"] != 100
            or heldout_summary.get("verified_steps") != expected_heldout_steps
            or type(heldout_summary.get("transition_count")) is not int
            or heldout_summary["transition_count"] < 1
        ):
            raise CompletionAttestationError(
                "held-out completion summary mismatch"
            )
        heldout_records = _heldout_telemetry_records(
            resolved["heldout_telemetry"],
            campaign_id=campaign_id,
            macro_round=macro_round,
            snapshot_file_sha256=bundle["heldout_snapshot_file_sha256"],
            snapshot_sha256=bundle["heldout_snapshot_sha256"],
            optimizer_steps=optimizer_steps,
            eval_interval=100,
            learned_action_indices=list(range(*train_action_slice)),
            world_size=world_size,
        )
        if (
            sorted(heldout_records) != expected_heldout_steps
            or next(iter(heldout_records.values()))[
                "heldout_transitions_global"
            ]
            != heldout_summary["transition_count"]
        ):
            raise CompletionAttestationError(
                "held-out telemetry differs from completion summary"
            )
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
    restart_document = _restart_document(
        resolved["restart"], "attested restart checkpoint"
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
        "completed_steps": optimizer_steps,
        "world_size": world_size,
        "gradient_reduction": "ordered_allgather_rank_sum_v1",
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
    expected_wandb_keys = {
        "entity",
        "project",
        "run_id",
        "run_state",
        "verified_optimizer_steps",
        "verified_training_quality_steps",
        "training_quality_probe_selection_sha256",
        "training_context_verified_step",
        "training_context_metric_keys",
    }
    if heldout_declared:
        expected_wandb_keys.add("verified_heldout_steps")
    if not isinstance(wandb, Mapping) or set(wandb) != expected_wandb_keys:
        raise CompletionAttestationError("W&B completion summary fields mismatch")
    if (
        wandb.get("entity") != "nvidia"
        or wandb.get("run_state") != "finished"
        or wandb.get("verified_optimizer_steps") != optimizer_steps
        or wandb.get("verified_training_quality_steps")
        != expected_training_quality_steps
        or wandb.get("training_quality_probe_selection_sha256")
        != probe_summary["selection_sha256"]
        or wandb.get("training_context_verified_step") != 0
        or wandb.get("training_context_metric_keys")
        != sorted(TRAINING_CONTEXT_METRIC_KEYS)
        or TOKEN_RE.fullmatch(str(wandb.get("project", ""))) is None
        or TOKEN_RE.fullmatch(str(wandb.get("run_id", ""))) is None
        or wandb.get("run_id")
        != learner_wandb_run_id(
            campaign_id=campaign_id,
            macro_round=macro_round,
            learner_identity_sha256=provided_identity_sha256,
        )
        or (
            heldout_declared
            and wandb.get("verified_heldout_steps")
            != _evaluation_steps(optimizer_steps, 100)
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
        optimizer_steps=optimizer_steps,
        training_quality_probe_summary=probe_summary,
        training_quality_verified_steps=expected_training_quality_steps,
        training_context_file_sha256=sha256_file(
            resolved["training_context"]
        ),
        training_context_metric_keys=sorted(TRAINING_CONTEXT_METRIC_KEYS),
        heldout_snapshot_file_sha256=(
            None
            if heldout_summary is None
            else str(heldout_summary["snapshot_file_sha256"])
        ),
        heldout_snapshot_sha256=(
            None
            if heldout_summary is None
            else str(heldout_summary["snapshot_sha256"])
        ),
        heldout_eval_interval=(
            None if heldout_summary is None else int(heldout_summary["eval_interval"])
        ),
        heldout_verified_steps=(
            None
            if heldout_summary is None
            else list(heldout_summary["verified_steps"])
        ),
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
    publish.add_argument(
        "--training-quality-telemetry",
        default="training_quality_telemetry.jsonl",
    )
    publish.add_argument("--training-context", default="training_context.json")
    publish.add_argument("--parent-checkpoint", required=True, type=Path)
    publish.add_argument("--snapshot", required=True, type=Path)
    publish.add_argument("--artifact-root", required=True, type=Path)
    publish.add_argument("--world-size", required=True, type=int)
    publish.add_argument("--heldout-snapshot", type=Path)
    publish.add_argument("--heldout-snapshot-file-sha256")
    publish.add_argument("--heldout-snapshot-sha256")
    publish.add_argument("--heldout-eval-interval", type=int)
    publish.add_argument(
        "--heldout-telemetry", default="heldout_telemetry.jsonl"
    )
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
