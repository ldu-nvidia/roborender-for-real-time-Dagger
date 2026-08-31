#!/usr/bin/env python3
"""Fail-closed student-versus-expert quality telemetry for sealed cohorts.

The actor payload already contains the frozen student's action, the expert
label, the behavior action selected by DAgger, episode identities, and done
flags.  This module derives policy-quality metrics only after the cohort seal
and every referenced payload have been verified byte-for-byte.  It deliberately
does not infer reward, task success, or safety outcomes that are absent from the
rollout schema.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import (
    ArtifactError,
    canonical_json_sha256,
    load_strict_json,
    publish_immutable_json,
    sha256_file,
    visit_verified_cohort_shards,
)


QUALITY_SCHEMA = "fast_roborender.round_dagger.student_quality.v1"
QUALITY_RECOVERY_SCHEMA = (
    "fast_roborender.round_dagger.student_quality_recovery.v1"
)
LEARNED_ACTION_DIMENSIONS = 11
LINEAGE_FIELDS = (
    "campaign_id",
    "macro_round",
    "cohort_index",
    "policy_sha256",
    "runtime_policy_sha256",
    "teacher_sha256",
    "renderer_sha256",
    "beta_sha256",
    "config_sha256",
)
SHA256_FIELDS = {
    "policy_sha256",
    "runtime_policy_sha256",
    "teacher_sha256",
    "renderer_sha256",
    "beta_sha256",
    "config_sha256",
}

# Cohort 00 of the first production run was finalized before trusted host
# hooks pinned their Python/NumPy runtime.  NumPy 2.3 and the subsequently
# pinned NumPy 2.0 use different pairwise-reduction trees for these two means;
# on that exact cohort each result differs by one binary64 ULP.  No other
# quality field is reduction-runtime compatible: accepting even a one-ULP
# difference elsewhere would hide corruption or semantic drift.
_LEGACY_REDUCTION_COMPATIBLE_METRICS = frozenset(
    {
        "student_quality/action_cosine_similarity_mean",
        "student_quality/action_l2_mean",
    }
)


class StudentQualityError(RuntimeError):
    """A cohort cannot support trustworthy student-quality telemetry."""


def _load_optional_regular_json(path: Path) -> dict[str, Any] | None:
    """Load an existing immutable document without following a symlink."""

    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise StudentQualityError(
            f"cannot inspect existing quality artifact {path}: {exc}"
        ) from exc
    if not stat.S_ISREG(info.st_mode):
        raise StudentQualityError(
            f"existing quality artifact is not a regular file: {path}"
        )
    if info.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise StudentQualityError(
            f"existing quality artifact is not immutable: {path}"
        )
    try:
        value = load_strict_json(path)
    except (ArtifactError, OSError) as exc:
        raise StudentQualityError(
            f"existing quality artifact is invalid: {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise StudentQualityError(
            f"existing quality artifact is not an object: {path}"
        )
    if sha256_file(path) != canonical_json_sha256(value):
        raise StudentQualityError(
            f"existing quality artifact is not canonical JSON: {path}"
        )
    return value


def _verify_legacy_reduction_compatibility(
    existing: Mapping[str, Any], recomputed: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Require exact identity except two explicitly known one-ULP means.

    This is intentionally narrower than an epsilon comparison.  It preserves
    a pre-runtime-pin immutable quality artifact while still rejecting schema,
    lineage, shard, count, ordering, type, and metric drift.
    """

    accepted: list[dict[str, Any]] = []

    def compare(left: Any, right: Any, path: tuple[str | int, ...]) -> None:
        if type(left) is not type(right):
            raise StudentQualityError(
                "existing quality artifact type mismatch at "
                + "/".join(map(str, path))
            )
        if isinstance(left, Mapping):
            if set(left) != set(right):
                raise StudentQualityError(
                    "existing quality artifact key mismatch at "
                    + "/".join(map(str, path))
                )
            for key in sorted(left):
                compare(left[key], right[key], (*path, key))
            return
        if isinstance(left, list):
            if len(left) != len(right):
                raise StudentQualityError(
                    "existing quality artifact length mismatch at "
                    + "/".join(map(str, path))
                )
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                compare(left_item, right_item, (*path, index))
            return
        if left == right:
            return
        metric = path[1] if len(path) == 2 and path[0] == "metrics" else None
        if (
            metric not in _LEGACY_REDUCTION_COMPATIBLE_METRICS
            or type(left) is not float
            or not math.isfinite(left)
            or not math.isfinite(right)
            or math.nextafter(left, right) != right
        ):
            raise StudentQualityError(
                "existing quality artifact semantic mismatch at "
                + "/".join(map(str, path))
            )
        accepted.append(
            {
                "json_path": f"metrics/{metric}",
                "existing_float_hex": left.hex(),
                "recomputed_float_hex": right.hex(),
                "ulp_distance": 1,
            }
        )

    compare(existing, recomputed, ())
    return accepted


def _expected_lineage(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(LINEAGE_FIELDS):
        raise StudentQualityError(
            "expected lineage must contain exactly the production lineage fields"
        )
    result = dict(value)
    campaign_id = result["campaign_id"]
    if not isinstance(campaign_id, str) or not campaign_id:
        raise StudentQualityError("expected campaign_id must be a non-empty string")
    for key in ("macro_round", "cohort_index"):
        if type(result[key]) is not int or result[key] < 0:
            raise StudentQualityError(f"expected {key} must be a non-negative integer")
    for key in SHA256_FIELDS:
        digest = result[key]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise StudentQualityError(f"expected {key} must be a lowercase SHA-256")
    return result


def _linear_quantile(values: np.ndarray, probability: float) -> float:
    """Deterministic NumPy-version-independent linear sample quantile."""

    if values.ndim != 1 or values.size == 0:
        raise StudentQualityError("cannot compute a quantile from empty data")
    ordered = np.sort(values.astype(np.float64, copy=False))
    position = (ordered.size - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _action_cosine(student: np.ndarray, expert: np.ndarray) -> np.ndarray:
    dot = np.sum(student * expert, axis=-1, dtype=np.float64)
    student_norm = np.linalg.norm(student, axis=-1)
    expert_norm = np.linalg.norm(expert, axis=-1)
    denominator = student_norm * expert_norm
    cosine = np.zeros_like(dot, dtype=np.float64)
    nonzero = denominator > 0.0
    np.divide(dot, denominator, out=cosine, where=nonzero)
    cosine[(student_norm == 0.0) & (expert_norm == 0.0)] = 1.0
    return np.clip(cosine, -1.0, 1.0)


def summarize_sealed_cohort(
    cohort_dir: str | os.PathLike[str],
    *,
    expected_lineage: Mapping[str, Any],
    expected_beta: float,
) -> dict[str, Any]:
    """Return deterministic metrics after exact seal, schema, and lineage gates."""

    expected = _expected_lineage(expected_lineage)
    if (
        isinstance(expected_beta, bool)
        or not isinstance(expected_beta, (int, float))
        or not math.isfinite(float(expected_beta))
        or not 0.0 <= float(expected_beta) <= 1.0
    ):
        raise StudentQualityError("expected beta must be finite and in [0,1]")
    expected_beta = float(expected_beta)
    if canonical_json_sha256({"beta": expected_beta}) != expected["beta_sha256"]:
        raise StudentQualityError("expected beta does not match beta_sha256")
    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    behavior_match_count = 0
    intervention_count = 0
    done_count = 0
    transition_count = 0
    episode_identities: set[tuple[int, int]] = set()
    environment_identities: set[int] = set()
    episode_steps_by_identity: dict[tuple[int, int], set[int]] = {}
    done_steps_by_identity: dict[tuple[int, int], set[int]] = {}
    l2_chunks: list[np.ndarray] = []
    cosine_chunks: list[np.ndarray] = []
    per_action_error_chunks: list[np.ndarray] = []
    per_action_squared_error_sum = np.zeros(
        LEARNED_ACTION_DIMENSIONS, dtype=np.float64
    )
    per_action_absolute_error_sum = np.zeros(
        LEARNED_ACTION_DIMENSIONS, dtype=np.float64
    )
    per_action_signed_error_sum = np.zeros(
        LEARNED_ACTION_DIMENSIONS, dtype=np.float64
    )
    shard_evidence: list[dict[str, Any]] = []

    def visit_shard(
        report: Mapping[str, Any], arrays: Mapping[str, np.ndarray]
    ) -> None:
        nonlocal squared_error_sum
        nonlocal absolute_error_sum
        nonlocal behavior_match_count
        nonlocal intervention_count
        nonlocal done_count
        nonlocal transition_count
        action_slice = slice(0, LEARNED_ACTION_DIMENSIONS)
        student = arrays["student_actions"][..., action_slice].astype(
            np.float64, copy=False
        )
        expert = arrays["expert_actions"][..., action_slice].astype(
            np.float64, copy=False
        )
        behavior = arrays["behavior_actions"][..., action_slice].astype(
            np.float64, copy=False
        )
        if student.shape != expert.shape or behavior.shape != expert.shape:
            raise StudentQualityError("verified action arrays disagree after learned slice")
        shard_transitions = int(student.shape[0] * student.shape[1])
        if shard_transitions != int(report["transition_count"]):
            raise StudentQualityError("verified shard transition shape/count mismatch")

        difference = student - expert
        squared_error_sum += float(np.sum(difference * difference, dtype=np.float64))
        absolute_error_sum += float(np.sum(np.abs(difference), dtype=np.float64))
        flat_difference = difference.reshape(-1, LEARNED_ACTION_DIMENSIONS)
        per_action_error_chunks.append(flat_difference)
        per_action_squared_error_sum[:] += np.sum(
            flat_difference * flat_difference, axis=0, dtype=np.float64
        )
        per_action_absolute_error_sum[:] += np.sum(
            np.abs(flat_difference), axis=0, dtype=np.float64
        )
        per_action_signed_error_sum[:] += np.sum(
            flat_difference, axis=0, dtype=np.float64
        )
        l2_chunks.append(np.linalg.norm(difference, axis=-1).reshape(-1))
        cosine_chunks.append(_action_cosine(student, expert).reshape(-1))
        behavior_match_count += int(
            np.count_nonzero(np.all(behavior == expert, axis=-1))
        )
        intervention_count += int(np.count_nonzero(arrays["expert_mask"]))
        done_count += int(np.count_nonzero(arrays["dones"]))
        transition_count += shard_transitions

        env_ids = arrays["global_env_ids"]
        if env_ids.ndim == 2:
            env_ids = env_ids[0]
        episodes = arrays["episode_ids"]
        episode_steps = arrays["episode_steps"]
        dones = arrays["dones"]
        for env_index, global_env_id in enumerate(env_ids.tolist()):
            env_id = int(global_env_id)
            environment_identities.add(env_id)
            for time_index in range(episodes.shape[0]):
                identity = (env_id, int(episodes[time_index, env_index]))
                episode_step = int(episode_steps[time_index, env_index])
                observed = episode_steps_by_identity.setdefault(identity, set())
                if episode_step in observed:
                    raise StudentQualityError(
                        "duplicate episode transition across verified shards"
                    )
                observed.add(episode_step)
                episode_identities.add(identity)
                if bool(dones[time_index, env_index]):
                    done_steps_by_identity.setdefault(identity, set()).add(
                        episode_step
                    )
        shard_evidence.append(
            {
                "shard_id": report["shard_id"],
                "manifest_sha256": report["manifest_sha256"],
                "payload_sha256": report["payload_sha256"],
                "transition_count": report["transition_count"],
            }
        )

    try:
        seal = visit_verified_cohort_shards(cohort_dir, visit_shard)
    except StudentQualityError:
        raise
    except (ArtifactError, OSError) as exc:
        raise StudentQualityError(
            f"sealed cohort verification failed: {exc}"
        ) from exc
    if seal.get("purpose") != "train":
        raise StudentQualityError("student-quality telemetry requires a train cohort")
    mismatches = [key for key in LINEAGE_FIELDS if seal.get(key) != expected[key]]
    if mismatches:
        raise StudentQualityError(
            "sealed cohort lineage mismatch: " + ", ".join(sorted(mismatches))
        )
    action_contract = seal.get("action_contract")
    shape = action_contract.get("shape") if isinstance(action_contract, Mapping) else None
    if (
        not isinstance(shape, list)
        or len(shape) != 1
        or type(shape[0]) is not int
        or shape[0] < LEARNED_ACTION_DIMENSIONS
    ):
        raise StudentQualityError(
            f"action contract must expose at least {LEARNED_ACTION_DIMENSIONS} dimensions"
        )

    if transition_count != int(seal["transition_count"]):
        raise StudentQualityError("quality aggregation did not cover the sealed cohort")
    if not l2_chunks or not cosine_chunks:
        raise StudentQualityError("sealed cohort contained no quality samples")
    l2 = np.concatenate(l2_chunks).astype(np.float64, copy=False)
    cosine = np.concatenate(cosine_chunks).astype(np.float64, copy=False)
    if l2.size != transition_count or cosine.size != transition_count:
        raise StudentQualityError("per-transition quality sample count mismatch")

    element_count = transition_count * LEARNED_ACTION_DIMENSIONS
    mse = squared_error_sum / element_count
    mae = absolute_error_sum / element_count
    observed_episode_lengths = np.asarray(
        [len(steps) for steps in episode_steps_by_identity.values()],
        dtype=np.float64,
    )
    if observed_episode_lengths.size != len(episode_identities):
        raise StudentQualityError("episode accounting coverage mismatch")
    for identity, done_steps in done_steps_by_identity.items():
        if len(done_steps) != 1:
            raise StudentQualityError(
                f"episode {identity} contains multiple done transitions"
            )
    fully_observed_done_lengths: list[float] = []
    for identity, done_steps in done_steps_by_identity.items():
        done_step = next(iter(done_steps))
        observed = episode_steps_by_identity[identity]
        if observed == set(range(done_step + 1)):
            fully_observed_done_lengths.append(float(done_step + 1))
    done_episode_count = len(done_steps_by_identity)
    censored_episode_count = len(episode_identities) - done_episode_count
    steps_per_environment, remainder = divmod(
        transition_count, len(environment_identities)
    )
    if remainder or steps_per_environment < 1:
        raise StudentQualityError("sealed cohort has invalid per-environment coverage")
    metrics: dict[str, int | float] = {
        "student_quality/valid": 1,
        "student_quality/learned_action_dimensions": LEARNED_ACTION_DIMENSIONS,
        "student_quality/action_mse": mse,
        "student_quality/action_mae": mae,
        "student_quality/action_rmse": math.sqrt(mse),
        "student_quality/action_cosine_similarity_mean": float(np.mean(cosine)),
        "student_quality/action_cosine_similarity_p50": _linear_quantile(cosine, 0.50),
        "student_quality/action_cosine_similarity_p90": _linear_quantile(cosine, 0.90),
        "student_quality/action_cosine_similarity_p99": _linear_quantile(cosine, 0.99),
        "student_quality/action_l2_mean": float(np.mean(l2)),
        "student_quality/action_l2_p50": _linear_quantile(l2, 0.50),
        "student_quality/action_l2_p90": _linear_quantile(l2, 0.90),
        "student_quality/action_l2_p99": _linear_quantile(l2, 0.99),
        "student_quality/behavior_expert_match_rate": behavior_match_count
        / transition_count,
        "student_quality/behavior_expert_matches": behavior_match_count,
        "student_quality/scheduled_beta": expected_beta,
        "student_quality/expert_intervention_rate": intervention_count
        / transition_count,
        "student_quality/expert_interventions": intervention_count,
        # The rollout payload carries an expert label for every visited state;
        # expert_mask records which label was actually executed by the DAgger
        # behavior mixture.  Keep query/label coverage distinct from behavior
        # intervention so beta does not masquerade as missing supervision.
        "student_quality/expert_label_coverage": 1.0,
        "student_quality/expert_labels": transition_count,
        "student_quality/expert_query_rate": 1.0,
        "student_quality/expert_queries": transition_count,
        "student_quality/student_behavior_rate": 1.0
        - intervention_count / transition_count,
        "student_quality/student_behavior_transitions": transition_count
        - intervention_count,
        "student_quality/transitions": transition_count,
        "student_quality/environments": len(environment_identities),
        "student_quality/steps_per_environment": steps_per_environment,
        "student_quality/episodes": len(episode_identities),
        "student_quality/dones": done_count,
        "student_quality/done_rate": done_count / transition_count,
        "student_quality/episodes_with_done": done_episode_count,
        "student_quality/episode_done_fraction": done_episode_count
        / len(episode_identities),
        "student_quality/right_censored_episodes": censored_episode_count,
        "student_quality/right_censored_episode_fraction": censored_episode_count
        / len(episode_identities),
        "student_quality/fully_observed_done_episodes": len(
            fully_observed_done_lengths
        ),
        "student_quality/observed_episode_length_mean": float(
            np.mean(observed_episode_lengths)
        ),
        "student_quality/observed_episode_length_p50": _linear_quantile(
            observed_episode_lengths, 0.50
        ),
        "student_quality/observed_episode_length_p90": _linear_quantile(
            observed_episode_lengths, 0.90
        ),
        "student_quality/macro_round": int(seal["macro_round"]),
        "student_quality/cohort_index": int(seal["cohort_index"]),
    }
    if fully_observed_done_lengths:
        done_lengths = np.asarray(fully_observed_done_lengths, dtype=np.float64)
        metrics.update(
            {
                "student_quality/done_episode_length_mean": float(
                    np.mean(done_lengths)
                ),
                "student_quality/done_episode_length_p50": _linear_quantile(
                    done_lengths, 0.50
                ),
                "student_quality/done_episode_length_p90": _linear_quantile(
                    done_lengths, 0.90
                ),
            }
        )
    per_action_errors = np.concatenate(per_action_error_chunks, axis=0)
    if per_action_errors.shape != (
        transition_count,
        LEARNED_ACTION_DIMENSIONS,
    ):
        raise StudentQualityError("per-action disagreement coverage mismatch")
    for action_index in range(LEARNED_ACTION_DIMENSIONS):
        action_mse = per_action_squared_error_sum[action_index] / transition_count
        prefix = f"student_quality/action_{action_index:02d}"
        metrics.update(
            {
                f"{prefix}/mse": float(action_mse),
                f"{prefix}/mae": float(
                    per_action_absolute_error_sum[action_index] / transition_count
                ),
                f"{prefix}/rmse": math.sqrt(float(action_mse)),
                f"{prefix}/mean_error": float(
                    per_action_signed_error_sum[action_index] / transition_count
                ),
                f"{prefix}/absolute_error_p90": _linear_quantile(
                    np.abs(per_action_errors[:, action_index]), 0.90
                ),
            }
        )
    for key, value in metrics.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise StudentQualityError(f"derived metric is non-finite: {key}")

    lineage = {key: seal[key] for key in LINEAGE_FIELDS}
    lineage.update(
        {
            "logical_id": seal["logical_id"],
            "contract_sha256": seal["contract_sha256"],
            "cohort_sha256": seal["cohort_sha256"],
        }
    )
    return {
        "schema": QUALITY_SCHEMA,
        "lineage": lineage,
        "action_slice": {
            "start_inclusive": 0,
            "stop_exclusive": LEARNED_ACTION_DIMENSIONS,
        },
        "counts": {
            "transitions": transition_count,
            "episodes": len(episode_identities),
            "dones": done_count,
            "steps_per_environment": steps_per_environment,
        },
        "metrics": metrics,
        "shards": sorted(shard_evidence, key=lambda item: item["shard_id"]),
        "wandb_step": steps_per_environment - 1,
    }


def publish_student_quality(
    cohort_dir: str | os.PathLike[str],
    run_dir: str | os.PathLike[str],
    *,
    expected_lineage: Mapping[str, Any],
    expected_beta: float,
) -> dict[str, Any]:
    """Publish an immutable lineage artifact and its numeric W&B projection."""

    run_root = Path(run_dir).expanduser()
    if not run_root.is_absolute():
        raise StudentQualityError("quality run directory must be absolute")
    document = summarize_sealed_cohort(
        cohort_dir,
        expected_lineage=expected_lineage,
        expected_beta=expected_beta,
    )
    lineage = document["lineage"]
    artifact_name = (
        f"macro_round_{lineage['macro_round']:09d}_"
        f"cohort_{lineage['cohort_index']:04d}_student_quality.json"
    )
    artifact_path = run_root / "quality" / artifact_name
    scalar_path = (
        run_root
        / "telemetry"
        / "scalars"
        / (
            f"step_{document['wandb_step']:09d}_student_quality_"
            f"r{lineage['macro_round']:09d}_c{lineage['cohort_index']:04d}.json"
        )
    )
    try:
        existing_document = _load_optional_regular_json(artifact_path)
        publication_document = document
        if existing_document is None:
            publish_immutable_json(artifact_path, document)
        else:
            accepted_differences = _verify_legacy_reduction_compatibility(
                existing_document, document
            )
            publication_document = existing_document
            if accepted_differences:
                existing_sha256 = sha256_file(artifact_path)
                recomputed_sha256 = canonical_json_sha256(document)
                recovery_path = (
                    run_root
                    / "quality"
                    / "recovery"
                    / (
                        f"existing_{existing_sha256}_"
                        f"recomputed_{recomputed_sha256}.json"
                    )
                )
                publish_immutable_json(
                    recovery_path,
                    {
                        "schema": QUALITY_RECOVERY_SCHEMA,
                        "status": "verified_legacy_reduction_adoption",
                        "reason": "numpy_pairwise_reduction_one_ulp",
                        "existing_artifact_sha256": existing_sha256,
                        "recomputed_artifact_sha256": recomputed_sha256,
                        "cohort_sha256": publication_document["lineage"][
                            "cohort_sha256"
                        ],
                        "lineage": publication_document["lineage"],
                        "accepted_differences": accepted_differences,
                    },
                )
        publish_immutable_json(
            scalar_path,
            {
                "step": publication_document["wandb_step"],
                "metrics": publication_document["metrics"],
            },
        )
    except (ArtifactError, KeyError, TypeError) as exc:
        raise StudentQualityError(f"quality publication failed: {exc}") from exc
    return {
        "schema": "fast_roborender.round_dagger.student_quality_publication.v1",
        "artifact_path": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "scalar_path": str(scalar_path),
        "scalar_sha256": sha256_file(scalar_path),
        "wandb_step": publication_document["wandb_step"],
        "metrics": publication_document["metrics"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--expected-campaign-id", required=True)
    parser.add_argument("--expected-macro-round", required=True, type=int)
    parser.add_argument("--expected-cohort-index", required=True, type=int)
    parser.add_argument("--expected-beta", required=True, type=float)
    for field in (
        "policy",
        "runtime-policy",
        "teacher",
        "renderer",
        "beta",
        "config",
    ):
        parser.add_argument(f"--expected-{field}-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    expected = {
        "campaign_id": arguments.expected_campaign_id,
        "macro_round": arguments.expected_macro_round,
        "cohort_index": arguments.expected_cohort_index,
        "policy_sha256": arguments.expected_policy_sha256,
        "runtime_policy_sha256": arguments.expected_runtime_policy_sha256,
        "teacher_sha256": arguments.expected_teacher_sha256,
        "renderer_sha256": arguments.expected_renderer_sha256,
        "beta_sha256": arguments.expected_beta_sha256,
        "config_sha256": arguments.expected_config_sha256,
    }
    report = publish_student_quality(
        arguments.cohort_dir,
        arguments.run_dir,
        expected_lineage=expected,
        expected_beta=arguments.expected_beta,
    )
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (StudentQualityError, ArtifactError) as error:
        print(f"student quality gate failed: {error}", file=__import__("sys").stderr)
        raise SystemExit(2)


__all__ = [
    "LEARNED_ACTION_DIMENSIONS",
    "LINEAGE_FIELDS",
    "QUALITY_SCHEMA",
    "QUALITY_RECOVERY_SCHEMA",
    "StudentQualityError",
    "publish_student_quality",
    "summarize_sealed_cohort",
]
